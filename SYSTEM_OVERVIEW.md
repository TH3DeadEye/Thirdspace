# ThirdSpace — Cultural Conflict Resolution System

**Author:** Arman Milani
**Purpose:** Meeting prep document. Walks through what was built, why each piece exists, what was added beyond the original spec, and known limitations.

---

## 1. What the system is

A multi-stage reasoning pipeline (not a chatbot) that takes a raw, emotionally-charged, one-sided description of a cultural conflict and returns a structured analysis the user can learn from.

It runs on **AWS Bedrock** using two Claude models:
- **Main agent:** Claude Sonnet 4.5 (`us.anthropic.claude-sonnet-4-5-20250929-v1:0`)
- **Challenger agent:** Claude Haiku 4.5 (`us.anthropic.claude-haiku-4-5-20251001-v1:0`)

Using different SKUs gives the Challenger a different checkpoint — different priors — without needing a separate fine-tune. This partially addresses the design requirement that the Challenger reason "with a separate LLM."

Orchestration is **LangGraph** `StateGraph`. Models, validation, and constraints are **Pydantic v2**.

---

## 2. The five-phase pipeline (high level)

```
raw scenario
    │
    ▼
[Phase 1] Reformulation
    Decomposes into 6 structured nodes ("The Occurring")
    │
    ▼
[Phase 2] Critical Thinking (Main Agent)
    Reasons across a 6×7 grid = 42 cells, one LLM call per cell
    │
    ▼
[Phase 3] Challenger
    Argues opposing cultural frame across the same 42 cells
    │
    ▼
[Phase 4] Mediation
    Compares grids, finds agreements/conflicts, extracts implications
    │
    ├── if not converged & cycles remain → loop back to Phase 2
    │   (after folding implying nodes back into The Occurring)
    │
    ▼
[Phase 5] Constrained Generation
    Writes the 5-section final document, bound to what mediation found
    │
    ▼
[Judge] LLM-as-a-judge scores empathy / consensus / escalation / insight
    │
    ├── if scores below threshold → regenerate with judge feedback
    │
    ▼
final output + persisted session JSON
```

Each cycle does ~85 LLM calls (42 main + 42 challenger + 1 mediation + 1 node update + 1 cycle judge + occasional refinement retries). Cells are parallelized with `ThreadPoolExecutor` so each phase finishes in roughly the slowest cell's wall time, not 42× that.

---

## 3. File structure

```
thirdspace/
├── models.py        — Pydantic data models, constants (FACET_LIST, DIMENSION_LIST)
├── llm.py           — Bedrock client, prompts, JSON extraction, refinement wrapper
├── agents.py        — One function per pipeline phase
├── graph.py         — LangGraph StateGraph wiring + routing functions
├── main.py          — Entry point + session persistence
├── budget.py        — Cost / latency / circuit-breaker guardrails (contextvar-based)
└── requirements.txt
```

Outputs are written to `outputs/{scenario_id}.json` — one file per scenario containing the entire session state (grids, mediations, judges, final document).

---

## 4. The 6×7 grid — what the cells actually are

### Six facets (rows) — "The Occurring" nodes

| Facet | What goes here |
|---|---|
| `individuals` | Each person *as they are already affected* by the conflict before responding — shaped by culture, relational history, and the event. Not neutral bios. |
| `interactional_event` | What a camera would have recorded. Stripped of all interpretation and backstory. |
| `process` | Timeline, context, and each party's model of time (linear vs cyclical, incident vs pattern). |
| `evidence_and_data` | Only what is verifiable. Interpretations are explicitly excluded — they go in `inferences`. |
| `concepts_theories_rules` | The cultural frameworks and norms each party is applying — usually without knowing it. Named for both sides. |
| `inferences` | What the *structure* of nodes 1-5 implies. Often contradicts the narrator's framing. |

### Seven critical-thinking dimensions (columns)

`objective`, `query_formulation`, `perspective`, `assumptions`, `evidence`, `concepts`, `implications`

Each cell is a focused micro-reasoning task: "reason about the **individuals** facet through the **assumptions** lens." This narrowness is the entire point — batched grid reasoning collapses into mush.

**Architectural rule:** one LLM call per cell. Never batch. Both agents make 42 separate calls. Enforced by `_cell_pairs()` in `agents.py:31`.

---

## 5. Phase-by-phase walkthrough

### Phase 1 — Reformulation (`run_reformulation`)

Single LLM call. Outputs 6 `OccurringNode`s.

**What it does:** Takes the raw scenario and decomposes it. The reformulation prompt explicitly forbids echoing the narrator's labels — every node must contain something the narrator did *not* explicitly state. Each node carries a `list[str]` of `assumptions`.

**Validation (`agents.py:41-89`):**
- Exactly 6 nodes returned (raises `ValueError` otherwise).
- `_validate_facet_set` rejects unknown facet names, duplicates, and missing facets — surfaces a clear error message for each case.
- Each node must carry ≥1 assumption.
- `_sort_by_facet` puts nodes into canonical `FACET_LIST` order so downstream indexing is stable regardless of what order the model returned them.

### Phase 2 — Critical Thinking (`run_critical_thinking`)

42 cell calls. Each cell:
1. Builds `critical_thinking_cell_system(cultural_frame, facet, dimension)` — the prompt names the agent's cultural frame as a *feature, not a flaw*, and requires the agent to flag whenever its frame shapes the reasoning.
2. Builds `critical_thinking_cell_user(occurring_json, facet, dimension)` — passes the full Occurring as context, then narrows the focus.
3. Wraps in `call_llm_json_with_refinement` (see §6 below).
4. Returns a `NodeDimensionCell` with content, `reflection_flags`, and refinement telemetry.

Parallelized via `ThreadPoolExecutor(max_workers=8)`. Configurable via `THIRDSPACE_CELL_WORKERS` env var.

### Phase 3 — Challenger (`run_challenger`)

Same 42-call structure, but:
- Uses `CHALLENGER_MODEL_ID` (Haiku 4.5 by default) — different checkpoint = different priors.
- Receives the main agent's grid linearized via `linearize_grid_to_matrix_map` — converts flat list to nested `{facet: {dimension: {content, ...}}}` lookup so each challenger cell can directly read what the main agent said about *that* cell.
- Classifies each response as `challenge`, `reframe`, or `confirm`.
- Uses `active_challenger_perspective` (refined each cycle by mediation) instead of the original cultural frame if mediation has produced a refinement.

**Argumentation style** is set independently of cultural identity (per design):

```python
challenger_style = (
    "procedural adversarial argumentation: surface load-bearing assumptions, "
    "demand they be justified rather than asserted, and refuse to let any "
    "single cultural norm stand in as a universal fact"
)
```

This separates *what the challenger argues for* (the cultural frame) from *how the challenger argues* (the procedural style). The mediator owns the latter.

### Phase 4 — Mediation (`run_mediation`)

Single LLM call. Inputs:
1. The Occurring (full)
2. Cell-pair comparison — for each of 42 cells, both agents' content + the challenger's `response_type`. Built compactly (not full grids) to keep the prompt under control.
3. Compressed history of prior mediation cycles via `compress_historical_context`:
   - Most recent cycle: kept verbatim
   - Older cycles: shrunk to `{cycle, converged, conflict_facets, updated_assumptions}` gists
   - Prevents the prompt from ballooning across cycles

Outputs an `ImplyingNodes` containing:
- `agreements: list[str]` — where the agents converged
- `conflicts: list[CellConflict]` — each with an `implication` field (the critical one — "this reveals that…")
- `what_conflicts_imply: list[str]` — pattern-level synthesis across all conflicts
- `updated_assumptions: list[str]` — literal "node X: remove [A] → replace with [B]" instructions
- `wholeness: WholenessImplyingNodes` — distilled output that feeds the next cycle and constrains final generation:
  - `query_assumptions`, `reformulated_query`, `new_objective`
  - `updated_challenger_perspective`, `objective_assumptions`, `perspective_assumptions`
- `converged: bool`

### Cycle integration — node update (`run_node_update`)

This is the step that turns the loop from a no-op into a real refinement cycle.

Between cycles, mediation's implying nodes are folded back into The Occurring. A node-update LLM call:
- Applies assumption updates literally
- Only modifies nodes mediation actually touched (others remain byte-identical)
- Stamps `last_updated_cycle` only on nodes that changed
- Validates the result with the same `_validate_facet_set` used by reformulation

The challenger's working perspective is also refreshed from `wholeness.updated_challenger_perspective` if set, so the next cycle's challenger reasons from a sharpened frame.

### Phase 5 — Constrained Generation (`run_constrained_generation`)

Single LLM call producing a 5-section document:
1. **What Actually Happened** — mediation's version, not the narrator's
2. **Where the Cultures Diverged** — significant conflicts, what each frame expected, the specific misread
3. **What the Conflict Actually Implies** — uses `what_conflicts_imply`
4. **What Each Person Likely Experienced** — empathy without blame or excuse
5. **How to Move Forward** — grounded only in `updated_assumptions` and `new_objective`; explicitly refuses to suggest "have an honest conversation" when the conflict is *about* what counts as honest communication

**The hard constraint:** every claim must trace back to something mediation found. No new inferences. No training-data wisdom.

**What gets passed to generation:**
- The Occurring (final, post-integration state)
- Distilled wholeness nodes from *all* cycles
- The final mediation in full
- `all_implications` — aggregated `what_conflicts_imply` + `implication` fields across every cycle (per the design: bound by implications gathered across the *entire* mediation cycle, not just the final one)
- Optional `evaluation_feedback` from the output judge if this is a regeneration pass

---

## 6. Quality controls

### Self-refinement loop (`call_llm_json_with_refinement`)

Every cell call goes through this wrapper. Each cell returns reasoning + `reflection_flags` — the model's own self-evaluation against 8 measures (`clarity, correctness, precision, proportion, depth, comprehensiveness, logic, fairness`).

If the response fails validation:
- Empty/short content (`<3` sentences via abbreviation-aware sentence count) → retry
- Any populated `reflection_flag` → corrective instruction injected per measure (`REFLECTION_INSTRUCTIONS` maps each measure to a specific reflection directive) → retry

Capped at 2 refine attempts. Telemetry is captured (`refinement_attempts`, `refinement_reasons`) and persisted on the cell so you can see which cells the model struggled with.

### Sentence-count heuristic

Uses an abbreviation-aware regex that won't split on "e.g." or "i.e." (Python fixed-width lookbehind). Edge case: doesn't handle trailing-period honorifics like "Dr." perfectly, but acceptable for the smoke check.

### JSON extraction (`_extract_json` / `_candidate_blocks`)

Bedrock occasionally wraps JSON in prose or fences. The extractor tries candidates in order:
1. Every fenced code block (` ```json ... ``` `)
2. The bracket-balanced slice of the full text (handles "Sure, here is the JSON:\n{…}")
3. The raw stripped text

Returns the first candidate that parses cleanly. If none parse, returns the most-promising candidate so `json.loads` produces a useful error.

### Two judge types

| Judge | When | What it scores | Where stored |
|---|---|---|---|
| **Cycle judge** | After mediation on cycles that will continue looping | The mediation reasoning itself | `cycle_judge_scores` |
| **Output judge** | After final generation | The complete output document | `output_judge_scores` |

Splitting these prevents fragile `judge_scores[-1]` indexing — the regen loop reads `output_judge_scores`, the plateau check reads `cycle_judge_scores`. They don't collide.

Both score 4 dimensions on 1-5: `empathy`, `consensus`, `escalation_risk`, `cultural_insight`.

A computed `judge_scores` property merges both lists for backwards-compat reads.

### Output regeneration loop

If `empathy < threshold` or `cultural_insight < threshold` (default threshold = 3) and `regen_attempts < max_regen` (default 1), the output is regenerated. The previous output judge's `reasoning` is passed back as `evaluation_feedback` so the model knows what to fix.

Max one regeneration by default — prevents runaway loops.

---

## 7. Convergence — multiple signals

Two ways the loop terminates besides `max_cycles`:

1. **Mediator-declared convergence** — the mediation LLM sets `converged: true` when this cycle's conflicts aren't meaningfully different from the previous cycle. Primary signal.
2. **Consensus plateau** — if the last two cycle judges both report `consensus >= consensus_plateau_threshold` (default 4) and differ by ≤1, the loop terminates even if the mediator didn't self-declare. Secondary signal that catches the case where the mediator stays cautious but the judge sees the consensus stabilizing.

**Note for the meeting:** the plateau signal needs ≥2 cycle judges to fire, and the terminating cycle is never judged in-loop, so plateau effectively requires `max_cycles >= 3`. The default smoke test in `main.py` uses `max_cycles=2`, so plateau is dormant in the default run.

---

## 8. Cost / latency guardrails (`budget.py`)

Per-scenario telemetry attached to `SessionMemory`:

| Field | Default | What it tracks |
|---|---|---|
| `llm_call_budget` | 1000 | Hard cap on LLM calls per scenario |
| `llm_calls_made` | 0 | Running counter |
| `total_latency_ms` | 0 | Wall-clock sum |
| `consecutive_failures` | 0 | Resets on success |
| `max_consecutive_failures` | 5 | Trips `CircuitOpen` |

`call_llm` calls `check_budget_before_call` (raises `BudgetExceeded`), `record_success(latency_ms)`, or `record_failure` (raises `CircuitOpen` if cap exceeded). Plus boto3's internal retry layer is disabled (`Config(retries={"max_attempts": 1})`) so app-level retries don't get double-billed.

The mechanism uses a `ContextVar` registered by `main.run_scenario` so it's per-scenario without globals.

**Known limitation to flag in the meeting:** `ContextVar`s do *not* automatically propagate from the parent thread to `ThreadPoolExecutor` workers. That means the budget check and circuit breaker currently no-op inside cell calls — only mediation, reformulation, node update, and judges (which run on the main thread) are counted. Fix is to pass `memory` explicitly into the worker and `set_active_memory(memory)` inside `_ct_cell` / `_challenger_cell`. Single-file change; not in yet.

---

## 9. Persistence

`main.persist_session` writes `outputs/{scenario_id}.json` containing the entire `SessionMemory` — grids, mediation history, judge scores, final document, and budget telemetry. One file per scenario.

Set `output_dir=None` in `run_scenario` to skip persistence (for tests / ad-hoc runs).

---

## 10. Configuration surface

| Env var | Default | Purpose |
|---|---|---|
| `BEDROCK_MODEL_ID` | Sonnet 4.5 | Main agent model |
| `CHALLENGER_MODEL_ID` | Haiku 4.5 | Challenger model |
| `AWS_REGION` | `us-east-1` | Bedrock region |
| `BEDROCK_CONNECT_TIMEOUT_S` | 10 | Network connect timeout |
| `BEDROCK_READ_TIMEOUT_S` | 120 | Response read timeout |
| `THIRDSPACE_CELL_WORKERS` | 8 | Parallel cell calls per phase |

Per-scenario knobs on `SessionMemory`:

| Field | Default | Purpose |
|---|---|---|
| `max_cycles` | 3 | Loop cap |
| `max_regen` | 1 | Output regenerations cap |
| `output_score_threshold` | 3 | Below → regenerate |
| `consensus_plateau_threshold` | 4 | Plateau cutoff |
| `llm_call_budget` | 1000 | Hard call cap |
| `max_consecutive_failures` | 5 | Circuit-breaker cap |

---

## 11. Things added beyond the original spec (defensible decisions)

The original `CLAUDE.md` build instructions describe a minimum viable pipeline. The current implementation extends it in several ways. Each is justified — be ready to explain why on each:

1. **`run_node_update` (cycle integration).** The original spec ran the loop but didn't actually fold mediation back into The Occurring, making subsequent cycles redundant. Now mediation's `updated_assumptions` and `wholeness` are literally applied to the nodes between cycles.

2. **Cycle judge in addition to output judge.** Original spec only judged the final document. Adding a per-cycle judge gives a secondary convergence signal (consensus plateau) and surfaces mediation quality issues before generation.

3. **Output regeneration loop.** If the output judge scores empathy/insight below threshold, regenerate once with judge feedback as a corrective. Single-shot generation gave no recovery path.

4. **Active challenger perspective.** Initially seeded from `challenger_cultural_frame`; refined each cycle from mediation's `updated_challenger_perspective`. Lets the challenger sharpen its frame as the loop progresses.

5. **Challenger argumentation style separated from cultural identity.** Per design slide: the mediator sets *how* the challenger argues (procedural adversarial), independent of *what* (the cultural frame).

6. **Separate model SKU for the Challenger.** Cheapest path to "separate LLM" without a fine-tune — Haiku vs Sonnet draws from a different checkpoint.

7. **Cell parallelization.** 42 cells × 2 agents × 3 cycles = 252 cells. Serial = serial * cell latency. Parallel = ~ceil(252/8) * cell latency. Order-of-magnitude latency win.

8. **Refinement telemetry on cells.** `refinement_attempts` and `refinement_reasons` per cell. Without this, refinement failures are silent — you can't tell which prompts are weak.

9. **Cost / latency / circuit-breaker guardrails.** A 100-cell scenario at $0.003/call is not catastrophic, but a runaway loop with no cap is. Belt and suspenders.

10. **Session persistence.** Outputs were originally `print()`-only. Now every scenario state is reconstructable from disk.

11. **Robust JSON extractor.** Original used naïve `cleaned.startswith("```")` heuristic. New version handles fenced blocks, prose wrappers, and bare JSON in the same code path.

12. **Strict facet-set validation.** The original `assert len(nodes) == 6` would pass on duplicates and unknown facet names. The new validator surfaces all three failure modes with explicit messages.

---

## 12. Things deliberately *not* included

Per build instructions:

- **`ContextPayload.assemble`** from Omid's update — string concatenation with a buggy `default_factory=field`. Mediation prompt already handles context assembly.
- **`run_cultural_conflict_pipeline`** function from the same update — replaced by `graph.py` proper LangGraph orchestration.
- **RAG retrieval for the Challenger** — design seam exists (was a stub function, now removed and noted as a future hookup point in `run_challenger`). No corpus or retriever to plug in yet.

---

## 13. Known limitations / what's next

Worth being honest about in the meeting:

| Limitation | Severity | Fix difficulty |
|---|---|---|
| `ContextVar` doesn't propagate into `ThreadPoolExecutor` workers → cell calls bypass budget/circuit-breaker | Medium — telemetry undercounts by ~95%, runaway protection on cells is off | Small — pass `memory` into worker, `set_active_memory` inside | 
| No tests | Medium | Medium — `tests/` with one fixture per phase |
| RAG / separate fine-tuned Challenger model is design-supported but not implemented | By design — explicit hookup point | Large — needs corpus + retriever |
| Plateau check is dormant at `max_cycles=2` | Low — works at `max_cycles=3+` | None |
| Sentence-count regex misses trailing-period abbreviations (e.g., "Dr.") | Trivial | Trivial if it matters |
| `judge_scores` computed property duplicates content in persisted JSON | Cosmetic | One-line `repr=False` / serialization-excluded |

---

## 14. Architectural rules (the non-negotiables)

If anyone proposes a change in the meeting, these are the rules that should not be broken:

1. **One LLM call per cell.** 42 separate calls per agent per cycle. Never batch.
2. **`call_llm_json_with_refinement` for all cell calls.** Reformulation and mediation use plain `call_llm_json` (they don't produce cells).
3. **`linearize_grid_to_matrix_map` before passing to the Challenger.** The challenger reads `matrix[facet][dimension]["content"]`.
4. **`compress_historical_context` before the mediation prompt.** Older cycles are gisted, most recent is verbatim.
5. **Constrained generation receives only distilled wholeness nodes** (+ aggregated implications). No full grids.
6. **Mediation receives full history; compression happens *inside* `run_mediation`**, not in the graph node.
7. **Bedrock client is a `@lru_cache(maxsize=1)` singleton.** No ad-hoc `boto3.client()` calls.
8. **No raw integers for grid sizes.** Use `NUM_FACETS`, `NUM_DIMENSIONS`, `NUM_CELLS`.

---

## 15. Demo flow (suggested)

If you need to run live:

1. Show `main.py` — Canadian/Iranian feedback scenario, `max_cycles=2`.
2. `python main.py` — runs end-to-end (~3-5 minutes with parallelization).
3. Open `outputs/{scenario_id}.json` to walk through:
   - The Occurring after reformulation
   - One main grid cell + one challenger cell at the same coordinates (show how the challenger names the universal-norm assumption)
   - A mediation `implication` showing the structural insight
   - The wholeness nodes that fed back into the next cycle
   - The cycle judge → final output → output judge → (if triggered) regeneration
4. Show the final 5-section document.

If anything fails live: the persisted JSON for a previous good run is your fallback.

---

## 16. One-line elevator pitch

> ThirdSpace is a five-phase reasoning pipeline that decomposes a one-sided account of a cultural conflict into a structured 6×7 grid, runs two LLMs with different cultural frames across every cell, mediates between them to surface assumptions neither party would see, loops until either convergence or a budget is hit, and writes a final analysis whose every claim must trace back to a mediation finding.
