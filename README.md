# ThirdSpace — Cultural Conflict Resolution System

A multi-stage reasoning pipeline that takes a raw, one-sided description of a cultural conflict and returns a structured analysis. Not a chatbot. One scenario in, one document out.

Built on **AWS Bedrock** (Claude Sonnet 4.5 + Claude Haiku 4.5), **LangGraph**, and **Pydantic v2**.

---

## How it works

```
raw scenario
    │
    ▼
[Phase 1] Reformulation
    Decomposes the scenario into 6 structured nodes — "The Occurring"
    │
    ▼
[Phase 2] Critical Thinking  (Main Agent — Sonnet 4.5)
    Reasons across a 6 × 7 grid = 42 cells, one LLM call per cell
    │
    ▼
[Phase 3] Challenger  (Haiku 4.5 — different priors, same prompts)
    Argues the opposing cultural frame across the same 42 cells
    │
    ▼
[Phase 4] Mediation
    Compares grids, finds agreements and conflicts, extracts implications
    │
    ├── not converged + cycles remain → update The Occurring → loop to Phase 2
    │
    ▼
[Phase 5] Constrained Generation
    Writes the 5-section final document using only what mediation found
    │
    ▼
[Judge] LLM-as-a-judge scores empathy / consensus / escalation / insight
    │
    ├── score below threshold → regenerate with judge feedback
    │
    ▼
final output + outputs/{scenario_id}.json
```

Each cycle runs ~85 LLM calls (42 main + 42 challenger + mediation + node update + in-loop judge). The 84 cell calls per cycle are parallelized with `ThreadPoolExecutor`.

---

## The 6 × 7 grid

### Six facets (rows) — "The Occurring"

| Facet | What it captures |
|---|---|
| `individuals` | Each person as they are already affected by the conflict, shaped by culture and relational history. Not neutral bios. |
| `interactional_event` | What a camera would have recorded. No interpretation. |
| `process` | Timeline, context, and each party's model of time. |
| `evidence_and_data` | Only what can actually be verified. |
| `concepts_theories_rules` | The cultural frameworks each party is applying, usually without knowing it. Named for both sides. |
| `inferences` | What the structure of nodes 1–5 implies. Often contradicts the narrator's framing. |

### Seven dimensions (columns) — critical-thinking lenses

`objective` · `query_formulation` · `perspective` · `assumptions` · `evidence` · `concepts` · `implications`

Every cell is a focused micro-reasoning task: "reason about the **individuals** facet through the **assumptions** lens." Narrowness is the point — batched reasoning collapses into mush. Each cell is its own LLM call.

---

## Project structure

```
thirdspace/
├── models.py     — Pydantic data models and grid constants
├── llm.py        — Bedrock client, all prompts, JSON extraction, self-refinement wrapper
├── agents.py     — One function per pipeline phase
├── graph.py      — LangGraph StateGraph wiring and routing logic
├── main.py       — Entry point and session persistence
├── budget.py     — Per-scenario cost / latency / circuit-breaker guardrails
└── requirements.txt

tests/
├── fixtures.py       — Reusable mock data builders
├── test_agents.py    — Unit tests for every phase function (LLM-stubbed)
├── test_budget.py    — Budget / circuit-breaker tests including parallel propagation
└── test_graph.py     — Integration tests for the LangGraph wiring

outputs/              — Created at runtime; one JSON file per scenario
```

---

## Setup

### Requirements

- Python 3.11+
- AWS account with Bedrock access
- Claude Sonnet 4.5 and Claude Haiku 4.5 enabled in Bedrock Model Access

### Install

```bash
pip install -r thirdspace/requirements.txt
```

### AWS credentials

```bash
aws configure          # access key + us-east-1
# or
aws sso login --profile <your-profile>
```

### Enable Bedrock models

In the AWS Console: `Bedrock → Model access → Request access` for:
- `Claude Sonnet 4.5`
- `Claude Haiku 4.5`

### Run the built-in sample scenario

```bash
cd thirdspace
python main.py
```

Runs the Canadian/Iranian code-review conflict scenario with `max_cycles=2`. Outputs the analysis to stdout and writes the full session to `outputs/{scenario_id}.json`.

---

## Run a custom scenario

```python
from thirdspace.main import run_scenario

memory = run_scenario(
    raw_scenario="""
        Your scenario here. Be specific and one-sided — that is the input.
        Include what was observed, what was assumed, and what was felt.
    """,
    main_cultural_frame="<narrator's cultural lens>",
    challenger_cultural_frame="<opposing cultural lens>",
    max_cycles=3,
)

print(memory.final_output)
print(f"Calls: {memory.llm_calls_made}, Latency: {memory.total_latency_ms / 1000:.1f}s")
```

---

## Run tests

No AWS credentials needed — all LLM calls are stubbed.

```bash
python -m unittest discover -s tests -v
```

41 tests, ~0.1 seconds.

---

## Configuration

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `BEDROCK_MODEL_ID` | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | Main agent model |
| `CHALLENGER_MODEL_ID` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Challenger model |
| `AWS_REGION` | `us-east-1` | Bedrock region |
| `BEDROCK_CONNECT_TIMEOUT_S` | `10` | Connection timeout |
| `BEDROCK_READ_TIMEOUT_S` | `120` | Response read timeout |
| `THIRDSPACE_CELL_WORKERS` | `8` | Parallel cell calls per phase |

### Per-scenario knobs (on `SessionMemory`)

| Field | Default | Purpose |
|---|---|---|
| `max_cycles` | `3` | Maximum reasoning cycles |
| `max_regen` | `1` | Maximum output regenerations |
| `output_score_threshold` | `3` | Judge score below this → regenerate |
| `consensus_plateau_threshold` | `4` | Secondary convergence signal |
| `llm_call_budget` | `1000` | Hard cap on LLM calls per scenario |
| `max_consecutive_failures` | `5` | Circuit-breaker threshold |

---

## Output format

The final document has five sections:

1. **What Actually Happened** — mediation's version, not the narrator's
2. **Where the Cultures Diverged** — what each frame expected and why the misread occurred
3. **What the Conflict Actually Implies** — pattern-level findings from mediation
4. **What Each Person Likely Experienced** — empathy for both sides without blame
5. **How to Move Forward** — grounded only in what mediation found; no generic advice

Every claim traces back to a mediation finding. The generation prompt explicitly forbids introducing new inferences from training data.

---

## Session persistence

Every run writes `outputs/{scenario_id}.json` containing the entire `SessionMemory`:
- The Occurring after each cycle
- Every reasoning grid (all 42 cells per agent per cycle)
- Full mediation history
- Cycle and output judge scores
- Final document
- Budget telemetry (`llm_calls_made`, `total_latency_ms`)

Pass `output_dir=None` to `run_scenario` to skip persistence.

---

## Architecture decisions

**One LLM call per cell — never batched.** Both agents make 42 separate calls per cycle. Batching collapses focus; each cell's call is deliberately narrow.

**Two models, different priors.** The challenger uses Haiku 4.5 by default. Same prompts, different checkpoint = different blind spots. Cheapest path toward the design requirement of a "separate LLM" without a fine-tune.

**Contextvars for budget propagation.** `budget.py` uses a `ContextVar` so `call_llm` can increment counters from anywhere in the call stack. Worker threads re-register the active memory explicitly via `with active_memory(memory):` before each cell call — because `ThreadPoolExecutor` workers do not inherit the parent thread's context.

**The grids are scratch work, not memory.** Only what mediation distills from them (the Occurring update + wholeness directives) carries forward. The raw cell content from cycle 1 does not influence cycle 2 directly.

**Constrained generation is hard-constrained.** The final writer receives only the distilled wholeness nodes and aggregated implications — not the full grids. It cannot regress into generic cultural advice from training data.

---

## What's not implemented yet

- **RAG for the Challenger** — the hookup point is `_challenger_cell` in `agents.py`. A retrieval layer over a cultural-ethnography corpus would significantly strengthen the challenge phase.
- **Fine-tuned Challenger model** — currently differentiated by SKU (Haiku vs Sonnet). A truly fine-tuned model on cultural argumentation data would be the next level.

---

## Dependencies

```
boto3>=1.34.0
langgraph>=0.2.0
pydantic>=2.0.0
```
