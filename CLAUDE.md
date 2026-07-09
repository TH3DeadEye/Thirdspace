# ThirdSpace — Cultural Conflict Analysis Pipeline

## Who you are building for
Arman (CS student, George Brown College). He knows Python well and has built
production systems on AWS. Do not explain basics. Write complete, working code.
The system runs on AWS Bedrock (model IDs come from env / `.env`: main + judge
on Sonnet, challenger on Haiku, Titan v2 for embeddings). Infrastructure is
already handled — focus only on application code.

> This file describes the AS-BUILT architecture. An earlier version of this
> file was a build-from-scratch spec that no longer matches the code — if an
> instruction here conflicts with what the code does, trust the code and
> `docs/TODO.md`, and update this file.

## What this system is
Not a chatbot — a reasoning pipeline. A raw, one-sided cultural conflict
description goes in; a structured five-section analysis document comes out.

As-built pipeline (LangGraph, `thirdspace/graph.py`):

0. **Frame detection** (`run_frame_detection`, 1 call) — when the caller passes
   no frames, two contrasting dimensional frames are inferred from the
   scenario, constrained to `CULTURE_DIMENSION_MENU` (Hofstede's six
   dimensions, binary values). Explicit frames and `""` (neutral mode) are
   also supported — see `run_scenario` in `main.py`.
1. **Reformulation** (1 call) — six structured "Occurring" nodes.
2. **Main agent** (42 calls, parallel workers) — one call per (facet ×
   dimension) cell, slim per-cell context, first-principles decorator,
   prior-cycle mediation guidance on cycle 2+.
3. **Challenger** (42 calls, different model SKU) — **fully blind**: never
   sees the main grid; reasons from its own frame + The Occurring only, with
   a red-team decorator. There is no challenge/reframe/confirm classification.
4. **Mediation** (1 large call) — compares all 42 cell pairs, classifies each
   conflict into the five-type taxonomy (semantic / attributional / normative
   / structural / factual) with severity + resolvability, enforced by
   Pydantic validators (`models.py` conflict registry).
5. **Cycle control** — bidirectional convergence in `should_loop`: cross-cycle
   similarity (embedding cosine or Jaccard, `convergence_metric`) is measured
   every cycle; the mediator's `converged` flag is advisory in BOTH directions
   (vetoed when content still moves; overridden to True when similarity is
   high and the conflict-type distribution is stable). How the loop ended is
   persisted as `SessionMemory.convergence_reason`. On loop: cycle judge
   (telemetry only) → node update folds findings back into The Occurring.
6. **Constrained generation** — receives distilled material only (wholeness,
   cross-cycle implications, high-severity conflicts, type distribution).
   Per-type guidance is plain prose; `models.LEAK_PHRASES` pins vocabulary
   that must never appear in the document (tested in
   `tests/test_leak_phrases.py`).
7. **Blind output judge** (5 dimensions incl. `intervention_fidelity`) with a
   one-shot regeneration loop.

## File map
```
thirdspace/
├── models.py     # Pydantic models, conflict-type registry, LEAK_PHRASES,
│                 # CultureAxis/CulturalIdentity, CULTURE_DIMENSION_MENU
├── llm.py        # Bedrock singleton client, call_llm/_json (+json_repair
│                 # recovery), structural-only refinement loop, embed_text
├── agents.py     # one function per phase; cell builders persist refinement
│                 # telemetry and reflection flags
├── graph.py      # LangGraph wiring; bidirectional should_loop
├── metrics.py    # jaccard / embedding similarity, type-distribution stability
├── budget.py     # call budget + circuit breaker + token accounting (locked)
├── main.py       # run_scenario entry point, session persistence
└── utils/        # prompts.py (ALL prompts), context.py, parsing.py, decorators
tests/            # offline unit tests (LLM callers stubbed) — run: pytest tests/
eval/             # dataset-driven evaluation harness — see eval/README.md
docs/TODO.md      # single source of truth for what's built and what's left
```

## Architecture rules (still binding)
1. **One LLM call per cell.** Main agent and challenger each make `NUM_CELLS`
   separate calls. Never batch cells into one call.
2. **The challenger stays blind.** It must never receive the main agent's grid
   or frame. (The old `linearize_grid_to_matrix_map` flow is dead.)
3. **Refinement retries are structural-only and context-complete.** Retry only
   on empty/too-short content; the retry prompt must contain the original user
   prompt and the previous response. `reflection_flags` are telemetry, never a
   retry trigger.
4. **Constrained generation gets distilled material only** — never full grids.
5. **No rubric vocabulary in generated documents.** Anything in
   `models.LEAK_PHRASES` must stay out of prompts' labeled-block style and out
   of outputs; keep the unit test and the harness grep green.
6. **boto3 client is a singleton** (`get_client()` with `lru_cache`). Never
   construct clients elsewhere.
7. **No raw grid integers.** Use `NUM_FACETS`, `NUM_DIMENSIONS`, `NUM_CELLS`.
8. **Malformed LLM JSON**: `call_llm_json` tries strict parsing then
   `json_repair`; if both fail, log the raw response and raise. Never continue
   silently with partial data.

## Evaluation (eval/)
Zero-spend by default: every runner is a dry run printing call counts + cost;
real runs need `--confirm-spend`. Label maps in `eval/label_maps/` are frozen
(pre-registered) — never edit a versioned map; create `_v2` and re-run.
Workflow and method notes: `eval/README.md`.

## Testing
`./venv/bin/python -m pytest tests/` — fully offline (LLM callers stubbed),
125 tests at last update. Add tests alongside behavior changes; fixtures in
`tests/fixtures.py`.
