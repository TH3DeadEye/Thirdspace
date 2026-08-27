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

**Frames are supplied by the caller — there is no inference step.** Evaluation
runs read them from the dataset record (`prompts.read_cultural_frames`: NormAd →
country, MDD → active/passive agent). `""` means neutral mode, where the prompt
drops the frame block entirely rather than rendering an empty one.

1. **Reformulation** (1 call) — five structured "Occurring" nodes, each capped
   at 120 words in the prompt. Nothing downstream truncates them.
2. **Main agent** (30 calls, parallel workers) — one call per (facet ×
   dimension) cell, per-cell context, first-principles decorator, prior-cycle
   mediation guidance on cycle 2+.
3. **Challenger** (30 calls, different model SKU) — **reactive, not blind**. It
   never sees the main grid, but it always contests a specific prior artifact:
   cycle 1 → The Occurring's own stated assumptions; cycle 2+ → the mediator's
   Wholeness. Red-team decorator.
4. **Mediation** (1 large call) — compares all 30 cell pairs (every pair; no
   pre-filtering), classifies each conflict into the six-type taxonomy (factual
   / procedural / interpretive / power_related / identity_related / normative)
   with severity + resolvability, enforced by Pydantic validators (`models.py`
   conflict registry). Frame-blind and label-blind: readings are shuffled A/B,
   and the un-shuffle keys on the (facet, dimension) WE generated — never on the
   id string the model echoes back.
5. **Cycle control** — `SessionMemory.cycle_policy` selects the rule:
   - `"fixed"` (default) runs exactly `max_cycles` and records the convergence
     measures as telemetry on `stability_log` without routing on them.
   - `"convergence"` treats the mediator's `converged` flag as advisory in BOTH
     directions (vetoed when content still moves; overridden to True when
     similarity is high and the conflict profile is stable).
   How the loop ended is persisted as `SessionMemory.convergence_reason`. On
   loop: node update folds findings back into The Occurring, and the
   challenger's `WorkingPerspective` accumulates the mediator's
   `updated_perspectives`.
6. **Constrained generation** — receives distilled material only (wholeness,
   cross-cycle implications, high-severity conflicts, type distribution), plus
   the interventions for every conflict type present, severity-weighted.
   `models.LEAK_PHRASES` pins vocabulary that must never appear in the document
   (tested in `tests/test_leak_phrases.py`, asserted per run by `verify.py`).
7. **Verification** (`verify.py`, no LLM calls) — invariant checks over the
   finished session: grid shape, ref resolution, drop counters, call accounting,
   rubric leakage. Runs automatically at the end of `run_scenario`.

The in-pipeline LLM judges are **gone** — they saturated at perfect scores and
were replaced by the external dual-classifier evaluation in `eval/`.

## File map
```
thirdspace/
├── models.py     # Pydantic models, conflict-type registry, LEAK_PHRASES,
│                 # CultureAxis/CulturalIdentity, CULTURE_DIMENSION_MENU
├── llm.py        # Bedrock singleton client, call_llm/_json (+json_repair
│                 # recovery), structural-only refinement loop, embed_text
├── agents.py     # one function per phase; cell builders persist refinement
│                 # telemetry and reflection flags
├── graph.py      # LangGraph wiring; cycle_policy router in should_loop
├── metrics.py    # jaccard / embedding similarity, type-distribution stability
├── budget.py     # call budget + circuit breaker + token accounting (locked)
├── verify.py     # post-run invariant checks; no LLM calls
├── main.py       # run_scenario entry point, session persistence
└── utils/        # prompts.py (ALL prompts), parsing.py, decorators
tests/            # offline unit tests (LLM callers stubbed) — run: pytest tests/
eval/             # dataset-driven evaluation harness — see eval/README.md
docs/TODO.md      # single source of truth for what's built and what's left
```

## Architecture rules (still binding)
1. **One LLM call per cell.** Main agent and challenger each make `NUM_CELLS`
   separate calls. Never batch cells into one call.
2. **The challenger never sees the main grid.** It is reactive — it contests
   The Occurring's assumptions (cycle 1) or the mediator's Wholeness (cycle 2+)
   — but the two grids must stay independently derived, or mediation is
   comparing a text against its own rebuttal.
3. **Refinement retries are structural-only and context-complete.** Retry only
   on empty/placeholder content — never on brevity, and never on
   `reflection_flags`. A minimum sentence count is a padding incentive that
   fights the cell prompts' own specificity rule. The retry prompt must contain
   the original user prompt and the previous response.
4. **Constrained generation gets distilled material only** — never full grids.
5. **No rubric vocabulary in generated documents.** Anything in
   `models.LEAK_PHRASES` must stay out of prompts' labeled-block style and out
   of outputs; keep the unit test and the harness grep green.
6. **boto3 client is a singleton** (`get_client()` with `lru_cache`). Never
   construct clients elsewhere.
7. **No raw grid integers.** Use `NUM_FACET`, `NUM_DIMENSION`, `NUM_CELLS`.
8. **Malformed LLM JSON**: `call_llm_json` tries strict parsing then
   `json_repair`; if both fail, log the raw response and raise. Never continue
   silently with partial data.
9. **Nothing silently edits what gets measured.** Two input optimizations were
   removed for this: a similarity filter that dropped "agreeing" cell pairs
   before mediation, and a dedupe pass that stripped sentences from one grid but
   not its pair. Both shrank the denominator of the conflict counts the
   classifiers read. Bound length in the prompt instead, and count every
   discard (`conflicts_dropped`, `refs_dropped`, `sentences_stripped`).
10. **Never key on a string the model echoed back.** Ids come back in varying
   case and spacing; normalize with `agents._norm_id`, and key internal maps on
   values the code generated.

## Evaluation (eval/)
Zero-spend by default: every runner is a dry run printing call counts + cost;
real runs need `--confirm-spend`. Label maps in `eval/label_maps/` are frozen
(pre-registered) — never edit a versioned map; create `_v2` and re-run.
Workflow and method notes: `eval/README.md`.

## Testing
`./venv/bin/python -m pytest tests/` — fully offline (LLM callers stubbed),
143 tests at last update. Add tests alongside behavior changes; fixtures in
`tests/fixtures.py`. `tests/test_bug_regressions.py` holds one test per bug
found in the August audit — each was verified to fail against the code as it
stood, so they are a real net rather than documentation.
