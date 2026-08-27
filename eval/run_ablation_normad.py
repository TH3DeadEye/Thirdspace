"""Single-agent vs full-pipeline ablation on NormAd-Eti (Pairing 4).

The most consequential open question in the design docs: does the
main+challenger+mediation architecture actually judge cultural scenarios
better than one model call, or does it only produce different text? This
mirrors the published single-agent vs multi-agent-debate design (Ki et al.,
ACL 2025) on the same dataset, scoring both accuracy and per-country parity.

Arms (both receive identical inputs — story + country; the rule-of-thumb is
withheld from both, since it nearly encodes the answer):
  single    one LLM call producing an adhere/violate/neither judgment
  pipeline  the full ThirdSpace run (frames read from the dataset record),
            plus one call reducing the final document to the same ternary
            judgment; the document is also grep'd for LEAK_PHRASES

Usage:
    python eval/run_ablation_normad.py                        # dry run
    python eval/run_ablation_normad.py --arm single --limit 15 --confirm-spend
    python eval/run_ablation_normad.py --arm both --limit 15 --confirm-spend
    python eval/report.py
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.harness import (  # noqa: E402
    add_common_args, load_gold, sample_stratified, estimate_cost_usd,
    dry_run_exit, ResumeLog, results_path,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

JUDGMENTS = ("adhere", "violate", "neither")  # matches label_maps/normad_v1.yaml

# How many cycles the pipeline arm runs. The loop policy is fixed-N (see
# SessionMemory.cycle_policy), so this is the whole cost story: each cycle is
# 2*NUM_CELLS cell calls plus one mediation.
PIPELINE_CYCLES = 2

# Dry-run cost model. Single arm: one small call.
#
# Pipeline arm, per cycle: NUM_CELLS main (Sonnet) + NUM_CELLS challenger (Haiku)
# + 1 mediation (Sonnet). Plus, once: reformulation and generation. Plus, per
# extra cycle: one node update and two convergence embeddings. Plus the verdict
# call this runner makes itself.
#
# Reference point: a pre-rewrite run of the same scenario cost 130 calls /
# 401K tokens for ONE cycle, with 50% of cells paying for a refinement retry.
# The retry fix removes that overhead and the grid went 42 -> 30 cells.
EST_SINGLE_IN, EST_SINGLE_OUT = 700, 250
EST_PIPELINE_CALLS = (
    1                                        # reformulation
    + PIPELINE_CYCLES * (2 * 30 + 1)         # cells + mediation per cycle
    + (PIPELINE_CYCLES - 1) * 3              # node update + 2 embeddings per loop
    + 1                                      # constrained generation
    + 1                                      # this runner's verdict call
)
EST_PIPELINE_USD = 0.65 * PIPELINE_CYCLES

SINGLE_SYSTEM = """\
You judge whether an action described in a story is appropriate under the
etiquette norms of the culture the story is set in.

Classify the central action as exactly one of:
- "adhere": the action is consistent with that culture's norms
- "violate": the action transgresses that culture's norms
- "neither": the described norms are not really engaged by the action

Respond ONLY with valid JSON. No preamble. No markdown fences.
Return: {"judgment": "adhere" | "violate" | "neither", "reasoning": "<2-3 sentences>"}
"""

VERDICT_SYSTEM = """\
You reduce a cultural conflict analysis document to a single verdict about
the central action it analyzes.

Classify that action as exactly one of:
- "adhere": the analysis concludes the action was consistent with the local culture's norms
- "violate": the analysis concludes the action transgressed the local culture's norms
- "neither": the analysis concludes the norms were not really engaged

Base the verdict ONLY on what the document says.
Respond ONLY with valid JSON. No preamble. No markdown fences.
Return: {"judgment": "adhere" | "violate" | "neither", "reasoning": "<1-2 sentences>"}
"""


def _single_user(story: str, country: str) -> str:
    return (
        f"Story (set in {country}):\n{story}\n\n"
        f"Given the etiquette norms of {country}, classify the central action."
    )


def _parse_judgment(result: dict) -> str | None:
    j = str(result.get("judgment", "")).strip().lower()
    return j if j in JUDGMENTS else None


def run_single_arm(rec: dict) -> dict:
    from llm import call_llm_json
    t0 = time.monotonic()
    result = call_llm_json(SINGLE_SYSTEM, _single_user(rec["story"], rec["country"]))
    return {
        "prediction": _parse_judgment(result),
        "reasoning": result.get("reasoning", ""),
        "latency_s": round(time.monotonic() - t0, 2),
        "llm_calls": 1,
    }


def run_pipeline_arm(rec: dict, sessions_dir: Path) -> dict:
    from llm import call_llm_json
    from main import run_scenario
    from models import LEAK_PHRASES
    from utils.prompts import read_cultural_frames
    from verify import verify_session

    # The frames come from the dataset labels, as a controlled variable. They used
    # to be smuggled into the prose as "(Setting: {country}) ..." with BOTH agents
    # left unframed — which meant the ablation was testing an unframed pipeline in
    # a study whose independent variable is the frame.
    main_frame, challenger_frame = read_cultural_frames(rec, "normad")

    t0 = time.monotonic()
    memory = run_scenario(
        raw_scenario=rec["story"],
        main_cultural_frame=main_frame,
        challenger_cultural_frame=challenger_frame,
        max_cycles=PIPELINE_CYCLES,
        output_dir=sessions_dir,
    )
    doc = memory.final_output or ""
    verdict = call_llm_json(
        VERDICT_SYSTEM,
        f"DOCUMENT:\n{doc}\n\nClassify the central action.",
    )
    doc_lower = doc.lower()
    leaks = [p for p in LEAK_PHRASES if p in doc_lower]
    return {
        "prediction": _parse_judgment(verdict),
        "reasoning": verdict.get("reasoning", ""),
        "latency_s": round(time.monotonic() - t0, 2),
        "llm_calls": memory.llm_calls_made + 1,
        "total_tokens": memory.total_tokens,
        "convergence_reason": memory.convergence_reason,
        "leak_phrases_found": leaks,
        "session_id": memory.scenario_id,
        "main_frame": main_frame,
        "challenger_frame": challenger_frame,
        # Per-scenario health, so a damaged run is visible in the results file
        # rather than only in a log nobody reads afterwards.
        "cycles": len(memory.mediation_history),
        "refinement_rate": round(memory.refinement_rate, 3),
        "conflicts_dropped": memory.conflicts_dropped,
        "verification": verify_session(memory),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--arm", choices=["single", "pipeline", "both"], default="both",
        help="Which arm(s) to run (default both).",
    )
    args = parser.parse_args()

    gold = load_gold(require_axes=False)
    sample = sample_stratified(gold, args.limit, args.seed)
    arms = ["single", "pipeline"] if args.arm == "both" else [args.arm]

    if not args.confirm_spend:
        lines = [
            f"Gold records: {len(gold)}",
            f"Sampled scenarios: {len(sample)} (stratified by country, seed={args.seed}) "
            f"x {args.repeats} repeat(s)",
        ]
        n = len(sample) * args.repeats
        if "single" in arms:
            c = estimate_cost_usd(n, EST_SINGLE_IN, EST_SINGLE_OUT, "sonnet")
            lines.append(f"single arm:   {n} calls, est. ${c:.2f}")
        if "pipeline" in arms:
            lines.append(
                f"pipeline arm: ~{n * EST_PIPELINE_CALLS} calls, "
                f"est. ${n * EST_PIPELINE_USD:.2f} (~${EST_PIPELINE_USD:.2f}/scenario "
                f"at max_cycles={PIPELINE_CYCLES}; budget breaker caps each run)"
            )
        dry_run_exit(lines)

    sessions_dir = results_path("sessions")
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log = ResumeLog(results_path("ablation.jsonl"))

    ran = skipped = 0
    for rec in sample:
        for arm in arms:
            for rep in range(args.repeats):
                key = f"{rec['id']}:{arm}:rep{rep}"
                if log.done(key):
                    skipped += 1
                    continue
                try:
                    result = (
                        run_single_arm(rec) if arm == "single"
                        else run_pipeline_arm(rec, sessions_dir)
                    )
                except Exception as e:
                    logger.error(f"[{key}] failed: {e}")
                    log.record(key, {
                        "id": rec["id"], "country": rec["country"], "arm": arm,
                        "repeat": rep, "gold": rec["gold_label"],
                        "prediction": None, "error": str(e),
                    })
                    continue
                log.record(key, {
                    "id": rec["id"], "country": rec["country"], "arm": arm,
                    "repeat": rep, "gold": rec["gold_label"], **result,
                })
                ran += 1
                logger.info(
                    f"[{key}] gold={rec['gold_label']} pred={result['prediction']} "
                    f"({result['latency_s']}s, {result['llm_calls']} calls)"
                )

    print(f"\nDone: {ran} run, {skipped} resumed from {log.path}")
    print("Aggregate with: python eval/report.py")


if __name__ == "__main__":
    main()
