"""Score Phase-0 frame detection against Hofstede-derived gold frames.

For each sampled NormAd scenario (country known), run_frame_detection makes
ONE LLM call and returns two contrasting dimensional frames. We score the
detector's MAIN frame per-dimension against the country's real Hofstede
values (eval/culture_scores.py), and separately report a best-of-either-frame
match as a secondary, more lenient metric.

Method caveat (pre-registered): NormAd stories are single-culture etiquette
narratives, not two-party conflicts, so the detector is running off its
design distribution. The main-frame score is the primary number; if it is low
while either-frame is high, the detector finds the culture but assigns it to
the wrong role.

Usage:
    python eval/run_frame_eval.py                 # dry run (default)
    python eval/run_frame_eval.py --limit 20 --confirm-spend
    python eval/report.py                          # aggregate results
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

# Rough per-call token shape for the dry-run estimate: scenario + dimension
# menu prompt in, two frames + rationale out.
EST_IN_TOKENS = 1600
EST_OUT_TOKENS = 400


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()

    gold = load_gold(require_axes=True)
    sample = sample_stratified(gold, args.limit, args.seed)
    total_calls = len(sample) * args.repeats

    if not args.confirm_spend:
        cost = estimate_cost_usd(total_calls, EST_IN_TOKENS, EST_OUT_TOKENS, "sonnet")
        dry_run_exit([
            f"Gold records with Hofstede frames: {len(gold)}",
            f"Sampled scenarios: {len(sample)} (stratified by country, seed={args.seed})",
            f"LLM calls: {total_calls} (1 frame-detection call per scenario per repeat)",
            f"Estimated cost: ${cost:.2f} (sonnet tier, ~{EST_IN_TOKENS} in / {EST_OUT_TOKENS} out tokens/call)",
        ])

    # Import inside the paid path so dry runs never touch boto3/Bedrock config.
    from agents import run_frame_detection  # noqa: E402

    log = ResumeLog(results_path("frame_eval.jsonl"))
    ran = skipped = 0
    for rec in sample:
        for rep in range(args.repeats):
            key = f"{rec['id']}:rep{rep}"
            if log.done(key):
                skipped += 1
                continue
            t0 = time.monotonic()
            detected = run_frame_detection(
                raw_scenario=rec["story"],
                scenario_id=f"frameeval-{rec['id']}-r{rep}",
            )
            latency_s = time.monotonic() - t0

            gold_axes = {a["dimension"]: a["value"] for a in rec["gold_axes"]}
            main_axes = {a.dimension: a.value for a in detected.main}
            chall_axes = {a.dimension: a.value for a in detected.challenger}

            per_dim = {}
            for dim, gold_val in gold_axes.items():
                main_val = main_axes.get(dim)
                per_dim[dim] = {
                    "gold": gold_val,
                    "main": main_val,
                    "challenger": chall_axes.get(dim),
                    "main_match": main_val == gold_val,
                    "either_match": gold_val in (main_val, chall_axes.get(dim)),
                }

            log.record(key, {
                "id": rec["id"],
                "country": rec["country"],
                "repeat": rep,
                "per_dimension": per_dim,
                "rationale": detected.rationale,
                "latency_s": round(latency_s, 2),
            })
            ran += 1
            matches = sum(d["main_match"] for d in per_dim.values())
            logger.info(
                f"[{rec['country']}/{rec['id']}] main-frame {matches}/{len(per_dim)} "
                f"dims correct ({latency_s:.1f}s)"
            )

    print(f"\nDone: {ran} run, {skipped} resumed from {log.path}")
    print("Aggregate with: python eval/report.py")


if __name__ == "__main__":
    main()
