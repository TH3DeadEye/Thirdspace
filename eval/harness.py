"""Shared harness plumbing: sampling, spend gating, cost estimates, resume log.

Spend policy (hard requirement): every runner is a dry run unless
--confirm-spend is passed. A dry run prints the scenario count, the exact LLM
call count, and a dollar estimate, then exits without touching Bedrock.
Real runs append one JSONL record per completed scenario so an interrupted run
resumes without re-paying for finished work.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import GOLD_DIR, RESULTS_DIR  # noqa: E402

GOLD_FILE = GOLD_DIR / "normad_frames.jsonl"

# Per-MTok Bedrock pricing (input, output) by model tier, USD. Bedrock matches
# first-party per-MTok rates: Sonnet-tier $3/$15, Haiku 4.5 $1/$5. The actual
# model IDs come from env (BEDROCK_MODEL_ID etc.); these are tier-level
# estimates for the dry-run gate, not billing-grade numbers.
PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "sonnet": (3.00, 15.00),
    "haiku": (1.00, 5.00),
}


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--limit", type=int, default=20,
        help="Number of scenarios to sample (default 20).",
    )
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="Runs per scenario; report mean±std across repeats (default 1).",
    )
    parser.add_argument(
        "--seed", type=int, default=7,
        help="Sampling seed. Keep fixed so reruns hit the same scenarios.",
    )
    parser.add_argument(
        "--confirm-spend", action="store_true",
        help="Actually call Bedrock. Without this flag the runner is a dry "
             "run: it prints call counts + cost estimate and exits.",
    )


def load_gold(require_axes: bool = False) -> list[dict]:
    if not GOLD_FILE.exists():
        raise FileNotFoundError(
            f"{GOLD_FILE} not found. Run `python eval/fetch_data.py` then "
            f"`python eval/build_gold.py` first."
        )
    records = []
    with GOLD_FILE.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if require_axes and not rec.get("gold_axes"):
                continue
            records.append(rec)
    return records


def sample_stratified(
    records: list[dict], limit: int, seed: int, key: str = "country"
) -> list[dict]:
    """Deterministic sample spread across values of `key` (round-robin over
    shuffled per-country buckets), so no single culture dominates a small
    sample. Same seed + same gold file => same sample."""
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = {}
    for rec in records:
        buckets.setdefault(rec.get(key, ""), []).append(rec)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    ordered_keys = sorted(buckets)
    sample: list[dict] = []
    depth = 0
    while len(sample) < min(limit, len(records)):
        progressed = False
        for k in ordered_keys:
            if depth < len(buckets[k]):
                sample.append(buckets[k][depth])
                progressed = True
                if len(sample) >= limit:
                    break
        if not progressed:
            break
        depth += 1
    return sample


def estimate_cost_usd(
    calls: int, in_tokens_per_call: int, out_tokens_per_call: int, tier: str
) -> float:
    in_price, out_price = PRICING_PER_MTOK[tier]
    return calls * (
        in_tokens_per_call / 1_000_000 * in_price
        + out_tokens_per_call / 1_000_000 * out_price
    )


def dry_run_exit(lines: list[str]) -> None:
    """Print the spend plan and exit. The default path for every runner."""
    print("=" * 62)
    print("DRY RUN — no Bedrock calls were made.")
    for line in lines:
        print(line)
    print("Re-run with --confirm-spend to execute.")
    print("=" * 62)
    sys.exit(0)


class ResumeLog:
    """Append-only JSONL of per-scenario results. Keyed records let an
    interrupted run resume without re-paying for completed scenarios
    (TODO item S1.6, implemented at the harness level)."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._done: set[str] = set()
        if path.exists():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        self._done.add(json.loads(line)["key"])
                    except (json.JSONDecodeError, KeyError):
                        continue

    def done(self, key: str) -> bool:
        return key in self._done

    def record(self, key: str, payload: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, **payload}, ensure_ascii=False) + "\n")
        self._done.add(key)

    def load_all(self) -> list[dict]:
        out = []
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return out


def results_path(name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / name
