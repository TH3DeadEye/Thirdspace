"""Aggregate eval results from eval/results/ into readable metrics. Offline.

Reads whatever exists:
  frame_eval.jsonl -> per-dimension accuracy + confusion counts
  ablation.jsonl   -> accuracy per arm (mean±std across repeats),
                      per-country parity, leak-phrase regression check

Usage:
    python eval/report.py
"""
from __future__ import annotations

import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.harness import ResumeLog, results_path  # noqa: E402


def report_frame_eval() -> None:
    rows = ResumeLog(results_path("frame_eval.jsonl")).load_all()
    if not rows:
        print("frame_eval.jsonl: no results yet")
        return
    print(f"\n== Frame detection vs Hofstede gold ({len(rows)} runs) ==")
    dim_totals: dict[str, Counter] = defaultdict(Counter)
    confusion: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        for dim, d in row["per_dimension"].items():
            dim_totals[dim]["n"] += 1
            dim_totals[dim]["main"] += bool(d["main_match"])
            dim_totals[dim]["either"] += bool(d["either_match"])
            confusion[dim][f"gold={d['gold']} -> main={d['main']}"] += 1

    print(f"{'dimension':<24}{'main-frame acc':>15}{'either-frame acc':>18}")
    overall_n = overall_main = overall_either = 0
    for dim, c in sorted(dim_totals.items()):
        print(f"{dim:<24}{c['main']/c['n']:>14.1%}{c['either']/c['n']:>17.1%}  (n={c['n']})")
        overall_n += c["n"]; overall_main += c["main"]; overall_either += c["either"]
    if overall_n:
        print(f"{'OVERALL':<24}{overall_main/overall_n:>14.1%}{overall_either/overall_n:>17.1%}")

    print("\nConfusion (gold -> detected main), top entries per dimension:")
    for dim, c in sorted(confusion.items()):
        top = ", ".join(f"{k} x{v}" for k, v in c.most_common(3))
        print(f"  {dim}: {top}")


def _acc(rows: list[dict]) -> float | None:
    scored = [r for r in rows if r.get("prediction")]
    if not scored:
        return None
    return sum(r["prediction"] == r["gold"] for r in scored) / len(scored)


def report_ablation() -> None:
    rows = ResumeLog(results_path("ablation.jsonl")).load_all()
    if not rows:
        print("ablation.jsonl: no results yet")
        return
    print(f"\n== Single-agent vs pipeline ablation ({len(rows)} runs) ==")
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_arm[r.get("arm", "?")].append(r)

    for arm, arm_rows in sorted(by_arm.items()):
        errors = sum(1 for r in arm_rows if r.get("error"))
        unparsed = sum(1 for r in arm_rows if not r.get("prediction") and not r.get("error"))
        # Accuracy per repeat -> mean ± std across repeats.
        by_rep: dict[int, list[dict]] = defaultdict(list)
        for r in arm_rows:
            by_rep[r.get("repeat", 0)].append(r)
        rep_accs = [a for a in (_acc(v) for v in by_rep.values()) if a is not None]
        if rep_accs:
            mean = statistics.mean(rep_accs)
            std = statistics.stdev(rep_accs) if len(rep_accs) > 1 else 0.0
            print(f"\n[{arm}] accuracy: {mean:.1%} ± {std:.1%} "
                  f"({len(arm_rows)} runs, {errors} errors, {unparsed} unparseable)")
        else:
            print(f"\n[{arm}] no scoreable runs ({errors} errors)")

        # Per-country parity: accuracy spread across countries with >=2 runs.
        by_country: dict[str, list[dict]] = defaultdict(list)
        for r in arm_rows:
            by_country[r.get("country", "?")].append(r)
        country_accs = {
            c: _acc(v) for c, v in by_country.items() if len(v) >= 2 and _acc(v) is not None
        }
        if country_accs:
            lo_c = min(country_accs, key=country_accs.get)
            hi_c = max(country_accs, key=country_accs.get)
            print(f"  parity across {len(country_accs)} countries: "
                  f"min {country_accs[lo_c]:.0%} ({lo_c}), max {country_accs[hi_c]:.0%} ({hi_c}), "
                  f"gap {country_accs[hi_c]-country_accs[lo_c]:.0%}")

        if arm == "pipeline":
            calls = [r["llm_calls"] for r in arm_rows if r.get("llm_calls")]
            if calls:
                print(f"  llm calls/scenario: mean {statistics.mean(calls):.0f} "
                      f"(min {min(calls)}, max {max(calls)})")
            reasons = Counter(r.get("convergence_reason") for r in arm_rows
                              if r.get("convergence_reason"))
            if reasons:
                print(f"  convergence reasons: {dict(reasons)}")
            leaky = [r for r in arm_rows if r.get("leak_phrases_found")]
            if leaky:
                phrases = Counter(p for r in leaky for p in r["leak_phrases_found"])
                print(f"  LEAK-PHRASE REGRESSION: {len(leaky)} document(s) leaked "
                      f"internal rubric text: {dict(phrases)}")
            else:
                print("  leak-phrase check: clean (no internal rubric text in documents)")


if __name__ == "__main__":
    report_frame_eval()
    report_ablation()
