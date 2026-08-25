"""Turn predictions.jsonl into a report.

Separated from the runner so a report can be re-rendered from cached rows at
any time — including after re-fitting the structural thresholds — without
spending a single call.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Sequence

from . import metrics


def _pairs(rows: Sequence[dict], key: str) -> list[tuple[str, str | None]]:
    """(gold, prediction) for one system. `key` is a top-level column
    ('prediction') or a nested head ('structural', 'readout', 'baseline')."""
    out = []
    for row in rows:
        gold = row.get("gold")
        if gold is None:
            continue
        if key == "prediction":
            pred = row.get("prediction")
        else:
            head = row.get(key) or {}
            pred = head.get("label") if isinstance(head, dict) else None
        out.append((gold, pred))
    return out


def build(rows: Sequence[dict], vocab, config: dict | None = None,
          seed: int = 0) -> dict:
    labels = list(vocab.labels) if hasattr(vocab, "labels") else list(vocab["labels"])
    usable = [r for r in rows if r.get("gold") is not None]

    systems = {"thirdspace": metrics.evaluate(_pairs(usable, "prediction"), labels, seed)}
    for head in ("structural", "readout", "baseline"):
        if any(isinstance(r.get(head), dict) for r in usable):
            systems[head] = metrics.evaluate(_pairs(usable, head), labels, seed)

    comparison = {}
    if "baseline" in systems:
        comparison["thirdspace_vs_baseline"] = metrics.mcnemar(
            _pairs(usable, "prediction"), _pairs(usable, "baseline"))
    if "structural" in systems and "readout" in systems:
        comparison["structural_vs_readout_agreement"] = metrics.agreement_rate(
            [p for _, p in _pairs(usable, "structural")],
            [p for _, p in _pairs(usable, "readout")])

    errors = [r for r in usable if r.get("error")]
    sig = [r.get("signals") or {} for r in usable if r.get("signals")]

    def _mean(field: str) -> float:
        vals = [float(s.get(field, 0) or 0) for s in sig]
        return sum(vals) / len(vals) if vals else 0.0

    cost = {
        "records": len(usable),
        "failed_records": len(errors),
        "llm_calls_total": sum(int(r.get("llm_calls") or 0) for r in usable),
        "llm_calls_per_record": (
            sum(int(r.get("llm_calls") or 0) for r in usable) / len(usable)) if usable else 0,
        "tokens_total": sum(int(r.get("total_tokens") or 0) for r in usable),
        "tokens_per_record": (
            sum(int(r.get("total_tokens") or 0) for r in usable) / len(usable)) if usable else 0,
        "wall_seconds_total": sum(float(r.get("wall_seconds") or 0) for r in usable),
    }

    diagnostics = {
        "mean_conflicts": _mean("n_conflicts"),
        "mean_agreements": _mean("n_agreements"),
        "mean_severity": _mean("mean_severity"),
        "mean_norm_pressure": _mean("norm_pressure"),
        "mean_agreement_ratio": _mean("agreement_ratio"),
        "dominant_type_distribution": dict(
            Counter(s.get("dominant_type") for s in sig if s.get("dominant_type"))),
        "convergence_reason_distribution": dict(
            Counter(s.get("convergence_reason") for s in sig)),
        # Does the pipeline's own conflict typing separate the gold classes at
        # all? If the dominant type is identical for every gold label, the
        # structural head has nothing to work with and that is worth knowing
        # regardless of what accuracy came out.
        "dominant_type_by_gold": _cross(usable, "dominant_type"),
    }

    slices = metrics.by_slice(usable, labels, key="group")

    return {"config": config or {}, "labels": labels, "systems": systems,
            "comparison": comparison, "cost": cost, "diagnostics": diagnostics,
            "slices": slices,
            "errors": [{"record_id": r["record_id"], "error": r["error"]} for r in errors][:20]}


def _cross(rows: Sequence[dict], signal_field: str) -> dict:
    out: dict[str, dict[str, int]] = {}
    for row in rows:
        gold = row.get("gold")
        value = (row.get("signals") or {}).get(signal_field)
        if gold is None or value is None:
            continue
        out.setdefault(gold, {})
        out[gold][value] = out[gold].get(value, 0) + 1
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def to_markdown(report: dict) -> str:
    cfg = report.get("config", {}).get("config", report.get("config", {}))
    labels = report["labels"]
    parts: list[str] = []

    parts.append(f"# ThirdSpace evaluation — {cfg.get('dataset', '?')}\n")
    parts.append(
        f"n={cfg.get('n')} · classifier={cfg.get('classifier')} · "
        f"max_cycles={cfg.get('max_cycles')} · seed={cfg.get('seed')} · "
        f"stratified={cfg.get('stratify')}\n")

    rows = []
    for name, m in report["systems"].items():
        lo, hi = m["accuracy_ci95"]
        rows.append([
            name, m["n_scored"], f"{m['accuracy']:.3f}", f"[{lo:.2f}, {hi:.2f}]",
            f"{m['balanced_accuracy']:.3f}", f"{m['macro_f1']:.3f}",
            f"{m['cohen_kappa']:.3f}", m["n_abstained"],
        ])
    majority = next(iter(report["systems"].values()))["majority_baseline"]
    parts.append("## Headline\n")
    parts.append(_table(
        ["system", "scored", "accuracy", "95% CI", "balanced acc", "macro F1",
         "kappa", "abstained"], rows))
    parts.append(
        f"\nAlways predicting `{majority['label']}` scores "
        f"{majority['accuracy']:.3f} on this sample. Any system at or below "
        f"that has learned nothing from the data.\n")

    main = report["systems"]["thirdspace"]
    parts.append("## Per class (ThirdSpace)\n")
    parts.append(_table(
        ["label", "precision", "recall", "F1", "support"],
        [[l, f"{s['precision']:.2f}", f"{s['recall']:.2f}", f"{s['f1']:.2f}", s["support"]]
         for l, s in main["per_class"].items()]))

    parts.append("\n## Confusion (rows = gold, cols = predicted)\n")
    parts.append(_table(
        ["gold \\ pred"] + labels,
        [[g] + [main["confusion"][g][p] for p in labels] for g in labels]))

    comp = report.get("comparison", {})
    if comp:
        parts.append("\n## Comparison\n")
        mn = comp.get("thirdspace_vs_baseline")
        if mn:
            parts.append(
                f"- McNemar vs single-call baseline on {mn['n_compared']} paired records: "
                f"ThirdSpace-only-correct={mn['a_only']}, baseline-only-correct={mn['b_only']}, "
                f"p={mn['p_value']:.3f}.")
        agree = comp.get("structural_vs_readout_agreement")
        if agree is not None:
            parts.append(
                f"- Structural and read-out heads agree on {agree:.1%} of records. "
                f"High agreement means the free head can replace the paid one.")

    d = report["diagnostics"]
    parts.append("\n## Pipeline diagnostics\n")
    parts.append(_table(
        ["signal", "mean"],
        [["conflicts per record", f"{d['mean_conflicts']:.1f}"],
         ["agreements per record", f"{d['mean_agreements']:.1f}"],
         ["mean conflict severity", f"{d['mean_severity']:.2f}"],
         ["norm pressure", f"{d['mean_norm_pressure']:.2f}"],
         ["agreement ratio", f"{d['mean_agreement_ratio']:.2f}"]]))
    parts.append(f"\nDominant conflict type overall: `{d['dominant_type_distribution']}`")
    parts.append(f"\nDominant conflict type by gold label: `{d['dominant_type_by_gold']}`")
    parts.append(f"\nTermination reasons: `{d['convergence_reason_distribution']}`")

    c = report["cost"]
    parts.append("\n## Cost\n")
    parts.append(_table(
        ["metric", "value"],
        [["records", c["records"]],
         ["failed records", c["failed_records"]],
         ["LLM calls (total)", c["llm_calls_total"]],
         ["LLM calls per record", f"{c['llm_calls_per_record']:.1f}"],
         ["tokens (total)", f"{c['tokens_total']:,}"],
         ["tokens per record", f"{c['tokens_per_record']:,.0f}"],
         ["wall clock (s)", f"{c['wall_seconds_total']:.0f}"]]))

    if report.get("slices"):
        parts.append("\n## By slice\n")
        parts.append(_table(
            ["slice", "n", "accuracy", "macro F1"],
            [[k, v["n"], f"{v['accuracy']:.2f}", f"{v['macro_f1']:.2f}"]
             for k, v in report["slices"].items()]))

    if report.get("errors"):
        parts.append("\n## Failures\n")
        for e in report["errors"]:
            parts.append(f"- `{e['record_id']}`: {e['error']}")

    return "\n".join(parts) + "\n"


def write(report: dict, run_dir: Path) -> tuple[Path, Path]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / "report.json"
    md_path = run_dir / "report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")
    return md_path, json_path
