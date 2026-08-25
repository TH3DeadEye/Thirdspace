"""Turn a finished SessionMemory into a flat bundle of numbers.

Everything downstream — the deterministic classifier, the calibration grid
search, the cost report — reads this bundle rather than the SessionMemory, so
signals are cached in predictions.jsonl and thresholds can be re-fitted later
without re-running a single Bedrock call.

Type fractions are recomputed here from the conflict list rather than read off
ConflictTypeSummary. Not because the summary is wrong, but because a run at
max_cycles > 1 needs the pooled view and the per-cycle summaries are the only
thing the pipeline stores; recomputing keeps one definition in one place.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

# Cells the mediator can compare: 5 facets x 6 dimensions.
NUM_CELLS = 30

# The dimensions where a disagreement is about what OUGHT to be the case
# rather than about what happened. Used to build `norm_pressure`.
VALUE_DIMENSIONS = ("value", "assumption")
VALUE_FACETS = ("representations", "consequences")


def _f(counter: Counter, key: str, total: int) -> float:
    return (counter.get(key, 0) / total) if total else 0.0


def extract_signals(memory: Any) -> dict:
    """Flat, JSON-serializable signal bundle for one completed run."""
    history = list(getattr(memory, "mediation_history", []) or [])
    conflicts = [c for m in history for c in m.conflicts]
    agreements = [a for m in history for a in m.agreements]
    total = len(conflicts)

    type_counts = Counter(str(c.conflict_type) for c in conflicts)
    resolv_counts = Counter(str(c.resolvability) for c in conflicts)
    severities = [int(c.severity) for c in conflicts]

    # Where the disagreement sits. A conflict on representations/value is a
    # disagreement about what the act MEANS and whether it was right; one on
    # actions/observation is a disagreement about what happened.
    value_cells = sum(
        1 for c in conflicts
        if str(c.dimension) in VALUE_DIMENSIONS or str(c.facet) in VALUE_FACETS
    )
    value_cell_fraction = (value_cells / total) if total else 0.0

    f_normative = _f(type_counts, "normative", total)
    f_identity = _f(type_counts, "identity_related", total)
    f_power = _f(type_counts, "power_related", total)

    # One number for "the two readings disagree about what ought to happen,
    # and they disagree about it where it matters". Half from the mediator's
    # own type call, half from where in the grid the conflicts landed, so a
    # single mis-typed conflict cannot swing it.
    norm_pressure = min(1.0, 0.5 * (f_normative + f_identity) + 0.5 * value_cell_fraction)

    n_agree, n_conf = len(agreements), total
    cycles = len(history)
    cells_considered = max(NUM_CELLS * max(cycles, 1), 1)

    last = history[-1] if history else None
    return {
        "scenario_id": getattr(memory, "scenario_id", ""),
        "cycles": cycles,
        "converged": bool(getattr(memory, "converged", False)),
        "convergence_reason": getattr(memory, "convergence_reason", None),

        "n_conflicts": n_conf,
        "n_agreements": n_agree,
        "conflict_density": n_conf / cells_considered,
        "agreement_ratio": (n_agree / (n_agree + n_conf)) if (n_agree + n_conf) else 1.0,

        "mean_severity": (sum(severities) / len(severities)) if severities else 0.0,
        "max_severity": max(severities) if severities else 0,
        "severe_count": sum(1 for s in severities if s >= 4),
        "severe_fraction": (sum(1 for s in severities if s >= 4) / total) if total else 0.0,

        "dominant_type": (type_counts.most_common(1)[0][0] if type_counts else None),
        "type_counts": dict(type_counts),
        "type_fractions": {t: c / total for t, c in type_counts.items()} if total else {},
        "normative_fraction": f_normative,
        "identity_fraction": f_identity,
        "power_fraction": f_power,
        "interpretive_fraction": _f(type_counts, "interpretive", total),
        "factual_fraction": _f(type_counts, "factual", total),
        "procedural_fraction": _f(type_counts, "procedural", total),
        "value_cell_fraction": value_cell_fraction,
        "norm_pressure": norm_pressure,

        "resolvability_counts": dict(resolv_counts),
        "unresolvable_fraction": _f(resolv_counts, "unresolvable", total),
        "conversational_fraction": _f(resolv_counts, "conversational", total),

        "has_final_output": bool(getattr(memory, "final_output", None)),
        "llm_calls": int(getattr(memory, "llm_calls_made", 0)),
        "input_tokens": int(getattr(memory, "total_input_tokens", 0)),
        "output_tokens": int(getattr(memory, "total_output_tokens", 0)),
        "total_tokens": int(getattr(memory, "total_tokens", 0)),
        "latency_ms": int(getattr(memory, "total_latency_ms", 0)),

        "last_cycle": getattr(last, "cycle", None),
    }


def mediation_digest(memory: Any, max_conflicts: int = 14) -> dict:
    """The artifact the read-out classifier is allowed to see.

    Deliberately narrow: The Occurring plus the FINAL cycle's mediation. Not
    the grids (60 cells of prose would swamp the call and reintroduce the
    token bloat the pipeline exists to avoid), and not the raw scenario beyond
    what reformulation already put into the nodes.
    """
    history = list(getattr(memory, "mediation_history", []) or [])
    if not history:
        return {"occurring": None, "agreements": [], "conflicts": [], "wholeness": {}}

    final = history[-1]
    conflicts = sorted(final.conflicts, key=lambda c: -int(c.severity))[:max_conflicts]
    occurring = getattr(memory, "occurring", None)

    return {
        "occurring": (
            [{"facet": str(n.facet), "content": n.content, "assumptions": n.assumptions}
             for n in occurring.nodes] if occurring else []),
        "agreements": [e.statement for e in final.agreements],
        "conflicts": [{
            "id": c.cell_id or f"{c.facet}/{c.dimension}",
            "facet": str(c.facet),
            "dimension": str(c.dimension),
            "reading_main": c.main_summary,
            "reading_challenger": c.challenger_summary,
            "implication": c.implication,
            "type": str(c.conflict_type),
            "severity": int(c.severity),
            "resolvability": str(c.resolvability),
        } for c in conflicts],
        "wholeness": final.wholeness.model_dump(mode="json"),
        "converged": bool(final.converged),
        "final_document": getattr(memory, "final_output", None),
    }
