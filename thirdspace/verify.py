"""Post-run invariant checks over a completed SessionMemory.

Makes no LLM calls and costs nothing, so it runs on every scenario. The point is
to turn "the run finished" into "the run finished AND is internally consistent":
before this existed, a run that quietly dropped half its conflicts and a clean run
produced output that looked identical.

Every check answers a question you would otherwise have to ask by hand while
reading a 300KB session JSON. `verify_session` returns the list of violations —
empty means clean — so callers can log it, assert on it in tests, or gate a
dataset run on it.
"""
from __future__ import annotations

from models import (
    LEAK_PHRASES, FACET_LIST, DIMENSION_LIST, NUM_CELLS, SessionMemory,
)


def _check_grids(mem: SessionMemory) -> list[str]:
    """Every grid holds exactly one cell per (facet, dimension) pair."""
    problems: list[str] = []
    expected = {(str(f), str(d)) for f in FACET_LIST for d in DIMENSION_LIST}
    for label, grids in (("main", mem.main_grids), ("challenger", mem.challenger_grids)):
        for grid in grids:
            got = [(str(c.facet), str(c.dimension)) for c in grid.cells]
            if len(got) != NUM_CELLS:
                problems.append(
                    f"{label} grid cycle {grid.cycle}: {len(got)} cells, expected {NUM_CELLS}")
            missing = expected - set(got)
            if missing:
                problems.append(
                    f"{label} grid cycle {grid.cycle}: missing {sorted(missing)[:4]}")
            if len(set(got)) != len(got):
                dupes = {p for p in got if got.count(p) > 1}
                problems.append(
                    f"{label} grid cycle {grid.cycle}: duplicate cells {sorted(dupes)[:4]}")
            empty = [f"{f}/{d}" for (f, d), c in zip(got, grid.cells) if not c.content.strip()]
            if empty:
                problems.append(
                    f"{label} grid cycle {grid.cycle}: {len(empty)} empty cell(s) {empty[:4]}")
    return problems


def _check_cycles_align(mem: SessionMemory) -> list[str]:
    """One main grid, one challenger grid and one mediation per completed cycle."""
    counts = {
        "main_grids": len(mem.main_grids),
        "challenger_grids": len(mem.challenger_grids),
        "mediation_history": len(mem.mediation_history),
    }
    if len(set(counts.values())) > 1:
        return [f"cycle artefacts out of step: {counts}"]
    return []


def _check_mediation(mem: SessionMemory) -> list[str]:
    """Conflicts fit the grid, and every wholeness ref resolves to a real id."""
    problems: list[str] = []
    for m in mem.mediation_history:
        total = len(m.conflicts) + len(m.agreements)
        if total > NUM_CELLS:
            problems.append(
                f"cycle {m.cycle}: {len(m.conflicts)} conflicts + {len(m.agreements)} "
                f"agreements = {total}, more than the {NUM_CELLS} cells compared")
        valid = m.conflict_ids | {str(r) for r in m.carried_forward}
        w = m.wholeness
        for field in ("what_conflicts_imply", "updated_assumptions",
                      "comparison_assumptions", "reformulated_queries",
                      "new_objectives", "updated_perspectives"):
            for entry in getattr(w, field):
                unknown = [r for r in entry.refs if r not in valid]
                if unknown:
                    problems.append(
                        f"cycle {m.cycle} {field}: unresolved ref(s) {unknown} "
                        f"in {entry.statement[:50]!r}")
    return problems


def _check_drops(mem: SessionMemory) -> list[str]:
    """Anything the pipeline discarded. Non-zero is not fatal, but it is never
    silent again: a dropped conflict deflates the counts the classifiers read."""
    problems: list[str] = []
    if mem.conflicts_dropped:
        problems.append(f"{mem.conflicts_dropped} conflict(s) dropped as invalid — "
                        f"the reported conflict count is an undercount")
    if mem.refs_dropped:
        problems.append(f"{mem.refs_dropped} wholeness ref(s) dropped as invented")
    return problems


def _check_output(mem: SessionMemory) -> list[str]:
    """The document exists when it should, and carries no internal vocabulary."""
    problems: list[str] = []
    if mem.skip_generation:
        return problems
    doc = mem.final_output or ""
    if not doc.strip():
        problems.append("skip_generation is False but final_output is empty")
        return problems
    leaked = [p for p in LEAK_PHRASES if p in doc.lower()]
    if leaked:
        problems.append(f"final_output contains internal rubric vocabulary: {leaked}")
    return problems


def _check_accounting(mem: SessionMemory) -> list[str]:
    """Call count sits in a plausible band for the cycles actually completed.

    A floor rather than an exact figure: node updates only fire when mediation
    produced signal, and embeddings only from cycle 2, so the exact total is not
    a fixed function of the cycle count.
    """
    problems: list[str] = []
    cycles = len(mem.mediation_history)
    if cycles == 0:
        return ["no mediation cycles completed"]
    floor = 1 + cycles * (2 * NUM_CELLS + 1)          # reformulation + grids + mediation
    if mem.llm_calls_made < floor:
        problems.append(
            f"llm_calls_made={mem.llm_calls_made} is below the {floor} required for "
            f"{cycles} cycle(s) — calls went unrecorded, or a phase was skipped")
    if mem.llm_calls_made > mem.llm_call_budget:
        problems.append(
            f"llm_calls_made={mem.llm_calls_made} exceeded budget {mem.llm_call_budget}")
    if mem.convergence_reason is None:
        problems.append("convergence_reason is unset — how the loop ended was not recorded")
    return problems


def verify_session(memory: SessionMemory) -> list[str]:
    """Run every invariant. Empty list means the session is internally consistent."""
    problems: list[str] = []
    for check in (_check_cycles_align, _check_grids, _check_mediation,
                  _check_drops, _check_output, _check_accounting):
        problems.extend(check(memory))
    return problems


def format_report(problems: list[str]) -> str:
    """One-line summary plus the detail, for logging."""
    if not problems:
        return "session verified: all invariants hold"
    lines = "\n".join(f"  - {p}" for p in problems)
    return f"session verification found {len(problems)} problem(s):\n{lines}"
