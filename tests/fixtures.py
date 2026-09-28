"""Reusable fixture builders for the test suite.

All tests stub `call_llm*` so nothing here talks to Bedrock. Everything is built
against the CURRENT ontology — 5 facets x 6 dimensions, the 6-type taxonomy,
WholenessEntry with refs, no judge — so a fixture that stops constructing is a
real signal about models.py rather than drift in this file.
"""
from models import (
    TheOccurring, OccurringNode,
    NodeReasoningGrid, NodeDimensionCell,
    ImplyingNodes, WholenessImplyingNodes, WholenessEntry, CellConflict,
    SessionMemory,
    FACET_LIST, DIMENSION_LIST, NUM_CELLS,
)

# A conflict implication must open with one of CellConflict's accepted starters
# and clear 60 characters, so build valid ones from here rather than by hand.
VALID_IMPLICATION = (
    "This reveals that the two readings price the same gesture on "
    "incompatible scales of obligation."
)
VALID_RATIONALE = (
    "Step six fired: both sides understand each other and still disagree "
    "about what ought to happen, so no earlier step applied."
)


def make_occurring(scenario_id: str = "test-scn", raw: str = "raw scenario") -> TheOccurring:
    return TheOccurring(
        scenario_id=scenario_id,
        raw_scenario=raw,
        nodes=[
            OccurringNode(node_id=str(f), facet=f, content=f"content for {f}",
                          assumptions=[f"assumption for {f}"], last_updated_cycle=0)
            for f in FACET_LIST
        ],
    )


def make_grid(agent: str, cycle: int = 1, content: str | None = None) -> NodeReasoningGrid:
    """A full NUM_CELLS grid. Cell content encodes agent/facet/dimension so tests
    can trace a specific cell's text through mediation and back."""
    return NodeReasoningGrid(
        cycle=cycle, agent=agent,
        cells=[
            NodeDimensionCell(
                facet=f, dimension=d,
                content=content or f"{agent.upper()}-SAYS-{f}/{d}",
                reflection_flags=[],
                cycle_last_updated=cycle,
            )
            for f in FACET_LIST for d in DIMENSION_LIST
        ],
    )


def make_conflict(facet: str = "agents", dimension: str = "value",
                  conflict_type: str = "normative",
                  resolvability: str = "facilitated",
                  severity: int = 4) -> CellConflict:
    return CellConflict(
        facet=facet, dimension=dimension,
        main_summary=f"main reading of {facet}/{dimension}",
        challenger_summary=f"challenger reading of {facet}/{dimension}",
        implication=VALID_IMPLICATION,
        conflict_type=conflict_type,
        conflict_type_rationale=VALID_RATIONALE,
        severity=severity, resolvability=resolvability,
    )


def make_wholeness(**overrides) -> WholenessImplyingNodes:
    """Empty by default — that is a legitimate mediator outcome, and the shape
    that used to crash cycle 2. Pass field=[WholenessEntry(...)] to populate."""
    return WholenessImplyingNodes(**overrides)


def make_mediation(cycle: int = 1, converged: bool = False,
                   conflicts: list[CellConflict] | None = None,
                   wholeness: WholenessImplyingNodes | None = None) -> ImplyingNodes:
    conflicts = [make_conflict()] if conflicts is None else conflicts
    refs = [c.cell_id for c in conflicts]
    if wholeness is None:
        wholeness = make_wholeness(
            what_conflicts_imply=[
                WholenessEntry(statement="pattern-level finding", refs=refs)],
            updated_assumptions=[
                WholenessEntry(statement="agents: remove [X] -> replace with [Y]", refs=refs)],
        )
    return ImplyingNodes(
        cycle=cycle,
        agreements=[WholenessEntry(statement="both agree the exchange happened", refs=refs)],
        conflicts=conflicts,
        wholeness=wholeness,
        converged=converged,
    )


def make_memory(scenario_id: str = "test-scn", **overrides) -> SessionMemory:
    fields = {
        "scenario_id": scenario_id,
        "main_cultural_frame": "germany",
        "challenger_cultural_frame": "jordan",
        "occurring": make_occurring(scenario_id),
        "max_cycles": 2,
    }
    fields.update(overrides)
    return SessionMemory(**fields)


def make_complete_session(cycles: int = 1, **overrides) -> SessionMemory:
    """A memory that should pass verify_session cleanly — the baseline a test
    perturbs one field at a time to prove a given check actually fires."""
    mem = make_memory(**overrides)
    for c in range(1, cycles + 1):
        mem.main_grids.append(make_grid("main", c))
        mem.challenger_grids.append(make_grid("challenger", c))
        mem.mediation_history.append(make_mediation(c))
    mem.convergence_reason = "max_cycles"
    mem.llm_calls_made = 1 + cycles * (2 * NUM_CELLS + 1)
    mem.final_output = "A plain document written for the two people involved."
    return mem
