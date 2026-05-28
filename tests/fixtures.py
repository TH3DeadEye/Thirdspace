"""
Reusable fixture builders for the test suite. All tests stub `call_llm*`
functions so nothing in here talks to Bedrock.
"""
from models import (
    TheOccurring, OccurringNode,
    NodeReasoningGrid, NodeDimensionCell,
    ImplyingNodes, WholenessImplyingNodes, JudgeScore,
    FACET_LIST, DIMENSION_LIST,
)


def make_occurring(scenario_id: str = "test-scn", raw: str = "raw scenario") -> TheOccurring:
    return TheOccurring(
        scenario_id=scenario_id,
        raw_scenario=raw,
        nodes=[
            OccurringNode(node_id=f, facet=f, content=f"content for {f}",
                          assumptions=[f"assumption for {f}"], last_updated_cycle=0)
            for f in FACET_LIST
        ],
    )


def make_grid(agent: str, cycle: int) -> NodeReasoningGrid:
    return NodeReasoningGrid(
        cycle=cycle, agent=agent,
        cells=[
            NodeDimensionCell(
                facet=f, dimension=d,
                content=f"{agent} {f}/{d}",
                reflection_flags=[],
                response_type=("challenge" if agent == "challenger" else None),
                cycle_last_updated=cycle,
            )
            for f in FACET_LIST for d in DIMENSION_LIST
        ],
    )


def make_wholeness(reformulated: str | None = "sharper question") -> WholenessImplyingNodes:
    return WholenessImplyingNodes(
        query_assumptions=["q-assumption"],
        reformulated_query=reformulated,
        new_objective=None,
        updated_challenger_perspective=None,
        objective_assumptions=[],
        perspective_assumptions=[],
    )


def make_mediation(cycle: int, converged: bool = False) -> ImplyingNodes:
    return ImplyingNodes(
        cycle=cycle,
        agreements=["both agree communication broke down"],
        conflicts=[],
        what_conflicts_imply=["pattern-level finding"],
        updated_assumptions=["individuals node: remove [X] -> replace with [Y]"],
        wholeness=make_wholeness(),
        converged=converged,
    )


def make_judge_score(cycle: int) -> JudgeScore:
    return JudgeScore(cycle=cycle, empathy=4, consensus=4, escalation_risk=2,
                      cultural_insight=4, reasoning="stub reasoning")
