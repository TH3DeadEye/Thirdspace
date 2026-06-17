import logging
from typing import TypedDict

from langgraph.graph import StateGraph, END

from metrics import convergence_similarity

from models import SessionMemory
from agents import (
    run_reformulation,
    run_critical_thinking,
    run_challenger,
    run_mediation,
    run_node_update,
    run_constrained_generation,
    run_judge,
    run_cycle_judge,
)

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    memory: SessionMemory


def node_reformulate(state: GraphState) -> GraphState:
    mem = state["memory"]
    logger.info(f"[{mem.scenario_id}] Phase 1: Reformulation")
    occurring = run_reformulation(
        raw_scenario=mem.occurring.raw_scenario,
        scenario_id=mem.scenario_id,
    )
    mem.occurring = occurring
    return {"memory": mem}


def node_main_agent(state: GraphState) -> GraphState:
    mem = state["memory"]
    cycle = mem.current_cycle
    logger.info(f"[{mem.scenario_id}] Phase 2: Main agent, cycle {cycle}")
    grid = run_critical_thinking(
        occurring=mem.occurring,
        cultural_frame=mem.main_cultural_frame,
        cycle=cycle,
        # Cycle 2+: feed the previous mediation's meta-level guidance (reformulated
        # query, new objective, named assumption corrections) into the cell prompts
        # so the main agent advances instead of re-deriving the same grid.
        prior_mediation=mem.mediation_history[-1] if mem.mediation_history else None,
    )
    mem.main_grids.append(grid)
    return {"memory": mem}


def node_challenger(state: GraphState) -> GraphState:
    mem = state["memory"]
    cycle = mem.current_cycle
    logger.info(f"[{mem.scenario_id}] Phase 3: Challenger, cycle {cycle}")
    # Use the working perspective if the loop has refined one; otherwise fall
    # back to the original cultural frame.
    challenger_frame = mem.active_challenger_perspective or mem.challenger_cultural_frame
    # The challenger is fully isolated — it does not receive the main agent's grid.
    grid = run_challenger(
        occurring=mem.occurring,
        challenger_frame=challenger_frame,
        main_frame=mem.main_cultural_frame,
        cycle=cycle,
        challenger_style=mem.challenger_style,
    )
    mem.challenger_grids.append(grid)
    return {"memory": mem}


def node_mediation(state: GraphState) -> GraphState:
    mem = state["memory"]
    cycle = mem.current_cycle
    logger.info(f"[{mem.scenario_id}] Phase 4: Mediation, cycle {cycle}")
    mediation = run_mediation(
        occurring=mem.occurring,
        main_grid=mem.main_grids[-1],
        challenger_grid=mem.challenger_grids[-1],
        all_mediation_history=mem.mediation_history,
        cycle=cycle,
    )
    mem.mediation_history.append(mediation)
    mem.converged = mediation.converged
    mem.current_cycle += 1
    return {"memory": mem}


def node_evaluate_cycle(state: GraphState) -> GraphState:
    """Score the just-completed mediation cycle. Only runs when we will loop —
    scoring a cycle we are about to discard is wasted budget."""
    mem = state["memory"]
    mediation = mem.mediation_history[-1]
    logger.info(f"[{mem.scenario_id}] Evaluate: scoring mediation cycle {mediation.cycle}")
    score = run_cycle_judge(mediation=mediation, cycle=mediation.cycle)
    mem.cycle_judge_scores.append(score)
    return {"memory": mem}


def node_integrate(state: GraphState) -> GraphState:
    mem = state["memory"]
    mediation = mem.mediation_history[-1]
    logger.info(
        f"[{mem.scenario_id}] Integrate: folding implying nodes into The Occurring"
    )
    mem.occurring = run_node_update(
        occurring=mem.occurring,
        mediation=mediation,
    )
    updated_perspective = mediation.wholeness.updated_challenger_perspective
    if updated_perspective:
        mem.active_challenger_perspective = updated_perspective
    return {"memory": mem}


def node_generate(state: GraphState) -> GraphState:
    mem = state["memory"]
    logger.info(f"[{mem.scenario_id}] Phase 5: Constrained generation")
    # A non-None final_output means we already generated once, so this run is a
    # regeneration: count the attempt and feed back the most recent OUTPUT
    # judge's reasoning (not a cycle judge — those score mediations, not docs).
    feedback = None
    if mem.final_output is not None:
        mem.regen_attempts += 1
        if mem.output_judge_scores:
            feedback = mem.output_judge_scores[-1].reasoning
    final_output = run_constrained_generation(
        occurring=mem.occurring,
        mediation_history=mem.mediation_history,
        evaluation_feedback=feedback,
    )
    mem.final_output = final_output
    return {"memory": mem}


def node_judge(state: GraphState) -> GraphState:
    mem = state["memory"]
    logger.info(f"[{mem.scenario_id}] Judge: evaluating output quality")
    # current_cycle was incremented by node_mediation; score against the cycle
    # that actually produced the output (the last mediation's cycle).
    score = run_judge(
        final_output=mem.final_output,
        mediation_history=mem.mediation_history,
        cycle=mem.mediation_history[-1].cycle,
    )
    mem.output_judge_scores.append(score)
    return {"memory": mem}


def _consensus_plateau(mem: SessionMemory) -> bool:
    """Multi-signal convergence: detect when consensus scores have plateaued.

    Returns True if the last two cycle judges both meet the consensus threshold
    and differ by no more than 1. Returns False when fewer than two cycle
    judges exist.
    """
    if len(mem.cycle_judge_scores) < 2:
        return False
    last, prev = mem.cycle_judge_scores[-1], mem.cycle_judge_scores[-2]
    threshold = mem.consensus_plateau_threshold
    return (
        last.consensus >= threshold
        and prev.consensus >= threshold
        and abs(last.consensus - prev.consensus) <= 1
    )


def should_loop(state: GraphState) -> str:
    """Decide whether to terminate after a mediation cycle.

    The in-loop judge (`evaluate_cycle`) only runs on the path that continues
    looping; terminating cycles skip it to save budget. The judge then feeds
    the plateau check via `should_continue_after_judge`.
    """

    mem = state["memory"]
    history = mem.mediation_history

    if mem.current_cycle >= mem.max_cycles:
        logger.info(f"{mem.scenario_id} Max cycles reached")
        return "generate"

    if mem.converged:
        if len(history) >= 2:
            sim = convergence_similarity(
                history[-2], history[-1], mem.convergence_metric
            )
            threshold = (
                mem.embedding_convergence_threshold
                if mem.convergence_metric == "embedding"
                else mem.jaccard_convergence_threshold
            )
            if sim < threshold:
                logger.info(
                    f"[{mem.scenario_id}] LLM converged but "
                    f"{mem.convergence_metric} similarity={sim:.2f} < {threshold}; "
                    f"content still moving, continuing"
                )
                return "evaluate_cycle"
        logger.info(f"[{mem.scenario_id}] Converged at cycle {mem.current_cycle}")
        return "generate"

    logger.info(f"[{mem.scenario_id}] Not converged, evaluating cycle {mem.current_cycle}")
    return "evaluate_cycle"

def should_continue_after_judge(state: GraphState) -> str:
    """After scoring the cycle, check the secondary plateau signal.

    If two consecutive cycle judges show a high, stable consensus, terminate
    early. Otherwise integrate findings and run another cycle.
    """
    mem = state["memory"]
    if _consensus_plateau(mem):
        logger.info(
            f"[{mem.scenario_id}] Consensus plateau at cycle {mem.current_cycle}; "
            f"terminating loop"
        )
        return "generate"
    return "integrate"


def should_regenerate(state: GraphState) -> str:
    # Pure router: decide only, do not mutate state. The attempt counter is
    # bumped inside node_generate when it actually regenerates.
    mem = state["memory"]
    score = mem.output_judge_scores[-1]
    threshold = mem.output_score_threshold
    below = score.empathy < threshold or score.cultural_insight < threshold
    if below and mem.regen_attempts < mem.max_regen:
        logger.info(
            f"[{mem.scenario_id}] Output below threshold "
            f"(empathy={score.empathy}, insight={score.cultural_insight}); "
            f"regenerating"
        )
        return "generate"
    logger.info(f"[{mem.scenario_id}] Output accepted")
    return "end"


def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("reformulate", node_reformulate)
    graph.add_node("main_agent", node_main_agent)
    graph.add_node("challenger", node_challenger)
    graph.add_node("mediation", node_mediation)
    graph.add_node("evaluate_cycle", node_evaluate_cycle)
    graph.add_node("integrate", node_integrate)
    graph.add_node("generate", node_generate)
    graph.add_node("judge", node_judge)

    graph.set_entry_point("reformulate")
    graph.add_edge("reformulate", "main_agent")
    graph.add_edge("main_agent", "challenger")
    graph.add_edge("challenger", "mediation")

    # mediation -> should_loop -> {evaluate_cycle | generate}
    # evaluate_cycle -> should_continue_after_judge -> {integrate | generate}
    # integrate -> main_agent (loop)
    graph.add_conditional_edges(
        "mediation",
        should_loop,
        {"evaluate_cycle": "evaluate_cycle", "generate": "generate"},
    )
    graph.add_conditional_edges(
        "evaluate_cycle",
        should_continue_after_judge,
        {"integrate": "integrate", "generate": "generate"},
    )
    graph.add_edge("integrate", "main_agent")

    graph.add_edge("generate", "judge")
    graph.add_conditional_edges(
        "judge",
        should_regenerate,
        {"generate": "generate", "end": END},
    )

    return graph.compile()
