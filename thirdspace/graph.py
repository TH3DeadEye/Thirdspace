import logging
from typing import TypedDict

from langgraph.graph import StateGraph, END

from metrics import convergence_similarity

from models import (
    SessionMemory, CompressedHistoryEntry, WorkingPerspective, CulturalIdentity,
)
from agents import (
    run_frame_detection,
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


def node_detect_frames(state: GraphState) -> GraphState:
    """Phase 0: infer both cultural frames from the raw scenario.

    Static, always-present entry node. It only does work when the caller passed
    no explicit frames (auto_detect_frames=True); otherwise it is a passthrough,
    so explicit-frame and neutral runs are unaffected. A detected side with an
    empty axis list renders to "" downstream (neutral mode for that side)."""
    mem = state["memory"]
    if not mem.auto_detect_frames:
        return {"memory": mem}
    logger.info(f"[{mem.scenario_id}] Phase 0: Frame detection")
    detected = run_frame_detection(
        raw_scenario=mem.occurring.raw_scenario,
        scenario_id=mem.scenario_id,
    )
    mem.main_cultural_frame = detected.main
    mem.challenger_cultural_frame = detected.challenger
    # Set the immutable structured identities directly from the detected axes.
    # The challenger's identity stays fixed across cycles; only its working
    # perspective (prose) evolves, so the structured frame never decays.
    mem.main_identity = CulturalIdentity.from_frame(detected.main)
    mem.challenger_identity = CulturalIdentity.from_frame(detected.challenger)
    mem.challenger_working_perspective = WorkingPerspective()
    mem.frame_detection = detected
    return {"memory": mem}


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
    # The challenger reasons from its fixed structured identity plus the mediator's
    # evolving working perspective. It is otherwise isolated — it never receives the
    # main agent's grid or frame, only The Occurring and its own frame.
    grid = run_challenger(
        occurring=mem.occurring,
        challenger_identity=mem.challenger_identity,
        challenger_working_perspective=mem.challenger_working_perspective,
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
        compressed_history=mem.compressed_history,
        cycle=cycle,
        mediator_frame=mem.mediator_frame,
    )
    mem.mediation_history.append(mediation)
    # Distill this cycle once and append (O(1) per cycle); the mediator prompt
    # reads this instead of re-flattening the whole history each call.
    mem.compressed_history.append(CompressedHistoryEntry.from_mediation(mediation))
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
    result = run_node_update(occurring=mem.occurring, mediation=mediation)
    mem.occurring = result.occurring

    changed = [d.facet for d in result.deltas if d.actually_changed]
    conflicts = result.conflicted_nodes
    logger.info(
        f"[{mem.scenario_id}] Node update cycle {mediation.cycle}: "
        f"changed={changed}, model_disagreements={conflicts}"
    )
    if conflicts:
        logger.warning(
            f"[{mem.scenario_id}] Model's changed flag disagreed with content "
            f"similarity on nodes: {conflicts}"
        )

    # Refine the challenger's WORKING PERSPECTIVE (prose) only when integration
    # actually changed something. The structured identity is never overwritten.
    if result.any_meaningful_change:
        prose = mediation.wholeness.updated_challenger_perspective
        if prose:
            mem.challenger_working_perspective = WorkingPerspective(
                cycle_last_updated=mediation.cycle, prose=prose
            )
            logger.info(
                f"[{mem.scenario_id}] Updated challenger working perspective "
                f"(identity preserved: {mem.challenger_identity.render()[:60]}...)"
            )
    return {"memory": mem}


def node_generate(state: GraphState) -> GraphState:
    """Pure generation node — no counter mutation. has_generated distinguishes a
    first run from a regeneration; the regen counter is owned by should_regenerate.
    On a regeneration, feed back the last OUTPUT judge's reasoning (cycle judges
    score mediations, not documents)."""
    mem = state["memory"]
    logger.info(f"[{mem.scenario_id}] Phase 5: Constrained generation")
    feedback = None
    if mem.has_generated and mem.output_judge_scores:
        feedback = mem.output_judge_scores[-1].reasoning
    try:
        mem.final_output = run_constrained_generation(
            occurring=mem.occurring,
            mediation_history=mem.mediation_history,
            mediator_frame=mem.mediator_frame,
            evaluation_feedback=feedback,
        )
    finally:
        # Mark attempted even on failure so a retry is treated as a regeneration.
        mem.has_generated = True
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
    # The router owns the regeneration decision AND its counter — a node should
    # not increment a counter that tracks a router's decision.
    mem = state["memory"]
    score = mem.output_judge_scores[-1]
    threshold = mem.output_score_threshold
    below = score.empathy < threshold or score.cultural_insight < threshold
    if below and mem.regen_attempts < mem.max_regen:
        mem.regen_attempts += 1
        logger.info(
            f"[{mem.scenario_id}] Output below threshold "
            f"(empathy={score.empathy}, insight={score.cultural_insight}); "
            f"regenerating (attempt {mem.regen_attempts}/{mem.max_regen})"
        )
        return "generate"
    if below:
        logger.warning(
            f"[{mem.scenario_id}] Output below threshold but max_regen="
            f"{mem.max_regen} exhausted; accepting subthreshold output."
        )
    logger.info(f"[{mem.scenario_id}] Output accepted after {mem.regen_attempts} regen(s)")
    return "end"


def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("detect_frames", node_detect_frames)
    graph.add_node("reformulate", node_reformulate)
    graph.add_node("main_agent", node_main_agent)
    graph.add_node("challenger", node_challenger)
    graph.add_node("mediation", node_mediation)
    graph.add_node("evaluate_cycle", node_evaluate_cycle)
    graph.add_node("integrate", node_integrate)
    graph.add_node("generate", node_generate)
    graph.add_node("judge", node_judge)

    graph.set_entry_point("detect_frames")
    graph.add_edge("detect_frames", "reformulate")
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
