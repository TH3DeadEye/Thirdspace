"""LangGraph wiring: the cyclic reasoning loop.

    reformulate -> main_agent -> challenger -> mediation
                        ^                          |
                        |                          v
                        +------ integrate <-- should_loop --> generate --> END
                                                         \\--> END (skip_generation)

Two nodes from the previous build are gone:
  - frame detection: frames are supplied by the caller. Evaluation runs read
    them from the dataset (NormAd -> country, MDD -> active/passive agent) via
    prompts.read_cultural_frames; there is no inference step.
  - the judges: the in-pipeline LLM judges saturated at perfect scores and are
    replaced by the external dual-classifier evaluation.

Generation is optional. The convergence-based (NormAd) and dominant-conflict-type
(MDD) classifiers read the mediation output, so an evaluation run that does not
need the document sets memory.skip_generation and terminates at mediation.

Cycles are 1-INDEXED (prompts.py guards assume it). current_cycle starts at 1
and is incremented by node_integrate — the node that means "begin another cycle".
"""
import logging
from typing import TypedDict

from langgraph.graph import StateGraph, END

from metrics import convergence_similarity, type_distribution_stable, conflicts_stable
from models import SessionMemory, CompressedHistoryEntry, WorkingPerspective
from agents import (
    run_reformulation,
    run_critical_thinking,
    run_challenger,
    run_mediation,
    run_node_update,
    run_constrained_generation,
)

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    memory: SessionMemory


# ---------------------------------------------------------------------------
# Phase nodes
# ---------------------------------------------------------------------------

def node_reformulate(state: GraphState) -> GraphState:
    """Raw scenario -> The Occurring (5 object nodes). Runs once, not per cycle."""
    mem = state["memory"]
    logger.info(f"[{mem.scenario_id}] Phase 1: reformulation")
    mem.occurring = run_reformulation(
        raw_scenario=mem.occurring.raw_scenario,
        scenario_id=mem.scenario_id,
    )
    return {"memory": mem}


def node_main_agent(state: GraphState) -> GraphState:
    """30 cells read through the main agent's frame."""
    mem = state["memory"]
    logger.info(f"[{mem.scenario_id}] Phase 2: main agent, cycle {mem.current_cycle}")
    mem.main_grids.append(run_critical_thinking(
        occurring=mem.occurring,
        cultural_frame=mem.main_cultural_frame,
        cycle=mem.current_cycle,
        # Cycle 2+: the previous mediation is refined DIRECTION for the main
        # agent (the challenger receives the same artifact as something to
        # contest — same input, opposite instruction).
        prior_mediation=mem.mediation_history[-1] if mem.mediation_history else None,
    ))
    return {"memory": mem}


def node_challenger(state: GraphState) -> GraphState:
    """30 cells read through the challenger's frame. Never sees the main grid."""
    mem = state["memory"]
    logger.info(f"[{mem.scenario_id}] Phase 3: challenger, cycle {mem.current_cycle}")
    mem.challenger_grids.append(run_challenger(
        occurring=mem.occurring,
        challenger_identity=mem.challenger_identity,
        challenger_working_perspective=mem.challenger_working_perspective,
        cycle=mem.current_cycle,
        prior_mediation=mem.mediation_history[-1] if mem.mediation_history else None,
    ))
    return {"memory": mem}


def node_mediation(state: GraphState) -> GraphState:
    """Compare the two grids: typed conflicts + the wholeness state."""
    mem = state["memory"]
    logger.info(f"[{mem.scenario_id}] Phase 4: mediation, cycle {mem.current_cycle}")
    mediation = run_mediation(
        occurring=mem.occurring,
        main_grid=mem.main_grids[-1],
        challenger_grid=mem.challenger_grids[-1],
        compressed_history=mem.compressed_history,
        cycle=mem.current_cycle,
    )
    mem.mediation_history.append(mediation)
    # Distil once per cycle (O(1)); the mediation prompt reads this instead of
    # re-flattening the whole history on every call.
    mem.compressed_history.append(CompressedHistoryEntry.from_mediation(mediation))
    mem.converged = mediation.converged
    summary = mediation.conflict_type_summary
    logger.info(
        f"[{mem.scenario_id}] cycle {mediation.cycle}: {len(mediation.conflicts)} conflicts, "
        f"dominant={summary.dominant_type if summary else 'n/a'}, "
        f"llm_converged={mediation.converged}"
    )
    return {"memory": mem}


def node_integrate(state: GraphState) -> GraphState:
    """Fold findings into The Occurring and open the next cycle.

    Only on the looping path — a run about to terminate never pays for an
    integration nobody will read.
    """
    mem = state["memory"]
    mediation = mem.mediation_history[-1]
    logger.info(f"[{mem.scenario_id}] Integrate: folding findings into The Occurring")

    before = {str(n.facet): n.content for n in mem.occurring.nodes}
    mem.occurring = run_node_update(occurring=mem.occurring, mediation=mediation)
    changed = [str(n.facet) for n in mem.occurring.nodes
               if before.get(str(n.facet)) != n.content]
    logger.info(f"[{mem.scenario_id}] node update cycle {mediation.cycle}: changed={changed}")

    # Refine the challenger's WORKING PERSPECTIVE (prose) only when something
    # actually moved. The structured identity is never overwritten, so a
    # dimensional frame cannot decay into free text across cycles.
    # NOTE: wholeness.updated_perspectives is agent-agnostic in the current
    # schema; it is applied to the challenger because that is the only side
    # carrying a WorkingPerspective. Revisit if the main agent needs one too.
    if changed and mediation.wholeness.updated_perspectives:
        prose = mediation.wholeness.updated_perspectives[0].statement
        mem.challenger_working_perspective = WorkingPerspective(
            cycle_last_updated=mediation.cycle, prose=prose
        )
        logger.info(f"[{mem.scenario_id}] challenger working perspective updated "
                    f"(identity preserved: {mem.challenger_identity.render()[:50]})")

    mem.current_cycle += 1
    return {"memory": mem}


def node_generate(state: GraphState) -> GraphState:
    """Write the final document from distilled findings only, never the grids."""
    mem = state["memory"]
    logger.info(f"[{mem.scenario_id}] Phase 5: constrained generation "
                f"({len(mem.mediation_history)} cycle(s), reason={mem.convergence_reason})")
    mem.final_output = run_constrained_generation(
        occurring=mem.occurring,
        mediation_history=mem.mediation_history,
        mediator_frame=mem.mediator_frame,
    )
    return {"memory": mem}


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def should_loop(state: GraphState) -> str:
    """Terminate, or run another cycle.

    The mediator's `converged` flag is ADVISORY IN BOTH DIRECTIONS. It has been
    observed to fail each way: declaring convergence while the content was still
    moving, and withholding it across cycles that produced identical conflict
    profiles (burning a full extra cycle re-deriving the same findings).

    So the loop verifies it, with three cheap measurements:
        similarity  - is the SUBSTANCE the same? (embedding cosine or Jaccard)
        type mix    - is the SHAPE of the disagreement the same?
        conflicts   - the design's own rule: no new (facet, dimension) cells,
                      and no conflict that changed type

        llm=True  + substance still moving   -> keep going (premature declaration)
        llm=False + all three stable         -> stop (structural_convergence)

    convergence_reason records which signal ended the loop, so a run's
    termination is auditable after the fact.
    """
    mem = state["memory"]
    history = mem.mediation_history
    # Where "stop" leads: the document, or straight out.
    done = "end" if mem.skip_generation else "generate"

    if mem.current_cycle >= mem.max_cycles:
        if not mem.converged:
            mem.convergence_reason = "max_cycles"
        logger.info(f"[{mem.scenario_id}] max cycles ({mem.max_cycles}) reached")
        return done

    if len(history) >= 2:
        prev, curr = history[-2], history[-1]
        sim = convergence_similarity(prev, curr, mem.convergence_metric)
        threshold = (mem.embedding_convergence_threshold
                     if mem.convergence_metric == "embedding"
                     else mem.jaccard_convergence_threshold)
        stable_types = type_distribution_stable(prev, curr)
        stable_conflicts, why = conflicts_stable(prev, curr)

        logger.info(f"[{mem.scenario_id}] cycle {mem.current_cycle} stability: "
                    f"{mem.convergence_metric}={sim:.2f}/{threshold}, "
                    f"types_stable={stable_types}, conflicts_stable={stable_conflicts} ({why})")

        if mem.converged and sim < threshold:
            logger.info(f"[{mem.scenario_id}] mediator declared converged but content is "
                        f"still moving ({sim:.2f} < {threshold}) — vetoing, continuing")
            return "integrate"

        if not mem.converged and sim >= threshold and stable_types and stable_conflicts:
            mem.converged = True
            mem.convergence_reason = "structural_convergence"
            logger.info(f"[{mem.scenario_id}] structural_convergence: all three measures "
                        f"stable — terminating despite llm_converged=False")
            return done

    if mem.converged:
        mem.convergence_reason = "llm_declared"
        logger.info(f"[{mem.scenario_id}] converged at cycle {mem.current_cycle}")
        return done

    logger.info(f"[{mem.scenario_id}] not converged; continuing to cycle {mem.current_cycle + 1}")
    return "integrate"


# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("reformulate", node_reformulate)
    graph.add_node("main_agent", node_main_agent)
    graph.add_node("challenger", node_challenger)
    graph.add_node("mediation", node_mediation)
    graph.add_node("integrate", node_integrate)
    graph.add_node("generate", node_generate)

    graph.set_entry_point("reformulate")
    graph.add_edge("reformulate", "main_agent")
    graph.add_edge("main_agent", "challenger")
    graph.add_edge("challenger", "mediation")

    # mediation -> should_loop -> {integrate (loop) | generate | end}
    graph.add_conditional_edges(
        "mediation",
        should_loop,
        {"integrate": "integrate", "generate": "generate", "end": END},
    )
    graph.add_edge("integrate", "main_agent")
    graph.add_edge("generate", END)

    return graph.compile()
