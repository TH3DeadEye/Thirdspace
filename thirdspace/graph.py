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

    # Refine the challenger's WORKING PERSPECTIVE. The structured identity is
    # never overwritten, so a dimensional frame cannot decay into free text.
    #
    # Gated on updated_perspectives alone — NOT on `changed`. `changed` measures
    # whether node CONTENT moved, so a cycle that revised only assumptions used to
    # leave the perspective stale even though mediation had explicitly asked for a
    # new lens. The field being non-empty is what licenses the update.
    #
    # NOTE: wholeness.updated_perspectives is agent-agnostic in the current
    # schema; it is applied to the challenger because that is the only side
    # carrying a WorkingPerspective. Revisit if the main agent needs one too.
    perspectives = mediation.wholeness.updated_perspectives
    if perspectives:
        before = len(mem.challenger_working_perspective.entries)
        mem.challenger_working_perspective = (
            mem.challenger_working_perspective.with_updates(mediation.cycle, perspectives)
        )
        added = len(mem.challenger_working_perspective.entries) - before
        logger.info(f"[{mem.scenario_id}] challenger working perspective: +{added} "
                    f"of {len(perspectives)} refinement(s) "
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

def _measure_stability(mem) -> dict | None:
    """Three cheap measures of whether the analysis has stopped moving.

        similarity  - is the SUBSTANCE the same? (embedding cosine or Jaccard)
        type mix    - is the SHAPE of the disagreement the same?
        conflicts   - the design's own rule: no new (facet, dimension) cells,
                      and no conflict that changed type

    Computed under BOTH cycle policies and appended to `stability_log`, so a
    fixed-N run still yields the convergence data even though it does not route
    on it. Returns None before cycle 2, when there is nothing to compare.

    `similarity` is None when the embedding call fails. It must never raise from
    here: this runs inside a LangGraph conditional edge, so an exception loses the
    entire paid run instead of terminating cleanly with a persisted session.
    """
    history = mem.mediation_history
    if len(history) < 2:
        return None
    prev, curr = history[-2], history[-1]
    threshold = (mem.embedding_convergence_threshold
                 if mem.convergence_metric == "embedding"
                 else mem.jaccard_convergence_threshold)
    try:
        sim = convergence_similarity(prev, curr, mem.convergence_metric)
    except Exception as exc:
        logger.warning(f"[{mem.scenario_id}] convergence metric unavailable "
                       f"({type(exc).__name__}: {exc}); treating as 'still moving'")
        sim = None
    stable_conflicts, why = conflicts_stable(prev, curr)
    return {
        "cycle": curr.cycle,
        "metric": mem.convergence_metric,
        "similarity": sim,
        "threshold": threshold,
        "types_stable": type_distribution_stable(prev, curr),
        "conflicts_stable": stable_conflicts,
        "conflicts_note": why,
        "llm_converged": mem.converged,
    }


def should_loop(state: GraphState) -> str:
    """Terminate, or run another cycle.

    Two policies, selected by `mem.cycle_policy`:

    "fixed" (default) — run exactly max_cycles. The convergence measures are
    still taken every cycle and recorded on `stability_log`, they just do not
    decide anything. This is the honest default: every eval path has always
    passed max_cycles=1, so the convergence branch below had never once
    executed. The policy was already fixed by accident; this states it.

    "convergence" — the mediator's `converged` flag is ADVISORY IN BOTH
    DIRECTIONS. It has been observed to fail each way: declaring convergence
    while content was still moving, and withholding it across cycles that
    produced identical conflict profiles (burning a full extra cycle).

        llm=True  + substance still moving   -> keep going (premature declaration)
        llm=False + all three stable         -> stop (structural_convergence)

    convergence_reason records which signal ended the loop, so a run's
    termination is auditable after the fact.
    """
    mem = state["memory"]
    done = "end" if mem.skip_generation else "generate"

    stability = _measure_stability(mem)
    if stability is not None:
        mem.stability_log.append(stability)
        sim = stability["similarity"]
        logger.info(
            f"[{mem.scenario_id}] cycle {mem.current_cycle} stability: "
            f"{stability['metric']}={'n/a' if sim is None else format(sim, '.2f')}"
            f"/{stability['threshold']}, types_stable={stability['types_stable']}, "
            f"conflicts_stable={stability['conflicts_stable']} ({stability['conflicts_note']})"
        )

    if mem.current_cycle >= mem.max_cycles:
        if not mem.converged:
            mem.convergence_reason = "max_cycles"
        logger.info(f"[{mem.scenario_id}] max cycles ({mem.max_cycles}) reached")
        return done

    if mem.cycle_policy == "fixed":
        logger.info(f"[{mem.scenario_id}] fixed policy; continuing to cycle "
                    f"{mem.current_cycle + 1} of {mem.max_cycles}")
        return "integrate"

    if stability is not None:
        sim, threshold = stability["similarity"], stability["threshold"]
        # sim is None only when the metric was unavailable; treat that as "still
        # moving" so an infrastructure failure can never manufacture convergence.
        moving = sim is None or sim < threshold

        if mem.converged and moving:
            logger.info(f"[{mem.scenario_id}] mediator declared converged but content is "
                        f"still moving — vetoing, continuing")
            return "integrate"

        if (not mem.converged and not moving
                and stability["types_stable"] and stability["conflicts_stable"]):
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
