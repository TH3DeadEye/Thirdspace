import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from models import (
    TheOccurring, OccurringNode,
    NodeReasoningGrid, NodeDimensionCell, ReflectionFlag,
    ImplyingNodes, CellConflict, WholenessImplyingNodes,
    JudgeScore,
    FACET_LIST, DIMENSION_LIST, NUM_FACETS, NUM_CELLS,
)
from llm import (
    call_llm, call_llm_json, call_llm_json_with_refinement,
    compress_historical_context, linearize_grid_to_matrix_map,
    CHALLENGER_MODEL_ID,
    REFORMULATION_SYSTEM, reformulation_user,
    critical_thinking_cell_system, critical_thinking_cell_user,
    challenger_cell_system, challenger_cell_user,
    NODE_UPDATE_SYSTEM, node_update_user,
    MEDIATION_SYSTEM, mediation_user,
    CONSTRAINED_GENERATION_SYSTEM, constrained_generation_user,
    judge_system, judge_user, judge_cycle_user,
)
from budget import active_memory, get_active_memory

# Cell-level concurrency. The Bedrock boto3 client is thread-safe; cells within
# a phase are independent so they can run in parallel. Tune via env var.
_CELL_WORKERS = int(os.environ.get("THIRDSPACE_CELL_WORKERS", "8"))


def _cell_pairs() -> list[tuple[str, str]]:
    return [(f, d) for f in FACET_LIST for d in DIMENSION_LIST]

logger = logging.getLogger(__name__)


def _cell_trace_extra(parent, agent: str, facet: str, dimension: str, cycle: int) -> dict:
    """LangSmith run config for one cell call.

    ThreadPoolExecutor workers start with a fresh contextvars context, so the
    @traceable cell run cannot discover its parent automatically. We capture
    the phase run in the main thread and pass it through explicitly as `parent`
    (no-op when tracing is off and parent is None). The name + metadata make
    each of the NUM_CELLS calls individually identifiable and filterable in the
    LangSmith UI.
    """
    return {
        "name": f"{agent}_cell[{facet}/{dimension}]",
        "parent": parent,
        "metadata": {
            "facet": facet,
            "dimension": dimension,
            "cycle": cycle,
            "agent": agent,
        },
        "tags": ["cell", agent],
    }


def _parse_flags(raw_flags: list) -> list[ReflectionFlag]:
    """Normalize reflection_flags from the model.

    Models sometimes return a list of dicts {"measure":..., "issue":...},
    sometimes a list of plain strings ("clarity"), sometimes a single string.
    This handles all three so neither agent crashes on malformed output.
    """
    flags: list[ReflectionFlag] = []
    if isinstance(raw_flags, str):
        raw_flags = [raw_flags]
    for f in raw_flags:
        if isinstance(f, dict):
            flags.append(ReflectionFlag(
                measure=f.get("measure", "unknown"),
                issue=f.get("issue", "unspecified"),
            ))
        elif isinstance(f, str) and f.strip():
            flags.append(ReflectionFlag(measure=f.strip(), issue="flagged by model"))
    return flags


# ---------------------------------------------------------------------------
# Phase 1 — Reformulation
# ---------------------------------------------------------------------------

def _validate_facet_set(raw_nodes: list[dict], where: str) -> None:
    """Reject reformulation/update results that don't cover all 6 facets exactly once.

    Surfaces three failure modes with explicit messages:
      - unknown facet names (typos or hallucinated facets)
      - duplicate facets
      - missing facets
    """
    facets = [n.get("facet") for n in raw_nodes]
    unknown = [f for f in facets if f not in FACET_LIST]
    if unknown:
        raise ValueError(f"{where}: unknown facet(s) {unknown}; expected subset of {FACET_LIST}")
    seen: set[str] = set()
    duplicates: list[str] = []
    for f in facets:
        if f in seen:
            duplicates.append(f)
        seen.add(f)
    if duplicates:
        raise ValueError(f"{where}: duplicate facet(s) {duplicates}")
    missing = [f for f in FACET_LIST if f not in seen]
    if missing:
        raise ValueError(f"{where}: missing facet(s) {missing}")


def _sort_by_facet(nodes: list[OccurringNode]) -> list[OccurringNode]:
    """Sort nodes into canonical FACET_LIST order for stable downstream indexing."""
    return sorted(nodes, key=lambda n: FACET_LIST.index(n.facet))


@traceable(run_type="chain", name="reformulation")
def run_reformulation(raw_scenario: str, scenario_id: str) -> TheOccurring:
    result = call_llm_json(
        system=REFORMULATION_SYSTEM,
        user=reformulation_user(raw_scenario, scenario_id),
    )
    raw_nodes = result.get("nodes", [])
    if len(raw_nodes) != NUM_FACETS:
        raise ValueError(
            f"Reformulation: expected {NUM_FACETS} nodes, got {len(raw_nodes)}"
        )
    _validate_facet_set(raw_nodes, where="Reformulation")
    # Require explicit, non-empty assumptions per node. Reformulation is the
    # only place we have to enforce this; once The Occurring is constructed,
    # downstream code trusts the structure.
    for n in raw_nodes:
        if not n.get("assumptions"):
            raise ValueError(
                f"Reformulation: node '{n.get('facet')}' must include at least one assumption"
            )
    nodes = [
        OccurringNode(
            node_id=n["node_id"],
            facet=n["facet"],
            content=n["content"],
            assumptions=n["assumptions"],
            last_updated_cycle=0,
        )
        for n in raw_nodes
    ]
    nodes = _sort_by_facet(nodes)
    return TheOccurring(
        scenario_id=scenario_id,
        raw_scenario=raw_scenario,
        nodes=nodes,
    )


# ---------------------------------------------------------------------------
# Phase 2 — Critical thinking (main agent)
# One LLM call per (facet, dimension) pair. Never batched.
# ---------------------------------------------------------------------------

@traceable(run_type="chain")
def _ct_cell(
    memory,  # SessionMemory | None — captured in the parent thread, re-registered here
    occurring_json: str,
    cultural_frame: str,
    facet: str,
    dimension: str,
    cycle: int,
) -> NodeDimensionCell:
    # ThreadPoolExecutor workers start with a fresh contextvars context, so we
    # re-register the active SessionMemory inside the worker before any LLM
    # call runs. Without this, call_llm's budget / circuit-breaker hooks all
    # see _active_memory.get() == None and silently no-op.
    with active_memory(memory):
        result = call_llm_json_with_refinement(
            system=critical_thinking_cell_system(cultural_frame, facet, dimension),
            user=critical_thinking_cell_user(occurring_json, facet, dimension),
        )
    flags = _parse_flags(result.get("reflection_flags", []))
    meta = result.get("_refinement_meta", {})
    return NodeDimensionCell(
        facet=facet,
        dimension=dimension,
        content=result["content"],
        reflection_flags=flags,
        response_type=None,
        cycle_last_updated=cycle,
        refinement_attempts=int(meta.get("attempts", 0)),
        refinement_reasons=list(meta.get("reasons", [])),
    )


@traceable(run_type="chain", name="critical_thinking")
def run_critical_thinking(
    occurring: TheOccurring,
    cultural_frame: str,
    cycle: int,
) -> NodeReasoningGrid:
    occurring_json = occurring.model_dump_json()
    pairs = _cell_pairs()
    # Capture the active memory in the main thread; each worker re-registers
    # it on its own thread (see threading note in budget.py).
    memory = get_active_memory()
    # Capture this phase's LangSmith run so the threaded cell runs can attach to
    # it explicitly (contextvars don't cross the ThreadPoolExecutor boundary).
    run = get_current_run_tree()
    if run is not None:
        run.add_metadata({"cycle": cycle, "agent": "main", "frame": cultural_frame})
    with ThreadPoolExecutor(max_workers=_CELL_WORKERS) as ex:
        cells = list(ex.map(
            lambda fd: _ct_cell(
                memory, occurring_json, cultural_frame, fd[0], fd[1], cycle,
                langsmith_extra=_cell_trace_extra(run, "main", fd[0], fd[1], cycle),
            ),
            pairs,
        ))
    if len(cells) != NUM_CELLS:
        raise ValueError(f"Main grid: expected {NUM_CELLS} cells, got {len(cells)}")
    return NodeReasoningGrid(cycle=cycle, agent="main", cells=cells)


# ---------------------------------------------------------------------------
# Phase 3 — Challenger
# Passes the main grid as a linearized matrix map (easier for LLM to read).
# One LLM call per cell with response_type classification.
# ---------------------------------------------------------------------------

@traceable(run_type="chain")
def _challenger_cell(
    memory,  # SessionMemory | None — captured in the parent thread
    occurring_json: str,
    main_matrix: dict,
    challenger_frame: str,
    main_frame: str,
    challenger_style: str,
    facet: str,
    dimension: str,
    cycle: int,
) -> NodeDimensionCell:
    main_cell_content = main_matrix.get(facet, {}).get(dimension, {}).get("content", "")
    # Re-register the active memory on this worker thread (see _ct_cell).
    with active_memory(memory):
        result = call_llm_json_with_refinement(
            system=challenger_cell_system(
                challenger_frame, main_frame, challenger_style, facet, dimension
            ),
            user=challenger_cell_user(
                occurring_json=occurring_json,
                main_cell_content=main_cell_content,
                facet=facet,
                dimension=dimension,
            ),
            model=CHALLENGER_MODEL_ID,
        )
    flags = _parse_flags(result.get("reflection_flags", []))
    meta = result.get("_refinement_meta", {})
    return NodeDimensionCell(
        facet=facet,
        dimension=dimension,
        content=result["content"],
        reflection_flags=flags,
        response_type=result.get("response_type"),
        cycle_last_updated=cycle,
        refinement_attempts=int(meta.get("attempts", 0)),
        refinement_reasons=list(meta.get("reasons", [])),
    )


@traceable(run_type="chain", name="challenger")
def run_challenger(
    occurring: TheOccurring,
    main_grid: NodeReasoningGrid,
    challenger_frame: str,
    main_frame: str,
    cycle: int,
    challenger_style: str,
) -> NodeReasoningGrid:
    # Future RAG/retrieval would plug in here: enrich the user prompt per cell
    # with retrieved context scoped to challenger_frame. Wire it up by editing
    # _challenger_cell when a corpus and retriever exist.
    occurring_json = occurring.model_dump_json()
    main_cells_raw = [
        {
            "facet": c.facet,
            "dimension": c.dimension,
            "content": c.content,
            "reflection_flags": [f.model_dump() for f in c.reflection_flags],
        }
        for c in main_grid.cells
    ]
    main_matrix = linearize_grid_to_matrix_map(main_cells_raw)

    pairs = _cell_pairs()
    memory = get_active_memory()
    run = get_current_run_tree()
    if run is not None:
        run.add_metadata({
            "cycle": cycle, "agent": "challenger",
            "challenger_frame": challenger_frame, "main_frame": main_frame,
        })
    with ThreadPoolExecutor(max_workers=_CELL_WORKERS) as ex:
        cells = list(ex.map(
            lambda fd: _challenger_cell(
                memory,
                occurring_json, main_matrix,
                challenger_frame, main_frame, challenger_style,
                fd[0], fd[1], cycle,
                langsmith_extra=_cell_trace_extra(run, "challenger", fd[0], fd[1], cycle),
            ),
            pairs,
        ))

    if len(cells) != NUM_CELLS:
        raise ValueError(f"Challenger grid: expected {NUM_CELLS} cells, got {len(cells)}")
    return NodeReasoningGrid(cycle=cycle, agent="challenger", cells=cells)


# ---------------------------------------------------------------------------
# Phase 4 — Mediation
# Passes a compact cell-pair comparison and compressed history.
# ---------------------------------------------------------------------------

@traceable(run_type="chain", name="mediation")
def run_mediation(
    occurring: TheOccurring,
    main_grid: NodeReasoningGrid,
    challenger_grid: NodeReasoningGrid,
    all_mediation_history: list[ImplyingNodes],
    cycle: int,
) -> ImplyingNodes:
    occurring_json = occurring.model_dump_json()

    cell_pairs = []
    for facet in FACET_LIST:
        for dimension in DIMENSION_LIST:
            main_cell = main_grid.get_cell(facet, dimension)
            chall_cell = challenger_grid.get_cell(facet, dimension)
            cell_pairs.append({
                "facet": facet,
                "dimension": dimension,
                "main": main_cell.content if main_cell else "",
                "challenger": chall_cell.content if chall_cell else "",
                "response_type": chall_cell.response_type if chall_cell else None,
            })

    cell_pairs_json = json.dumps(cell_pairs, ensure_ascii=False)

    raw_history = [m.model_dump() for m in all_mediation_history]
    compressed = compress_historical_context(raw_history)
    compressed_history_json = json.dumps(compressed, ensure_ascii=False)

    result = call_llm_json(
        system=MEDIATION_SYSTEM,
        user=mediation_user(
            occurring_json=occurring_json,
            cell_pairs_json=cell_pairs_json,
            compressed_history_json=compressed_history_json,
            cycle=cycle,
        ),
        max_tokens=16000,
    )

    conflicts = [
        CellConflict(
            facet=c["facet"],
            dimension=c["dimension"],
            main_summary=c["main_content"],
            challenger_summary=c["challenger_content"],
            challenger_response_type=c.get("response_type", "challenge"),
            implication=c["implication"],
        )
        for c in result.get("conflicts", [])
    ]

    w = result.get("wholeness", {})
    wholeness = WholenessImplyingNodes(
        query_assumptions=w.get("query_assumptions", []),
        reformulated_query=w.get("reformulated_query"),
        new_objective=w.get("new_objective"),
        updated_challenger_perspective=w.get("updated_challenger_perspective"),
        objective_assumptions=w.get("objective_assumptions", []),
        perspective_assumptions=w.get("perspective_assumptions", []),
    )

    return ImplyingNodes(
        cycle=cycle,
        agreements=result.get("agreements", []),
        conflicts=conflicts,
        what_conflicts_imply=result.get("what_conflicts_imply", []),
        updated_assumptions=result.get("updated_assumptions", []),
        wholeness=wholeness,
        converged=result.get("converged", False),
    )


# ---------------------------------------------------------------------------
# Cycle integration — fold implying nodes back into The Occurring
# Turns the loop from a no-op into a real refinement cycle.
# ---------------------------------------------------------------------------

@traceable(run_type="chain", name="node_update")
def run_node_update(
    occurring: TheOccurring,
    mediation: ImplyingNodes,
) -> TheOccurring:
    # Nothing to fold in: keep the occurring untouched.
    if not mediation.updated_assumptions and not _wholeness_has_signal(mediation.wholeness):
        return occurring

    # Stamp updated nodes with the mediation cycle that produced the change.
    cycle = mediation.cycle

    result = call_llm_json(
        system=NODE_UPDATE_SYSTEM,
        user=node_update_user(
            occurring_json=occurring.model_dump_json(),
            updated_assumptions_json=json.dumps(
                mediation.updated_assumptions, ensure_ascii=False
            ),
            wholeness_json=mediation.wholeness.model_dump_json(),
            cycle=cycle,
        ),
    )

    raw_nodes = result.get("nodes", [])
    if len(raw_nodes) != NUM_FACETS:
        raise ValueError(f"Node update: expected {NUM_FACETS} nodes, got {len(raw_nodes)}")
    _validate_facet_set(raw_nodes, where="Node update")

    existing = {n.facet: n for n in occurring.nodes}
    nodes: list[OccurringNode] = []
    for n in raw_nodes:
        facet = n["facet"]
        prev = existing.get(facet)
        changed = bool(n.get("changed", False))
        nodes.append(OccurringNode(
            node_id=n.get("node_id", facet),
            facet=facet,
            content=n["content"],
            assumptions=n.get("assumptions", []),
            # Only advance the stamp on nodes the update actually touched.
            last_updated_cycle=cycle if changed else (
                prev.last_updated_cycle if prev else 0
            ),
        ))

    nodes = _sort_by_facet(nodes)
    return TheOccurring(
        scenario_id=occurring.scenario_id,
        raw_scenario=occurring.raw_scenario,
        nodes=nodes,
    )


def _wholeness_has_signal(wholeness: WholenessImplyingNodes) -> bool:
    return any([
        wholeness.query_assumptions,
        wholeness.reformulated_query,
        wholeness.new_objective,
        wholeness.objective_assumptions,
        wholeness.perspective_assumptions,
    ])


# ---------------------------------------------------------------------------
# Phase 5 — Constrained generation
# Receives ONLY distilled wholeness nodes. Not the full grids.
# ---------------------------------------------------------------------------

@traceable(run_type="chain", name="constrained_generation")
def run_constrained_generation(
    occurring: TheOccurring,
    mediation_history: list[ImplyingNodes],
    evaluation_feedback: str | None = None,
) -> str:
    distilled = [m.wholeness.model_dump() for m in mediation_history]
    final_mediation = mediation_history[-1]

    # Slide 8: output is bound by implications gathered across the ENTIRE
    # mediation cycle, not just the final one. Aggregate findings from every cycle.
    all_implications: list[str] = []
    for m in mediation_history:
        all_implications.extend(m.what_conflicts_imply)
        all_implications.extend(c.implication for c in m.conflicts)
    all_findings_json = json.dumps(all_implications, ensure_ascii=False)

    return call_llm(
        system=CONSTRAINED_GENERATION_SYSTEM,
        user=constrained_generation_user(
            occurring_json=occurring.model_dump_json(),
            distilled_wholeness_json=json.dumps(distilled, ensure_ascii=False),
            final_mediation_json=final_mediation.model_dump_json(),
            all_findings_json=all_findings_json,
            evaluation_feedback=evaluation_feedback,
        ),
        temperature=0.0
    )


# ---------------------------------------------------------------------------
# Judge (runs after generation, scores output quality)
# ---------------------------------------------------------------------------

@traceable(run_type="chain", name="judge")
def run_judge(
    final_output: str,
    mediation_history: list[ImplyingNodes],
    cycle: int,
) -> JudgeScore:
    mediation_json = json.dumps(
        [m.model_dump() for m in mediation_history],
        ensure_ascii=False,
    )
    result = call_llm_json(
        system=judge_system(),
        user=judge_user(final_output, mediation_json),
        temperature=0.0
    )
    return JudgeScore(
        cycle=cycle,
        empathy=result["empathy"],
        consensus=result["consensus"],
        escalation_risk=result["escalation_risk"],
        cultural_insight=result["cultural_insight"],
        reasoning=result["reasoning"],
    )


@traceable(run_type="chain", name="cycle_judge")
def run_cycle_judge(
    mediation: ImplyingNodes,
    cycle: int,
) -> JudgeScore:
    """In-loop evaluation of a single cycle's mediation (no final output yet)."""
    result = call_llm_json(
        system=judge_system(),
        user=judge_cycle_user(
            mediation_json=mediation.model_dump_json(),
            cycle=cycle,
        ),
        temperature=0.0
    )
    return JudgeScore(
        cycle=cycle,
        empathy=result["empathy"],
        consensus=result["consensus"],
        escalation_risk=result["escalation_risk"],
        cultural_insight=result["cultural_insight"],
        reasoning=result["reasoning"],
    )
