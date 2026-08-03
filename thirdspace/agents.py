from __future__ import annotations

import difflib
import hashlib
import json 
import logging
import os 
import random 
import re
from concurrent.futures import ThreadPoolExecutor

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from models import (
    TheOccurring, OccurringNode,
    NodeReasoningGrid, NodeDimensionCell, ReflectionFlag,
    ImplyingNodes, CellConflict, WholenessImplyingNodes, WholenessEntry,
    ConflictTypeSummary, CompressedHistoryEntry,
    FACET_LIST, DIMENSION_LIST, NUM_FACET, NUM_CELLS,
    CulturalFrame, CulturalIdentity, WorkingPerspective, CultureAxis,
    CONFLICT_TYPE_DEFS,
)
from llm import call_llm, call_llm_json, call_llm_json_with_refinement, CHALLENGER_MODEL_ID
from budget import active_memory, get_active_memory
from utils import prompts

logger = logging.getLogger(__name__)

#cell-level concurrecnty.
_CELL_WORKERS = max(1, int(os.environ.get("THIRDSPACE_CELL_WORKERS", "8")))

#skipping a cell pair in mediation when both readings are near-identical
AGREEMENT_SIMILARITY_THRESHOLD = 0.90

DEDUPE_MIN_RATIO= 0.4

GENERATION_MAX_TOKENS = 4000
GENERATION_WORD_BUDGET = 900

def _cell_pairs() -> list[tuple[str, str]]:
    return [(str(f), str(d)) for f in FACET_LIST for d in DIMENSION_LIST]

def _similar(a:str, b:str) -> float:
    #1.0 means they are identical , 0.0 means they have nothing in common
    return difflib.SequenceMatcher(None, a.strip(), b.strip()).ratio()

def _render_frame(frame:CulturalFrame | CulturalIdentity | None) -> str:
    #string | list[culturalAxis] | CulturalIdentity - > prompt string.
    """ means NEUTRAL mode — the prompt drops its frame block entirely rather
    than rendering an empty one. Local to agents.py because prompts.py does
    not export a render_frame.
    """

    if frame is None:
        return ""
    if isinstance(frame, CulturalIdentity):
        return frame.render()
    if isinstance(frame, list):
        return "; ".join(f"{a.dimension}: {a.value}" for a in frame if isinstance(a, CultureAxis))
    return str(frame)


#cell helpers
def _parse_flags(raw_flags) -> list[ReflectionFlag]:
    """Normalize reflection_flags: models return dicts, bare strings, or one string.

    Flags are TELEMETRY ONLY — they must never trigger a retry. An honest model
    asked to self-criticize always finds something; treating that as a failure
    cost a previous run 252 wasted calls that fixed nothing.
    """
    flags: list[ReflectionFlag] =[]
    if isinstance(raw_flags, str):
        raw_flags = [raw_flags]
    for f in raw_flags or []:
        if isinstance(f, dict):
            flags.append(ReflectionFlag(measure=f.get("measure", "unknown"),
                                        issue=f.get("issue", "unspecified")))
        elif isinstance(f,str) and f.strip():
            flags.append(ReflectionFlag(measure=f.strip(), issue="flagged by model"))
    return flags


def _cell_trace_extra(parent, agent:str, facet:str, dimension:str, cycle:int) -> dict:
    # LangSmith run config for one cell call. ThreadPoolExecutor workers start
    # with a fresh contextvars context, so a @traceable cell cannot discover its
    # parent run on its own — we capture the phase run in the main thread and
    # pass it in. Metadata values must be plain JSON types (a set is not).
    return {
        "name":f"{agent}_cell[{facet}/{dimension}]",
        "parent": parent,
        "metadata": {"facet": facet,
                     "dimension": dimension,
                     "cycle": cycle,
                     "agent": agent},
        "tags": ["cell", agent]
    }


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
 
def _split_sentences(text:str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text or "") if s.strip()]

def _hash_sentence(sentence:str) -> str:
    "Normalize first"
    norm = re.sub(r"[^a-z0-9 ]", "", sentence.lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]

def _dedupe_grid(cells: list[NodeDimensionCell], agent: str, cycle: int) -> list[NodeDimensionCell]:
    """Strip sentences an EARLIER cell in this grid already used.

    30 cells each restating the same background is the token leak the design
    calls out. Runs as a sequential post-pass because parallel cells can't
    share the `seen` set. First cell to use a sentence keeps it; the safety
    floor means a repetitive cell degrades instead of vanishing. content_hashes
    is recorded either way, so you can measure repetition even when nothing
    was stripped.
    """
    seen: set[str] = set()
    stripped = 0
    for cell in cells:
        sentences = _split_sentences(cell.content)
        hashes = [_hash_sentence(s) for s in sentences]
        kept = [s for s, h in zip(sentences, hashes) if h not in seen]
        seen.update(hashes)
        cell.content_hashes = hashes
        if not kept or len(kept) == len(sentences):
            continue
        new_content = " ".join(kept)
        if len(new_content) < DEDUPE_MIN_RATIO * len(cell.content):
            continue                       # would gut the cell; leave it alone
        stripped += len(sentences) - len(kept)
        cell.content = new_content
    if stripped:
        logger.info(f"[dedupe {agent} cycle={cycle}] removed {stripped} repeated sentences")
    return cells


#validators

def _validate_facet_set(raw_nodes: list[dict], where:str) -> None:
    "reject a node lis tthat doesn't cover all 5 objects exactly once"

    facets = [n.get("facet") for n in raw_nodes]
    unknown = [f for f in facets if f not in FACET_LIST]
    if unknown:
        raise ValueError(f"{where}: unknown facet(s) {unknown}; "
                         f"expected subset of {[str(f) for f in FACET_LIST]}")
    seen: set[str] = set()
    duplicates: list[str] = []
    for f in facets:
        if f in seen:
            duplicates.append(f)
        seen.add(f)
    if duplicates:
        raise ValueError(f"{where}: duplicate facet(s) {duplicates}")
    missing = [str(f) for f in FACET_LIST if f not in seen]
    if missing:
        raise ValueError(f"{where}: missing facet(s) {missing}")


def _sort_by_facet(nodes:list[OccurringNode]) -> list[OccurringNode]:
    return sorted(nodes, key= lambda n: FACET_LIST.index(n.facet))

def _entries(raw, valid_ids: set[str], where: str) -> list[WholenessEntry]:
    """Parse [{statement, refs}] into WholenessEntry, dropping invented refs.

    This is the whole point of carrying refs: a fabricated finding usually
    cites fabricated conflicts. An entry whose refs are ALL unknown is dropped;
    one with some valid refs keeps only those. Bare strings are tolerated so a
    model that regresses to the old shape doesn't kill the run.
    """
    out: list[WholenessEntry] = []
    for e in raw or []:
        if isinstance(e, str) and e.strip():
            out.append(WholenessEntry(statement=e.strip()))
            continue
        if not isinstance(e, dict):
            continue
        statement = str(e.get("statement", "")).strip()
        if not statement:
            continue
        refs = [str(r) for r in (e.get("refs") or [])]
        good = [r for r in refs if r in valid_ids]
        bogus = [r for r in refs if r not in valid_ids]
        if refs and not good:
            logger.warning(f"[{where}] dropping finding, every ref invented {bogus}: "
                           f"{statement[:70]}")
            continue
        if bogus:
            logger.warning(f"[{where}] dropped invented refs {bogus} from: {statement[:70]}")
        out.append(WholenessEntry(statement=statement, refs=good))
    return out


#Phase 1 reformulation. turning the raw scenario to The Occuring (5 nodes)

@traceable(run_type="chain", name="reformulation")
def run_reformulation(raw_scenario:str, scenario_id:str) -> TheOccurring:
    results = call_llm_json(
        system=prompts.REFORMULATION_SYSTEM, 
        user=prompts.reformulation_user(raw_scenario, scenario_id)
    )
    raw_nodes = results.get("nodes",[])

    #Fail loud here
    if len(raw_nodes) != NUM_FACET:
        raise ValueError(f"Reformulation: expected {NUM_FACET} nodes, got {len(raw_nodes)}")
    _validate_facet_set(raw_nodes, where="Reformulation")

    for n in raw_nodes:
        if not n.get("assumptions"):
            #assumptions_by_facet() is the challenger's cycle-1 baseline; an
            # empty list leaves it with nothing to contest.
            raise ValueError(f"Reformulation:node '{n.get('facet')} needs >=1 assumptions'")


    nodes = _sort_by_facet([
        OccurringNode(node_id=n.get("node_id", n["facet"]), facet = n["facet"],
        content=n["content"],
        assumptions=n["assumptions"],
        last_updated_cycle=0)
        for n in raw_nodes
    ]) 

    return TheOccurring(scenario_id=scenario_id, raw_scenario=raw_scenario, nodes= nodes)



#phase and 3 the two grids. 5 x 6

@traceable(run_type="chain")
#critical thinking cells
def _ct_cell(memory, occurring_json: str, cultural_frame: str, facet: str,
             dimension: str, cycle: int, prior_mediation_json: str | None) -> NodeDimensionCell:

    system, user = prompts.main_agent_cell_prompt(
        cultural_frame=cultural_frame, facet=facet, dimension=dimension,
        occurring_json=occurring_json, cycle=cycle,
        prior_mediation_json=prior_mediation_json,
    )

    # Re-register the SessionMemory INSIDE the worker. ThreadPoolExecutor does
    # not propagate contextvars, so without this the budget cap and circuit
    # breaker silently no-op for every cell call — no error, just zeros.

    with active_memory(memory=memory):
        result = call_llm_json_with_refinement(system, user)
    meta = result.get("_refinement_meta", {})
    return NodeDimensionCell(
        facet=facet, dimension=dimension, content=result["content"],
        reflection_flags=_parse_flags(result.get("reflection_flags", [])),
        cycle_last_updated=cycle,
        # Telemetry only. If a run shows attempts > 0 across the board, the old
        # "retry on self-criticism" bug is back.
        refinement_attempts=int(meta.get("attempts", 0)),
        refinement_reasons=list(meta.get("reasons", [])),
    )


@traceable(run_type = "chain", name='critical_thinking_main')
def run_critical_thinking(occurring: TheOccurring, cultural_frame: CulturalFrame,
                          cycle: int, prior_mediation: ImplyingNodes | None = None
                          ) -> NodeReasoningGrid:

    frame = _render_frame(cultural_frame)
    # Each cell gets its focus node in full and the others truncated, instead
    # of the whole Occurring on all 30 calls.
    slim = {str(f): occurring.slim_context(str(f)) for f in FACET_LIST}

    prior_context = prior_mediation.to_context(prior_mediation.cycle) if prior_mediation else None
    prior_json = json.dumps(prior_context, ensure_ascii=False) if prior_context else None 

    memory = get_active_memory()
    run  = get_current_run_tree()
    
    if run is not None:
        run.add_metadata({"cycle": cycle, 
                          "agent": "main",
                           "frame":frame })

    with ThreadPoolExecutor(max_workers=_CELL_WORKERS) as ex:
        cells = list(ex.map(
            lambda fd: _ct_cell(
                memory, slim[fd[0]], frame, fd[0], fd[1], cycle, prior_json,
                langsmith_extra=_cell_trace_extra(run, "main", fd[0], fd[1], cycle)),
            _cell_pairs()))


    if len(cells) != NUM_CELLS:
        raise ValueError(f"Main grid: expected {NUM_CELLS} cells, got {len(cells)}")
    return NodeReasoningGrid(cycle=cycle, agent="main",
                             cells=_dedupe_grid(cells, "main", cycle))


@traceable(run_type="chain")
def _challenger_cell(memory, occurring_json: str, challenger_frame: str, facet: str,
                     dimension: str, cycle: int, occurring_assumptions_json: str | None,
                     prior_mediation_json: str | None) -> NodeDimensionCell:

    system, user = prompts.challenger_cell_prompt(
        challenger_frame=challenger_frame, facet=facet, dimension=dimension,
        occurring_json=occurring_json, cycle=cycle,
        occurring_assumptions_json=occurring_assumptions_json,
        prior_mediation_json=prior_mediation_json,
    )

    with active_memory(memory):
        result = call_llm_json_with_refinement(system, user, model = CHALLENGER_MODEL_ID)
    meta = result.get("_refinement_meta", {})
    return NodeDimensionCell(
        facet=facet, dimension=dimension, content=result["content"],
        reflection_flags=_parse_flags(result.get("reflection_flags", [])),
        cycle_last_updated=cycle,
        refinement_attempts=int(meta.get("attempts", 0)),
        refinement_reasons=list(meta.get("reasons", [])),
    )

@traceable(run_type="chain", name="challenger")
def run_challenger(occurring: TheOccurring, challenger_identity: CulturalIdentity,
                   challenger_working_perspective: WorkingPerspective,
                   cycle: int, prior_mediation: ImplyingNodes | None = None
                   ) -> NodeReasoningGrid:
    """Adversarial, but NOT given the main grid.

    It contests the Occurring's own assumptions (cycle 1) or the mediator's
    synthesis (cycle 2+), so the two grids stay independently derived and the
    mediator has two genuine readings to compare rather than a text and its
    rebuttal.
    """
    frame = challenger_working_perspective.render_for_prompt(challenger_identity)
    slim = {str(f): occurring.slim_context(str(f)) for f in FACET_LIST}
    assumption = occurring.assumptions_by_facet()
    prior_context = prior_mediation.to_context(prior_mediation.cycle) if prior_mediation else None
    prior_json = json.dumps(prior_context, ensure_ascii=False) if prior_context else None

    memory = get_active_memory()
    run = get_current_run_tree()
    if run is not None:
        run.add_metadata({"cycle": cycle, 
                          "agent": "challenger",
                           "challenger_frame":frame })

    def _one(fd):
        facet, dimension = fd
        assumptions_json = (json.dumps({facet: assumption.get(facet, [])}, ensure_ascii=False) if cycle == 1 else None)
        return _challenger_cell(
            memory, slim[facet], frame, facet, dimension, cycle,
            assumptions_json, prior_json,
            langsmith_extra=_cell_trace_extra(run, "challenger", facet, dimension, cycle))


    with ThreadPoolExecutor(max_workers=_CELL_WORKERS) as ex:
        cells = list(ex.map(
            _one, _cell_pairs()
        ))

    if len(cells) != NUM_CELLS:
        raise ValueError(f"Challenger grid: expected {NUM_CELLS} cells, got {len(cells)}")

    return NodeReasoningGrid(cycle=cycle, agent="challenger",
                             cells=_dedupe_grid(cells, "challenger", cycle))




#Phase 4-6: mediation, nodeupdate and generation
@traceable(run_type="chain", name="mediation")
def run_mediation(occurring: TheOccurring, main_grid: NodeReasoningGrid,
                  challenger_grid: NodeReasoningGrid,
                  compressed_history: list[CompressedHistoryEntry],
                  cycle: int) -> ImplyingNodes:

    """Frame-blind AND label-blind adjudication of the two grids.
    Shuffle which agent is "reading_a" per cell. The prompt promises the
    mediator can't tell who wrote what — emitting main/challenger keys would
    break that promise, and a fixed order invites position bias. Seeded on
    scenario+cycle so a run is reproducible"""

    rng = random.Random(f"{occurring.scenario_id}:{cycle}")
    ab_map: dict[str, tuple[str, str]] = {}

    pairs_by_facet:dict = {}
    skipped = 0
    for f in FACET_LIST:
        facet, row = str(f), {}
        for d in DIMENSION_LIST:
            dim = str(d)
            mc = main_grid.get_cell(facet, dim)
            cc = challenger_grid.get_cell(facet, dim)
            main_txt = mc.content if mc else ""
            chall_txt = cc.content if cc else ""

            # Input trimming: don't spend tokens on pairs that plainly agree.
            if main_txt and chall_txt and _similar(main_txt, chall_txt) >= AGREEMENT_SIMILARITY_THRESHOLD:
                skipped += 1
                continue
            cell_id = f"{facet}/{dim}"
            if rng.random() < 0.5:
                row[dim] = {"reading_a": main_txt, "reading_b": chall_txt}
                ab_map[cell_id] = ("main", "challenger")
            else:
                row[dim] = {"reading_a": chall_txt, "reading_b": main_txt}
                ab_map[cell_id] = ("challenger", "main")
        # Facet level — after the dimension loop, so all 6 dimensions land.
        if row:
            pairs_by_facet[facet] = row
    if skipped:
        logger.info(f"[mediation cycle={cycle}] skipped {skipped}/{NUM_CELLS} near-identical pairs")

    result = call_llm_json(
        system=prompts.MEDIATION_SYSTEM,
        user=prompts.mediation_user(
            occurring_json=occurring.to_nodes_json(),
            cell_pairs_json=json.dumps(pairs_by_facet, ensure_ascii=False),
            compressed_history_json=json.dumps(
                [e.model_dump(mode="json") for e in compressed_history], ensure_ascii=False),
            cycle=cycle),
        max_tokens=16000, temperature=0.1,
    )
    # Un-shuffle A/B back to main/challenger, then validate. A conflict that
    # violates the resolvability whitelist is DROPPED with a log line, never
    # coerced — models.py is the enforcement point for the taxonomy.
    conflicts: list[CellConflict] = []
    for c in result.get("conflicts", []):
        try:
            facet, dim = c["facet"], c["dimension"]
            cell_id = c.get("id") or f"{facet}/{dim}"
            role_a, _ = ab_map.get(cell_id, ("main", "challenger"))
            a, b = c.get("reading_a_content", ""), c.get("reading_b_content", "")
            main_summary, chall_summary = (a, b) if role_a == "main" else (b, a)
            conflicts.append(CellConflict(
                facet=facet, dimension=dim, cell_id=cell_id,
                main_summary=main_summary, challenger_summary=chall_summary,
                implication=c["implication"],
                conflict_type=c["conflict_type"],
                conflict_type_rationale=c.get("conflict_type_rationale", ""),
                is_semantic_subtype=bool(c.get("is_semantic_subtype", False)),
                severity=c.get("severity", 3),
                resolvability=c["resolvability"],
            ))
        except Exception as exc:
            logger.warning(f"[mediation cycle={cycle}] dropped invalid conflict "
                           f"({c.get('facet')}, {c.get('dimension')}): {exc}")
    # A ref may cite this cycle's conflicts or a finding carried from history.
    carried = [str(r) for r in result.get("carried_forward", [])]
    valid_ids = {c.cell_id for c in conflicts} | set(carried)
    where = f"mediation cycle={cycle}"
    w = result.get("wholeness", {}) or {}
    wholeness = WholenessImplyingNodes(
        what_conflicts_imply=_entries(w.get("what_conflicts_imply"), valid_ids, where),
        updated_assumptions=_entries(w.get("updated_assumptions"), valid_ids, where),
        comparison_assumptions=_entries(w.get("comparison_assumptions"), valid_ids, where),
        reformulated_queries=_entries(w.get("reformulated_queries"), valid_ids, where),
        new_objectives=_entries(w.get("new_objectives"), valid_ids, where),
        updated_perspectives=_entries(w.get("updated_perspectives"), valid_ids, where),
    )
    return ImplyingNodes(
        cycle=cycle,
        agreements=_entries(result.get("agreements"), valid_ids, where),
        conflicts=conflicts,
        wholeness=wholeness,
        carried_forward=carried,
        converged=bool(result.get("converged", False)),
    )



@traceable(run_type="chain", name="node_update")
def run_node_update(occurring: TheOccurring, mediation: ImplyingNodes) -> TheOccurring:
    """Fold the mediator's findings back into The Occurring.

    This is what makes the loop a refinement cycle rather than a re-derivation:
    without it, cycle 2 reasons over the same nodes cycle 1 did and convergence
    measures nothing but sampling noise.
    """
    if not mediation.wholeness.has_signal():
        logger.info(f"[node_update cycle={mediation.cycle}] no signal; Occurring unchanged")
        return occurring

    cycle = mediation.cycle
    result = call_llm_json(
        system=prompts.NODE_UPDATE_SYSTEM,
        user=prompts.node_update_user(
            occurring_json=occurring.to_nodes_json(),
            wholeness_json=mediation.wholeness.model_dump_json(),
            cycle=cycle),
    )
    raw_nodes = result.get("nodes", [])
    if len(raw_nodes) != NUM_FACET:
        raise ValueError(f"Node update: expected {NUM_FACET} nodes, got {len(raw_nodes)}")
    _validate_facet_set(raw_nodes, where="Node update")

    # Telemetry: instructions the model reported it could not apply. Reporting
    # one is correct behaviour; inventing a place to apply it is not.
    for u in result.get("unapplied", []) or []:
        logger.info(f"[node_update cycle={cycle}] unapplied: "
                    f"{str(u.get('instruction', '?'))[:80]} — {u.get('reason', '?')}")

    existing = {str(n.facet): n for n in occurring.nodes}
    nodes: list[OccurringNode] = []
    for n in raw_nodes:
        facet = n["facet"]
        prev = existing.get(facet)
        # changed=false means content-IDENTITY: reuse the previous node object
        # verbatim, so a no-op update cannot silently reword it.
        if prev is not None and not bool(n.get("changed", False)):
            nodes.append(prev)
            continue
        nodes.append(OccurringNode(
            node_id=n.get("node_id", facet), facet=facet, content=n["content"],
            assumptions=n.get("assumptions", list(prev.assumptions) if prev else []),
            last_updated_cycle=cycle,
        ))
    return TheOccurring(scenario_id=occurring.scenario_id,
                        raw_scenario=occurring.raw_scenario,
                        nodes=_sort_by_facet(nodes))


@traceable(run_type="chain", name="constrained_generation")
def run_constrained_generation(occurring: TheOccurring,
                               mediation_history: list[ImplyingNodes],
                               mediator_frame: str) -> str:
    """Write the final document from DISTILLED findings only — never the grids."""
    if not mediation_history:
        raise ValueError("run_constrained_generation requires >=1 completed mediation cycle.")

    final = mediation_history[-1]
    implications: list[str] = []
    for m in mediation_history:
        implications.extend(e.statement for e in m.wholeness.what_conflicts_imply)
        implications.extend(c.implication for c in m.conflicts)

    # Pool the per-cycle summaries instead of re-flattening every conflict.
    cross_cycle = ConflictTypeSummary.aggregate(
        [m.conflict_type_summary for m in mediation_history])
    # Severity >= 3 only: the document should lead with what's load-bearing.
    salient = sorted([c for m in mediation_history for c in m.conflicts if c.severity >= 3],
                     key=lambda c: c.severity, reverse=True)

    doc = call_llm(
        system=prompts.build_constrained_generation_system(cross_cycle, GENERATION_WORD_BUDGET),
        user=prompts.constrained_generation_user(
            occurring_json=occurring.to_nodes_json(),
            distilled_wholeness_json=json.dumps(
                [m.wholeness.model_dump(mode="json") for m in mediation_history],
                ensure_ascii=False),
            final_mediation_json=final.model_dump_json(),
            all_findings_json=json.dumps(implications, ensure_ascii=False),
            conflict_type_summary_json=(
                cross_cycle.model_dump_json() if cross_cycle else "null"),
            salient_conflicts_json=json.dumps(
                [c.model_dump(mode="json") for c in salient], ensure_ascii=False),
            word_budget=GENERATION_WORD_BUDGET),
        temperature=0.5,
        max_tokens=GENERATION_MAX_TOKENS,   # hard cap; the word budget is only a request
    )

    # Provenance header — prepended in code, never LLM-authored, so it's exact.
    header = ("──────────────────────────────────────────────\n"
              f'ANALYSIS FRAME: mediated from a "{mediator_frame}" standpoint. A finding that '
              "reads as neutral from this frame may systematically advantage one party; read "
              "accordingly.\n"
              "──────────────────────────────────────────────\n\n")
    return header + doc