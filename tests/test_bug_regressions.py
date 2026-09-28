"""One test per bug found in the August audit.

Each was verified to fail against the code as it stood, so each of these would
have caught its bug. Offline: no Bedrock, no spend.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT, _ROOT / "thirdspace"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytest

import agents
from tests.fixtures import (
    make_occurring, make_grid, make_mediation, make_conflict, make_wholeness,
    VALID_IMPLICATION, VALID_RATIONALE,
)
from models import (
    ImplyingNodes, WholenessImplyingNodes, WholenessEntry, WorkingPerspective,
    CulturalIdentity, FACET_LIST, DIMENSION_LIST, NUM_CELLS,
)
from utils import prompts


# --- Bug 1: cycle 2 crashed when the mediator's wholeness came back empty -----

def test_empty_wholeness_still_yields_prior_context():
    """to_context must never return None.

    An all-empty wholeness is a documented outcome — the mediation prompt tells
    the model four of six fields should stay empty unless warranted. Returning
    None for it made both cell prompt builders raise, killing cycle 2 after
    cycle 1 had been paid for.
    """
    empty = ImplyingNodes(cycle=1, agreements=[], conflicts=[],
                          wholeness=WholenessImplyingNodes(), converged=False)
    ctx = empty.to_context(1)
    assert ctx is not None
    assert ctx["guidance"] == {}
    assert "no new findings" in ctx["_note"]


@pytest.mark.parametrize("builder,frame_kw", [
    (prompts.main_agent_cell_prompt, "cultural_frame"),
    (prompts.challenger_cell_prompt, "challenger_frame"),
])
def test_cycle_two_prompt_builds_from_empty_wholeness(builder, frame_kw):
    """The end of the same path: a cycle-2 prompt must build from that payload."""
    import json
    empty = ImplyingNodes(cycle=1, agreements=[], conflicts=[],
                          wholeness=WholenessImplyingNodes(), converged=False)
    prior = json.dumps(empty.to_context(1))
    system, user = builder(facet="agents", dimension="value", occurring_json="{}",
                           cycle=2, prior_mediation_json=prior, **{frame_kw: "germany"})
    assert system and user


# --- Bug 2: the A/B un-shuffle keyed on a model-supplied id string ------------

def test_ab_unshuffle_survives_wrong_case_ids(monkeypatch):
    """Every conflict must be attributed to the agent that actually wrote it,
    even when the model echoes the id back in a different case."""
    occurring = make_occurring()
    captured = {}

    def fake(system, user, **kw):
        import json
        payload = user.split("agreeing and not):\n")[1].split("\n\nPRIOR")[0]
        captured.update(json.loads(payload))
        return {"cycle": 1, "agreements": [], "wholeness": {}, "carried_forward": [],
                "converged": False,
                "conflicts": [
                    {"id": f"{f.upper()}/{d.upper()}",          # wrong case on purpose
                     "facet": f, "dimension": d,
                     "reading_a_content": pair["reading_a"],
                     "reading_b_content": pair["reading_b"],
                     "implication": VALID_IMPLICATION,
                     "conflict_type": "normative", "is_semantic_subtype": False,
                     "conflict_type_rationale": VALID_RATIONALE,
                     "severity": 3, "resolvability": "facilitated"}
                    for f, row in captured.items() for d, pair in row.items()]}

    monkeypatch.setattr(agents, "call_llm_json", fake)
    med = agents.run_mediation(occurring, make_grid("main"), make_grid("challenger"), [], 1)

    assert len(med.conflicts) == NUM_CELLS
    # Meaningful only if the shuffle actually swapped some cells.
    shuffled = sum(1 for row in captured.values() for p in row.values()
                   if p["reading_a"].startswith("CHALLENGER"))
    assert shuffled > 0, "seed produced no shuffled cells; test proves nothing"
    for c in med.conflicts:
        assert c.main_summary.startswith("MAIN"), f"{c.cell_id} misattributed"
        assert c.challenger_summary.startswith("CHALLENGER"), f"{c.cell_id} misattributed"


# --- Bug 3: dropped conflicts were logged but never counted ------------------

def test_invalid_conflict_is_dropped_and_counted(monkeypatch):
    from budget import active_memory
    from tests.fixtures import make_memory

    def fake(system, user, **kw):
        return {"cycle": 1, "agreements": [], "wholeness": {}, "carried_forward": [],
                "converged": False,
                "conflicts": [{
                    "id": "agents/value", "facet": "agents", "dimension": "value",
                    "reading_a_content": "a", "reading_b_content": "b",
                    "implication": "Too short.",              # fails the validator
                    "conflict_type": "normative", "is_semantic_subtype": False,
                    "conflict_type_rationale": VALID_RATIONALE,
                    "severity": 3, "resolvability": "facilitated"}]}

    monkeypatch.setattr(agents, "call_llm_json", fake)
    mem = make_memory()
    with active_memory(mem):
        med = agents.run_mediation(make_occurring(), make_grid("main"),
                                   make_grid("challenger"), [], 1)
    assert med.conflicts == []
    assert mem.conflicts_dropped == 1, "a dropped conflict must be counted, not just logged"


def test_wholeness_ref_matching_is_case_insensitive():
    """A ref differing only in case is a match, not an invention — and is stored
    in canonical form so downstream ref resolution agrees."""
    entries = agents._entries(
        [{"statement": "finding", "refs": ["AGENTS/VALUE", " agents / value "]}],
        valid_ids={"agents/value"}, where="test")
    assert len(entries) == 1
    assert entries[0].refs == ["agents/value", "agents/value"]


# --- Bug 4: dedupe edited what mediation compared ----------------------------

def test_repetition_is_recorded_not_removed():
    """Cell text must reach mediation exactly as the agent wrote it."""
    grid = make_grid("main", content="Shared opening line. Cell specific point.")
    before = [c.content for c in grid.cells]
    agents._record_repetition(grid.cells, "main", 1)
    assert [c.content for c in grid.cells] == before, "content must not be edited"
    assert all(c.content_hashes for c in grid.cells), "repetition must still be measurable"


# --- Bug 5: scenario_id ignored the run configuration ------------------------

def test_scenario_id_distinguishes_run_configurations():
    from main import _make_scenario_id
    base = ("same story", "germany", "jordan")
    ids = {
        _make_scenario_id(*base, max_cycles=1),
        _make_scenario_id(*base, max_cycles=3),
        _make_scenario_id(*base, max_cycles=1, mediator_frame="other"),
    }
    assert len(ids) == 3, "different configurations must not share an output file"
    assert _make_scenario_id(*base, max_cycles=2) == _make_scenario_id(*base, max_cycles=2)


# --- Bug 6: skip_generation had two conflicting defaults ---------------------

def test_skip_generation_default_agrees_across_entry_points():
    import inspect
    from main import run_scenario
    from models import SessionMemory
    param = inspect.signature(run_scenario).parameters["skip_generation"].default
    field = SessionMemory.model_fields["skip_generation"].default
    assert param == field, "one switch must not mean opposite things by construction path"


# --- Bug 7: the router could raise on a failed embedding ---------------------

def test_router_survives_embedding_failure(monkeypatch):
    """A budget or network failure in the convergence metric must end the loop
    cleanly, not raise out of a conditional edge and lose the paid run."""
    import graph
    from tests.fixtures import make_memory

    def boom(*a, **kw):
        raise RuntimeError("budget exceeded")

    monkeypatch.setattr(graph, "convergence_similarity", boom)
    mem = make_memory(cycle_policy="convergence", max_cycles=5)
    mem.mediation_history = [make_mediation(1), make_mediation(2)]
    mem.current_cycle = 2
    result = graph.should_loop({"memory": mem})
    assert result in {"integrate", "generate", "end"}
    assert mem.stability_log[-1]["similarity"] is None


# --- Working perspective: no longer an arbitrary [0] pick --------------------

def test_working_perspective_keeps_every_refinement():
    wp = WorkingPerspective()
    entries = [WholenessEntry(statement="lens one"), WholenessEntry(statement="lens two")]
    wp = wp.with_updates(2, entries)
    assert len(wp.entries) == 2, "the old code kept only updated_perspectives[0]"
    # Accumulates across cycles, and does not duplicate.
    wp = wp.with_updates(3, [WholenessEntry(statement="lens one"),
                             WholenessEntry(statement="lens three")])
    assert [e.statement for e in wp.entries] == ["lens one", "lens two", "lens three"]
    assert wp.cycle_last_updated == 3
    assert "lens three" in wp.render_for_prompt(CulturalIdentity(label="jordan"))


# --- Interventions are wired, not orphaned -----------------------------------

def test_every_conflict_carries_its_intervention():
    for ctype, resolvability in [("normative", "facilitated"),
                                 ("power_related", "structural_change"),
                                 ("factual", "external_standard")]:
        c = make_conflict(conflict_type=ctype, resolvability=resolvability)
        assert c.intervention, f"{ctype} conflict carries no intervention"


def test_generation_prompt_carries_all_present_types():
    """Not just the dominant type — five of six interventions used to be unreachable."""
    system = prompts.build_constrained_generation_system(
        None, 900, weighted_types=["power_related", "normative"])
    assert "structural asymmetry" in system      # power_related's prose
    assert "incompatible value frameworks" in system   # normative's prose
