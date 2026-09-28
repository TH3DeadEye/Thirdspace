"""Phase functions in agents.py. Every LLM caller is stubbed — no Bedrock, no spend."""
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT, _ROOT / "thirdspace"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytest

import agents
from models import (
    CulturalIdentity, WorkingPerspective, WholenessEntry, WholenessImplyingNodes,
    FACET_LIST, DIMENSION_LIST, NUM_FACET, NUM_CELLS,
)
from tests.fixtures import (
    make_occurring, make_grid, make_mediation, make_conflict,
    VALID_IMPLICATION, VALID_RATIONALE,
)


def _raw_nodes(**overrides):
    nodes = [{"node_id": str(f), "facet": str(f), "content": f"content {f}",
              "assumptions": [f"assumes {f}"]} for f in FACET_LIST]
    nodes = overrides.get("nodes", nodes)
    return {"nodes": nodes}


def _cell_response(*_a, **_kw):
    return {"content": "A specific claim about this cell. It rests on one named "
                       "detail. It runs out where the material stops.",
            "reflection_flags": [], "_refinement_meta": {"attempts": 0, "reasons": []}}


class TestReformulation:
    def test_happy_path_returns_five_nodes(self, monkeypatch):
        monkeypatch.setattr(agents, "call_llm_json", lambda *a, **k: _raw_nodes())
        occ = agents.run_reformulation("raw", "scn")
        assert len(occ.nodes) == NUM_FACET
        assert [str(n.facet) for n in occ.nodes] == [str(f) for f in FACET_LIST]

    def test_wrong_node_count_raises(self, monkeypatch):
        monkeypatch.setattr(agents, "call_llm_json",
                            lambda *a, **k: _raw_nodes(nodes=[{"facet": "agents",
                                                               "content": "c",
                                                               "assumptions": ["a"]}]))
        with pytest.raises(ValueError, match="expected"):
            agents.run_reformulation("raw", "scn")

    def test_unknown_facet_rejected(self, monkeypatch):
        bad = [{"node_id": "x", "facet": "wormholes", "content": "c", "assumptions": ["a"]}]
        bad += [{"node_id": str(f), "facet": str(f), "content": "c", "assumptions": ["a"]}
                for f in FACET_LIST[1:]]
        monkeypatch.setattr(agents, "call_llm_json", lambda *a, **k: _raw_nodes(nodes=bad))
        with pytest.raises(ValueError, match="unknown facet"):
            agents.run_reformulation("raw", "scn")

    def test_empty_assumptions_rejected(self, monkeypatch):
        """The challenger's cycle-1 baseline is these assumptions; an empty list
        leaves it nothing to contest."""
        nodes = [{"node_id": str(f), "facet": str(f), "content": "c",
                  "assumptions": [] if f == FACET_LIST[0] else ["a"]} for f in FACET_LIST]
        monkeypatch.setattr(agents, "call_llm_json", lambda *a, **k: _raw_nodes(nodes=nodes))
        with pytest.raises(ValueError, match="assumptions"):
            agents.run_reformulation("raw", "scn")

    def test_nodes_sorted_into_canonical_order(self, monkeypatch):
        shuffled = list(reversed([{"node_id": str(f), "facet": str(f), "content": "c",
                                   "assumptions": ["a"]} for f in FACET_LIST]))
        monkeypatch.setattr(agents, "call_llm_json", lambda *a, **k: _raw_nodes(nodes=shuffled))
        occ = agents.run_reformulation("raw", "scn")
        assert [str(n.facet) for n in occ.nodes] == [str(f) for f in FACET_LIST]


class TestGrids:
    def test_main_grid_covers_every_cell(self, monkeypatch):
        monkeypatch.setattr(agents, "call_llm_json_with_refinement", _cell_response)
        grid = agents.run_critical_thinking(make_occurring(), "germany", cycle=1)
        assert len(grid.cells) == NUM_CELLS
        assert {(str(c.facet), str(c.dimension)) for c in grid.cells} == {
            (str(f), str(d)) for f in FACET_LIST for d in DIMENSION_LIST}
        assert grid.agent == "main"

    def test_challenger_grid_covers_every_cell(self, monkeypatch):
        monkeypatch.setattr(agents, "call_llm_json_with_refinement", _cell_response)
        grid = agents.run_challenger(
            make_occurring(), CulturalIdentity(label="jordan"),
            WorkingPerspective(), cycle=1)
        assert len(grid.cells) == NUM_CELLS
        assert grid.agent == "challenger"

    def test_challenger_never_receives_the_main_grid(self, monkeypatch):
        """The two readings must be independently derived, or mediation is
        comparing a text against its own rebuttal."""
        seen = []

        def spy(system, user, **kw):
            seen.append(system + user)
            return _cell_response()

        monkeypatch.setattr(agents, "call_llm_json_with_refinement", spy)
        agents.run_challenger(make_occurring(), CulturalIdentity(label="jordan"),
                              WorkingPerspective(), cycle=1)
        assert seen and not any("MAIN-SAYS" in s for s in seen)


class TestRefinementTelemetry:
    def test_first_pass_records_zero_attempts(self, monkeypatch):
        monkeypatch.setattr(agents, "call_llm_json_with_refinement", _cell_response)
        grid = agents.run_critical_thinking(make_occurring(), "germany", cycle=1)
        assert all(c.refinement_attempts == 0 for c in grid.cells)

    def test_retry_metadata_propagates_to_cell(self, monkeypatch):
        def retried(*a, **k):
            r = _cell_response()
            r["_refinement_meta"] = {"attempts": 1, "reasons": ["empty content"]}
            return r

        monkeypatch.setattr(agents, "call_llm_json_with_refinement", retried)
        grid = agents.run_critical_thinking(make_occurring(), "germany", cycle=1)
        assert all(c.refinement_attempts == 1 for c in grid.cells)
        assert all(c.refinement_reasons == ["empty content"] for c in grid.cells)


class TestMediation:
    def _response(self, **over):
        base = {
            "cycle": 1,
            "agreements": [{"statement": "both accept the sequence", "refs": ["agents/value"]}],
            "conflicts": [{
                "id": "agents/value", "facet": "agents", "dimension": "value",
                "reading_a_content": "A", "reading_b_content": "B",
                "implication": VALID_IMPLICATION, "conflict_type": "normative",
                "is_semantic_subtype": False, "conflict_type_rationale": VALID_RATIONALE,
                "severity": 4, "resolvability": "facilitated"}],
            "wholeness": {"what_conflicts_imply": [
                {"statement": "obligation, not money", "refs": ["agents/value"]}]},
            "carried_forward": [], "converged": False}
        base.update(over)
        return base

    def test_parses_into_implying_nodes(self, monkeypatch):
        monkeypatch.setattr(agents, "call_llm_json", lambda *a, **k: self._response())
        med = agents.run_mediation(make_occurring(), make_grid("main"),
                                   make_grid("challenger"), [], 1)
        assert len(med.conflicts) == 1
        assert med.conflict_type_summary.dominant_type == "normative"
        assert med.wholeness.what_conflicts_imply[0].refs == ["agents/value"]

    def test_every_cell_pair_is_sent(self, monkeypatch):
        """No pre-filtering: a pair the mediator never sees produces neither a
        conflict nor an agreement, which silently shrinks the denominator."""
        captured = {}

        def spy(system, user, **kw):
            payload = user.split("agreeing and not):\n")[1].split("\n\nPRIOR")[0]
            captured.update(json.loads(payload))
            return self._response()

        monkeypatch.setattr(agents, "call_llm_json", spy)
        agents.run_mediation(make_occurring(), make_grid("main"),
                             make_grid("challenger"), [], 1)
        assert sum(len(row) for row in captured.values()) == NUM_CELLS

    def test_invented_refs_are_dropped(self, monkeypatch):
        resp = self._response(wholeness={"what_conflicts_imply": [
            {"statement": "unsupported finding", "refs": ["nowhere/atall"]}]})
        monkeypatch.setattr(agents, "call_llm_json", lambda *a, **k: resp)
        med = agents.run_mediation(make_occurring(), make_grid("main"),
                                   make_grid("challenger"), [], 1)
        assert med.wholeness.what_conflicts_imply == []


class TestNodeUpdate:
    def _nodes(self, changed_facet=None):
        return {"nodes": [{"node_id": str(f), "facet": str(f),
                           "content": "NEW content" if str(f) == changed_facet else f"content {f}",
                           "assumptions": [f"assumes {f}"],
                           "changed": str(f) == changed_facet, "change_note": ""}
                          for f in FACET_LIST], "unapplied": []}

    def test_no_signal_returns_same_object(self, monkeypatch):
        called = []
        monkeypatch.setattr(agents, "call_llm_json",
                            lambda *a, **k: called.append(1) or self._nodes())
        occ = make_occurring()
        med = make_mediation(1, wholeness=WholenessImplyingNodes())
        assert agents.run_node_update(occ, med) is occ
        assert not called, "an empty wholeness must not pay for a node-update call"

    def test_only_changed_nodes_get_cycle_bump(self, monkeypatch):
        monkeypatch.setattr(agents, "call_llm_json",
                            lambda *a, **k: self._nodes(changed_facet="agents"))
        out = agents.run_node_update(make_occurring(), make_mediation(2))
        bumped = {str(n.facet) for n in out.nodes if n.last_updated_cycle == 2}
        assert bumped == {"agents"}

    def test_duplicate_facets_rejected(self, monkeypatch):
        dupes = self._nodes()
        dupes["nodes"][1] = dict(dupes["nodes"][0])
        monkeypatch.setattr(agents, "call_llm_json", lambda *a, **k: dupes)
        with pytest.raises(ValueError, match="duplicate facet"):
            agents.run_node_update(make_occurring(), make_mediation(2))


class TestConstrainedGeneration:
    def test_aggregates_across_cycles_and_prepends_provenance(self, monkeypatch):
        seen = {}

        def spy(system, user, **kw):
            seen["system"], seen["user"] = system, user
            return "The document body."

        monkeypatch.setattr(agents, "call_llm", spy)
        history = [make_mediation(1), make_mediation(2)]
        doc = agents.run_constrained_generation(make_occurring(), history, "procedural rationalist")
        assert "ANALYSIS FRAME" in doc and "procedural rationalist" in doc
        assert "The document body." in doc

    def test_requires_at_least_one_cycle(self):
        with pytest.raises(ValueError, match="requires"):
            agents.run_constrained_generation(make_occurring(), [], "frame")

    def test_grids_never_reach_the_generator(self, monkeypatch):
        """Constrained generation receives distilled findings only."""
        seen = {}
        monkeypatch.setattr(agents, "call_llm",
                            lambda system, user, **kw: seen.update(u=user) or "doc")
        agents.run_constrained_generation(make_occurring(), [make_mediation(1)], "frame")
        assert "MAIN-SAYS" not in seen["u"] and "CHALLENGER-SAYS" not in seen["u"]
