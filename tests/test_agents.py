"""
Unit tests for each phase function in agents.py.

Strategy: stub the LLM callers (`call_llm`, `call_llm_json`,
`call_llm_json_with_refinement`) so no Bedrock traffic happens. Each test
focuses on the shape, control flow, and data plumbing of a single function.
"""
import sys
import unittest
from pathlib import Path

# Make thirdspace/ and tests/ importable regardless of how tests are launched.
_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT, _ROOT / "thirdspace"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import agents
from models import (
    NodeReasoningGrid,
    JudgeScore,
    FACET_LIST, DIMENSION_LIST, NUM_FACETS, NUM_CELLS,
)
from tests.fixtures import (
    make_occurring, make_grid, make_mediation, make_wholeness,
)


def _ok_cell_response():
    """A reflection-flag-free response so refinement loop exits on first try."""
    return {
        "content": "Sentence one. Sentence two. Sentence three.",
        "reflection_flags": [],
    }


class TestReformulation(unittest.TestCase):
    def test_happy_path_returns_six_nodes(self):
        agents.call_llm_json = lambda **k: {
            "nodes": [
                {"node_id": f, "facet": f, "content": f"c {f}", "assumptions": ["a"]}
                for f in FACET_LIST
            ]
        }
        occ = agents.run_reformulation("scn", "id1")
        self.assertEqual(len(occ.nodes), NUM_FACETS)
        self.assertEqual([n.facet for n in occ.nodes], FACET_LIST)

    def test_wrong_node_count_raises_valueerror(self):
        agents.call_llm_json = lambda **k: {"nodes": []}
        with self.assertRaises(ValueError) as cm:
            agents.run_reformulation("scn", "id1")
        self.assertIn("expected 6 nodes", str(cm.exception))

    def test_unknown_facet_rejected(self):
        # 6 nodes total but one has a hallucinated facet name.
        bad_facets = FACET_LIST[:-1] + ["bogus_facet"]
        agents.call_llm_json = lambda **k: {
            "nodes": [
                {"node_id": f, "facet": f, "content": "c", "assumptions": ["a"]}
                for f in bad_facets
            ]
        }
        with self.assertRaises(ValueError) as cm:
            agents.run_reformulation("scn", "id1")
        self.assertIn("unknown facet", str(cm.exception))

    def test_empty_assumptions_rejected(self):
        agents.call_llm_json = lambda **k: {
            "nodes": [
                {"node_id": f, "facet": f, "content": "c", "assumptions": []}
                for f in FACET_LIST
            ]
        }
        with self.assertRaises(ValueError) as cm:
            agents.run_reformulation("scn", "id1")
        self.assertIn("at least one assumption", str(cm.exception))

    def test_returned_nodes_sorted_in_canonical_order(self):
        # Send the LLM stub in reversed order; expect canonical FACET_LIST order out.
        reversed_facets = list(reversed(FACET_LIST))
        agents.call_llm_json = lambda **k: {
            "nodes": [
                {"node_id": f, "facet": f, "content": "c", "assumptions": ["a"]}
                for f in reversed_facets
            ]
        }
        occ = agents.run_reformulation("scn", "id1")
        self.assertEqual([n.facet for n in occ.nodes], FACET_LIST)


class TestCriticalThinking(unittest.TestCase):
    def test_produces_42_cells_with_correct_keys(self):
        agents.call_llm_json_with_refinement = lambda **k: _ok_cell_response()
        grid = agents.run_critical_thinking(make_occurring(), "frame", cycle=0)
        self.assertIsInstance(grid, NodeReasoningGrid)
        self.assertEqual(grid.agent, "main")
        self.assertEqual(len(grid.cells), NUM_CELLS)
        keys = {(c.facet, c.dimension) for c in grid.cells}
        expected = {(f, d) for f in FACET_LIST for d in DIMENSION_LIST}
        self.assertEqual(keys, expected)
        # Cycle stamp propagates.
        self.assertTrue(all(c.cycle_last_updated == 0 for c in grid.cells))


class TestChallenger(unittest.TestCase):
    def test_independent_grid_has_no_response_type(self):
        # The challenger is blind to the main agent: it produces an independent
        # grid and does NOT classify challenge/reframe/confirm. response_type is
        # always None, even if the model erroneously emits one.
        agents.call_llm_json_with_refinement = lambda **k: {
            "content": "One. Two. Three.",
            "reflection_flags": [],
            "response_type": "challenge",
        }
        grid = agents.run_challenger(
            occurring=make_occurring(),
            challenger_frame="X",
            cycle=0,
            challenger_style="procedural",
        )
        self.assertEqual(grid.agent, "challenger")
        self.assertEqual(len(grid.cells), NUM_CELLS)
        self.assertTrue(all(c.response_type is None for c in grid.cells))


class TestMediation(unittest.TestCase):
    def test_parses_implying_nodes(self):
        agents.call_llm_json = lambda **k: {
            "cycle": 0,
            "agreements": ["a"],
            "conflicts": [{
                "facet": "individuals",
                "dimension": "objective",
                "response_type": "challenge",
                "main_content": "main says",
                "challenger_content": "challenger says",
                "implication": "This reveals that both parties operated under incompatible definitions of professional respect.",
                "conflict_type": "normative",
                "conflict_type_rationale": "Classified normative because both understand each other yet neither will change position.",
                "severity": 4,
                "resolvability": "facilitated",
            }],
            "what_conflicts_imply": ["pattern"],
            "updated_assumptions": ["individuals node: remove [X] -> replace with [Y]"],
            "wholeness": {
                "query_assumptions": ["qa"],
                "reformulated_query": "sharper",
                "new_objective": None,
                "updated_challenger_perspective": None,
                "objective_assumptions": [],
                "perspective_assumptions": [],
            },
            "converged": False,
        }
        med = agents.run_mediation(
            occurring=make_occurring(),
            main_grid=make_grid("main", 0),
            challenger_grid=make_grid("challenger", 0),
            all_mediation_history=[],
            cycle=0,
        )
        self.assertEqual(med.cycle, 0)
        self.assertEqual(len(med.conflicts), 1)
        self.assertEqual(med.conflicts[0].conflict_type, "normative")
        self.assertEqual(med.conflicts[0].resolvability, "facilitated")
        self.assertEqual(med.wholeness.reformulated_query, "sharper")


class TestNodeUpdate(unittest.TestCase):
    def test_only_changed_nodes_get_cycle_bump(self):
        # Stub: claim 'individuals' and 'process' changed, others unchanged.
        def stub(**_kw):
            return {
                "nodes": [
                    {"node_id": f, "facet": f, "content": f"new {f}" if f in {"individuals", "process"} else f"content for {f}",
                     "assumptions": ["a"], "changed": f in {"individuals", "process"}}
                    for f in FACET_LIST
                ]
            }
        agents.call_llm_json = stub
        occ = make_occurring()
        med = make_mediation(cycle=2)
        updated = agents.run_node_update(occ, med)
        stamps = {n.facet: n.last_updated_cycle for n in updated.nodes}
        self.assertEqual(stamps["individuals"], 2)
        self.assertEqual(stamps["process"], 2)
        self.assertEqual(stamps["interactional_event"], 0)
        self.assertEqual(stamps["inferences"], 0)

    def test_no_signal_returns_same_object(self):
        from models import ImplyingNodes
        empty_med = ImplyingNodes(
            cycle=1, agreements=[], conflicts=[], what_conflicts_imply=[],
            updated_assumptions=[],
            wholeness=make_wholeness(reformulated=None),
            converged=True,
        )
        # Even though wholeness has query_assumptions, set to empty:
        empty_med.wholeness.query_assumptions = []
        occ = make_occurring()
        out = agents.run_node_update(occ, empty_med)
        self.assertIs(out, occ)

    def test_duplicate_facets_rejected(self):
        # Six nodes but two share the same facet — should fail set-equality check.
        dup_facets = [FACET_LIST[0]] * 2 + FACET_LIST[1:-1]  # length 6, one duplicate
        agents.call_llm_json = lambda **k: {
            "nodes": [
                {"node_id": f, "facet": f, "content": f"new {f}",
                 "assumptions": ["a"], "changed": True}
                for f in dup_facets
            ]
        }
        with self.assertRaises(ValueError) as cm:
            agents.run_node_update(make_occurring(), make_mediation(cycle=1))
        self.assertIn("duplicate facet", str(cm.exception))


class TestRefinementTelemetry(unittest.TestCase):
    def test_first_pass_records_zero_attempts(self):
        agents.call_llm_json_with_refinement = lambda **k: {
            "content": "One. Two. Three.",
            "reflection_flags": [],
            "_refinement_meta": {"attempts": 0, "reasons": []},
        }
        grid = agents.run_critical_thinking(make_occurring(), "frame", cycle=0)
        self.assertTrue(all(c.refinement_attempts == 0 for c in grid.cells))
        self.assertTrue(all(c.refinement_reasons == [] for c in grid.cells))

    def test_retry_metadata_propagates_to_cell(self):
        agents.call_llm_json_with_refinement = lambda **k: {
            "content": "One. Two. Three.",
            "reflection_flags": [],
            "_refinement_meta": {
                "attempts": 2,
                "reasons": ["Self-evaluation failed [DEPTH]: foo",
                            "Content has only 1 sentence(s). Write at least 3 sentences."],
            },
        }
        grid = agents.run_critical_thinking(make_occurring(), "frame", cycle=3)
        sample = grid.cells[0]
        self.assertEqual(sample.refinement_attempts, 2)
        self.assertEqual(len(sample.refinement_reasons), 2)
        self.assertIn("DEPTH", sample.refinement_reasons[0])


class TestConstrainedGeneration(unittest.TestCase):
    def test_aggregates_implications_across_cycles_and_threads_feedback(self):
        captured = {}
        def fake_call_llm(system, user, **kw):
            captured["user"] = user
            return "DOC"
        agents.call_llm = fake_call_llm

        med0 = make_mediation(cycle=0)
        med1 = make_mediation(cycle=1, converged=True)
        # Different implication on the second cycle to verify aggregation.
        med1.what_conflicts_imply = ["second-cycle pattern"]

        out = agents.run_constrained_generation(
            occurring=make_occurring(),
            mediation_history=[med0, med1],
            evaluation_feedback="judge says fix tone",
        )
        self.assertEqual(out, "DOC")
        self.assertIn("pattern-level finding", captured["user"])  # from cycle 0
        self.assertIn("second-cycle pattern", captured["user"])   # from cycle 1
        self.assertIn("judge says fix tone", captured["user"])    # feedback


class TestJudges(unittest.TestCase):
    def test_run_judge_builds_score(self):
        agents.call_llm_json = lambda **k: {
            "reasoning": "good", "empathy": 4, "consensus": 3,
            "escalation_risk": 2, "cultural_insight": 5, "intervention_fidelity": 4,
        }
        score = agents.run_judge("FINAL DOC", [make_mediation(0)], cycle=0)
        self.assertIsInstance(score, JudgeScore)
        self.assertEqual(score.cycle, 0)
        self.assertEqual(score.empathy, 4)

    def test_run_cycle_judge_builds_score(self):
        agents.call_llm_json = lambda **k: {
            "reasoning": "ok", "empathy": 3, "consensus": 3,
            "escalation_risk": 3, "cultural_insight": 3, "intervention_fidelity": 3,
        }
        score = agents.run_cycle_judge(make_mediation(7), cycle=7)
        self.assertEqual(score.cycle, 7)


if __name__ == "__main__":
    unittest.main()
