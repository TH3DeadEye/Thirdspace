"""Graph wiring and the loop router.

The judge, regeneration and frame-detection suites that used to live here are
gone with the features they covered: the in-pipeline judges saturated at perfect
scores and were replaced by external classifiers, and frames are now read from
the dataset rather than inferred.

What remains is the part that decides how many cycles you pay for.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT, _ROOT / "thirdspace"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytest

import graph
from models import WholenessEntry, WholenessImplyingNodes, WorkingPerspective
from tests.fixtures import make_memory, make_mediation, make_conflict, make_grid


def _mem_at_cycle(cycle, cycles_done, **over):
    mem = make_memory(**over)
    mem.current_cycle = cycle
    mem.mediation_history = [make_mediation(c) for c in range(1, cycles_done + 1)]
    return mem


class TestFixedPolicy:
    """The default. Runs exactly max_cycles and does not route on convergence."""

    def test_loops_until_max_cycles(self):
        mem = _mem_at_cycle(1, 1, max_cycles=2)
        assert graph.should_loop({"memory": mem}) == "integrate"

    def test_terminates_at_max_cycles(self):
        mem = _mem_at_cycle(2, 2, max_cycles=2)
        assert graph.should_loop({"memory": mem}) == "generate"
        assert mem.convergence_reason == "max_cycles"

    def test_ignores_an_llm_convergence_declaration(self):
        """Under a fixed policy the mediator's flag is telemetry, not a decision."""
        mem = _mem_at_cycle(1, 1, max_cycles=3)
        mem.converged = True
        assert graph.should_loop({"memory": mem}) == "integrate"

    def test_skip_generation_routes_to_end(self):
        mem = _mem_at_cycle(2, 2, max_cycles=2, skip_generation=True)
        assert graph.should_loop({"memory": mem}) == "end"

    def test_stability_is_still_recorded(self, monkeypatch):
        """Fixed policy must not cost you the convergence data."""
        monkeypatch.setattr(graph, "convergence_similarity", lambda *a, **k: 0.91)
        mem = _mem_at_cycle(2, 2, max_cycles=3)
        graph.should_loop({"memory": mem})
        assert len(mem.stability_log) == 1
        assert mem.stability_log[0]["similarity"] == 0.91


class TestConvergencePolicy:
    def test_structural_convergence_overrides_llm_false(self, monkeypatch):
        """The mediator has withheld `converged` across cycles with identical
        conflict profiles, burning a full extra cycle re-deriving the same thing."""
        monkeypatch.setattr(graph, "convergence_similarity", lambda *a, **k: 0.99)
        conflicts = [make_conflict("agents", "value")]
        mem = make_memory(cycle_policy="convergence", max_cycles=5)
        mem.current_cycle = 2
        mem.mediation_history = [make_mediation(1, conflicts=conflicts),
                                 make_mediation(2, conflicts=conflicts)]
        assert graph.should_loop({"memory": mem}) == "generate"
        assert mem.convergence_reason == "structural_convergence"
        assert mem.converged is True

    def test_premature_llm_convergence_is_vetoed(self, monkeypatch):
        """The opposite failure: declared converged while content still moved."""
        monkeypatch.setattr(graph, "convergence_similarity", lambda *a, **k: 0.10)
        mem = make_memory(cycle_policy="convergence", max_cycles=5)
        mem.current_cycle = 2
        mem.converged = True
        mem.mediation_history = [make_mediation(1), make_mediation(2)]
        assert graph.should_loop({"memory": mem}) == "integrate"

    def test_shifting_type_distribution_blocks_termination(self, monkeypatch):
        monkeypatch.setattr(graph, "convergence_similarity", lambda *a, **k: 0.99)
        mem = make_memory(cycle_policy="convergence", max_cycles=5)
        mem.current_cycle = 2
        mem.mediation_history = [
            make_mediation(1, conflicts=[make_conflict("agents", "value", "normative",
                                                       "facilitated")]),
            make_mediation(2, conflicts=[
                make_conflict("agents", "value", "power_related", "structural_change"),
                make_conflict("actions", "value", "power_related", "structural_change"),
                make_conflict("context", "value", "power_related", "structural_change")]),
        ]
        assert graph.should_loop({"memory": mem}) == "integrate"

    def test_embedding_failure_never_manufactures_convergence(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("throttled")

        monkeypatch.setattr(graph, "convergence_similarity", boom)
        conflicts = [make_conflict("agents", "value")]
        mem = make_memory(cycle_policy="convergence", max_cycles=5)
        mem.current_cycle = 2
        mem.mediation_history = [make_mediation(1, conflicts=conflicts),
                                 make_mediation(2, conflicts=conflicts)]
        assert graph.should_loop({"memory": mem}) == "integrate"
        assert mem.stability_log[-1]["similarity"] is None


class TestIntegrate:
    def test_working_perspective_updates_without_content_change(self, monkeypatch):
        """Gating on `changed` (node CONTENT) meant a cycle that revised only
        assumptions left the lens stale even though mediation asked for a new one."""
        monkeypatch.setattr(graph, "run_node_update", lambda occurring, mediation: occurring)
        mem = make_memory()
        mem.mediation_history = [make_mediation(1, wholeness=WholenessImplyingNodes(
            updated_perspectives=[WholenessEntry(statement="read refusal as role maintenance"),
                                  WholenessEntry(statement="a second lens")]))]
        graph.node_integrate({"memory": mem})
        assert len(mem.challenger_working_perspective.entries) == 2
        assert mem.current_cycle == 2

    def test_identity_is_never_overwritten(self, monkeypatch):
        monkeypatch.setattr(graph, "run_node_update", lambda occurring, mediation: occurring)
        mem = make_memory()
        before = mem.challenger_identity.render()
        mem.mediation_history = [make_mediation(1, wholeness=WholenessImplyingNodes(
            updated_perspectives=[WholenessEntry(statement="a new lens")]))]
        graph.node_integrate({"memory": mem})
        assert mem.challenger_identity.render() == before


class TestGraphShape:
    def test_compiles(self):
        assert graph.build_graph() is not None

    def test_no_judge_or_frame_detection_nodes_remain(self):
        for gone in ("node_judge", "node_detect_frames", "node_evaluate_cycle",
                     "should_regenerate"):
            assert not hasattr(graph, gone), f"{gone} should have been removed"
