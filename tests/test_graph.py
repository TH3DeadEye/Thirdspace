"""
Integration tests for the LangGraph wiring in graph.py. All agent functions
are stubbed so the test exercises only the orchestration: routing, cycle
bookkeeping, integrate node, and the regeneration loop.
"""
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT, _ROOT / "thirdspace"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import graph as g
import metrics
from models import (
    SessionMemory, JudgeScore, ImplyingNodes, DetectedFrames, CultureAxis,
    TheOccurringUpdateResult, NodeUpdateDelta,
)
from tests.fixtures import make_occurring, make_grid, make_mediation, make_wholeness


def _stub_node_update(occurring, mediation):
    """Stand-in for run_node_update returning the new TheOccurringUpdateResult.

    Marks every node as actually changed so node_integrate exercises the
    working-perspective update path.
    """
    occ = make_occurring("updated")
    deltas = [
        NodeUpdateDelta(
            facet=n.facet, similarity=0.0, model_claimed_changed=True,
            actually_changed=True, content_kept=n.content,
            content_proposed=n.content, conflict=False,
        )
        for n in occ.nodes
    ]
    return TheOccurringUpdateResult(occurring=occ, deltas=deltas)


def _char_embed(text):
    """Deterministic offline embedding (per-character counts)."""
    alphabet = "abcdefghijklmnopqrstuvwxyz "
    low = text.lower()
    return [float(low.count(c)) for c in alphabet]


def _install_stubs(g_module, *, low_score: bool = False, max_loop_cycles: int = 2):
    """Replace every agent function bound in graph.py's namespace with a stub."""
    # Only exercised when auto_detect_frames=True (passthrough otherwise), but
    # bound here so the auto-detect test has a deterministic detector.
    g_module.run_frame_detection = lambda raw_scenario, scenario_id: DetectedFrames(
        main=[CultureAxis(dimension="power_distance", value="low")],
        challenger=[CultureAxis(dimension="power_distance", value="high")],
        rationale="stub detection",
    )
    g_module.run_reformulation = lambda raw_scenario, scenario_id: make_occurring(scenario_id)
    g_module.run_critical_thinking = lambda occurring, cultural_frame, cycle, prior_mediation=None: make_grid("main", cycle)
    g_module.run_challenger = lambda **kw: make_grid("challenger", kw["cycle"])
    g_module.run_mediation = lambda **kw: make_mediation(
        cycle=kw["cycle"],
        # Force convergence on the last allowed cycle so the loop terminates.
        converged=(kw["cycle"] >= max_loop_cycles - 1),
    )
    g_module.run_node_update = _stub_node_update
    g_module.run_cycle_judge = lambda mediation, cycle: JudgeScore(
        cycle=cycle, empathy=5, consensus=5, escalation_risk=2,
        cultural_insight=5, intervention_fidelity=5, reasoning="cycle ok",
    )
    output_empathy = 1 if low_score else 5
    output_insight = 1 if low_score else 5
    g_module.run_judge = lambda final_output, mediation_history, cycle: JudgeScore(
        cycle=cycle, empathy=output_empathy, consensus=3, escalation_risk=3,
        cultural_insight=output_insight, intervention_fidelity=5,
        reasoning="output judge reasoning",
    )
    g_module.run_constrained_generation = (
        lambda occurring, mediation_history, mediator_frame=None, evaluation_feedback=None:
            f"DOC(feedback={evaluation_feedback!r})"
    )


def _mem(max_cycles: int = 2) -> SessionMemory:
    return SessionMemory(
        scenario_id="t1",
        main_cultural_frame="A",
        challenger_cultural_frame="B",
        occurring=make_occurring("t1"),
        max_cycles=max_cycles,
        # These orchestration tests exercise routing, not the metric. Pin to
        # jaccard so the convergence guard stays offline (the embedding metric
        # has its own dedicated test with a stubbed embedder).
        convergence_metric="jaccard",
    )


class TestGraphHappyPath(unittest.TestCase):
    def test_no_regen_when_scores_above_threshold(self):
        _install_stubs(g, low_score=False, max_loop_cycles=2)
        compiled = g.build_graph()
        out = compiled.invoke({"memory": _mem(max_cycles=2)})["memory"]
        self.assertEqual(out.regen_attempts, 0)
        self.assertIn("feedback=None", out.final_output)
        # mediation_history should contain one entry per cycle that ran.
        self.assertGreaterEqual(len(out.mediation_history), 1)

    def test_final_judge_cycle_matches_last_mediation_cycle(self):
        """Regression test for the off-by-one fix in node_judge."""
        _install_stubs(g, low_score=False, max_loop_cycles=2)
        compiled = g.build_graph()
        out = compiled.invoke({"memory": _mem(max_cycles=2)})["memory"]
        # The last judge score is the post-output judge; its cycle field must
        # equal the cycle of the mediation that produced the output, not
        # current_cycle (which was already bumped).
        last_mediation_cycle = out.mediation_history[-1].cycle
        output_judges = [s for s in out.judge_scores if s.reasoning == "output judge reasoning"]
        self.assertTrue(output_judges, "expected at least one output judge score")
        self.assertEqual(output_judges[-1].cycle, last_mediation_cycle)


class TestGraphRegenLoop(unittest.TestCase):
    def test_low_score_triggers_one_regen(self):
        _install_stubs(g, low_score=True, max_loop_cycles=1)
        compiled = g.build_graph()
        out = compiled.invoke({"memory": _mem(max_cycles=1)})["memory"]
        self.assertEqual(out.regen_attempts, 1)
        # On the regen pass, feedback should be the output-judge reasoning.
        self.assertIn("output judge reasoning", out.final_output)

    def test_regen_caps_at_max_regen(self):
        _install_stubs(g, low_score=True, max_loop_cycles=1)
        compiled = g.build_graph()
        mem = _mem(max_cycles=1)
        mem.max_regen = 2
        out = compiled.invoke({"memory": mem})["memory"]
        self.assertEqual(out.regen_attempts, 2)

    def test_regen_feedback_uses_output_judge_not_cycle_judge(self):
        """Bug 4 regression: node_generate reads output_judge_scores[-1],
        not the most recent entry across all judge lists. If we ran cycles
        whose cycle-judge reasoning differs from the output judge, the regen
        feedback must come from the OUTPUT judge."""
        _install_stubs(g, low_score=True, max_loop_cycles=2)
        compiled = g.build_graph()
        out = compiled.invoke({"memory": _mem(max_cycles=2)})["memory"]
        # cycle_judge_scores should have at least one entry (from the looped cycle).
        self.assertGreaterEqual(len(out.cycle_judge_scores), 1)
        self.assertTrue(all(s.reasoning == "cycle ok" for s in out.cycle_judge_scores))
        # output_judge_scores: original generation + 1 regen attempt = 2 entries.
        self.assertEqual(len(out.output_judge_scores), 2)
        # Final output's feedback string is the OUTPUT judge reasoning.
        self.assertIn("output judge reasoning", out.final_output)
        # First generation had no feedback yet.
        # Both judges scored the same; just confirm regen counter and feedback origin.
        self.assertEqual(out.regen_attempts, 1)


class TestEvaluateCycleSkippedOnTerminal(unittest.TestCase):
    """Bug 3 regression: evaluate_cycle should NOT run on the final cycle that
    triggers termination. Otherwise we burn judge calls on a discarded cycle."""

    def test_no_cycle_judge_when_converged_on_first_cycle(self):
        # With max_loop_cycles=1, the very first mediation converges and the
        # graph goes straight to generate without ever calling evaluate_cycle.
        _install_stubs(g, low_score=False, max_loop_cycles=1)
        compiled = g.build_graph()
        out = compiled.invoke({"memory": _mem(max_cycles=1)})["memory"]
        # current_cycle bumped to 1 after one mediation.
        self.assertEqual(out.current_cycle, 1)
        self.assertEqual(len(out.cycle_judge_scores), 0)
        # Output judge still ran.
        self.assertEqual(len(out.output_judge_scores), 1)

    def test_cycle_judge_count_equals_completed_cycles_minus_terminal(self):
        # With max_loop_cycles=2: cycle 0 loops (evaluate_cycle fires once),
        # cycle 1 terminates (evaluate_cycle skipped).
        _install_stubs(g, low_score=False, max_loop_cycles=2)
        compiled = g.build_graph()
        out = compiled.invoke({"memory": _mem(max_cycles=2)})["memory"]
        self.assertEqual(out.current_cycle, 2)
        self.assertEqual(len(out.cycle_judge_scores), 1)


class TestConsensusPlateau(unittest.TestCase):
    """Convergence plateau is a secondary termination signal: two consecutive
    cycle-judge scores at/above the threshold should terminate the loop even
    if the mediator did not flag converged=True."""

    def _install_never_converged_stubs(self, g_module):
        """Stubs where mediation never self-declares converged. The plateau
        check is the only way the loop can terminate before max_cycles."""
        g_module.run_reformulation = lambda raw_scenario, scenario_id: make_occurring(scenario_id)
        g_module.run_critical_thinking = lambda occurring, cultural_frame, cycle, prior_mediation=None: make_grid("main", cycle)
        g_module.run_challenger = lambda **kw: make_grid("challenger", kw["cycle"])
        g_module.run_mediation = lambda **kw: make_mediation(cycle=kw["cycle"], converged=False)
        g_module.run_node_update = _stub_node_update
        # Cycle judge: always returns consensus=5 so two in a row trigger plateau.
        g_module.run_cycle_judge = lambda mediation, cycle: JudgeScore(
            cycle=cycle, empathy=5, consensus=5, escalation_risk=2,
            cultural_insight=5, intervention_fidelity=5, reasoning="cycle ok",
        )
        g_module.run_judge = lambda final_output, mediation_history, cycle: JudgeScore(
            cycle=cycle, empathy=5, consensus=5, escalation_risk=2,
            cultural_insight=5, intervention_fidelity=5, reasoning="output judge reasoning",
        )
        g_module.run_constrained_generation = (
            lambda occurring, mediation_history, mediator_frame=None, evaluation_feedback=None: "DOC"
        )

    def test_plateau_triggers_termination_before_max_cycles(self):
        self._install_never_converged_stubs(g)
        compiled = g.build_graph()
        # max_cycles=5 so plateau is the actual termination cause.
        out = compiled.invoke({"memory": _mem(max_cycles=5)})["memory"]
        # Cycle 0 -> evaluate (score 1) -> integrate -> cycle 1 -> evaluate (score 2)
        # Now two consecutive consensus=5 scores, plateau triggers, go to generate.
        self.assertEqual(len(out.cycle_judge_scores), 2)
        # The loop terminated well before max_cycles=5.
        self.assertLess(out.current_cycle, 5)
        self.assertEqual(out.current_cycle, 2)


class TestBackwardsCompatibleJudgeScores(unittest.TestCase):
    """The computed `judge_scores` property must return the concatenation so
    callers that don't know about the split (persistence, print loops) still work."""

    def test_combined_view(self):
        _install_stubs(g, low_score=False, max_loop_cycles=2)
        compiled = g.build_graph()
        out = compiled.invoke({"memory": _mem(max_cycles=2)})["memory"]
        combined = out.judge_scores
        self.assertEqual(
            len(combined), len(out.cycle_judge_scores) + len(out.output_judge_scores)
        )


class TestEmbeddingConvergence(unittest.TestCase):
    """With convergence_metric='embedding', a premature LLM convergence whose
    findings still differ semantically from the prior cycle must NOT terminate
    the loop — the embedding guard keeps it going."""

    def _med(self, cycle, converged, implies):
        return ImplyingNodes(
            cycle=cycle, agreements=[], conflicts=[],
            what_conflicts_imply=implies, updated_assumptions=[],
            wholeness=make_wholeness(reformulated=None), converged=converged,
        )

    def _install(self, g_module):
        g_module.run_reformulation = lambda raw_scenario, scenario_id: make_occurring(scenario_id)
        g_module.run_critical_thinking = lambda occurring, cultural_frame, cycle, prior_mediation=None: make_grid("main", cycle)
        g_module.run_challenger = lambda **kw: make_grid("challenger", kw["cycle"])
        # Highly different per-cycle findings => low embedding cosine between
        # consecutive cycles. The mediator self-declares converged from cycle 1.
        blocks = {0: ["aaaa aaaa aaaa"], 1: ["zzzz zzzz zzzz"], 2: ["mmmm mmmm mmmm"]}
        g_module.run_mediation = lambda **kw: self._med(
            kw["cycle"], kw["cycle"] >= 1, blocks[kw["cycle"]]
        )
        # Consensus BELOW the plateau threshold (4), so the plateau signal never
        # fires and the embedding guard is the only loop control.
        g_module.run_cycle_judge = lambda mediation, cycle: JudgeScore(
            cycle=cycle, empathy=5, consensus=2, escalation_risk=2,
            cultural_insight=5, intervention_fidelity=5, reasoning="cycle ok",
        )
        g_module.run_node_update = _stub_node_update
        g_module.run_judge = lambda final_output, mediation_history, cycle: JudgeScore(
            cycle=cycle, empathy=5, consensus=5, escalation_risk=2,
            cultural_insight=5, intervention_fidelity=5, reasoning="output judge",
        )
        g_module.run_constrained_generation = (
            lambda occurring, mediation_history, mediator_frame=None, evaluation_feedback=None: "DOC"
        )

    def test_low_embedding_similarity_overrides_premature_convergence(self):
        orig = metrics.embed_text
        metrics.embed_text = _char_embed
        self._install(g)
        try:
            compiled = g.build_graph()
            mem = SessionMemory(
                scenario_id="emb", main_cultural_frame="A",
                challenger_cultural_frame="B",
                occurring=make_occurring("emb"), max_cycles=3,
                convergence_metric="embedding",
            )
            out = compiled.invoke({"memory": mem})["memory"]
        finally:
            metrics.embed_text = orig
        # The mediator declared converged at cycle 1, but cycle 0 vs 1 findings
        # are semantically far apart, so the loop continued to max_cycles=3.
        # (Under jaccard, empty conflict sets would have stopped it at cycle 1.)
        self.assertEqual(len(out.mediation_history), 3)
        self.assertEqual(out.final_output, "DOC")


class TestAutoDetectFrames(unittest.TestCase):
    """Phase 0 entry node fills both frames from the scenario when
    auto_detect_frames=True, and is a passthrough otherwise."""

    def test_detected_frames_populate_before_pipeline(self):
        _install_stubs(g, low_score=False, max_loop_cycles=1)
        compiled = g.build_graph()
        mem = SessionMemory(
            scenario_id="auto",
            main_cultural_frame="", challenger_cultural_frame="",
            occurring=make_occurring("auto"),
            max_cycles=1, convergence_metric="jaccard",
            auto_detect_frames=True,
        )
        out = compiled.invoke({"memory": mem})["memory"]
        # Detection ran: placeholder frames were overwritten and recorded.
        self.assertIsNotNone(out.frame_detection)
        self.assertEqual([a.dimension for a in out.main_cultural_frame], ["power_distance"])
        self.assertEqual(out.main_cultural_frame[0].value, "low")
        self.assertEqual(out.challenger_cultural_frame[0].value, "high")
        # challenger_identity set (structured) from the detected challenger frame.
        self.assertEqual(out.challenger_identity.axes[0].value, "high")

    def test_passthrough_leaves_explicit_frames_untouched(self):
        _install_stubs(g, low_score=False, max_loop_cycles=1)
        compiled = g.build_graph()
        mem = SessionMemory(
            scenario_id="explicit",
            main_cultural_frame="Frame A", challenger_cultural_frame="Frame B",
            occurring=make_occurring("explicit"),
            max_cycles=1, convergence_metric="jaccard",
            # auto_detect_frames defaults False -> node is a no-op.
        )
        out = compiled.invoke({"memory": mem})["memory"]
        self.assertIsNone(out.frame_detection)
        self.assertEqual(out.main_cultural_frame, "Frame A")
        self.assertEqual(out.challenger_cultural_frame, "Frame B")


class TestMediatorFrameWiring(unittest.TestCase):
    """node_mediation and node_generate must thread mem.mediator_frame into the
    mediation and constrained-generation calls."""

    def test_mediator_frame_reaches_both_agent_calls(self):
        _install_stubs(g, low_score=False, max_loop_cycles=1)
        captured = {}
        base_med = g.run_mediation
        base_gen = g.run_constrained_generation

        def cap_med(**kw):
            captured["mediation"] = kw.get("mediator_frame")
            return base_med(**kw)

        def cap_gen(occurring, mediation_history, mediator_frame=None, evaluation_feedback=None):
            captured["generation"] = mediator_frame
            return base_gen(
                occurring=occurring, mediation_history=mediation_history,
                mediator_frame=mediator_frame, evaluation_feedback=evaluation_feedback,
            )

        g.run_mediation = cap_med
        g.run_constrained_generation = cap_gen
        try:
            compiled = g.build_graph()
            mem = _mem(max_cycles=1)
            mem.mediator_frame = "utilitarian arbiter"
            compiled.invoke({"memory": mem})
        finally:
            g.run_mediation = base_med
            g.run_constrained_generation = base_gen

        self.assertEqual(captured["mediation"], "utilitarian arbiter")
        self.assertEqual(captured["generation"], "utilitarian arbiter")


if __name__ == "__main__":
    unittest.main()
