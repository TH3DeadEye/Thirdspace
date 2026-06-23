"""
Unit tests for Phase 0 cultural frame detection (agents.run_frame_detection and
its _validate_axes / _axes_identical helpers).

Strategy: stub agents.call_llm_json so no Bedrock traffic happens. A small
sequence stub lets tests assert the corrective-retry path (two calls).
"""
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT, _ROOT / "thirdspace"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import agents
from models import CultureAxis, DetectedFrames


class _SeqStub:
    """Returns queued responses in order; repeats the last once exhausted.
    Records how many times it was called so tests can assert the retry path."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, **kwargs):
        i = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return self.responses[i]


def _good_response():
    return {
        "main": [
            {"dimension": "power_distance", "value": "low"},
            {"dimension": "individualism", "value": "individualist"},
            {"dimension": "uncertainty_avoidance", "value": "low"},
            {"dimension": "indulgence", "value": "indulgent"},
        ],
        "challenger": [
            {"dimension": "power_distance", "value": "high"},
            {"dimension": "individualism", "value": "collectivist"},
            {"dimension": "uncertainty_avoidance", "value": "high"},
            {"dimension": "indulgence", "value": "restrained"},
        ],
        "rationale": "Behaviors in the scenario imply contrasting frames.",
    }


class TestValidateAxes(unittest.TestCase):
    def test_keeps_valid_and_restores_canonical_value(self):
        # Mixed case + whitespace must normalize to the canonical menu form.
        axes = agents._validate_axes(
            [{"dimension": "Power_Distance", "value": " HIGH "}], where="t"
        )
        self.assertEqual(len(axes), 1)
        self.assertEqual(axes[0].dimension, "power_distance")
        self.assertEqual(axes[0].value, "high")

    def test_drops_unknown_dimension(self):
        # "context" is not in the Hofstede-6 menu.
        self.assertEqual(
            agents._validate_axes([{"dimension": "context", "value": "high"}], where="t"),
            [],
        )

    def test_drops_illegal_value(self):
        self.assertEqual(
            agents._validate_axes(
                [{"dimension": "power_distance", "value": "medium"}], where="t"
            ),
            [],
        )

    def test_dedupes_dimension_first_valid_wins(self):
        axes = agents._validate_axes(
            [
                {"dimension": "power_distance", "value": "high"},
                {"dimension": "power_distance", "value": "low"},
            ],
            where="t",
        )
        self.assertEqual(len(axes), 1)
        self.assertEqual(axes[0].value, "high")

    def test_skips_non_dict_entries(self):
        self.assertEqual(agents._validate_axes(["nope", None, 3], where="t"), [])


class TestAxesIdentical(unittest.TestCase):
    def test_same_dims_same_values_is_identical(self):
        a = [CultureAxis(dimension="power_distance", value="high")]
        b = [CultureAxis(dimension="power_distance", value="high")]
        self.assertTrue(agents._axes_identical(a, b))

    def test_contrasting_values_not_identical(self):
        a = [CultureAxis(dimension="power_distance", value="high")]
        b = [CultureAxis(dimension="power_distance", value="low")]
        self.assertFalse(agents._axes_identical(a, b))


class TestRunFrameDetection(unittest.TestCase):
    def test_happy_path_single_call(self):
        stub = _SeqStub([_good_response()])
        agents.call_llm_json = stub
        out = agents.run_frame_detection("scn", "id1")
        self.assertIsInstance(out, DetectedFrames)
        self.assertEqual(stub.calls, 1)
        self.assertEqual(len(out.main), 4)
        self.assertEqual(len(out.challenger), 4)
        self.assertFalse(agents._axes_identical(out.main, out.challenger))

    def test_drops_invalid_axes_but_keeps_valid(self):
        resp = _good_response()
        resp["main"].append({"dimension": "bogus", "value": "x"})          # unknown dim
        resp["main"].append({"dimension": "power_distance", "value": "mid"})  # illegal value
        stub = _SeqStub([resp])
        agents.call_llm_json = stub
        out = agents.run_frame_detection("scn", "id1")
        dims = [a.dimension for a in out.main]
        self.assertNotIn("bogus", dims)
        # The original valid power_distance:low survives; the illegal dup is gone.
        self.assertEqual(dims.count("power_distance"), 1)
        self.assertEqual(stub.calls, 1)

    def test_retry_on_degenerate_then_succeeds(self):
        degenerate = {
            "main": [{"dimension": "power_distance", "value": "high"}],
            "challenger": [{"dimension": "power_distance", "value": "high"}],
            "rationale": "identical, no contrast",
        }
        stub = _SeqStub([degenerate, _good_response()])
        agents.call_llm_json = stub
        out = agents.run_frame_detection("scn", "id1")
        self.assertEqual(stub.calls, 2)  # one corrective retry
        self.assertFalse(agents._axes_identical(out.main, out.challenger))

    def test_empty_side_falls_back_to_neutral(self):
        # main has only invalid axes -> empty after validation on BOTH attempts.
        empty_main = {
            "main": [{"dimension": "bogus", "value": "x"}],
            "challenger": [{"dimension": "power_distance", "value": "high"}],
            "rationale": "x",
        }
        stub = _SeqStub([empty_main, empty_main])
        agents.call_llm_json = stub
        out = agents.run_frame_detection("scn", "id1")
        self.assertEqual(out.main, [])          # neutral mode for main downstream
        self.assertEqual(len(out.challenger), 1)
        self.assertEqual(stub.calls, 2)         # retried once before giving up


if __name__ == "__main__":
    unittest.main()
