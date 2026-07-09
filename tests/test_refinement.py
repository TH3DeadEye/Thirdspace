"""
Unit tests for call_llm_json_with_refinement's structural-only retry loop.

The old loop treated any populated reflection_flag as a validation failure
(run-june-10-1: 126/126 cells retried on their own honesty, 49% of the run's
calls) and its retry prompt dropped the original context entirely. These tests
pin the corrected contract:
  - reflection_flags never trigger a retry
  - retries fire only on structural defects (empty / too-short content)
  - the retry prompt contains the original user prompt AND the previous answer
  - _refinement_meta records attempts and reasons
"""
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT, _ROOT / "thirdspace"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import llm


GOOD_CONTENT = (
    "First sentence with real substance here. Second sentence with more. "
    "Third sentence closes the reasoning."
)


class RefinementLoopTests(unittest.TestCase):
    def setUp(self):
        self._orig = llm.call_llm_json
        self.calls: list[dict] = []

    def tearDown(self):
        llm.call_llm_json = self._orig

    def _stub(self, responses):
        """Queue canned responses; record every (system, user) pair."""
        queue = list(responses)

        def fake(system, user, **kwargs):
            self.calls.append({"system": system, "user": user})
            return queue.pop(0)

        llm.call_llm_json = fake

    def test_reflection_flags_do_not_trigger_retry(self):
        self._stub([
            {
                "content": GOOD_CONTENT,
                "reflection_flags": [
                    {"measure": "fairness", "issue": "weighted narrator more heavily"}
                ],
            }
        ])
        result = llm.call_llm_json_with_refinement("sys", "user prompt")
        self.assertEqual(len(self.calls), 1)  # honesty is not punished
        self.assertEqual(result["_refinement_meta"]["attempts"], 0)
        # Flags survive untouched for the cell builders to persist.
        self.assertEqual(result["reflection_flags"][0]["measure"], "fairness")

    def test_empty_content_retries_with_full_context(self):
        self._stub([
            {"content": "", "reflection_flags": []},
            {"content": GOOD_CONTENT, "reflection_flags": []},
        ])
        result = llm.call_llm_json_with_refinement("sys", "THE OCCURRING: full scenario")
        self.assertEqual(len(self.calls), 2)
        retry_user = self.calls[1]["user"]
        # Context-complete retry: original prompt + previous response + feedback.
        self.assertIn("THE OCCURRING: full scenario", retry_user)
        self.assertIn("YOUR PREVIOUS RESPONSE", retry_user)
        self.assertIn("VALIDATION FEEDBACK", retry_user)
        self.assertEqual(result["_refinement_meta"]["attempts"], 1)
        self.assertTrue(any("empty" in r for r in result["_refinement_meta"]["reasons"]))

    def test_short_content_retries_then_gives_up_at_cap(self):
        short = {"content": "One sentence only.", "reflection_flags": []}
        self._stub([short, short, short])
        result = llm.call_llm_json_with_refinement("sys", "user", max_refine_attempts=2)
        self.assertEqual(len(self.calls), 3)  # initial + 2 retries
        self.assertEqual(result["_refinement_meta"]["attempts"], 2)

    def test_non_dict_response_raises(self):
        self._stub([["not", "a", "dict"]])
        with self.assertRaises(ValueError):
            llm.call_llm_json_with_refinement("sys", "user")


if __name__ == "__main__":
    unittest.main()
