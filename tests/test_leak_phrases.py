"""
Prompt-hygiene regression tests for rubric leakage (TODO Step 2).

run-june-10-1's final document contained internal rubric text verbatim:
"CRITICAL PROHIBITION: Do NOT offer dialogue...", "This is the false
resolution trap.", and headers like "FOR STRUCTURAL CONFLICTS (16 conflicts,
50% of total)". The generator parrots labeled rubric blocks, so the fix is to
keep them out of the prompts entirely. models.LEAK_PHRASES is the shared
banned-vocabulary list; the eval harness greps generated documents against it.
"""
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT, _ROOT / "thirdspace"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from models import (
    INTERVENTION_CLASS, LEAK_PHRASES, ConflictTypeSummary,
)
from utils.prompts import build_constrained_generation_system
from tests.test_graph import _conflict


# Rubric-block markers that must never appear in any prompt sent to the
# generator. (A subset of LEAK_PHRASES: the full list includes words like
# "severity" that the system prompt legitimately mentions when instructing the
# model NOT to use them.)
RUBRIC_MARKERS = [
    "intervention class",
    "absolute prohibition",
    "critical prohibition",
    "false resolution trap",
    "prohibited",
    "══",
    "──",
]


def _full_summary() -> ConflictTypeSummary:
    """A summary with every type present, structural fraction >= 0.30, and
    unresolvable conflicts, so every conditional prompt block is exercised."""
    conflicts = [
        _conflict("individuals", "assumptions", "structural"),
        _conflict("individuals", "evidence", "structural"),
        _conflict("concepts_theories_rules", "concepts", "structural"),
        _conflict("process", "assumptions", "normative"),
        _conflict("process", "evidence", "semantic"),
        _conflict("inferences", "assumptions", "attributional"),
        _conflict("inferences", "evidence", "factual"),
    ]
    # Mark one unresolvable so the unresolvable_note block renders too.
    unresolvable = _conflict("evidence_and_data", "evidence", "factual")
    unresolvable = unresolvable.model_copy(update={"resolvability": "unresolvable"})
    return ConflictTypeSummary.from_conflicts(conflicts + [unresolvable])


class InterventionTextHygiene(unittest.TestCase):
    def test_intervention_texts_contain_no_leak_phrases(self):
        for ctype, text in INTERVENTION_CLASS.items():
            low = text.lower()
            for phrase in LEAK_PHRASES:
                self.assertNotIn(
                    phrase, low,
                    f"intervention_text for '{ctype}' contains banned phrase {phrase!r}",
                )

    def test_intervention_texts_are_prose_not_labeled_blocks(self):
        for ctype, text in INTERVENTION_CLASS.items():
            self.assertFalse(
                text.strip().split("\n")[0].isupper(),
                f"intervention_text for '{ctype}' starts with an ALL-CAPS label line",
            )


class GenerationSystemPromptHygiene(unittest.TestCase):
    def test_full_prompt_contains_no_rubric_markers(self):
        prompt = build_constrained_generation_system(_full_summary())
        low = prompt.lower()
        for marker in RUBRIC_MARKERS:
            self.assertNotIn(marker, low, f"generation prompt contains {marker!r}")

    def test_empty_summary_prompt_contains_no_rubric_markers(self):
        for summary in (None,):
            prompt = build_constrained_generation_system(summary)
            low = prompt.lower()
            for marker in RUBRIC_MARKERS:
                self.assertNotIn(marker, low, f"generation prompt contains {marker!r}")

    def test_prompt_still_carries_the_substance(self):
        """De-fanging must not delete the guidance itself: the structural
        anti-dialogue rule and per-type paragraphs must survive as prose."""
        prompt = build_constrained_generation_system(_full_summary())
        self.assertIn("Do not present dialogue as", prompt)
        self.assertIn("structural conflict", prompt)
        self.assertIn("HOW TO MOVE FORWARD", prompt)


if __name__ == "__main__":
    unittest.main()
