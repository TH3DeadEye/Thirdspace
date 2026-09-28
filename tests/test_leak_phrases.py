"""The generated document must never carry the vocabulary that produced it.

A real run once emitted "CRITICAL PROHIBITION: Do NOT offer dialogue..." verbatim
into the finished analysis, alongside internal headers and seven uses of the word
"severity" — the two people in the conflict were handed the rubric. These tests
pin the fix at the source: the intervention prose and the generation system
prompt must both be free of rubric markers.

models.LEAK_PHRASES is the shared list; the eval harness greps produced documents
against the same constant, and verify_session asserts on it per run.
"""
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT, _ROOT / "thirdspace"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from models import (
    CONFLICT_TYPE_DEFS, LEAK_PHRASES, ConflictTypeSummary, ConflictType,
)
from utils import prompts
from tests.fixtures import make_conflict

# Markers from the old labelled-block style. These are what leaked verbatim.
RUBRIC_MARKERS = [
    "INTERVENTION CLASS", "ABSOLUTE PROHIBITION", "CRITICAL PROHIBITION",
    "FALSE RESOLUTION TRAP", "PROHIBITED:",
]


def _full_summary() -> ConflictTypeSummary:
    conflicts = [
        make_conflict("agents", "value", "power_related", "structural_change", 5),
        make_conflict("actions", "constraint", "power_related", "structural_change", 4),
        make_conflict("context", "interpretation", "interpretive", "conversational", 2),
        make_conflict("representations", "assumption", "normative", "facilitated", 3),
    ]
    return ConflictTypeSummary.from_conflicts(conflicts)


class TestInterventionTextHygiene:
    def test_intervention_texts_contain_no_leak_phrases(self):
        for ctype, defn in CONFLICT_TYPE_DEFS.items():
            low = defn.intervention_text.lower()
            for phrase in LEAK_PHRASES:
                assert phrase not in low, (
                    f"intervention_text for {ctype!r} contains banned phrase {phrase!r}")

    def test_intervention_texts_are_prose_not_labeled_blocks(self):
        for ctype, defn in CONFLICT_TYPE_DEFS.items():
            first = defn.intervention_text.strip().splitlines()[0]
            assert not re.match(r"^[A-Z][A-Z \-]{6,}:", first), (
                f"intervention_text for {ctype!r} opens with an ALL-CAPS label line; "
                f"the generator parrots these into the document")

    def test_every_type_has_intervention_prose(self):
        for ctype in ConflictType:
            assert CONFLICT_TYPE_DEFS[ctype].intervention_text.strip()


class TestGenerationSystemPromptHygiene:
    def test_full_prompt_contains_no_rubric_markers(self):
        system = prompts.build_constrained_generation_system(_full_summary(), 900)
        for marker in RUBRIC_MARKERS:
            assert marker not in system
        low = system.lower()
        for phrase in LEAK_PHRASES:
            assert phrase not in low, f"generation prompt contains {phrase!r}"

    def test_empty_summary_prompt_contains_no_rubric_markers(self):
        system = prompts.build_constrained_generation_system(None, 900)
        for marker in RUBRIC_MARKERS:
            assert marker not in system

    def test_prompt_still_carries_the_substance(self):
        """Hygiene must not have hollowed the guidance out."""
        system = prompts.build_constrained_generation_system(_full_summary(), 900)
        assert "WHAT THIS SITUATION NEEDS" in system
        assert "900" in system

    def test_multi_type_prompt_carries_each_present_type(self):
        """Only the dominant type's remedy used to reach the prompt, so five of
        the six interventions were unreachable in practice."""
        system = prompts.build_constrained_generation_system(
            None, 900, weighted_types=["power_related", "interpretive"])
        assert "structural asymmetry" in system
        assert "shared framework" in system
        for marker in RUBRIC_MARKERS:
            assert marker not in system
