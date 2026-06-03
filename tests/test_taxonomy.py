"""
Tests for the e5 conflict taxonomy (models.py).

These are pure-model tests — no Bedrock, no tokens. They lock in:
  - CONFLICT_TYPE has all five lowercase members (regression for the
    recurring "broken Literal" revert that silently zeroed all conflicts)
  - the false-resolution trap is a runtime ValidationError
  - the implication / rationale field validators
  - ConflictTypeSummary auto-computation from ImplyingNodes
"""
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

# Make thirdspace/ importable regardless of how tests are launched.
_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT, _ROOT / "thirdspace"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from models import (
    CellConflict, ImplyingNodes, WholenessImplyingNodes,
    INTERVENTION_CLASS,
)

# A valid resolvability for each conflict type (avoids the forbidden combos).
_VALID_RESOLVABILITY = {
    "semantic": "conversational",
    "attributional": "facilitated",
    "normative": "facilitated",
    "structural": "structural_change",
    "factual": "external_standard",
}

_GOOD_IMPLICATION = (
    "This reveals that both parties operated under incompatible definitions "
    "of professional respect, making the divergence invisible to both."
)
_GOOD_RATIONALE = (
    "Classified by the diagnostic procedure because the cited evidence in "
    "the cell pair triggered that step."
)


def _make_conflict(
    conflict_type: str = "semantic",
    resolvability: str | None = None,
    severity: int = 3,
    implication: str = _GOOD_IMPLICATION,
    rationale: str = _GOOD_RATIONALE,
) -> CellConflict:
    if resolvability is None:
        resolvability = _VALID_RESOLVABILITY[conflict_type]
    return CellConflict(
        facet="individuals",
        dimension="assumptions",
        main_summary="main agent reasoning",
        challenger_summary="challenger reasoning",
        challenger_response_type="challenge",
        implication=implication,
        conflict_type=conflict_type,
        conflict_type_rationale=rationale,
        severity=severity,
        resolvability=resolvability,
    )


def _implying(conflicts: list[CellConflict]) -> ImplyingNodes:
    return ImplyingNodes(
        cycle=1, agreements=[], conflicts=conflicts,
        what_conflicts_imply=[], updated_assumptions=[],
        wholeness=WholenessImplyingNodes(
            query_assumptions=[], reformulated_query=None, new_objective=None,
            updated_challenger_perspective=None, objective_assumptions=[],
            perspective_assumptions=[],
        ),
        converged=False,
    )


# ---------------------------------------------------------------------------
# 0. Regression: all five types must be valid Literal members.
#    This is the test that catches the "broken CONFLICT_TYPE" revert.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "conflict_type",
    ["semantic", "attributional", "normative", "structural", "factual"],
)
def test_all_five_conflict_types_are_valid(conflict_type):
    c = _make_conflict(conflict_type=conflict_type)
    assert c.conflict_type == conflict_type


def test_intervention_class_has_all_five_types():
    for t in ["semantic", "attributional", "normative", "structural", "factual"]:
        assert t in INTERVENTION_CLASS


# ---------------------------------------------------------------------------
# 1. False-resolution trap: type/resolvability consistency enforcement.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("resolvability", ["conversational", "facilitated"])
def test_structural_rejects_conversational_and_facilitated(resolvability):
    with pytest.raises(ValidationError, match="inconsistent"):
        _make_conflict("structural", resolvability)


def test_normative_rejects_conversational():
    with pytest.raises(ValidationError, match="inconsistent"):
        _make_conflict("normative", "conversational")


@pytest.mark.parametrize("resolvability", ["conversational", "facilitated"])
def test_factual_rejects_conversational_and_facilitated(resolvability):
    with pytest.raises(ValidationError, match="inconsistent"):
        _make_conflict("factual", resolvability)


@pytest.mark.parametrize("resolvability", ["structural_change", "external_standard", "unresolvable"])
def test_structural_accepts_valid_resolvability(resolvability):
    c = _make_conflict("structural", resolvability)
    assert c.resolvability == resolvability


@pytest.mark.parametrize("conflict_type", ["semantic", "attributional"])
def test_conversational_allowed_for_soft_types(conflict_type):
    c = _make_conflict(conflict_type, "conversational")
    assert c.resolvability == "conversational"


# ---------------------------------------------------------------------------
# 2. implication field validator.
# ---------------------------------------------------------------------------

def test_implication_must_start_with_structural_marker():
    with pytest.raises(ValidationError, match="must begin"):
        _make_conflict(implication="They simply have different communication preferences here.")


def test_implication_too_short_rejected():
    with pytest.raises(ValidationError, match="too short"):
        _make_conflict(implication="This reveals a gap.")


@pytest.mark.parametrize("starter", [
    "This reveals that", "This suggests that", "This exposes that",
    "This shows that", "This uncovers that", "This indicates that",
])
def test_all_valid_starters_accepted(starter):
    impl = starter + (
        " both parties operated with incompatible definitions of what "
        "professional feedback is for, invisible to both sides."
    )
    c = _make_conflict(implication=impl)
    assert c.implication.lower().startswith(starter.lower())


# ---------------------------------------------------------------------------
# 3. conflict_type_rationale validator.
# ---------------------------------------------------------------------------

def test_rationale_too_short_rejected():
    with pytest.raises(ValidationError):
        _make_conflict(rationale="too short")


# ---------------------------------------------------------------------------
# 4. ConflictTypeSummary auto-computation.
# ---------------------------------------------------------------------------

def test_summary_counts_and_dominant_type():
    conflicts = [
        _make_conflict("structural", severity=4),
        _make_conflict("structural", severity=5),
        _make_conflict("semantic", "conversational", severity=2),
        _make_conflict("normative", severity=3),
    ]
    s = _implying(conflicts).conflict_type_summary
    assert s is not None
    assert s.structural == 2
    assert s.semantic == 1
    assert s.normative == 1
    assert s.total == 4
    assert s.dominant_type == "structural"


def test_summary_fractions():
    conflicts = [
        _make_conflict("structural", severity=4),
        _make_conflict("semantic", "conversational", severity=2),
    ]
    s = _implying(conflicts).conflict_type_summary
    assert abs(s.structural_fraction - 0.5) < 1e-6
    assert abs(s.conversational_fraction - 0.5) < 1e-6
    # severity-weighted: one structural with severity 4, out of 2 total → 4/(5*2)=0.4
    assert abs(s.structural_severity_score - 0.4) < 1e-6


def test_empty_conflicts_yields_none_summary():
    assert _implying([]).conflict_type_summary is None


def test_unresolvable_fraction():
    conflicts = [
        _make_conflict("structural", "unresolvable"),
        _make_conflict("structural", "structural_change"),
    ]
    s = _implying(conflicts).conflict_type_summary
    assert abs(s.unresolvable_fraction - 0.5) < 1e-6
