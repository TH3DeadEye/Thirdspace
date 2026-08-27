"""The six-type conflict taxonomy and the validators that enforce it.

models.py is the enforcement point: a conflict type paired with a remedy that
cannot work for it is rejected rather than coerced, because a mismatch prescribes
an intervention that is guaranteed to fail (telling parties in a power conflict to
talk it out, or offering an external standard for a recognition injury).
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT, _ROOT / "thirdspace"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytest

from models import (
    CellConflict, ConflictType, ConflictTypeSummary, Resolvability,
    CONFLICT_TYPE_DEFS, get_allowed_resolvabilities,
)
from tests.fixtures import make_conflict, VALID_IMPLICATION, VALID_RATIONALE


def _build(conflict_type, resolvability, **kw):
    return CellConflict(
        facet=kw.get("facet", "agents"), dimension=kw.get("dimension", "value"),
        main_summary="m", challenger_summary="c",
        implication=kw.get("implication", VALID_IMPLICATION),
        conflict_type=conflict_type,
        conflict_type_rationale=kw.get("rationale", VALID_RATIONALE),
        severity=kw.get("severity", 3), resolvability=resolvability,
    )


# --- registry integrity ------------------------------------------------------

def test_registry_covers_every_conflict_type():
    assert set(CONFLICT_TYPE_DEFS) == set(ConflictType)


def test_exactly_one_default_type():
    assert sum(d.is_default for d in CONFLICT_TYPE_DEFS.values()) == 1
    assert CONFLICT_TYPE_DEFS[ConflictType.NORMATIVE].is_default


def test_prompt_table_matches_registry():
    """prompts.py asserts this at import; pin it here so the failure names itself."""
    from utils.prompts import RESOLVABILITY_BY_TYPE
    assert set(RESOLVABILITY_BY_TYPE) == {str(t) for t in ConflictType}


# --- type -> resolvability whitelist ----------------------------------------

@pytest.mark.parametrize("ctype,bad", [
    (ConflictType.POWER_RELATED, Resolvability.CONVERSATIONAL),
    (ConflictType.POWER_RELATED, Resolvability.FACILITATED),
    (ConflictType.NORMATIVE, Resolvability.CONVERSATIONAL),
    (ConflictType.FACTUAL, Resolvability.CONVERSATIONAL),
    (ConflictType.FACTUAL, Resolvability.FACILITATED),
    # Recognition cannot be mandated by an outside rule.
    (ConflictType.IDENTITY_RELATED, Resolvability.EXTERNAL_STANDARD),
])
def test_illegal_pairings_rejected(ctype, bad):
    with pytest.raises(ValueError):
        _build(ctype, bad)


@pytest.mark.parametrize("ctype,good", [
    (ConflictType.POWER_RELATED, Resolvability.STRUCTURAL_CHANGE),
    (ConflictType.INTERPRETIVE, Resolvability.CONVERSATIONAL),
    (ConflictType.PROCEDURAL, Resolvability.EXTERNAL_STANDARD),
    (ConflictType.IDENTITY_RELATED, Resolvability.FACILITATED),
    (ConflictType.NORMATIVE, Resolvability.FACILITATED),
    (ConflictType.FACTUAL, Resolvability.EXTERNAL_STANDARD),
])
def test_legal_pairings_accepted(ctype, good):
    assert _build(ctype, good).resolvability == good


def test_every_type_has_at_least_one_legal_remedy():
    for ctype in ConflictType:
        assert get_allowed_resolvabilities(ctype)


# --- field validators --------------------------------------------------------

def test_implication_must_open_with_a_reveal_marker():
    with pytest.raises(ValueError, match="implication"):
        _build(ConflictType.NORMATIVE, Resolvability.FACILITATED,
               implication="The parties simply disagree about what matters most here, at length.")


@pytest.mark.parametrize("starter", [
    "This reveals", "This suggests", "This exposes",
    "This shows", "This uncovers", "This indicates",
])
def test_all_documented_starters_accepted(starter):
    text = f"{starter} that the two readings rest on incompatible premises about obligation."
    assert _build(ConflictType.NORMATIVE, Resolvability.FACILITATED, implication=text)


def test_implication_too_short_rejected():
    with pytest.raises(ValueError, match="too short"):
        _build(ConflictType.NORMATIVE, Resolvability.FACILITATED,
               implication="This reveals a gap.")


def test_rationale_too_short_rejected():
    with pytest.raises(ValueError, match="rationale"):
        _build(ConflictType.NORMATIVE, Resolvability.FACILITATED, rationale="Step six.")


def test_semantic_subtype_only_valid_for_interpretive():
    with pytest.raises(ValueError, match="is_semantic_subtype"):
        CellConflict(facet="agents", dimension="value", main_summary="m",
                     challenger_summary="c", implication=VALID_IMPLICATION,
                     conflict_type=ConflictType.NORMATIVE,
                     conflict_type_rationale=VALID_RATIONALE, severity=3,
                     resolvability=Resolvability.FACILITATED, is_semantic_subtype=True)


def test_cell_id_derived_when_absent():
    assert _build(ConflictType.NORMATIVE, Resolvability.FACILITATED).cell_id == "agents/value"


def test_intervention_is_derived_from_type_not_supplied():
    c = _build(ConflictType.POWER_RELATED, Resolvability.STRUCTURAL_CHANGE)
    assert c.intervention == CONFLICT_TYPE_DEFS[ConflictType.POWER_RELATED].intervention_text


# --- summary -----------------------------------------------------------------

def test_summary_counts_and_dominant_type():
    s = ConflictTypeSummary.from_conflicts([
        make_conflict("agents", "value", "power_related", "structural_change"),
        make_conflict("actions", "value", "power_related", "structural_change"),
        make_conflict("context", "value", "normative", "facilitated"),
    ])
    assert s.total == 3
    assert s.power_related == 2
    assert s.normative == 1
    assert s.dominant_type == ConflictType.POWER_RELATED


def test_summary_fractions():
    s = ConflictTypeSummary.from_conflicts([
        make_conflict("agents", "interpretation", "interpretive", "conversational"),
        make_conflict("actions", "value", "normative", "unresolvable"),
    ])
    assert abs(s.conversational_fraction - 0.5) < 1e-6
    assert abs(s.unresolvable_fraction - 0.5) < 1e-6


def test_empty_conflicts_yields_none_summary():
    assert ConflictTypeSummary.from_conflicts([]) is None


def test_aggregate_across_cycles():
    a = ConflictTypeSummary.from_conflicts(
        [make_conflict("agents", "value", "normative", "facilitated")], cycles_included=[1])
    b = ConflictTypeSummary.from_conflicts(
        [make_conflict("actions", "value", "power_related", "structural_change")],
        cycles_included=[2])
    agg = ConflictTypeSummary.aggregate([a, b])
    assert agg.total == 2
    assert agg.scope == "cross_cycle"
    assert agg.cycles_included == [1, 2]
