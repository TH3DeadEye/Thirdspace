import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT, _ROOT / "thirdspace"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from metrics import conflict_jaccard
from models import ImplyingNodes, WholenessImplyingNodes, CellConflict


def _wholeness():
    return WholenessImplyingNodes(
        query_assumptions=[], reformulated_query=None, new_objective=None,
        updated_challenger_perspective=None, objective_assumptions=[],
        perspective_assumptions=[],
    )


def _conflict(facet, dimension):
    return CellConflict(
        facet=facet, dimension=dimension,
        main_summary="m", challenger_summary="c",
        challenger_response_type="challenge",
        implication=(
            "This reveals that the two frames hold incompatible definitions "
            "of professional respect in this exchange."
        ),
        conflict_type="structural",
        conflict_type_rationale=(
            "Classified structural because the asymmetry reproduces under any "
            "occupants of these roles."
        ),
        severity=3, resolvability="structural_change",
    )


def _cycle(cycle, conflicts):
    return ImplyingNodes(
        cycle=cycle, agreements=[], conflicts=conflicts,
        what_conflicts_imply=[], updated_assumptions=[],
        wholeness=_wholeness(), converged=False,
    )


def test_identical_sets_is_one():
    a = _cycle(0, [_conflict("individuals", "objective")])
    b = _cycle(1, [_conflict("individuals", "objective")])
    assert conflict_jaccard(a, b) == 1.0


def test_disjoint_sets_is_zero():
    a = _cycle(0, [_conflict("individuals", "objective")])
    b = _cycle(1, [_conflict("process", "evidence")])
    assert conflict_jaccard(a, b) == 0.0


def test_both_empty_is_one():
    assert conflict_jaccard(_cycle(0, []), _cycle(1, [])) == 1.0


def test_partial_overlap():
    a = _cycle(0, [_conflict("individuals", "objective")])
    b = _cycle(1, [
        _conflict("individuals", "objective"),
        _conflict("process", "evidence"),
    ])
    # intersection {1}, union {2} → 0.5
    assert conflict_jaccard(a, b) == 0.5
