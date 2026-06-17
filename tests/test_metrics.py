import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT, _ROOT / "thirdspace"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import metrics
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


def _cycle_implies(cycle, implies):
    return ImplyingNodes(
        cycle=cycle, agreements=[], conflicts=[],
        what_conflicts_imply=implies, updated_assumptions=[],
        wholeness=_wholeness(), converged=False,
    )


def _char_embed(text):
    """Deterministic offline embedding: per-character counts over a fixed
    alphabet. Identical strings -> identical vectors -> cosine 1.0."""
    alphabet = "abcdefghijklmnopqrstuvwxyz "
    low = text.lower()
    return [float(low.count(c)) for c in alphabet]


# --- conflict_jaccard ---

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


# --- conflict_signature ---

def test_signature_joins_implications_and_patterns():
    c = _cycle(0, [_conflict("individuals", "objective")])
    c.what_conflicts_imply = ["pattern A", "pattern B"]
    sig = metrics.conflict_signature(c)
    assert "This reveals that" in sig
    assert "pattern A" in sig and "pattern B" in sig


def test_signature_skips_empty_and_whitespace():
    c = _cycle(0, [])
    c.what_conflicts_imply = ["", "real pattern", "   "]
    assert metrics.conflict_signature(c) == "real pattern"


def test_signature_empty_when_nothing():
    assert metrics.conflict_signature(_cycle(0, [])) == ""


# --- cosine ---

def test_cosine_identical():
    assert metrics._cosine([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_cosine_orthogonal():
    assert metrics._cosine([1, 0], [0, 1]) == 0.0


def test_cosine_zero_vector_guard():
    assert metrics._cosine([0, 0], [1, 1]) == 0.0


# --- conflict_embedding_similarity ---

def test_embed_both_empty_short_circuits_without_calling():
    def boom(_):
        raise AssertionError("embed_fn must not be called for empty signatures")
    assert metrics.conflict_embedding_similarity(
        _cycle(0, []), _cycle(1, []), embed_fn=boom
    ) == 1.0


def test_embed_one_empty_short_circuits_without_calling():
    def boom(_):
        raise AssertionError("embed_fn must not be called when one side is empty")
    a = _cycle_implies(0, ["something"])
    assert metrics.conflict_embedding_similarity(a, _cycle(1, []), embed_fn=boom) == 0.0


def test_embed_identical_content_high():
    a = _cycle_implies(0, ["the same finding text about silence"])
    b = _cycle_implies(1, ["the same finding text about silence"])
    assert metrics.conflict_embedding_similarity(a, b, embed_fn=_char_embed) == pytest.approx(1.0)


def test_embed_different_content_low():
    a = _cycle_implies(0, ["aaaa aaaa aaaa"])
    b = _cycle_implies(1, ["zzzz zzzz zzzz"])
    assert metrics.conflict_embedding_similarity(a, b, embed_fn=_char_embed) < 0.5


def test_embed_resolves_module_global(monkeypatch):
    # convergence_similarity does not pass embed_fn, so monkeypatching the module
    # global must take effect (validates call-time resolution, not def-time bind).
    monkeypatch.setattr(metrics, "embed_text", _char_embed)
    a = _cycle_implies(0, ["alpha beta gamma"])
    b = _cycle_implies(1, ["alpha beta gamma"])
    assert metrics.conflict_embedding_similarity(a, b) == pytest.approx(1.0)


# --- convergence_similarity dispatch ---

def test_dispatch_jaccard():
    a = _cycle(0, [_conflict("individuals", "objective")])
    b = _cycle(1, [_conflict("process", "evidence")])
    assert metrics.convergence_similarity(a, b, "jaccard") == 0.0


def test_dispatch_embedding_compares_content_not_cells(monkeypatch):
    # Same implication text on DIFFERENT cells: jaccard reads 0 (no cell overlap),
    # but the embedding metric compares content -> ~1.0.
    monkeypatch.setattr(metrics, "embed_text", _char_embed)
    a = _cycle(0, [_conflict("individuals", "objective")])
    b = _cycle(1, [_conflict("process", "evidence")])
    assert metrics.convergence_similarity(a, b, "jaccard") == 0.0
    assert metrics.convergence_similarity(a, b, "embedding") == pytest.approx(1.0)