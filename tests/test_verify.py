"""Post-run invariant checks.

Each test perturbs exactly one thing on an otherwise-clean session, so a check
that stops firing is visible immediately. If these pass but the checks are dead,
the whole point of verify_session — making a damaged run distinguishable from a
clean one — is lost.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT, _ROOT / "thirdspace"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from models import WholenessEntry, WholenessImplyingNodes, NUM_CELLS
from verify import verify_session, format_report
from tests.fixtures import make_complete_session, make_mediation, make_conflict


def test_clean_session_has_no_problems():
    assert verify_session(make_complete_session(cycles=2)) == []


def test_clean_report_says_so():
    assert "all invariants hold" in format_report([])


def test_short_grid_is_caught():
    mem = make_complete_session()
    mem.main_grids[0].cells.pop()
    problems = verify_session(mem)
    assert any(f"expected {NUM_CELLS}" in p for p in problems)


def test_empty_cell_is_caught():
    mem = make_complete_session()
    mem.main_grids[0].cells[3].content = "   "
    assert any("empty cell" in p for p in verify_session(mem))


def test_cycle_artefacts_out_of_step_is_caught():
    mem = make_complete_session(cycles=2)
    mem.challenger_grids.pop()
    assert any("out of step" in p for p in verify_session(mem))


def test_unresolved_wholeness_ref_is_caught():
    mem = make_complete_session()
    mem.mediation_history[0].wholeness.what_conflicts_imply = [
        WholenessEntry(statement="unsupported", refs=["nowhere/atall"])]
    assert any("unresolved ref" in p for p in verify_session(mem))


def test_dropped_conflicts_are_surfaced():
    mem = make_complete_session()
    mem.conflicts_dropped = 3
    assert any("undercount" in p for p in verify_session(mem))


def test_leaked_rubric_vocabulary_is_caught():
    mem = make_complete_session()
    mem.final_output = ("You should know the false resolution trap applies here.")
    assert any("rubric vocabulary" in p for p in verify_session(mem))


def test_missing_document_is_caught_only_when_generation_expected():
    mem = make_complete_session()
    mem.final_output = ""
    assert any("final_output is empty" in p for p in verify_session(mem))

    mem.skip_generation = True
    assert not any("final_output" in p for p in verify_session(mem))


def test_undercounted_calls_are_caught():
    mem = make_complete_session()
    mem.llm_calls_made = 3
    assert any("below the" in p for p in verify_session(mem))


def test_missing_convergence_reason_is_caught():
    mem = make_complete_session()
    mem.convergence_reason = None
    assert any("convergence_reason" in p for p in verify_session(mem))


def test_more_conflicts_than_cells_is_caught():
    mem = make_complete_session()
    mem.mediation_history[0].conflicts = [
        make_conflict("agents", "value") for _ in range(NUM_CELLS + 2)]
    assert any("more than the" in p for p in verify_session(mem))
