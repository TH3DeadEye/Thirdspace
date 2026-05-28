"""
Unit tests for the budget / circuit-breaker module.

These tests do NOT go through call_llm / Bedrock. Instead they verify the
context registration mechanics and the two exception classes that call_llm is
wired to raise. The integration with call_llm is exercised indirectly by the
test that monkey-patches boto3 ClientError below.
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT, _ROOT / "thirdspace"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import budget
from models import SessionMemory


def _fresh_memory(**overrides) -> SessionMemory:
    fields = {
        "scenario_id": "budget-test",
        "main_cultural_frame": "A",
        "challenger_cultural_frame": "B",
    }
    fields.update(overrides)
    return SessionMemory(**fields)


class TestBudgetContext(unittest.TestCase):
    def test_no_active_memory_is_noop(self):
        # Calling the budget functions without an active memory should silently
        # do nothing — supports unit tests that exercise call_llm directly.
        self.assertIsNone(budget.get_active_memory())
        budget.check_budget_before_call()  # must not raise
        budget.record_success(123)
        budget.record_failure()

    def test_set_and_reset(self):
        mem = _fresh_memory()
        token = budget.set_active_memory(mem)
        try:
            self.assertIs(budget.get_active_memory(), mem)
        finally:
            budget.reset_active_memory(token)
        self.assertIsNone(budget.get_active_memory())


class TestBudgetAccounting(unittest.TestCase):
    def test_success_increments_calls_and_latency(self):
        mem = _fresh_memory(llm_call_budget=10)
        token = budget.set_active_memory(mem)
        try:
            budget.record_success(100)
            budget.record_success(250)
            self.assertEqual(mem.llm_calls_made, 2)
            self.assertEqual(mem.total_latency_ms, 350)
            self.assertEqual(mem.consecutive_failures, 0)
        finally:
            budget.reset_active_memory(token)

    def test_budget_exceeded_raises_before_call(self):
        mem = _fresh_memory(llm_call_budget=2)
        token = budget.set_active_memory(mem)
        try:
            budget.check_budget_before_call()
            budget.record_success(10)
            budget.check_budget_before_call()
            budget.record_success(10)
            # Third call would exceed budget=2.
            with self.assertRaises(budget.BudgetExceeded):
                budget.check_budget_before_call()
        finally:
            budget.reset_active_memory(token)

    def test_success_resets_consecutive_failures(self):
        mem = _fresh_memory(max_consecutive_failures=5)
        token = budget.set_active_memory(mem)
        try:
            mem.consecutive_failures = 3
            budget.record_success(10)
            self.assertEqual(mem.consecutive_failures, 0)
        finally:
            budget.reset_active_memory(token)


class TestCircuitBreaker(unittest.TestCase):
    def test_circuit_opens_after_max_failures(self):
        mem = _fresh_memory(max_consecutive_failures=2)
        token = budget.set_active_memory(mem)
        try:
            budget.record_failure()  # 1, below cap
            budget.record_failure()  # 2, at cap, still ok
            with self.assertRaises(budget.CircuitOpen):
                budget.record_failure()  # 3, > cap
            self.assertEqual(mem.consecutive_failures, 3)
        finally:
            budget.reset_active_memory(token)

    def test_circuit_does_not_open_when_no_active_memory(self):
        # No registered memory -> no breaker state to maintain.
        budget.record_failure()  # must not raise


class TestCallLLMIntegration(unittest.TestCase):
    """Verify call_llm wires budget checks + circuit breaker correctly without
    actually hitting Bedrock."""

    def test_call_llm_increments_budget_on_success(self):
        import llm
        mem = _fresh_memory(llm_call_budget=10)
        token = budget.set_active_memory(mem)
        try:
            mock_client = MagicMock()
            mock_client.converse.return_value = {
                "output": {"message": {"content": [{"text": "ok"}]}}
            }
            with patch.object(llm, "get_client", return_value=mock_client):
                result = llm.call_llm("sys", "user")
            self.assertEqual(result, "ok")
            self.assertEqual(mem.llm_calls_made, 1)
            self.assertGreaterEqual(mem.total_latency_ms, 0)
            self.assertEqual(mem.consecutive_failures, 0)
        finally:
            budget.reset_active_memory(token)

    def test_call_llm_raises_budget_exceeded_before_calling_bedrock(self):
        import llm
        mem = _fresh_memory(llm_call_budget=0)  # zero budget = block all calls
        token = budget.set_active_memory(mem)
        try:
            mock_client = MagicMock()
            with patch.object(llm, "get_client", return_value=mock_client):
                with self.assertRaises(budget.BudgetExceeded):
                    llm.call_llm("sys", "user")
            mock_client.converse.assert_not_called()
        finally:
            budget.reset_active_memory(token)

    def test_call_llm_failure_does_not_increment_calls_made(self):
        """Failed calls count against the circuit breaker but NOT the success
        counter — otherwise transient failures eat the budget twice."""
        import llm
        from botocore.exceptions import ClientError
        mem = _fresh_memory(llm_call_budget=10, max_consecutive_failures=10)
        token = budget.set_active_memory(mem)
        try:
            mock_client = MagicMock()
            mock_client.converse.side_effect = ClientError(
                {"Error": {"Code": "ValidationException", "Message": "bad"}}, "Converse"
            )
            with patch.object(llm, "get_client", return_value=mock_client):
                with self.assertRaises(ClientError):
                    llm.call_llm("sys", "user")
            self.assertEqual(mem.llm_calls_made, 0)
            self.assertEqual(mem.consecutive_failures, 1)
        finally:
            budget.reset_active_memory(token)


class TestActiveMemoryContextManager(unittest.TestCase):
    """The `active_memory` context manager is what worker threads use to
    re-register the parent's SessionMemory inside a fresh contextvars context."""

    def test_none_is_noop(self):
        with budget.active_memory(None):
            self.assertIsNone(budget.get_active_memory())

    def test_registers_and_restores(self):
        outer = _fresh_memory()
        outer_token = budget.set_active_memory(outer)
        try:
            inner = _fresh_memory(scenario_id="inner")
            with budget.active_memory(inner):
                self.assertIs(budget.get_active_memory(), inner)
            # On exit the previous registration is restored.
            self.assertIs(budget.get_active_memory(), outer)
        finally:
            budget.reset_active_memory(outer_token)


class TestParallelBudgetPropagation(unittest.TestCase):
    """Regression test for the contextvar-not-propagated bug.

    Without the `active_memory(memory)` wrapper inside `_ct_cell`, the 42
    cell-level LLM calls run in fresh contextvars contexts and silently
    bypass the budget / circuit-breaker hooks in call_llm. This verifies
    each cell call still increments mem.llm_calls_made.
    """

    def setUp(self):
        """Other test modules (test_agents) replace
        agents.call_llm_json_with_refinement with a lambda and don't restore
        it. We need the real chain (call_llm_json_with_refinement -> call_llm
        -> get_client) to exercise the budget hooks, so restore it here."""
        import llm
        import agents
        self._saved = {
            "call_llm": agents.call_llm,
            "call_llm_json": agents.call_llm_json,
            "call_llm_json_with_refinement": agents.call_llm_json_with_refinement,
        }
        agents.call_llm = llm.call_llm
        agents.call_llm_json = llm.call_llm_json
        agents.call_llm_json_with_refinement = llm.call_llm_json_with_refinement

    def tearDown(self):
        import agents
        for name, fn in self._saved.items():
            setattr(agents, name, fn)

    def test_cell_calls_accumulate_into_budget_via_thread_pool(self):
        import llm
        import agents
        from models import NUM_CELLS
        from tests.fixtures import make_occurring

        mem = _fresh_memory(llm_call_budget=NUM_CELLS * 2)
        token = budget.set_active_memory(mem)
        try:
            mock_client = MagicMock()
            mock_client.converse.return_value = {
                "output": {"message": {"content": [{
                    "text": (
                        '{"content": "Sentence one. Sentence two. '
                        'Sentence three.", "reflection_flags": []}'
                    ),
                }]}}
            }
            with patch.object(llm, "get_client", return_value=mock_client):
                grid = agents.run_critical_thinking(
                    make_occurring(), "frame", cycle=0
                )
            self.assertEqual(len(grid.cells), NUM_CELLS)
            # Without the fix, this would be 0 because each worker's
            # contextvar would see None and silently skip record_success.
            self.assertEqual(mem.llm_calls_made, NUM_CELLS)
            self.assertEqual(mem.consecutive_failures, 0)
        finally:
            budget.reset_active_memory(token)

    def test_circuit_trips_from_worker_thread_failures(self):
        """A worker-thread failure must also tick the circuit breaker —
        otherwise a degraded Bedrock could burn the full per-scenario budget."""
        import llm
        import agents
        from botocore.exceptions import ClientError
        from tests.fixtures import make_occurring

        # Tight breaker: any worker thread that fails should trip it quickly.
        mem = _fresh_memory(llm_call_budget=1000, max_consecutive_failures=2)
        token = budget.set_active_memory(mem)
        try:
            mock_client = MagicMock()
            mock_client.converse.side_effect = ClientError(
                {"Error": {"Code": "ValidationException", "Message": "bad"}},
                "Converse",
            )
            with patch.object(llm, "get_client", return_value=mock_client):
                with self.assertRaises((ClientError, budget.CircuitOpen)):
                    agents.run_critical_thinking(make_occurring(), "frame", cycle=0)
            # Even one worker failure should have incremented the counter.
            self.assertGreater(mem.consecutive_failures, 0)
        finally:
            budget.reset_active_memory(token)


class TestRefinementDictGuard(unittest.TestCase):
    """call_llm_json_with_refinement must reject non-object JSON loudly
    rather than TypeError later on result['_refinement_meta']."""

    def test_list_raises_valueerror(self):
        import llm
        original = llm.call_llm_json
        try:
            llm.call_llm_json = lambda *a, **kw: ["not", "a", "dict"]
            with self.assertRaises(ValueError) as cm:
                llm.call_llm_json_with_refinement("sys", "user")
            self.assertIn("JSON object", str(cm.exception))
        finally:
            llm.call_llm_json = original

    def test_scalar_raises_valueerror(self):
        import llm
        original = llm.call_llm_json
        try:
            llm.call_llm_json = lambda *a, **kw: "bare string"
            with self.assertRaises(ValueError):
                llm.call_llm_json_with_refinement("sys", "user")
        finally:
            llm.call_llm_json = original


if __name__ == "__main__":
    unittest.main()
