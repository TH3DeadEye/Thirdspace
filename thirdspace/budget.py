"""
Cost / latency guardrails.

A scenario run registers its active SessionMemory in this module's context
variable. `call_llm` then increments call/latency counters on every Bedrock
request, and raises BudgetExceeded or CircuitOpen when the configured limits
are hit.

Threading note: `concurrent.futures.ThreadPoolExecutor` does NOT propagate
the parent's contextvars to worker threads — each worker starts with a fresh
context where `_active_memory` is None. The cell agents
(run_critical_thinking, run_challenger) capture the active memory in the main
thread and re-register it inside each worker via `with active_memory(mem):`.
Without that wrapper, the ~84 cell calls per cycle would silently bypass both
the budget check and the circuit breaker.

If no memory is registered (e.g. during unit tests that exercise call_llm
directly), the counters and breaker are no-ops.
"""
from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models import SessionMemory


class BudgetExceeded(RuntimeError):
    """Raised when the scenario exceeds its configured LLM call budget."""


class CircuitOpen(RuntimeError):
    """Raised when consecutive LLM failures exceed the configured cap.

    Trips the circuit and forces the scenario to abort instead of burning
    further budget on a degraded backend.
    """


_active_memory: contextvars.ContextVar["SessionMemory | None"] = contextvars.ContextVar(
    "thirdspace_active_memory", default=None,
)


def set_active_memory(memory: "SessionMemory | None") -> contextvars.Token:
    """Register the SessionMemory that should accumulate LLM telemetry.

    Returns the contextvars Token so the caller can `reset_active_memory(token)`
    after the scenario finishes.
    """
    return _active_memory.set(memory)


def reset_active_memory(token: contextvars.Token) -> None:
    _active_memory.reset(token)


def get_active_memory() -> "SessionMemory | None":
    return _active_memory.get()


def check_budget_before_call() -> None:
    """Raise BudgetExceeded if a new call would exceed the budget.

    Called by call_llm immediately before issuing a Bedrock request.
    """
    mem = _active_memory.get()
    if mem is None:
        return
    # +1 because we're about to make a call that hasn't been counted yet.
    if mem.llm_calls_made + 1 > mem.llm_call_budget:
        raise BudgetExceeded(
            f"Scenario {mem.scenario_id} exceeded LLM call budget: "
            f"{mem.llm_calls_made + 1} > {mem.llm_call_budget}"
        )


def record_success(latency_ms: int) -> None:
    """Called by call_llm on a successful Bedrock response."""
    mem = _active_memory.get()
    if mem is None:
        return
    mem.llm_calls_made += 1
    mem.total_latency_ms += latency_ms
    mem.consecutive_failures = 0


def record_failure() -> None:
    """Called by call_llm when a Bedrock call exhausts retries.

    Raises CircuitOpen if the failure count is now over the cap.
    """
    mem = _active_memory.get()
    if mem is None:
        return
    mem.consecutive_failures += 1
    if mem.consecutive_failures > mem.max_consecutive_failures:
        raise CircuitOpen(
            f"Scenario {mem.scenario_id} circuit open: "
            f"{mem.consecutive_failures} consecutive failures "
            f"(> {mem.max_consecutive_failures})"
        )


@contextmanager
def active_memory(memory: "SessionMemory | None"):
    """Register `memory` as the active SessionMemory for the duration of the block.

    Used inside ThreadPoolExecutor workers so cell-level LLM calls see the
    same SessionMemory the main thread registered. If `memory` is None the
    context manager is a no-op (supports tests that don't register a memory).

    Concurrency note: under CPython the GIL makes most attribute reads atomic,
    but `mem.llm_calls_made += 1` is a read-modify-write and could miss a
    handful of increments under high parallelism. For the 42-cells-per-cycle
    workload this is acceptable; if exact counts become important, add a lock
    around `record_success` / `record_failure`.
    """
    if memory is None:
        yield
        return
    token = set_active_memory(memory)
    try:
        yield
    finally:
        reset_active_memory(token)
