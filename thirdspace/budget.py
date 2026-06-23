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
import logging
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models import SessionMemory

logger = logging.getLogger(__name__)

# One lock per process, held only around the integer counter mutations in
# record_success / record_failure (never across an LLM call), so contention is
# microseconds. Fixes the read-modify-write race on `mem.llm_calls_made += 1`
# under the cell-level ThreadPoolExecutor.
_counter_lock = threading.Lock()


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
    with _counter_lock:
        mem.llm_calls_made += 1
        mem.total_latency_ms += latency_ms
        mem.consecutive_failures = 0
        # Post-increment guard: check_budget_before_call is intentionally
        # lock-free (locking 84 cell calls through one chokepoint would kill the
        # parallelism), so one call can slip past a nearly-exhausted budget.
        # Catch and log that here rather than preventing it there.
        if mem.llm_calls_made > mem.llm_call_budget:
            logger.warning(
                f"[{mem.scenario_id}] Budget exceeded post-call: "
                f"{mem.llm_calls_made} > {mem.llm_call_budget}. "
                f"A race allowed one extra call through."
            )


def record_failure() -> None:
    """Called by call_llm when a Bedrock call exhausts retries.

    Raises CircuitOpen if the failure count is now over the cap.
    """
    mem = _active_memory.get()
    if mem is None:
        return
    with _counter_lock:
        mem.consecutive_failures += 1
        over_limit = mem.consecutive_failures > mem.max_consecutive_failures
        failures = mem.consecutive_failures
    # Raise outside the lock — never raise while holding it.
    if over_limit:
        raise CircuitOpen(
            f"Scenario {mem.scenario_id} circuit open: "
            f"{failures} consecutive failures "
            f"(> {mem.max_consecutive_failures})"
        )


@contextmanager
def active_memory(memory: "SessionMemory | None"):
    """Register `memory` as the active SessionMemory for the duration of the block.

    Used inside ThreadPoolExecutor workers so cell-level LLM calls see the
    same SessionMemory the main thread registered. If `memory` is None the
    context manager is a no-op (supports tests that don't register a memory).

    Concurrency note: `mem.llm_calls_made += 1` is a read-modify-write that
    could otherwise miss increments under high parallelism, so record_success /
    record_failure now mutate the counters under `_counter_lock`.
    """
    if memory is None:
        yield
        return
    token = set_active_memory(memory)
    try:
        yield
    finally:
        reset_active_memory(token)
