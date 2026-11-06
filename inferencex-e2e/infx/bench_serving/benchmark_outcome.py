"""The serving client's existing request-failure gate, preserved with results."""

from __future__ import annotations

from typing import Literal, TypedDict

MAX_FAILURE_RATE = 0.05


class BenchmarkOutcome(TypedDict):
    status: Literal["passed", "failed"]
    requested: int
    completed: int
    failed: int
    max_failure_rate: float


def benchmark_outcome(requested: int, completed: int) -> BenchmarkOutcome:
    """Apply the five-percent failure gate to ``requested`` and ``completed`` and return the outcome.

    Raises ``ValueError`` unless both are ints with ``0 <= completed <= requested`` and ``requested > 0``.
    """
    if (
        type(requested) is not int
        or type(completed) is not int
        or requested <= 0
        or not 0 <= completed <= requested
    ):
        raise ValueError("Benchmark request counts must satisfy 0 <= completed <= requested > 0")
    failed = requested - completed
    return {
        "status": "failed" if failed / requested > MAX_FAILURE_RATE else "passed",
        "requested": requested,
        "completed": completed,
        "failed": failed,
        "max_failure_rate": MAX_FAILURE_RATE,
    }
