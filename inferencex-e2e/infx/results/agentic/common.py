"""Shared helpers for agentic aggregate generation."""

from __future__ import annotations

import math
import statistics
from typing import Any


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def stats_for(prefix: str, values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        f"mean_{prefix}": statistics.mean(values),
        f"p50_{prefix}": percentile(values, 50),
        f"p75_{prefix}": percentile(values, 75),
        f"p90_{prefix}": percentile(values, 90),
        f"p95_{prefix}": percentile(values, 95),
        f"std_{prefix}": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def round_floats(obj: Any, decimal_places: int = 5) -> Any:
    """Round every finite float in a nested JSON-like object."""
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return obj
        rounded = round(obj, decimal_places)
        if abs(rounded) >= 1 and rounded.is_integer():
            return int(rounded)
        return rounded
    if isinstance(obj, dict):
        return {key: round_floats(value, decimal_places) for key, value in obj.items()}
    if isinstance(obj, list):
        return [round_floats(value, decimal_places) for value in obj]
    return obj
