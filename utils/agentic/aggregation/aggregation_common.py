"""Shared helpers for agentic aggregate generation."""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
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
    except (TypeError, ValueError):
        return None


def rate(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def normalize_fraction(value: float | None) -> float | None:
    """Normalize gauges that may be exported as 0..1 or 0..100."""
    if value is None:
        return None
    if value > 1.5:
        return value / 100.0
    return value


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


def index_server_metrics(server_metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the metrics dict from aiperf's server_metrics_export.json."""
    if not isinstance(server_metrics, dict):
        return {}
    metrics = server_metrics.get("metrics")
    if isinstance(metrics, dict):
        return metrics
    return {}


def metric_series(
    metrics: dict[str, dict[str, Any]],
    metric_names: str | list[str],
) -> list[dict[str, Any]]:
    names = [metric_names] if isinstance(metric_names, str) else metric_names
    out: list[dict[str, Any]] = []
    for name in names:
        entry = metrics.get(name)
        if not isinstance(entry, dict):
            continue
        series = entry.get("series")
        if not isinstance(series, list):
            continue
        out.extend(s for s in series if isinstance(s, dict))
    return out


def series_stat(
    series: dict[str, Any],
    preferred_keys: tuple[str, ...] = ("total", "sum", "max", "avg"),
) -> float | None:
    stats = series.get("stats")
    if not isinstance(stats, dict):
        return None
    for key in preferred_keys:
        value = to_float(stats.get(key))
        if value is not None:
            return value
    return None


def sum_stat(
    metrics: dict[str, dict[str, Any]],
    metric_names: str | list[str],
    *,
    preferred_keys: tuple[str, ...] = ("total", "sum", "max", "avg"),
    series_filter: Callable[[dict[str, Any]], bool] | None = None,
) -> float | None:
    total = 0.0
    found = False
    for series in metric_series(metrics, metric_names):
        if series_filter is not None and not series_filter(series):
            continue
        value = series_stat(series, preferred_keys)
        if value is None:
            continue
        total += value
        found = True
    return total if found else None


def gauge_stat(
    metrics: dict[str, dict[str, Any]],
    metric_names: str | list[str],
    *,
    preferred_keys: tuple[str, ...] = ("max", "avg", "total"),
    combine: str = "max",
    series_filter: Callable[[dict[str, Any]], bool] | None = None,
) -> float | None:
    values: list[float] = []
    for series in metric_series(metrics, metric_names):
        if series_filter is not None and not series_filter(series):
            continue
        value = series_stat(series, preferred_keys)
        if value is not None:
            values.append(value)
    if not values:
        return None
    if combine == "avg":
        return statistics.mean(values)
    if combine == "sum":
        return sum(values)
    return max(values)


def label_value(series: dict[str, Any], key: str) -> str | None:
    labels = series.get("labels")
    if not isinstance(labels, dict):
        return None
    value = labels.get(key)
    if value is None:
        return None
    return str(value)


def label_equals(key: str, value: str) -> Callable[[dict[str, Any]], bool]:
    return lambda series: label_value(series, key) == value


def sum_by_label(
    metrics: dict[str, dict[str, Any]],
    metric_names: str | list[str],
    label_keys: str | list[str],
    *,
    preferred_keys: tuple[str, ...] = ("total", "sum", "max", "avg"),
) -> dict[str, float]:
    keys = [label_keys] if isinstance(label_keys, str) else label_keys
    out: dict[str, float] = {}
    for series in metric_series(metrics, metric_names):
        labels = series.get("labels")
        if not isinstance(labels, dict):
            continue
        label = None
        for key in keys:
            raw = labels.get(key)
            if raw is not None:
                label = str(raw)
                break
        if label is None:
            continue
        value = series_stat(series, preferred_keys)
        if value is None:
            continue
        out[label] = out.get(label, 0.0) + value
    return out
