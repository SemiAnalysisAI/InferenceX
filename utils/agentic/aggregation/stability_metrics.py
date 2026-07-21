"""Descriptive fixed-window diagnostics for AgentX request records."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from .aggregation_common import percentile, to_float


OBSERVED_WINDOW_SECONDS = 600.0
_PERCENTILES = (75, 90)


def _metric_value(record: dict[str, Any], key: str) -> float | None:
    metric = record.get("metrics", {}).get(key)
    if isinstance(metric, dict):
        metric = metric.get("value")
    return to_float(metric)


def _timestamp_ns(record: dict[str, Any]) -> int | None:
    metadata = record.get("metadata", {})
    value = metadata.get("credit_issued_ns")
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
    ):
        value = metadata.get("request_start_ns")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if not math.isfinite(value):
        return None
    return int(value)


def _source_trajectory(record: dict[str, Any]) -> str:
    metadata = record.get("metadata", {})
    source = metadata.get("source_trace_id") or metadata.get("conversation_id")
    return str(source) if source not in (None, "") else "unknown"


def _range(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {"min": min(values), "max": max(values)}


def _latency_ranges(
    blocks: dict[int, list[dict[str, Any]]],
) -> dict[str, dict[str, dict[str, float] | None]]:
    metric_keys = {
        "ttft": "time_to_first_token",
        "e2el": "request_latency",
        "itl": "inter_token_latency",
    }
    block_percentiles: dict[str, dict[int, list[float]]] = {
        name: {p: [] for p in _PERCENTILES} for name in metric_keys
    }

    for records in blocks.values():
        for name, key in metric_keys.items():
            values_ms = [
                value
                for record in records
                if (value := _metric_value(record, key)) is not None and value > 0
            ]
            if not values_ms:
                continue
            values_s = [value / 1000.0 for value in values_ms]
            for p in _PERCENTILES:
                block_percentiles[name][p].append(percentile(values_s, p))

    output: dict[str, dict[str, dict[str, float] | None]] = {
        "ttft": {},
        "e2el": {},
        "intvty": {},
    }
    for p in _PERCENTILES:
        output["ttft"][f"p{p}"] = _range(block_percentiles["ttft"][p])
        output["e2el"][f"p{p}"] = _range(block_percentiles["e2el"][p])
        itl = block_percentiles["itl"][p]
        output["intvty"][f"p{p}"] = _range([1.0 / value for value in itl if value > 0])
    return output


def compute_stability_metrics(
    records: list[dict[str, Any]],
    *,
    benchmark_duration_s: float | None,
    window_seconds: float = OBSERVED_WINDOW_SECONDS,
) -> dict[str, Any]:
    """Summarize realized workload coverage and non-overlapping time windows.

    These are descriptive diagnostics for one run. They deliberately do not
    claim to be confidence intervals or estimates of rerun reproducibility.
    """
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")

    timed: list[tuple[int, dict[str, Any]]] = []
    for record in records:
        timestamp = _timestamp_ns(record)
        if timestamp is not None:
            timed.append((timestamp, record))
    trajectory_counts = Counter(_source_trajectory(record) for record in records)
    n_requests = sum(trajectory_counts.values())
    squared = sum(count * count for count in trajectory_counts.values())
    kish_effective = n_requests * n_requests / squared if squared > 0 else 0.0
    largest_share = (
        max(trajectory_counts.values(), default=0) / n_requests if n_requests else 0.0
    )

    configured_duration = to_float(benchmark_duration_s)
    expected_windows = (
        int(configured_duration // window_seconds)
        if configured_duration is not None and configured_duration > 0
        else 0
    )

    blocks: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if timed and expected_windows > 0:
        origin_ns = min(timestamp for timestamp, _ in timed)
        window_ns = window_seconds * 1e9
        for timestamp, record in timed:
            block = int((timestamp - origin_ns) // window_ns)
            if 0 <= block < expected_windows:
                blocks[block].append(record)

    block_sizes = [len(block) for block in blocks.values()]
    return {
        "window_seconds": window_seconds,
        "expected_window_count": expected_windows,
        "observed_window_count": len(blocks),
        "min_window_requests": min(block_sizes) if block_sizes else 0,
        "root_trajectory_count": len(trajectory_counts),
        "root_trajectory_kish_effective_count": kish_effective,
        "root_trajectory_largest_share": largest_share,
        "observed_ranges": _latency_ranges(blocks) if len(blocks) >= 2 else {},
    }
