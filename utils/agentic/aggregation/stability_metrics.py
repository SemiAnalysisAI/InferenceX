"""Within-run stability diagnostics for AgentX request records."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from .aggregation_common import percentile, to_float


OBSERVED_WINDOW_SECONDS = 600.0
CONVERGENCE_CHECKPOINT_SECONDS = 300.0
CONVERGENCE_TOLERANCE_RATIO = 0.05
CONVERGENCE_MIN_CONFIRMATION_SECONDS = 1200.0
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


def _convergence_summary(
    checkpoints: list[dict[str, float | int]],
    *,
    tolerance_ratio: float,
    min_confirmation_seconds: float,
) -> dict[str, float | int] | None:
    """Return the earliest retrospectively stable suffix of a checkpoint series.

    The ratio test is performed in log space. This makes the tolerance
    symmetric under reciprocals: a latency series and its interactivity
    reciprocal stabilize at the same checkpoint.
    """
    if not checkpoints:
        return None

    final_value = float(checkpoints[-1]["value"])
    horizon_seconds = float(checkpoints[-1]["seconds"])
    if not math.isfinite(final_value) or final_value <= 0:
        return None

    log_tolerance = math.log1p(tolerance_ratio)
    boundary_epsilon = 1e-12
    for index, checkpoint in enumerate(checkpoints):
        checkpoint_seconds = float(checkpoint["seconds"])
        if horizon_seconds - checkpoint_seconds < min_confirmation_seconds:
            continue

        suffix = checkpoints[index:]
        values = [float(item["value"]) for item in suffix]
        if any(not math.isfinite(value) or value <= 0 for value in values):
            continue
        if any(
            abs(math.log(value / final_value)) > log_tolerance + boundary_epsilon
            for value in values
        ):
            continue

        return {
            "time_seconds": checkpoint["seconds"],
            "requests": checkpoint["requests"],
            "min": min(values),
            "max": max(values),
            "max_relative_deviation": max(
                abs(value / final_value - 1.0) for value in values
            ),
        }
    return None


def _cumulative_convergence(
    timed: list[tuple[int, dict[str, Any]]],
    *,
    horizon_seconds: float,
    checkpoint_seconds: float,
    tolerance_ratio: float,
    min_confirmation_seconds: float,
) -> dict[str, dict[str, dict[str, float | int] | None]]:
    """Compute retrospective convergence from cumulative request prefixes."""
    output: dict[str, dict[str, dict[str, float | int] | None]] = {
        "ttft": {f"p{p}": None for p in _PERCENTILES},
        "e2el": {f"p{p}": None for p in _PERCENTILES},
        "intvty": {f"p{p}": None for p in _PERCENTILES},
    }
    if not timed or horizon_seconds <= 0:
        return output

    origin_ns = min(timestamp for timestamp, _ in timed)
    sorted_timed = sorted(timed, key=lambda item: item[0])
    metric_keys = {
        "ttft": "time_to_first_token",
        "e2el": "request_latency",
        "itl": "inter_token_latency",
    }
    values: dict[str, list[float]] = {name: [] for name in metric_keys}
    series: dict[str, dict[int, list[dict[str, float | int]]]] = {
        "ttft": {p: [] for p in _PERCENTILES},
        "e2el": {p: [] for p in _PERCENTILES},
        "intvty": {p: [] for p in _PERCENTILES},
    }

    cursor = 0
    checkpoint_count = int(horizon_seconds // checkpoint_seconds)
    for checkpoint_index in range(1, checkpoint_count + 1):
        elapsed_seconds = checkpoint_index * checkpoint_seconds
        cutoff_ns = origin_ns + elapsed_seconds * 1e9
        while cursor < len(sorted_timed) and sorted_timed[cursor][0] < cutoff_ns:
            timestamp, record = sorted_timed[cursor]
            if timestamp >= origin_ns:
                for name, key in metric_keys.items():
                    value_ms = _metric_value(record, key)
                    if value_ms is not None and value_ms > 0:
                        values[name].append(value_ms / 1000.0)
            cursor += 1

        for p in _PERCENTILES:
            for name in ("ttft", "e2el"):
                if values[name]:
                    series[name][p].append(
                        {
                            "seconds": elapsed_seconds,
                            "requests": len(values[name]),
                            "value": percentile(values[name], p),
                        }
                    )
            if values["itl"]:
                itl_percentile = percentile(values["itl"], p)
                if itl_percentile > 0:
                    series["intvty"][p].append(
                        {
                            "seconds": elapsed_seconds,
                            "requests": len(values["itl"]),
                            "value": 1.0 / itl_percentile,
                        }
                    )

    for name in output:
        for p in _PERCENTILES:
            checkpoint_series = series[name][p]
            output[name][f"p{p}"] = _convergence_summary(
                checkpoint_series,
                tolerance_ratio=tolerance_ratio,
                min_confirmation_seconds=min_confirmation_seconds,
            )
    return output


def compute_stability_metrics(
    records: list[dict[str, Any]],
    *,
    benchmark_duration_s: float | None,
    window_seconds: float = OBSERVED_WINDOW_SECONDS,
    checkpoint_seconds: float = CONVERGENCE_CHECKPOINT_SECONDS,
    tolerance_ratio: float = CONVERGENCE_TOLERANCE_RATIO,
    min_confirmation_seconds: float = CONVERGENCE_MIN_CONFIRMATION_SECONDS,
) -> dict[str, Any]:
    """Summarize workload coverage, window drift, and cumulative convergence.

    These are retrospective diagnostics for one run. They deliberately do not
    claim to be confidence intervals or estimates of rerun reproducibility.
    """
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if checkpoint_seconds <= 0:
        raise ValueError("checkpoint_seconds must be positive")
    if tolerance_ratio < 0:
        raise ValueError("tolerance_ratio must be non-negative")
    if min_confirmation_seconds < 0:
        raise ValueError("min_confirmation_seconds must be non-negative")

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
    convergence_horizon = (
        math.floor(configured_duration / checkpoint_seconds) * checkpoint_seconds
        if configured_duration is not None and configured_duration > 0
        else 0.0
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
        "convergence": {
            "checkpoint_seconds": checkpoint_seconds,
            "tolerance_ratio": tolerance_ratio,
            "min_confirmation_seconds": min_confirmation_seconds,
            "horizon_seconds": convergence_horizon,
            "metrics": _cumulative_convergence(
                timed,
                horizon_seconds=convergence_horizon,
                checkpoint_seconds=checkpoint_seconds,
                tolerance_ratio=tolerance_ratio,
                min_confirmation_seconds=min_confirmation_seconds,
            ),
        },
    }
