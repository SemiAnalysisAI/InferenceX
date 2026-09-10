"""Shared benchmark, integration, and artifact primitives for power validators."""

from __future__ import annotations

import bisect
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from . import POWER_METRIC_SCHEMA_VERSION, with_power_metrics


@dataclass(frozen=True)
class BenchmarkData:
    """Raw benchmark fields required for energy normalization."""

    start_unix: float
    end_unix: float
    reported_duration_s: float
    completed: int
    total_input_tokens: int
    total_output_tokens: int

    @property
    def integration_duration_s(self) -> float:
        return self.end_unix - self.start_unix


def _append_reason(reasons: list[str], reason: str) -> None:
    """Append a validation reason once while preserving discovery order."""
    if reason not in reasons:
        reasons.append(reason)


def _interpolate_power(samples: list[tuple[float, float]], timestamp: float) -> float:
    """Linearly interpolate power at a timestamp bracketed by ``samples``."""
    times = [sample_time for sample_time, _ in samples]
    right_index = bisect.bisect_left(times, timestamp)
    if right_index < len(samples) and math.isclose(
        samples[right_index][0], timestamp, rel_tol=0.0, abs_tol=1e-9
    ):
        return samples[right_index][1]
    left_index = right_index - 1
    if left_index < 0 or right_index >= len(samples):
        raise ValueError("timestamp is not bracketed")
    left_time, left_power = samples[left_index]
    right_time, right_power = samples[right_index]
    fraction = (timestamp - left_time) / (right_time - left_time)
    return left_power + fraction * (right_power - left_power)


def _integrate_device(
    samples: list[tuple[float, float]],
    *,
    start_unix: float,
    end_unix: float,
) -> float:
    """Integrate one device over ``[start_unix, end_unix]``."""
    start_power = _interpolate_power(samples, start_unix)
    end_power = _interpolate_power(samples, end_unix)
    clipped = [(start_unix, start_power)]
    clipped.extend(
        (timestamp, power)
        for timestamp, power in samples
        if start_unix < timestamp < end_unix
    )
    clipped.append((end_unix, end_power))

    energy_j = 0.0
    for (left_time, left_power), (right_time, right_power) in zip(
        clipped, clipped[1:]
    ):
        energy_j += (right_time - left_time) * (left_power + right_power) / 2.0
    return energy_j


def _p90_total_power(
    device_samples: list[list[tuple[float, float]]],
    *,
    start_unix: float,
    end_unix: float,
) -> float:
    """Time-weighted P90 of synchronized fleet power with linear interpolation.

    Sum device curves before taking the percentile. Each linear segment's
    distribution is uniform over its power range, weighted by elapsed time;
    constant segments contribute a point mass. Sampling cadence cannot bias
    the result. Call only after all streams and their shared window validate.
    """
    slope_changes: dict[float, float] = {start_unix: 0.0, end_unix: 0.0}
    total_power = 0.0
    for samples in device_samples:
        first = _interpolate_power(samples, start_unix)
        total_power += first
        clipped = [(start_unix, first)]
        clipped.extend((t, p) for t, p in samples if start_unix < t < end_unix)
        clipped.append((end_unix, _interpolate_power(samples, end_unix)))
        for (left_t, left_p), (right_t, right_p) in zip(clipped, clipped[1:]):
            slope = (right_p - left_p) / (right_t - left_t)
            slope_changes[left_t] = slope_changes.get(left_t, 0.0) + slope
            slope_changes[right_t] = slope_changes.get(right_t, 0.0) - slope

    # The union of all device timestamps gives the exact knots of their sum.
    segments: list[tuple[float, float, float]] = []
    slope = 0.0
    times = sorted(slope_changes)
    for left_t, right_t in zip(times, times[1:]):
        slope += slope_changes[left_t]
        next_power = total_power + slope * (right_t - left_t)
        segments.append((
            min(total_power, next_power), max(total_power, next_power), right_t - left_t
        ))
        total_power = next_power
    lower = min(low for low, _, _ in segments)
    upper = max(high for _, high, _ in segments)
    target_time = (end_unix - start_unix) * 0.9
    # Bisection includes point masses without averaging device percentiles.
    for _ in range(60):
        value = lower + (upper - lower) / 2
        time_below = sum(
            duration if value >= high
            else duration * (value - low) / (high - low) if value > low
            else 0.0
            for low, high, duration in segments
        )
        if time_below >= target_time:
            upper = value
        else:
            lower = value
    return upper


def _load_benchmark_data(
    bench_result_path: Path,
) -> tuple[BenchmarkData | None, list[str]]:
    """Load the strict energy-normalization contract from raw benchmark JSON."""
    try:
        bench = json.loads(bench_result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, ["invalid_benchmark_result"]

    start = bench.get("benchmark_start_time_unix")
    end = bench.get("benchmark_end_time_unix")
    duration = bench.get("duration")
    numeric_window = all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in (start, end, duration)
    )
    if not numeric_window:
        return None, ["invalid_benchmark_window"]

    start = float(start)
    end = float(end)
    duration = float(duration)
    if (
        not all(math.isfinite(value) for value in (start, end, duration))
        or end <= start
        or duration <= 0
    ):
        return None, ["invalid_benchmark_window"]

    reasons: list[str] = []
    integration_duration = end - start
    duration_tolerance = max(0.5, integration_duration * 0.01)
    if abs(duration - integration_duration) > duration_tolerance:
        _append_reason(reasons, "benchmark_duration_mismatch")

    completed = bench.get("completed")
    if not isinstance(completed, int) or isinstance(completed, bool) or completed <= 0:
        _append_reason(reasons, "invalid_successful_query_count")
        completed = 0

    total_input = bench.get("total_input_tokens")
    if not isinstance(total_input, int) or isinstance(total_input, bool) or total_input <= 0:
        _append_reason(reasons, "invalid_input_token_count")
        total_input = 0

    total_output = bench.get("total_output_tokens")
    if not isinstance(total_output, int) or isinstance(total_output, bool) or total_output <= 0:
        _append_reason(reasons, "invalid_output_token_count")
        total_output = 0

    return (
        BenchmarkData(
            start_unix=start,
            end_unix=end,
            reported_duration_s=duration,
            completed=completed,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
        ),
        reasons,
    )


def _write_json_atomic(path: Path, data: dict) -> None:
    """Atomically replace a JSON artifact with formatted UTF-8 content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def benchmark_window_payload(benchmark: BenchmarkData | None) -> dict[str, float | int] | None:
    """Serialize the common benchmark-window contract for validation sidecars."""
    if benchmark is None:
        return None
    return {
        "start_time_unix": benchmark.start_unix,
        "end_time_unix": benchmark.end_unix,
        "reported_duration_s": benchmark.reported_duration_s,
        "integration_duration_s": benchmark.integration_duration_s,
        "completed": benchmark.completed,
        "total_input_tokens": benchmark.total_input_tokens,
        "total_output_tokens": benchmark.total_output_tokens,
    }


def audit_metrics(metrics: Mapping[str, float | None]) -> dict[str, float]:
    """Keep finite sidecar metrics at audit precision, independent of display units."""
    return {
        key: round(value, 6)
        for key, value in metrics.items()
        if value is not None and math.isfinite(value)
    }


def patch_power_metrics(
    path: Path,
    *,
    metric_keys: Iterable[str],
    power_valid: bool,
    metrics: Mapping[str, float],
) -> None:
    """Validate replacement metrics before atomically updating an aggregate."""
    data = json.loads(path.read_text(encoding="utf-8"))
    data = with_power_metrics(
        data, metric_keys=metric_keys, schema_version=POWER_METRIC_SCHEMA_VERSION,
        power_valid=power_valid, metrics=metrics,
    )
    _write_json_atomic(path, data)
