"""Integrate GPU-board power over a measurement window from a telemetry CSV.

A self-contained port of InferenceX's single-node power integration, retained here so the
H3 harness validates the telemetry it records without the end-to-end project on its path.
Energy is the per-device trapezoidal integral with linear interpolation at both window
boundaries; a stream is valid only when every device brackets the window with no sampling
gap above the limit.
"""

from __future__ import annotations

import bisect
import csv
import itertools
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

UTC = timezone.utc

_POWER_COL_RE = re.compile(r"power", re.IGNORECASE)
_POWER_EXCLUDE_RE = re.compile(r"limit|cap|max|min", re.IGNORECASE)
_TIMESTAMP_COL_RE = re.compile(r"time", re.IGNORECASE)
_GPU_INDEX_COL_RE = re.compile(r"^(index|gpu|gpu_id|gpu_index|card|device)$", re.IGNORECASE)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

INTEGRATION_METHOD = "per_device_trapezoidal_with_linear_boundary_interpolation"
_DEFAULT_MAX_SAMPLE_GAP_S = 3.0


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
        (timestamp, power) for timestamp, power in samples if start_unix < timestamp < end_unix
    )
    clipped.append((end_unix, end_power))

    energy_j = 0.0
    for (left_time, left_power), (right_time, right_power) in itertools.pairwise(clipped):
        energy_j += (right_time - left_time) * (left_power + right_power) / 2.0
    return energy_j

def _percentile_total_power(
    device_samples: list[list[tuple[float, float]]],
    *,
    start_unix: float,
    end_unix: float,
    quantile: float,
) -> float:
    """Time-weighted quantile of synchronized fleet power with linear interpolation.

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
        for (left_t, left_p), (right_t, right_p) in itertools.pairwise(clipped):
            slope = (right_p - left_p) / (right_t - left_t)
            slope_changes[left_t] = slope_changes.get(left_t, 0.0) + slope
            slope_changes[right_t] = slope_changes.get(right_t, 0.0) - slope

    segments: list[tuple[float, float, float]] = []
    slope = 0.0
    times = sorted(slope_changes)
    for left_t, right_t in itertools.pairwise(times):
        slope += slope_changes[left_t]
        next_power = total_power + slope * (right_t - left_t)
        segments.append(
            (
                min(total_power, next_power),
                max(total_power, next_power),
                right_t - left_t,
            )
        )
        total_power = next_power
    lower = min(low for low, _, _ in segments)
    upper = max(high for _, high, _ in segments)
    target_time = (end_unix - start_unix) * quantile
    for _ in range(60):
        value = lower + (upper - lower) / 2
        time_below = sum(
            duration
            if value >= high
            else duration * (value - low) / (high - low)
            if value > low
            else 0.0
            for low, high, duration in segments
        )
        if time_below >= target_time:
            upper = value
        else:
            lower = value
    return upper

@dataclass(frozen=True)
class PowerIntegration:
    """Auditable result of integrating one single-node telemetry stream."""

    power_valid: bool
    invalid_reasons: tuple[str, ...]
    expected_num_gpus: int | None
    observed_gpu_ids: tuple[str, ...]
    per_gpu_sample_counts: dict[str, int]
    per_gpu_max_sample_gap_s: dict[str, float]
    per_gpu_energy_j: dict[str, float]
    device_issues: dict[str, list[str]]
    # Rows in the ingest band but outside the formal window whose power was
    # missing/N-A/non-finite/<=0, skipped and counted per GPU; "unknown"
    # buckets rows without a GPU identity.
    boundary_degenerate_rows: dict[str, int] = field(default_factory=dict)
    avg_power_w: float | None = None
    p75_power_w: float | None = None
    p75_total_gpu_power_w: float | None = None
    p90_power_w: float | None = None
    p90_total_gpu_power_w: float | None = None
    avg_total_gpu_power_w: float | None = None
    total_gpu_energy_j: float | None = None

    @property
    def observed_num_gpus(self) -> int:
        return len(self.observed_gpu_ids)

def _parse_timestamp(value: str, *, naive_timezone: timezone | None = None) -> float | None:
    """Best-effort timestamp parse to Unix epoch seconds (local wall clock).

    Handles the formats observed in practice:
      - nvidia-smi: "2025/01/15 12:34:56.789" (local time, no TZ)
      - amd-smi:    ISO 8601 "2025-01-15T12:34:56.789" or epoch seconds
      - Plain numeric epoch (int or float, s or ms)
    """
    value = value.strip()
    if not value:
        return None
    # Plain epoch number — accept both seconds and milliseconds.
    if _NUMBER_RE.fullmatch(value):
        n = float(value)
        return n / 1000.0 if n > 1e12 else n
    # nvidia-smi: "YYYY/MM/DD HH:MM:SS.ffffff"
    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=naive_timezone).timestamp()
        except ValueError:
            pass
    # ISO 8601 (amd-smi variants). fromisoformat tolerates 'T' or space separator
    # in Python 3.11+; older versions need 'T'.
    iso_value = value.replace(" ", "T", 1) if " " in value and "T" not in value else value
    try:
        dt = datetime.fromisoformat(iso_value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        # Treat naive timestamps as local time (matches nvidia-smi convention).
        return dt.replace(tzinfo=naive_timezone).timestamp()
    return dt.astimezone(UTC).timestamp()

def _parse_power(value: str) -> float | None:
    """Extract the first numeric value from a power cell.

    nvidia-smi formats power as "412.34 W"; some configurations report
    "[N/A]" when power capping is disabled. AMD reports a bare number.
    """
    value = value.strip()
    if not value or value.lower() in {"[n/a]", "n/a", "na"}:
        return None
    m = _NUMBER_RE.search(value)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None

def _detect_columns(header: list[str]) -> tuple[str | None, str | None, str | None]:
    """Return (timestamp_col, power_col, gpu_index_col) from a CSV header.

    Power column: contains "power" and not "limit"/"cap"/"max"/"min".
    Timestamp column: contains "time".
    GPU index column: optional — used to count distinct GPUs per sample.
    """
    timestamp_col = next((c for c in header if _TIMESTAMP_COL_RE.search(c)), None)
    power_col = next(
        (c for c in header if _POWER_COL_RE.search(c) and not _POWER_EXCLUDE_RE.search(c)),
        None,
    )
    gpu_col = next((c for c in header if _GPU_INDEX_COL_RE.match(c.strip())), None)
    return timestamp_col, power_col, gpu_col

def _telemetry_timezone(csv_path: Path) -> timezone | None:
    context = csv_path.with_name(f"{csv_path.stem}_context.json")
    if not context.is_file():
        return None
    payload = json.loads(context.read_text())
    if not isinstance(payload, dict) or payload.get("timestamp_timezone") != "UTC":
        raise ValueError("unsupported_telemetry_timezone")
    return UTC

def _gpu_sort_key(gpu_id: str) -> tuple[int, int | str]:
    """Sort numeric GPU identities before arbitrary string identities."""
    return (0, int(gpu_id)) if gpu_id.isdigit() else (1, gpu_id)

def _empty_integration(
    *,
    expected_num_gpus: int | None,
    reasons: list[str],
    boundary_degenerate_rows: dict[str, int] | None = None,
) -> PowerIntegration:
    """Build an invalid integration result when no device data is available."""
    return PowerIntegration(
        power_valid=False,
        invalid_reasons=tuple(reasons),
        expected_num_gpus=expected_num_gpus,
        observed_gpu_ids=(),
        per_gpu_sample_counts={},
        per_gpu_max_sample_gap_s={},
        per_gpu_energy_j={},
        device_issues={},
        boundary_degenerate_rows=boundary_degenerate_rows or {},
    )

def integrate_power(
    csv_path: Path,
    *,
    start_unix: float,
    end_unix: float,
    expected_num_gpus: int | None = None,
    max_sample_gap_s: float = _DEFAULT_MAX_SAMPLE_GAP_S,
) -> PowerIntegration:
    """Validate and integrate per-device GPU power over the formal window.

    A valid stream must contain parseable timestamps and power values, expose
    stable GPU identities, match the expected topology when supplied, bracket
    both window boundaries for every device, and have no in-window sampling gap
    larger than ``max_sample_gap_s``.
    """
    reasons: list[str] = []
    if not math.isfinite(start_unix) or not math.isfinite(end_unix) or end_unix <= start_unix:
        return _empty_integration(
            expected_num_gpus=expected_num_gpus,
            reasons=["invalid_benchmark_window"],
        )
    if expected_num_gpus is not None and expected_num_gpus <= 0:
        _append_reason(reasons, "invalid_expected_gpu_count")
    if not math.isfinite(max_sample_gap_s) or max_sample_gap_s <= 0:
        return _empty_integration(
            expected_num_gpus=expected_num_gpus,
            reasons=["invalid_max_sample_gap"],
        )

    if not csv_path.is_file():
        _append_reason(reasons, "telemetry_file_missing")
        return _empty_integration(
            expected_num_gpus=expected_num_gpus,
            reasons=reasons,
        )
    try:
        if csv_path.stat().st_size == 0:
            _append_reason(reasons, "telemetry_file_empty")
            return _empty_integration(
                expected_num_gpus=expected_num_gpus,
                reasons=reasons,
            )
    except OSError:
        _append_reason(reasons, "telemetry_file_unreadable")
        return _empty_integration(
            expected_num_gpus=expected_num_gpus,
            reasons=reasons,
        )

    # Values are first grouped by GPU and exact timestamp. Some SMI versions
    # expose timestamps at lower resolution than their sampling cadence, so
    # duplicate-timestamp readings are averaged rather than treated as corrupt.
    raw_samples: dict[str, dict[float, list[float]]] = {}
    boundary_degenerate: dict[str, int] = {}
    saw_missing_gpu_identity = False
    try:
        timestamp_timezone = _telemetry_timezone(csv_path)
        with csv_path.open("r", newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            header = [column.strip() for column in (reader.fieldnames or [])]
            reader.fieldnames = header
            timestamp_col, power_col, gpu_col = _detect_columns(header)
            missing_required_columns = False
            if not timestamp_col:
                _append_reason(reasons, "timestamp_column_missing")
                missing_required_columns = True
            if not power_col:
                _append_reason(reasons, "power_column_missing")
                missing_required_columns = True
            if not gpu_col:
                _append_reason(reasons, "gpu_identity_column_missing")
                missing_required_columns = True
            if missing_required_columns:
                return _empty_integration(
                    expected_num_gpus=expected_num_gpus,
                    reasons=reasons,
                )

            for row in reader:
                timestamp = _parse_timestamp(
                    (row.get(timestamp_col) or "").strip(),
                    naive_timezone=timestamp_timezone,
                )
                if timestamp is None or not math.isfinite(timestamp):
                    _append_reason(reasons, "invalid_timestamp_sample")
                    continue
                # Note (wenyao): corrupt warmup/eval rows are skipped only when
                # their parseable timestamp proves they cannot affect the
                # window. An unparseable timestamp has unknown position, so it
                # was rejected above instead of skipped.
                if (
                    timestamp < start_unix - max_sample_gap_s
                    or timestamp > end_unix + max_sample_gap_s
                ):
                    continue

                power = _parse_power((row.get(power_col) or "").strip())
                gpu_id = (row.get(gpu_col) or "").strip()
                if (power is None or not math.isfinite(power) or power <= 0.0) and (
                    timestamp < start_unix or timestamp > end_unix
                ):
                    # SMI teardown rows can carry N/A or 0 W cells: outside the
                    # formal window they are counted, never used to satisfy
                    # bracketing or to poison in-window validity.
                    key = gpu_id or "unknown"
                    boundary_degenerate[key] = boundary_degenerate.get(key, 0) + 1
                    continue
                if power is None:
                    _append_reason(reasons, "invalid_power_sample")
                    continue
                if not math.isfinite(power) or power < 0:
                    _append_reason(reasons, "invalid_power_sample")
                    continue
                if not gpu_id:
                    saw_missing_gpu_identity = True
                    continue
                values = raw_samples.setdefault(gpu_id, {}).setdefault(timestamp, [])
                values.append(power)
    except (OSError, csv.Error, ValueError):
        _append_reason(reasons, "telemetry_file_unreadable")
        return _empty_integration(
            expected_num_gpus=expected_num_gpus,
            reasons=reasons,
            boundary_degenerate_rows=boundary_degenerate,
        )

    if saw_missing_gpu_identity:
        _append_reason(reasons, "gpu_identity_missing")
    if not raw_samples:
        _append_reason(reasons, "no_usable_power_samples")
        return _empty_integration(
            expected_num_gpus=expected_num_gpus,
            reasons=reasons,
            boundary_degenerate_rows=boundary_degenerate,
        )

    observed_gpu_ids = tuple(sorted(raw_samples, key=_gpu_sort_key))
    if (
        expected_num_gpus is not None
        and expected_num_gpus > 0
        and len(observed_gpu_ids) != expected_num_gpus
    ):
        _append_reason(reasons, "expected_gpu_count_mismatch")

    per_gpu_sample_counts: dict[str, int] = {}
    per_gpu_max_sample_gap_s: dict[str, float] = {}
    per_gpu_energy_j: dict[str, float] = {}
    device_issues: dict[str, list[str]] = {}
    device_samples: list[list[tuple[float, float]]] = []

    for gpu_id in observed_gpu_ids:
        timestamp_values = raw_samples[gpu_id]
        samples = sorted(
            (timestamp, mean(values)) for timestamp, values in timestamp_values.items()
        )
        device_samples.append(samples)
        per_gpu_sample_counts[gpu_id] = len(samples)
        issues: list[str] = []

        if len(samples) < 2:
            issues.append("insufficient_power_samples")
            _append_reason(reasons, "insufficient_power_samples")
        elif samples[0][0] > start_unix or samples[-1][0] < end_unix:
            issues.append("benchmark_window_not_bracketed")
            _append_reason(reasons, "benchmark_window_not_bracketed")
        else:
            times = [timestamp for timestamp, _ in samples]
            left_index = bisect.bisect_right(times, start_unix) - 1
            right_index = bisect.bisect_left(times, end_unix)
            relevant = samples[left_index : right_index + 1]
            gaps = [right[0] - left[0] for left, right in itertools.pairwise(relevant)]
            max_gap = max(gaps, default=0.0)
            per_gpu_max_sample_gap_s[gpu_id] = max_gap
            if max_gap > max_sample_gap_s:
                issues.append("sampling_gap_exceeded")
                _append_reason(reasons, "sampling_gap_exceeded")
            per_gpu_energy_j[gpu_id] = _integrate_device(
                samples,
                start_unix=start_unix,
                end_unix=end_unix,
            )

        if issues:
            device_issues[gpu_id] = issues

    duration_s = end_unix - start_unix
    total_gpu_energy_j: float | None = None
    avg_total_gpu_power_w: float | None = None
    avg_power_w: float | None = None
    if len(per_gpu_energy_j) == len(observed_gpu_ids):
        total_gpu_energy_j = sum(per_gpu_energy_j.values())
        avg_total_gpu_power_w = total_gpu_energy_j / duration_s
        avg_power_w = avg_total_gpu_power_w / len(observed_gpu_ids)

    p75_total = (
        None
        if reasons
        else _percentile_total_power(
            device_samples, start_unix=start_unix, end_unix=end_unix, quantile=0.75
        )
    )
    p90_total = (
        None
        if reasons
        else _percentile_total_power(
            device_samples, start_unix=start_unix, end_unix=end_unix, quantile=0.9
        )
    )
    return PowerIntegration(
        power_valid=not reasons,
        invalid_reasons=tuple(reasons),
        expected_num_gpus=expected_num_gpus,
        observed_gpu_ids=observed_gpu_ids,
        per_gpu_sample_counts=per_gpu_sample_counts,
        per_gpu_max_sample_gap_s=per_gpu_max_sample_gap_s,
        per_gpu_energy_j=per_gpu_energy_j,
        device_issues=device_issues,
        boundary_degenerate_rows=boundary_degenerate,
        avg_power_w=avg_power_w,
        p75_power_w=p75_total / len(observed_gpu_ids) if p75_total is not None else None,
        p75_total_gpu_power_w=p75_total,
        p90_power_w=p90_total / len(observed_gpu_ids) if p90_total is not None else None,
        p90_total_gpu_power_w=p90_total,
        avg_total_gpu_power_w=avg_total_gpu_power_w,
        total_gpu_energy_j=total_gpu_energy_j,
    )
