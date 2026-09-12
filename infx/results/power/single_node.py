"""Validate and aggregate single-node GPU-board power into benchmark results.

The formal benchmark window comes from ``benchmark_serving.py``. Power is
integrated independently for every observed GPU with trapezoidal integration
and linear interpolation at the window boundaries. Unqualified energy metrics
therefore always describe the whole deployment, never a prefill/decode role.

Ordinary benchmark runs are best-effort: invalid telemetry is recorded in the
aggregate and a validation sidecar, but does not fail the benchmark. Power
studies can set ``REQUIRE_POWER=1`` to fail after those audit artifacts exist.
The aggregate carries numeric ``power_valid`` (1/0) for metric ingestion; the
sidecar is the canonical source for boolean validity and reason codes. Rows in
the ingest band but outside the formal window whose power is missing,
non-finite, or <= 0 are teardown noise: they are skipped and counted in the
sidecar's ``boundary_degenerate_rows`` instead of poisoning validity or faking
window bracketing.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from . import POWER_METRIC_SCHEMA_VERSION, WHOLE_METRIC_KEYS, with_power_metrics
from .common import (
    BenchmarkData,
    _append_reason,
    _integrate_device,
    _percentile_total_power,
    _interpolate_power,
    _load_benchmark_data,
    _write_json_atomic,
    audit_metrics,
    benchmark_window_payload,
    patch_power_metrics,
)

_POWER_COL_RE = re.compile(r"power", re.IGNORECASE)
_POWER_EXCLUDE_RE = re.compile(r"limit|cap|max|min", re.IGNORECASE)
_TIMESTAMP_COL_RE = re.compile(r"time", re.IGNORECASE)
_GPU_INDEX_COL_RE = re.compile(r"^(index|gpu|gpu_id|gpu_index|card|device)$", re.IGNORECASE)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

_INTEGRATION_METHOD = "per_device_trapezoidal_with_linear_boundary_interpolation"
_DEFAULT_MAX_SAMPLE_GAP_S = 3.0
_ACCUMULATOR_TOLERANCE = 0.05
_POWER_METRIC_KEYS = set(WHOLE_METRIC_KEYS)


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
    return dt.astimezone(timezone.utc).timestamp()


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
    return timezone.utc


def aggregate_power(
    csv_path: Path,
    start_unix: float,
    end_unix: float,
) -> tuple[float, int] | None:
    """Legacy arithmetic mean retained for callers of the pre-PR1 helper.

    Published PR1 metrics use :func:`integrate_power`; this compatibility API
    must not be used for energy calculations.
    """
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        return None
    if end_unix <= start_unix:
        return None

    try:
        timestamp_timezone = _telemetry_timezone(csv_path)
        with csv_path.open("r", newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            header = [c.strip() for c in (reader.fieldnames or [])]
            reader.fieldnames = header
            timestamp_col, power_col, gpu_col = _detect_columns(header)
            if not timestamp_col or not power_col:
                return None

            # Group power readings by sample timestamp so per-sample total power
            # (sum across GPUs) is computed correctly even if rows are interleaved.
            #
            # per_sample_row_count is the structural divisor: it's incremented for
            # every contributing row regardless of whether a GPU-index column was
            # detected. per_sample_gpus / gpu_keys are only populated when gpu_col
            # is present and provide the canonical num_gpus via distinct-id count.
            # When gpu_col is absent (vendor schema variant whose header doesn't
            # match _GPU_INDEX_COL_RE), we fall back to inferring num_gpus from
            # the modal row count per timestamp — assuming one row per GPU per
            # sample, which is what every SMI tool we've seen actually emits.
            per_sample_total: dict[float, float] = {}
            per_sample_row_count: dict[float, int] = {}
            per_sample_gpus: dict[float, set[str]] = {}
            gpu_keys: set[str] = set()

            for row in reader:
                ts_raw = (row.get(timestamp_col) or "").strip()
                pw_raw = (row.get(power_col) or "").strip()
                ts = _parse_timestamp(ts_raw, naive_timezone=timestamp_timezone)
                pw = _parse_power(pw_raw)
                if ts is None or pw is None:
                    continue
                if ts < start_unix or ts > end_unix:
                    continue
                # Bucket by sample timestamp (rounded to ms to absorb sub-ms drift).
                bucket = round(ts, 3)
                per_sample_total[bucket] = per_sample_total.get(bucket, 0.0) + pw
                per_sample_row_count[bucket] = per_sample_row_count.get(bucket, 0) + 1
                if gpu_col:
                    gpu_id = (row.get(gpu_col) or "").strip()
                    if gpu_id:
                        per_sample_gpus.setdefault(bucket, set()).add(gpu_id)
                        gpu_keys.add(gpu_id)
    except (OSError, csv.Error, ValueError):
        return None

    if not per_sample_total:
        return None

    # Per-sample divisor and overall num_gpus.
    # - If a GPU column was detected, trust distinct GPU IDs (correct for any
    #   sampling pattern, including hot-swap or partial visibility).
    # - Otherwise, infer from row count (one row per GPU per sample).
    if gpu_col and gpu_keys:
        num_gpus = len(gpu_keys)
        per_sample_mean_per_gpu = [
            total / max(len(per_sample_gpus.get(ts, ())), 1)
            for ts, total in per_sample_total.items()
        ]
    else:
        num_gpus = max(per_sample_row_count.values())
        per_sample_mean_per_gpu = [
            total / per_sample_row_count[ts] for ts, total in per_sample_total.items()
        ]
    return mean(per_sample_mean_per_gpu), num_gpus


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
    if (
        not math.isfinite(start_unix)
        or not math.isfinite(end_unix)
        or end_unix <= start_unix
    ):
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
                timestamp = _parse_timestamp((row.get(timestamp_col) or "").strip(), naive_timezone=timestamp_timezone)
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
            gaps = [
                right[0] - left[0] for left, right in zip(relevant, relevant[1:])
            ]
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

    p75_total = None if reasons else _percentile_total_power(
        device_samples, start_unix=start_unix, end_unix=end_unix, quantile=0.75
    )
    p90_total = None if reasons else _percentile_total_power(
        device_samples, start_unix=start_unix, end_unix=end_unix, quantile=0.9
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


def _read_energy_snapshot(path: Path) -> dict[str, float] | None:
    """Map GPU id to accumulator joules from one ``amd-smi metric -E`` snapshot."""
    with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        header = [column.strip() for column in (reader.fieldnames or [])]
        reader.fieldnames = header
        if "gpu" not in header or "total_energy_consumption" not in header:
            return None
        energy_by_gpu: dict[str, float] = {}
        for row in reader:
            gpu_id = (row.get("gpu") or "").strip()
            if not gpu_id:
                continue
            try:
                energy_j = float((row.get("total_energy_consumption") or "").strip())
            except ValueError:
                continue
            if math.isfinite(energy_j):
                energy_by_gpu[gpu_id] = energy_j
    return energy_by_gpu or None


def _stream_samples_by_gpu(csv_path: Path) -> dict[str, list[tuple[float, float]]]:
    """Group every parseable (timestamp, watt) sample per GPU, sorted in time."""
    samples: dict[str, list[tuple[float, float]]] = {}
    timestamp_timezone = _telemetry_timezone(csv_path)
    with csv_path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        header = [column.strip() for column in (reader.fieldnames or [])]
        reader.fieldnames = header
        timestamp_col, power_col, gpu_col = _detect_columns(header)
        if not timestamp_col or not power_col or not gpu_col:
            return {}
        for row in reader:
            timestamp = _parse_timestamp((row.get(timestamp_col) or "").strip(), naive_timezone=timestamp_timezone)
            power = _parse_power((row.get(power_col) or "").strip())
            gpu_id = (row.get(gpu_col) or "").strip()
            if timestamp is None or power is None or not gpu_id:
                continue
            if not math.isfinite(timestamp) or not math.isfinite(power):
                continue
            samples.setdefault(gpu_id, []).append((timestamp, power))
    return {gpu_id: sorted(values) for gpu_id, values in samples.items()}


def _trapezoid(samples: list[tuple[float, float]]) -> float:
    """Integrate consecutive (timestamp, watt) samples with no window clipping."""
    return float(
        sum(
            (right_time - left_time) * (left_power + right_power) / 2.0
            for (left_time, left_power), (right_time, right_power) in zip(
                samples, samples[1:]
            )
        )
    )


def cross_check_accumulator(csv_path: Path) -> dict | None:
    """Compare the hardware energy-accumulator delta with the full stream integral.

    The snapshots bracket the whole monitor lifetime rather than one benchmark
    window, so the comparison integrates the entire stream span: it validates the
    sampling instrument, not a window's metrics. Advisory only — the result never
    affects ``power_valid`` or the validation reason codes.
    """
    start_path = csv_path.with_name(f"{csv_path.stem}_energy_start.csv")
    end_path = csv_path.with_name(f"{csv_path.stem}_energy_end.csv")
    start_exists = start_path.is_file()
    end_exists = end_path.is_file()
    if not start_exists and not end_exists:
        return None
    if not start_exists:
        return {"available": False, "reason": "missing_start_snapshot"}
    if not end_exists:
        return {"available": False, "reason": "missing_end_snapshot"}

    start_energy = _read_energy_snapshot(start_path)
    if start_energy is None:
        return {"available": False, "reason": "unparseable_start_snapshot"}
    end_energy = _read_energy_snapshot(end_path)
    if end_energy is None:
        return {"available": False, "reason": "unparseable_end_snapshot"}

    matched_gpu_ids = sorted(set(start_energy) & set(end_energy), key=_gpu_sort_key)
    per_gpu_delta_j = {
        gpu_id: end_energy[gpu_id] - start_energy[gpu_id] for gpu_id in matched_gpu_ids
    }
    unmatched_gpus = sorted(set(start_energy) ^ set(end_energy), key=_gpu_sort_key)
    negative_delta_gpus = [
        gpu_id for gpu_id, delta in per_gpu_delta_j.items() if delta < 0
    ]

    stream = _stream_samples_by_gpu(csv_path)
    integrated_gpu_ids = sorted(set(stream) & set(per_gpu_delta_j), key=_gpu_sort_key)
    integrated_stream_j = float(
        sum(_trapezoid(stream[gpu_id]) for gpu_id in integrated_gpu_ids)
    )
    boundaries = [
        timestamp
        for gpu_id in integrated_gpu_ids
        for timestamp in (stream[gpu_id][0][0], stream[gpu_id][-1][0])
    ]
    stream_span_s = max(boundaries) - min(boundaries) if boundaries else 0.0

    accumulator_delta_j = float(sum(per_gpu_delta_j.values()))
    relative_error: float | None = None
    within_tolerance = False
    if accumulator_delta_j > 0:
        relative_error = (
            abs(accumulator_delta_j - integrated_stream_j) / accumulator_delta_j
        )
        within_tolerance = (
            not negative_delta_gpus and relative_error <= _ACCUMULATOR_TOLERANCE
        )

    return {
        "available": True,
        "accumulator_delta_j": accumulator_delta_j,
        "integrated_stream_j": integrated_stream_j,
        "relative_error": relative_error,
        "within_tolerance": within_tolerance,
        "tolerance": _ACCUMULATOR_TOLERANCE,
        "per_gpu_delta_j": per_gpu_delta_j,
        "integrated_gpu_ids": integrated_gpu_ids,
        "unmatched_gpus": unmatched_gpus,
        "negative_delta_gpus": negative_delta_gpus,
        "stream_span_s": stream_span_s,
    }


def patch_agg_result(
    agg_path: Path,
    avg_power_w: float,
    joules_per_output_token: float,
    joules_per_total_token: float,
) -> None:
    """Read the agg JSON, add the three power keys, and write it back atomically."""
    data = json.loads(agg_path.read_text(encoding="utf-8"))
    data["avg_power_w"] = round(avg_power_w, 3)
    data["joules_per_output_token"] = round(joules_per_output_token, 6)
    data["joules_per_total_token"] = round(joules_per_total_token, 6)
    tmp_path = agg_path.with_suffix(agg_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp_path.replace(agg_path)


def _derived_metrics(
    integration: PowerIntegration,
    benchmark: BenchmarkData,
) -> dict[str, float]:
    """Return whole-deployment energy metrics for a valid measurement."""
    if (
        integration.avg_power_w is None
        or integration.p75_power_w is None
        or integration.p75_total_gpu_power_w is None
        or integration.p90_power_w is None
        or integration.p90_total_gpu_power_w is None
        or integration.avg_total_gpu_power_w is None
        or integration.total_gpu_energy_j is None
    ):
        raise ValueError("valid power integration has incomplete power metrics")
    avg_power_w = integration.avg_power_w
    avg_total_gpu_power_w = integration.avg_total_gpu_power_w
    energy = integration.total_gpu_energy_j
    total_tokens = benchmark.total_input_tokens + benchmark.total_output_tokens
    return {
        "avg_power_w": avg_power_w,
        "p75_power_w": integration.p75_power_w,
        "p75_total_gpu_power_w": integration.p75_total_gpu_power_w,
        "p90_power_w": integration.p90_power_w,
        "p90_total_gpu_power_w": integration.p90_total_gpu_power_w,
        "avg_total_gpu_power_w": avg_total_gpu_power_w,
        "total_gpu_energy_j": energy,
        "joules_per_successful_query": energy / benchmark.completed,
        "joules_per_input_token": energy / benchmark.total_input_tokens,
        "joules_per_output_token": energy / benchmark.total_output_tokens,
        "joules_per_total_token": energy / total_tokens,
    }


def _patch_power_result(
    agg_path: Path,
    *,
    power_valid: bool,
    metrics: dict[str, float],
) -> None:
    """Replace aggregate power fields with one validated metric set."""
    patch_power_metrics(
        agg_path, metric_keys=_POWER_METRIC_KEYS, power_valid=power_valid, metrics=metrics,
    )


def _validation_payload(
    *,
    csv_path: Path,
    bench_result: Path,
    benchmark: BenchmarkData | None,
    integration: PowerIntegration,
    power_valid: bool,
    reasons: list[str],
    metrics: dict[str, float],
    accumulator_check: dict | None,
) -> dict:
    """Build the complete auditable power-validation sidecar payload."""
    return {
        "schema_version": 1,
        "power_valid": power_valid,
        "reasons": reasons,
        "telemetry_source": str(csv_path),
        "benchmark_result": str(bench_result),
        "benchmark_window": benchmark_window_payload(benchmark),
        "integration_method": _INTEGRATION_METHOD,
        "power_percentile_method": "time_weighted_synchronized_total_piecewise_linear",
        "expected_gpu_count": integration.expected_num_gpus,
        "observed_gpu_count": integration.observed_num_gpus,
        "observed_gpu_ids": list(integration.observed_gpu_ids),
        "per_gpu_sample_counts": integration.per_gpu_sample_counts,
        "per_gpu_max_sample_gap_s": integration.per_gpu_max_sample_gap_s,
        "per_gpu_energy_j": integration.per_gpu_energy_j,
        "device_issues": integration.device_issues,
        "boundary_degenerate_rows": integration.boundary_degenerate_rows,
        "accumulator_check": accumulator_check,
        "metrics": audit_metrics(metrics),
    }


def invalid_validation_payload(
    *,
    csv_path: Path,
    bench_result: Path,
    expected_num_gpus: int | None,
    reasons: list[str],
) -> dict:
    """Build a single-node audit when an adapter cannot supply a usable window."""
    return _validation_payload(
        csv_path=csv_path, bench_result=bench_result, benchmark=None,
        integration=_empty_integration(expected_num_gpus=expected_num_gpus, reasons=reasons),
        power_valid=False, reasons=reasons, metrics={}, accumulator_check=None,
    )


def run(
    csv_path: Path,
    bench_result: Path,
    agg_result: Path,
    *,
    expected_num_gpus: int | None = None,
    validation_result: Path | None = None,
    require_power: bool = False,
) -> int:
    """Aggregate power, always preserving validity, and optionally fail closed."""
    validation_result = validation_result or bench_result.with_name(
        f"power_validation_{bench_result.stem}.json"
    )
    benchmark, benchmark_reasons = _load_benchmark_data(bench_result)
    if benchmark is None:
        integration = _empty_integration(
            expected_num_gpus=expected_num_gpus,
            reasons=benchmark_reasons.copy(),
        )
    else:
        integration = integrate_power(
            csv_path,
            start_unix=benchmark.start_unix,
            end_unix=benchmark.end_unix,
            expected_num_gpus=expected_num_gpus,
        )

    reasons = benchmark_reasons.copy()
    for reason in integration.invalid_reasons:
        _append_reason(reasons, reason)

    metrics: dict[str, float] = {}
    power_valid = not reasons
    if power_valid and benchmark is not None:
        metrics = _derived_metrics(integration, benchmark)

    if not agg_result.is_file():
        _append_reason(reasons, "aggregate_result_missing")
        power_valid = False
        metrics = {}
    else:
        try:
            _patch_power_result(
                agg_result,
                power_valid=power_valid,
                metrics=metrics,
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            _append_reason(reasons, "aggregate_result_unwritable")
            power_valid = False
            metrics = {}
            print(f"[aggregate_power] Failed to patch {agg_result}: {exc}", file=sys.stderr)

    try:
        accumulator_check = cross_check_accumulator(csv_path)
    except (OSError, csv.Error, ValueError):
        accumulator_check = {"available": False, "reason": "cross_check_error"}

    try:
        _write_json_atomic(
            validation_result,
            _validation_payload(
                csv_path=csv_path,
                bench_result=bench_result,
                benchmark=benchmark,
                integration=integration,
                power_valid=power_valid,
                reasons=reasons,
                metrics=metrics,
                accumulator_check=accumulator_check,
            ),
        )
    except OSError as exc:
        print(
            f"[aggregate_power] Failed to write validation artifact "
            f"{validation_result}: {exc}",
            file=sys.stderr,
        )
        return 1 if require_power else 0

    if not power_valid:
        print(
            f"[aggregate_power] Power validation failed: {', '.join(reasons)} "
            f"(details: {validation_result})",
            file=sys.stderr,
        )
        return 1 if require_power else 0

    print(
        f"[aggregate_power] avg_power_w={metrics['avg_power_w']:.2f} "
        f"avg_total_gpu_power_w={metrics['avg_total_gpu_power_w']:.2f} "
        f"total_gpu_energy_j={metrics['total_gpu_energy_j']:.2f} "
        f"(n={integration.observed_num_gpus}) -> {agg_result}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("/workspace/gpu_metrics.csv"),
        help="Path to gpu_metrics.csv from start_gpu_monitor (default: /workspace/gpu_metrics.csv)",
    )
    parser.add_argument(
        "--bench-result",
        type=Path,
        required=True,
        help="Path to the raw benchmark_serving.py result JSON (provides bench window + token counts)",
    )
    parser.add_argument(
        "--agg-result",
        type=Path,
        required=True,
        help="Path to the agg_<run>.json output of process_result.py (will be patched in place)",
    )
    parser.add_argument(
        "--expected-num-gpus",
        type=int,
        required=True,
        help="Expected single-node GPU count from TP * PP * PCP",
    )
    parser.add_argument(
        "--validation-result",
        type=Path,
        help="Path for the power validation sidecar",
    )
    parser.add_argument(
        "--require-power",
        action="store_true",
        default=os.environ.get("REQUIRE_POWER", "").lower() in {"1", "true", "yes"},
        help="Fail when power telemetry is invalid (also enabled by REQUIRE_POWER=1)",
    )
    args = parser.parse_args()
    return run(
        args.csv,
        args.bench_result,
        args.agg_result,
        expected_num_gpus=args.expected_num_gpus,
        validation_result=args.validation_result,
        require_power=args.require_power,
    )


if __name__ == "__main__":
    sys.exit(main())
