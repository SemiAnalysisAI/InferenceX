"""Validate and aggregate single-node GPU-board power into benchmark results.

The formal benchmark window comes from ``benchmark_serving.py``. Power is
integrated independently for every observed GPU with trapezoidal integration
and linear interpolation at the window boundaries. Unqualified energy metrics
therefore always describe the whole deployment, never a prefill/decode role.

Ordinary benchmark runs are best-effort: invalid telemetry is recorded in the
aggregate and a validation sidecar, but does not fail the benchmark. Power
studies can set ``REQUIRE_POWER=1`` to fail after those audit artifacts exist.
The aggregate carries numeric ``power_valid`` (1/0) for metric ingestion; the
sidecar is the canonical source for boolean validity and reason codes.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

_POWER_COL_RE = re.compile(r"power", re.IGNORECASE)
_POWER_EXCLUDE_RE = re.compile(r"limit|cap|max|min", re.IGNORECASE)
_TIMESTAMP_COL_RE = re.compile(r"time", re.IGNORECASE)
_GPU_INDEX_COL_RE = re.compile(r"^(index|gpu|gpu_id|gpu_index|card|device)$", re.IGNORECASE)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

_INTEGRATION_METHOD = "per_device_trapezoidal_with_linear_boundary_interpolation"
_DEFAULT_MAX_SAMPLE_GAP_S = 3.0
_POWER_METRIC_KEYS = {
    "avg_power_w",
    "avg_total_gpu_power_w",
    "total_gpu_energy_j",
    "joules_per_successful_query",
    "joules_per_input_token",
    "joules_per_output_token",
    "joules_per_total_token",
}


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
    avg_power_w: float | None = None
    avg_total_gpu_power_w: float | None = None
    total_gpu_energy_j: float | None = None

    @property
    def observed_num_gpus(self) -> int:
        return len(self.observed_gpu_ids)


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


def _parse_timestamp(value: str) -> float | None:
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
            return datetime.strptime(value, fmt).timestamp()
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
        return dt.timestamp()
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
                ts = _parse_timestamp(ts_raw)
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
    except (OSError, csv.Error):
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


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _gpu_sort_key(gpu_id: str) -> tuple[int, int | str]:
    return (0, int(gpu_id)) if gpu_id.isdigit() else (1, gpu_id)


def _empty_integration(
    *,
    expected_num_gpus: int | None,
    reasons: list[str],
) -> PowerIntegration:
    return PowerIntegration(
        power_valid=False,
        invalid_reasons=tuple(reasons),
        expected_num_gpus=expected_num_gpus,
        observed_gpu_ids=(),
        per_gpu_sample_counts={},
        per_gpu_max_sample_gap_s={},
        per_gpu_energy_j={},
        device_issues={},
    )


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


def integrate_power(
    csv_path: Path,
    *,
    start_unix: float,
    end_unix: float,
    expected_num_gpus: int | None = None,
    max_sample_gap_s: float = _DEFAULT_MAX_SAMPLE_GAP_S,
) -> PowerIntegration:
    """Validate and integrate per-device GPU power over the formal window.

    A valid stream must expose stable GPU identities, match the expected
    topology when supplied, bracket both window boundaries for every device,
    and have no in-window sampling gap larger than ``max_sample_gap_s``.
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
    saw_missing_gpu_identity = False
    try:
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
                timestamp = _parse_timestamp((row.get(timestamp_col) or "").strip())
                if timestamp is None or not math.isfinite(timestamp):
                    continue
                # Only the formal window and its possible boundary neighbors
                # can affect integration. Bad warmup/eval rows farther away
                # must not invalidate an otherwise sound measurement.
                if (
                    timestamp < start_unix - max_sample_gap_s
                    or timestamp > end_unix + max_sample_gap_s
                ):
                    continue

                power = _parse_power((row.get(power_col) or "").strip())
                gpu_id = (row.get(gpu_col) or "").strip()
                if power is None:
                    continue
                if not math.isfinite(power) or power < 0:
                    _append_reason(reasons, "invalid_power_sample")
                    continue
                if not gpu_id:
                    saw_missing_gpu_identity = True
                    continue
                values = raw_samples.setdefault(gpu_id, {}).setdefault(timestamp, [])
                values.append(power)
    except (OSError, csv.Error):
        _append_reason(reasons, "telemetry_file_unreadable")
        return _empty_integration(
            expected_num_gpus=expected_num_gpus,
            reasons=reasons,
        )

    if saw_missing_gpu_identity:
        _append_reason(reasons, "gpu_identity_missing")
    if not raw_samples:
        _append_reason(reasons, "no_usable_power_samples")
        return _empty_integration(
            expected_num_gpus=expected_num_gpus,
            reasons=reasons,
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

    for gpu_id in observed_gpu_ids:
        timestamp_values = raw_samples[gpu_id]
        samples = sorted(
            (timestamp, mean(values)) for timestamp, values in timestamp_values.items()
        )
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

    return PowerIntegration(
        power_valid=not reasons,
        invalid_reasons=tuple(reasons),
        expected_num_gpus=expected_num_gpus,
        observed_gpu_ids=observed_gpu_ids,
        per_gpu_sample_counts=per_gpu_sample_counts,
        per_gpu_max_sample_gap_s=per_gpu_max_sample_gap_s,
        per_gpu_energy_j=per_gpu_energy_j,
        device_issues=device_issues,
        avg_power_w=avg_power_w,
        avg_total_gpu_power_w=avg_total_gpu_power_w,
        total_gpu_energy_j=total_gpu_energy_j,
    )


def _load_bench_window(
    bench_result_path: Path,
) -> tuple[float, float, float, int, int] | None:
    """Read (start_unix, end_unix, duration_s, total_output_tokens, total_input_tokens)
    from the raw bench JSON. Returns None if any required field is missing.

    total_input_tokens defaults to 0 if absent (older bench JSONs may not have it);
    this only degrades joules_per_total_token to equal joules_per_output_token in
    that case, never breaks the rest of the aggregation.
    """
    try:
        bench = json.loads(bench_result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    start = bench.get("benchmark_start_time_unix")
    end = bench.get("benchmark_end_time_unix")
    duration = bench.get("duration")
    total_output = bench.get("total_output_tokens")
    total_input = bench.get("total_input_tokens", 0)
    if not all(isinstance(v, (int, float)) for v in (start, end, duration)):
        return None
    if not isinstance(total_output, int) or total_output <= 0:
        return None
    if not isinstance(total_input, int) or total_input < 0:
        total_input = 0
    return float(start), float(end), float(duration), int(total_output), int(total_input)


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


def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _derived_metrics(
    integration: PowerIntegration,
    benchmark: BenchmarkData,
) -> dict[str, float]:
    """Return whole-deployment energy metrics for a valid measurement."""
    if (
        integration.avg_power_w is None
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
    data = json.loads(agg_path.read_text(encoding="utf-8"))
    for key in _POWER_METRIC_KEYS:
        data.pop(key, None)
    # Keep the canonical aggregate numeric-only for InferenceX-app's metric
    # auto-capture. Detailed reason codes live in the validation sidecar.
    data["power_valid"] = int(power_valid)
    data.pop("power_invalid_reasons", None)
    if power_valid:
        for key, value in metrics.items():
            if value is None or not math.isfinite(value):
                raise ValueError(f"non-finite power metric: {key}")
            precision = 3 if key.endswith(("_w", "_j")) else 6
            data[key] = round(value, precision)
    _write_json_atomic(agg_path, data)


def _validation_payload(
    *,
    csv_path: Path,
    bench_result: Path,
    benchmark: BenchmarkData | None,
    integration: PowerIntegration,
    power_valid: bool,
    reasons: list[str],
    metrics: dict[str, float],
) -> dict:
    benchmark_window = None
    if benchmark is not None:
        benchmark_window = {
            "start_time_unix": benchmark.start_unix,
            "end_time_unix": benchmark.end_unix,
            "reported_duration_s": benchmark.reported_duration_s,
            "integration_duration_s": benchmark.integration_duration_s,
            "completed": benchmark.completed,
            "total_input_tokens": benchmark.total_input_tokens,
            "total_output_tokens": benchmark.total_output_tokens,
        }
    return {
        "schema_version": 1,
        "power_valid": power_valid,
        "reasons": reasons,
        "telemetry_source": str(csv_path),
        "benchmark_result": str(bench_result),
        "benchmark_window": benchmark_window,
        "integration_method": _INTEGRATION_METHOD,
        "expected_gpu_count": integration.expected_num_gpus,
        "observed_gpu_count": integration.observed_num_gpus,
        "observed_gpu_ids": list(integration.observed_gpu_ids),
        "per_gpu_sample_counts": integration.per_gpu_sample_counts,
        "per_gpu_max_sample_gap_s": integration.per_gpu_max_sample_gap_s,
        "per_gpu_energy_j": integration.per_gpu_energy_j,
        "device_issues": integration.device_issues,
        "metrics": {
            key: round(value, 6)
            for key, value in metrics.items()
            if value is not None and math.isfinite(value)
        },
    }


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
