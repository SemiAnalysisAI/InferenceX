"""Validate and aggregate multinode srt-slurm DCGM power artifacts.

Consumes the srt-slurm ``dcgm-power`` artifact package (v1 wire contract:
``power/{manifest.json, samples.csv, windows/*.json}``) produced for
multinode runs, re-validates it independently, and patches whole-deployment
energy metrics plus role-level metrics for disaggregated deployments into the
aggregate JSON.

Trust model: the producer's stored ``publication_valid`` verdict is never
trusted. Validity is recomputed here from the persisted bytes (mirroring
srt-slurm's own offline validator), then cross-checked against the stored
verdict; any disagreement invalidates the measurement. The producer git
commit must equal the expected pin — in every mode — because energy numbers
from an unknown producer contract must never be published.

Role energy semantics: ``prefill_gpu_energy_j`` / ``decode_gpu_energy_j`` are
the board-level energy of that role's GPUs integrated over the FULL formal
serving window. They are not kernel-level prefill/decode phase energies.
``prefill_avg_power_w`` / ``decode_avg_power_w`` divide that role energy by the
same full window and by the role's GPU count, so they are the mean board draw
of that role's GPUs across the whole serving window, not a phase power.

Ordinary benchmark runs are best-effort: invalid telemetry records
``power_valid=0`` (and no energy metrics) in the aggregate plus a validation
sidecar, but never fails the benchmark. Power studies set ``REQUIRE_POWER=1``
to fail after those audit artifacts exist.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

try:
    from .aggregate_power import (
        POWER_METRIC_SCHEMA_VERSION,
        BenchmarkData,
        _append_reason,
        _integrate_device,
        _load_benchmark_data,
        _write_json_atomic,
    )
except ImportError:  # Direct execution: python utils/aggregate_power_multinode.py
    from aggregate_power import (
        POWER_METRIC_SCHEMA_VERSION,
        BenchmarkData,
        _append_reason,
        _integrate_device,
        _load_benchmark_data,
        _write_json_atomic,
    )

# --- srt-slurm dcgm-power v1 wire contract (mirrored constants) -------------

SCHEMA_VERSION = 1
PRODUCER = "srt-slurm.dcgm-power"
POWER_METRIC = "DCGM_FI_DEV_POWER_USAGE"
POWER_UNIT = "W"
POWER_SCOPE = "gpu_device_board_as_reported_by_dcgm"
CLOCK_SOURCE = "head_node_unix_clock"

MANIFEST_FILENAME = "manifest.json"
SAMPLES_FILENAME = "samples.csv"
WINDOWS_DIRNAME = "windows"

SAMPLES_HEADER = (
    "schema_version",
    "timestamp_unix",
    "scrape_seq",
    "hostname",
    "gpu_index",
    "gpu_uuid",
    "power_w",
)

# Fixed by the producer contract (srt-slurm contract.MAX_SAMPLE_GAP_SECONDS),
# NOT a multiple of the configured sample interval.
MAX_SAMPLE_GAP_SECONDS = 3.0

WORKER_ROLES = ("prefill", "decode", "agg")

STATUS_COMPLETE = "complete"
WINDOW_STATUS_RUNNING = "running"
WINDOW_STATUS_COMPLETED = "completed"
_WINDOW_ALLOWED_STATUSES = ("running", "completed", "failed", "interrupted")

_CLOCK_TOLERANCE_SECONDS = 0.5
_CLOCK_TOLERANCE_FRACTION = 0.01

# Lifecycle reason codes that contradict a "complete" manifest.
_FATAL_LIFECYCLE_REASONS = (
    "exporter_exited",
    "collector_exception",
    "collector_interrupted",
    "collector_join_timeout",
    "benchmark_child_reap_timeout",
)
_STARTUP_FAILURE_REASONS = (
    "exporter_startup_timeout",
    "exporter_launch_failed",
    "endpoint_resolution_failed",
)

_INTEGRATION_METHOD = "per_device_trapezoidal_with_linear_boundary_interpolation"

WHOLE_METRIC_KEYS = (
    "avg_power_w",
    "avg_total_gpu_power_w",
    "total_gpu_energy_j",
    "joules_per_successful_query",
    "joules_per_input_token",
    "joules_per_output_token",
    "joules_per_total_token",
)
ROLE_METRIC_KEYS = (
    "prefill_gpu_energy_j",
    "decode_gpu_energy_j",
    "prefill_avg_power_w",
    "decode_avg_power_w",
    "prefill_joules_per_input_token",
    "decode_joules_per_output_token",
)
_ALL_POWER_METRIC_KEYS = WHOLE_METRIC_KEYS + ROLE_METRIC_KEYS


def _dedupe(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _is_safe_relative_subpath(value: str) -> bool:
    if not value or value.startswith(("/", "~")):
        return False
    parts = PurePosixPath(value).parts
    return bool(parts) and not any(part in ("..", "") for part in parts)


def _stays_below(root: Path, relative: str) -> bool:
    try:
        (root / relative).resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        return False
    return True


def _is_finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_positive_finite(value) -> bool:
    return _is_finite(value) and value > 0


# --- strict manifest parsing (mirrors srt-slurm validate_artifacts) ---------


@dataclass(frozen=True)
class DeviceAssignment:
    worker_role: str
    worker_index: int
    worker_process: int
    het_group: int | None


@dataclass(frozen=True)
class ExpectedDevice:
    hostname: str
    gpu_index: int
    assignments: tuple[DeviceAssignment, ...]

    @property
    def key(self) -> tuple[str, int]:
        return (self.hostname, self.gpu_index)


@dataclass(frozen=True)
class ExpectedWindow:
    benchmark_type: str
    concurrency: int

    @property
    def key(self) -> tuple[str, int]:
        return (self.benchmark_type, self.concurrency)


@dataclass(frozen=True)
class SampleRow:
    timestamp_unix: float
    scrape_seq: int
    hostname: str
    gpu_index: int
    gpu_uuid: str
    power_w: float

    @property
    def key(self) -> tuple[int, str, int]:
        return (self.scrape_seq, self.hostname, self.gpu_index)


@dataclass(frozen=True)
class ObservedDevice:
    hostname: str
    gpu_index: int
    gpu_uuids: tuple[str, ...]
    first_sample_time_unix: float
    last_sample_time_unix: float
    sample_times: tuple[float, ...] = field(default=(), repr=False)

    @property
    def key(self) -> tuple[str, int]:
        return (self.hostname, self.gpu_index)

    def to_dict(self) -> dict:
        return {
            "hostname": self.hostname,
            "gpu_index": self.gpu_index,
            "gpu_uuids": list(self.gpu_uuids),
            "first_sample_time_unix": self.first_sample_time_unix,
            "last_sample_time_unix": self.last_sample_time_unix,
        }


def _text(value, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is not a non-empty string: {value!r}")
    return value


def _whole(value, label: str, *, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{label} is not an integer >= {minimum}: {value!r}")
    return value


def _role(value) -> str:
    if value not in WORKER_ROLES:
        raise ValueError(f"worker_role is not one of {WORKER_ROLES}: {value!r}")
    return value


def _parse_expected_devices(manifest: dict) -> list[ExpectedDevice]:
    entries = manifest.get("expected_devices") or []
    if not isinstance(entries, list):
        raise TypeError(f"expected_devices is not a list: {entries!r}")
    devices: list[ExpectedDevice] = []
    for entry in entries:
        raw_assignments = entry.get("assignments") or []
        if not isinstance(raw_assignments, list) or not raw_assignments:
            raise ValueError(f"assignments is not a non-empty list: {raw_assignments!r}")
        assignments = tuple(
            DeviceAssignment(
                worker_role=_role(assignment["worker_role"]),
                worker_index=_whole(assignment["worker_index"], "worker_index", minimum=0),
                worker_process=_whole(assignment["worker_process"], "worker_process", minimum=0),
                het_group=(
                    None
                    if assignment.get("het_group") is None
                    else _whole(assignment["het_group"], "het_group", minimum=0)
                ),
            )
            for assignment in raw_assignments
        )
        devices.append(
            ExpectedDevice(
                hostname=_text(entry["hostname"], "hostname"),
                gpu_index=_whole(entry["gpu_index"], "gpu_index", minimum=0),
                assignments=assignments,
            )
        )
    return devices


def _parse_expected_windows(manifest: dict) -> list[ExpectedWindow]:
    entries = manifest.get("expected_windows") or []
    if not isinstance(entries, list):
        raise TypeError(f"expected_windows is not a list: {entries!r}")
    return [
        ExpectedWindow(
            benchmark_type=_text(entry["benchmark_type"], "benchmark_type"),
            concurrency=_whole(entry["concurrency"], "concurrency", minimum=1),
        )
        for entry in entries
    ]


def _check_wire_contract(manifest: dict) -> list[str]:
    """Mirror srt-slurm's manifest wire/lifecycle checks (verdict excluded)."""
    failures: list[str] = []
    for key, expected in (
        ("schema_version", SCHEMA_VERSION),
        ("producer", PRODUCER),
        ("source_metric", POWER_METRIC),
        ("unit", POWER_UNIT),
        ("power_scope", POWER_SCOPE),
        ("timestamp_source", CLOCK_SOURCE),
    ):
        if manifest.get(key) != expected:
            failures.append(f"{key} is {manifest.get(key)!r}, expected {expected!r}")

    status = manifest.get("status")
    if status != STATUS_COMPLETE:
        failures.append(f"status is {status!r}, expected {STATUS_COMPLETE!r}")

    if not _is_finite(manifest.get("started_at_unix")):
        failures.append("started_at_unix is not a finite number")
    if not _is_finite(manifest.get("stopped_at_unix")):
        failures.append("stopped_at_unix is not finite in a terminal manifest")
    for key in ("sample_interval_seconds", "request_timeout_seconds"):
        if not _is_positive_finite(manifest.get(key)):
            failures.append(f"{key} is not finite and positive")
    for key in ("scrape_count", "sample_row_count"):
        value = manifest.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            failures.append(f"{key} is not a non-negative integer")

    required = manifest.get("required")
    if not isinstance(required, bool):
        failures.append(f"required is {required!r}, expected a boolean")

    reasons = manifest.get("reason_codes")
    if not isinstance(reasons, list) or not all(isinstance(reason, str) for reason in reasons):
        failures.append("reason_codes is not a list of strings")
    else:
        blocking = set(_FATAL_LIFECYCLE_REASONS)
        if required is True:
            blocking |= set(_STARTUP_FAILURE_REASONS)
        incompatible = sorted({str(reason) for reason in reasons} & blocking)
        if incompatible:
            failures.append(
                f"complete manifest carries lifecycle-failure reasons: {', '.join(incompatible)}"
            )

    exporter = manifest.get("dcgm_exporter")
    if not isinstance(exporter, dict):
        failures.append("dcgm_exporter is not an object")
    elif not exporter.get("container_image_resolved"):
        failures.append("dcgm_exporter.container_image_resolved is empty")

    return failures


# --- strict samples parsing (mirrors srt-slurm samples.read_samples) --------


def _parse_sample_row(raw: list[str]) -> SampleRow | None:
    if len(raw) != len(SAMPLES_HEADER):
        return None
    try:
        schema_version = int(raw[0])
        timestamp_unix = float(raw[1])
        scrape_seq = int(raw[2])
        gpu_index = int(raw[4])
        power_w = float(raw[6])
    except ValueError:
        return None
    hostname, gpu_uuid = raw[3], raw[5]
    if schema_version != SCHEMA_VERSION or not hostname or not gpu_uuid:
        return None
    if not math.isfinite(timestamp_unix) or not math.isfinite(power_w) or power_w < 0:
        return None
    if scrape_seq < 0 or gpu_index < 0:
        return None
    return SampleRow(
        timestamp_unix=timestamp_unix,
        scrape_seq=scrape_seq,
        hostname=hostname,
        gpu_index=gpu_index,
        gpu_uuid=gpu_uuid,
        power_w=power_w,
    )


def read_samples(path: Path) -> tuple[tuple[SampleRow, ...], tuple[str, ...]]:
    if not path.is_file():
        return (), ("samples_csv_missing",)

    reasons: list[str] = []
    rows: list[SampleRow] = []
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header != list(SAMPLES_HEADER):
                return (), ("samples_csv_header_mismatch",)
            for raw in reader:
                row = _parse_sample_row(raw)
                if row is None:
                    reasons.append("samples_csv_malformed")
                    continue
                rows.append(row)
    except (OSError, UnicodeDecodeError, csv.Error):
        reasons.append("samples_csv_malformed")

    seen: set[tuple[int, str, int]] = set()
    unique: list[SampleRow] = []
    for row in rows:
        if row.key in seen:
            reasons.append("duplicate_sample_row")
            continue
        seen.add(row.key)
        unique.append(row)

    per_device: dict[tuple[str, int], list[SampleRow]] = {}
    for row in unique:
        per_device.setdefault((row.hostname, row.gpu_index), []).append(row)
    for device_rows in per_device.values():
        ordered = sorted(device_rows, key=lambda row: row.scrape_seq)
        if any(
            current.timestamp_unix <= previous.timestamp_unix
            for previous, current in itertools.pairwise(ordered)
        ):
            reasons.append("timestamp_non_monotonic")
            break

    return tuple(unique), _dedupe(reasons)


def derive_observed_devices(rows: tuple[SampleRow, ...]) -> list[ObservedDevice]:
    ordered: dict[tuple[str, int], list[SampleRow]] = {}
    for row in rows:
        ordered.setdefault((row.hostname, row.gpu_index), []).append(row)

    devices: list[ObservedDevice] = []
    for key in sorted(ordered):
        device_rows = sorted(ordered[key], key=lambda row: row.scrape_seq)
        times = tuple(row.timestamp_unix for row in device_rows)
        devices.append(
            ObservedDevice(
                hostname=key[0],
                gpu_index=key[1],
                gpu_uuids=tuple(dict.fromkeys(row.gpu_uuid for row in device_rows)),
                first_sample_time_unix=min(times),
                last_sample_time_unix=max(times),
                sample_times=tuple(sorted(times)),
            )
        )
    return devices


# --- device identity / topology (mirrors srt-slurm validation.py) -----------


def _resolve_roles(devices: list[ExpectedDevice]) -> tuple[dict[tuple[str, int], str], list[str]]:
    roles: dict[tuple[str, int], str] = {}
    for device in devices:
        distinct = {assignment.worker_role for assignment in device.assignments}
        if len(distinct) != 1:
            return {}, ["conflicting_worker_roles"]
        roles[device.key] = distinct.pop()
    return roles, []


def _resolve_het_groups(devices: list[ExpectedDevice]) -> tuple[dict[str, int | None], list[str]]:
    groups: dict[str, int | None] = {}
    for device in devices:
        distinct = {assignment.het_group for assignment in device.assignments}
        if len(distinct) != 1:
            return {}, ["conflicting_het_groups"]
        group = distinct.pop()
        if device.hostname in groups and groups[device.hostname] != group:
            return {}, ["conflicting_het_groups"]
        groups[device.hostname] = group
    return groups, []


def _validate_devices(
    expected: list[ExpectedDevice],
    observed: list[ObservedDevice],
) -> tuple[list[str], dict[tuple[str, int], str]]:
    reasons: list[str] = []
    expected_keys = {device.key for device in expected}
    observed_keys = {device.key for device in observed}

    if not expected_keys or expected_keys - observed_keys:
        reasons.append("expected_device_missing")
    if observed_keys - expected_keys:
        reasons.append("unexpected_device")
    # A UUID must map 1:1 to a device key, or one physical GPU is counted twice.
    if any(len(device.gpu_uuids) != 1 for device in observed):
        reasons.append("gpu_uuid_changed")
    else:
        uuids = [device.gpu_uuids[0] for device in observed]
        if len(set(uuids)) != len(uuids):
            reasons.append("gpu_uuid_changed")

    roles, role_conflicts = _resolve_roles(expected)
    _, group_conflicts = _resolve_het_groups(expected)
    reasons.extend(role_conflicts)
    reasons.extend(group_conflicts)
    return list(_dedupe(reasons)), roles


def _check_role_topology(
    expected_devices: list[ExpectedDevice],
    roles: dict[tuple[str, int], str],
    expected_roles: dict[str, int],
) -> list[str]:
    """Exact role-count match against the workflow env; distinct het groups when present."""
    failures: list[str] = []
    counts: dict[str, int] = {}
    for role in roles.values():
        counts[role] = counts.get(role, 0) + 1

    if set(counts) != set(expected_roles):
        failures.append(f"expected roles {sorted(expected_roles)}, found {sorted(counts)}")
    for role, count in sorted(expected_roles.items()):
        if counts.get(role, 0) != count:
            failures.append(f"expected {count} {role} GPUs, found {counts.get(role, 0)}")

    groups, group_conflicts = _resolve_het_groups(expected_devices)
    if group_conflicts:
        return failures + [f"{reason} (topology)" for reason in group_conflicts]

    per_role: dict[str, set] = {}
    for device in expected_devices:
        if device.key in roles:
            per_role.setdefault(roles[device.key], set()).add(groups[device.hostname])

    # Non-het deployments (e.g. GB200 single-job 1P1D) legitimately report
    # het_group=None for every device; the exemption is deployment-wide, so a
    # mixed manifest (one role het, another None) stays a topology failure.
    non_het_deployment = all(group is None for group in groups.values())

    assigned: list[int] = []
    for role, role_groups in sorted(per_role.items()):
        if len(role_groups) != 1:
            failures.append(
                f"role {role} spans het groups {sorted(role_groups, key=str)}, expected exactly one"
            )
            continue
        group = next(iter(role_groups))
        if group is None:
            if not non_het_deployment:
                failures.append(
                    f"role {role} has het group None while other devices report real het groups"
                )
            continue
        if not isinstance(group, int) or isinstance(group, bool) or group < 0:
            failures.append(f"role {role} has het group {group!r}, expected a non-negative integer")
            continue
        assigned.append(group)
    if len(set(assigned)) != len(assigned):
        failures.append("roles share a het group")
    return failures


# --- window scan + audit (mirrors srt-slurm windows.py) ---------------------


@dataclass(frozen=True)
class ParsedWindow:
    relative_path: str
    benchmark_type: str
    concurrency: int
    result_path: str
    status: str
    start_unix: float
    end_unix: float | None
    duration: float | None


def _window_status_invariants_hold(status: str, end, duration, reason) -> bool:
    if status == WINDOW_STATUS_RUNNING:
        return end is None and duration is None and reason is None
    if status == "interrupted":
        return end is None and duration is None and isinstance(reason, str) and bool(reason)
    if not _is_finite(end) or not _is_finite(duration) or duration <= 0:
        return False
    if status == WINDOW_STATUS_COMPLETED:
        return reason is None
    return isinstance(reason, str) and bool(reason)


def _parse_window(path: Path, relative: str, result_root: Path) -> tuple[ParsedWindow | None, list[str]]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return None, ["measurement_window_malformed"]
    if not isinstance(payload, dict):
        return None, ["measurement_window_malformed"]

    status = payload.get("status")
    concurrency = payload.get("concurrency")
    benchmark_type = payload.get("benchmark_type")
    start = payload.get("benchmark_start_time_unix")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("clock_source") != CLOCK_SOURCE
        or status not in _WINDOW_ALLOWED_STATUSES
        or not isinstance(benchmark_type, str)
        or not isinstance(concurrency, int)
        or not _is_finite(start)
    ):
        return None, ["measurement_window_malformed"]

    result_path = payload.get("result_path")
    if not isinstance(result_path, str) or not _is_safe_relative_subpath(result_path):
        return None, ["measurement_window_result_path_invalid"]
    if not _stays_below(result_root, result_path) or Path(result_path).stem != path.stem:
        return None, ["measurement_window_result_path_invalid"]

    end = payload.get("benchmark_end_time_unix")
    duration = payload.get("duration")
    if not _window_status_invariants_hold(status, end, duration, payload.get("reason")):
        return None, ["measurement_window_malformed"]

    return (
        ParsedWindow(
            relative_path=relative,
            benchmark_type=benchmark_type,
            concurrency=concurrency,
            result_path=result_path,
            status=status,
            start_unix=float(start),
            end_unix=float(end) if end is not None else None,
            duration=float(duration) if duration is not None else None,
        ),
        [],
    )


def _scan_windows(
    windows_dir: Path,
    result_root: Path,
    artifact_errors: list[dict],
) -> tuple[dict[tuple[str, int], ParsedWindow], set[tuple[str, int]]]:
    parsed: dict[tuple[str, int], ParsedWindow] = {}
    duplicates: set[tuple[str, int]] = set()
    if not windows_dir.is_dir():
        return parsed, duplicates

    if not _stays_below(windows_dir.parent, WINDOWS_DIRNAME):
        artifact_errors.append(
            {"path": WINDOWS_DIRNAME, "reason_codes": ["measurement_window_artifact_path_invalid"]}
        )
        return parsed, duplicates

    for path in sorted(windows_dir.iterdir()):
        relative = f"{WINDOWS_DIRNAME}/{path.name}"
        if path.is_symlink() or not path.is_file():
            artifact_errors.append(
                {"path": relative, "reason_codes": ["measurement_window_artifact_path_invalid"]}
            )
            continue

        window, reasons = _parse_window(path, relative, result_root)
        if window is None:
            artifact_errors.append({"path": relative, "reason_codes": reasons})
            continue

        key = (window.benchmark_type, window.concurrency)
        if key in parsed:
            duplicates.add(key)
            artifact_errors.append(
                {"path": parsed[key].relative_path, "reason_codes": ["measurement_window_duplicate"]}
            )
            artifact_errors.append({"path": relative, "reason_codes": ["measurement_window_duplicate"]})
            continue
        parsed[key] = window

    for key in duplicates:
        parsed.pop(key, None)
    return parsed, duplicates


def _check_window_result(window: ParsedWindow, result_root: Path) -> list[str]:
    result_file = result_root / window.result_path
    if not result_file.is_file():
        return ["measurement_window_result_missing"]
    try:
        result = json.loads(result_file.read_text())
    except (OSError, ValueError):
        return ["measurement_window_result_missing"]
    if not isinstance(result, dict):
        return ["measurement_window_result_mismatch"]

    if (
        result.get("benchmark_start_time_unix") != window.start_unix
        or result.get("benchmark_end_time_unix") != window.end_unix
        or result.get("duration") != window.duration
    ):
        return ["measurement_window_result_mismatch"]

    wall = window.end_unix - window.start_unix
    tolerance = max(_CLOCK_TOLERANCE_SECONDS, _CLOCK_TOLERANCE_FRACTION * window.duration)
    if abs(wall - window.duration) > tolerance:
        return ["measurement_window_clock_mismatch"]
    return []


def _bracketing_sequence(times: tuple[float, ...], start: float, end: float) -> list[float] | None:
    before = [value for value in times if value <= start]
    after = [value for value in times if value >= end]
    if not before or not after:
        return None
    inside = [value for value in times if start < value < end]
    return [before[-1], *inside, after[0]]


def _check_coverage(
    start: float,
    end: float,
    expected_device_keys: set[tuple[str, int]],
    observed_devices: list[ObservedDevice],
) -> tuple[dict[str, float], list[str]]:
    by_key = {device.key: device for device in observed_devices}
    gaps: dict[str, float] = {}
    reasons: list[str] = []

    for key in sorted(expected_device_keys):
        device = by_key.get(key)
        if device is None:
            reasons.append("measurement_window_not_bracketed")
            continue
        if len(device.gpu_uuids) != 1:
            reasons.append("gpu_uuid_changed")
            continue
        sequence = _bracketing_sequence(device.sample_times, start, end)
        if sequence is None:
            reasons.append("measurement_window_not_bracketed")
            continue
        largest = max(
            (later - earlier for earlier, later in itertools.pairwise(sequence)), default=0.0
        )
        gaps[f"{device.hostname}/{device.gpu_uuids[0]}"] = largest
        if largest > MAX_SAMPLE_GAP_SECONDS:
            reasons.append("sample_gap_exceeded")

    return gaps, reasons


def _validate_expected_windows(
    *,
    power_dir: Path,
    result_root: Path,
    expected_windows: list[ExpectedWindow],
    expected_device_keys: set[tuple[str, int]],
    observed_devices: list[ObservedDevice],
    artifact_errors: list[dict],
) -> tuple[list[dict], dict[tuple[str, int], ParsedWindow]]:
    """Mirror the producer's per-expected-window audit rows exactly."""
    parsed, duplicates = _scan_windows(power_dir / WINDOWS_DIRNAME, result_root, artifact_errors)
    expected_keys = {window.key for window in expected_windows}

    for key, window in sorted(parsed.items()):
        if key not in expected_keys:
            artifact_errors.append(
                {"path": window.relative_path, "reason_codes": ["measurement_window_unexpected"]}
            )

    validations: list[dict] = []
    for expected in expected_windows:
        window = parsed.get(expected.key)
        if window is None:
            reasons = (
                ["measurement_window_duplicate"]
                if expected.key in duplicates
                else ["measurement_window_missing"]
            )
            validations.append(
                {
                    "benchmark_type": expected.benchmark_type,
                    "concurrency": expected.concurrency,
                    "window_file": None,
                    "power_coverage_valid": False,
                    "reason_codes": reasons,
                    "per_device_max_sample_gap_seconds": {},
                }
            )
            continue

        reasons = []
        if window.status != WINDOW_STATUS_COMPLETED:
            reasons.append("measurement_window_incomplete")
        else:
            reasons.extend(_check_window_result(window, result_root))

        gaps: dict[str, float] = {}
        if not reasons:
            gaps, coverage_reasons = _check_coverage(
                window.start_unix, window.end_unix, expected_device_keys, observed_devices
            )
            reasons.extend(coverage_reasons)
            if coverage_reasons:
                gaps = {}

        validations.append(
            {
                "benchmark_type": expected.benchmark_type,
                "concurrency": expected.concurrency,
                "window_file": window.relative_path,
                "power_coverage_valid": not reasons,
                "reason_codes": list(_dedupe(reasons)),
                "per_device_max_sample_gap_seconds": gaps,
            }
        )
    return validations, parsed


# --- stored-evidence cross-check (mirrors srt-slurm validate_artifacts) -----


def _check_stored_evidence(
    manifest: dict,
    expected_devices: list[ExpectedDevice],
    expected_windows: list[ExpectedWindow],
    rows: tuple[SampleRow, ...],
    observed: list[ObservedDevice],
    validations: list[dict],
    artifact_errors: list[dict],
) -> list[str]:
    failures: list[str] = []

    stored_rows = manifest.get("sample_row_count")
    if stored_rows != len(rows):
        failures.append(f"sample_row_count is {stored_rows!r}, disk has {len(rows)} rows")

    stored_scrapes = manifest.get("scrape_count")
    if rows:
        least = max(row.scrape_seq for row in rows) + 1
        if not isinstance(stored_scrapes, int) or isinstance(stored_scrapes, bool) or stored_scrapes < least:
            failures.append(f"scrape_count is {stored_scrapes!r}, disk needs at least {least}")

    if not observed:
        failures.append("observed_devices is empty")
    if manifest.get("observed_devices") != [device.to_dict() for device in observed]:
        failures.append("observed_devices does not match the devices derived from samples.csv")
    if manifest.get("window_validations") != validations:
        failures.append("window_validations does not match the recomputed window audit")
    if manifest.get("artifact_errors") != artifact_errors:
        failures.append("artifact_errors does not match the recomputed artifact scan")

    for label, keys in (
        ("expected_devices", [device.key for device in expected_devices]),
        ("expected_windows", [window.key for window in expected_windows]),
    ):
        if len(set(keys)) != len(keys):
            failures.append(f"{label} contains duplicate keys")

    return failures


# --- consumer-side verdict, integration, and metrics ------------------------


@dataclass
class MultinodePowerAudit:
    """Everything the sidecar needs to explain the verdict."""

    power_valid: bool = False
    reasons: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    stored_publication_valid: bool | None = None
    recomputed_publication_valid: bool | None = None
    producer_git_commit: str | None = None
    expected_producer_git_commit: str | None = None
    exporter_image_sha256: str | None = None
    window: dict | None = None
    per_gpu_energy_j: dict[str, float] = field(default_factory=dict)
    per_gpu_role: dict[str, str] = field(default_factory=dict)
    per_gpu_max_sample_gap_s: dict[str, float] = field(default_factory=dict)
    observed_gpu_count: int = 0
    expected_gpu_count: int = 0
    metrics: dict[str, float] = field(default_factory=dict)


def _canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _add_reason(audit: MultinodePowerAudit, reason: str, detail: str | None = None) -> None:
    _append_reason(audit.reasons, reason)
    if detail:
        audit.failures.append(detail)


def validate_and_integrate(
    *,
    power_dir: Path,
    logs_root: Path,
    bench_result_path: Path,
    benchmark: BenchmarkData | None,
    prefill_gpus: int,
    decode_gpus: int,
    aggregate_gpus: int,
    expected_producer_sha: str | None,
) -> MultinodePowerAudit:
    """Recompute package validity, cross-check verdicts, and integrate energy."""
    audit = MultinodePowerAudit(
        expected_producer_git_commit=expected_producer_sha,
    )

    if not power_dir.is_dir():
        _add_reason(audit, "power_artifacts_missing", f"power dir missing: {power_dir}")
        return audit

    manifest_path = power_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        _add_reason(audit, "manifest_missing", f"manifest missing: {manifest_path}")
        return audit
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as exc:
        _add_reason(audit, "manifest_malformed", f"manifest unreadable: {exc}")
        return audit
    if not isinstance(manifest, dict):
        _add_reason(audit, "manifest_malformed", "manifest is not an object")
        return audit

    stored_valid = manifest.get("publication_valid")
    audit.stored_publication_valid = stored_valid if isinstance(stored_valid, bool) else None
    commit = manifest.get("producer_git_commit")
    audit.producer_git_commit = commit if isinstance(commit, str) else None
    exporter = manifest.get("dcgm_exporter")
    if isinstance(exporter, dict):
        sha = exporter.get("container_image_sha256")
        audit.exporter_image_sha256 = sha if isinstance(sha, str) else None

    # Gate: producer pin. Unknown producer contract → never publish numbers.
    if not expected_producer_sha:
        _add_reason(audit, "producer_pin_missing", "no expected producer SHA configured")
    elif audit.producer_git_commit != expected_producer_sha:
        _add_reason(
            audit,
            "producer_commit_mismatch",
            f"producer_git_commit {audit.producer_git_commit!r} != expected {expected_producer_sha!r}",
        )

    # Recompute publication validity exactly like srt-slurm's offline validator.
    recompute_failures: list[str] = _check_wire_contract(manifest)

    try:
        expected_devices = _parse_expected_devices(manifest)
        expected_windows = _parse_expected_windows(manifest)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        _add_reason(audit, "manifest_malformed", f"manifest malformed: {exc!r}")
        audit.recomputed_publication_valid = False
        return audit

    rows, sample_reasons = read_samples(power_dir / SAMPLES_FILENAME)
    recompute_failures += [f"{reason} in {SAMPLES_FILENAME}" for reason in sample_reasons]

    observed = derive_observed_devices(rows)
    device_reasons, roles = _validate_devices(expected_devices, observed)
    recompute_failures += [f"{reason} (device identity/topology)" for reason in device_reasons]

    artifact_errors: list[dict] = []
    validations, parsed_windows = _validate_expected_windows(
        power_dir=power_dir,
        result_root=logs_root,
        expected_windows=expected_windows,
        expected_device_keys={device.key for device in expected_devices},
        observed_devices=observed,
        artifact_errors=artifact_errors,
    )
    if not expected_windows:
        recompute_failures.append("no expected measurement window")
    for validation in validations:
        recompute_failures += [
            f"{reason} (window {validation['benchmark_type']}/{validation['concurrency']})"
            for reason in validation["reason_codes"]
        ]
    recompute_failures += [
        f"{reason} ({error['path']})" for error in artifact_errors for reason in error["reason_codes"]
    ]
    recompute_failures += _check_stored_evidence(
        manifest, expected_devices, expected_windows, rows, observed, validations, artifact_errors
    )

    audit.recomputed_publication_valid = not recompute_failures
    audit.failures.extend(recompute_failures)
    if recompute_failures:
        _append_reason(audit.reasons, "package_recompute_invalid")

    # Gate: stored verdict must agree with the recomputation, and both must be
    # true. The stored verdict is never trusted on its own.
    if audit.stored_publication_valid is not True or (
        audit.stored_publication_valid != audit.recomputed_publication_valid
    ):
        if audit.stored_publication_valid != audit.recomputed_publication_valid:
            _add_reason(
                audit,
                "producer_verdict_mismatch",
                f"stored publication_valid={audit.stored_publication_valid!r}, "
                f"recomputed={audit.recomputed_publication_valid!r}",
            )
        elif audit.stored_publication_valid is not True:
            _add_reason(audit, "package_recompute_invalid", "stored and recomputed both invalid")

    # Gate: role topology must match the workflow's own GPU counts. Aggregate
    # workers are mutually exclusive with disaggregated prefill/decode workers;
    # this prevents phase-local fields from being fabricated for shared GPUs.
    expected_roles = {}
    if aggregate_gpus > 0 and (prefill_gpus > 0 or decode_gpus > 0):
        _add_reason(
            audit,
            "topology_env_mismatch",
            "aggregate GPUs cannot be combined with prefill/decode GPUs",
        )
    if prefill_gpus > 0:
        expected_roles["prefill"] = prefill_gpus
    if decode_gpus > 0:
        expected_roles["decode"] = decode_gpus
    if aggregate_gpus > 0:
        expected_roles["agg"] = aggregate_gpus
    if not expected_roles:
        _add_reason(
            audit,
            "topology_env_mismatch",
            "prefill, decode, and aggregate GPU counts are all zero",
        )
    elif roles:
        topology_failures = _check_role_topology(expected_devices, roles, expected_roles)
        for failure in topology_failures:
            _add_reason(audit, "topology_env_mismatch", failure)

    audit.expected_gpu_count = len(expected_devices)
    audit.observed_gpu_count = len(observed)
    # Sidecar per-GPU maps share the hostname/uuid namespace the producer uses
    # for its stored evidence, so role/energy/gap entries join on one key;
    # fall back to hostname/index for devices that were never observed.
    uuid_label = {
        device.key: f"{device.hostname}/{device.gpu_uuids[0]}"
        for device in observed
        if len(device.gpu_uuids) == 1
    }
    audit.per_gpu_role = {
        uuid_label.get(key, f"{key[0]}/{key[1]}"): role
        for key, role in sorted(roles.items())
    }

    # Gate: exactly one completed window must belong to THIS processed result.
    # The workspace copy was renamed by the launcher, so the link is by content:
    # the original result at LOGS/<result_path> must equal the processed copy.
    window = _select_window_for_result(
        audit, parsed_windows, logs_root, bench_result_path, benchmark
    )

    if audit.reasons or window is None or benchmark is None:
        return audit

    # Integration: per-GPU trapezoid clipped to the formal window; roles sum.
    per_key_samples: dict[tuple[str, int], list[tuple[float, float]]] = {}
    for row in rows:
        per_key_samples.setdefault((row.hostname, row.gpu_index), []).append(
            (row.timestamp_unix, row.power_w)
        )
    by_key = {device.key: device for device in observed}
    per_gpu_energy: dict[str, float] = {}
    per_gpu_gap: dict[str, float] = {}
    role_energy: dict[str, float] = {"prefill": 0.0, "decode": 0.0}
    for device in expected_devices:
        samples = sorted(per_key_samples.get(device.key, []))
        label = uuid_label[device.key]
        energy = _integrate_device(
            samples, start_unix=window.start_unix, end_unix=window.end_unix
        )
        per_gpu_energy[label] = energy
        role = roles.get(device.key)
        if role in role_energy:
            role_energy[role] += energy
        sequence = _bracketing_sequence(
            by_key[device.key].sample_times, window.start_unix, window.end_unix
        )
        per_gpu_gap[label] = max(
            (later - earlier for earlier, later in itertools.pairwise(sequence)), default=0.0
        )

    audit.per_gpu_energy_j = per_gpu_energy
    audit.per_gpu_max_sample_gap_s = per_gpu_gap

    duration_s = window.end_unix - window.start_unix
    total_energy = sum(per_gpu_energy.values())
    total_tokens = benchmark.total_input_tokens + benchmark.total_output_tokens
    metrics = {
        "avg_power_w": total_energy / duration_s / len(expected_devices),
        "avg_total_gpu_power_w": total_energy / duration_s,
        "total_gpu_energy_j": total_energy,
        "joules_per_successful_query": total_energy / benchmark.completed,
        "joules_per_input_token": total_energy / benchmark.total_input_tokens,
        "joules_per_output_token": total_energy / benchmark.total_output_tokens,
        "joules_per_total_token": total_energy / total_tokens,
    }
    if prefill_gpus > 0:
        metrics["prefill_gpu_energy_j"] = role_energy["prefill"]
        metrics["prefill_avg_power_w"] = role_energy["prefill"] / duration_s / prefill_gpus
        metrics["prefill_joules_per_input_token"] = (
            role_energy["prefill"] / benchmark.total_input_tokens
        )
    if decode_gpus > 0:
        metrics["decode_gpu_energy_j"] = role_energy["decode"]
        metrics["decode_avg_power_w"] = role_energy["decode"] / duration_s / decode_gpus
        metrics["decode_joules_per_output_token"] = (
            role_energy["decode"] / benchmark.total_output_tokens
        )
    # Absurd-but-finite sample values can overflow to inf during integration;
    # publish an invalid verdict instead of dying in _patch_agg with a stale agg.
    non_finite = sorted(key for key, value in metrics.items() if not math.isfinite(value))
    if non_finite:
        _add_reason(audit, "non_finite_power_metric", f"non-finite: {', '.join(non_finite)}")
        return audit
    audit.metrics = metrics
    audit.power_valid = True
    return audit


def _select_window_for_result(
    audit: MultinodePowerAudit,
    parsed_windows: dict[tuple[str, int], ParsedWindow],
    logs_root: Path,
    bench_result_path: Path,
    benchmark: BenchmarkData | None,
) -> ParsedWindow | None:
    """Bind the processed result to its completed window by content, not name."""
    try:
        processed = json.loads(bench_result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        _add_reason(audit, "invalid_benchmark_result", f"unreadable result: {bench_result_path}")
        return None
    if not isinstance(processed, dict):
        _add_reason(audit, "invalid_benchmark_result", "processed result is not an object")
        return None

    concurrency = processed.get("max_concurrency")
    matches = [
        window
        for window in parsed_windows.values()
        if window.status == WINDOW_STATUS_COMPLETED and window.concurrency == concurrency
    ]
    if len(matches) != 1:
        _add_reason(
            audit,
            "window_for_result_missing",
            f"found {len(matches)} completed windows for concurrency {concurrency!r}",
        )
        return None
    window = matches[0]

    original_path = logs_root / window.result_path
    try:
        original = json.loads(original_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        _add_reason(
            audit, "measurement_window_result_missing", f"original result unreadable: {original_path}"
        )
        return None
    if _canonical_sha256(original) != _canonical_sha256(processed):
        _add_reason(
            audit,
            "result_content_mismatch",
            f"workspace copy {bench_result_path.name} differs from original {window.result_path}",
        )
        return None

    if benchmark is not None and (
        benchmark.start_unix != window.start_unix or benchmark.end_unix != window.end_unix
    ):
        _add_reason(
            audit,
            "measurement_window_result_mismatch",
            "benchmark window fields do not match the selected window file",
        )
        return None

    audit.window = {
        "window_file": window.relative_path,
        "result_path": window.result_path,
        "benchmark_type": window.benchmark_type,
        "concurrency": window.concurrency,
        "start_time_unix": window.start_unix,
        "end_time_unix": window.end_unix,
        "duration": window.duration,
    }
    return window


# --- aggregate patch + sidecar + entry point ---------------------------------


def _patch_agg(agg_path: Path, audit: MultinodePowerAudit) -> None:
    data = json.loads(agg_path.read_text(encoding="utf-8"))
    for key in _ALL_POWER_METRIC_KEYS:
        data.pop(key, None)
    data["power_metric_schema_version"] = POWER_METRIC_SCHEMA_VERSION
    data["power_valid"] = int(audit.power_valid)
    data.pop("power_invalid_reasons", None)
    if audit.power_valid:
        for key, value in audit.metrics.items():
            if value is None or not math.isfinite(value):
                raise ValueError(f"non-finite power metric: {key}")
            precision = 3 if key.endswith(("_w", "_j")) else 6
            data[key] = round(value, precision)
    _write_json_atomic(agg_path, data)


def _sidecar_payload(
    *,
    audit: MultinodePowerAudit,
    power_dir: Path,
    bench_result: Path,
    benchmark: BenchmarkData | None,
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
        "power_valid": audit.power_valid,
        "reasons": list(audit.reasons),
        "failures": list(audit.failures),
        "telemetry_source": str(power_dir),
        "benchmark_result": str(bench_result),
        "benchmark_window": benchmark_window,
        "selected_window": audit.window,
        "integration_method": _INTEGRATION_METHOD,
        "producer": {
            "producer_git_commit": audit.producer_git_commit,
            "expected_producer_git_commit": audit.expected_producer_git_commit,
            "exporter_image_sha256": audit.exporter_image_sha256,
            "stored_publication_valid": audit.stored_publication_valid,
            "recomputed_publication_valid": audit.recomputed_publication_valid,
        },
        "expected_gpu_count": audit.expected_gpu_count,
        "observed_gpu_count": audit.observed_gpu_count,
        "per_gpu_role": audit.per_gpu_role,
        "per_gpu_max_sample_gap_s": audit.per_gpu_max_sample_gap_s,
        # Overflowed integrations reach here as inf; json.dumps would emit a
        # bare Infinity token, which strict RFC 8259 parsers reject.
        "per_gpu_energy_j": {
            key: value if math.isfinite(value) else None
            for key, value in audit.per_gpu_energy_j.items()
        },
        "metrics": {
            key: round(value, 6)
            for key, value in audit.metrics.items()
            if value is not None and math.isfinite(value)
        },
    }


def run(
    power_dir: Path,
    bench_result: Path,
    agg_result: Path,
    *,
    prefill_gpus: int,
    decode_gpus: int,
    expected_producer_sha: str | None,
    logs_root: Path,
    aggregate_gpus: int = 0,
    validation_result: Path | None = None,
    require_power: bool = False,
) -> int:
    """Validate multinode power artifacts, patch metrics, and audit the verdict."""
    validation_result = validation_result or bench_result.with_name(
        f"power_validation_{bench_result.stem}.json"
    )
    benchmark, benchmark_reasons = _load_benchmark_data(bench_result)
    # Integration/normalization must only ever see a fully valid benchmark
    # contract; a partially valid one (e.g. zero token counts) is invalid
    # anyway and would divide by zero.
    effective_benchmark = benchmark if benchmark is not None and not benchmark_reasons else None

    audit = validate_and_integrate(
        power_dir=power_dir,
        logs_root=logs_root,
        bench_result_path=bench_result,
        benchmark=effective_benchmark,
        prefill_gpus=prefill_gpus,
        decode_gpus=decode_gpus,
        aggregate_gpus=aggregate_gpus,
        expected_producer_sha=expected_producer_sha,
    )
    for reason in benchmark_reasons:
        _add_reason(audit, reason)
    if benchmark is None or audit.reasons:
        audit.power_valid = False
        audit.metrics = {}

    if not agg_result.is_file():
        _add_reason(audit, "aggregate_result_missing", f"aggregate missing: {agg_result}")
        audit.power_valid = False
        audit.metrics = {}
    else:
        try:
            _patch_agg(agg_result, audit)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            _add_reason(audit, "aggregate_result_unwritable", str(exc))
            audit.power_valid = False
            audit.metrics = {}
            print(
                f"[aggregate_power_multinode] Failed to patch {agg_result}: {exc}",
                file=sys.stderr,
            )

    try:
        _write_json_atomic(
            validation_result,
            _sidecar_payload(
                audit=audit,
                power_dir=power_dir,
                bench_result=bench_result,
                benchmark=benchmark,
            ),
        )
    except OSError as exc:
        print(
            f"[aggregate_power_multinode] Failed to write validation artifact "
            f"{validation_result}: {exc}",
            file=sys.stderr,
        )
        return 1 if require_power else 0

    if not audit.power_valid:
        print(
            f"[aggregate_power_multinode] Power validation failed: "
            f"{', '.join(audit.reasons)} (details: {validation_result})",
            file=sys.stderr,
        )
        return 1 if require_power else 0

    print(
        f"[aggregate_power_multinode] avg_power_w={audit.metrics['avg_power_w']:.2f} "
        f"total_gpu_energy_j={audit.metrics['total_gpu_energy_j']:.2f} "
        f"prefill_gpu_energy_j={audit.metrics.get('prefill_gpu_energy_j', 0.0):.2f} "
        f"decode_gpu_energy_j={audit.metrics.get('decode_gpu_energy_j', 0.0):.2f} "
        f"(n={audit.expected_gpu_count}) -> {agg_result}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--power-dir",
        type=Path,
        default=Path(os.environ.get("POWER_ARTIFACT_DIR", "LOGS/power")),
        help="Path to the srt-slurm power artifact dir with manifest.json, samples.csv, "
        "windows/ (default: POWER_ARTIFACT_DIR or LOGS/power)",
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path(os.environ.get("POWER_RESULT_ROOT", "LOGS")),
        help="Root directory that window result_path values resolve under "
        "(default: POWER_RESULT_ROOT or LOGS)",
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
        "--prefill-gpus",
        type=int,
        required=True,
        help="Expected prefill GPU count from the workflow's PREFILL_GPUS",
    )
    parser.add_argument(
        "--decode-gpus",
        type=int,
        required=True,
        help="Expected decode GPU count from the workflow's DECODE_GPUS",
    )
    parser.add_argument(
        "--aggregate-gpus",
        type=int,
        default=0,
        help="Expected aggregate-worker GPU count (mutually exclusive with prefill/decode)",
    )
    parser.add_argument(
        "--expected-producer-sha",
        default=os.environ.get("POWER_PRODUCER_SHA") or None,
        help="Required srt-slurm producer git commit (env POWER_PRODUCER_SHA)",
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
        args.power_dir,
        args.bench_result,
        args.agg_result,
        prefill_gpus=args.prefill_gpus,
        decode_gpus=args.decode_gpus,
        aggregate_gpus=args.aggregate_gpus,
        expected_producer_sha=args.expected_producer_sha,
        logs_root=args.logs_root,
        validation_result=args.validation_result,
        require_power=args.require_power,
    )


if __name__ == "__main__":
    sys.exit(main())
