"""Best-effort Grace CPU-side power leg of the multinode srt-slurm package.

srt-slurm's ``cpu_power_exporter`` leg writes ``power/cpu/samples.csv`` and a
non-authoritative ``power/cpu/cpu_manifest.json`` beside the GPU DCGM package
(srt-slurm ``src/srtctl/core/power/cpu_session.py``). Historical v2.2.1
packages use the v1 long format, one row per sensor reading. The current
producer uses the v2 wide format, one row per (scrape, host, socket) with
the ACPI component rails as reference columns. Both are accepted here.

Each (hostname, socket) headline series is integrated over the GPU leg's
bound formal window with the shared trapezoid and boundary interpolation and
published as additive metrics next to the GPU-board numbers. The v2 ACPI
CPU-only and SysIO rails are integrated the same way when every socket reports
them; a rail with gaps drops only its own metrics. The leg is
best-effort by contract: any failure records ``cpu_power_valid=0`` with
reason codes and leaves every GPU field untouched. Strict qualification also
requires a recipe-declared CPU source; GPU-only callers keep best-effort CPU
behavior. A package without ``cpu/`` emits nothing unless a source is expected.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .common import _append_reason, _bracketing_sequence, _integrate_device, audit_metrics

CPU_DIRNAME = "cpu"
CPU_SAMPLES_FILENAME = "samples.csv"
CPU_MANIFEST_FILENAME = "cpu_manifest.json"

# srt-slurm contract.py: CPU_SAMPLES_HEADER_V1 (written by v2.2.1) and the
# wide CPU_SAMPLES_HEADER (written by v2.15.0, which reads both).
CPU_SAMPLES_HEADER_V1 = (
    "schema_version",
    "timestamp_unix",
    "hostname",
    "source",
    "sensor",
    "socket_id",
    "power_w",
    "total_power_w",
)
CPU_SAMPLES_HEADER_V2 = (
    *CPU_SAMPLES_HEADER_V1[:7],
    "cpu_rail_w",
    "soc_w",
    "dram_w",
    "total_power_w",
)
_HEADERS = {1: CPU_SAMPLES_HEADER_V1, 2: CPU_SAMPLES_HEADER_V2}

# One GB200/GB300 compute tray per worker host, two Grace sockets per tray.
SOCKETS_PER_HOST = 2
# Shared with the GPU leg (srt-slurm contract.MAX_SAMPLE_GAP_SECONDS).
MAX_SAMPLE_GAP_SECONDS = 3.0

SENSOR_MODULE = "module"
SENSOR_GRACE = "grace_socket"
SENSOR_DCGM = "dcgm_cpu_rail"
# Headline preference per socket. ACPI component rails ride on the Grace row
# and never select the headline or its verdict.
HEADLINE_PREFERENCE = (SENSOR_MODULE, SENSOR_GRACE, SENSOR_DCGM)
# v2 Grace-row column -> published rail family: "CPU Power Socket N" (cores,
# no SysIO) and "SysIO Power Socket N". dram_w stays blank on NVL72 firmware.
_RAIL_COLUMNS = {"cpu_rail_w": "cpu_rail", "soc_w": "cpu_sysio"}
# Grace-side kinds, in fallback order, that feed the ``*_cpu_*`` keys.
_GRACE_SIDE_PREFERENCE = (SENSOR_GRACE, SENSOR_DCGM)

# ``CPU<socket>:<suffix>`` sensor names (srt-slurm cpu_rails.SENSOR_SUFFIXES)
# and firmware OEM labels (srt-slurm cpu_rails.ACPI_LABEL_PATTERNS for the
# Grace total). The Module label is the NVL72 tray firmware's whole-module
# sensor, which srt-slurm does not classify yet; a v2.2.1 package therefore
# yields the Grace socket total until the exporter learns the label.
_SUFFIX_KINDS = {"cpuSidePowerUsageW": SENSOR_GRACE, "cpuPowerUsageW": SENSOR_DCGM}
_LABEL_KINDS = (
    (SENSOR_MODULE, re.compile(r"\bModule\s+Power\s+Socket\s+\d+\b", re.IGNORECASE)),
    (SENSOR_GRACE, re.compile(r"\bGrace\s+Power\s+Socket\s+\d+\b", re.IGNORECASE)),
    (
        SENSOR_GRACE,
        re.compile(r"\bTotal(?:\s+Input)?\s+Power(?:\s+in\s+uW)?\s+Socket\s+\d+\b", re.IGNORECASE),
    ),
)

_INTEGRATION_METHOD = "per_socket_trapezoidal_with_linear_boundary_interpolation"


@dataclass(frozen=True)
class CpuSampleRow:
    timestamp_unix: float
    hostname: str
    source: str
    sensor: str
    socket_id: int
    power_w: float
    rails: tuple[tuple[str, float], ...] = ()


@dataclass
class CpuPowerAudit:
    """Everything the sidecar needs to explain the CPU-side verdict."""

    valid: bool = False
    sensor_kind: str | None = None
    source: str | None = None
    expected_sockets: int = 0
    observed_sockets: int = 0
    sample_row_count: int = 0
    reason_codes: list[str] = field(default_factory=list)
    per_series_energy_j: dict[str, float] = field(default_factory=dict)
    per_series_max_sample_gap_s: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)

    def invalidate(self, *reasons: str) -> None:
        for reason in reasons:
            _append_reason(self.reason_codes, reason)
        self.valid = False
        self.metrics = {}

    def to_payload(self) -> dict[str, Any]:
        return {
            "cpu_power_valid": self.valid,
            "reason_codes": list(self.reason_codes),
            "sensor_kind": self.sensor_kind,
            "source": self.source,
            "sockets_per_host": SOCKETS_PER_HOST,
            "expected_sockets": self.expected_sockets,
            "observed_sockets": self.observed_sockets,
            "sample_row_count": self.sample_row_count,
            "headline_preference": list(HEADLINE_PREFERENCE),
            "integration_method": _INTEGRATION_METHOD,
            "max_sample_gap_seconds": MAX_SAMPLE_GAP_SECONDS,
            "per_series_max_sample_gap_s": dict(self.per_series_max_sample_gap_s),
            # Overflowed integrations reach here as inf; keep the sidecar strict JSON.
            "per_series_energy_j": {
                key: value if math.isfinite(value) else None
                for key, value in self.per_series_energy_j.items()
            },
            "metrics": audit_metrics(self.metrics),
        }


def classify_sensor(sensor: str) -> str | None:
    """Headline kind for a ``sensor`` cell from either writer version, else None."""
    _, _, suffix = sensor.partition(":")
    if suffix in _SUFFIX_KINDS:
        return _SUFFIX_KINDS[suffix]
    return next((kind for kind, pattern in _LABEL_KINDS if pattern.search(sensor)), None)


def _parse_row(raw: list[str], version: int) -> CpuSampleRow | None:
    if len(raw) != len(_HEADERS[version]):
        return None
    try:
        schema_version = int(raw[0])
        timestamp_unix = float(raw[1])
        socket_id = int(raw[5])
        power_w = float(raw[6])
    except ValueError:
        return None
    hostname, source, sensor = raw[2], raw[3], raw[4]
    if schema_version != version or not hostname or not source or not sensor:
        return None
    if not math.isfinite(timestamp_unix) or not math.isfinite(power_w) or power_w < 0:
        return None
    if socket_id < 0:
        return None
    rails = []
    for column, family in _RAIL_COLUMNS.items():
        cell = raw[_HEADERS[version].index(column)] if column in _HEADERS[version] else ""
        if not cell:
            continue
        try:
            watts = float(cell)
        except ValueError:
            return None
        if not math.isfinite(watts) or watts < 0:
            return None
        rails.append((family, watts))
    return CpuSampleRow(
        timestamp_unix=timestamp_unix,
        hostname=hostname,
        source=source,
        sensor=sensor,
        socket_id=socket_id,
        power_w=power_w,
        rails=tuple(rails),
    )


def read_cpu_samples(path: Path) -> tuple[tuple[CpuSampleRow, ...], tuple[str, ...]]:
    """Strictly parse either sample generation; malformed rows invalidate the leg."""
    if not path.is_file():
        return (), ("cpu_samples_missing",)
    rows: list[CpuSampleRow] = []
    reasons: list[str] = []
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            version = next((v for v, h in _HEADERS.items() if header == list(h)), None)
            if version is None:
                return (), ("cpu_samples_header_mismatch",)
            for raw in reader:
                row = _parse_row(raw, version)
                if row is None:
                    _append_reason(reasons, "cpu_samples_malformed")
                    continue
                rows.append(row)
    except (OSError, UnicodeDecodeError, csv.Error):
        _append_reason(reasons, "cpu_samples_malformed")
    return tuple(rows), tuple(reasons)


def _manifest_is_object(path: Path) -> bool:
    try:
        return isinstance(json.loads(path.read_text(encoding="utf-8")), dict)
    except (OSError, UnicodeDecodeError, ValueError):
        return False


SocketKey = tuple[str, int]


def _headline_series(
    rows: Iterable[CpuSampleRow],
) -> tuple[dict[SocketKey, dict[str, dict[float, float]]], dict[str, set[str]], bool]:
    """Group headline rows as (host, socket) -> kind -> {timestamp: watts}."""
    series: dict[SocketKey, dict[str, dict[float, float]]] = {}
    sources: dict[str, set[str]] = {}
    duplicate = False
    for row in rows:
        kind = classify_sensor(row.sensor)
        if kind is None:
            continue
        by_kind = series.setdefault((row.hostname, row.socket_id), {})
        points = by_kind.setdefault(kind, {})
        if row.timestamp_unix in points:
            duplicate = True
        points[row.timestamp_unix] = row.power_w
        sources.setdefault(kind, set()).add(row.source)
        if kind == SENSOR_GRACE:
            for family, watts in row.rails:
                by_kind.setdefault(family, {})[row.timestamp_unix] = watts
    return series, sources, duplicate


def _select_feeds(common_kinds: set[str]) -> tuple[str | None, dict[str, str]]:
    """Pick the headline kind and which kind feeds each metric family."""
    selected = next((kind for kind in HEADLINE_PREFERENCE if kind in common_kinds), None)
    if selected is None:
        return None, {}
    if selected != SENSOR_MODULE:
        return selected, {selected: "cpu"}
    feeds = {SENSOR_MODULE: "module"}
    grace = next((kind for kind in _GRACE_SIDE_PREFERENCE if kind in common_kinds), None)
    if grace is not None:
        feeds[grace] = "cpu"
    return selected, feeds


def validate_cpu_leg(
    cpu_dir: Path,
    *,
    window: tuple[float, float] | None,
    expected_hosts: Iterable[str],
    expected_source: str | None = None,
) -> CpuPowerAudit | None:
    """Validate and integrate the CPU sub-package; None when the run carried none."""
    if not cpu_dir.is_dir():
        if expected_source is None:
            return None
        audit = CpuPowerAudit()
        audit.invalidate("cpu_artifacts_missing")
        return audit
    audit = CpuPowerAudit()
    if not _manifest_is_object(cpu_dir / CPU_MANIFEST_FILENAME):
        audit.invalidate("cpu_manifest_invalid")
    rows, sample_reasons = read_cpu_samples(cpu_dir / CPU_SAMPLES_FILENAME)
    audit.sample_row_count = len(rows)
    audit.invalidate(*sample_reasons)
    series, sources, duplicate = _headline_series(rows)
    if duplicate:
        audit.invalidate("cpu_samples_malformed")
    expected_keys = {
        (host, socket) for host in expected_hosts for socket in range(SOCKETS_PER_HOST)
    }
    audit.expected_sockets = len(expected_keys)
    audit.observed_sockets = len(series)
    if window is None:
        audit.invalidate("cpu_window_unavailable")
    if audit.reason_codes:
        return audit

    if not expected_keys or set(series) != expected_keys:
        audit.invalidate("cpu_socket_count_mismatch")
        return audit
    common_kinds = set.intersection(*(set(kinds) for kinds in series.values()))
    selected, feeds = _select_feeds(common_kinds)
    if selected is None or len(sources[selected]) != 1:
        audit.invalidate("cpu_sensor_kind_mixed")
        return audit
    audit.sensor_kind = selected
    audit.source = next(iter(sources[selected]))
    if expected_source is not None and (
        audit.source != expected_source
        or (expected_source == "acpi" and selected not in {SENSOR_GRACE, SENSOR_MODULE})
    ):
        audit.invalidate("cpu_sensor_source_mismatch")
        return audit

    if SENSOR_GRACE in feeds:
        feeds.update({rail: rail for rail in _RAIL_COLUMNS.values() if rail in common_kinds})
    start, end = window
    energy: dict[str, float] = {}
    dropped_rails: set[str] = set()
    for (host, socket), by_kind in sorted(series.items()):
        for kind, family in feeds.items():
            label = f"{host}/socket{socket}/{kind}"
            samples = sorted(by_kind[kind].items())
            sequence = _bracketing_sequence(tuple(t for t, _ in samples), start, end)
            gap = (
                None
                if sequence is None
                else max(
                    (later - earlier for earlier, later in itertools.pairwise(sequence)),
                    default=0.0,
                )
            )
            if gap is not None:
                audit.per_series_max_sample_gap_s[label] = gap
            if gap is None or gap > MAX_SAMPLE_GAP_SECONDS:
                if family in _RAIL_COLUMNS.values():
                    dropped_rails.add(family)
                elif gap is None:
                    audit.invalidate("cpu_window_not_bracketed")
                else:
                    audit.invalidate("cpu_sample_gap_exceeded")
                continue
            joules = _integrate_device(samples, start_unix=start, end_unix=end)
            audit.per_series_energy_j[label] = joules
            energy[family] = energy.get(family, 0.0) + joules
    if audit.reason_codes:
        audit.per_series_energy_j = {}
        return audit
    for family in dropped_rails:
        energy.pop(family, None)

    duration = end - start
    metrics: dict[str, float] = {}
    if "cpu" in energy:
        metrics["avg_cpu_socket_power_w"] = energy["cpu"] / duration / len(expected_keys)
        metrics["avg_total_cpu_power_w"] = energy["cpu"] / duration
        metrics["total_cpu_energy_j"] = energy["cpu"]
    for family in ("module", *_RAIL_COLUMNS.values()):
        if family in energy:
            metrics[f"avg_total_{family}_power_w"] = energy[family] / duration
            metrics[f"total_{family}_energy_j"] = energy[family]
    if any(not math.isfinite(value) for value in metrics.values()):
        audit.invalidate("non_finite_power_metric")
        return audit
    audit.metrics = metrics
    audit.valid = True
    return audit
