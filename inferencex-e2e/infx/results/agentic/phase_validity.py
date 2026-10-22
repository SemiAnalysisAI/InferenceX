from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .common import to_float

VALID = "VALID"
UNKNOWN = "UNKNOWN"
BOUNDARY_RESIDUAL = "BOUNDARY_RESIDUAL"
BOUNDARY_UNCERTAIN = "BOUNDARY_UNCERTAIN"
PARTIAL_COVERAGE = "PARTIAL_COVERAGE"
PHASE_CONTAMINATED = "PHASE_CONTAMINATED"
REJECTED = "REJECTED"
_SEVERITY = {
    VALID: 0,
    UNKNOWN: 1,
    BOUNDARY_RESIDUAL: 2,
    BOUNDARY_UNCERTAIN: 3,
    PARTIAL_COVERAGE: 4,
    PHASE_CONTAMINATED: 5,
    REJECTED: 6,
}
# Rate built only from VALID series: an interval measurement, not a request-bound actual.
CONDITIONAL_INTERVAL = "CONDITIONAL_INTERVAL"

NS_PER_S = 1_000_000_000
COUNT_TOLERANCE = 0.5
SUM_RELATIVE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class PhaseWindows:
    profiling: tuple[int, int]
    warmup: tuple[int, int] | None

    def as_dict(self) -> dict[str, list[int] | None]:
        return {
            "profiling": list(self.profiling),
            "warmup": list(self.warmup) if self.warmup else None,
        }


@dataclass(frozen=True)
class PhaseContext:
    """What a backend needs to judge its own series; kept out of the published result.

    ``unscraped`` is None when the export does not say which endpoints were
    configured and scraped. ``disaggregated`` is None when no authoritative
    deployment mode was supplied.
    """

    windows: PhaseWindows | None
    warmup_totals: Mapping[tuple, float | None] = field(default_factory=dict)
    unscraped: tuple[str, ...] | None = ()
    disaggregated: bool | None = None

    def has_reference(self, metric: str, series: Mapping[str, Any]) -> bool:
        return series_key(metric, series) in self.warmup_totals

    def warmup_total(self, metric: str, series: Mapping[str, Any]) -> float | None:
        return self.warmup_totals.get(series_key(metric, series))


def base_name(metric: str) -> str:
    return metric.removesuffix("_total")


def series_key(metric: str, series: Mapping[str, Any]) -> tuple:
    labels = series.get("labels") if isinstance(series.get("labels"), Mapping) else {}
    return (
        base_name(metric),
        str(series.get("endpoint_url", "")),
        tuple(sorted((str(k), str(v)) for k, v in labels.items())),
    )


def phase_windows(server_metrics: Any) -> PhaseWindows | None:
    summary = server_metrics.get("summary") if isinstance(server_metrics, Mapping) else None
    ranges = summary.get("phase_time_ranges") if isinstance(summary, Mapping) else None
    if not isinstance(ranges, Mapping):
        return None
    profiling = _time_range(ranges.get("profiling"))
    if profiling is None:
        return None
    return PhaseWindows(profiling, _time_range(ranges.get("warmup")))


def warmup_totals(server_metrics: Any) -> dict[tuple, float | None]:
    """Warmup-phase delta of every series exported under ``warmup_metrics``.

    Presence means the series has a sample before its first PROFILING sample; a
    zero total means no increment was counted during warmup.
    """
    warmup = server_metrics.get("warmup_metrics") if isinstance(server_metrics, Mapping) else None
    totals: dict[tuple, float | None] = {}
    for name, entry in warmup.items() if isinstance(warmup, Mapping) else ():
        if not isinstance(entry, Mapping):
            continue
        key_name = "count" if entry.get("type") == "histogram" else "total"
        series_list = entry.get("series")
        for series in series_list if isinstance(series_list, list) else ():
            if isinstance(series, Mapping):
                stats = series.get("stats") if isinstance(series.get("stats"), Mapping) else {}
                totals[series_key(name, series)] = to_float(stats.get(key_name))
    return totals


def export_context(server_metrics: Any, windows: PhaseWindows | None) -> dict[str, Any]:
    summary = server_metrics.get("summary") if isinstance(server_metrics, Mapping) else None
    summary = summary if isinstance(summary, Mapping) else {}
    configured = summary.get("endpoints_configured")
    successful = summary.get("endpoints_successful")
    if isinstance(configured, list) and isinstance(successful, list):
        scraped = {str(url) for url in successful}
        endpoints = {
            "configured": len(configured),
            "unscraped": [str(url) for url in configured if str(url) not in scraped],
        }
    else:
        endpoints = {"configured": None, "unscraped": None}
    return {"phase_windows": windows.as_dict() if windows else None, "endpoints": endpoints}


def worst(statuses: Iterable[str]) -> str:
    return max(statuses, key=_SEVERITY.__getitem__, default=VALID)


def classify_series(
    series: Mapping[str, Any],
    metric_type: str,
    windows: PhaseWindows | None,
    *,
    has_reference: bool = False,
    warmup_total: float | None = None,
) -> dict[str, Any]:
    histogram = metric_type == "histogram"
    keys = ("sum", "count") if histogram else ("total",)
    stats = series.get("stats")
    stats = stats if isinstance(stats, Mapping) else {}
    totals = {key: to_float(stats.get(key)) for key in keys}
    raw_slices = series.get("timeslices")
    slices = (
        [s for s in raw_slices if isinstance(s, Mapping)] if isinstance(raw_slices, list) else []
    )
    findings: list[tuple[str, str]] = []
    record: dict[str, Any] = {
        "endpoint_url": series.get("endpoint_url"),
        "total": totals[keys[0]],
        "slice_sum": None,
        "residual": None,
        "first_slice_offset_s": None,
        "last_slice_offset_s": None,
    }

    for value in totals.values():
        if value is None:
            findings.append((UNKNOWN, "missing_total"))
        elif not math.isfinite(value):
            findings.append((REJECTED, "nonfinite_total"))
        elif value < 0:
            findings.append((REJECTED, "negative_total"))
    if windows is None:
        findings.append((UNKNOWN, "phase_window_missing"))
    if not slices:
        findings.append((UNKNOWN, "no_timeslices"))
    else:
        findings.extend(
            _slice_findings(slices, windows, record, histogram, has_reference, warmup_total)
        )
        for key in keys:
            slice_sum = _slice_sum(slices, key, histogram)
            if slice_sum is None:
                findings.append((REJECTED, "negative_or_nonfinite_slice"))
                continue
            total = totals[key]
            if key == keys[0]:
                record["slice_sum"] = slice_sum
            if total is None or not math.isfinite(total):
                continue
            residual = total - slice_sum
            if key == keys[0]:
                record["residual"] = residual
            tolerance = COUNT_TOLERANCE
            if key == "sum":
                tolerance = max(COUNT_TOLERANCE, SUM_RELATIVE_TOLERANCE * abs(total))
            if residual > tolerance:
                findings.append((BOUNDARY_RESIDUAL, "total_exceeds_slice_sum"))
            elif residual < -tolerance:
                findings.append((REJECTED, "slice_sum_exceeds_total"))

    record["status"] = worst(status for status, _ in findings)
    record["reasons"] = list(dict.fromkeys(code for _, code in findings))
    return record


def summarize(
    metrics: Mapping[str, Any],
    windows: PhaseWindows | None,
    warmup: Mapping[tuple, float | None] | None = None,
) -> dict[str, Any]:
    warmup = warmup or {}
    series_by_status: Counter[str] = Counter()
    metrics_by_status: dict[str, list[str]] = {}
    for name in sorted(metrics):
        entry = metrics[name]
        if not isinstance(entry, Mapping) or entry.get("type") not in ("counter", "histogram"):
            continue
        series_list = entry.get("series")
        if not isinstance(series_list, list):
            continue
        statuses = [
            classify_series(
                series,
                entry["type"],
                windows,
                has_reference=series_key(name, series) in warmup,
                warmup_total=warmup.get(series_key(name, series)),
            )["status"]
            for series in series_list
            if isinstance(series, Mapping)
        ]
        series_by_status.update(statuses)
        status = worst(statuses)
        if statuses and status != VALID:
            metrics_by_status.setdefault(status, []).append(name)
    return {
        "series_by_status": dict(series_by_status),
        "metrics_by_status": metrics_by_status,
    }


def resolve_aliases(
    metrics: Mapping[str, Any], base: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Series of ``base`` and its Prometheus ``_total`` alias, one per endpoint and label set.

    Cross-name aliases collapse only when they are the same observation (equal
    ``stats`` and ``timeslices``); any other overlap is a conflict, whatever
    the order. Repeated series under one name are reported as duplicates. Both
    are dropped so they are never summed.
    """
    groups: dict[tuple, list[tuple[str, Mapping[str, Any]]]] = {}
    for name in (base, f"{base}_total"):
        entry = metrics.get(name)
        series_list = entry.get("series") if isinstance(entry, Mapping) else None
        for series in series_list if isinstance(series_list, list) else []:
            if isinstance(series, Mapping):
                groups.setdefault(series_key(name, series), []).append((name, series))
    chosen: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for key, items in groups.items():
        names = [name for name, _ in items]
        if len(items) == 1:
            chosen.append(items[0][1])
            continue
        kind = "duplicate_series" if len(set(names)) < len(names) else "alias_conflict"
        first = items[0][1]
        if kind == "alias_conflict" and all(_same_observation(first, s) for _, s in items[1:]):
            chosen.append(first)
            continue
        conflicts.append(
            {"metric": base, "endpoint_url": key[1], "names": sorted(names), "kind": kind}
        )
    return chosen, conflicts


def _time_range(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, Mapping):
        return None
    start, end = value.get("start_ns"), value.get("end_ns")
    if not isinstance(start, int) or not isinstance(end, int) or start >= end:
        return None
    return start, end


def _slice_findings(
    slices: list[Mapping[str, Any]],
    windows: PhaseWindows | None,
    record: dict[str, Any],
    histogram: bool,
    has_reference: bool,
    warmup_total: float | None,
) -> list[tuple[str, str]]:
    bounds = [(s.get("start_ns"), s.get("end_ns")) for s in slices]
    if not all(isinstance(a, int) and isinstance(b, int) for a, b in bounds):
        return [(REJECTED, "slice_bounds_missing")]
    findings: list[tuple[str, str]] = []
    if any(end <= start for start, end in bounds):
        findings.append((REJECTED, "slice_nonpositive_duration"))
    if any(bounds[i][0] < bounds[i - 1][1] for i in range(1, len(bounds))):
        findings.append((REJECTED, "slice_overlap_or_backward"))
    if any(bounds[i][0] > bounds[i - 1][1] for i in range(1, len(bounds))):
        findings.append((REJECTED, "slice_gap"))
    if findings or windows is None:
        return findings

    start, end = windows.profiling
    first, last = bounds[0][0], bounds[-1][1]
    step = max(b - a for a, b in bounds)
    activity = [_slice_activity(s, histogram) for s in slices]
    before = sum(v for (a, b), v in zip(bounds, activity, strict=True) if b <= start)
    straddle_start = sum(v for (a, b), v in zip(bounds, activity, strict=True) if a < start < b)
    straddle_end = sum(v for (a, b), v in zip(bounds, activity, strict=True) if a < end < b)
    after = sum(v for (a, b), v in zip(bounds, activity, strict=True) if a >= end)
    record.update(
        first_slice_offset_s=(first - start) / NS_PER_S,
        last_slice_offset_s=(last - end) / NS_PER_S,
        activity_before_profiling=before,
        activity_straddling_start=straddle_start,
        activity_straddling_end=straddle_end,
        activity_after_profiling=after,
    )
    # A window that opens before the phase holds PROFILING-tagged samples from
    # before it. Increments between the last of those and the first in-phase
    # sample land in whichever slice holds that later sample, so slice position
    # cannot clear them; only a zero warmup-phase total can.
    if first < start:
        if before > 0 or (straddle_start > 0 and first <= start - step):
            findings.append((PHASE_CONTAMINATED, "activity_before_profiling"))
        elif warmup_total != 0 and first <= start - step:
            findings.append((PHASE_CONTAMINATED, "window_opens_before_profiling"))
        elif straddle_start > 0:
            findings.append((BOUNDARY_UNCERTAIN, "activity_straddles_profiling_start"))
        elif warmup_total != 0:
            findings.append((BOUNDARY_UNCERTAIN, "window_opens_before_profiling"))
    if after > 0:
        findings.append((PHASE_CONTAMINATED, "activity_after_profiling"))
    elif straddle_end > 0:
        findings.append((BOUNDARY_UNCERTAIN, "activity_straddles_profiling_end"))
    # A late first slice is covered only if a reference sample exists, so that
    # stats.total spans the gap and the residual check sees any increment in it.
    if first > start and not has_reference:
        findings.append((PARTIAL_COVERAGE, "window_starts_after_profiling"))
    if last < end:
        findings.append((PARTIAL_COVERAGE, "window_ends_before_profiling"))
    return findings


def _slice_activity(item: Mapping[str, Any], histogram: bool) -> float:
    value = to_float(item.get("count" if histogram else "total"))
    return value if value is not None and math.isfinite(value) and value > 0 else 0.0


def _slice_sum(slices: list[Mapping[str, Any]], key: str, histogram: bool) -> float | None:
    total = 0.0
    for item in slices:
        if histogram:
            count = to_float(item.get("count"))
            value = count if key == "count" else _product(count, to_float(item.get("avg")))
        else:
            value = to_float(item.get("total"))
        if value is None or not math.isfinite(value) or value < 0:
            return None
        total += value
    return total


def _product(count: float | None, avg: float | None) -> float | None:
    if count is None or avg is None:
        return None
    return count * avg


def _same_observation(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return left.get("stats") == right.get("stats") and left.get("timeslices") == right.get(
        "timeslices"
    )
