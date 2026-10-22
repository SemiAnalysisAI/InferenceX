from __future__ import annotations

import math
from collections.abc import Mapping
from itertools import pairwise
from typing import Any

from .common import metric_series, series_stat

FULL = "prefix_cache_full_tokens"
CACHED = "prefix_cache_cached_tokens"
OFFLOAD = "prefix_cache_offload_tokens"
LOADED = "lmcache_loaded_tokens"
REQUESTS = "prefix_cache_requests"
_COUNTERS = (FULL, CACHED, OFFLOAD, LOADED, REQUESTS)
_CORE = (FULL, CACHED, OFFLOAD)


def _series(section: Mapping[str, Any], stem: str) -> tuple[dict[tuple, dict], set[str | None]]:
    names = [
        f"{prefix}{stem}{suffix}" for prefix in ("atom:", "atom_") for suffix in ("", "_total")
    ]
    found: dict[tuple, dict] = {}
    duplicates: set[str | None] = set()
    for series in metric_series(dict(section), names):
        url, labels = series.get("endpoint_url"), series.get("labels")
        if labels is None:
            labels = {}
        if not isinstance(url, str) or not url or not isinstance(labels, dict):
            duplicates.add(None)
            continue
        key = (url, tuple(sorted((str(k), str(v)) for k, v in labels.items())))
        if key in found:
            duplicates.add(url)
        found[key] = series
    return found, duplicates


def _total(series: dict[str, Any]) -> float | None:
    stats = series.get("stats")
    raw = stats.get("total") if isinstance(stats, dict) else None
    value = series_stat(series, preferred_keys=("total",))
    return (
        value
        if not isinstance(raw, bool) and value is not None and math.isfinite(value) and value >= 0
        else None
    )


def _window_total(
    series: dict[str, Any], window: tuple[int, int], warmup: dict[str, Any] | None
) -> tuple[float | None, bool]:
    slices = series.get("timeslices")
    if not isinstance(slices, list) or not slices:
        return None, False
    parsed = []
    for item in slices:
        if not isinstance(item, dict):
            return None, False
        lo, hi, value = item.get("start_ns"), item.get("end_ns"), item.get("total")
        if (
            type(lo) is not int
            or type(hi) is not int
            or hi <= lo
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            return None, False
        parsed.append((lo, hi, value))
    parsed.sort()
    if any(b[0] < a[1] for a, b in pairwise(parsed)):
        return None, False
    start, end = window
    covering = [item for item in parsed if item[1] > start and item[0] < end]
    if (
        not covering
        or covering[0][0] > start
        or covering[-1][1] < end
        or any(b[0] != a[1] for a, b in pairwise(covering))
    ):
        return None, False
    before = sum(value for lo, hi, value in parsed if hi <= start)
    start_straddle = sum(value for lo, hi, value in parsed if lo < start < hi)
    end_straddle = sum(value for lo, hi, value in parsed if lo < end < hi)
    warm = _total(warmup) if warmup is not None else None
    boundary = end_straddle > 0 or (start_straddle > 0 and before + start_straddle != warm)
    return float(sum(value for lo, hi, value in parsed if start <= lo and hi <= end)), boundary


def _modern_export(export: Mapping[str, Any], endpoints: list[str]) -> bool:
    summary = export.get("summary") or {}
    return (
        isinstance(summary, dict)
        and export.get("schema_version") == "1.1"
        and export.get("aiperf_version") == "1.0.12"
        and export.get("metrics_phase") == "profiling"
        and all(
            isinstance(summary.get(key), list)
            and all(isinstance(url, str) for url in summary[key])
            and set(endpoints) <= set(summary[key])
            for key in ("endpoints_configured", "endpoints_successful")
        )
    )


def account_cache(
    metrics: dict[str, dict[str, Any]],
    *,
    framework: str,
    endpoint_roles: Mapping[str, str] | None,
    profiling_window: tuple[int, int] | None,
    warmup_metrics: Mapping[str, Any] | None = None,
    request_scope: Mapping[str, Any] | None = None,
    disagg: bool | None = None,
    kv_offloading: str | None = None,
    collector_export: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Qualify counter totals against the profiling window and successful request cohort."""
    result: dict[str, Any] = dict.fromkeys(
        ("gpu", "cpu", "overall", "full", "cached", "offload", "loaded", "requests")
    )
    reasons: list[str] = []
    validity: dict[str, Any] = {
        "schema": "atom-cache-validity/1",
        "status": None,
        "pricing_eligible": False,
        "observed_scope": None,
        "requested_scope": "profiling_successful_requests",
        "cpu_component": None,
        "roles_source": None,
        "accounted_endpoints": [],
        "window": None,
        "request_scope": None,
        "diagnostics": {},
        "reasons": reasons,
    }
    result.update(validity=validity, warnings=[], replace_telemetry=False)

    def refuse(status: str, reason: str) -> dict[str, Any]:
        validity["status"] = status
        reasons.append(reason)
        result["warnings"].append(f"atom cache accounting: {reason}")
        return result

    series, duplicates = {}, {}
    for stem in _COUNTERS:
        series[stem], duplicates[stem] = _series(metrics, stem)
    observed = {key[0] for stem in _CORE for key in series[stem]}
    if not series[FULL]:
        return refuse("MISSING_OBSERVATION", "no prefix_cache_full_tokens series")
    if endpoint_roles is None and disagg is False and len(observed) == 1:
        accounted = sorted(observed)
        validity["roles_source"] = "single_endpoint_non_disagg"
    elif endpoint_roles:
        accounted = sorted(
            url for url, role in endpoint_roles.items() if role in {"prefill", "unified"}
        )
        if observed - endpoint_roles.keys() or not accounted:
            return refuse("ROLE_OR_COVERAGE_UNRESOLVED", f"{framework}: endpoint roles incomplete")
        validity["roles_source"] = "launcher_env"
    else:
        return refuse("ROLE_OR_COVERAGE_UNRESOLVED", f"{framework}: endpoint roles unresolved")
    validity["accounted_endpoints"] = accounted
    series = {
        stem: {key: s for key, s in group.items() if key[0] in accounted}
        for stem, group in series.items()
    }
    duplicates = {stem: bad & {*accounted, None} for stem, bad in duplicates.items()}
    if any(duplicates[stem] for stem in (FULL, CACHED)):
        result["replace_telemetry"] = True
        return refuse("INVALID_COUNTER", "duplicate or unidentifiable cache counter series")
    keys = set(series[FULL])
    if set(accounted) - {key[0] for key in keys}:
        return refuse("ROLE_OR_COVERAGE_UNRESOLVED", "accounted endpoints missing full counters")
    if not keys or set(series[CACHED]) != keys:
        return refuse("MISSING_OBSERVATION", "full/cached series coverage differs")
    scoped = {stem: {key: _total(s) for key, s in group.items()} for stem, group in series.items()}
    raw = {
        stem: sum(values) if values and all(value is not None for value in values) else None
        for stem in _COUNTERS
        for values in [list(scoped[stem].values())]
    }
    validity["diagnostics"]["cumulative_stats_total"] = {
        f"atom:{stem}": value for stem, value in raw.items() if value is not None
    }
    if raw[FULL] is None or raw[CACHED] is None:
        result["replace_telemetry"] = True
        return refuse("INVALID_COUNTER", "full/cached stats.total missing or invalid")
    cpu_missing = not series[OFFLOAD]
    cpu_complete = (
        set(series[OFFLOAD]) == keys and raw[OFFLOAD] is not None and not duplicates[OFFLOAD]
    )
    cpu_invalid = bool(duplicates[OFFLOAD]) or (bool(series[OFFLOAD]) and raw[OFFLOAD] is None)
    if profiling_window is None:
        return refuse("WINDOW_UNRESOLVED", "no summary.phase_time_ranges.profiling window")
    totals = dict(raw)
    boundary = False
    modern = _modern_export(collector_export or {}, accounted)
    if not modern:
        for stem in _COUNTERS:
            if raw[stem] is None or (stem == OFFLOAD and not cpu_complete):
                totals[stem] = None
                continue
            warm, warm_duplicate = _series(warmup_metrics or {}, stem)
            windows = {
                key: _window_total(s, profiling_window, None if warm_duplicate else warm.get(key))
                for key, s in series[stem].items()
            }
            values = list(windows.values())
            if any(value is None for value, _ in values):
                if stem in (FULL, CACHED) or (stem == OFFLOAD and cpu_complete):
                    return refuse(
                        "WINDOW_UNRESOLVED", f"{stem}: missing, invalid or gapped timeslices"
                    )
                totals[stem] = None
            else:
                scoped[stem] = {key: value for key, (value, _) in windows.items()}
                totals[stem] = sum(value for value, _ in values)
                if stem in _CORE:
                    boundary |= any(issue for _, issue in values)
    validity["observed_scope"] = (
        "profiling_phase_counter_totals" if modern else "profiling_window_interior_timeslices"
    )
    validity["window"] = {
        "start_ns": profiling_window[0],
        "end_ns": profiling_window[1],
        "boundary_issues": ["activity crosses the profiling boundary"] if boundary else [],
    }
    result["replace_telemetry"] = True
    full, cached, offload = totals[FULL], totals[CACHED], totals[OFFLOAD]
    if (
        full is None
        or not math.isfinite(full)
        or full <= 0
        or cached is None
        or not 0 <= cached <= full
    ):
        return refuse("INVALID_COUNTER", "full/cached tokens outside their valid range")
    for key in keys:
        if scoped[CACHED][key] > scoped[FULL][key]:
            return refuse("INVALID_COUNTER", "cached tokens exceed full tokens in a paired series")
        if cpu_complete and scoped[CACHED][key] + scoped[OFFLOAD][key] > scoped[FULL][key]:
            cpu_invalid = True
    if cpu_missing and (kv_offloading or "").lower() == "none":
        offload = 0.0
        validity["cpu_component"] = "NO_CPU_TIER_CONFIRMED"
    elif cpu_complete and not cpu_invalid and offload is not None and cached + offload <= full:
        validity["cpu_component"] = "MEASURED"
    else:
        offload = None
        validity["cpu_component"] = "MISSING"
    result.update(
        full=full,
        cached=cached,
        offload=offload,
        loaded=totals[LOADED],
        requests=totals[REQUESTS],
        gpu=cached / full,
    )
    if offload is not None:
        result.update(cpu=offload / full, overall=(cached + offload) / full)
    scope = dict(request_scope or {})
    admissions = (
        totals[REQUESTS] if set(series[REQUESTS]) == keys and not duplicates[REQUESTS] else None
    )
    scope_counts = [
        scope.get(key) for key in ("profiling_requests", "profiling_prompt_tokens", "error_records")
    ]
    if "cancelled_records" in scope:
        scope_counts.append(scope["cancelled_records"])
    scope_valid = all(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
        and int(value) == value
        for value in scope_counts
    )
    conserved = (
        scope_valid
        and admissions is not None
        and scope.get("profiling_prompt_tokens") is not None
        and admissions == scope.get("profiling_requests")
        and full == scope["profiling_prompt_tokens"]
        and scope.get("error_records") == 0
        and not scope.get("cancelled_records")
        and (not modern or scope.get("cancelled_records") == 0)
    )
    validity["request_scope"] = {
        "server_window_admissions": admissions,
        "server_window_prompt_tokens": full,
        **scope,
        "conserved": conserved,
    }
    if duplicates[LOADED] or totals[LOADED] is None:
        result["loaded"] = None
        validity["diagnostics"]["loaded_tokens_unavailable"] = True
    if boundary:
        return refuse(
            "WINDOW_UNRESOLVED", "activity crosses the profiling boundary without warmup proof"
        )
    if cpu_invalid or (offload is None and cpu_complete):
        return refuse(
            "INVALID_COUNTER", "offload tokens invalid or incompatible with full/cached tokens"
        )
    if offload is None:
        return refuse(
            "PARTIAL_COMPONENT", "CPU reuse missing and no confirmed absence of a CPU tier"
        )
    if not conserved:
        return refuse(
            "REQUEST_SCOPE_UNRESOLVED",
            "server admissions/prompt tokens do not match successful profiling requests",
        )
    validity.update(
        status="VALID_ZERO" if cached + offload == 0 else "VALID_COMPLETE", pricing_eligible=True
    )
    return result
