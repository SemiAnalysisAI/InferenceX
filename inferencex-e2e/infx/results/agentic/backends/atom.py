"""Native ATOM server metric adapter."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from infx.results.agentic.common import gauge_stat, normalize_fraction, series_stat, sum_stat

from .base import ServerMetricsBackend, counter_int

# P/D decode workers re-count prefilled prompts in prefix_cache_full_tokens without reuse.
ACCOUNTED_ROLES = frozenset({"prefill", "unified"})
FULL = "prefix_cache_full_tokens"
CACHED = "prefix_cache_cached_tokens"
OFFLOAD = "prefix_cache_offload_tokens"
LOADED = "lmcache_loaded_tokens"
REQUESTS = "prefix_cache_requests"
_COUNTERS = (FULL, CACHED, OFFLOAD, LOADED, REQUESTS)
VALIDITY_SCHEMA = "atom-cache-validity/1"
PRICING_STATUSES = frozenset({"VALID_COMPLETE", "VALID_ZERO"})


def _atom_names(suffix: str) -> list[str]:
    """Accept direct worker metrics and Atomesh's colon-normalized form."""
    return [f"atom:{suffix}", f"atom_{suffix}"]


def _atom_counter_names(stem: str) -> list[str]:
    """Accept Prometheus raw names and AIPerf's counter-family names."""
    return [*_atom_names(stem), *_atom_names(f"{stem}_total")]


def _series_by_endpoint(
    section: Mapping[str, Any], stem: str
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """One series per endpoint across all name aliases; a second one is a duplicate."""
    by_endpoint: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for name in _atom_counter_names(stem):
        entry = section.get(name)
        if not isinstance(entry, dict):
            continue
        for series in entry.get("series") or []:
            url = series.get("endpoint_url") if isinstance(series, dict) else None
            if not isinstance(url, str):
                continue
            if url in by_endpoint:
                duplicates.append(url)
            by_endpoint[url] = series
    return by_endpoint, duplicates


def _slices(series: dict[str, Any]) -> list[tuple[int, int, float]] | None:
    """Ordered, non-overlapping, finite, non-negative AIPerf timeslices, or None."""
    raw = series.get("timeslices")
    if not isinstance(raw, list) or not raw:
        return None
    out: list[tuple[int, int, float]] = []
    for item in raw:
        if not isinstance(item, dict):
            return None
        lo, hi, value = item.get("start_ns"), item.get("end_ns"), item.get("total")
        if not isinstance(lo, int) or not isinstance(hi, int) or hi <= lo:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if not math.isfinite(value) or value < 0:
            return None
        out.append((lo, hi, float(value)))
    out.sort()
    if any(out[i][0] < out[i - 1][1] for i in range(1, len(out))):
        return None
    return out


def _window_parts(slices: list[tuple[int, int, float]], start: int, end: int) -> dict[str, Any]:
    """Split slice totals around [start, end]; flag sampling gaps inside the window."""
    parts = {"before": 0.0, "start_straddle": 0.0, "inside": 0.0, "end_straddle": 0.0}
    covering = [s for s in slices if s[1] > start and s[0] < end]
    gap = (
        not covering
        or covering[0][0] > start
        or covering[-1][1] < end
        or any(covering[i][0] != covering[i - 1][1] for i in range(1, len(covering)))
    )
    for lo, hi, value in slices:
        if hi <= start:
            parts["before"] += value
        elif lo >= start and hi <= end:
            parts["inside"] += value
        else:
            if lo < start < hi:
                parts["start_straddle"] += value
            if lo < end < hi and not lo < start < hi:
                parts["end_straddle"] += value
    parts["gap"] = gap
    return parts


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
) -> dict[str, Any]:
    """Prefix-cache reuse on prefill (or unified) workers inside the profiling window.

    GPU = cached/full and CPU = offload/full on the same endpoints and timeslices. ATOM records
    LMCache reuse in prefix_cache_offload_tokens, disjoint from the HBM `cached` counter, so
    overall = GPU + CPU. lmcache_loaded_tokens is transfer volume and stays diagnostic only.
    `validity` tells consumers whether the rates are a complete, same-scope actual observation
    (`pricing_eligible`); warnings alone are not the guard.
    """
    result: dict[str, Any] = dict.fromkeys(
        ("gpu", "cpu", "overall", "full", "cached", "offload", "loaded", "requests")
    )
    reasons: list[str] = []
    validity: dict[str, Any] = {
        "schema": VALIDITY_SCHEMA,
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
    result.update(validity=validity, warnings=[])

    def refuse(status: str, reason: str) -> dict[str, Any]:
        validity["status"] = status
        reasons.append(reason)
        result["warnings"].append(f"atom cache accounting: {reason}")
        return result

    series: dict[str, dict[str, dict[str, Any]]] = {}
    for stem in _COUNTERS:
        series[stem], duplicates = _series_by_endpoint(metrics, stem)
        if duplicates:
            return refuse("INVALID_COUNTER", f"duplicate {stem} series for {duplicates}")
    observed = sorted(series[FULL])
    if not observed:
        return refuse("MISSING_OBSERVATION", "no prefix_cache_full_tokens series")

    if endpoint_roles:
        unmapped = [url for url in observed if url not in endpoint_roles]
        accounted = sorted(u for u, role in endpoint_roles.items() if role in ACCOUNTED_ROLES)
        missing = [url for url in accounted if url not in observed]
        if unmapped or missing or not accounted:
            return refuse(
                "ROLE_OR_COVERAGE_UNRESOLVED",
                "endpoint roles do not cover the scraped endpoints "
                f"(unmapped={unmapped}, missing={missing})",
            )
        validity["roles_source"] = "launcher_env"
    elif disagg is False and len(observed) == 1:
        accounted = observed
        validity["roles_source"] = "single_endpoint_non_disagg"
    else:
        return refuse(
            "ROLE_OR_COVERAGE_UNRESOLVED",
            f"{framework or 'atom'}: no launcher endpoint roles and not a proven single unified "
            "instance (SRT_PREFILL_ENDPOINTS/SRT_DECODE_ENDPOINTS or SRT_AGG_ENDPOINTS needed)",
        )
    validity["accounted_endpoints"] = accounted

    for stem in (FULL, CACHED):
        if any(url not in series[stem] for url in accounted):
            return refuse("MISSING_OBSERVATION", f"{stem} missing on accounted endpoints")
    cumulative = {}
    for stem in _COUNTERS:
        values = [series_stat(series[stem][u]) for u in accounted if u in series[stem]]
        if values and len(values) == len(accounted) and None not in values:
            cumulative[f"atom:{stem}"] = float(sum(values))
    validity["diagnostics"]["cumulative_stats_total"] = cumulative

    if profiling_window is None:
        return refuse("WINDOW_UNRESOLVED", "no summary.phase_time_ranges.profiling window")
    start, end = profiling_window
    totals: dict[str, float | None] = {}
    boundary: list[str] = []
    for stem in _COUNTERS:
        present = [u for u in accounted if u in series[stem]]
        if not present:
            totals[stem] = None
            continue
        if len(present) != len(accounted):
            totals[stem] = None
            continue
        inside = 0.0
        for url in present:
            slices = _slices(series[stem][url])
            if slices is None:
                return refuse("WINDOW_UNRESOLVED", f"{stem}@{url}: timeslices missing or invalid")
            parts = _window_parts(slices, start, end)
            if parts["gap"]:
                return refuse("WINDOW_UNRESOLVED", f"{stem}@{url}: sampling gap in the window")
            inside += parts["inside"]
            if stem in (LOADED, REQUESTS):
                continue
            if parts["end_straddle"] > 0:
                boundary.append(f"{stem}@{url}: activity in the slice straddling profiling end")
            if parts["start_straddle"] > 0:
                warm_series, _ = _series_by_endpoint(warmup_metrics or {}, stem)
                warm = series_stat(warm_series[url]) if url in warm_series else None
                if warm is None or parts["before"] + parts["start_straddle"] != warm:
                    boundary.append(
                        f"{stem}@{url}: activity in the slice straddling profiling start is not "
                        "proven warmup-only by warmup_metrics"
                    )
        totals[stem] = inside
    validity["observed_scope"] = "profiling_window_interior_timeslices"
    validity["window"] = {"start_ns": start, "end_ns": end, "boundary_issues": boundary}

    full, cached = totals[FULL], totals[CACHED]
    offload, loaded, requests = totals[OFFLOAD], totals[LOADED], totals[REQUESTS]
    if full is None or cached is None:
        return refuse("MISSING_OBSERVATION", "cached/full counters missing on accounted endpoints")
    if offload is not None:
        validity["cpu_component"] = "MEASURED"
    elif (kv_offloading or "").lower() == "none":
        validity["cpu_component"] = "NO_CPU_TIER_CONFIRMED"
        offload = 0.0
    else:
        validity["cpu_component"] = "MISSING"
    if not math.isfinite(full) or full <= 0:
        return refuse("INVALID_COUNTER", "no prompt tokens in the accounting window")
    if not math.isfinite(cached) or not 0 <= cached <= full:
        return refuse(
            "INVALID_COUNTER", f"cached tokens ({cached:.0f}) outside [0, full ({full:.0f})]"
        )
    if offload is not None and (
        not math.isfinite(offload) or offload < 0 or not cached + offload <= full
    ):
        return refuse(
            "INVALID_COUNTER",
            f"cached+offload ({cached + offload:.0f}) exceeds full tokens ({full:.0f})",
        )
    result.update(full=full, cached=cached, offload=offload, loaded=loaded, requests=requests)
    result["gpu"] = cached / full
    if offload is not None:
        result.update(cpu=offload / full, overall=(cached + offload) / full)

    scope = dict(request_scope or {})
    client_requests = scope.get("profiling_requests")
    client_tokens = scope.get("profiling_prompt_tokens")
    conserved = (
        requests is not None
        and client_tokens is not None
        and requests == client_requests
        and full == client_tokens
        and not scope.get("error_records")
        and not scope.get("cancelled_records")
    )
    validity["request_scope"] = {
        "server_window_admissions": requests,
        "server_window_prompt_tokens": full,
        **scope,
        "conserved": conserved,
    }

    if boundary:
        reasons.extend(boundary)
        validity["status"] = "WINDOW_UNRESOLVED"
    elif validity["cpu_component"] == "MISSING":
        reasons.append("CPU reuse counter missing and no confirmed absence of a CPU tier")
        validity["status"] = "PARTIAL_COMPONENT"
    elif not conserved:
        reasons.append(
            "server window admissions/prompt tokens do not match the successful profiling "
            f"requests ({requests} vs {client_requests}; {full} vs {client_tokens})"
        )
        validity["status"] = "REQUEST_SCOPE_UNRESOLVED"
    else:
        validity["status"] = "VALID_ZERO" if cached + offload == 0 else "VALID_COMPLETE"
    validity["pricing_eligible"] = validity["status"] in PRICING_STATUSES
    if not validity["pricing_eligible"]:
        result["warnings"].append(
            f"atom cache accounting: {validity['status']}; rates are telemetry, not pricing input"
        )
    return result


class AtomBackend(ServerMetricsBackend):
    name = "atom"
    accepts_context = True

    def matches(self, metrics: dict[str, dict[str, Any]], framework: str) -> bool:
        metric_names = set(metrics)
        return any(name.startswith(("atom:", "atom_")) for name in metric_names) or (
            not metrics and framework.lower() == "atom"
        )

    def populate(
        self,
        metrics: dict[str, dict[str, Any]],
        flat: dict[str, Any],
        nested: dict[str, Any],
        *,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        prompt_total = sum_stat(
            metrics,
            _atom_counter_names("prompt_tokens"),
            preferred_keys=("total", "sum", "max", "avg"),
        )
        generation_total = sum_stat(
            metrics,
            _atom_counter_names("generation_tokens"),
            preferred_keys=("total", "sum", "max", "avg"),
        )
        flat["total_prompt_tokens"] = counter_int(prompt_total)
        flat["total_generation_tokens"] = counter_int(generation_total)

        context = context or {}
        accounting = account_cache(
            metrics,
            framework=str(context.get("framework") or ""),
            endpoint_roles=context.get("endpoint_roles"),
            profiling_window=context.get("profiling_window"),
            warmup_metrics=context.get("warmup_metrics"),
            request_scope=context.get("request_scope"),
            disagg=context.get("disagg"),
            kv_offloading=context.get("kv_offloading"),
        )
        sink = context.get("warnings")
        if isinstance(sink, list):
            sink.extend(accounting["warnings"])

        flat["server_gpu_cache_hit_rate"] = accounting["gpu"]
        flat["server_cpu_cache_hit_rate"] = accounting["cpu"]
        # LocalCPUBackend is ATOM's only external tier, so external aliases the CPU reuse.
        flat["server_external_cache_hit_rate"] = accounting["cpu"]
        flat["server_overall_cache_hit_rate"] = accounting["overall"]
        flat["gpu_kv_cache_usage_pct"] = normalize_fraction(
            gauge_stat(
                metrics,
                _atom_names("kv_cache_usage_ratio"),
                preferred_keys=("max", "avg", "total"),
                combine="max",
            )
        )

        nested["cache"].update(
            {
                "gpu_cache_hit_rate": accounting["gpu"],
                "cpu_cache_hit_rate": accounting["cpu"],
                "external_cache_hit_rate": accounting["cpu"],
                "overall_cache_hit_rate": accounting["overall"],
                "prefix_cache_hits": accounting["cached"],
                "prefix_cache_queries": accounting["full"],
                "external_prefix_cache_hits": accounting["offload"],
                "external_prefix_cache_queries": (
                    accounting["full"] if accounting["offload"] is not None else None
                ),
            }
        )
        nested["cache_validity"] = accounting["validity"]
        by_source = nested["tokens"]["prompt_by_source"]
        if accounting["full"] is not None and accounting["offload"] is not None:
            by_source.update(
                {
                    "gpu_cache_hit": accounting["cached"],
                    "cpu_or_external_cache_hit": accounting["offload"],
                    "computed": accounting["full"] - accounting["cached"] - accounting["offload"],
                }
            )
        by_source["raw"] = {
            f"atom:{stem}": accounting[key]
            for stem, key in (
                (FULL, "full"),
                (CACHED, "cached"),
                (OFFLOAD, "offload"),
                (LOADED, "loaded"),
                (REQUESTS, "requests"),
            )
            if accounting[key] is not None
        }
        nested["kv_cache"]["gpu_usage_pct"] = flat["gpu_kv_cache_usage_pct"]
        nested["tokens"].update(
            {
                "prompt_total": flat["total_prompt_tokens"],
                "generation_total": flat["total_generation_tokens"],
            }
        )
