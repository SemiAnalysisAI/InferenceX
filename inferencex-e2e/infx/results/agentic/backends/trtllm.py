"""TensorRT-LLM server metric adapter."""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Iterable
from typing import Any

from infx.results.agentic.common import (
    gauge_stat,
    label_value,
    metric_series,
    normalize_fraction,
    rate,
    series_stat,
    sum_stat,
)
from infx.results.agentic.phase_validity import (
    BOUNDARY_UNCERTAIN,
    CONDITIONAL_INTERVAL,
    PARTIAL_COVERAGE,
    REJECTED,
    UNKNOWN,
    VALID,
    PhaseContext,
    classify_series,
    resolve_aliases,
    series_key,
    worst,
)

from .base import ServerMetricsBackend, counter_int

PROMPT_TOKENS = "trtllm_prompt_tokens"
CACHED_TOKENS = "trtllm_prompt_cached_tokens"
GENERATION_TOKENS = "trtllm_generation_tokens"
# Interval counters do not identify the billed cohort or prove expected worker coverage.
UNBOUND_REASONS = ("request_scope_not_bound", "expected_instances_not_verified")
KNOWN_ROLES = ("prefill", "decode", "combined")


class TrtllmBackend(ServerMetricsBackend):
    name = "trtllm"

    def matches(self, metrics: dict[str, dict[str, Any]], framework: str) -> bool:
        metric_names = set(metrics)
        return any(name.startswith("trtllm_") for name in metric_names) or (
            not metrics and framework.lower() in ("trtllm", "dynamo-trt", "dynamo-trtllm")
        )

    def populate(
        self,
        metrics: dict[str, dict[str, Any]],
        flat: dict[str, Any],
        nested: dict[str, Any],
    ) -> None:
        prompt_total = sum_stat(
            metrics, "dynamo_frontend_input_sequence_tokens", preferred_keys=("total", "sum")
        )
        generation_total = _first_total(metrics, "dynamo_frontend_output_tokens", GENERATION_TOKENS)
        diagnostic = _hit_rate_gauge_diagnostic(metrics)
        if diagnostic is not None:
            nested["validity"]["diagnostics"]["trtllm_kv_cache_hit_rate"] = diagnostic

        flat["total_prompt_tokens"] = counter_int(prompt_total)
        flat["total_generation_tokens"] = counter_int(generation_total)
        flat["server_gpu_cache_hit_rate"] = None
        flat["server_overall_cache_hit_rate"] = None

        flat["gpu_kv_cache_usage_pct"] = normalize_fraction(
            gauge_stat(
                metrics,
                "trtllm_kv_cache_utilization",
                preferred_keys=("max", "avg", "total"),
                combine="max",
            )
        )
        flat["cpu_kv_cache_usage_pct"] = normalize_fraction(
            gauge_stat(
                metrics,
                "trtllm_kv_cache_host_utilization",
                preferred_keys=("max", "avg", "total"),
                combine="max",
            )
        )
        flat["kv_offload_bytes_gpu_to_cpu"] = _alias_total(metrics, "trtllm_kv_cache_offload_bytes")
        flat["kv_offload_bytes_cpu_to_gpu"] = _alias_total(metrics, "trtllm_kv_cache_onboard_bytes")

        nested["cache"].update(
            {
                "gpu_cache_hit_rate": None,
                "overall_cache_hit_rate": None,
                "cached_tokens_by_source": {},
            }
        )
        nested["kv_cache"].update(
            {
                "gpu_usage_pct": flat["gpu_kv_cache_usage_pct"],
                "cpu_usage_pct": flat["cpu_kv_cache_usage_pct"],
            }
        )
        nested["kv_offload"].update(
            {
                "bytes_gpu_to_cpu": flat["kv_offload_bytes_gpu_to_cpu"],
                "bytes_cpu_to_gpu": flat["kv_offload_bytes_cpu_to_gpu"],
            }
        )
        nested["tokens"].update(
            {
                "prompt_total": flat["total_prompt_tokens"],
                "generation_total": flat["total_generation_tokens"],
                "prompt_by_source": {
                    "gpu_cache_hit": None,
                    "cpu_or_external_cache_hit": None,
                    "computed": None,
                    "raw": {},
                },
            }
        )
        nested["sources"] = _trtllm_sources(metrics)

    def apply_validity(
        self,
        metrics: dict[str, dict[str, Any]],
        flat: dict[str, Any],
        nested: dict[str, Any],
        context: PhaseContext,
    ) -> None:
        reuse = _reuse_rate(metrics, context)
        nested["validity"]["cache_rate"] = reuse
        nested["sources"] = _trtllm_sources(metrics, reuse=reuse)
        if not reuse["published"]:
            return
        cached, prompt = reuse["cached_tokens"], reuse["prompt_tokens"]
        if flat["total_prompt_tokens"] is None:
            flat["total_prompt_tokens"] = nested["tokens"]["prompt_total"] = counter_int(prompt)
        flat["server_overall_cache_hit_rate"] = reuse["raw_rate"]
        nested["cache"].update(
            {
                "overall_cache_hit_rate": reuse["raw_rate"],
                "cached_tokens_by_source": {"combined": cached},
            }
        )
        nested["tokens"]["prompt_by_source"].update(
            {
                "computed": prompt - cached if nested["tokens"]["prompt_total"] == prompt else None,
                "raw": {"combined": cached},
            }
        )

    def gpu_kv_capacity_tokens(
        self,
        metrics: dict[str, dict[str, Any]],
        server_logs: Iterable[str | None],
    ) -> int | None:
        del server_logs
        max_blocks = sum_stat(
            metrics,
            "trtllm_kv_cache_max_blocks",
            preferred_keys=("max", "avg", "total", "sum"),
        )
        tokens_per_block = gauge_stat(
            metrics,
            "trtllm_kv_cache_tokens_per_block",
            preferred_keys=("max", "avg", "total"),
            combine="max",
        )
        if max_blocks is None or tokens_per_block is None:
            return None
        return counter_int(max_blocks * tokens_per_block)


def _reuse_rate(metrics: dict[str, dict[str, Any]], context: PhaseContext) -> dict[str, Any]:
    # Decode transfers are not reuse; pair prefill counters within each endpoint and label domain.
    findings: list[tuple[str, str]] = []
    if context.windows is None:
        findings.append((UNKNOWN, "phase_window_missing"))
    resolved: dict[str, list[dict[str, Any]]] = {}
    conflicts: list[dict[str, Any]] = []
    for metric in (PROMPT_TOKENS, CACHED_TOKENS):
        resolved[metric], metric_conflicts = resolve_aliases(metrics, metric)
        conflicts.extend(metric_conflicts)
    endpoint_roles = _endpoint_role_sets(metrics, context.disaggregated)
    labelled = []
    for metric, series_list in resolved.items():
        for series in series_list:
            role = _series_role(series, endpoint_roles, context.disaggregated)
            labelled.append((metric, series, role if role in KNOWN_ROLES else "unknown"))
    present = {role for _, _, role in labelled}
    scope, scope_findings = _deployment_scope(present, context.disaggregated)
    findings.extend(scope_findings)
    scoped_endpoints = {
        str(series.get("endpoint_url", ""))
        for _, series, role in labelled
        if role == scope or (role == "unknown" and context.disaggregated is False)
    }
    if any(len(endpoint_roles.get(url, set())) > 1 for url in scoped_endpoints):
        findings.append((UNKNOWN, "role_conflict"))
    findings.extend(
        (REJECTED, conflict["kind"])
        for conflict in conflicts
        if conflict["endpoint_url"] in scoped_endpoints
        or _series_role(conflict, endpoint_roles, context.disaggregated) != "decode"
    )

    in_scope: dict[str, list[dict[str, Any]]] = {PROMPT_TOKENS: [], CACHED_TOKENS: []}
    excluded: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for metric, series, role in labelled:
        entry = {
            "metric": metric,
            "endpoint_url": str(series.get("endpoint_url", "")),
            "role": role,
        }
        if (
            role == "unknown"
            and context.disaggregated is False
            and not label_value(series, "disaggregation_mode")
        ):
            role = "combined"
        if scope is not None and role == scope:
            record = classify_series(
                series,
                "counter",
                context.windows,
                has_reference=context.has_reference(metric, series),
                warmup_total=context.warmup_total(metric, series),
            )
            in_scope[metric].append(
                {**entry, "role": role, "key": series_key(metric, series), **record}
            )
            findings.extend((record["status"], code) for code in record["reasons"])
        elif scope == "prefill" and role == "decode":
            excluded.append({**entry, "reason": "decode_role_excluded"})
        elif role == "unknown":
            unresolved.append(entry)
    if unresolved and scope is not None:
        findings.append((UNKNOWN, "role_unknown"))

    domains = _pair_domains(in_scope, findings) if scope is not None else []
    if context.unscraped is None:
        findings.append((UNKNOWN, "scrape_coverage_unknown"))
    elif context.unscraped:
        findings.append((PARTIAL_COVERAGE, "endpoint_not_scraped"))

    prompt = _finite_sum(r["total"] for r in in_scope[PROMPT_TOKENS])
    cached = _finite_sum(r["total"] for r in in_scope[CACHED_TOKENS])
    raw_rate = rate(cached, prompt)
    status = worst(status for status, _ in findings)
    if status == VALID and raw_rate is None:
        status = UNKNOWN
        findings.append((UNKNOWN, "zero_denominator"))
    published = status == VALID
    if published:
        status = CONDITIONAL_INTERVAL
    for records in in_scope.values():
        for record in records:
            record.pop("key")
    return {
        "status": status,
        "published": published,
        "eligible_exact_actual": False,
        "role_scope": scope or "unresolved",
        "deployment_disaggregated": context.disaggregated,
        "tier": "combined",
        "numerator": CACHED_TOKENS,
        "denominator": PROMPT_TOKENS,
        "cached_tokens": cached,
        "prompt_tokens": prompt,
        "raw_rate": raw_rate,
        "reasons": list(dict.fromkeys([code for _, code in findings] + list(UNBOUND_REASONS))),
        "series": in_scope[PROMPT_TOKENS] + in_scope[CACHED_TOKENS],
        "domains": domains,
        "excluded": excluded,
        "unresolved": unresolved,
        "alias_conflicts": conflicts,
    }


def _deployment_scope(
    present: set[str], disaggregated: bool | None
) -> tuple[str | None, list[tuple[str, str]]]:
    if disaggregated is True:
        findings = [(UNKNOWN, "role_conflicts_with_deployment")] if "combined" in present else []
        if "prefill" in present:
            return "prefill", findings
        return None, [*findings, (UNKNOWN, "no_prefill_scope")]
    if disaggregated is False:
        if present & {"prefill", "decode"}:
            return None, [(UNKNOWN, "role_conflicts_with_deployment")]
        return "combined", []
    if "prefill" in present:
        if "combined" in present:
            return None, [(UNKNOWN, "role_conflict")]
        return "prefill", []
    if "combined" in present and "decode" not in present:
        return "combined", []
    if "decode" in present:
        return None, [(UNKNOWN, "decode_only_no_prefill_scope")]
    return None, [(UNKNOWN, "deployment_mode_unknown")]


def _pair_domains(
    in_scope: dict[str, list[dict[str, Any]]], findings: list[tuple[str, str]]
) -> list[dict[str, Any]]:
    prompt = {r["key"][1:]: r for r in in_scope[PROMPT_TOKENS]}
    cached = {r["key"][1:]: r for r in in_scope[CACHED_TOKENS]}
    if not prompt and not cached:
        findings.append((UNKNOWN, "counter_series_missing"))
        return []
    if set(prompt) != set(cached):
        findings.append((PARTIAL_COVERAGE, "missing_pair"))
    domains = []
    for key in sorted(set(prompt) & set(cached)):
        f, k = prompt[key], cached[key]
        supports = [(r["first_slice_offset_s"], r["last_slice_offset_s"]) for r in (f, k)]
        windows_match = None
        if None not in supports[0] + supports[1]:
            windows_match = supports[0] == supports[1]
            if not windows_match:
                findings.append((BOUNDARY_UNCERTAIN, "window_support_mismatch"))
        within = None
        if _finite(f["total"]) and _finite(k["total"]):
            within = 0 <= k["total"] <= f["total"]
            if not within:
                findings.append((REJECTED, "cached_exceeds_prompt"))
        domains.append(
            {
                "endpoint_url": key[0],
                "labels": dict(key[1]),
                "prompt": f["total"],
                "cached": k["total"],
                "cached_within_prompt": within,
                "windows_match": windows_match,
            }
        )
    return domains


def _hit_rate_gauge_diagnostic(metrics: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    roles = _endpoint_role_sets(metrics)
    values: list[float] = []
    by_role: dict[str, list[float]] = {}
    for series in metric_series(metrics, "trtllm_kv_cache_hit_rate"):
        value = normalize_fraction(series_stat(series, ("avg", "max", "total")))
        if value is None:
            continue
        values.append(value)
        by_role.setdefault(_series_role(series, roles), []).append(value)
    if not values:
        return None
    return {
        "status": "GAUGE_DERIVED",
        "pricing_eligible": False,
        "all_series_avg": statistics.mean(values),
        "by_role": {role: statistics.mean(v) for role, v in sorted(by_role.items())},
    }


def _first_total(
    metrics: dict[str, dict[str, Any]], frontend_metric: str, trtllm_metric: str
) -> float | None:
    frontend = sum_stat(metrics, frontend_metric, preferred_keys=("total", "sum", "max", "avg"))
    if frontend is not None:
        return frontend
    return _alias_total(metrics, trtllm_metric)


def _alias_total(
    metrics: dict[str, dict[str, Any]],
    base: str,
    series_filter: Callable[[dict[str, Any]], bool] | None = None,
) -> float | None:
    series_list, conflicts = resolve_aliases(metrics, base)
    if any(
        series_filter is None or series_filter({"endpoint_url": c["endpoint_url"]})
        for c in conflicts
    ):
        return None
    values = [
        series_stat(series, ("total", "sum", "max", "avg"))
        for series in series_list
        if series_filter is None or series_filter(series)
    ]
    values = [value for value in values if value is not None]
    return sum(values) if values else None


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def _finite_sum(values: Iterable[float | None]) -> float | None:
    total = 0.0
    found = False
    for value in values:
        if not _finite(value):
            return None
        total += value
        found = True
    return total if found else None


def _label_role(series: dict[str, Any], disaggregated: bool | None = None) -> str | None:
    mode = label_value(series, "disaggregation_mode")
    if mode:
        role = _normalize_role(mode)
        return role if role in KNOWN_ROLES else "unknown"
    component = label_value(series, "dynamo_component")
    if component == "backend" and disaggregated is False:
        return "combined"
    role = _normalize_role(component) if component else None
    return role if role in KNOWN_ROLES else None


def _endpoint_role_sets(
    metrics: dict[str, dict[str, Any]], disaggregated: bool | None = None
) -> dict[str, set[str]]:
    explicit: dict[str, set[str]] = {}
    inferred: dict[str, set[str]] = {}
    for entry in metrics.values():
        if not isinstance(entry, dict):
            continue
        series_list = entry.get("series")
        for series in series_list if isinstance(series_list, list) else []:
            if not isinstance(series, dict) or not series.get("endpoint_url"):
                continue
            role = _label_role(series, disaggregated)
            if role in KNOWN_ROLES:
                target = explicit if label_value(series, "disaggregation_mode") else inferred
                target.setdefault(str(series["endpoint_url"]), set()).add(role)
    return {**inferred, **explicit}


def _series_role(
    series: dict[str, Any], roles: dict[str, set[str]], disaggregated: bool | None = None
) -> str:
    endpoint_roles = roles.get(str(series.get("endpoint_url", "")), set())
    if label_value(series, "disaggregation_mode"):
        return _label_role(series, disaggregated) or "unknown"
    if len(endpoint_roles) == 1:
        return next(iter(endpoint_roles))
    return _label_role(series, disaggregated) or "unknown"


def _trtllm_sources(
    metrics: dict[str, dict[str, Any]], *, reuse: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    endpoints: set[str] = set()
    for entry in metrics.values():
        if not isinstance(entry, dict):
            continue
        series_list = entry.get("series")
        if not isinstance(series_list, list):
            continue
        for series in series_list:
            if isinstance(series, dict) and series.get("endpoint_url"):
                endpoints.add(str(series["endpoint_url"]))
    roles = {
        endpoint: next(iter(found)) if len(found) == 1 else "unknown"
        for endpoint, found in _endpoint_role_sets(
            metrics, reuse.get("deployment_disaggregated") if reuse else None
        ).items()
    }

    sources: list[dict[str, Any]] = []
    for endpoint in sorted(endpoints):

        def series_filter(series: dict, endpoint: str = endpoint) -> bool:
            return str(series.get("endpoint_url", "")) == endpoint

        prompt_tokens = _alias_total(metrics, PROMPT_TOKENS, series_filter)
        generation_tokens = _alias_total(metrics, GENERATION_TOKENS, series_filter)
        hit_rate = normalize_fraction(
            gauge_stat(
                metrics,
                "trtllm_kv_cache_hit_rate",
                combine="avg",
                series_filter=series_filter,
            )
        )
        if hit_rate is None and reuse is not None and reuse["published"]:
            domains = [d for d in reuse["domains"] if d["endpoint_url"] == endpoint]
            if domains:
                hit_rate = rate(
                    sum(d["cached"] for d in domains), sum(d["prompt"] for d in domains)
                )
        sources.append(
            {
                "id": f"{roles.get(endpoint, 'worker')}|{endpoint}",
                "role": roles.get(endpoint, "worker"),
                "prompt_tokens": prompt_tokens,
                "generation_tokens": generation_tokens,
                "prefix_cache_hit_rate": hit_rate,
                "gpu_kv_cache_usage_pct": normalize_fraction(
                    gauge_stat(
                        metrics,
                        "trtllm_kv_cache_utilization",
                        series_filter=series_filter,
                    )
                ),
            }
        )
    return sources


def _normalize_role(mode: str) -> str:
    lowered = mode.lower()
    if lowered in ("prefill", "context", "ctx"):
        return "prefill"
    if lowered in ("decode", "generation", "gen", "backend"):
        return "decode"
    if lowered in ("aggregated", "prefill_and_decode", "agg"):
        return "combined"
    return lowered
