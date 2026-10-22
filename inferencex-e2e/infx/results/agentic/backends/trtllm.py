"""TensorRT-LLM server metric adapter."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

from infx.results.agentic.cache_validity import modern_profile_error
from infx.results.agentic.common import (
    gauge_stat,
    label_value,
    metric_series,
    normalize_fraction,
    rate,
    sum_stat,
)

from .base import ServerMetricsBackend, counter_int

PROMPT_TOKENS = "trtllm_prompt_tokens"
CACHED_TOKENS = "trtllm_prompt_cached_tokens"
GENERATION_TOKENS = "trtllm_generation_tokens"
# Interval counters do not identify the billed cohort or prove expected worker coverage.
UNBOUND_REASONS = ("request_scope_not_bound", "expected_instances_not_verified")
ROLE_LABELS = {"disaggregation_mode", "dynamo_component"}


class TrtllmBackend(ServerMetricsBackend):
    name = "trtllm"
    accepts_context = True

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
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        context = context or {}
        prompt_total = sum_stat(
            metrics, "dynamo_frontend_input_sequence_tokens", preferred_keys=("total", "sum")
        )
        generation_total = _first_total(metrics, "dynamo_frontend_output_tokens", GENERATION_TOKENS)

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
        reuse, paired = _reuse_rate(metrics, context)
        nested["validity"]["cache_rate"] = reuse
        nested["sources"] = _trtllm_sources(metrics, paired, context.get("disagg"))
        if reuse["published"]:
            prompt = reuse["prompt_tokens"]
            cached = sum(item[1] for item in paired.values())
            if flat["total_prompt_tokens"] is None:
                flat["total_prompt_tokens"] = nested["tokens"]["prompt_total"] = prompt
            flat["server_overall_cache_hit_rate"] = cached / prompt
            nested["cache"].update(
                {
                    "overall_cache_hit_rate": cached / prompt,
                    "cached_tokens_by_source": {"combined": cached},
                }
            )
            nested["tokens"]["prompt_by_source"].update(
                {
                    "computed": prompt - cached
                    if nested["tokens"]["prompt_total"] == prompt
                    else None,
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


def _reuse_rate(
    metrics: dict[str, dict[str, Any]], context: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, tuple[int, int]]]:
    result = {
        "status": "UNKNOWN",
        "published": False,
        "eligible_exact_actual": False,
        "reasons": [],
        "role_scope": None,
        "prompt_tokens": None,
    }
    disaggregated = context.get("disagg")
    roles = _endpoint_role_sets(metrics, disaggregated, context.get("endpoint_roles"))
    observed = {
        _series_role(series, roles, disaggregated)
        for name in (PROMPT_TOKENS, CACHED_TOKENS)
        for series in metric_series(metrics, [name, f"{name}_total"])
    }
    scope = "prefill" if disaggregated is True else "combined" if disaggregated is False else None
    if scope is None:
        candidates = observed - {"decode"}
        scope = next(iter(candidates)) if len(candidates) == 1 else None
    if (
        scope not in ("prefill", "combined")
        or observed - {scope, "decode"}
        or (scope == "combined" and "decode" in observed)
    ):
        result["reasons"] = ["role_or_deployment_unresolved"]
        return result, {}
    if any(
        len(roles.get(str(series.get("endpoint_url", "")), set())) > 1
        for name in (PROMPT_TOKENS, CACHED_TOKENS)
        for series in metric_series(metrics, [name, f"{name}_total"])
        if _series_role(series, roles, disaggregated) == scope
    ):
        result["reasons"] = ["role_conflict"]
        return result, {}
    result["role_scope"] = scope
    try:
        prompt = _counter_domains(metrics, PROMPT_TOKENS, roles, scope, disaggregated)
        cached = _counter_domains(metrics, CACHED_TOKENS, roles, scope, disaggregated)
    except ValueError as error:
        result.update(status="REJECTED", reasons=[str(error)])
        return result, {}
    if not prompt or prompt.keys() != cached.keys():
        result.update(status="PARTIAL_COVERAGE", reasons=["missing_counter_pair"])
        return result, {}
    error = modern_profile_error(context.get("collector_export") or {}, {key[0] for key in prompt})
    if error:
        result["reasons"] = [error]
        return result, {}
    if any(cached[key] > value for key, value in prompt.items()):
        result.update(status="REJECTED", reasons=["cached_exceeds_prompt"])
        return result, {}
    total = sum(prompt.values())
    if total <= 0:
        result["reasons"] = ["no_prompt_tokens"]
        return result, {}
    paired: dict[str, tuple[int, int]] = {}
    for key, value in prompt.items():
        full, hit = paired.get(key[0], (0, 0))
        paired[key[0]] = (full + value, hit + cached[key])
    result.update(
        status="CONDITIONAL_INTERVAL",
        published=True,
        reasons=list(UNBOUND_REASONS),
        prompt_tokens=total,
    )
    return result, paired


def _counter_domains(
    metrics: dict[str, dict[str, Any]],
    name: str,
    roles: dict[str, set[str]],
    scope: str,
    disaggregated: bool | None,
) -> dict[tuple, int]:
    totals: dict[tuple, int] = {}
    for alias in (name, f"{name}_total"):
        entry = metrics.get(alias)
        if entry is None:
            continue
        if (
            not isinstance(entry, dict)
            or entry.get("type") != "counter"
            or not isinstance(entry.get("series"), list)
        ):
            raise ValueError("invalid_counter_series")
        seen = set()
        for series in entry["series"]:
            if not isinstance(series, dict):
                raise ValueError("invalid_counter_series")
            if _series_role(series, roles, disaggregated) != scope:
                continue
            endpoint = series.get("endpoint_url")
            labels = series.get("labels")
            labels = {} if labels is None else labels
            if (
                not isinstance(endpoint, str)
                or not endpoint
                or not isinstance(labels, dict)
                or any(
                    not isinstance(key, str) or not isinstance(value, str)
                    for key, value in labels.items()
                )
            ):
                raise ValueError("invalid_counter_domain")
            key = endpoint, tuple(sorted((k, v) for k, v in labels.items() if k not in ROLE_LABELS))
            stats = series.get("stats")
            value = stats.get("total") if isinstance(stats, dict) else None
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or (isinstance(value, float) and not math.isfinite(value))
                or value < 0
                or int(value) != value
            ):
                raise ValueError("invalid_counter_total")
            if key in seen or (key in totals and totals[key] != value):
                raise ValueError("counter_alias_conflict")
            seen.add(key)
            totals[key] = int(value)
    return totals


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
    # Native and exporter-stripped names represent the same counter.
    for name in (base, f"{base}_total"):
        if name in metrics:
            return sum_stat(metrics, name, series_filter=series_filter)
    return None


def _endpoint_role_sets(
    metrics: dict[str, dict[str, Any]],
    disaggregated: bool | None = None,
    declared: dict[str, str] | None = None,
) -> dict[str, set[str]]:
    explicit: dict[str, set[str]] = {}
    inferred: dict[str, set[str]] = {}
    for entry in metrics.values():
        if not isinstance(entry, dict):
            continue
        series_list = entry.get("series")
        for series in series_list if isinstance(series_list, list) else []:
            if not isinstance(series, dict):
                continue
            endpoint = str(series.get("endpoint_url", ""))
            labels = series.get("labels")
            component = label_value(series, "dynamo_component")
            if isinstance(labels, dict) and "disaggregation_mode" in labels:
                mode = labels["disaggregation_mode"]
                role = _normalize_role(mode) if isinstance(mode, str) and mode else "unknown"
                explicit.setdefault(endpoint, set()).add(role)
            elif component:
                role = (
                    "combined"
                    if component == "backend" and disaggregated is False
                    else _normalize_role(component)
                )
                if role in ("prefill", "decode", "combined"):
                    inferred.setdefault(endpoint, set()).add(role)
    roles = {**inferred, **explicit}
    for endpoint, role in (declared or {}).items():
        roles.setdefault(endpoint, set()).add(_normalize_role(role))
    return roles


def _series_role(
    series: dict[str, Any], roles: dict[str, set[str]], disaggregated: bool | None = None
) -> str:
    labels = series.get("labels")
    if isinstance(labels, dict) and "disaggregation_mode" in labels:
        mode = labels["disaggregation_mode"]
        return _normalize_role(mode) if isinstance(mode, str) and mode else "unknown"
    found = roles.get(str(series.get("endpoint_url", "")), set())
    if len(found) == 1:
        return next(iter(found))
    return "combined" if not found and disaggregated is False else "unknown"


def _trtllm_sources(
    metrics: dict[str, dict[str, Any]],
    paired: dict[str, tuple[int, int]],
    disaggregated: bool | None,
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
        for endpoint, found in _endpoint_role_sets(metrics, disaggregated).items()
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
        if hit_rate is None and endpoint in paired:
            prompt, cached = paired[endpoint]
            hit_rate = rate(cached, prompt)
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
    if lowered in ("aggregated", "prefill_and_decode", "agg", "combined", "unified"):
        return "combined"
    return lowered
