"""vLLM server metric adapter."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from infx.results.agentic.common import (
    gauge_stat,
    label_value,
    metric_series,
    normalize_fraction,
    rate,
    sum_by_label,
    sum_server_log_capacities,
    sum_stat,
)

from .base import ServerMetricsBackend, counter_int


class VllmBackend(ServerMetricsBackend):
    name = "vllm"
    accepts_context = True
    _ENGINE_TAG_RE = re.compile(r"\((?P<tag>EngineCore(?:_DP\d+)?)\s+pid=\d+\)")
    _GPU_KV_SIZE_RE = re.compile(r"GPU KV cache size:\s*(?P<tokens>[\d,]+)\s*tokens")

    def matches(self, metrics: dict[str, dict[str, Any]], framework: str) -> bool:
        """Return True for vllm: metric names, or for an empty scrape when ``framework`` is vllm."""
        metric_names = set(metrics)
        return any(name.startswith("vllm:") for name in metric_names) or (
            not metrics and framework.lower() == "vllm"
        )

    def populate(
        self,
        metrics: dict[str, dict[str, Any]],
        flat: dict[str, Any],
        nested: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Fill token totals, cache rates, KV usage, offload statistics and sources."""
        endpoint_roles = (context or {}).get("endpoint_roles")
        prompt_total, generation_total = self.prompt_generation_totals(metrics)
        flat["total_prompt_tokens"] = counter_int(prompt_total)
        flat["total_generation_tokens"] = counter_int(generation_total)

        disaggregated = nested["validity"]["deployment"]["disaggregated"]
        role_scope, role_filter = _cache_role_filter(metrics, disaggregated, endpoint_roles)
        prefix_hits = sum_stat(
            metrics,
            "vllm:prefix_cache_hits",
            preferred_keys=("total", "sum", "max", "avg"),
            series_filter=role_filter,
        )
        prefix_queries = sum_stat(
            metrics,
            "vllm:prefix_cache_queries",
            preferred_keys=("total", "sum", "max", "avg"),
            series_filter=role_filter,
        )
        gpu_rate = rate(prefix_hits, prefix_queries)
        flat["server_gpu_cache_hit_rate"] = gpu_rate

        external_hits = sum_stat(
            metrics,
            "vllm:external_prefix_cache_hits",
            preferred_keys=("total", "sum", "max", "avg"),
            series_filter=role_filter,
        )
        external_queries = sum_stat(
            metrics,
            "vllm:external_prefix_cache_queries",
            preferred_keys=("total", "sum", "max", "avg"),
            series_filter=role_filter,
        )
        external_rate = rate(external_hits, external_queries)
        flat["server_external_cache_hit_rate"] = external_rate
        flat["server_cpu_cache_hit_rate"] = external_rate

        prompt_by_source = sum_by_label(
            metrics,
            "vllm:prompt_tokens_by_source",
            "source",
            preferred_keys=("total", "sum", "max", "avg"),
            series_filter=role_filter,
        )
        if prompt_by_source:
            local_cache_hit = prompt_by_source.get("local_cache_hit")
            external_transfer = prompt_by_source.get("external_kv_transfer")
            local_compute = prompt_by_source.get("local_compute")
            source_total = sum(prompt_by_source.values())
            if source_total > 0:
                if local_cache_hit is not None:
                    flat["server_gpu_cache_hit_rate"] = local_cache_hit / source_total
                if external_transfer is not None:
                    flat["server_cpu_cache_hit_rate"] = external_transfer / source_total
                    flat["server_external_cache_hit_rate"] = external_transfer / source_total
                cached_total = (local_cache_hit or 0.0) + (external_transfer or 0.0)
                flat["server_overall_cache_hit_rate"] = cached_total / source_total
            nested["tokens"]["prompt_by_source"] = {
                "gpu_cache_hit": local_cache_hit,
                "cpu_or_external_cache_hit": external_transfer,
                "computed": local_compute,
                "raw": prompt_by_source,
            }
        elif gpu_rate is not None:
            flat["server_overall_cache_hit_rate"] = gpu_rate

        gpu_usage = gauge_stat(
            metrics,
            ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"],
            preferred_keys=("max", "avg", "total"),
            combine="max",
        )
        flat["gpu_kv_cache_usage_pct"] = normalize_fraction(gpu_usage)

        cpu_usage = gauge_stat(
            metrics,
            "vllm:cpu_kv_cache_usage_perc",
            preferred_keys=("max", "avg", "total"),
            combine="max",
        )
        flat["cpu_kv_cache_usage_pct"] = normalize_fraction(cpu_usage)

        for metric_name, field_name in (
            ("vllm:kv_offload_bytes_gpu_to_cpu", "kv_offload_bytes_gpu_to_cpu"),
            ("vllm:kv_offload_bytes_cpu_to_gpu", "kv_offload_bytes_cpu_to_gpu"),
            ("vllm:kv_offload_time_gpu_to_cpu", "kv_offload_time_gpu_to_cpu"),
            ("vllm:kv_offload_time_cpu_to_gpu", "kv_offload_time_cpu_to_gpu"),
        ):
            flat[field_name] = sum_stat(
                metrics,
                metric_name,
                preferred_keys=("total", "sum", "max", "avg"),
            )

        flat["kv_offload_bandwidth_gpu_to_cpu_bytes_per_second"] = rate(
            flat["kv_offload_bytes_gpu_to_cpu"],
            flat["kv_offload_time_gpu_to_cpu"],
        )
        flat["kv_offload_bandwidth_cpu_to_gpu_bytes_per_second"] = rate(
            flat["kv_offload_bytes_cpu_to_gpu"],
            flat["kv_offload_time_cpu_to_gpu"],
        )

        nested["cache"].update(
            {
                "gpu_cache_hit_rate": flat["server_gpu_cache_hit_rate"],
                "cpu_cache_hit_rate": flat["server_cpu_cache_hit_rate"],
                "external_cache_hit_rate": flat["server_external_cache_hit_rate"],
                "overall_cache_hit_rate": flat["server_overall_cache_hit_rate"],
                "prefix_cache_hits": prefix_hits,
                "prefix_cache_queries": prefix_queries,
                "external_prefix_cache_hits": external_hits,
                "external_prefix_cache_queries": external_queries,
                "overall_source": "worker_counters"
                if flat["server_overall_cache_hit_rate"] is not None
                else None,
                "role_scope": role_scope,
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
                "time_gpu_to_cpu": flat["kv_offload_time_gpu_to_cpu"],
                "time_cpu_to_gpu": flat["kv_offload_time_cpu_to_gpu"],
                "bandwidth_gpu_to_cpu_bytes_per_second": flat[
                    "kv_offload_bandwidth_gpu_to_cpu_bytes_per_second"
                ],
                "bandwidth_cpu_to_gpu_bytes_per_second": flat[
                    "kv_offload_bandwidth_cpu_to_gpu_bytes_per_second"
                ],
            }
        )
        nested["tokens"].update(
            {
                "prompt_total": flat["total_prompt_tokens"],
                "generation_total": flat["total_generation_tokens"],
            }
        )
        nested["sources"] = _vllm_sources(metrics, disaggregated, endpoint_roles)

    def prompt_generation_totals(
        self,
        metrics: dict[str, dict[str, Any]],
    ) -> tuple[float | None, float | None]:
        """Return the vLLM prompt and generation token totals from ``metrics``."""
        prompt_total = sum_stat(
            metrics,
            "vllm:prompt_tokens",
            preferred_keys=("total", "sum", "max", "avg"),
        )
        generation_total = sum_stat(
            metrics,
            "vllm:generation_tokens",
            preferred_keys=("total", "sum", "max", "avg"),
        )
        return prompt_total, generation_total

    def gpu_kv_capacity_tokens(
        self,
        metrics: dict[str, dict[str, Any]],  # noqa: ARG002
        server_logs: Iterable[str | None],
    ) -> int | None:
        """Return the summed vLLM GPU KV cache size parsed from ``server_logs``, ignoring ``metrics``."""
        return sum_server_log_capacities(
            server_logs,
            self.kv_cache_pool_tokens_from_server_log,
        )

    @classmethod
    def kv_cache_pool_tokens_from_server_log(cls, server_log: str | None) -> int | None:
        """Return the GPU KV cache size in tokens from ``server_log``, summed per EngineCore, or None."""
        if not server_log:
            return None

        per_engine: dict[str, int] = {}
        bare_total = 0
        bare_found = False

        for line in server_log.splitlines():
            if "GPU KV cache size" not in line:
                continue

            size_match = cls._GPU_KV_SIZE_RE.search(line)
            if not size_match:
                continue

            try:
                tokens = int(size_match.group("tokens").replace(",", ""))
            except ValueError:
                continue
            if tokens <= 0:
                continue

            tag_match = cls._ENGINE_TAG_RE.search(line)
            if tag_match:
                per_engine[tag_match.group("tag")] = tokens
            else:
                bare_total += tokens
                bare_found = True

        if per_engine:
            return sum(per_engine.values())
        return bare_total if bare_found else None


def first_counter_total(
    metrics: dict[str, dict[str, Any]],
    metric_names: list[str],
) -> float | None:
    """Return the first non-None counter total among ``metric_names`` in ``metrics``."""
    for metric_name in metric_names:
        value = sum_stat(
            metrics,
            metric_name,
            preferred_keys=("total", "sum", "max", "avg"),
        )
        if value is not None:
            return value
    return None


def _vllm_sources(
    metrics: dict[str, dict[str, Any]],
    disaggregated: bool | None,
    endpoint_roles: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Return per-source token, hit rate and KV usage rows, sorted by source id."""
    source_ids = set()
    for metric_name in (
        "vllm:prompt_tokens",
        "vllm:generation_tokens",
        "vllm:prefix_cache_hits",
        "vllm:prefix_cache_queries",
        "vllm:kv_cache_usage_perc",
        "vllm:prompt_tokens_by_source",
    ):
        for series in metric_series(metrics, metric_name):
            source_ids.add(_source_id(series, disaggregated, endpoint_roles))

    sources: list[dict[str, Any]] = []
    for source_id in sorted(source_ids):
        if not source_id:
            continue

        def series_filter(series: dict, source_id: str = source_id) -> bool:
            """Return whether the series belongs to the selected source."""
            return _source_id(series, disaggregated, endpoint_roles) == source_id

        prompt_tokens = sum_stat(metrics, "vllm:prompt_tokens", series_filter=series_filter)
        generation_tokens = sum_stat(metrics, "vllm:generation_tokens", series_filter=series_filter)
        hits = sum_stat(metrics, "vllm:prefix_cache_hits", series_filter=series_filter)
        queries = sum_stat(metrics, "vllm:prefix_cache_queries", series_filter=series_filter)
        kv_usage = normalize_fraction(
            gauge_stat(
                metrics,
                ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"],
                series_filter=series_filter,
            )
        )
        role = source_id.split("|", 1)[0]
        sources.append(
            {
                "id": source_id,
                "role": role,
                "prompt_tokens": prompt_tokens,
                "generation_tokens": generation_tokens,
                "prefix_cache_hit_rate": rate(hits, queries),
                "gpu_kv_cache_usage_pct": kv_usage,
            }
        )
    return sources


_CACHE_METRICS = (
    "vllm:prompt_tokens_by_source",
    "vllm:prefix_cache_hits",
    "vllm:prefix_cache_queries",
    "vllm:external_prefix_cache_hits",
    "vllm:external_prefix_cache_queries",
)


def _series_role(
    series: dict[str, Any],
    disaggregated: bool | None = None,
    endpoint_roles: Mapping[str, str] | None = None,
) -> str:
    role = _label_role(series, disaggregated)
    declared = (endpoint_roles or {}).get(str(series.get("endpoint_url", "")))
    if declared is None:
        return role
    declared = {
        "prefill": "prefill",
        "decode": "decode",
        "unified": "combined",
        "combined": "combined",
    }.get(declared, "unknown")
    labels = series.get("labels") if isinstance(series.get("labels"), dict) else {}
    if any(key in labels for key in ("disaggregation_mode", "dynamo_component", "component")):
        return role if role == declared else "unknown"
    return declared


def _label_role(series: dict[str, Any], disaggregated: bool | None = None) -> str:
    labels = series.get("labels") if isinstance(series.get("labels"), dict) else {}
    mode = str(labels.get("disaggregation_mode") or "").lower()
    if mode in ("prefill", "context"):
        return "prefill"
    if mode == "decode":
        return "decode"
    if mode in ("combined", "unified", "prefill_and_decode"):
        return "combined"
    if "disaggregation_mode" in labels:
        return "unknown"
    component = str(labels.get("dynamo_component") or labels.get("component") or "")
    if component == "prefill":
        return "prefill"
    if component == "backend" and disaggregated is False:
        return "combined"
    if component in ("backend", "decode"):
        return "decode"
    if component in ("frontend", "router"):
        return "router"
    return "combined"


def _cache_role_filter(
    metrics: dict[str, dict[str, Any]],
    disaggregated: bool | None,
    endpoint_roles: Mapping[str, str] | None = None,
) -> tuple[str, Any]:
    roles = {
        _series_role(s, disaggregated, endpoint_roles)
        for name in _CACHE_METRICS
        for s in metric_series(metrics, name)
    }
    if "prefill" in roles:
        scope = "mixed" if "combined" in roles else "prefill"
    else:
        scope = "unresolved" if "decode" in roles or disaggregated is True else "combined"
    if "unknown" in roles or (disaggregated is True and "combined" in roles):
        scope = "unresolved"
    # Known decode counters measure KV transfer, including alongside unlabelled workers.
    return (
        scope,
        lambda series: (
            _series_role(series, disaggregated, endpoint_roles)
            in (
                "prefill",
                "combined",
            )
        ),
    )


def _source_id(
    series: dict[str, Any],
    disaggregated: bool | None,
    endpoint_roles: Mapping[str, str] | None = None,
) -> str:
    """Build a role, endpoint and rank identity for a worker series."""
    role = _series_role(series, disaggregated, endpoint_roles)

    parts = [role]
    endpoint = series.get("endpoint_url")
    if endpoint:
        parts.append(str(endpoint))
    for key in ("worker_id", "dp_rank", "engine"):
        value = label_value(series, key)
        if value is not None:
            parts.append(f"{key}={value}")
    return "|".join(parts)
