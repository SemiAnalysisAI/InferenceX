"""vLLM server metric adapter."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..aggregation_common import (
    gauge_stat,
    label_value,
    metric_series,
    normalize_fraction,
    rate,
    sum_by_label,
    sum_stat,
)
from ..server_log_metrics import sum_server_log_capacities
from .base import ServerMetricsBackend, counter_int


class VllmBackend(ServerMetricsBackend):
    name = "vllm"
    _ENGINE_TAG_RE = re.compile(r"\((?P<tag>EngineCore(?:_DP\d+)?)\s+pid=\d+\)")
    _GPU_KV_SIZE_RE = re.compile(r"GPU KV cache size:\s*(?P<tokens>[\d,]+)\s*tokens")

    def matches(self, metrics: dict[str, dict[str, Any]], framework: str) -> bool:
        metric_names = set(metrics)
        return any(name.startswith("vllm:") for name in metric_names) or (
            not metrics and framework.lower() == "vllm"
        )

    def populate(
        self,
        metrics: dict[str, dict[str, Any]],
        flat: dict[str, Any],
        nested: dict[str, Any],
    ) -> None:
        prompt_total, generation_total = self.prompt_generation_totals(metrics)
        flat["total_prompt_tokens"] = counter_int(prompt_total)
        flat["total_generation_tokens"] = counter_int(generation_total)

        prefix_hits = sum_stat(
            metrics,
            "vllm:prefix_cache_hits",
            preferred_keys=("total", "sum", "max", "avg"),
        )
        prefix_queries = sum_stat(
            metrics,
            "vllm:prefix_cache_queries",
            preferred_keys=("total", "sum", "max", "avg"),
        )
        gpu_rate = rate(prefix_hits, prefix_queries)
        flat["server_gpu_cache_hit_rate"] = gpu_rate

        external_hits = sum_stat(
            metrics,
            "vllm:external_prefix_cache_hits",
            preferred_keys=("total", "sum", "max", "avg"),
        )
        external_queries = sum_stat(
            metrics,
            "vllm:external_prefix_cache_queries",
            preferred_keys=("total", "sum", "max", "avg"),
        )
        external_rate = rate(external_hits, external_queries)
        flat["server_external_cache_hit_rate"] = external_rate
        flat["server_cpu_cache_hit_rate"] = external_rate

        prompt_by_source = sum_by_label(
            metrics,
            "vllm:prompt_tokens_by_source",
            "source",
            preferred_keys=("total", "sum", "max", "avg"),
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
        nested["sources"] = _vllm_sources(metrics)

    def prompt_generation_totals(
        self,
        metrics: dict[str, dict[str, Any]],
    ) -> tuple[float | None, float | None]:
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
        metrics: dict[str, dict[str, Any]],
        server_log_paths: list[Path],
    ) -> int | None:
        return sum_server_log_capacities(
            server_log_paths,
            self.kv_cache_pool_tokens_from_server_log,
        )

    @classmethod
    def kv_cache_pool_tokens_from_server_log(cls, server_log: str | None) -> int | None:
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
    for metric_name in metric_names:
        value = sum_stat(
            metrics,
            metric_name,
            preferred_keys=("total", "sum", "max", "avg"),
        )
        if value is not None:
            return value
    return None


def _vllm_sources(metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
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
            source_ids.add(_source_id(series))

    sources: list[dict[str, Any]] = []
    for source_id in sorted(source_ids):
        if not source_id:
            continue
        series_filter = lambda series, source_id=source_id: _source_id(series) == source_id
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


def _source_id(series: dict[str, Any]) -> str:
    labels = series.get("labels") if isinstance(series.get("labels"), dict) else {}
    component = str(labels.get("dynamo_component") or labels.get("component") or "")
    if component == "prefill":
        role = "prefill"
    elif component in ("backend", "decode"):
        role = "decode"
    elif component in ("frontend", "router"):
        role = "router"
    else:
        role = "combined"

    parts = [role]
    endpoint = series.get("endpoint_url")
    if endpoint:
        parts.append(str(endpoint))
    for key in ("worker_id", "dp_rank", "engine"):
        value = label_value(series, key)
        if value is not None:
            parts.append(f"{key}={value}")
    return "|".join(parts)
