"""TensorRT-LLM server metric adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..aggregation_common import (
    gauge_stat,
    label_value,
    normalize_fraction,
    rate,
    sum_stat,
)
from .base import ServerMetricsBackend, counter_int


class TrtllmBackend(ServerMetricsBackend):
    name = "trtllm"

    def matches(self, metrics: dict[str, dict[str, Any]], framework: str) -> bool:
        metric_names = set(metrics)
        return any(name.startswith("trtllm_") for name in metric_names) or (
            not metrics
            and framework.lower() in ("trtllm", "dynamo-trt", "dynamo-trtllm")
        )

    def populate(
        self,
        metrics: dict[str, dict[str, Any]],
        flat: dict[str, Any],
        nested: dict[str, Any],
    ) -> None:
        prompt_total = _first_counter_total(
            metrics,
            ["dynamo_frontend_input_sequence_tokens", "trtllm_prompt_tokens_total"],
        )
        generation_total = _first_counter_total(
            metrics,
            ["dynamo_frontend_output_tokens", "trtllm_generation_tokens_total"],
        )
        cached_tokens = sum_stat(
            metrics,
            "trtllm_prompt_cached_tokens_total",
            preferred_keys=("total", "sum", "max", "avg"),
        )

        flat["total_prompt_tokens"] = counter_int(prompt_total)
        flat["total_generation_tokens"] = counter_int(generation_total)
        flat["server_gpu_cache_hit_rate"] = normalize_fraction(
            gauge_stat(
                metrics,
                "trtllm_kv_cache_hit_rate",
                preferred_keys=("avg", "max", "total"),
                combine="avg",
            )
        )
        if flat["server_gpu_cache_hit_rate"] is None:
            flat["server_gpu_cache_hit_rate"] = rate(cached_tokens, prompt_total)
        flat["server_overall_cache_hit_rate"] = flat["server_gpu_cache_hit_rate"]

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
        flat["kv_offload_bytes_gpu_to_cpu"] = sum_stat(
            metrics,
            "trtllm_kv_cache_offload_bytes_total",
            preferred_keys=("total", "sum", "max", "avg"),
        )
        flat["kv_offload_bytes_cpu_to_gpu"] = sum_stat(
            metrics,
            "trtllm_kv_cache_onboard_bytes_total",
            preferred_keys=("total", "sum", "max", "avg"),
        )

        computed_tokens = None
        if prompt_total is not None and cached_tokens is not None:
            computed_tokens = max(prompt_total - cached_tokens, 0.0)

        nested["cache"].update(
            {
                "gpu_cache_hit_rate": flat["server_gpu_cache_hit_rate"],
                "overall_cache_hit_rate": flat["server_overall_cache_hit_rate"],
                "cached_tokens_by_source": {"device": cached_tokens}
                if cached_tokens is not None
                else {},
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
                    "gpu_cache_hit": cached_tokens,
                    "cpu_or_external_cache_hit": None,
                    "computed": computed_tokens,
                    "raw": {"device": cached_tokens}
                    if cached_tokens is not None
                    else {},
                },
            }
        )
        nested["sources"] = _trtllm_sources(metrics)

    def gpu_kv_capacity_tokens(
        self,
        metrics: dict[str, dict[str, Any]],
        server_log_paths: list[Path],
    ) -> int | None:
        del server_log_paths
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


def _first_counter_total(
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


def _trtllm_sources(metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    endpoints: set[str] = set()
    roles: dict[str, str] = {}
    for entry in metrics.values():
        if not isinstance(entry, dict):
            continue
        series_list = entry.get("series")
        if not isinstance(series_list, list):
            continue
        for series in series_list:
            if not isinstance(series, dict) or not series.get("endpoint_url"):
                continue
            endpoint = str(series["endpoint_url"])
            endpoints.add(endpoint)
            mode = label_value(series, "disaggregation_mode") or label_value(
                series, "dynamo_component"
            )
            if mode:
                roles[endpoint] = _normalize_role(mode)

    sources: list[dict[str, Any]] = []
    for endpoint in sorted(endpoints):
        series_filter = lambda series, endpoint=endpoint: str(
            series.get("endpoint_url", "")
        ) == endpoint
        prompt_tokens = sum_stat(
            metrics,
            "trtllm_prompt_tokens_total",
            series_filter=series_filter,
        )
        generation_tokens = sum_stat(
            metrics,
            "trtllm_generation_tokens_total",
            series_filter=series_filter,
        )
        cached_tokens = sum_stat(
            metrics,
            "trtllm_prompt_cached_tokens_total",
            series_filter=series_filter,
        )
        hit_rate = normalize_fraction(
            gauge_stat(
                metrics,
                "trtllm_kv_cache_hit_rate",
                combine="avg",
                series_filter=series_filter,
            )
        )
        if hit_rate is None:
            hit_rate = rate(cached_tokens, prompt_tokens)
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
