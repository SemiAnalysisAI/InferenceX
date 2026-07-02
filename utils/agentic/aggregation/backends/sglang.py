"""SGLang server metric adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..aggregation_common import (
    gauge_stat,
    label_equals,
    normalize_fraction,
    rate,
    sum_by_label,
    sum_stat,
)
from ..server_log_metrics import sglang_kv_cache_pool_tokens_from_server_logs
from .base import ServerMetricsBackend, counter_int


class SglangBackend(ServerMetricsBackend):
    name = "sglang"

    def matches(self, metrics: dict[str, dict[str, Any]], framework: str) -> bool:
        metric_names = set(metrics)
        return any(name.startswith("sglang:") for name in metric_names) or (
            not metrics and framework.lower() == "sglang"
        )

    def populate(
        self,
        metrics: dict[str, dict[str, Any]],
        flat: dict[str, Any],
        nested: dict[str, Any],
    ) -> None:
        prompt_total = sum_stat(
            metrics,
            "sglang:prompt_tokens",
            preferred_keys=("total", "sum", "max", "avg"),
        )
        generation_total = sum_stat(
            metrics,
            "sglang:generation_tokens",
            preferred_keys=("total", "sum", "max", "avg"),
        )
        flat["total_prompt_tokens"] = counter_int(prompt_total)
        flat["total_generation_tokens"] = counter_int(generation_total)

        cached_by_source = sum_by_label(
            metrics,
            "sglang:cached_tokens",
            "cache_source",
            preferred_keys=("total", "sum", "max", "avg"),
        )
        device_hits = cached_by_source.get("device")
        host_hits = cached_by_source.get("host")
        total_cached = sum(cached_by_source.values()) if cached_by_source else None

        flat["server_gpu_cache_hit_rate"] = rate(device_hits, prompt_total)
        flat["server_cpu_cache_hit_rate"] = rate(host_hits, prompt_total)
        flat["server_overall_cache_hit_rate"] = rate(total_cached, prompt_total)

        if flat["server_overall_cache_hit_rate"] is None:
            flat["server_overall_cache_hit_rate"] = normalize_fraction(
                gauge_stat(
                    metrics,
                    "sglang:cache_hit_rate",
                    preferred_keys=("avg", "max", "total"),
                    combine="avg",
                )
            )

        flat["gpu_kv_cache_usage_pct"] = normalize_fraction(
            gauge_stat(
                metrics,
                "sglang:token_usage",
                preferred_keys=("max", "avg", "total"),
                combine="max",
            )
        )
        max_total_num_tokens = sum_stat(
            metrics,
            "sglang:max_total_num_tokens",
            preferred_keys=("max", "avg", "total", "sum"),
        )

        host_used = gauge_stat(
            metrics,
            "sglang:hicache_host_used_tokens",
            preferred_keys=("max", "avg", "total"),
            combine="max",
        )
        host_total = gauge_stat(
            metrics,
            "sglang:hicache_host_total_tokens",
            preferred_keys=("max", "avg", "total"),
            combine="max",
        )
        flat["cpu_kv_cache_usage_pct"] = rate(host_used, host_total)

        prefill_compute = sum_stat(
            metrics,
            "sglang:realtime_tokens",
            preferred_keys=("total", "sum", "max", "avg"),
            series_filter=label_equals("mode", "prefill_compute"),
        )

        nested["cache"].update(
            {
                "gpu_cache_hit_rate": flat["server_gpu_cache_hit_rate"],
                "cpu_cache_hit_rate": flat["server_cpu_cache_hit_rate"],
                "external_cache_hit_rate": flat["server_external_cache_hit_rate"],
                "overall_cache_hit_rate": flat["server_overall_cache_hit_rate"],
                "cached_tokens_by_source": cached_by_source,
            }
        )
        nested["kv_cache"].update(
            {
                "gpu_usage_pct": flat["gpu_kv_cache_usage_pct"],
                "gpu_total_tokens": counter_int(max_total_num_tokens),
                "cpu_usage_pct": flat["cpu_kv_cache_usage_pct"],
                "cpu_used_tokens": host_used,
                "cpu_total_tokens": host_total,
            }
        )
        nested["tokens"].update(
            {
                "prompt_total": flat["total_prompt_tokens"],
                "generation_total": flat["total_generation_tokens"],
                "prompt_by_source": {
                    "gpu_cache_hit": device_hits,
                    "cpu_or_external_cache_hit": host_hits,
                    "computed": prefill_compute,
                    "raw": cached_by_source,
                },
            }
        )

    def gpu_kv_capacity_tokens(
        self,
        metrics: dict[str, dict[str, Any]],
        server_log_paths: list[Path],
    ) -> int | None:
        return sglang_kv_cache_pool_tokens_from_server_logs(server_log_paths)
