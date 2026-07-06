"""Dynamo-vLLM server metric adapter."""

from __future__ import annotations

from typing import Any

from ..aggregation_common import gauge_stat, normalize_fraction, rate, sum_stat
from .vllm import VllmBackend, first_counter_total


class DynamoVllmBackend(VllmBackend):
    name = "dynamo-vllm"

    def matches(self, metrics: dict[str, dict[str, Any]], framework: str) -> bool:
        metric_names = set(metrics)
        framework = framework.lower()
        return framework.startswith("dynamo") or any(
            name.startswith("dynamo_") for name in metric_names
        )

    def prompt_generation_totals(
        self,
        metrics: dict[str, dict[str, Any]],
    ) -> tuple[float | None, float | None]:
        prompt_total = first_counter_total(
            metrics,
            ["dynamo_frontend_input_sequence_tokens", "vllm:prompt_tokens"],
        )
        generation_total = first_counter_total(
            metrics,
            ["dynamo_frontend_output_tokens", "vllm:generation_tokens"],
        )
        return prompt_total, generation_total

    def populate(
        self,
        metrics: dict[str, dict[str, Any]],
        flat: dict[str, Any],
        nested: dict[str, Any],
    ) -> None:
        super().populate(metrics, flat, nested)
        self._populate_dynamo(metrics, flat, nested)

    def _populate_dynamo(
        self,
        metrics: dict[str, dict[str, Any]],
        flat: dict[str, Any],
        nested: dict[str, Any],
    ) -> None:
        dynamo_gpu_usage = normalize_fraction(
            gauge_stat(
                metrics,
                "dynamo_component_gpu_cache_usage_percent",
                preferred_keys=("max", "avg", "total"),
                combine="max",
            )
        )
        if flat["gpu_kv_cache_usage_pct"] is None:
            flat["gpu_kv_cache_usage_pct"] = dynamo_gpu_usage
            nested["kv_cache"]["gpu_usage_pct"] = dynamo_gpu_usage

        frontend_cached = sum_stat(
            metrics,
            "dynamo_frontend_cached_tokens",
            preferred_keys=("total", "sum", "max", "avg"),
        )
        frontend_input = sum_stat(
            metrics,
            "dynamo_frontend_input_sequence_tokens",
            preferred_keys=("total", "sum", "max", "avg"),
        )
        frontend_hit_rate = rate(frontend_cached, frontend_input)
        router_kv_hit_rate = normalize_fraction(
            gauge_stat(
                metrics,
                "dynamo_component_router_kv_hit_rate",
                preferred_keys=("avg", "max", "total"),
                combine="avg",
            )
        )
        router_shared_hit_rate = normalize_fraction(
            gauge_stat(
                metrics,
                "dynamo_component_router_shared_cache_hit_rate",
                preferred_keys=("avg", "max", "total"),
                combine="avg",
            )
        )
        if flat["server_overall_cache_hit_rate"] is None:
            flat["server_overall_cache_hit_rate"] = frontend_hit_rate or router_shared_hit_rate

        nested["cache"].update(
            {
                "frontend_cache_hit_rate": frontend_hit_rate,
                "router_kv_hit_rate": router_kv_hit_rate,
                "router_shared_cache_hit_rate": router_shared_hit_rate,
                "frontend_cached_tokens": frontend_cached,
                "frontend_input_tokens": frontend_input,
            }
        )
