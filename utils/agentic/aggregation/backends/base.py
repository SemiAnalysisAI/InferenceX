"""Shared server metric adapter helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


SERVER_CACHE_FLAT_FIELDS = {
    "server_gpu_cache_hit_rate": None,
    "server_cpu_cache_hit_rate": None,
    "server_external_cache_hit_rate": None,
    "server_overall_cache_hit_rate": None,
    "gpu_kv_cache_usage_pct": None,
    "cpu_kv_cache_usage_pct": None,
    "kv_offload_bytes_gpu_to_cpu": None,
    "kv_offload_bytes_cpu_to_gpu": None,
    "kv_offload_time_gpu_to_cpu": None,
    "kv_offload_time_cpu_to_gpu": None,
    "kv_offload_bandwidth_gpu_to_cpu_bytes_per_second": None,
    "kv_offload_bandwidth_cpu_to_gpu_bytes_per_second": None,
    "total_prompt_tokens": None,
    "total_generation_tokens": None,
    "total_requests_completed": None,
}


class ServerMetricsBackend:
    name = ""

    def matches(self, metrics: dict[str, dict[str, Any]], framework: str) -> bool:
        raise NotImplementedError

    def populate(
        self,
        metrics: dict[str, dict[str, Any]],
        flat: dict[str, Any],
        nested: dict[str, Any],
    ) -> None:
        raise NotImplementedError

    def gpu_kv_capacity_tokens(
        self,
        metrics: dict[str, dict[str, Any]],
        server_log_paths: list[Path],
    ) -> int | None:
        return None


def empty_server_metrics(present: bool, metric_count: int) -> dict[str, Any]:
    return {
        "present": present,
        "adapter": "none",
        "metric_count": metric_count,
        "cache": {
            "gpu_cache_hit_rate": None,
            "cpu_cache_hit_rate": None,
            "external_cache_hit_rate": None,
            "overall_cache_hit_rate": None,
            "prefix_cache_hits": None,
            "prefix_cache_queries": None,
            "external_prefix_cache_hits": None,
            "external_prefix_cache_queries": None,
            "cached_tokens_by_source": {},
            "frontend_cache_hit_rate": None,
            "router_kv_hit_rate": None,
            "router_shared_cache_hit_rate": None,
            "frontend_cached_tokens": None,
            "frontend_input_tokens": None,
        },
        "kv_cache": {
            "gpu_usage_pct": None,
            "gpu_total_tokens": None,
            "cpu_usage_pct": None,
            "cpu_used_tokens": None,
            "cpu_total_tokens": None,
        },
        "kv_offload": {
            "bytes_gpu_to_cpu": None,
            "bytes_cpu_to_gpu": None,
            "time_gpu_to_cpu": None,
            "time_cpu_to_gpu": None,
            "bandwidth_gpu_to_cpu_bytes_per_second": None,
            "bandwidth_cpu_to_gpu_bytes_per_second": None,
        },
        "tokens": {
            "prompt_total": None,
            "generation_total": None,
            "requests_completed": None,
            "prompt_by_source": {
                "gpu_cache_hit": None,
                "cpu_or_external_cache_hit": None,
                "computed": None,
                "raw": {},
            },
        },
        "sources": [],
    }


def apply_profile_totals(flat: dict[str, Any], records: list[dict[str, Any]]) -> None:
    input_tokens = _record_token_sum(records, "input_sequence_length")
    output_tokens = _record_token_sum(records, "output_sequence_length")
    if flat["total_prompt_tokens"] is None and input_tokens is not None:
        flat["total_prompt_tokens"] = input_tokens
    if flat["total_generation_tokens"] is None and output_tokens is not None:
        flat["total_generation_tokens"] = output_tokens
    flat["total_requests_completed"] = len(records)


def counter_int(value: float | None) -> int | None:
    if value is None:
        return None
    return int(round(value))


def _record_token_sum(records: list[dict[str, Any]], metric_name: str) -> int | None:
    total = 0
    found = False
    for record in records:
        metric = record.get("metrics", {}).get(metric_name)
        value = metric.get("value") if isinstance(metric, dict) else metric
        if value is None:
            continue
        try:
            total += int(value)
            found = True
        except (TypeError, ValueError):
            continue
    return total if found else None
