from __future__ import annotations

from typing import Any

from infx.results.agentic.common import gauge_stat, rate, sum_stat

from .base import ServerMetricsBackend, counter_int


def _counter(metrics: dict[str, dict[str, Any]], stem: str) -> float | None:
    """Return the window total of ``tokenspeed:<stem>``, exported with or without ``_total``."""
    for metric_name in (f"tokenspeed:{stem}", f"tokenspeed:{stem}_total"):
        value = sum_stat(metrics, metric_name)
        if value is not None:
            return value
    return None


class TokenSpeedBackend(ServerMetricsBackend):
    name = "tokenspeed"

    def matches(self, metrics: dict[str, dict[str, Any]], framework: str) -> bool:
        return any(name.startswith("tokenspeed:") for name in metrics) or (
            not metrics and framework.lower() == "tokenspeed"
        )

    def populate(
        self,
        metrics: dict[str, dict[str, Any]],
        flat: dict[str, Any],
        nested: dict[str, Any],
    ) -> None:
        """Fill token totals, Device and Host prefix cache hits and KV usage from ``metrics``."""
        prompt_total = _counter(metrics, "prompt_tokens")
        generation_total = _counter(metrics, "generation_tokens")
        flat["total_prompt_tokens"] = counter_int(prompt_total)
        flat["total_generation_tokens"] = counter_int(generation_total)

        # Hits are counted at admission, so they divide by admitted, not ended, prompt tokens.
        admitted = _counter(metrics, "prefix_prompt_tokens")
        device_hits = _counter(metrics, "prefix_device_hit_tokens")
        # Includes tokens the Host cache loaded from L3; absent when no Host cache is configured.
        host_hits = _counter(metrics, "prefix_host_hit_tokens")
        cached_by_source = {
            source: hits
            for source, hits in (("device", device_hits), ("host", host_hits))
            if hits is not None
        }
        total_cached = sum(cached_by_source.values()) if cached_by_source else None
        computed = None
        if admitted is not None and total_cached is not None:
            computed = max(admitted - total_cached, 0.0)

        flat["server_gpu_cache_hit_rate"] = rate(device_hits, admitted)
        flat["server_cpu_cache_hit_rate"] = rate(host_hits, admitted)
        flat["server_external_cache_hit_rate"] = flat["server_cpu_cache_hit_rate"]
        flat["server_overall_cache_hit_rate"] = rate(total_cached, admitted)

        # TokenSpeed exports device-page occupancy as a 0..1 fraction, despite the suffix.
        usage = gauge_stat(metrics, "tokenspeed:kv_cache_usage_perc", combine="max")
        flat["gpu_kv_cache_usage_pct"] = usage

        nested["cache"].update(
            {
                "gpu_cache_hit_rate": flat["server_gpu_cache_hit_rate"],
                "cpu_cache_hit_rate": flat["server_cpu_cache_hit_rate"],
                "external_cache_hit_rate": flat["server_external_cache_hit_rate"],
                "overall_cache_hit_rate": flat["server_overall_cache_hit_rate"],
                "prefix_cache_hits": device_hits,
                "prefix_cache_queries": admitted,
                "external_prefix_cache_hits": host_hits,
                "external_prefix_cache_queries": admitted if host_hits is not None else None,
                "cached_tokens_by_source": cached_by_source,
            }
        )
        nested["kv_cache"]["gpu_usage_pct"] = usage
        nested["tokens"].update(
            {
                "prompt_total": flat["total_prompt_tokens"],
                "generation_total": flat["total_generation_tokens"],
                "prompt_by_source": {
                    "gpu_cache_hit": device_hits,
                    "cpu_or_external_cache_hit": host_hits,
                    "computed": computed,
                    "raw": cached_by_source,
                },
            }
        )
