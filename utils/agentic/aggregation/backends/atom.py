"""Native ATOM server metric adapter."""

from __future__ import annotations

from typing import Any

from ..aggregation_common import gauge_stat, normalize_fraction, rate, sum_stat
from .base import ServerMetricsBackend, counter_int


def _atom_names(suffix: str) -> list[str]:
    """Accept direct worker metrics and Atomesh's colon-normalized form."""
    return [f"atom:{suffix}", f"atom_{suffix}"]


def _atom_counter_names(stem: str) -> list[str]:
    """Accept Prometheus raw names and AIPerf's counter-family names."""
    return [*_atom_names(stem), *_atom_names(f"{stem}_total")]


class AtomBackend(ServerMetricsBackend):
    name = "atom"

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

        cached_tokens = sum_stat(
            metrics,
            _atom_counter_names("prefix_cache_cached_tokens"),
            preferred_keys=("total", "sum", "max", "avg"),
        )
        full_tokens = sum_stat(
            metrics,
            _atom_counter_names("prefix_cache_full_tokens"),
            preferred_keys=("total", "sum", "max", "avg"),
        )
        cache_hit_rate = rate(cached_tokens, full_tokens)
        external_tokens = sum_stat(
            metrics,
            _atom_counter_names("lmcache_loaded_tokens"),
            preferred_keys=("total", "sum", "max", "avg"),
        )
        external_hit_rate = rate(external_tokens, full_tokens)
        # ATOM's admitted cache counter may already include a completed
        # LMCache load, so do not add external tokens a second time.
        overall_hit_rate = cache_hit_rate

        flat["server_gpu_cache_hit_rate"] = cache_hit_rate
        flat["server_cpu_cache_hit_rate"] = external_hit_rate
        flat["server_external_cache_hit_rate"] = external_hit_rate
        flat["server_overall_cache_hit_rate"] = overall_hit_rate
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
                "gpu_cache_hit_rate": cache_hit_rate,
                "cpu_cache_hit_rate": external_hit_rate,
                "external_cache_hit_rate": external_hit_rate,
                "overall_cache_hit_rate": overall_hit_rate,
                "prefix_cache_hits": cached_tokens,
                "prefix_cache_queries": full_tokens,
                "external_prefix_cache_hits": external_tokens,
                "external_prefix_cache_queries": full_tokens,
            }
        )
        nested["kv_cache"]["gpu_usage_pct"] = flat["gpu_kv_cache_usage_pct"]
        nested["tokens"].update(
            {
                "prompt_total": flat["total_prompt_tokens"],
                "generation_total": flat["total_generation_tokens"],
            }
        )
