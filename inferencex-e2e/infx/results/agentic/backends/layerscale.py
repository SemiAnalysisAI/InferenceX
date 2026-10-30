"""LayerScale server metric adapter."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from infx.results.agentic.common import gauge_stat, label_value, metric_series, rate, sum_stat

from .base import ServerMetricsBackend, counter_int


def _counter_names(stem: str) -> list[str]:
    """Accept Prometheus raw names and AIPerf's counter-family names."""
    return [f"layerscale_{stem}", f"layerscale_{stem}_total"]


def _counter(metrics: dict[str, dict[str, Any]], stem: str) -> float | None:
    """Return the run total of the LayerScale counter ``stem``, or None when it was not scraped."""
    return sum_stat(metrics, _counter_names(stem), preferred_keys=("total", "sum", "max", "avg"))


class LayerScaleBackend(ServerMetricsBackend):
    name = "layerscale"

    def matches(self, metrics: dict[str, dict[str, Any]], framework: str) -> bool:
        """Return True for layerscale_ metric names, or for an empty scrape when ``framework`` is layerscale."""
        return any(name.startswith("layerscale_") for name in metrics) or (
            not metrics and framework.lower() == "layerscale"
        )

    def populate(
        self,
        metrics: dict[str, dict[str, Any]],
        flat: dict[str, Any],
        nested: dict[str, Any],
    ) -> None:
        """Fill token totals, the GPU prefix hit rate and peak KV usage from LayerScale ``metrics``."""
        prompt_total = _counter(metrics, "prompt_tokens")
        generation_total = _counter(metrics, "generated_tokens")
        flat["total_prompt_tokens"] = counter_int(prompt_total)
        flat["total_generation_tokens"] = counter_int(generation_total)

        prefix_hits = _counter(metrics, "prefix_cached_tokens")
        session_hits = _counter(metrics, "session_cached_tokens")
        # Prefix pages and session state are both held in GPU memory; there is no host tier.
        gpu_hits = None if prefix_hits is None else prefix_hits + (session_hits or 0.0)
        hit_rate = rate(gpu_hits, prompt_total)
        flat["server_gpu_cache_hit_rate"] = hit_rate
        flat["server_overall_cache_hit_rate"] = hit_rate

        pages = gauge_stat(metrics, "layerscale_kv_pages_total", preferred_keys=("max", "avg"))
        fewest_free = gauge_stat(
            metrics, "layerscale_kv_pages_free", preferred_keys=("min", "avg"), combine="avg"
        )
        if pages is not None and fewest_free is not None:
            flat["gpu_kv_cache_usage_pct"] = rate(pages - fewest_free, pages)

        nested["cache"].update(
            {
                "gpu_cache_hit_rate": hit_rate,
                "overall_cache_hit_rate": hit_rate,
                "prefix_cache_hits": prefix_hits,
                "prefix_cache_queries": prompt_total,
                "cached_tokens_by_source": {
                    name: value
                    for name, value in (("prefix", prefix_hits), ("session", session_hits))
                    if value is not None
                },
            }
        )
        nested["kv_cache"]["gpu_usage_pct"] = flat["gpu_kv_cache_usage_pct"]
        nested["tokens"].update(
            {
                "prompt_total": flat["total_prompt_tokens"],
                "generation_total": flat["total_generation_tokens"],
            }
        )
        nested["tokens"]["prompt_by_source"]["gpu_cache_hit"] = gpu_hits

    def gpu_kv_capacity_tokens(
        self,
        metrics: dict[str, dict[str, Any]],
        server_logs: Iterable[str | None],  # noqa: ARG002
    ) -> int | None:
        """Return the KV pool's pages times the page size ``layerscale_build_info`` reports."""
        pages = gauge_stat(metrics, "layerscale_kv_pages_total", preferred_keys=("max", "avg"))
        sizes = {
            label_value(s, "page_size") for s in metric_series(metrics, "layerscale_build_info")
        }
        sizes.discard(None)
        if pages is None or len(sizes) != 1:
            return None
        return round(pages) * int(sizes.pop())
