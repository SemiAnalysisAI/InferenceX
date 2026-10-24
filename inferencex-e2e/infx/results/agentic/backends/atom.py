"""Native ATOM server metric adapter."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any

from infx.results.agentic.common import (
    gauge_stat,
    metric_series,
    normalize_fraction,
    rate,
    series_stat,
    sum_stat,
)

from .base import ServerMetricsBackend, counter_int

logger = logging.getLogger(__name__)


def _atom_names(suffix: str) -> list[str]:
    """Accept direct worker metrics and Atomesh's colon-normalized form."""
    return [f"atom:{suffix}", f"atom_{suffix}"]


def _atom_counter_names(stem: str) -> list[str]:
    """Accept Prometheus raw names and AIPerf's counter-family names."""
    return [*_atom_names(stem), *_atom_names(f"{stem}_total")]


def _counter_total(metrics: dict[str, dict[str, Any]], stem: str, endpoint: str) -> float | None:
    names = _atom_counter_names(f"prefix_cache_{stem}_tokens")
    series = [s for s in metric_series(metrics, names) if s.get("endpoint_url") == endpoint]
    if not series or any(
        (value := series_stat(s, preferred_keys=("total",))) is None
        or not math.isfinite(value)
        or value < 0
        for s in series
    ):
        logger.warning("ATOM %s cache counter missing or invalid at %s", stem, endpoint)
        return None
    total = sum_stat(
        metrics,
        names,
        preferred_keys=("total",),
        series_filter=lambda s: s.get("endpoint_url") == endpoint,
    )
    return total if total is not None and math.isfinite(total) else None


def _cache_counts(
    metrics: dict[str, dict[str, Any]],
    endpoint_roles: Mapping[str, str] | None,
    disaggregated: bool | None,
    cpu_offload_disabled: bool = False,
) -> tuple[float | None, float | None, float | None]:
    names = [
        name
        for stem in ("full", "cached", "offload")
        for name in _atom_counter_names(f"prefix_cache_{stem}_tokens")
    ]
    endpoints = [s.get("endpoint_url") for s in metric_series(metrics, names)]
    if any(not isinstance(endpoint, str) or not endpoint for endpoint in endpoints):
        return None, None, None
    observed = set(endpoints)
    if endpoint_roles is None:
        accounted = observed if disaggregated is False and len(observed) == 1 else set()
    else:
        # Decode counts the prompt again without serving its prefill cache reuse.
        accounted = {url for url, role in endpoint_roles.items() if role in {"prefill", "unified"}}
        if observed - endpoint_roles.keys():
            accounted = set()
    if not accounted:
        logger.warning("ATOM cache endpoint roles missing or ambiguous")
        return None, None, None
    full_total = cached_total = offload_total = 0.0
    cpu_complete = True
    offload_absent = not any(
        s.get("endpoint_url") in accounted
        for s in metric_series(metrics, _atom_counter_names("prefix_cache_offload_tokens"))
    )
    for endpoint in accounted:
        full, cached, offload = (
            _counter_total(metrics, stem, endpoint) for stem in ("full", "cached", "offload")
        )
        if cpu_offload_disabled and offload_absent:
            offload = 0.0
        if full is None or cached is None or cached > full:
            return None, None, None
        full_total += full
        cached_total += cached
        if offload is None or cached + offload > full:
            cpu_complete = False
        else:
            offload_total += offload
    if full_total <= 0 or not all(math.isfinite(v) for v in (full_total, cached_total)):
        return None, None, None
    return (
        full_total,
        cached_total,
        offload_total if cpu_complete and math.isfinite(offload_total) else None,
    )


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
        *,
        endpoint_roles: Mapping[str, str] | None = None,
        disaggregated: bool | None = None,
        cpu_offload_disabled: bool = False,
    ) -> None:
        prompt_names = _atom_counter_names("prompt_tokens")
        prompt_endpoints = {s.get("endpoint_url") for s in metric_series(metrics, prompt_names)}
        if endpoint_roles is None:
            accounted = (
                prompt_endpoints if disaggregated is False and len(prompt_endpoints) == 1 else set()
            )
        else:
            accounted = {
                url for url, role in endpoint_roles.items() if role in {"prefill", "unified"}
            }
            if prompt_endpoints - endpoint_roles.keys():
                accounted = set()
        accounted = {url for url in accounted if isinstance(url, str) and url}
        if accounted - prompt_endpoints:
            accounted = set()
        prompt_total = sum_stat(
            metrics,
            prompt_names,
            preferred_keys=("total",),
            series_filter=lambda s: s.get("endpoint_url") in accounted,
        )
        generation_total = sum_stat(
            metrics,
            _atom_counter_names("generation_tokens"),
            preferred_keys=("total", "sum", "max", "avg"),
        )
        flat["total_prompt_tokens"] = counter_int(prompt_total)
        flat["total_generation_tokens"] = counter_int(generation_total)

        full_tokens, cached_tokens, external_tokens = _cache_counts(
            metrics, endpoint_roles, disaggregated, cpu_offload_disabled
        )
        cache_hit_rate = rate(cached_tokens, full_tokens)
        # Offload is CPU reuse beyond the GPU prefix; loaded tokens measure transfer volume.
        external_hit_rate = rate(external_tokens, full_tokens)
        overall_hit_rate = (
            rate(cached_tokens + external_tokens, full_tokens)
            if cached_tokens is not None and external_tokens is not None
            else None
        )

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
                "external_prefix_cache_queries": (
                    full_tokens if external_tokens is not None else None
                ),
            }
        )
        nested["kv_cache"]["gpu_usage_pct"] = flat["gpu_kv_cache_usage_pct"]
        nested["tokens"].update(
            {
                "prompt_total": flat["total_prompt_tokens"],
                "generation_total": flat["total_generation_tokens"],
            }
        )
