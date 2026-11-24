"""TensorRT-LLM server metric adapter."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from infx.results.agentic.common import (
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
        """Return True for trtllm_ metric names, or for an empty scrape from a TRT-LLM ``framework``."""
        metric_names = set(metrics)
        return any(name.startswith("trtllm_") for name in metric_names) or (
            not metrics and framework.lower() in ("trtllm", "dynamo-trt", "dynamo-trtllm")
        )

    def populate(
        self,
        metrics: dict[str, dict[str, Any]],
        flat: dict[str, Any],
        nested: dict[str, Any],
    ) -> None:
        """Fill token totals, KV hit rate, usage and offload bytes and per-endpoint sources from ``metrics``."""
        roles = _endpoint_roles(metrics)
        prefill_endpoints = {endpoint for endpoint, role in roles.items() if role == "prefill"}
        scope = _endpoint_filter(prefill_endpoints) if prefill_endpoints else None
        prompt_total = sum_stat(metrics, "dynamo_frontend_input_sequence_tokens")
        if prompt_total is None:
            prompt_total = _counter(metrics, "trtllm_prompt_tokens", scope)
        generation_total = _first_counter_total(
            metrics,
            [
                "dynamo_frontend_output_tokens",
                "trtllm_generation_tokens_total",
                "trtllm_generation_tokens",
            ],
        )
        # Prefix reuse saves compute only where prompts are prefilled; a disaggregated decode
        # worker receives its KV by transfer, so its hits measure transfer savings instead.
        hit_rate, cached_tokens, queried_tokens = _prefix_hit_rate(
            metrics, scope, None if scope else prompt_total
        )

        flat["total_prompt_tokens"] = counter_int(prompt_total)
        flat["total_generation_tokens"] = counter_int(generation_total)
        flat["server_gpu_cache_hit_rate"] = hit_rate
        flat["server_overall_cache_hit_rate"] = hit_rate

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
        flat["kv_offload_bytes_gpu_to_cpu"] = _counter(metrics, "trtllm_kv_cache_offload_bytes")
        flat["kv_offload_bytes_cpu_to_gpu"] = _counter(metrics, "trtllm_kv_cache_onboard_bytes")

        computed_tokens = None
        if queried_tokens is not None and cached_tokens is not None:
            computed_tokens = max(queried_tokens - cached_tokens, 0.0)

        nested["cache"].update(
            {
                "gpu_cache_hit_rate": flat["server_gpu_cache_hit_rate"],
                "overall_cache_hit_rate": flat["server_overall_cache_hit_rate"],
                "prefix_cache_hits": cached_tokens,
                "prefix_cache_queries": queried_tokens,
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
                    "raw": {"device": cached_tokens} if cached_tokens is not None else {},
                },
            }
        )
        nested["sources"] = _trtllm_sources(metrics)

    def gpu_kv_capacity_tokens(
        self,
        metrics: dict[str, dict[str, Any]],
        server_logs: Iterable[str | None],
    ) -> int | None:
        """Return KV max blocks times tokens per block from ``metrics``, or None if either is missing."""
        del server_logs
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


def _counter(
    metrics: dict[str, dict[str, Any]],
    base_name: str,
    series_filter: Callable[[dict[str, Any]], bool] | None = None,
) -> float | None:
    """Return counter ``base_name``'s window total over ``series_filter``.

    aiperf exports a counter under its ``_total`` name or, as aiperf 1.0.12 does, its base name.
    """
    for metric_name in (f"{base_name}_total", base_name):
        value = sum_stat(metrics, metric_name, series_filter=series_filter)
        if value is not None:
            return value
    return None


def _first_counter_total(
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


def _endpoint_roles(metrics: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Map every scraped endpoint URL in ``metrics`` to its role, ``worker`` when no series names one."""
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
            roles.setdefault(endpoint, "worker")
            # Core trtllm_ series carry no role label; Dynamo's own series on the same endpoint do.
            mode = label_value(series, "disaggregation_mode") or label_value(
                series, "dynamo_component"
            )
            if mode:
                role = _normalize_role(mode)
                if role in {"prefill", "decode", "combined"} or roles[endpoint] == "worker":
                    roles[endpoint] = role
    return roles


def _endpoint_filter(endpoints: set[str]) -> Callable[[dict[str, Any]], bool]:
    """Return a series filter that keeps series scraped from one of ``endpoints``."""
    return lambda series: str(series.get("endpoint_url", "")) in endpoints


def _prefix_hit_rate(
    metrics: dict[str, dict[str, Any]],
    series_filter: Callable[[dict[str, Any]], bool] | None,
    fallback_queried: float | None,
) -> tuple[float | None, float | None, float | None]:
    """Return the prefix hit rate with its cached and queried token counts over ``series_filter``.

    Prefers window token counters, then window block counters, then the hit-rate gauge;
    ``fallback_queried`` stands in for a missing prompt-token counter. Counts are None unless
    the rate came from token counters.
    """
    cached = _counter(metrics, "trtllm_prompt_cached_tokens", series_filter)
    queried = _counter(metrics, "trtllm_prompt_tokens", series_filter)
    if queried is None:
        queried = fallback_queried
    token_rate = rate(cached, queried)
    if token_rate is not None:
        return token_rate, cached, queried

    reused = _counter(metrics, "trtllm_kv_cache_reused_blocks", series_filter)
    missed = _counter(metrics, "trtllm_kv_cache_missed_blocks", series_filter)
    if reused is not None and missed is not None:
        block_rate = rate(reused, reused + missed)
        if block_rate is not None:
            return block_rate, None, None

    # The gauge is TRT-LLM's process-lifetime ratio, so it includes warmup traffic.
    gauge_rate = normalize_fraction(
        gauge_stat(
            metrics,
            "trtllm_kv_cache_hit_rate",
            preferred_keys=("avg", "max", "total"),
            combine="avg",
            series_filter=series_filter,
        )
    )
    return gauge_rate, None, None


def _trtllm_sources(metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Return per-endpoint token, hit rate and KV usage rows from TRT-LLM ``metrics``, sorted by endpoint."""
    roles = _endpoint_roles(metrics)

    sources: list[dict[str, Any]] = []
    for endpoint in sorted(roles):
        series_filter = _endpoint_filter({endpoint})

        prompt_tokens = _counter(metrics, "trtllm_prompt_tokens", series_filter)
        generation_tokens = _counter(metrics, "trtllm_generation_tokens", series_filter)
        hit_rate, _, _ = _prefix_hit_rate(metrics, series_filter, None)
        sources.append(
            {
                "id": f"{roles[endpoint]}|{endpoint}",
                "role": roles[endpoint],
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
    """Map a disaggregation ``mode`` label to prefill, decode or combined, else return it lowercased."""
    lowered = mode.lower()
    if lowered in ("prefill", "context", "ctx"):
        return "prefill"
    if lowered in ("decode", "generation", "gen", "backend"):
        return "decode"
    if lowered in ("aggregated", "prefill_and_decode", "agg"):
        return "combined"
    return lowered
