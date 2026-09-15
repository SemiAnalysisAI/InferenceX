"""SGLang server metric adapter."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any

from ..common import (
    gauge_stat,
    label_equals,
    metric_series,
    normalize_fraction,
    rate,
    series_stat,
    sum_by_label,
    sum_stat,
    sum_server_log_capacities,
    to_float,
)
from .base import ServerMetricsBackend, counter_int


class SglangBackend(ServerMetricsBackend):
    name = "sglang"
    _RANK_RE = re.compile(r"\b(?P<tag>DP\d+\s+TP\d+\s+EP\d+)\b")
    _MAX_TOKENS_RE = re.compile(r"\bmax_total_num_tokens=(?P<tokens>\d+)\b")
    _DP_SIZE_RE = re.compile(r"\bdp_size=(?P<dp_size>\d+)\b")

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
        # HiCache host hits are external to the GPU cache.
        flat["server_external_cache_hit_rate"] = flat["server_cpu_cache_hit_rate"]
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
        max_total_num_tokens = self.kv_cache_pool_tokens_from_metrics(metrics)

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
        server_logs: Iterable[str | None],
    ) -> int | None:
        if "sglang:max_total_num_tokens" in metrics:
            # Metrics cover all scraped roles; a log bundle may omit decode workers.
            return self.kv_cache_pool_tokens_from_metrics(metrics)
        return sum_server_log_capacities(
            server_logs,
            self.kv_cache_pool_tokens_from_server_log,
        )

    @staticmethod
    def kv_cache_pool_tokens_from_metrics(
        metrics: dict[str, dict[str, Any]],
    ) -> int | None:
        """Count each endpoint/DP pool once across replicated scheduler gauges."""
        series_list = metric_series(metrics, "sglang:max_total_num_tokens")
        if not any("tp_rank" in (series.get("labels") or {}) for series in series_list):
            return counter_int(
                sum_stat(
                    metrics, "sglang:max_total_num_tokens",
                    preferred_keys=("max", "avg", "total", "sum"),
                )
            )
        pools: dict[tuple[Any, ...], float] = {}
        shard_labels = {"tp_rank", "pp_rank", "ep_rank", "moe_ep_rank"}
        for series in series_list:
            value = series_stat(series, ("max", "avg", "total", "sum"))
            if (
                value is None or not math.isfinite(value)
                or value <= 0 or not value.is_integer()
            ):
                return None
            stats = series.get("stats", {})
            if any(
                stats.get(key) is not None and to_float(stats[key]) != value
                for key in ("min", "avg")
            ):
                # A changing gauge does not establish one constant pool ceiling.
                return None
            labels = series.get("labels") or {}
            if "tp_rank" in labels:
                endpoint = (
                    labels.get("worker_id") or series.get("worker_id")
                    or series.get("endpoint_url")
                )
                if not endpoint or not str(labels["tp_rank"]).isdigit():
                    return None
                identity = (
                    str(endpoint),
                    tuple(sorted(
                        (key, str(val)) for key, val in labels.items()
                        if key not in shard_labels
                    )),
                )
            else:
                # Mixing ranked and unranked gauges cannot establish ownership.
                return None
            if identity in pools and pools[identity] != value:
                return None
            pools[identity] = value
        return int(sum(pools.values())) if pools else None

    @classmethod
    def kv_cache_pool_tokens_from_server_log(cls, server_log: str | None) -> int | None:
        if not server_log:
            return None

        per_rank: dict[str, int] = {}
        bare_total = 0
        bare_count = 0
        dp_size = cls._dp_size(server_log)

        for line in server_log.splitlines():
            if "max_total_num_tokens" not in line:
                continue
            size_match = cls._MAX_TOKENS_RE.search(line)
            if not size_match:
                continue
            tokens = int(size_match.group("tokens"))
            if tokens <= 0:
                continue
            tag_match = cls._RANK_RE.search(line)
            if tag_match:
                per_rank[tag_match.group("tag")] = tokens
            else:
                bare_total += tokens
                bare_count += 1

        if per_rank:
            if dp_size is not None and len(per_rank) == 1 and dp_size > 1:
                return next(iter(per_rank.values())) * dp_size
            return sum(per_rank.values())
        if bare_count == 1 and dp_size is not None and dp_size > 1:
            return bare_total * dp_size
        return bare_total if bare_count else None

    @classmethod
    def _dp_size(cls, server_log: str) -> int | None:
        match = cls._DP_SIZE_RE.search(server_log)
        if not match:
            return None
        dp_size = int(match.group("dp_size"))
        return dp_size if dp_size > 0 else None
