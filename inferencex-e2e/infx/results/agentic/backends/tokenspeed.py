from __future__ import annotations

from typing import Any

from infx.results.agentic.common import gauge_stat

from .base import ServerMetricsBackend


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
        # TokenSpeed exports device-page occupancy as a 0..1 fraction, despite the suffix.
        usage = gauge_stat(metrics, "tokenspeed:kv_cache_usage_perc", combine="max")
        flat["gpu_kv_cache_usage_pct"] = usage
        nested["kv_cache"]["gpu_usage_pct"] = usage
