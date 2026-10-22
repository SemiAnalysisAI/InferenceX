from __future__ import annotations

from typing import Any

CORE_SCHEMA = "infx-cache-validity-core/1"
STATUS_CLASSES = (
    "VALID_COMPLETE",
    "VALID_ZERO",
    "VALID_INTERVAL",
    "PARTIAL",
    "UNRESOLVED",
    "CONTAMINATED",
    "INVALID",
    "MISSING",
    "NOT_EVALUATED",
)
_STATUS_CLASS = {
    "VALID_COMPLETE": "VALID_COMPLETE",
    "VALID_ZERO": "VALID_ZERO",
    "PARTIAL_COMPONENT": "PARTIAL",
    "WINDOW_UNRESOLVED": "UNRESOLVED",
    "ROLE_OR_COVERAGE_UNRESOLVED": "UNRESOLVED",
    "REQUEST_SCOPE_UNRESOLVED": "UNRESOLVED",
    "MISSING_OBSERVATION": "MISSING",
    "INVALID_COUNTER": "INVALID",
    "CONDITIONAL_INTERVAL": "VALID_INTERVAL",
    "UNKNOWN": "UNRESOLVED",
    "BOUNDARY_RESIDUAL": "UNRESOLVED",
    "BOUNDARY_UNCERTAIN": "UNRESOLVED",
    "PARTIAL_COVERAGE": "PARTIAL",
    "PHASE_CONTAMINATED": "CONTAMINATED",
    "REJECTED": "INVALID",
    "NOT_EVALUATED": "NOT_EVALUATED",
}
_TIER_RATES = ("gpu_cache_hit_rate", "cpu_cache_hit_rate", "external_cache_hit_rate")


def summarize_cache_validity(nested: dict[str, Any]) -> dict[str, Any]:
    cache = nested["cache"]
    tiers = any(cache.get(key) is not None for key in _TIER_RATES)
    combined = cache.get("overall_cache_hit_rate") is not None
    core = {
        "core_schema": CORE_SCHEMA,
        "published": tiers or combined,
        "basis": "tiers" if tiers else "combined" if combined else None,
    }
    detail = nested.get("cache_validity")
    if isinstance(detail, dict):
        return {
            **detail,
            **core,
            "status_class": _STATUS_CLASS.get(detail.get("status"), "UNRESOLVED"),
        }
    rate = nested["validity"]["cache_rate"]
    present = bool(nested.get("present"))
    status = rate.get("status") if present else "MISSING_OBSERVATION"
    return {
        "schema": CORE_SCHEMA,
        "status": status,
        "status_class": _STATUS_CLASS.get(status, "UNRESOLVED"),
        "pricing_eligible": present and bool(rate.get("eligible_exact_actual")),
        "reasons": list(rate.get("reasons") or []) if present else ["server_metrics_missing"],
        **core,
    }
