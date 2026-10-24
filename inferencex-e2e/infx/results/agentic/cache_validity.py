from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

CORE_SCHEMA = "infx-cache-validity-core/1"
_STATUS_CLASS = {
    "VALID_COMPLETE": "VALID_COMPLETE",
    "VALID_ZERO": "VALID_ZERO",
    "CONDITIONAL_INTERVAL": "VALID_INTERVAL",
    "PARTIAL_COMPONENT": "PARTIAL",
    "PARTIAL_COVERAGE": "PARTIAL",
    "WINDOW_UNRESOLVED": "UNRESOLVED",
    "ROLE_OR_COVERAGE_UNRESOLVED": "UNRESOLVED",
    "REQUEST_SCOPE_UNRESOLVED": "UNRESOLVED",
    "MISSING_OBSERVATION": "MISSING",
    "INVALID_COUNTER": "INVALID",
    "REJECTED": "INVALID",
    "PHASE_CONTAMINATED": "CONTAMINATED",
    "NOT_EVALUATED": "NOT_EVALUATED",
}


def modern_profile_error(export: Mapping[str, Any], endpoints: Iterable[str] = ()) -> str | None:
    if (
        export.get("schema_version") != "1.1"
        or export.get("aiperf_version") != "1.0.12"
        or export.get("metrics_phase") != "profiling"
    ):
        return "AgentX 1.0.12 profiling export required"
    summary = export.get("summary")
    ranges = summary.get("phase_time_ranges") if isinstance(summary, dict) else None
    window = ranges.get("profiling") if isinstance(ranges, dict) else None
    if not isinstance(window, dict) or not (
        type(window.get("start_ns")) is int
        and type(window.get("end_ns")) is int
        and 0 <= window["start_ns"] < window["end_ns"]
    ):
        return "profiling time bounds missing or invalid"
    expected = set(endpoints)
    for key in ("endpoints_configured", "endpoints_successful"):
        covered = summary.get(key)
        if (
            not isinstance(covered, list)
            or not all(isinstance(url, str) for url in covered)
            or not expected.issubset(covered)
        ):
            return "profiling endpoint coverage incomplete"
    return None


def summarize_cache_validity(nested: dict[str, Any]) -> dict[str, Any]:
    cache = nested["cache"]
    tiers = any(
        cache.get(f"{tier}_cache_hit_rate") is not None for tier in ("gpu", "cpu", "external")
    )
    combined = cache.get("overall_cache_hit_rate") is not None
    detail = nested.get("cache_validity")
    if not isinstance(detail, dict):
        rate = nested.get("validity", {}).get("cache_rate", {})
        detail = {
            "status": rate.get("status", "NOT_EVALUATED"),
            "pricing_eligible": bool(rate.get("eligible_exact_actual")),
            "reasons": list(rate.get("reasons") or []),
        }
    if not nested.get("present"):
        detail = {"status": "MISSING_OBSERVATION", "reasons": ["server_metrics_missing"]}
    status = detail.get("status")
    core = {
        "schema": detail.get("schema", CORE_SCHEMA),
        "core_schema": CORE_SCHEMA,
        "status": status,
        "status_class": _STATUS_CLASS.get(status, "UNRESOLVED"),
        "published": tiers or combined,
        "pricing_eligible": bool(detail.get("pricing_eligible")),
        "basis": "tiers" if tiers else "combined" if combined else None,
        "reasons": list(detail.get("reasons") or []),
    }
    if isinstance(detail.get("request_scope"), dict):
        core["request_scope"] = detail["request_scope"]
    return core
