"""Server metric normalization for agentic aggregate generation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .backends import detect_backend
from .backends.base import (
    SERVER_CACHE_FLAT_FIELDS,
    apply_profile_totals,
    empty_server_metrics,
)
from .cache_validity import summarize_cache_validity
from .common import index_server_metrics
from .phase_validity import (
    PhaseContext,
    export_context,
    phase_windows,
    summarize,
    warmup_totals,
)
from .request_metrics import extract_per_record_ints


def request_scope_from_records(
    records: list[dict[str, Any]], request_accounting: Mapping[str, Any]
) -> dict[str, Any]:
    """Client-side profiling cohort in server token units (``usage_prompt_tokens``).

    ``profiling_prompt_tokens`` is None when any successful record lacks usage, so a partial
    sum is never compared with server counters. ``error_records`` counts failed profiling
    requests; accounting without that split falls back to the all-phase error total.
    """
    prompt = extract_per_record_ints(records, "usage_prompt_tokens")
    return {
        "profiling_requests": len(records),
        "profiling_prompt_tokens": sum(prompt) if len(prompt) == len(records) else None,
        "error_records": request_accounting.get(
            "records_profiling_error_dropped", request_accounting.get("records_error_dropped")
        ),
        "cancelled_records": sum(
            1 for record in records if (record.get("metadata") or {}).get("was_cancelled")
        ),
    }


def deployment_from_env(env: Mapping[str, str]) -> dict[str, Any]:
    """Launcher-declared topology facts; absent or unrecognised values stay unknown (None)."""
    disagg = {"true": True, "false": False}.get((env.get("DISAGG") or "").lower())
    return {"disagg": disagg, "kv_offloading": env.get("KV_OFFLOADING") or None}


def endpoint_roles_from_env(env: Mapping[str, str]) -> dict[str, str] | None:
    """Map each scraped metrics URL to its launcher role, or None when roles are unknown.

    Mirrors ``infx.bench.agentic.replay._server_metrics_urls``: aggregated workers are
    ``unified``; otherwise prefill endpoints precede decode endpoints. Caller-supplied
    AIPERF_SERVER_METRICS_URLS and Dynamo frontends carry no roles, and an endpoint listed
    under two roles is ambiguous.
    """
    if env.get("AIPERF_SERVER_METRICS_URLS") or env.get("SRTCTL_FRONTEND_TYPE") == "dynamo":
        return None
    if env.get("SRT_AGG_ENDPOINTS"):
        groups = [("unified", env["SRT_AGG_ENDPOINTS"])]
    else:
        groups = [
            ("prefill", env.get("SRT_PREFILL_ENDPOINTS", "")),
            ("decode", env.get("SRT_DECODE_ENDPOINTS", "")),
        ]
    roles: dict[str, str] = {}
    for role, endpoints in groups:
        for endpoint in endpoints.split(","):
            if endpoint and roles.setdefault(f"http://{endpoint}/metrics", role) != role:
                return None
    return roles or None


def _profiling_window(server_metrics: dict[str, Any]) -> tuple[int, int] | None:
    summary = server_metrics.get("summary")
    ranges = summary.get("phase_time_ranges") if isinstance(summary, dict) else None
    profiling = ranges.get("profiling") if isinstance(ranges, dict) else None
    if not isinstance(profiling, dict):
        return None
    start, end = profiling.get("start_ns"), profiling.get("end_ns")
    if isinstance(start, int) and isinstance(end, int) and start < end:
        return start, end
    return None


def compute_server_metrics(
    server_metrics: dict[str, Any],
    *,
    framework: str,
    records: list[dict[str, Any]],
    server_logs: Iterable[str | None] = (),
    endpoint_roles: Mapping[str, str] | None = None,
    request_scope: Mapping[str, Any] | None = None,
    deployment: Mapping[str, Any] | None = None,
    disaggregated: bool | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    metrics = index_server_metrics(server_metrics)
    flat = SERVER_CACHE_FLAT_FIELDS.copy()
    warnings: list[str] = []
    nested = empty_server_metrics(bool(metrics), len(metrics))
    windows = phase_windows(server_metrics)
    exported = export_context(server_metrics, windows)
    unscraped = exported["endpoints"]["unscraped"]
    context = PhaseContext(
        windows=windows,
        warmup_totals=warmup_totals(server_metrics),
        unscraped=None if unscraped is None else tuple(unscraped),
        disaggregated=disaggregated,
    )
    nested["validity"].update(exported)
    nested["validity"]["deployment"] = {"disaggregated": disaggregated}
    nested["validity"].update(summarize(metrics, windows, context.warmup_totals))
    adapter = detect_backend(metrics, framework)

    if not metrics:
        warnings.append("server_metrics_export.json missing or empty")
        if adapter is not None:
            capacity = adapter.gpu_kv_capacity_tokens(metrics, server_logs)
            if capacity is not None:
                nested["kv_cache"]["gpu_total_tokens"] = capacity
        apply_profile_totals(flat, records)
        nested["tokens"]["prompt_total"] = flat["total_prompt_tokens"]
        nested["tokens"]["generation_total"] = flat["total_generation_tokens"]
        nested["tokens"]["requests_completed"] = flat["total_requests_completed"]
        nested["cache_validity"] = summarize_cache_validity(nested)
        return flat, nested, warnings

    if adapter is None:
        preview = ", ".join(sorted(metrics)[:10])
        raise ValueError(
            "Unsupported agentic server metrics backend; "
            f"framework={framework.lower()!r}, metric_names=[{preview}]"
        )
    nested["adapter"] = adapter.name
    if getattr(adapter, "accepts_context", False):
        warmup = server_metrics.get("warmup_metrics")
        adapter_context = {
            "framework": framework,
            "endpoint_roles": endpoint_roles,
            "profiling_window": _profiling_window(server_metrics),
            "warmup_metrics": warmup if isinstance(warmup, dict) else None,
            "request_scope": request_scope,
            **dict(deployment or {}),
            "warnings": warnings,
        }
        adapter.populate(metrics, flat, nested, context=adapter_context)
    else:
        adapter.populate(metrics, flat, nested)
    adapter.apply_validity(metrics, flat, nested, context)

    if framework.lower() == "dynamo-sglang" and adapter.name == "sglang":
        nested["kv_cache"]["gpu_total_tokens"] = None
        warnings.append(
            "Dynamo-SGLang logical KV capacity is unsupported: rank replicas are not normalized"
        )
    else:
        capacity = adapter.gpu_kv_capacity_tokens(metrics, server_logs)
        if capacity is not None:
            nested["kv_cache"]["gpu_total_tokens"] = capacity

    apply_profile_totals(flat, records)
    if nested["tokens"]["prompt_total"] is None:
        nested["tokens"]["prompt_total"] = flat["total_prompt_tokens"]
    if nested["tokens"]["generation_total"] is None:
        nested["tokens"]["generation_total"] = flat["total_generation_tokens"]
    nested["tokens"]["requests_completed"] = flat["total_requests_completed"]

    nested["cache_validity"] = summarize_cache_validity(nested)
    return flat, nested, warnings
