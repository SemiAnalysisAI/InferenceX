"""Server metric normalization for agentic aggregate generation."""

from __future__ import annotations

import math
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
from .phase_validity import PhaseContext, export_context, phase_windows, summarize, warmup_totals


def request_scope_from_records(
    records: list[dict[str, Any]], request_accounting: Mapping[str, Any]
) -> dict[str, Any]:
    prompt = []
    for record in records:
        metric = record.get("metrics", {}).get("usage_prompt_tokens")
        value = metric.get("value") if isinstance(metric, dict) else metric
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or int(value) != value
        ):
            break
        prompt.append(int(value))
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
    offload = (env.get("KV_OFFLOADING") or "").strip().lower() or None
    if offload == "none" and (
        env.get("KV_OFFLOAD_BACKEND") or env.get("KV_OFFLOAD_BACKEND_METADATA")
    ):
        offload = None
    return {
        "disagg": {"true": True, "false": False}.get((env.get("DISAGG") or "").lower()),
        "kv_offloading": offload,
    }


def _profiling_window(server_metrics: dict[str, Any]) -> tuple[int, int] | None:
    summary = server_metrics.get("summary")
    ranges = summary.get("phase_time_ranges") if isinstance(summary, dict) else None
    profiling = ranges.get("profiling") if isinstance(ranges, dict) else None
    if isinstance(profiling, dict):
        start, end = profiling.get("start_ns"), profiling.get("end_ns")
        if type(start) is int and type(end) is int and start < end:
            return start, end
    return None


def endpoint_roles_from_env(env: Mapping[str, str]) -> dict[str, str] | None:
    if env.get("SRTCTL_FRONTEND_TYPE") == "dynamo":
        return {}
    if env.get("AIPERF_SERVER_METRICS_URLS"):
        return None
    groups = (
        [("unified", env["SRT_AGG_ENDPOINTS"])]
        if env.get("SRT_AGG_ENDPOINTS")
        else [
            ("prefill", env.get("SRT_PREFILL_ENDPOINTS", "")),
            ("decode", env.get("SRT_DECODE_ENDPOINTS", "")),
        ]
    )
    roles: dict[str, str] = {}
    for role, endpoints in groups:
        for endpoint in endpoints.split(","):
            if endpoint:
                url = f"http://{endpoint}/metrics"
                if roles.setdefault(url, role) != role:
                    return {}
    return roles or None


def _integer_total(values: list[Any]) -> int | None:
    if not values or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or (isinstance(value, float) and not math.isfinite(value))
        or value < 0
        or int(value) != value
        for value in values
    ):
        return None
    return sum(int(value) for value in values)


def _disaggregated_prompt_total(
    metrics: dict[str, dict[str, Any]], records: list[dict[str, Any]], nested: dict[str, Any]
) -> int | None:
    values = []
    for record in records:
        metric = record.get("metrics", {}).get("input_sequence_length")
        values.append(metric.get("value") if isinstance(metric, dict) else metric)
    prompt = _integer_total(values)
    if prompt is not None:
        return prompt
    frontend = metrics.get("dynamo_frontend_input_sequence_tokens")
    if frontend is not None:
        series = frontend.get("series")
        if not isinstance(series, list):
            return None
        return _integer_total(
            [
                item.get("stats", {}).get("total")
                if isinstance(item, dict) and isinstance(item.get("stats"), dict)
                else None
                for item in series
            ]
        )
    reuse = nested["validity"]["cache_rate"]
    if reuse.get("published") and reuse.get("role_scope") == "prefill":
        return _integer_total([reuse.get("prompt_tokens")])
    return None


def _apply_disaggregated_prompt_total(
    metrics: dict[str, dict[str, Any]],
    records: list[dict[str, Any]],
    flat: dict[str, Any],
    nested: dict[str, Any],
) -> None:
    prompt = _disaggregated_prompt_total(metrics, records, nested)
    if prompt != nested["tokens"]["prompt_total"]:
        nested["tokens"]["prompt_by_source"]["computed"] = None
    flat["total_prompt_tokens"] = nested["tokens"]["prompt_total"] = prompt


def compute_server_metrics(
    server_metrics: dict[str, Any],
    *,
    framework: str,
    records: list[dict[str, Any]],
    server_logs: Iterable[str | None] = (),
    env: Mapping[str, str] | None = None,
    endpoint_roles: Mapping[str, str] | None = None,
    request_scope: Mapping[str, Any] | None = None,
    deployment: Mapping[str, Any] | None = None,
    disaggregated: bool | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    metrics = index_server_metrics(server_metrics)
    flat = SERVER_CACHE_FLAT_FIELDS.copy()
    warnings: list[str] = []
    nested = empty_server_metrics(bool(metrics), len(metrics))
    source_env = env or {}
    deployment = {**deployment_from_env(source_env), **dict(deployment or {})}
    if disaggregated is None:
        disaggregated = deployment.get("disagg")
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
        if disaggregated is True and (adapter is None or adapter.name != "atom"):
            _apply_disaggregated_prompt_total(metrics, records, flat, nested)
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
        adapter.populate(
            metrics,
            flat,
            nested,
            context={
                "framework": framework,
                "endpoint_roles": endpoint_roles
                if endpoint_roles is not None
                else endpoint_roles_from_env(source_env),
                "profiling_window": _profiling_window(server_metrics),
                "warmup_metrics": warmup if isinstance(warmup, dict) else None,
                "request_scope": request_scope,
                **deployment,
                "collector_export": server_metrics,
                "warnings": warnings,
            },
        )
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

    if disaggregated is True and adapter.name != "atom":
        _apply_disaggregated_prompt_total(metrics, records, flat, nested)

    nested["cache_validity"] = summarize_cache_validity(nested)
    return flat, nested, warnings
