"""Server metric normalization for agentic aggregate generation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .backends import detect_backend
from .backends.atom import AtomBackend
from .backends.base import (
    SERVER_CACHE_FLAT_FIELDS,
    apply_profile_totals,
    empty_server_metrics,
)
from .common import index_server_metrics


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


def compute_server_metrics(
    server_metrics: dict[str, Any],
    *,
    framework: str,
    records: list[dict[str, Any]],
    server_logs: Iterable[str | None] = (),
    env: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    metrics = index_server_metrics(server_metrics)
    flat = SERVER_CACHE_FLAT_FIELDS.copy()
    warnings: list[str] = []
    nested = empty_server_metrics(bool(metrics), len(metrics))
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
        return flat, nested, warnings

    if adapter is None:
        preview = ", ".join(sorted(metrics)[:10])
        raise ValueError(
            "Unsupported agentic server metrics backend; "
            f"framework={framework.lower()!r}, metric_names=[{preview}]"
        )
    nested["adapter"] = adapter.name
    if isinstance(adapter, AtomBackend):
        source_env = env or {}
        adapter.populate(
            metrics,
            flat,
            nested,
            endpoint_roles=endpoint_roles_from_env(source_env),
            disaggregated={"true": True, "false": False}.get(source_env.get("DISAGG", "").lower()),
            cpu_offload_disabled=(source_env.get("KV_OFFLOADING", "").strip().lower() == "none")
            and not (
                source_env.get("KV_OFFLOAD_BACKEND")
                or source_env.get("KV_OFFLOAD_BACKEND_METADATA")
            ),
        )
    else:
        adapter.populate(metrics, flat, nested)

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

    return flat, nested, warnings
