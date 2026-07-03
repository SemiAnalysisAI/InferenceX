"""Server metric normalization for agentic aggregate generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .aggregation_common import index_server_metrics
from .backends import detect_backend
from .backends.base import (
    SERVER_CACHE_FLAT_FIELDS,
    apply_profile_totals,
    empty_server_metrics,
)


def load_server_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def compute_server_metrics(
    server_metrics: dict[str, Any],
    *,
    framework: str,
    records: list[dict[str, Any]],
    server_log_paths: list[Path] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    metrics = index_server_metrics(server_metrics)
    flat = SERVER_CACHE_FLAT_FIELDS.copy()
    warnings: list[str] = []
    nested = empty_server_metrics(bool(metrics), len(metrics))
    log_paths = server_log_paths or []
    adapter = detect_backend(metrics, framework)

    if not metrics:
        warnings.append("server_metrics_export.json missing or empty")
        if adapter is not None:
            capacity = adapter.gpu_kv_capacity_tokens(metrics, log_paths)
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
    adapter.populate(metrics, flat, nested)

    capacity = adapter.gpu_kv_capacity_tokens(metrics, log_paths)
    if capacity is not None:
        nested["kv_cache"]["gpu_total_tokens"] = capacity

    apply_profile_totals(flat, records)
    if nested["tokens"]["prompt_total"] is None:
        nested["tokens"]["prompt_total"] = flat["total_prompt_tokens"]
    if nested["tokens"]["generation_total"] is None:
        nested["tokens"]["generation_total"] = flat["total_generation_tokens"]
    nested["tokens"]["requests_completed"] = flat["total_requests_completed"]

    return flat, nested, warnings
