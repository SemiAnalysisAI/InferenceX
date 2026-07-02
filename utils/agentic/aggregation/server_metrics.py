"""Server metric normalization for agentic aggregate generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .aggregation_common import (
    gauge_stat,
    index_server_metrics,
    label_equals,
    label_value,
    metric_series,
    normalize_fraction,
    rate,
    sum_by_label,
    sum_stat,
)


SERVER_CACHE_FLAT_FIELDS = {
    "server_gpu_cache_hit_rate": None,
    "server_cpu_cache_hit_rate": None,
    "server_external_cache_hit_rate": None,
    "server_overall_cache_hit_rate": None,
    "gpu_kv_cache_usage_pct": None,
    "cpu_kv_cache_usage_pct": None,
    "kv_offload_bytes_gpu_to_cpu": None,
    "kv_offload_bytes_cpu_to_gpu": None,
    "kv_offload_time_gpu_to_cpu": None,
    "kv_offload_time_cpu_to_gpu": None,
    "total_prompt_tokens": None,
    "total_generation_tokens": None,
    "total_requests_completed": None,
}


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
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    metrics = index_server_metrics(server_metrics)
    flat = SERVER_CACHE_FLAT_FIELDS.copy()
    warnings: list[str] = []
    nested = _empty_server_metrics(bool(metrics), len(metrics))

    if not metrics:
        warnings.append("server_metrics_export.json missing or empty")
        _apply_profile_totals(flat, records)
        nested["tokens"]["prompt_total"] = flat["total_prompt_tokens"]
        nested["tokens"]["generation_total"] = flat["total_generation_tokens"]
        nested["tokens"]["requests_completed"] = flat["total_requests_completed"]
        return flat, nested, warnings

    adapter = _detect_adapter(metrics, framework)
    nested["adapter"] = adapter

    if adapter == "sglang":
        _populate_sglang(metrics, flat, nested)
    else:
        _populate_vllm_family(metrics, flat, nested, adapter)
        if adapter == "dynamo-vllm":
            _populate_dynamo(metrics, flat, nested)

    _apply_profile_totals(flat, records)
    if nested["tokens"]["prompt_total"] is None:
        nested["tokens"]["prompt_total"] = flat["total_prompt_tokens"]
    if nested["tokens"]["generation_total"] is None:
        nested["tokens"]["generation_total"] = flat["total_generation_tokens"]
    nested["tokens"]["requests_completed"] = flat["total_requests_completed"]

    return flat, nested, warnings


def _empty_server_metrics(present: bool, metric_count: int) -> dict[str, Any]:
    return {
        "present": present,
        "adapter": "none",
        "metric_count": metric_count,
        "cache": {
            "gpu_cache_hit_rate": None,
            "cpu_cache_hit_rate": None,
            "external_cache_hit_rate": None,
            "overall_cache_hit_rate": None,
            "prefix_cache_hits": None,
            "prefix_cache_queries": None,
            "external_prefix_cache_hits": None,
            "external_prefix_cache_queries": None,
            "cached_tokens_by_source": {},
            "frontend_cache_hit_rate": None,
            "router_kv_hit_rate": None,
            "router_shared_cache_hit_rate": None,
            "frontend_cached_tokens": None,
            "frontend_input_tokens": None,
        },
        "kv_cache": {
            "gpu_usage_pct": None,
            "cpu_usage_pct": None,
            "cpu_used_tokens": None,
            "cpu_total_tokens": None,
        },
        "tokens": {
            "prompt_total": None,
            "generation_total": None,
            "requests_completed": None,
            "prompt_by_source": {
                "gpu_cache_hit": None,
                "cpu_or_external_cache_hit": None,
                "computed": None,
                "raw": {},
            },
        },
        "sources": [],
    }


def _detect_adapter(metrics: dict[str, dict[str, Any]], framework: str) -> str:
    metric_names = set(metrics)
    framework = framework.lower()
    if framework.startswith("dynamo") or any(name.startswith("dynamo_") for name in metric_names):
        return "dynamo-vllm"
    if any(name.startswith("sglang:") for name in metric_names):
        return "sglang"
    if any(name.startswith("vllm:") for name in metric_names):
        return "vllm"
    preview = ", ".join(sorted(metric_names)[:10])
    raise ValueError(
        "Unsupported agentic server metrics backend; "
        f"framework={framework!r}, metric_names=[{preview}]"
    )


def _apply_profile_totals(flat: dict[str, Any], records: list[dict[str, Any]]) -> None:
    input_tokens = _record_token_sum(records, "input_sequence_length")
    output_tokens = _record_token_sum(records, "output_sequence_length")
    if flat["total_prompt_tokens"] is None and input_tokens is not None:
        flat["total_prompt_tokens"] = input_tokens
    if flat["total_generation_tokens"] is None and output_tokens is not None:
        flat["total_generation_tokens"] = output_tokens
    flat["total_requests_completed"] = len(records)


def _record_token_sum(records: list[dict[str, Any]], metric_name: str) -> int | None:
    total = 0
    found = False
    for record in records:
        metric = record.get("metrics", {}).get(metric_name)
        value = metric.get("value") if isinstance(metric, dict) else metric
        if value is None:
            continue
        try:
            total += int(value)
            found = True
        except (TypeError, ValueError):
            continue
    return total if found else None


def _counter_int(value: float | None) -> int | None:
    if value is None:
        return None
    return int(round(value))


def _populate_vllm_family(
    metrics: dict[str, dict[str, Any]],
    flat: dict[str, Any],
    nested: dict[str, Any],
    adapter: str,
) -> None:
    if adapter == "dynamo-vllm":
        prompt_total = _first_counter_total(
            metrics,
            ["dynamo_frontend_input_sequence_tokens", "vllm:prompt_tokens"],
        )
        generation_total = _first_counter_total(
            metrics,
            ["dynamo_frontend_output_tokens", "vllm:generation_tokens"],
        )
    else:
        prompt_total = sum_stat(metrics, "vllm:prompt_tokens", preferred_keys=("total", "sum", "max", "avg"))
        generation_total = sum_stat(
            metrics,
            "vllm:generation_tokens",
            preferred_keys=("total", "sum", "max", "avg"),
        )
    flat["total_prompt_tokens"] = _counter_int(prompt_total)
    flat["total_generation_tokens"] = _counter_int(generation_total)

    prefix_hits = sum_stat(metrics, "vllm:prefix_cache_hits", preferred_keys=("total", "sum", "max", "avg"))
    prefix_queries = sum_stat(metrics, "vllm:prefix_cache_queries", preferred_keys=("total", "sum", "max", "avg"))
    gpu_rate = rate(prefix_hits, prefix_queries)
    flat["server_gpu_cache_hit_rate"] = gpu_rate

    external_hits = sum_stat(
        metrics,
        "vllm:external_prefix_cache_hits",
        preferred_keys=("total", "sum", "max", "avg"),
    )
    external_queries = sum_stat(
        metrics,
        "vllm:external_prefix_cache_queries",
        preferred_keys=("total", "sum", "max", "avg"),
    )
    external_rate = rate(external_hits, external_queries)
    flat["server_external_cache_hit_rate"] = external_rate
    flat["server_cpu_cache_hit_rate"] = external_rate

    prompt_by_source = sum_by_label(
        metrics,
        "vllm:prompt_tokens_by_source",
        "source",
        preferred_keys=("total", "sum", "max", "avg"),
    )
    if prompt_by_source:
        local_cache_hit = prompt_by_source.get("local_cache_hit")
        external_transfer = prompt_by_source.get("external_kv_transfer")
        local_compute = prompt_by_source.get("local_compute")
        source_total = sum(prompt_by_source.values())
        if source_total > 0:
            if local_cache_hit is not None:
                flat["server_gpu_cache_hit_rate"] = local_cache_hit / source_total
            if external_transfer is not None:
                flat["server_cpu_cache_hit_rate"] = external_transfer / source_total
                flat["server_external_cache_hit_rate"] = external_transfer / source_total
            cached_total = (local_cache_hit or 0.0) + (external_transfer or 0.0)
            flat["server_overall_cache_hit_rate"] = cached_total / source_total
        nested["tokens"]["prompt_by_source"] = {
            "gpu_cache_hit": local_cache_hit,
            "cpu_or_external_cache_hit": external_transfer,
            "computed": local_compute,
            "raw": prompt_by_source,
        }
    elif gpu_rate is not None:
        flat["server_overall_cache_hit_rate"] = gpu_rate

    gpu_usage = gauge_stat(
        metrics,
        ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"],
        preferred_keys=("max", "avg", "total"),
        combine="max",
    )
    flat["gpu_kv_cache_usage_pct"] = normalize_fraction(gpu_usage)

    cpu_usage = gauge_stat(
        metrics,
        "vllm:cpu_kv_cache_usage_perc",
        preferred_keys=("max", "avg", "total"),
        combine="max",
    )
    flat["cpu_kv_cache_usage_pct"] = normalize_fraction(cpu_usage)

    for metric_name, field_name in (
        ("vllm:kv_offload_bytes_gpu_to_cpu", "kv_offload_bytes_gpu_to_cpu"),
        ("vllm:kv_offload_bytes_cpu_to_gpu", "kv_offload_bytes_cpu_to_gpu"),
        ("vllm:kv_offload_time_gpu_to_cpu", "kv_offload_time_gpu_to_cpu"),
        ("vllm:kv_offload_time_cpu_to_gpu", "kv_offload_time_cpu_to_gpu"),
    ):
        flat[field_name] = sum_stat(metrics, metric_name, preferred_keys=("total", "sum", "max", "avg"))

    nested["cache"].update(
        {
            "gpu_cache_hit_rate": flat["server_gpu_cache_hit_rate"],
            "cpu_cache_hit_rate": flat["server_cpu_cache_hit_rate"],
            "external_cache_hit_rate": flat["server_external_cache_hit_rate"],
            "overall_cache_hit_rate": flat["server_overall_cache_hit_rate"],
            "prefix_cache_hits": prefix_hits,
            "prefix_cache_queries": prefix_queries,
            "external_prefix_cache_hits": external_hits,
            "external_prefix_cache_queries": external_queries,
        }
    )
    nested["kv_cache"].update(
        {
            "gpu_usage_pct": flat["gpu_kv_cache_usage_pct"],
            "cpu_usage_pct": flat["cpu_kv_cache_usage_pct"],
        }
    )
    nested["tokens"].update(
        {
            "prompt_total": flat["total_prompt_tokens"],
            "generation_total": flat["total_generation_tokens"],
        }
    )
    nested["sources"] = _vllm_sources(metrics)

def _populate_sglang(
    metrics: dict[str, dict[str, Any]],
    flat: dict[str, Any],
    nested: dict[str, Any],
) -> None:
    prompt_total = sum_stat(metrics, "sglang:prompt_tokens", preferred_keys=("total", "sum", "max", "avg"))
    generation_total = sum_stat(
        metrics,
        "sglang:generation_tokens",
        preferred_keys=("total", "sum", "max", "avg"),
    )
    flat["total_prompt_tokens"] = _counter_int(prompt_total)
    flat["total_generation_tokens"] = _counter_int(generation_total)

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


def _first_counter_total(
    metrics: dict[str, dict[str, Any]],
    metric_names: list[str],
) -> float | None:
    for metric_name in metric_names:
        value = sum_stat(metrics, metric_name, preferred_keys=("total", "sum", "max", "avg"))
        if value is not None:
            return value
    return None


def _populate_dynamo(
    metrics: dict[str, dict[str, Any]],
    flat: dict[str, Any],
    nested: dict[str, Any],
) -> None:
    dynamo_gpu_usage = normalize_fraction(
        gauge_stat(
            metrics,
            "dynamo_component_gpu_cache_usage_percent",
            preferred_keys=("max", "avg", "total"),
            combine="max",
        )
    )
    if flat["gpu_kv_cache_usage_pct"] is None:
        flat["gpu_kv_cache_usage_pct"] = dynamo_gpu_usage
        nested["kv_cache"]["gpu_usage_pct"] = dynamo_gpu_usage

    frontend_cached = sum_stat(
        metrics,
        "dynamo_frontend_cached_tokens",
        preferred_keys=("total", "sum", "max", "avg"),
    )
    frontend_input = sum_stat(
        metrics,
        "dynamo_frontend_input_sequence_tokens",
        preferred_keys=("total", "sum", "max", "avg"),
    )
    frontend_hit_rate = rate(frontend_cached, frontend_input)
    router_kv_hit_rate = normalize_fraction(
        gauge_stat(
            metrics,
            "dynamo_component_router_kv_hit_rate",
            preferred_keys=("avg", "max", "total"),
            combine="avg",
        )
    )
    router_shared_hit_rate = normalize_fraction(
        gauge_stat(
            metrics,
            "dynamo_component_router_shared_cache_hit_rate",
            preferred_keys=("avg", "max", "total"),
            combine="avg",
        )
    )
    if flat["server_overall_cache_hit_rate"] is None:
        flat["server_overall_cache_hit_rate"] = frontend_hit_rate or router_shared_hit_rate

    nested["cache"].update(
        {
            "frontend_cache_hit_rate": frontend_hit_rate,
            "router_kv_hit_rate": router_kv_hit_rate,
            "router_shared_cache_hit_rate": router_shared_hit_rate,
            "frontend_cached_tokens": frontend_cached,
            "frontend_input_tokens": frontend_input,
        }
    )


def _vllm_sources(metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    source_ids = set()
    for metric_name in (
        "vllm:prompt_tokens",
        "vllm:generation_tokens",
        "vllm:prefix_cache_hits",
        "vllm:prefix_cache_queries",
        "vllm:kv_cache_usage_perc",
        "vllm:prompt_tokens_by_source",
    ):
        for series in metric_series(metrics, metric_name):
            source_ids.add(_source_id(series))

    sources: list[dict[str, Any]] = []
    for source_id in sorted(source_ids):
        if not source_id:
            continue
        series_filter = lambda series, source_id=source_id: _source_id(series) == source_id
        prompt_tokens = sum_stat(metrics, "vllm:prompt_tokens", series_filter=series_filter)
        generation_tokens = sum_stat(metrics, "vllm:generation_tokens", series_filter=series_filter)
        hits = sum_stat(metrics, "vllm:prefix_cache_hits", series_filter=series_filter)
        queries = sum_stat(metrics, "vllm:prefix_cache_queries", series_filter=series_filter)
        kv_usage = normalize_fraction(
            gauge_stat(
                metrics,
                ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"],
                series_filter=series_filter,
            )
        )
        role = source_id.split("|", 1)[0]
        sources.append(
            {
                "id": source_id,
                "role": role,
                "prompt_tokens": prompt_tokens,
                "generation_tokens": generation_tokens,
                "prefix_cache_hit_rate": rate(hits, queries),
                "gpu_kv_cache_usage_pct": kv_usage,
            }
        )
    return sources


def _source_id(series: dict[str, Any]) -> str:
    labels = series.get("labels") if isinstance(series.get("labels"), dict) else {}
    component = str(labels.get("dynamo_component") or labels.get("component") or "")
    if component == "prefill":
        role = "prefill"
    elif component in ("backend", "decode"):
        role = "decode"
    elif component in ("frontend", "router"):
        role = "router"
    else:
        role = "combined"

    parts = [role]
    endpoint = series.get("endpoint_url")
    if endpoint:
        parts.append(str(endpoint))
    for key in ("worker_id", "dp_rank", "engine"):
        value = label_value(series, key)
        if value is not None:
            parts.append(f"{key}={value}")
    return "|".join(parts)
