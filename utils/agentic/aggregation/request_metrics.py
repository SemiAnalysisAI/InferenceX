"""Request/profile aggregation for aiperf agentic artifacts."""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .aggregation_common import percentile, stats_for, to_float, to_int
from .trace_metadata import expected_output_lengths


def load_aggregate(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def load_records(path: Path) -> list[dict[str, Any]]:
    records, _ = load_records_with_accounting(path)
    return records


def load_records_with_accounting(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load profiling records from profile_export.jsonl.

    Warmup rows are diagnostics only. Older artifacts did not have
    metadata.benchmark_phase, so missing phase is treated as profiling.
    """
    records: list[dict[str, Any]] = []
    accounting: dict[str, Any] = {
        "records_total": 0,
        "records_profiled": 0,
        "records_dropped_total": 0,
        "records_warmup_dropped": 0,
        "records_error_dropped": 0,
        "error_categories": {},
    }
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            accounting["records_total"] += 1
            phase = obj.get("metadata", {}).get("benchmark_phase")
            is_warmup = phase is not None and phase != "profiling"
            error = obj.get("error")
            if is_warmup:
                accounting["records_warmup_dropped"] += 1
            if error:
                accounting["records_error_dropped"] += 1
                category = _error_category(error)
                categories = accounting["error_categories"]
                categories[category] = categories.get(category, 0) + 1
            if error or is_warmup:
                continue
            records.append(obj)
    accounting["records_profiled"] = len(records)
    accounting["records_dropped_total"] = accounting["records_total"] - len(records)
    return records, accounting


def _error_category(error: Any) -> str:
    if isinstance(error, dict):
        for key in ("type", "error_type", "code", "class", "status"):
            value = error.get(key)
            if value not in (None, ""):
                return str(value)
        message = error.get("message") or error.get("error")
    else:
        message = error

    if not message:
        return "unknown"
    first_line = str(message).strip().splitlines()[0]
    return (first_line.split(":", 1)[0] or "unknown")[:120]


def _metric_value(record: dict[str, Any], key: str) -> Any:
    metric = record.get("metrics", {}).get(key)
    if isinstance(metric, dict):
        return metric.get("value")
    return metric


def extract_per_record_floats(records: list[dict[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for record in records:
        value = to_float(_metric_value(record, key))
        if value is not None:
            out.append(value)
    return out


def extract_per_record_ints(records: list[dict[str, Any]], key: str) -> list[int]:
    out: list[int] = []
    for record in records:
        value = to_int(_metric_value(record, key))
        if value is not None:
            out.append(value)
    return out


def _ms_to_s(values_ms: Iterable[float]) -> list[float]:
    return [value / 1000.0 for value in values_ms if value is not None and value > 0]


def _distribution(prefix: str, values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    return {
        f"mean_{prefix}": statistics.mean(values),
        f"p50_{prefix}": percentile([float(v) for v in values], 50),
        f"p75_{prefix}": percentile([float(v) for v in values], 75),
        f"p90_{prefix}": percentile([float(v) for v in values], 90),
        f"p95_{prefix}": percentile([float(v) for v in values], 95),
        f"std_{prefix}": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def _nest_stats(prefix: str, flat: dict[str, Any]) -> dict[str, Any]:
    suffix = f"_{prefix}"
    return {
        key[: -len(suffix)]: value
        for key, value in flat.items()
        if key.endswith(suffix)
    }


def _interactivity_stats(itl_stats: dict[str, Any], itls: list[float]) -> dict[str, float]:
    """Derive slow-tail interactivity from the matching ITL statistic."""
    out: dict[str, float] = {}
    for key in ("mean", "p50", "p75", "p90", "p95"):
        value = itl_stats.get(f"{key}_itl")
        if isinstance(value, int | float) and not isinstance(value, bool) and value > 0:
            out[f"{key}_intvty"] = 1.0 / value

    per_request = [1.0 / value for value in itls if value > 0]
    if per_request:
        out["std_intvty"] = (
            statistics.pstdev(per_request) if len(per_request) > 1 else 0.0
        )
    return out


def compute_latency_stats(records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    ttfts = _ms_to_s(extract_per_record_floats(records, "time_to_first_token"))
    e2els = _ms_to_s(extract_per_record_floats(records, "request_latency"))
    itls = _ms_to_s(extract_per_record_floats(records, "inter_token_latency"))
    ttft_stats = stats_for("ttft", ttfts)
    e2el_stats = stats_for("e2el", e2els)
    itl_stats = stats_for("itl", itls)
    tpot_stats = stats_for("tpot", itls)
    intvty_stats = _interactivity_stats(itl_stats, itls)

    flat: dict[str, Any] = {}
    flat.update(ttft_stats)
    flat.update(e2el_stats)
    flat.update(itl_stats)
    flat.update(tpot_stats)
    flat.update(intvty_stats)

    nested = {
        "ttft": _nest_stats("ttft", ttft_stats),
        "e2el": _nest_stats("e2el", e2el_stats),
        "itl": _nest_stats("itl", itl_stats),
        "tpot": _nest_stats("tpot", tpot_stats),
        "intvty": _nest_stats("intvty", intvty_stats),
    }
    return flat, nested


def compute_qps_stats(records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    ends_ns = [
        int(record["metadata"]["request_end_ns"])
        for record in records
        if record.get("metadata", {}).get("request_end_ns")
    ]
    if len(ends_ns) < 2:
        return {}, {}
    ends = sorted(timestamp / 1e9 for timestamp in ends_ns)
    duration = ends[-1] - ends[0]
    if duration <= 0:
        return {}, {}

    window = 1.0
    qps_values: list[float] = []
    current = ends[0]
    while current + window <= ends[-1]:
        count = sum(1 for completed_at in ends if current <= completed_at < current + window)
        qps_values.append(count / window)
        current += window

    if qps_values:
        flat = {
            "mean_qps": statistics.mean(qps_values),
            "p50_qps": percentile(qps_values, 50),
            "p75_qps": percentile(qps_values, 75),
            "p90_qps": percentile(qps_values, 90),
            "p95_qps": percentile(qps_values, 95),
            "std_qps": statistics.pstdev(qps_values) if len(qps_values) > 1 else 0.0,
        }
    else:
        flat = {"mean_qps": len(ends) / duration}
    return flat, {"window_seconds": window, "samples": len(qps_values), **_nest_stats("qps", flat)}


def compute_workload_stats(records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    input_tokens = extract_per_record_ints(records, "input_sequence_length")
    output_tokens = extract_per_record_ints(records, "output_sequence_length")

    flat: dict[str, Any] = {}
    flat.update(_distribution("input_tokens", input_tokens))
    flat.update(_distribution("output_tokens_actual", output_tokens))

    expected = expected_output_lengths(records)
    if expected:
        flat.update(_distribution("output_tokens_expected", expected))

    nested = {
        "input": _nest_stats("input_tokens", flat),
        "output_actual": _nest_stats("output_tokens_actual", flat),
        "output_expected": _nest_stats("output_tokens_expected", flat),
    }
    return flat, nested


def compute_throughput_stats(
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    input_tokens = extract_per_record_ints(records, "input_sequence_length")
    output_tokens = extract_per_record_ints(records, "output_sequence_length")
    starts_ns = [
        int(record["metadata"]["request_start_ns"])
        for record in records
        if record.get("metadata", {}).get("request_start_ns")
    ]
    ends_ns = [
        int(record["metadata"]["request_end_ns"])
        for record in records
        if record.get("metadata", {}).get("request_end_ns")
    ]
    if not starts_ns or not ends_ns:
        return {}, {}
    duration = (max(ends_ns) - min(starts_ns)) / 1e9
    if duration <= 0:
        return {}, {}

    total_input = sum(input_tokens)
    total_output = sum(output_tokens)
    flat = {
        "input_tput_tps": total_input / duration,
        "output_tput_tps": total_output / duration,
        "total_tput_tps": (total_input + total_output) / duration,
        "duration_seconds": duration,
    }
    nested = {
        "input": {"tokens_per_second": flat["input_tput_tps"]},
        "output": {"tokens_per_second": flat["output_tput_tps"]},
        "total": {"tokens_per_second": flat["total_tput_tps"]},
        "duration_seconds": duration,
        "per_gpu": {},
    }
    return flat, nested


def _aiperf_percent_metric_as_rate(
    aggregate: dict[str, Any],
    metric_name: str,
) -> float | None:
    metric = aggregate.get(metric_name)
    if not isinstance(metric, dict):
        return None

    hit_blocks = to_float(metric.get("sum"))
    total_blocks = to_float(metric.get("count"))
    if hit_blocks is not None and total_blocks is not None and total_blocks > 0:
        return hit_blocks / total_blocks

    avg = to_float(metric.get("avg"))
    if avg is None:
        return None
    unit = str(metric.get("unit", "")).strip()
    return avg / 100.0 if unit == "%" else avg


def compute_cache_stats(
    records: list[dict[str, Any]],
    aggregate: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    flat: dict[str, Any] = {
        "theoretical_cache_hit_rate": _aiperf_percent_metric_as_rate(
            aggregate,
            "theoretical_prefix_cache_hit",
        ),
    }

    return flat, {
        "theoretical_cache_hit_rate": flat["theoretical_cache_hit_rate"],
    }


def compute_request_metrics(
    records: list[dict[str, Any]],
    aggregate: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    aggregate = aggregate or {}
    flat: dict[str, Any] = {}
    nested: dict[str, Any] = {}

    qps_flat, qps_nested = compute_qps_stats(records)
    latency_flat, latency_nested = compute_latency_stats(records)
    workload_flat, workload_nested = compute_workload_stats(records)
    cache_flat, cache_nested = compute_cache_stats(records, aggregate)
    throughput_flat, throughput_nested = compute_throughput_stats(records)

    for part in (qps_flat, latency_flat, workload_flat, cache_flat, throughput_flat):
        flat.update(part)

    nested.update(
        {
            "qps": qps_nested,
            "latency": latency_nested,
            "tokens": workload_nested,
            "throughput": throughput_nested,
            "cache": cache_nested,
        }
    )
    return flat, nested
