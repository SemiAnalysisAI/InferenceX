#!/usr/bin/env python3
"""Summarize TensorRT-LLM PyTorch Chrome traces without dumping raw JSON."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from io_utils import write_json


def discover_traces(inputs: Iterable[Path]) -> list[Path]:
    traces = []
    for path in inputs:
        if path.is_dir():
            traces.extend(
                path.rglob("*_torch_profile-rank-*.json")
            )
        elif path.is_file():
            traces.append(path)
        else:
            raise FileNotFoundError(path)
    return sorted(set(trace.resolve() for trace in traces))


def trace_payload(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if isinstance(payload, dict):
        events = payload.get("traceEvents", [])
        metadata = payload
    elif isinstance(payload, list):
        events = payload
        metadata = {}
    else:
        raise ValueError(f"Unsupported trace root in {path}")
    if not isinstance(events, list):
        raise ValueError(f"traceEvents is not a list in {path}")
    return (
        [event for event in events if isinstance(event, dict)],
        metadata,
    )


def trace_events(path: Path) -> list[dict[str, Any]]:
    events, _ = trace_payload(path)
    return events


def is_gpu_event(category: str) -> bool:
    category = category.lower()
    return (
        "kernel" in category
        or "gpu_memcpy" in category
        or "gpu_memset" in category
    )


def is_cpu_event(category: str) -> bool:
    category = category.lower()
    return category in {"cpu_op", "user_annotation"}


def top_rows(
    totals_us: dict[str, float],
    counts: dict[str, int],
    limit: int,
) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "total_ms": total_us / 1000.0,
            "count": counts[name],
            "mean_us": total_us / counts[name],
        }
        for name, total_us in sorted(
            totals_us.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:limit]
    ]


def kernel_family(name: str) -> str:
    lower = name.lower()
    patterns = (
        ("moe_gemm1", ("bmm_mxe4m3",)),
        ("moe_gemm2", ("bmm_bfloat16",)),
        ("moe_dispatch", ("moea2adispatch",)),
        ("moe_combine", ("moea2acombine",)),
        ("moe_sanitize", ("moea2asanitize",)),
        ("moe_finalize", ("finalizekernel",)),
        ("routing_topk", ("gate_forward_kernel", "topk")),
        ("mhc", ("fused_mhc", "hc_prenorm", "::mhc::")),
        ("dense_deepgemm", ("deep_gemm",)),
        ("dsa_blockwise_gemm", ("blockwise_gemm",)),
        ("attention_fmha", ("fmha",)),
        ("kv_compressor", ("compressor",)),
        ("mla_rope_kv", ("applymlarope", "inverserope")),
        ("quantization", ("quantize",)),
        ("nvjet_gemm", ("nvjet",)),
        ("collective", ("nccl", "allreduce", "all_reduce")),
        ("memcpy", ("memcpy",)),
    )
    for family, markers in patterns:
        if any(marker in lower for marker in markers):
            return family
    return "other"


def architecture_tag(name: str) -> str:
    lower = name.lower()
    if "sm103" in lower:
        return "sm103"
    if "sm100f" in lower:
        return "sm100f"
    if "sm100" in lower:
        return "sm100"
    return "untagged"


def interval_summary(
    intervals: list[tuple[float, float]],
) -> dict[str, float | None]:
    if not intervals:
        return {
            "sum_us": 0.0,
            "busy_union_us": 0.0,
            "window_us": None,
            "idle_within_window_us": None,
            "overlap_factor": None,
            "max_idle_gap_us": None,
        }
    ordered = sorted(intervals)
    sum_us = sum(stop - start for start, stop in ordered)
    window_start = ordered[0][0]
    current_start, current_stop = ordered[0]
    busy_union_us = 0.0
    max_idle_gap_us = 0.0
    for start, stop in ordered[1:]:
        if start <= current_stop:
            current_stop = max(current_stop, stop)
            continue
        busy_union_us += current_stop - current_start
        max_idle_gap_us = max(max_idle_gap_us, start - current_stop)
        current_start, current_stop = start, stop
    busy_union_us += current_stop - current_start
    window_us = max(stop for _, stop in ordered) - window_start
    return {
        "sum_us": sum_us,
        "busy_union_us": busy_union_us,
        "window_us": window_us,
        "idle_within_window_us": window_us - busy_union_us,
        "overlap_factor": (
            sum_us / busy_union_us if busy_union_us > 0 else None
        ),
        "max_idle_gap_us": max_idle_gap_us,
    }


def device_properties(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    devices = metadata.get("deviceProperties", [])
    if not isinstance(devices, list):
        return []
    rows = []
    for device in devices:
        if not isinstance(device, dict):
            continue
        major = device.get("computeMajor")
        minor = device.get("computeMinor")
        rows.append(
            {
                "id": device.get("id"),
                "name": device.get("name"),
                "total_global_mem_bytes": device.get("totalGlobalMem"),
                "compute_capability": (
                    f"{major}.{minor}"
                    if isinstance(major, int) and isinstance(minor, int)
                    else None
                ),
                "num_sms": device.get("numSms"),
            }
        )
    return rows


def summarize_trace(path: Path, limit: int) -> dict[str, Any]:
    events, metadata = trace_payload(path)
    category_totals: dict[str, float] = defaultdict(float)
    gpu_totals: dict[str, float] = defaultdict(float)
    gpu_counts: dict[str, int] = defaultdict(int)
    family_totals: dict[str, float] = defaultdict(float)
    family_counts: dict[str, int] = defaultdict(int)
    architecture_totals: dict[str, float] = defaultdict(float)
    architecture_counts: dict[str, int] = defaultdict(int)
    stream_totals: dict[str, float] = defaultdict(float)
    stream_counts: dict[str, int] = defaultdict(int)
    cpu_totals: dict[str, float] = defaultdict(float)
    cpu_counts: dict[str, int] = defaultdict(int)
    gpu_intervals: list[tuple[float, float]] = []
    starts = []
    stops = []
    duration_events = 0

    for event in events:
        if event.get("ph") != "X":
            continue
        duration = event.get("dur")
        timestamp = event.get("ts")
        if not isinstance(duration, (int, float)) or duration < 0:
            continue
        duration_events += 1
        category = str(event.get("cat") or "uncategorized")
        name = str(event.get("name") or "unnamed")
        category_totals[category] += float(duration)
        if is_gpu_event(category):
            gpu_totals[name] += float(duration)
            gpu_counts[name] += 1
            family = kernel_family(name)
            family_totals[family] += float(duration)
            family_counts[family] += 1
            architecture = architecture_tag(name)
            architecture_totals[architecture] += float(duration)
            architecture_counts[architecture] += 1
            stream = f"pid={event.get('pid')} tid={event.get('tid')}"
            stream_totals[stream] += float(duration)
            stream_counts[stream] += 1
            if isinstance(timestamp, (int, float)):
                gpu_intervals.append(
                    (
                        float(timestamp),
                        float(timestamp) + float(duration),
                    )
                )
        if is_cpu_event(category):
            cpu_totals[name] += float(duration)
            cpu_counts[name] += 1
        if isinstance(timestamp, (int, float)):
            starts.append(float(timestamp))
            stops.append(float(timestamp) + float(duration))

    intervals = interval_summary(gpu_intervals)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "device_properties": device_properties(metadata),
        "event_count": len(events),
        "duration_event_count": duration_events,
        "trace_span_ms": (
            (max(stops) - min(starts)) / 1000.0
            if starts and stops
            else None
        ),
        "category_total_ms": {
            category: duration / 1000.0
            for category, duration in sorted(
                category_totals.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        },
        "gpu_event_sum_ms": intervals["sum_us"] / 1000.0,
        "gpu_busy_union_ms": intervals["busy_union_us"] / 1000.0,
        "gpu_kernel_window_ms": (
            intervals["window_us"] / 1000.0
            if intervals["window_us"] is not None
            else None
        ),
        "gpu_idle_within_window_ms": (
            intervals["idle_within_window_us"] / 1000.0
            if intervals["idle_within_window_us"] is not None
            else None
        ),
        "gpu_overlap_factor": intervals["overlap_factor"],
        "max_gpu_idle_gap_ms": (
            intervals["max_idle_gap_us"] / 1000.0
            if intervals["max_idle_gap_us"] is not None
            else None
        ),
        "gpu_family_total_ms": {
            name: total_us / 1000.0
            for name, total_us in sorted(
                family_totals.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        },
        "top_gpu_families": top_rows(
            family_totals,
            family_counts,
            limit,
        ),
        "gpu_architecture_tags": top_rows(
            architecture_totals,
            architecture_counts,
            len(architecture_totals),
        ),
        "top_gpu_streams": top_rows(
            stream_totals,
            stream_counts,
            limit,
        ),
        "top_gpu_events": top_rows(gpu_totals, gpu_counts, limit),
        "top_cpu_events": top_rows(cpu_totals, cpu_counts, limit),
    }


def aggregate_traces(
    traces: list[dict[str, Any]],
    limit: int,
) -> dict[str, Any]:
    scalar_fields = (
        "gpu_event_sum_ms",
        "gpu_busy_union_ms",
        "gpu_kernel_window_ms",
        "gpu_idle_within_window_ms",
        "gpu_overlap_factor",
        "max_gpu_idle_gap_ms",
    )
    rank_mean = {}
    for field in scalar_fields:
        values = [
            float(trace[field])
            for trace in traces
            if trace.get(field) is not None
        ]
        rank_mean[field] = sum(values) / len(values) if values else None

    family_totals: dict[str, float] = defaultdict(float)
    family_counts: dict[str, int] = defaultdict(int)
    for trace in traces:
        for name, total_ms in trace["gpu_family_total_ms"].items():
            family_totals[name] += float(total_ms)
            family_counts[name] += 1
    family_rank_means = {
        name: total / family_counts[name]
        for name, total in family_totals.items()
    }
    return {
        "rank_mean": rank_mean,
        "gpu_family_rank_mean_ms": [
            {"name": name, "mean_total_ms": total}
            for name, total in sorted(
                family_rank_means.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:limit]
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top <= 0:
        raise ValueError("--top must be positive")
    traces = discover_traces(args.inputs)
    if not traces:
        raise FileNotFoundError("No *_torch_profile-rank-*.json traces found")
    trace_summaries = [
        summarize_trace(trace, args.top)
        for trace in traces
    ]
    summary = {
        "trace_count": len(traces),
        "aggregate": aggregate_traces(trace_summaries, args.top),
        "traces": trace_summaries,
    }
    if args.json_out is not None:
        write_json(args.json_out, summary)
    for trace in summary["traces"]:
        span = trace["trace_span_ms"]
        span_text = f"{span:.3f} ms" if span is not None else "unavailable"
        overlap = trace["gpu_overlap_factor"]
        overlap_text = f"{overlap:.3f}x" if overlap is not None else "n/a"
        print(
            f"{Path(trace['path']).name}: "
            f"{trace['event_count']} events, "
            f"span={span_text}, "
            f"gpu_busy={trace['gpu_busy_union_ms']:.3f} ms, "
            f"overlap={overlap_text}"
        )
        for row in trace["top_gpu_families"][: args.top]:
            print(
                f"  FAMILY {row['total_ms']:10.3f} ms "
                f"{row['count']:7d}x {row['name']}"
            )
        for row in trace["top_gpu_events"][: args.top]:
            print(
                f"  GPU {row['total_ms']:10.3f} ms "
                f"{row['count']:7d}x {row['name']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
