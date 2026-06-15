#!/usr/bin/env python3
"""Summarize a single-iteration PyTorch/Perfetto profile trace."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

ANNOTATION_RE = re.compile(
    r"execute_context_(?P<context_requests>\d+)"
    r"\((?P<context_tokens>\d+)\)_generation_"
    r"(?P<generation_requests>\d+)\((?P<generation_tokens>\d+)\)"
)


def _open_trace(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _duration(event: dict[str, Any]) -> float:
    value = event.get("dur", 0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _is_kernel(event: dict[str, Any]) -> bool:
    if event.get("ph") != "X" or _duration(event) <= 0:
        return False
    category = str(event.get("cat", "")).lower()
    return "kernel" in category


def _classify_kernel(name: str) -> str | None:
    lowered = name.lower()
    normalized = re.sub(r"[^a-z0-9]", "", lowered)
    aiter_fused_tokens = (
        "fusedallreducermsnorm",
        "fusedarrms",
        "allreducefusionkernel",
        "localdeviceloadrmsnorm",
        "reducescattercrossdevicestore",
    )
    if any(token in normalized for token in aiter_fused_tokens):
        return "aiter_fused_allreduce_rmsnorm"
    if "gemma" in normalized and "rmsnorm" in normalized:
        return "gemma_rmsnorm"
    if any(
        token in normalized
        for token in ("allreduce", "crossdevicereduce", "rccl", "nccl")
    ):
        return "allreduce"
    return None


def _is_m3_qk_norm(name: str) -> bool:
    return "fusedminimaxm3qnormropekvinsertkernel" in re.sub(
        r"[^a-z0-9]", "", name.lower()
    )


def _is_generic_gemm(name: str) -> bool:
    return name.startswith("Cijk_")


def _is_collective_or_norm(name: str) -> bool:
    return _classify_kernel(name) is not None


def _m3_named_phase(name: str) -> str | None:
    lowered = name.lower()
    normalized = re.sub(r"[^a-z0-9]", "", lowered)
    if _is_m3_qk_norm(name):
        return "attention_qk_norm_rope_cache"
    if "_decode_index_score_kernel" in lowered:
        return "sparse_index_score"
    if "_topk_index_partial_kernel" in lowered:
        return "sparse_index_topk_partial"
    if "_topk_index_merge_kernel" in lowered:
        return "sparse_index_topk_merge"
    if "_gqa_sparse_decode_kernel" in lowered:
        return "sparse_attention"
    if "_merge_topk_attn_out_kernel" in lowered:
        return "sparse_attention_merge"
    if "kernel_unified_attention" in lowered:
        return "dense_attention"
    if lowered == "reduce_segments":
        return "dense_attention_reduce"
    if "reshape_and_cache_kernel" in lowered:
        return "dense_kv_cache_write"
    if "topkgating" in normalized:
        return "moe_router_topk"
    if "moealignblocks" in normalized or "moealignblocksize" in normalized:
        return "moe_token_align"
    if "countandsortexperttokens" in normalized:
        return "moe_token_sort"
    if "fusedmoekernel" in normalized or "mxfp8groupedgemm" in normalized:
        return "moe_expert_gemm"
    if "actandmulkernel" in normalized:
        return "moe_expert_activation"
    if "_swiglu_oai_quant_kernel" in lowered:
        return "moe_expert_activation_quant"
    if "moesumkernel" in normalized:
        return "moe_weighted_sum"
    if "bfloat16tofloat32copykernel" in normalized:
        return "moe_router_cast"
    if "_compute_slot_mapping_kernel" in lowered:
        return "decode_metadata"
    return None


def _merged_interval_duration(
    events: list[dict[str, Any]],
    start: float | None = None,
    end: float | None = None,
) -> float:
    intervals: list[tuple[float, float]] = []
    for event in events:
        event_start = float(event.get("ts", 0))
        event_end = event_start + _duration(event)
        if start is not None:
            event_start = max(event_start, start)
        if end is not None:
            event_end = min(event_end, end)
        if event_end > event_start:
            intervals.append((event_start, event_end))
    if not intervals:
        return 0.0

    intervals.sort()
    merged_duration = 0.0
    current_start, current_end = intervals[0]
    for interval_start, interval_end in intervals[1:]:
        if interval_start <= current_end:
            current_end = max(current_end, interval_end)
            continue
        merged_duration += current_end - current_start
        current_start, current_end = interval_start, interval_end
    return merged_duration + current_end - current_start


def _kernel_stream(event: dict[str, Any]) -> tuple[str, str]:
    args = event.get("args", {})
    if not isinstance(args, dict):
        args = {}
    return (
        str(args.get("device", "<unknown>")),
        str(args.get("stream", event.get("tid", "<unknown>"))),
    )


def _cross_stream_overlap(
    events: list[dict[str, Any]],
) -> tuple[float, int]:
    points: list[tuple[float, int, tuple[str, str]]] = []
    for event in events:
        start = float(event.get("ts", 0))
        end = start + _duration(event)
        stream = _kernel_stream(event)
        points.append((start, 1, stream))
        points.append((end, -1, stream))
    if not points:
        return 0.0, 0

    # Process ends before starts at the same timestamp so back-to-back kernels
    # on different streams do not count as overlapping.
    points.sort(key=lambda item: (item[0], item[1]))
    active: dict[tuple[str, str], int] = defaultdict(int)
    active_streams = 0
    max_active_streams = 0
    overlap_us = 0.0
    previous_ts = points[0][0]
    for timestamp, delta, stream in points:
        if timestamp > previous_ts and active_streams >= 2:
            overlap_us += timestamp - previous_ts

        previous_count = active[stream]
        active[stream] += delta
        if previous_count == 0 and active[stream] > 0:
            active_streams += 1
        elif previous_count > 0 and active[stream] == 0:
            active_streams -= 1
        max_active_streams = max(max_active_streams, active_streams)
        previous_ts = timestamp

    return overlap_us, max_active_streams


def _stream_summary(
    kernels: list[dict[str, Any]],
) -> list[dict[str, float | int | str]]:
    by_stream: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in kernels:
        by_stream[_kernel_stream(event)].append(event)

    summaries: list[dict[str, float | int | str]] = []
    for (device, stream), events in by_stream.items():
        start = min(float(event["ts"]) for event in events)
        end = max(float(event["ts"]) + _duration(event) for event in events)
        summaries.append(
            {
                "device": device,
                "stream": stream,
                "count": len(events),
                "duration_us": sum(_duration(event) for event in events),
                "busy_us": _merged_interval_duration(events),
                "span_us": end - start,
            }
        )
    return sorted(
        summaries,
        key=lambda item: float(item["duration_us"]),
        reverse=True,
    )


def _is_kernel_launch(event: dict[str, Any]) -> bool:
    if event.get("ph") != "X" or _duration(event) <= 0:
        return False
    category = str(event.get("cat", "")).lower()
    if category not in {"cuda_runtime", "hip_runtime"}:
        return False
    normalized = re.sub(r"[^a-z0-9]", "", str(event.get("name", "")).lower())
    return normalized.startswith("hip") and "launchkernel" in normalized


def _kernel_launch_analysis(
    events: list[dict[str, Any]],
    kernels: list[dict[str, Any]],
) -> dict[str, Any]:
    launches_by_correlation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if not isinstance(event, dict) or not _is_kernel_launch(event):
            continue
        args = event.get("args", {})
        if not isinstance(args, dict) or "correlation" not in args:
            continue
        launches_by_correlation[str(args["correlation"])].append(event)

    correlated: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for kernel in kernels:
        args = kernel.get("args", {})
        if not isinstance(args, dict) or "correlation" not in args:
            continue
        candidates = launches_by_correlation.get(str(args["correlation"]), [])
        if not candidates:
            continue
        kernel_start = float(kernel.get("ts", 0))
        launch = min(
            candidates,
            key=lambda candidate: (
                float(candidate.get("ts", 0)) > kernel_start,
                abs(float(candidate.get("ts", 0)) - kernel_start),
            ),
        )
        correlated.append((kernel, launch))

    if not correlated:
        return {
            "correlated_kernel_count": 0,
            "correlated_kernel_percent": 0.0,
        }

    launch_durations = [_duration(launch) for _, launch in correlated]
    queue_delays = [
        max(
            0.0,
            float(kernel.get("ts", 0))
            - (float(launch.get("ts", 0)) + _duration(launch)),
        )
        for kernel, launch in correlated
    ]
    launch_starts = [float(launch.get("ts", 0)) for _, launch in correlated]
    launch_ends = [
        float(launch.get("ts", 0)) + _duration(launch)
        for _, launch in correlated
    ]

    previous_end_by_stream: dict[tuple[str, str], float] = {}
    same_stream_gap_us = 0.0
    definite_host_starvation_us = 0.0
    launch_call_in_gap_us = 0.0
    post_launch_device_gap_us = 0.0
    host_late_launch_count = 0
    gap_count = 0
    for kernel, launch in sorted(
        correlated,
        key=lambda item: float(item[0].get("ts", 0)),
    ):
        kernel_start = float(kernel.get("ts", 0))
        kernel_end = kernel_start + _duration(kernel)
        stream = _kernel_stream(kernel)
        previous_end = previous_end_by_stream.get(stream)
        previous_end_by_stream[stream] = max(
            kernel_end,
            previous_end if previous_end is not None else kernel_end,
        )
        if previous_end is None or kernel_start <= previous_end:
            continue

        gap_count += 1
        gap = kernel_start - previous_end
        same_stream_gap_us += gap
        launch_start = float(launch.get("ts", 0))
        launch_end = launch_start + _duration(launch)
        if launch_start > previous_end:
            host_late_launch_count += 1
        definite_host_starvation_us += max(
            0.0,
            min(kernel_start, launch_start) - previous_end,
        )
        launch_call_in_gap_us += max(
            0.0,
            min(kernel_start, launch_end) - max(previous_end, launch_start),
        )
        post_launch_device_gap_us += max(
            0.0,
            kernel_start - max(previous_end, launch_end),
        )

    return {
        "correlated_kernel_count": len(correlated),
        "correlated_kernel_percent": 100.0 * len(correlated) / len(kernels),
        "launch_submission_span_us": max(launch_ends) - min(launch_starts),
        "launch_api_duration_us": {
            "total": sum(launch_durations),
            "p50": _percentile(launch_durations, 0.50),
            "p95": _percentile(launch_durations, 0.95),
            "max": max(launch_durations),
        },
        "launch_to_kernel_start_us": {
            "p50": _percentile(queue_delays, 0.50),
            "p95": _percentile(queue_delays, 0.95),
            "max": max(queue_delays),
        },
        "same_stream_gap_count": gap_count,
        "same_stream_gap_us": same_stream_gap_us,
        "host_late_launch_count": host_late_launch_count,
        "definite_host_starvation_us": definite_host_starvation_us,
        "launch_call_in_gap_us": launch_call_in_gap_us,
        "post_launch_device_gap_us": post_launch_device_gap_us,
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _summarize_events_by_name(
    events: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, float | int | str]]:
    by_name: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"duration_us": 0.0, "count": 0}
    )
    for event in events:
        name = str(event.get("name", "<unnamed>"))
        by_name[name]["duration_us"] = float(by_name[name]["duration_us"]) + _duration(
            event
        )
        by_name[name]["count"] = int(by_name[name]["count"]) + 1
    return sorted(
        ({"name": name, **values} for name, values in by_name.items()),
        key=lambda item: float(item["duration_us"]),
        reverse=True,
    )[:limit]


def _phase_summary(
    kernels: list[dict[str, Any]],
    phases: dict[int, str],
    total_kernel_us: float,
) -> tuple[
    dict[str, dict[str, Any]],
    list[dict[str, float | str]],
    list[dict[str, float | int | str]],
]:
    by_phase: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"duration_us": 0.0, "count": 0, "preceding_gap_us": 0.0}
    )
    phase_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    largest_gaps: list[dict[str, float | str]] = []
    gap_transitions: dict[tuple[str, str], dict[str, float | int]] = defaultdict(
        lambda: {"duration_us": 0.0, "count": 0, "max_us": 0.0}
    )
    cursor = min(float(event["ts"]) for event in kernels)
    cursor_index: int | None = None
    for index, event in enumerate(kernels):
        phase = phases[index]
        event_start = float(event["ts"])
        gap = max(0.0, event_start - cursor)
        values = by_phase[phase]
        values["duration_us"] = float(values["duration_us"]) + _duration(event)
        values["count"] = int(values["count"]) + 1
        values["preceding_gap_us"] = float(values["preceding_gap_us"]) + gap
        phase_events[phase].append(event)
        if gap > 0 and cursor_index is not None:
            previous_event = kernels[cursor_index]
            previous_phase = phases[cursor_index]
            previous_device, previous_stream = _kernel_stream(previous_event)
            next_device, next_stream = _kernel_stream(event)
            largest_gaps.append(
                {
                    "duration_us": gap,
                    "previous_phase": previous_phase,
                    "previous_kernel": str(
                        previous_event.get("name", "<unnamed>")
                    ),
                    "previous_device": previous_device,
                    "previous_stream": previous_stream,
                    "next_phase": phase,
                    "next_kernel": str(event.get("name", "<unnamed>")),
                    "next_device": next_device,
                    "next_stream": next_stream,
                }
            )
            transition = gap_transitions[(previous_phase, phase)]
            transition["duration_us"] = float(transition["duration_us"]) + gap
            transition["count"] = int(transition["count"]) + 1
            transition["max_us"] = max(float(transition["max_us"]), gap)

        event_end = event_start + _duration(event)
        if event_end > cursor:
            cursor = event_end
            cursor_index = index

    summary: dict[str, dict[str, Any]] = {}
    for phase, values in sorted(
        by_phase.items(),
        key=lambda item: float(item[1]["duration_us"]),
        reverse=True,
    ):
        duration_us = float(values["duration_us"])
        summary[phase] = {
            **values,
            "percent_of_kernel_time": (
                100.0 * duration_us / total_kernel_us if total_kernel_us else 0.0
            ),
            "top_kernels": _summarize_events_by_name(phase_events[phase], 5),
        }
    transitions = sorted(
        (
            {
                "previous_phase": previous_phase,
                "next_phase": next_phase,
                **values,
                "mean_us": (
                    float(values["duration_us"]) / int(values["count"])
                    if values["count"]
                    else 0.0
                ),
            }
            for (previous_phase, next_phase), values in gap_transitions.items()
        ),
        key=lambda item: float(item["duration_us"]),
        reverse=True,
    )
    return (
        summary,
        sorted(
            largest_gaps,
            key=lambda item: float(item["duration_us"]),
            reverse=True,
        )[:20],
        transitions,
    )


def _analyze_m3(
    kernels: list[dict[str, Any]],
    total_kernel_us: float,
) -> dict[str, Any] | None:
    qk_norm_indices = [
        index
        for index, event in enumerate(kernels)
        if _is_m3_qk_norm(str(event.get("name", "")))
    ]
    if not qk_norm_indices:
        return None

    layer_starts: list[int] = []
    previous_start = -1
    for qk_norm_index in qk_norm_indices:
        layer_start = next(
            (
                index
                for index in range(qk_norm_index - 1, previous_start, -1)
                if _is_generic_gemm(str(kernels[index].get("name", "")))
            ),
            -1,
        )
        if layer_start < 0:
            return {
                "recognized": False,
                "reason": "could not find qkv projection before QK norm",
                "qk_norm_count": len(qk_norm_indices),
            }
        layer_starts.append(layer_start)
        previous_start = layer_start

    phases: dict[int, str] = {}
    layers: list[dict[str, Any]] = []
    first_layer_start = layer_starts[0]
    for index in range(first_layer_start):
        name = str(kernels[index].get("name", ""))
        if _is_collective_or_norm(name):
            phases[index] = "model_input_collective_norm"
        else:
            phases[index] = _m3_named_phase(name) or "decode_setup"

    tail_start = len(kernels)
    dense_layer_count = 0
    sparse_layer_count = 0
    for layer_id, (layer_start, qk_norm_index) in enumerate(
        zip(layer_starts, qk_norm_indices, strict=True)
    ):
        segment_end = (
            layer_starts[layer_id + 1]
            if layer_id + 1 < len(layer_starts)
            else len(kernels)
        )
        segment_indices = range(layer_start, segment_end)
        sparse = any(
            _m3_named_phase(str(kernels[index].get("name", ""))) == "sparse_index_score"
            for index in segment_indices
        )
        layer_kind = "sparse_moe" if sparse else "dense"
        if sparse:
            sparse_layer_count += 1
        else:
            dense_layer_count += 1

        gemm_indices = [
            index
            for index in segment_indices
            if _is_generic_gemm(str(kernels[index].get("name", "")))
        ]
        expected_gemms = 5 if sparse else 4
        if len(gemm_indices) < expected_gemms:
            return {
                "recognized": False,
                "reason": (
                    f"layer {layer_id} has {len(gemm_indices)} GEMMs, "
                    f"expected at least {expected_gemms}"
                ),
                "qk_norm_count": len(qk_norm_indices),
            }
        if layer_id == len(layer_starts) - 1 and len(gemm_indices) > expected_gemms:
            tail_start = gemm_indices[expected_gemms]
            segment_end = tail_start
            gemm_indices = gemm_indices[:expected_gemms]

        gemm_roles = (
            [
                "attention_qkv_projection",
                "attention_output_projection",
                "moe_router_projection",
                "moe_shared_gate_up_projection",
                "moe_shared_down_projection",
            ]
            if sparse
            else [
                "attention_qkv_projection",
                "attention_output_projection",
                "dense_ffn_gate_up_projection",
                "dense_ffn_down_projection",
            ]
        )
        phases.update(dict(zip(gemm_indices, gemm_roles, strict=True)))
        ffn_start = gemm_indices[2]
        expert_gemm_indices = [
            index
            for index in range(ffn_start, segment_end)
            if _m3_named_phase(str(kernels[index].get("name", ""))) == "moe_expert_gemm"
        ]
        first_expert_gemm_index = (
            expert_gemm_indices[0] if expert_gemm_indices else segment_end
        )
        token_sort_indices = [
            index
            for index in range(ffn_start, first_expert_gemm_index)
            if _m3_named_phase(str(kernels[index].get("name", ""))) == "moe_token_sort"
        ]
        token_sort_index = token_sort_indices[-1] if token_sort_indices else -1
        final_collective_indices = [
            index
            for index in range(ffn_start, segment_end)
            if _is_collective_or_norm(str(kernels[index].get("name", "")))
        ]
        final_collective_index = (
            final_collective_indices[-1] if final_collective_indices else -1
        )
        expert_gemm_count = 0
        weighted_sum_index: int | None = None

        for index in range(layer_start, segment_end):
            if index in phases:
                continue
            name = str(kernels[index].get("name", ""))
            named_phase = _m3_named_phase(name)
            if named_phase == "moe_expert_gemm":
                expert_gemm_count += 1
                phases[index] = f"moe_expert_gemm_{expert_gemm_count}"
            elif named_phase == "moe_weighted_sum":
                weighted_sum_index = index
                phases[index] = named_phase
            elif named_phase is not None:
                phases[index] = named_phase
            elif _is_collective_or_norm(name):
                phases[index] = (
                    "attention_collective_norm"
                    if index < ffn_start
                    else "ffn_collective_norm"
                )
            elif "_swiglu_oai_kernel" in name.lower():
                phases[index] = (
                    "moe_shared_activation" if sparse else "dense_ffn_activation"
                )
            elif sparse and token_sort_index < index < first_expert_gemm_index:
                phases[index] = "moe_expert_input_prepare"
            elif (
                layer_id == len(layer_starts) - 1
                and final_collective_index >= 0
                and index > final_collective_index
            ):
                phases[index] = "model_output_prepare"
            elif (
                sparse and weighted_sum_index is not None and index > weighted_sum_index
            ):
                phases[index] = "moe_shared_combine"
            elif index < qk_norm_index:
                phases[index] = "attention_qkv_aux"
            elif index < gemm_indices[1]:
                phases[index] = "attention_other"
            elif index < ffn_start:
                phases[index] = "attention_output_other"
            else:
                phases[index] = "moe_other" if sparse else "dense_ffn_other"

        layer_events = kernels[layer_start:segment_end]
        layer_start_us = min(float(event["ts"]) for event in layer_events)
        layer_end_us = max(
            float(event["ts"]) + _duration(event) for event in layer_events
        )
        layer_kernel_us = sum(_duration(event) for event in layer_events)
        layer_busy_us = _merged_interval_duration(layer_events)
        layer_phase_totals: dict[str, float] = defaultdict(float)
        for index in range(layer_start, segment_end):
            layer_phase_totals[phases[index]] += _duration(kernels[index])
        layers.append(
            {
                "layer": layer_id,
                "kind": layer_kind,
                "span_us": layer_end_us - layer_start_us,
                "kernel_duration_us": layer_kernel_us,
                "kernel_busy_us": layer_busy_us,
                "idle_us": layer_end_us - layer_start_us - layer_busy_us,
                "phases": dict(
                    sorted(
                        layer_phase_totals.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                ),
            }
        )

    for index in range(tail_start, len(kernels)):
        name = str(kernels[index].get("name", ""))
        if _is_generic_gemm(name):
            phases[index] = "logits_projection"
        elif "nccl" in name.lower() or "allgather" in name.lower():
            phases[index] = "logits_all_gather"
        else:
            phases[index] = _m3_named_phase(name) or "sampling_and_output"

    for index in range(len(kernels)):
        phases.setdefault(index, "other")

    phase_summary, largest_gaps, gap_transitions = _phase_summary(
        kernels, phases, total_kernel_us
    )
    layer_spans = [float(layer["span_us"]) for layer in layers]
    dense_spans = [
        float(layer["span_us"]) for layer in layers if layer["kind"] == "dense"
    ]
    sparse_spans = [
        float(layer["span_us"]) for layer in layers if layer["kind"] == "sparse_moe"
    ]
    return {
        "recognized": True,
        "layer_count": len(layers),
        "dense_layer_count": dense_layer_count,
        "sparse_moe_layer_count": sparse_layer_count,
        "qk_norm_rope_cache_count": len(qk_norm_indices),
        "phases": phase_summary,
        "largest_kernel_gaps": largest_gaps,
        "kernel_gap_transitions": gap_transitions,
        "layer_span_us": {
            "min": min(layer_spans),
            "p50": _percentile(layer_spans, 0.50),
            "p95": _percentile(layer_spans, 0.95),
            "max": max(layer_spans),
            "dense_p50": _percentile(dense_spans, 0.50),
            "sparse_moe_p50": _percentile(sparse_spans, 0.50),
        },
        "layers": layers,
    }


def analyze_trace(
    path: Path,
    expected_concurrency: int | None = None,
    minimum_concurrency_fraction: float = 1.0,
) -> dict[str, Any]:
    if not 0 < minimum_concurrency_fraction <= 1:
        raise ValueError("minimum concurrency fraction must be in (0, 1]")

    with _open_trace(path) as trace_file:
        payload = json.load(trace_file)

    events = payload.get("traceEvents") if isinstance(payload, dict) else None
    if not isinstance(events, list):
        raise ValueError("traceEvents is missing or is not a list")

    annotations: list[tuple[dict[str, Any], re.Match[str]]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        match = ANNOTATION_RE.fullmatch(str(event.get("name", "")))
        if match:
            annotations.append((event, match))

    if not annotations:
        raise ValueError("no execute_context/generation annotation found")

    matching_annotations: list[tuple[dict[str, Any], re.Match[str]]] = []
    minimum_generation_requests = (
        math.ceil(expected_concurrency * minimum_concurrency_fraction)
        if expected_concurrency is not None
        else None
    )
    for annotation in annotations:
        values = {key: int(value) for key, value in annotation[1].groupdict().items()}
        if values["context_requests"] != 0 or values["context_tokens"] != 0:
            continue
        if expected_concurrency is not None and not (
            minimum_generation_requests
            <= values["generation_requests"]
            <= expected_concurrency
        ):
            continue
        matching_annotations.append(annotation)

    if not matching_annotations:
        observed = [item[0].get("name") for item in annotations]
        raise ValueError(
            "no steady-state decode annotation matched the expected concurrency; "
            f"observed={observed}"
        )

    # Torch emits both CPU and GPU copies of the same annotation. Kernel
    # timestamps align with gpu_user_annotation; the CPU window can end while
    # queued GPU work is still running.
    selected = max(
        matching_annotations,
        key=lambda item: (
            "gpu_user_annotation" in str(item[0].get("cat", "")).lower(),
            int(item[1].group("generation_requests")),
            _duration(item[0]),
        ),
    )
    annotation_event, annotation_match = selected
    annotation_values = {
        key: int(value) for key, value in annotation_match.groupdict().items()
    }
    start = float(annotation_event.get("ts", 0))
    end = start + _duration(annotation_event)

    all_kernels = [
        event for event in events if isinstance(event, dict) and _is_kernel(event)
    ]
    kernels = [
        event for event in all_kernels if start <= float(event.get("ts", -1)) < end
    ]
    used_annotation_window = True
    if not kernels:
        kernels = all_kernels
        used_annotation_window = False
    if not kernels:
        raise ValueError("no GPU kernel events found")

    by_name: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"duration_us": 0.0, "count": 0}
    )
    by_category: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"duration_us": 0.0, "count": 0}
    )
    kernel_start = min(float(event["ts"]) for event in kernels)
    kernel_end = max(float(event["ts"]) + _duration(event) for event in kernels)

    for event in kernels:
        name = str(event.get("name", "<unnamed>"))
        duration = _duration(event)
        by_name[name]["duration_us"] = float(by_name[name]["duration_us"]) + duration
        by_name[name]["count"] = int(by_name[name]["count"]) + 1
        category = _classify_kernel(name)
        if category is not None:
            by_category[category]["duration_us"] = (
                float(by_category[category]["duration_us"]) + duration
            )
            by_category[category]["count"] = int(by_category[category]["count"]) + 1

    total_kernel_us = sum(_duration(event) for event in kernels)
    kernel_busy_us = _merged_interval_duration(kernels)
    kernel_streams = _stream_summary(kernels)
    cross_stream_overlap_us, max_concurrent_kernel_streams = _cross_stream_overlap(
        kernels
    )
    kernel_launch_analysis = _kernel_launch_analysis(events, kernels)
    categories = {}
    for name, values in sorted(by_category.items()):
        duration_us = float(values["duration_us"])
        categories[name] = {
            **values,
            "percent_of_kernel_time": (
                100.0 * duration_us / total_kernel_us if total_kernel_us else 0.0
            ),
        }

    top_kernels = sorted(
        (
            {
                "name": name,
                **values,
                "percent_of_kernel_time": (
                    100.0 * float(values["duration_us"]) / total_kernel_us
                    if total_kernel_us
                    else 0.0
                ),
            }
            for name, values in by_name.items()
        ),
        key=lambda item: float(item["duration_us"]),
        reverse=True,
    )[:20]

    cpu_annotations = [
        annotation
        for annotation in matching_annotations
        if "gpu_user_annotation" not in str(annotation[0].get("cat", "")).lower()
    ]
    top_cpu_ops: list[dict[str, float | int | str]] = []
    top_runtime_calls: list[dict[str, float | int | str]] = []
    if cpu_annotations:
        cpu_annotation = max(
            cpu_annotations,
            key=lambda item: (
                int(item[1].group("generation_requests")),
                _duration(item[0]),
            ),
        )[0]
        cpu_start = float(cpu_annotation.get("ts", 0))
        cpu_end = cpu_start + _duration(cpu_annotation)
        cpu_ops = [
            event
            for event in events
            if isinstance(event, dict)
            and event.get("ph") == "X"
            and _duration(event) > 0
            and event.get("cat") == "cpu_op"
            and cpu_start <= float(event.get("ts", -1)) < cpu_end
        ]
        runtime_calls = [
            event
            for event in events
            if isinstance(event, dict)
            and event.get("ph") == "X"
            and _duration(event) > 0
            and str(event.get("cat", "")).lower() in {"cuda_runtime", "hip_runtime"}
            and cpu_start <= float(event.get("ts", -1)) < cpu_end
        ]
        top_cpu_ops = _summarize_events_by_name(cpu_ops, 20)
        top_runtime_calls = _summarize_events_by_name(runtime_calls, 20)

    annotation_duration_us = _duration(annotation_event)
    annotation_kernel_busy_us = _merged_interval_duration(
        kernels,
        start=start,
        end=end,
    )
    return {
        "trace": str(path),
        "annotation": annotation_event["name"],
        "annotation_category": annotation_event.get("cat", ""),
        "expected_concurrency": expected_concurrency,
        "minimum_concurrency_fraction": minimum_concurrency_fraction,
        **annotation_values,
        "annotation_window_used": used_annotation_window,
        "annotation_duration_us": annotation_duration_us,
        "annotation_kernel_busy_us": annotation_kernel_busy_us,
        "annotation_gpu_idle_us": max(
            0.0, annotation_duration_us - annotation_kernel_busy_us
        ),
        "decode_kernel_span_us": kernel_end - kernel_start,
        "total_kernel_duration_us": total_kernel_us,
        "kernel_busy_time_us": kernel_busy_us,
        "kernel_idle_time_us": max(0.0, kernel_end - kernel_start - kernel_busy_us),
        "kernel_stream_count": len(kernel_streams),
        "kernel_streams": kernel_streams,
        "cross_stream_overlap_us": cross_stream_overlap_us,
        "max_concurrent_kernel_streams": max_concurrent_kernel_streams,
        "kernel_launch_analysis": kernel_launch_analysis,
        "kernel_busy_percent": (
            100.0 * kernel_busy_us / (kernel_end - kernel_start)
            if kernel_end > kernel_start
            else 0.0
        ),
        "kernel_count": len(kernels),
        "categories": categories,
        "top_kernels": top_kernels,
        "top_cpu_ops": top_cpu_ops,
        "top_runtime_calls": top_runtime_calls,
        "minimax_m3": _analyze_m3(kernels, total_kernel_us),
    }


def validate_categories(
    summary: dict[str, Any],
    required: list[str],
    forbidden: list[str],
) -> None:
    """Validate experiment kernels without accepting out-of-window events."""
    if (required or forbidden) and not summary["annotation_window_used"]:
        raise ValueError(
            "kernel category validation requires events inside the decode annotation"
        )

    observed_categories = set(summary["categories"])
    missing = set(required) - observed_categories
    present_forbidden = set(forbidden) & observed_categories
    if missing:
        raise ValueError(f"required kernel categories not found: {sorted(missing)}")
    if present_forbidden:
        raise ValueError(
            f"forbidden kernel categories found: {sorted(present_forbidden)}"
        )


def validate_kernel_count(summary: dict[str, Any], minimum: int | None) -> None:
    """Reject partial traces that do not cover the expected model execution."""
    if minimum is not None and summary["kernel_count"] < minimum:
        raise ValueError(
            f"kernel count {summary['kernel_count']} is below required minimum {minimum}"
        )


def validate_kernel_stream_count(
    summary: dict[str, Any],
    minimum: int | None,
) -> None:
    """Require an experiment to expose the expected number of GPU streams."""
    if minimum is not None and summary["kernel_stream_count"] < minimum:
        raise ValueError(
            f"kernel stream count {summary['kernel_stream_count']} "
            f"is below required minimum {minimum}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--expected-concurrency", type=int)
    parser.add_argument(
        "--minimum-concurrency-fraction",
        type=float,
        default=1.0,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-kernel-count", type=int)
    parser.add_argument("--min-kernel-stream-count", type=int)
    parser.add_argument("--require-category", action="append", default=[])
    parser.add_argument("--forbid-category", action="append", default=[])
    args = parser.parse_args()

    summary = analyze_trace(
        args.trace,
        args.expected_concurrency,
        args.minimum_concurrency_fraction,
    )
    try:
        validate_kernel_count(summary, args.min_kernel_count)
        validate_kernel_stream_count(summary, args.min_kernel_stream_count)
        validate_categories(
            summary,
            required=args.require_category,
            forbidden=args.forbid_category,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    output = json.dumps(summary, indent=2)
    print(output)
    if args.output is not None:
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
