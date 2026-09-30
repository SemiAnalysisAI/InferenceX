"""Summarize native serving traces without adding overlapping durations."""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path


def interval_union(intervals: list[tuple[float, float]]) -> float:
    end = None
    total = 0.0
    for start, stop in sorted(intervals):
        if stop < start:
            raise ValueError("Negative interval")
        total += max(0.0, stop - (start if end is None else max(start, end)))
        end = stop if end is None else max(end, stop)
    return total


def family(name: str) -> str:
    n = name.lower()
    if any(
        s in n
        for s in (
            "allreduce",
            "all_reduce",
            "allgather",
            "reduce_scatter",
            "nccl",
            "mnnvl",
        )
    ):
        return "communication"
    if "engram" in n:
        return "engram"
    if "_hc_" in n or "mhc" in n:
        return "hyper_connections"
    if any(s in n for s in ("mqa_logits", "index_q_", "index_k_", "fp4_index")):
        return "indexer"
    if any(s in n for s in ("flash_fwd", "sparse_attn", "mla_")):
        return "attention"
    if "mxe2m1" in n or "moe" in n or "mixedinput" in n:
        return "routed_experts"
    if "topk" in n or "top_k" in n or "router" in n or "radixselect" in n:
        return "selection_and_routing"
    if any(s in n for s in ("gemm", "bmm", "matmul", "nvjet")):
        return "dense_gemm"
    if "compress" in n or "cache" in n:
        return "cache_and_compression"
    if "quant" in n or "rope" in n or "norm" in n:
        return "quantization_rope_norm"
    return "other"


def describe(work: list[dict]) -> dict:
    if not work:
        raise ValueError("No device events in timing scope")
    start = min(float(e["ts"]) for e in work)
    end = max(float(e["ts"]) + e["dur"] for e in work)
    kernels = [e for e in work if e.get("cat") == "kernel"]
    groups = defaultdict(float)
    names = defaultdict(lambda: {"calls": 0, "sum_us": 0.0})
    for e in kernels:
        groups[family(e["name"])] += e["dur"]
        names[e["name"]]["calls"] += 1
        names[e["name"]]["sum_us"] += e["dur"]
    active = interval_union([(float(e["ts"]), float(e["ts"]) + e["dur"]) for e in work])
    return {
        "device_span_us": end - start,
        "device_active_union_us": active,
        "uncovered_device_span_us": end - start - active,
        "kernel_sum_us": sum(e["dur"] for e in kernels),
        "kernel_calls": len(kernels),
        "kernel_family_sums_us": dict(groups),
        "kernels": dict(sorted(names.items(), key=lambda item: -item[1]["sum_us"])),
    }


def summarize(trace: dict) -> dict:
    events = trace["traceEvents"]
    work = [
        e
        for e in events
        if e.get("ph") == "X" and e.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}
    ]
    by_correlation = defaultdict(list)
    for e in work:
        by_correlation[e.get("args", {}).get("correlation")].append(e)
    launches = [
        e
        for e in events
        if e.get("cat") in {"cuda_runtime", "cuda_driver"}
        and "Launch" in e.get("name", "")
    ]
    scopes = [
        e
        for e in events
        if e.get("cat") == "user_annotation"
        and e.get("name", "").startswith(("step[", "execute_"))
    ]
    rows = []
    for scope in scopes:
        calls = [
            e
            for e in launches
            if e["pid"] == scope["pid"]
            and e["tid"] == scope["tid"]
            and scope["ts"] <= e["ts"] < scope["ts"] + scope["dur"]
        ]
        correlations = {e.get("args", {}).get("correlation") for e in calls} - {None}
        selected = [e for c in correlations for e in by_correlation[c]]
        if selected:
            rows.append(
                {
                    "scope": scope["name"],
                    "cpu_scope_us": scope["dur"],
                    "graph_launches": sum(
                        e["name"] == "cudaGraphLaunch" for e in calls
                    ),
                    **describe(selected),
                }
            )
    stages = {}
    for stage in ("EXTEND", "DRAFT", "VERIFY", "VLLM_EXECUTE"):
        prefix = "execute_" if stage == "VLLM_EXECUTE" else "step[" + stage
        chosen = [r for r in rows if r["scope"].startswith(prefix)]
        if not chosen:
            continue
        stages[stage] = {"samples": len(chosen)}
        for field in (
            "device_span_us",
            "device_active_union_us",
            "kernel_sum_us",
            "cpu_scope_us",
        ):
            values = [r[field] for r in chosen]
            stages[stage][field] = {
                "median": statistics.median(values),
                "mean": statistics.mean(values),
                "min": min(values),
                "max": max(values),
            }
    return {
        "whole_capture": describe(work),
        "stages": stages,
        "iterations": rows,
        "qualification": "Profiled device spans and kernel sums are distinct. Kernel sums overlap; CPU scope is asynchronous submission time, not completed model execution. Family classification is name-based; raw kernel names are retained.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = {}
    for path in args.traces:
        data = (
            gzip.decompress(path.read_bytes())
            if path.suffix == ".gz"
            else path.read_bytes()
        )
        result[path.name] = summarize(json.loads(data))
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
