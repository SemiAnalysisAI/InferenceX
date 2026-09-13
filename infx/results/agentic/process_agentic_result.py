#!/usr/bin/env python3
"""Process aiperf agentic-replay output into InferenceX aggregate JSON."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from infx.results.agentic import build_result
from infx.results.agentic.common import round_floats

from .artifacts import (
    find_server_log_paths,
    iter_trace_blobs,
    load_aggregate,
    load_records_with_accounting,
    load_server_log_head,
    load_server_metrics,
    resolve_artifact_dir,
)


def main() -> int:
    result_filename = os.environ.get("RESULT_FILENAME", "")
    if not result_filename:
        print("ERROR: RESULT_FILENAME env var not set", file=sys.stderr)
        return 1

    result_dir = Path(os.environ.get("RESULT_DIR", "results"))
    output_dir = Path(os.environ.get("AGENTIC_OUTPUT_DIR", "."))

    artifact_dir = resolve_artifact_dir(result_dir)
    aggregate_path = artifact_dir / "profile_export_aiperf.json"
    jsonl_path = artifact_dir / "profile_export.jsonl"
    server_metrics_path = artifact_dir / "server_metrics_export.json"

    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} not found", file=sys.stderr)
        return 1

    records, request_accounting = load_records_with_accounting(jsonl_path)
    aggregate = load_aggregate(aggregate_path) if aggregate_path.exists() else {}
    server_metrics = load_server_metrics(server_metrics_path)
    server_log_paths = find_server_log_paths(result_dir)
    agg = round_floats(
        build_result(
            records,
            aggregate,
            server_metrics,
            os.environ,
            request_accounting=request_accounting,
            traces=iter_trace_blobs(aggregate, os.environ),
            server_logs=(load_server_log_head(path) for path in server_log_paths),
        )
    )

    output_path = output_dir / f"{result_filename}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(agg, f, indent=2)

    print(f"Saved aggregated agentic result to {output_path}")
    print(
        f"  Requests: {len(records)} successful / "
        f"{request_accounting['records_total']} total "
        f"({request_accounting['records_warmup_dropped']} warmup, "
        f"{request_accounting['records_error_dropped']} error dropped)"
    )
    request_metrics = agg.get("request_metrics", {})
    qps_metrics = request_metrics.get("qps", {})
    if "mean" in qps_metrics:
        print(
            f"  QPS: mean={qps_metrics['mean']:.2f} "
            f"p75={qps_metrics.get('p75', 0):.2f} "
            f"p95={qps_metrics.get('p95', 0):.2f}"
        )
    server_metrics = agg.get("server_metrics", {})
    server_cache = server_metrics.get("cache", {})
    server_kv_cache = server_metrics.get("kv_cache", {})
    if server_cache.get("gpu_cache_hit_rate") is not None:
        print(f"  GPU cache hit rate: {server_cache['gpu_cache_hit_rate']:.1%}")
    if server_cache.get("cpu_cache_hit_rate") is not None:
        print(f"  CPU/offload cache hit rate: {server_cache['cpu_cache_hit_rate']:.1%}")
    if server_cache.get("external_cache_hit_rate") is not None:
        print(f"  External cache hit rate: {server_cache['external_cache_hit_rate']:.1%}")
    if server_kv_cache.get("gpu_usage_pct") is not None:
        print(f"  GPU KV cache usage:  {server_kv_cache['gpu_usage_pct']:.1%}")
    if server_kv_cache.get("gpu_total_tokens") is not None:
        print(f"  GPU KV cache capacity: {server_kv_cache['gpu_total_tokens']} tokens")
    request_cache = request_metrics.get("cache", {})
    if request_cache.get("theoretical_cache_hit_rate") is not None:
        print(f"  Theoretical cache hit rate: {request_cache['theoretical_cache_hit_rate']:.1%}")
    throughput_per_gpu = request_metrics.get("throughput", {}).get("per_gpu", {})
    if throughput_per_gpu.get("total_tput_tps") is not None:
        print(f"  Throughput per GPU: {throughput_per_gpu['total_tput_tps']:.0f} tok/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
