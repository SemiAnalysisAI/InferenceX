#!/usr/bin/env python3
"""Process agentic trace replay benchmark results into an aggregated JSON file.

Reads detailed_results.csv and metrics_server_metrics.csv from the benchmark
output directory and produces an agg_*.json file matching the naming convention
of fixed-seq-len results.

Expected env vars:
    RESULT_FILENAME - base name for output file (e.g., dsr1_tp4_users8_offloadon_...)
    MODEL, MODEL_PREFIX, FRAMEWORK, PRECISION, TP, EP_SIZE, DP_ATTENTION
    USERS, OFFLOAD_MODE, RUNNER_TYPE
"""

import csv
import json
import os
import sys
import statistics

csv.field_size_limit(sys.maxsize)
from pathlib import Path


def percentile(data, p):
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def load_detailed_results(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load_server_metrics(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def compute_qps_stats(rows):
    """Compute QPS from request completion timestamps using 1-second sliding windows."""
    if len(rows) < 2:
        return {}

    complete_times = sorted(float(r['request_complete_time']) for r in rows if r.get('success') == 'True')
    if len(complete_times) < 2:
        return {}

    start = complete_times[0]
    end = complete_times[-1]
    duration = end - start
    if duration <= 0:
        return {}

    window = 1.0
    qps_values = []
    t = start
    while t + window <= end:
        count = sum(1 for ct in complete_times if t <= ct < t + window)
        qps_values.append(count / window)
        t += window

    if not qps_values:
        overall_qps = len(complete_times) / duration
        return {"mean_qps": overall_qps}

    return {
        "mean_qps": statistics.mean(qps_values),
        "median_qps": statistics.median(qps_values),
        "p90_qps": percentile(qps_values, 90),
        "p99_qps": percentile(qps_values, 99),
        "p99.9_qps": percentile(qps_values, 99.9),
    }


def compute_latency_stats(rows):
    ttfts = [float(r['ttft']) for r in rows if r.get('success') == 'True' and float(r['ttft']) > 0]
    ttlts = [float(r['ttlt']) for r in rows if r.get('success') == 'True' and float(r['ttlt']) > 0]
    itls = [float(r['itl']) for r in rows if r.get('success') == 'True' and float(r['itl']) > 0]

    result = {}
    for name, values in [("ttft", ttfts), ("ttlt", ttlts), ("itl", itls)]:
        if values:
            result[f"mean_{name}"] = statistics.mean(values)
            result[f"median_{name}"] = statistics.median(values)
            result[f"p90_{name}"] = percentile(values, 90)
            result[f"p99_{name}"] = percentile(values, 99)
            result[f"p99.9_{name}"] = percentile(values, 99.9)
    return result


def compute_workload_stats(rows):
    input_tokens = [int(r['input_tokens']) for r in rows if r.get('success') == 'True']
    output_expected = [int(r['output_tokens_expected']) for r in rows if r.get('success') == 'True']
    output_actual = [int(r['output_tokens_actual']) for r in rows if r.get('success') == 'True']

    result = {}
    for name, values in [("input_tokens", input_tokens), ("output_tokens_expected", output_expected), ("output_tokens_actual", output_actual)]:
        if values:
            result[f"mean_{name}"] = statistics.mean(values)
            result[f"median_{name}"] = statistics.median(values)
            result[f"p90_{name}"] = percentile(values, 90)
            result[f"p99_{name}"] = percentile(values, 99)
    return result


def compute_cache_stats(rows, server_metrics):
    """Compute cache hit rates from both detailed results and server metrics."""
    result = {
        "theoretical_cache_hit_rate": None,
        "theoretical_infinite_cache_hit_rate": None,
        "server_gpu_cache_hit_rate": None,
        "server_cpu_cache_hit_rate": None,
        "kv_offload_bytes_gpu_to_cpu": None,
        "kv_offload_bytes_cpu_to_gpu": None,
        "kv_offload_time_gpu_to_cpu": None,
        "kv_offload_time_cpu_to_gpu": None,
        "cpu_kv_cache_usage_pct": None,
        "total_prompt_tokens": None,
        "total_generation_tokens": None,
        "total_requests_completed": None,
    }

    # From detailed results: theoretical cache hit rate (infinite cache)
    total_hit_blocks = sum(int(r.get('cache_hit_blocks', 0)) for r in rows)
    total_miss_blocks = sum(int(r.get('cache_miss_blocks', 0)) for r in rows)
    total_blocks = total_hit_blocks + total_miss_blocks
    if total_blocks > 0:
        result["theoretical_cache_hit_rate"] = total_hit_blocks / total_blocks

    theoretical_hits = sum(int(r.get('theoretical_cache_hit_blocks', 0)) for r in rows)
    if total_blocks > 0:
        result["theoretical_infinite_cache_hit_rate"] = theoretical_hits / total_blocks

    # From server metrics: actual prefix cache hit rate (last row)
    if server_metrics:
        last = server_metrics[-1]
        hits = int(last.get('prefix_cache_hits', 0))
        queries = int(last.get('prefix_cache_queries', 0))
        if queries > 0:
            result["server_gpu_cache_hit_rate"] = hits / queries

        cpu_hits = int(last.get('cpu_prefix_cache_hits', 0))
        cpu_queries = int(last.get('cpu_prefix_cache_queries', 0))
        if cpu_queries > 0:
            result["server_cpu_cache_hit_rate"] = cpu_hits / cpu_queries

        offload_g2c = float(last.get('kv_offload_bytes_gpu_to_cpu', 0))
        offload_c2g = float(last.get('kv_offload_bytes_cpu_to_gpu', 0))
        if offload_g2c > 0 or offload_c2g > 0:
            result["kv_offload_bytes_gpu_to_cpu"] = offload_g2c
            result["kv_offload_bytes_cpu_to_gpu"] = offload_c2g
            result["kv_offload_time_gpu_to_cpu"] = float(last.get('kv_offload_time_gpu_to_cpu', 0))
            result["kv_offload_time_cpu_to_gpu"] = float(last.get('kv_offload_time_cpu_to_gpu', 0))

        cpu_cache_pct = float(last.get('cpu_kv_cache_usage_pct', 0))
        if cpu_cache_pct > 0:
            result["cpu_kv_cache_usage_pct"] = cpu_cache_pct

        result["total_prompt_tokens"] = int(last.get('prompt_tokens_total', 0))
        result["total_generation_tokens"] = int(last.get('generation_tokens_total', 0))
        result["total_requests_completed"] = int(last.get('request_success_total', 0))

    return result


def compute_throughput_stats(rows, server_metrics):
    """Compute throughput from completed requests."""
    successful = [r for r in rows if r.get('success') == 'True']
    if len(successful) < 2:
        return {}

    start = min(float(r['request_start_time']) for r in successful)
    end = max(float(r['request_complete_time']) for r in successful)
    duration = end - start
    if duration <= 0:
        return {}

    total_input = sum(int(r['input_tokens']) for r in successful)
    total_output = sum(int(r['output_tokens_actual']) for r in successful)

    return {
        "input_tput_tps": total_input / duration,
        "output_tput_tps": total_output / duration,
        "total_tput_tps": (total_input + total_output) / duration,
        "duration_seconds": duration,
    }


def main():
    result_filename = os.environ.get('RESULT_FILENAME', '')
    if not result_filename:
        print("ERROR: RESULT_FILENAME env var not set", file=sys.stderr)
        sys.exit(1)

    detailed_path = Path("results/trace_replay/detailed_results.csv")
    metrics_path = Path("results/metrics_server_metrics.csv")

    if not detailed_path.exists():
        print(f"ERROR: {detailed_path} not found", file=sys.stderr)
        sys.exit(1)

    rows = load_detailed_results(detailed_path)
    server_metrics = load_server_metrics(metrics_path) if metrics_path.exists() else []

    successful = [r for r in rows if r.get('success') == 'True']

    tp = int(os.environ.get('TP', '1'))
    num_gpus = tp

    agg = {
        "hw": os.environ.get('RUNNER_TYPE', ''),
        "users": int(os.environ.get('USERS', '0')),
        "image": os.environ.get('IMAGE', ''),
        "model": os.environ.get('MODEL', ''),
        "infmax_model_prefix": os.environ.get('MODEL_PREFIX', ''),
        "framework": os.environ.get('FRAMEWORK', ''),
        "precision": os.environ.get('PRECISION', ''),
        "scenario_type": "agentic-coding",
        "is_multinode": False,
        "tp": tp,
        "ep": int(os.environ.get('EP_SIZE', '1')),
        "dp_attention": os.environ.get('DP_ATTENTION', 'false'),
        "offload_mode": os.environ.get('OFFLOAD_MODE', 'off'),
        "num_requests_total": len(rows),
        "num_requests_successful": len(successful),
    }

    agg.update(compute_qps_stats(successful))
    agg.update(compute_latency_stats(successful))
    agg.update(compute_workload_stats(successful))
    agg.update(compute_cache_stats(successful, server_metrics))
    agg.update(compute_throughput_stats(successful, server_metrics))

    # Per-GPU throughput
    if "total_tput_tps" in agg and num_gpus > 0:
        agg["tput_per_gpu"] = agg["total_tput_tps"] / num_gpus
        agg["output_tput_per_gpu"] = agg.get("output_tput_tps", 0) / num_gpus
        agg["input_tput_per_gpu"] = agg.get("input_tput_tps", 0) / num_gpus

    output_path = f"agg_{result_filename}.json"
    with open(output_path, 'w') as f:
        json.dump(agg, f, indent=2)

    print(f"Saved aggregated agentic result to {output_path}")
    print(f"  Requests: {len(successful)}/{len(rows)} successful")
    if "mean_qps" in agg:
        print(f"  QPS: mean={agg['mean_qps']:.2f} median={agg.get('median_qps', 0):.2f} p99={agg.get('p99_qps', 0):.2f}")
    if agg.get("server_gpu_cache_hit_rate") is not None:
        print(f"  GPU cache hit rate: {agg['server_gpu_cache_hit_rate']:.1%}")
    if agg.get("tput_per_gpu") is not None:
        print(f"  Throughput per GPU: {agg['tput_per_gpu']:.0f} tok/s")


if __name__ == "__main__":
    main()
