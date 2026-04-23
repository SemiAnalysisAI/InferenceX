#!/usr/bin/env python3
"""Process agentic trace replay benchmark results into an aggregated JSON file.

Reads detailed_results.csv and metrics_server_metrics.csv from the benchmark
output directory and produces an agg_*.json file matching the naming convention
of fixed-seq-len results.

Expected env vars:
    RESULT_FILENAME - base name for output file (e.g., dsr1_tp4_users8_offloadcpu_...)
    MODEL, MODEL_PREFIX, FRAMEWORK, PRECISION, TP, EP_SIZE, DP_ATTENTION
    USERS, OFFLOADING, RUNNER_TYPE
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


def env_int(name, default=0):
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return int(value)


def env_bool(name, default=False):
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value.lower() in ("1", "true", "yes", "on")


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
        "std_qps": statistics.pstdev(qps_values) if len(qps_values) > 1 else 0.0,
    }


def compute_latency_stats(rows):
    """Emit the same keys fixed-seq-len emits (mean/median/std/p90/p99/p99.9
    for ttft, tpot, intvty, itl, e2el) so downstream consumers can treat
    both scenarios identically.

    - ttft: time to first token (s) — direct from trace replay
    - e2el: end-to-end request latency (s) — what trace replay calls ttlt
    - itl:  inter-token latency (s) — direct from trace replay
    - tpot: time per output token (s) — same measure as itl; aliased for
            fixed-seq-len compatibility
    - intvty: interactivity (1/tpot) — tokens/s per-request decode rate
    """
    ttfts = [float(r['ttft']) for r in rows if r.get('success') == 'True' and float(r['ttft']) > 0]
    e2els = [float(r['ttlt']) for r in rows if r.get('success') == 'True' and float(r['ttlt']) > 0]
    itls = [float(r['itl']) for r in rows if r.get('success') == 'True' and float(r['itl']) > 0]

    def stats_for(prefix, values):
        if not values:
            return {}
        out = {
            f"mean_{prefix}": statistics.mean(values),
            f"median_{prefix}": statistics.median(values),
            f"p90_{prefix}": percentile(values, 90),
            f"p99_{prefix}": percentile(values, 99),
            f"p99.9_{prefix}": percentile(values, 99.9),
        }
        out[f"std_{prefix}"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        return out

    result = {}
    result.update(stats_for("ttft", ttfts))
    result.update(stats_for("e2el", e2els))
    result.update(stats_for("itl", itls))
    # tpot = itl (agentic has no speculative-decoding distinction)
    result.update(stats_for("tpot", itls))
    # intvty = 1 / tpot (tokens/second per-request decode rate)
    if itls:
        intvtys = [1.0 / v for v in itls if v > 0]
        result.update(stats_for("intvty", intvtys))
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
            result[f"p99.9_{name}"] = percentile(values, 99.9)
            result[f"std_{name}"] = statistics.pstdev(values) if len(values) > 1 else 0.0
    return result


def compute_cache_stats(rows, server_metrics):
    """Compute cache hit rates from both detailed results and server metrics."""
    result = {
        "theoretical_cache_hit_rate": None,
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

    # Theoretical infinite-cache hit rate from detailed results.
    # A block counts as a hit iff its hash_id was seen earlier in the session.
    total_hit_blocks = sum(int(r.get('cache_hit_blocks', 0)) for r in rows)
    total_miss_blocks = sum(int(r.get('cache_miss_blocks', 0)) for r in rows)
    total_blocks = total_hit_blocks + total_miss_blocks
    if total_blocks > 0:
        result["theoretical_cache_hit_rate"] = total_hit_blocks / total_blocks

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

    # Result paths are relative to RESULT_DIR (set by the agentic script, e.g.
    # /workspace/results). When run standalone from the repo root, fall back
    # to ./results.
    result_dir = Path(os.environ.get('RESULT_DIR', 'results'))
    output_dir = Path(os.environ.get('AGENTIC_OUTPUT_DIR', '.'))

    detailed_path = result_dir / "trace_replay/detailed_results.csv"
    metrics_path = result_dir / "metrics_server_metrics.csv"

    if not detailed_path.exists():
        print(f"ERROR: {detailed_path} not found", file=sys.stderr)
        sys.exit(1)

    rows = load_detailed_results(detailed_path)
    server_metrics = load_server_metrics(metrics_path) if metrics_path.exists() else []

    successful = [r for r in rows if r.get('success') == 'True']

    is_multinode = env_bool('IS_MULTINODE')
    tp = env_int('TP', 1)
    ep = env_int('EP_SIZE', 1)
    dp_attention = os.environ.get('DP_ATTENTION', 'false')
    num_gpus = tp

    if is_multinode:
        prefill_num_workers = env_int('PREFILL_NUM_WORKERS')
        prefill_tp = env_int('PREFILL_TP')
        prefill_ep = env_int('PREFILL_EP', 1)
        prefill_dp_attention = os.environ.get('PREFILL_DP_ATTN', 'false')
        decode_num_workers = env_int('DECODE_NUM_WORKERS')
        decode_tp = env_int('DECODE_TP')
        decode_ep = env_int('DECODE_EP', 1)
        decode_dp_attention = os.environ.get('DECODE_DP_ATTN', 'false')
        num_prefill_gpu = prefill_num_workers * prefill_tp
        num_decode_gpu = decode_num_workers * decode_tp
        num_gpus = num_prefill_gpu + num_decode_gpu
        # Keep legacy fields populated for consumers that have not split by topology yet.
        tp = prefill_tp + decode_tp
        ep = max(prefill_ep, decode_ep)
        dp_attention = "true" if env_bool('PREFILL_DP_ATTN') or env_bool('DECODE_DP_ATTN') else "false"

    users = int(os.environ.get('USERS', '0'))
    agg = {
        "hw": os.environ.get('RUNNER_TYPE', ''),
        # conc mirrors fixed-seq-len's field; users is the historical agentic
        # name. Keep both so consumers can use either.
        "conc": users,
        "users": users,
        "image": os.environ.get('IMAGE', ''),
        "model": os.environ.get('MODEL', ''),
        "infmax_model_prefix": os.environ.get('MODEL_PREFIX', ''),
        "framework": os.environ.get('FRAMEWORK', ''),
        "precision": os.environ.get('PRECISION', ''),
        "spec_decoding": os.environ.get('SPEC_DECODING', 'none'),
        "disagg": env_bool('DISAGG'),
        "scenario_type": "agentic-coding",
        "is_multinode": is_multinode,
        "tp": tp,
        "ep": ep,
        "dp_attention": dp_attention,
        "offloading": os.environ.get('OFFLOADING', 'none'),
        "num_requests_total": len(rows),
        "num_requests_successful": len(successful),
    }

    if is_multinode:
        agg.update({
            "prefill_num_workers": prefill_num_workers,
            "prefill_tp": prefill_tp,
            "prefill_ep": prefill_ep,
            "prefill_dp_attention": prefill_dp_attention,
            "num_prefill_gpu": num_prefill_gpu,
            "decode_num_workers": decode_num_workers,
            "decode_tp": decode_tp,
            "decode_ep": decode_ep,
            "decode_dp_attention": decode_dp_attention,
            "num_decode_gpu": num_decode_gpu,
        })

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

    output_path = output_dir / f"{result_filename}.json"
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
