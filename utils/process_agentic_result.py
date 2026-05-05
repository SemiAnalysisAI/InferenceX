#!/usr/bin/env python3
"""Process aiperf agentic-replay output into the InferenceX agg_*.json shape.

Reads aiperf's three artifact files from $RESULT_DIR/trace_replay/ and emits
$AGENTIC_OUTPUT_DIR/$RESULT_FILENAME.json with the same key schema fixed-seq-len
and the legacy kv-cache-tester pipeline produce, so utils/summarize.py and
sibling aggregators keep working without changes.

Inputs:
- profile_export_aiperf.json  -- per-metric aggregate stats (avg/p50/p90/p99/...)
- profile_export.jsonl        -- one record per request (metadata + metrics)
- server_metrics_export.json  -- Prometheus scrape aggregates from the inference
                                 server (vLLM cache hit counters, KV usage, etc.)

Theoretical-cache-hit and output-tokens-expected stats are computed from the
trace metadata directly: aiperf records carry a ``conversation_id`` (= trace
id under the inferencex-agentx-mvp scenario) and ``turn_index``. We resolve
the original trace JSON via huggingface_hub's local cache and walk hash_ids
in trace order, counting hits whenever a hash_id has appeared earlier in the
same trace.

Required env vars:
    RESULT_FILENAME   - base name for output file
    MODEL, MODEL_PREFIX, FRAMEWORK, PRECISION, TP, EP_SIZE, DP_ATTENTION,
    CONC, OFFLOADING, RUNNER_TYPE
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from collections.abc import Iterable
from pathlib import Path

# Trace metadata lookup: conversation_id (= trace id) -> per-turn dict with
# ``hash_ids`` and ``output_length``. Built lazily from the HF dataset cache.
_TRACE_METADATA_CACHE: dict[str, list[dict]] | None = None
_HF_DATASET = "semianalysisai/cc-traces-weka-042026"


# ---- helpers ---------------------------------------------------------------


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def env_int(name: str, default: int = 0) -> int:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return int(value)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value.lower() in ("1", "true", "yes", "on")


def stats_for(prefix: str, values: list[float]) -> dict:
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


# ---- aiperf artifact loaders -----------------------------------------------


def load_aggregate(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_records(path: Path) -> list[dict]:
    """Load profile_export.jsonl as a list of (metadata, metrics) dicts."""
    records: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("error"):
                continue
            records.append(obj)
    return records


def load_server_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ---- trace metadata --------------------------------------------------------


def _hf_traces_dir() -> Path | None:
    """Locate the HuggingFace cache directory for the weka traces dataset.

    Returns the directory containing per-trace JSON files, or None if the
    dataset isn't present locally. Mirrors the layout
    huggingface_hub.snapshot_download() produces:
    ``$HF_HUB_CACHE/datasets--<org>--<name>/snapshots/<revision>/``.
    """
    hub_cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        cache_root = Path(hub_cache)
    else:
        home = os.environ.get("HF_HOME")
        cache_root = Path(home) / "hub" if home else Path.home() / ".cache" / "huggingface" / "hub"

    org, name = _HF_DATASET.split("/", 1)
    snapshots = cache_root / f"datasets--{org}--{name}" / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates = sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    # Prefer the snapshot that contains usable trace files. The published HF
    # dataset ships a single ``traces.jsonl`` (one trace per line); older /
    # local mirrors may use per-trace ``*.json`` files instead. Accept either.
    for c in candidates:
        if not c.is_dir():
            continue
        if any(c.glob("*.jsonl")) or any(c.glob("*.json")):
            return c
    return None


def _iter_trace_blobs(traces_dir: Path):
    """Yield each trace JSON dict from the local HF cache.

    Handles both layouts:
    - one JSONL file (e.g. ``traces.jsonl``) with one trace per line — the
      shape published HF dataset format.
    - one ``*.json`` per trace — the legacy per-file layout.
    """
    for path in sorted(traces_dir.glob("*.jsonl")):
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue
    for path in sorted(traces_dir.glob("*.json")):
        try:
            with open(path) as f:
                yield json.load(f)
        except (json.JSONDecodeError, OSError):
            continue


def _load_trace_metadata() -> dict[str, list[dict]]:
    """Build {trace_id: [{hash_ids, output_length}, ...]} from local HF cache."""
    global _TRACE_METADATA_CACHE
    if _TRACE_METADATA_CACHE is not None:
        return _TRACE_METADATA_CACHE
    out: dict[str, list[dict]] = {}
    traces_dir = _hf_traces_dir()
    if traces_dir is None:
        _TRACE_METADATA_CACHE = out
        return out
    for blob in _iter_trace_blobs(traces_dir):
        trace_id = blob.get("id")
        if not trace_id:
            continue
        per_turn: list[dict] = []
        for req in blob.get("requests", []):
            if req.get("type") not in ("n", "s"):
                continue
            # The on-disk trace uses ``in``/``out`` (the loader's Pydantic
            # aliases for ``input_length`` / ``output_length``); accept either.
            output_length = req.get("output_length")
            if output_length is None:
                output_length = req.get("out")
            per_turn.append(
                {
                    "hash_ids": list(req.get("hash_ids") or []),
                    "output_length": int(output_length or 0),
                }
            )
        if per_turn:
            out[trace_id] = per_turn
    _TRACE_METADATA_CACHE = out
    return out


def _conversation_id_to_trace_id(conv_id: str | None) -> str | None:
    """Strip aiperf's ``::sa:<agent>`` suffix to recover the parent trace id."""
    if not conv_id:
        return None
    return conv_id.split("::", 1)[0]


# ---- metric extraction -----------------------------------------------------


def _ms_to_s(values_ms: Iterable[float]) -> list[float]:
    return [v / 1000.0 for v in values_ms if v is not None and v > 0]


def _extract_per_record_floats(records: list[dict], key: str) -> list[float]:
    """Pull a scalar metric value from each record. Skips missing/null."""
    out: list[float] = []
    for r in records:
        m = r.get("metrics", {}).get(key)
        if m is None:
            continue
        v = m.get("value") if isinstance(m, dict) else m
        if v is None:
            continue
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def _extract_per_record_ints(records: list[dict], key: str) -> list[int]:
    out: list[int] = []
    for r in records:
        m = r.get("metrics", {}).get(key)
        if m is None:
            continue
        v = m.get("value") if isinstance(m, dict) else m
        if v is None:
            continue
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            continue
    return out


def compute_latency_stats(records: list[dict]) -> dict:
    """Per-request latencies. aiperf reports ms; legacy schema is seconds."""
    ttfts = _ms_to_s(_extract_per_record_floats(records, "time_to_first_token"))
    e2els = _ms_to_s(_extract_per_record_floats(records, "request_latency"))
    itls = _ms_to_s(_extract_per_record_floats(records, "inter_token_latency"))

    result: dict = {}
    result.update(stats_for("ttft", ttfts))
    result.update(stats_for("e2el", e2els))
    result.update(stats_for("itl", itls))
    # tpot ≡ itl (no spec-decoding distinction in agentic).
    result.update(stats_for("tpot", itls))
    if itls:
        intvtys = [1.0 / v for v in itls if v > 0]
        result.update(stats_for("intvty", intvtys))
    return result


def compute_qps_stats(records: list[dict]) -> dict:
    """1s sliding-window QPS from request_end_ns."""
    ends_ns = [
        int(r["metadata"]["request_end_ns"])
        for r in records
        if r.get("metadata", {}).get("request_end_ns")
    ]
    if len(ends_ns) < 2:
        return {}
    ends = sorted(t / 1e9 for t in ends_ns)
    duration = ends[-1] - ends[0]
    if duration <= 0:
        return {}

    window = 1.0
    qps_values: list[float] = []
    t = ends[0]
    while t + window <= ends[-1]:
        count = sum(1 for ct in ends if t <= ct < t + window)
        qps_values.append(count / window)
        t += window

    if not qps_values:
        return {"mean_qps": len(ends) / duration}

    return {
        "mean_qps": statistics.mean(qps_values),
        "median_qps": statistics.median(qps_values),
        "p90_qps": percentile(qps_values, 90),
        "p99_qps": percentile(qps_values, 99),
        "p99.9_qps": percentile(qps_values, 99.9),
        "std_qps": statistics.pstdev(qps_values) if len(qps_values) > 1 else 0.0,
    }


def compute_workload_stats(records: list[dict]) -> dict:
    """Input/output token distributions, plus expected from trace metadata."""
    isls = _extract_per_record_ints(records, "input_sequence_length")
    osls = _extract_per_record_ints(records, "output_sequence_length")

    result: dict = {}
    for name, values in (("input_tokens", isls), ("output_tokens_actual", osls)):
        if not values:
            continue
        result[f"mean_{name}"] = statistics.mean(values)
        result[f"median_{name}"] = statistics.median(values)
        result[f"p90_{name}"] = percentile(values, 90)
        result[f"p99_{name}"] = percentile(values, 99)
        result[f"p99.9_{name}"] = percentile(values, 99.9)
        result[f"std_{name}"] = (
            statistics.pstdev(values) if len(values) > 1 else 0.0
        )

    # output_tokens_expected: trace's recorded output_length per turn.
    metadata = _load_trace_metadata()
    if metadata:
        expected: list[int] = []
        for r in records:
            md = r.get("metadata", {})
            trace_id = _conversation_id_to_trace_id(md.get("conversation_id"))
            turn_index = md.get("turn_index")
            if trace_id is None or turn_index is None:
                continue
            turns = metadata.get(trace_id)
            if not turns or turn_index >= len(turns):
                continue
            expected.append(turns[turn_index]["output_length"])
        if expected:
            result["mean_output_tokens_expected"] = statistics.mean(expected)
            result["median_output_tokens_expected"] = statistics.median(expected)
            result["p90_output_tokens_expected"] = percentile(expected, 90)
            result["p99_output_tokens_expected"] = percentile(expected, 99)
            result["p99.9_output_tokens_expected"] = percentile(expected, 99.9)
            result["std_output_tokens_expected"] = (
                statistics.pstdev(expected) if len(expected) > 1 else 0.0
            )

    return result


def compute_throughput_stats(records: list[dict], aggregate: dict) -> dict:
    """Wall-clock throughput. Prefer aiperf's aggregate when available."""
    isls = _extract_per_record_ints(records, "input_sequence_length")
    osls = _extract_per_record_ints(records, "output_sequence_length")
    starts_ns = [
        int(r["metadata"]["request_start_ns"])
        for r in records
        if r.get("metadata", {}).get("request_start_ns")
    ]
    ends_ns = [
        int(r["metadata"]["request_end_ns"])
        for r in records
        if r.get("metadata", {}).get("request_end_ns")
    ]
    if not starts_ns or not ends_ns:
        return {}
    duration = (max(ends_ns) - min(starts_ns)) / 1e9
    if duration <= 0:
        return {}
    total_input = sum(isls)
    total_output = sum(osls)
    return {
        "input_tput_tps": total_input / duration,
        "output_tput_tps": total_output / duration,
        "total_tput_tps": (total_input + total_output) / duration,
        "duration_seconds": duration,
    }


def compute_cache_stats(records: list[dict], server_metrics: dict) -> dict:
    """Cache-hit metrics: theoretical (from trace metadata) + actual (server)."""
    result: dict = {
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
        "response_cache_hit_rate": None,
    }

    # -- Theoretical infinite-cache hit rate from trace metadata ------------
    # For each completed request, walk its trace's hash_ids in trace order
    # (turns 0..k_i where k_i = max turn_index seen for this conversation).
    # A block is a hit if its hash_id appeared earlier in the same trace.
    metadata = _load_trace_metadata()
    if metadata:
        max_turn_per_conv: dict[str, int] = {}
        for r in records:
            md = r.get("metadata", {})
            conv_id = md.get("conversation_id")
            ti = md.get("turn_index")
            if conv_id is None or ti is None:
                continue
            prev = max_turn_per_conv.get(conv_id, -1)
            if ti > prev:
                max_turn_per_conv[conv_id] = ti

        total_hits = 0
        total_blocks = 0
        for conv_id, max_turn in max_turn_per_conv.items():
            trace_id = _conversation_id_to_trace_id(conv_id)
            turns = metadata.get(trace_id) if trace_id else None
            if not turns:
                continue
            seen: set[int] = set()
            for ti in range(min(max_turn + 1, len(turns))):
                for h in turns[ti]["hash_ids"]:
                    if h in seen:
                        total_hits += 1
                    else:
                        seen.add(h)
                    total_blocks += 1
        if total_blocks > 0:
            result["theoretical_cache_hit_rate"] = total_hits / total_blocks

    # -- Per-response cache-hit rate (vLLM/OpenAI cached_tokens) ------------
    cached = _extract_per_record_ints(records, "usage_prompt_cache_read_tokens")
    isls_per_record = _extract_per_record_ints(records, "input_sequence_length")
    if cached and isls_per_record and len(cached) == len(isls_per_record):
        prompt_total = sum(isls_per_record)
        if prompt_total > 0:
            result["response_cache_hit_rate"] = sum(cached) / prompt_total

    # -- Server-side Prometheus scrape (vLLM-specific keys) -----------------
    # aiperf's server_metrics_export.json shape:
    #   {"metrics": {<name>: {"type": ..., "series": [{"stats": {...}}, ...]}}}
    # We aggregate across series (multiple endpoints / label sets) and prefer
    # ``total`` for counters, then ``max``/``avg`` for gauges.
    metrics_by_name = _index_server_metrics(server_metrics)

    def _final_value(metric_name: str) -> float | None:
        entry = metrics_by_name.get(metric_name)
        if not isinstance(entry, dict):
            return None
        series = entry.get("series") or []
        if not isinstance(series, list):
            return None
        for stats_key in ("total", "max", "avg"):
            agg = 0.0
            found = False
            for s in series:
                if not isinstance(s, dict):
                    continue
                stats = s.get("stats")
                if not isinstance(stats, dict):
                    continue
                v = stats.get(stats_key)
                if v is None:
                    continue
                try:
                    agg += float(v)
                    found = True
                except (TypeError, ValueError):
                    continue
            if found:
                return agg
        return None

    hits = _final_value("vllm:prefix_cache_hits")
    queries = _final_value("vllm:prefix_cache_queries")
    if hits is not None and queries and queries > 0:
        result["server_gpu_cache_hit_rate"] = hits / queries

    cpu_hits = _final_value("vllm:cpu_prefix_cache_hits")
    cpu_queries = _final_value("vllm:cpu_prefix_cache_queries")
    if cpu_hits is not None and cpu_queries and cpu_queries > 0:
        result["server_cpu_cache_hit_rate"] = cpu_hits / cpu_queries

    for src_key, dst_key in (
        ("vllm:kv_offload_bytes_gpu_to_cpu", "kv_offload_bytes_gpu_to_cpu"),
        ("vllm:kv_offload_bytes_cpu_to_gpu", "kv_offload_bytes_cpu_to_gpu"),
        ("vllm:kv_offload_time_gpu_to_cpu", "kv_offload_time_gpu_to_cpu"),
        ("vllm:kv_offload_time_cpu_to_gpu", "kv_offload_time_cpu_to_gpu"),
        ("vllm:cpu_kv_cache_usage_perc", "cpu_kv_cache_usage_pct"),
    ):
        v = _final_value(src_key)
        if v is not None:
            result[dst_key] = v

    pt = _final_value("vllm:prompt_tokens")
    if pt is not None:
        result["total_prompt_tokens"] = int(pt)
    gt = _final_value("vllm:generation_tokens")
    if gt is not None:
        result["total_generation_tokens"] = int(gt)

    # Fallback to per-record sums when server metrics aren't present.
    isls = _extract_per_record_ints(records, "input_sequence_length")
    osls = _extract_per_record_ints(records, "output_sequence_length")
    if result["total_prompt_tokens"] is None and isls:
        result["total_prompt_tokens"] = sum(isls)
    if result["total_generation_tokens"] is None and osls:
        result["total_generation_tokens"] = sum(osls)
    result["total_requests_completed"] = len(records)

    return result


def _index_server_metrics(server_metrics: dict) -> dict[str, dict]:
    """Return the metrics dict from aiperf's server_metrics_export.json.

    aiperf v0.8 schema: top-level ``{"metrics": {<name>: {"type": ...,
    "series": [{"stats": {...}}, ...]}}}``. The ``metrics`` value is a
    ``dict`` keyed by metric name, NOT a list. We just return it as-is so
    callers can do ``out[metric_name]`` lookups.

    See ``utils/aiperf/docs/server-metrics/server-metrics-json-schema.md``
    for the full schema.
    """
    if not isinstance(server_metrics, dict):
        return {}
    metrics = server_metrics.get("metrics")
    if isinstance(metrics, dict):
        return metrics
    return {}


# ---- main ------------------------------------------------------------------


def build_agg(
    records: list[dict],
    aggregate: dict,
    server_metrics: dict,
) -> dict:
    """Compose the agg_*.json body from the three aiperf inputs."""
    is_multinode = env_bool("IS_MULTINODE")
    tp = env_int("TP", 1)
    ep = env_int("EP_SIZE", 1)
    dp_attention = os.environ.get("DP_ATTENTION", "false")
    num_gpus = tp

    if is_multinode:
        prefill_num_workers = env_int("PREFILL_NUM_WORKERS")
        prefill_tp = env_int("PREFILL_TP")
        prefill_ep = env_int("PREFILL_EP", 1)
        prefill_dp_attention = os.environ.get("PREFILL_DP_ATTN", "false")
        decode_num_workers = env_int("DECODE_NUM_WORKERS")
        decode_tp = env_int("DECODE_TP")
        decode_ep = env_int("DECODE_EP", 1)
        decode_dp_attention = os.environ.get("DECODE_DP_ATTN", "false")
        num_prefill_gpu = prefill_num_workers * prefill_tp
        num_decode_gpu = decode_num_workers * decode_tp
        num_gpus = num_prefill_gpu + num_decode_gpu
        tp = prefill_tp + decode_tp
        ep = max(prefill_ep, decode_ep)
        dp_attention = (
            "true"
            if env_bool("PREFILL_DP_ATTN") or env_bool("DECODE_DP_ATTN")
            else "false"
        )

    conc = int(os.environ.get("CONC", "0"))
    agg: dict = {
        "hw": os.environ.get("RUNNER_TYPE", ""),
        "conc": conc,
        "image": os.environ.get("IMAGE", ""),
        "model": os.environ.get("MODEL", ""),
        "infmax_model_prefix": os.environ.get("MODEL_PREFIX", ""),
        "framework": os.environ.get("FRAMEWORK", ""),
        "precision": os.environ.get("PRECISION", ""),
        "spec_decoding": os.environ.get("SPEC_DECODING", "none"),
        "disagg": env_bool("DISAGG"),
        "scenario_type": "agentic-coding",
        "is_multinode": is_multinode,
        "tp": tp,
        "ep": ep,
        "dp_attention": dp_attention,
        "offloading": os.environ.get("OFFLOADING", "none"),
        "num_requests_total": len(records),
        "num_requests_successful": len(records),
    }

    if is_multinode:
        agg.update(
            {
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
            }
        )

    agg.update(compute_qps_stats(records))
    agg.update(compute_latency_stats(records))
    agg.update(compute_workload_stats(records))
    agg.update(compute_cache_stats(records, server_metrics))
    agg.update(compute_throughput_stats(records, aggregate))

    if "total_tput_tps" in agg and num_gpus > 0:
        agg["tput_per_gpu"] = agg["total_tput_tps"] / num_gpus
        agg["output_tput_per_gpu"] = agg.get("output_tput_tps", 0) / num_gpus
        agg["input_tput_per_gpu"] = agg.get("input_tput_tps", 0) / num_gpus

    return agg


def _resolve_artifact_dir(result_dir: Path) -> Path:
    """Find the dir containing aiperf's profile_export* files.

    aiperf accepts ``--output-artifact-dir`` and writes directly into it when
    ``--num-profile-runs == 1`` (our default), but creates a per-run subdir
    when that flag is > 1. Handle both: prefer ``result_dir/trace_replay``
    when it has the export files, else descend into the first child dir
    that does.
    """
    base = result_dir / "trace_replay"
    if (base / "profile_export.jsonl").is_file():
        return base
    if base.is_dir():
        for child in sorted(base.iterdir()):
            if child.is_dir() and (child / "profile_export.jsonl").is_file():
                return child
    return base


def main() -> int:
    result_filename = os.environ.get("RESULT_FILENAME", "")
    if not result_filename:
        print("ERROR: RESULT_FILENAME env var not set", file=sys.stderr)
        return 1

    result_dir = Path(os.environ.get("RESULT_DIR", "results"))
    output_dir = Path(os.environ.get("AGENTIC_OUTPUT_DIR", "."))

    artifact_dir = _resolve_artifact_dir(result_dir)
    aggregate_path = artifact_dir / "profile_export_aiperf.json"
    jsonl_path = artifact_dir / "profile_export.jsonl"
    server_metrics_path = artifact_dir / "server_metrics_export.json"

    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} not found", file=sys.stderr)
        return 1

    records = load_records(jsonl_path)
    aggregate = load_aggregate(aggregate_path) if aggregate_path.exists() else {}
    server_metrics = load_server_metrics(server_metrics_path)

    agg = build_agg(records, aggregate, server_metrics)

    output_path = output_dir / f"{result_filename}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(agg, f, indent=2)

    print(f"Saved aggregated agentic result to {output_path}")
    print(f"  Requests: {len(records)} successful (aiperf drops error records)")
    if "mean_qps" in agg:
        print(
            f"  QPS: mean={agg['mean_qps']:.2f} "
            f"median={agg.get('median_qps', 0):.2f} "
            f"p99={agg.get('p99_qps', 0):.2f}"
        )
    if agg.get("server_gpu_cache_hit_rate") is not None:
        print(f"  GPU cache hit rate: {agg['server_gpu_cache_hit_rate']:.1%}")
    if agg.get("response_cache_hit_rate") is not None:
        print(f"  Response cache hit rate: {agg['response_cache_hit_rate']:.1%}")
    if agg.get("theoretical_cache_hit_rate") is not None:
        print(
            f"  Theoretical cache hit rate: "
            f"{agg['theoretical_cache_hit_rate']:.1%}"
        )
    if agg.get("tput_per_gpu") is not None:
        print(f"  Throughput per GPU: {agg['tput_per_gpu']:.0f} tok/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
