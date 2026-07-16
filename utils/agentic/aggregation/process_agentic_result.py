#!/usr/bin/env python3
"""Process aiperf agentic-replay output into InferenceX aggregate JSON."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from .aggregation_common import round_floats
from .request_metrics import compute_request_metrics, load_aggregate, load_records_with_accounting
from .server_log_metrics import find_server_log_paths
from .server_metrics import compute_server_metrics, load_server_metrics


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


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if value in (None, ""):
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def optional_component_metadata(env_name: str) -> dict[str, str] | None:
    """Parse strict optional component metadata from a JSON environment value."""
    raw_value = os.environ.get(env_name)
    if raw_value in (None, "", "null"):
        return None

    try:
        metadata = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{env_name} must contain valid JSON") from exc

    if not isinstance(metadata, dict) or set(metadata) != {"name", "version"}:
        raise SystemExit(f"{env_name} must contain exactly 'name' and 'version'")
    if not all(isinstance(metadata[key], str) and metadata[key] for key in metadata):
        raise SystemExit(f"{env_name} name and version must be non-empty strings")
    return metadata


def optional_kv_offload_backend_metadata(
    env_name: str,
) -> dict[str, str] | None:
    """Parse KV offload backend metadata with an optional version."""
    raw_value = os.environ.get(env_name)
    if raw_value in (None, "", "null"):
        return None

    try:
        metadata = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{env_name} must contain valid JSON") from exc

    if not isinstance(metadata, dict) or not set(metadata) <= {"name", "version"}:
        raise SystemExit(f"{env_name} may contain only 'name' and 'version'")
    if set(metadata) not in ({"name"}, {"name", "version"}):
        raise SystemExit(f"{env_name} must contain 'name' and optional 'version'")
    if not all(isinstance(value, str) and value for value in metadata.values()):
        raise SystemExit(f"{env_name} values must be non-empty strings")
    return metadata


def _validate_kv_offload_env() -> tuple[str, dict[str, str] | None]:
    kv_offloading = required_env("KV_OFFLOADING")
    backend_name = os.environ.get("KV_OFFLOAD_BACKEND", "")
    backend_metadata = optional_kv_offload_backend_metadata(
        "KV_OFFLOAD_BACKEND_METADATA"
    )
    if kv_offloading == "none":
        if backend_name or backend_metadata is not None:
            raise SystemExit("KV_OFFLOAD_BACKEND must be empty when KV_OFFLOADING=none")
    else:
        if not backend_name or backend_name == "none" or backend_metadata is None:
            raise SystemExit("KV_OFFLOAD_BACKEND is required when KV_OFFLOADING is enabled")
        if backend_metadata["name"] != backend_name:
            raise SystemExit(
                "KV_OFFLOAD_BACKEND must match KV_OFFLOAD_BACKEND_METADATA.name"
            )
    return kv_offloading, backend_metadata


def _gpu_shape() -> tuple[dict[str, Any], int, int, int, str]:
    is_multinode = env_bool("IS_MULTINODE")
    tp = env_int("TP", 1)
    ep = env_int("EP_SIZE", 1)
    dp_attention = os.environ.get("DP_ATTENTION", "false")
    fields: dict[str, Any] = {}

    if not is_multinode:
        pp = env_int("PP_SIZE", 1)
        dcp_size = env_int("DCP_SIZE", 1)
        pcp_size = env_int("PCP_SIZE", 1)
        if pp <= 0 or dcp_size <= 0 or pcp_size <= 0:
            raise SystemExit(
                "PP_SIZE, DCP_SIZE, and PCP_SIZE must be positive integers."
            )
        fields.update({"pp": pp, "dcp_size": dcp_size, "pcp_size": pcp_size})
        return fields, tp * pp * pcp_size, tp, ep, dp_attention

    prefill_num_workers = env_int("PREFILL_NUM_WORKERS")
    prefill_tp = env_int("PREFILL_TP")
    prefill_pp = env_int("PREFILL_PP_SIZE", 1)
    prefill_dcp_size = env_int("PREFILL_DCP_SIZE", 1)
    prefill_pcp_size = env_int("PREFILL_PCP_SIZE", 1)
    prefill_ep = env_int("PREFILL_EP", 1)
    prefill_dp_attention = os.environ.get("PREFILL_DP_ATTN", "false")
    decode_num_workers = env_int("DECODE_NUM_WORKERS")
    decode_tp = env_int("DECODE_TP")
    decode_pp = env_int("DECODE_PP_SIZE", 1)
    decode_dcp_size = env_int("DECODE_DCP_SIZE", 1)
    decode_pcp_size = env_int("DECODE_PCP_SIZE", 1)
    decode_ep = env_int("DECODE_EP", 1)
    decode_dp_attention = os.environ.get("DECODE_DP_ATTN", "false")
    worker_parallelism = (
        prefill_pp,
        prefill_dcp_size,
        prefill_pcp_size,
        decode_pp,
        decode_dcp_size,
        decode_pcp_size,
    )
    if any(value <= 0 for value in worker_parallelism):
        raise SystemExit(
            "Multinode PP, DCP, and PCP sizes must be positive integers."
        )
    prefill_hardware = os.environ.get("PREFILL_HARDWARE", "")
    decode_hardware = os.environ.get("DECODE_HARDWARE", "")
    if bool(prefill_hardware) != bool(decode_hardware):
        raise SystemExit(
            "PREFILL_HARDWARE and DECODE_HARDWARE must be specified together."
        )
    num_prefill_gpu = prefill_num_workers * prefill_tp * prefill_pp * prefill_pcp_size
    num_decode_gpu = decode_num_workers * decode_tp * decode_pp * decode_pcp_size
    num_gpus = num_prefill_gpu + num_decode_gpu
    # Aggregated configs set decode num-worker 0 (prefill+decode co-located on one
    # worker), so there are no separate decode GPUs. Mirror process_result.py and drop
    # the decode-side parallelism, so TP/EP and the per-GPU throughput denominator
    # reflect the single aggregated worker instead of double-counting its GPUs.
    if num_decode_gpu <= 0:
        decode_tp = 0
        decode_ep = 0
        decode_pp = 1
        decode_dcp_size = 1
        decode_pcp_size = 1
    tp = prefill_tp + decode_tp
    ep = max(prefill_ep, decode_ep)
    dp_attention = (
        "true"
        if env_bool("PREFILL_DP_ATTN") or env_bool("DECODE_DP_ATTN")
        else "false"
    )
    fields.update(
        {
            "prefill_num_workers": prefill_num_workers,
            "prefill_tp": prefill_tp,
            "prefill_pp": prefill_pp,
            "prefill_dcp_size": prefill_dcp_size,
            "prefill_pcp_size": prefill_pcp_size,
            "prefill_ep": prefill_ep,
            "prefill_dp_attention": prefill_dp_attention,
            "num_prefill_gpu": num_prefill_gpu,
            "decode_num_workers": decode_num_workers,
            "decode_tp": decode_tp,
            "decode_pp": decode_pp,
            "decode_dcp_size": decode_dcp_size,
            "decode_pcp_size": decode_pcp_size,
            "decode_ep": decode_ep,
            "decode_dp_attention": decode_dp_attention,
            "num_decode_gpu": num_decode_gpu,
        }
    )
    if prefill_hardware:
        fields["prefill_hw"] = prefill_hardware
        fields["decode_hw"] = decode_hardware
    return fields, num_gpus, tp, ep, dp_attention


def build_agg(
    records: list[dict[str, Any]],
    aggregate: dict[str, Any],
    server_metrics: dict[str, Any],
    *,
    request_accounting: dict[str, Any] | None = None,
    server_log_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Compose the agg_*.json body from the three aiperf inputs."""
    kv_offloading, kv_offload_backend = _validate_kv_offload_env()
    multinode_fields, num_gpus, tp, ep, dp_attention = _gpu_shape()
    framework = os.environ.get("FRAMEWORK", "")
    request_accounting = request_accounting or {
        "records_total": len(records),
        "records_profiled": len(records),
        "records_dropped_total": 0,
        "records_warmup_dropped": 0,
        "records_error_dropped": 0,
        "error_categories": {},
    }

    agg: dict[str, Any] = {
        "hw": os.environ.get("RUNNER_TYPE", ""),
        "conc": int(os.environ.get("CONC", "0")),
        "image": os.environ.get("IMAGE", ""),
        "model": os.environ.get("MODEL", ""),
        "infmax_model_prefix": os.environ.get("MODEL_PREFIX", ""),
        "framework": framework,
        "precision": os.environ.get("PRECISION", ""),
        "spec_decoding": os.environ.get("SPEC_DECODING", "none"),
        "disagg": env_bool("DISAGG"),
        "scenario_type": "agentic-coding",
        "is_multinode": env_bool("IS_MULTINODE"),
        "tp": tp,
        "ep": ep,
        "dp_attention": dp_attention,
        "kv_offloading": kv_offloading,
        "kv_offload_backend": kv_offload_backend,
        "allocated_cpu_dram_gb": env_int("TOTAL_CPU_DRAM_GB"),
        "num_requests_total": request_accounting["records_total"],
        "num_requests_successful": len(records),
        "request_accounting": request_accounting,
    }
    agg.update(multinode_fields)

    router = optional_component_metadata("ROUTER_METADATA")
    if router is not None:
        agg["router"] = router

    kv_p2p_transfer = os.environ.get("KV_P2P_TRANSFER")
    if kv_p2p_transfer:
        agg["kv_p2p_transfer"] = kv_p2p_transfer

    metadata = aggregate.get("metadata")
    if isinstance(metadata, dict):
        dataset = metadata.get("dataset")
        if isinstance(dataset, dict):
            agg["dataset"] = dataset

    request_flat, request_nested = compute_request_metrics(records, aggregate)
    _, server_nested, warnings = compute_server_metrics(
        server_metrics,
        framework=framework,
        records=records,
        server_log_paths=server_log_paths,
    )

    if "total_tput_tps" in request_flat and num_gpus > 0:
        request_nested["throughput"]["per_gpu"] = {
            "total_tput_tps": request_flat["total_tput_tps"] / num_gpus,
            "output_tput_tps": request_flat.get("output_tput_tps", 0) / num_gpus,
            "input_tput_tps": request_flat.get("input_tput_tps", 0) / num_gpus,
        }

    agg["request_metrics"] = request_nested
    agg["server_metrics"] = server_nested
    agg["kv_cache_pool_tokens"] = server_nested["kv_cache"]["gpu_total_tokens"]
    if warnings:
        agg["warnings"] = warnings
    return agg


def _resolve_artifact_dir(result_dir: Path) -> Path:
    """Find the dir containing aiperf's profile_export* files."""
    base = result_dir / "aiperf_artifacts"
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

    records, request_accounting = load_records_with_accounting(jsonl_path)
    aggregate = load_aggregate(aggregate_path) if aggregate_path.exists() else {}
    server_metrics = load_server_metrics(server_metrics_path)
    server_log_paths = find_server_log_paths(result_dir)
    agg = round_floats(
        build_agg(
            records,
            aggregate,
            server_metrics,
            request_accounting=request_accounting,
            server_log_paths=server_log_paths,
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
