"""AgentX aggregate construction with explicit inputs and no implicit I/O."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ..metadata import parse_component_metadata
from ..topology import Parallelism, validate_parallelism
from .request_metrics import compute_request_metrics
from .server_metrics import compute_server_metrics


def _env_int(env: Mapping[str, str], name: str, default: int = 0) -> int:
    value = env.get(name)
    if value in (None, ""):
        return default
    return int(value)


def _env_bool(env: Mapping[str, str], name: str, default: bool = False) -> bool:
    value = env.get(name)
    if value in (None, ""):
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _required_env(env: Mapping[str, str], name: str) -> str:
    value = env.get(name)
    if value in (None, ""):
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def _optional_component_metadata(env: Mapping[str, str], env_name: str) -> dict[str, str] | None:
    return parse_component_metadata(
        env.get(env_name), env_name, error_type=SystemExit,
    )


def _optional_kv_offload_backend_metadata(env: Mapping[str, str], env_name: str) -> dict[str, str] | None:
    return parse_component_metadata(
        env.get(env_name), env_name, version_optional=True, error_type=SystemExit,
    )


def _validate_kv_offload_env(env: Mapping[str, str]) -> tuple[str, dict[str, str] | None]:
    kv_offloading = _required_env(env, "KV_OFFLOADING")
    backend_name = env.get("KV_OFFLOAD_BACKEND", "")
    backend_metadata = _optional_kv_offload_backend_metadata(
        env, "KV_OFFLOAD_BACKEND_METADATA"
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


def _gpu_shape(env: Mapping[str, str]) -> tuple[dict[str, Any], int, int, int, str]:
    is_multinode = _env_bool(env, "IS_MULTINODE")
    tp = _env_int(env, "TP", 1)
    ep = _env_int(env, "EP_SIZE", 1)
    dp_attention = env.get("DP_ATTENTION", "false")
    if not is_multinode:
        parallelism = Parallelism(
            tp=tp, pp=_env_int(env, "PP_SIZE", 1), dcp_size=_env_int(env, "DCP_SIZE", 1),
            pcp_size=_env_int(env, "PCP_SIZE", 1), ep=ep,
        )
        validate_parallelism(parallelism, error_type=SystemExit)
        fields = {"pp": parallelism.pp, "dcp_size": parallelism.dcp_size, "pcp_size": parallelism.pcp_size}
        return fields, parallelism.gpus_per_worker, tp, ep, dp_attention

    prefill_num_workers = _env_int(env, "PREFILL_NUM_WORKERS")
    prefill = Parallelism(
        tp=_env_int(env, "PREFILL_TP"), pp=_env_int(env, "PREFILL_PP_SIZE", 1),
        dcp_size=_env_int(env, "PREFILL_DCP_SIZE", 1), pcp_size=_env_int(env, "PREFILL_PCP_SIZE", 1),
        ep=_env_int(env, "PREFILL_EP", 1),
    )
    prefill_dp_attention = env.get("PREFILL_DP_ATTN", "false")
    decode_num_workers = _env_int(env, "DECODE_NUM_WORKERS")
    decode = Parallelism(
        tp=_env_int(env, "DECODE_TP"), pp=_env_int(env, "DECODE_PP_SIZE", 1),
        dcp_size=_env_int(env, "DECODE_DCP_SIZE", 1), pcp_size=_env_int(env, "DECODE_PCP_SIZE", 1),
        ep=_env_int(env, "DECODE_EP", 1),
    )
    decode_dp_attention = env.get("DECODE_DP_ATTN", "false")
    validate_parallelism(prefill, decode, error_type=SystemExit)
    prefill_hardware = env.get("PREFILL_HARDWARE", "")
    decode_hardware = env.get("DECODE_HARDWARE", "")
    if bool(prefill_hardware) != bool(decode_hardware):
        raise SystemExit(
            "PREFILL_HARDWARE and DECODE_HARDWARE must be specified together."
        )
    num_prefill_gpu = prefill_num_workers * prefill.gpus_per_worker
    num_decode_gpu = decode_num_workers * decode.gpus_per_worker
    num_gpus = num_prefill_gpu + num_decode_gpu
    decode = decode.for_decode(num_decode_gpu)
    tp = prefill.tp + decode.tp
    ep = max(prefill.ep, decode.ep)
    dp_attention = (
        "true"
        if _env_bool(env, "PREFILL_DP_ATTN") or _env_bool(env, "DECODE_DP_ATTN")
        else "false"
    )
    fields = {
        "prefill_num_workers": prefill_num_workers,
        **prefill.fields("prefill_"),
        "prefill_dp_attention": prefill_dp_attention,
        "num_prefill_gpu": num_prefill_gpu,
        "decode_num_workers": decode_num_workers,
        **decode.fields("decode_"),
        "decode_dp_attention": decode_dp_attention,
        "num_decode_gpu": num_decode_gpu,
    }
    if prefill_hardware:
        fields["prefill_hw"] = prefill_hardware
        fields["decode_hw"] = decode_hardware
    return fields, num_gpus, tp, ep, dp_attention


def build_result(
    records: list[dict[str, Any]],
    aggregate: dict[str, Any],
    server_metrics: dict[str, Any],
    env: Mapping[str, str],
    *,
    request_accounting: dict[str, Any] | None = None,
    traces: Iterable[dict[str, Any]] = (),
    server_logs: Iterable[str | None] = (),
) -> dict[str, Any]:
    """Build an unrounded AgentX aggregate from explicit inputs.

    Does not read process environment or open files. Inputs are not mutated;
    dataset and request-accounting mappings remain shared with the result.
    Optional traces and logs are consumed once, only when needed. Traces must
    belong to the aggregate's dataset; each log item is one decoded file head.
    Preserve CLI validation order and errors, including SystemExit for invalid
    required metadata. The caller owns serialization and output rounding.
    """
    kv_offloading, kv_offload_backend = _validate_kv_offload_env(env)
    multinode_fields, num_gpus, tp, ep, dp_attention = _gpu_shape(env)
    framework = env.get("FRAMEWORK", "")
    request_accounting = request_accounting or {
        "records_total": len(records),
        "records_profiled": len(records),
        "records_dropped_total": 0,
        "records_warmup_dropped": 0,
        "records_error_dropped": 0,
        "error_categories": {},
    }

    agg: dict[str, Any] = {
        "hw": env.get("RUNNER_TYPE", ""),
        "conc": int(env.get("CONC", "0")),
        "image": env.get("IMAGE", ""),
        "recipe_fingerprint": env.get("RECIPE_FINGERPRINT", ""),
        "model": env.get("MODEL", ""),
        "infmax_model_prefix": env.get("MODEL_PREFIX", ""),
        "framework": framework,
        "precision": env.get("PRECISION", ""),
        "spec_decoding": env.get("SPEC_DECODING", "none"),
        "disagg": _env_bool(env, "DISAGG"),
        "scenario_type": "agentic-coding",
        "is_multinode": _env_bool(env, "IS_MULTINODE"),
        "num_gpus": num_gpus,
        "tp": tp,
        "ep": ep,
        "dp_attention": dp_attention,
        "kv_offloading": kv_offloading,
        "kv_offload_backend": kv_offload_backend,
        "allocated_cpu_dram_gb": _env_int(env, "TOTAL_CPU_DRAM_GB"),
        "num_requests_total": request_accounting["records_total"],
        "num_requests_successful": len(records),
        "request_accounting": request_accounting,
    }
    agg.update(multinode_fields)

    router = _optional_component_metadata(env, "ROUTER_METADATA")
    if router is not None:
        agg["router"] = router

    kv_p2p_transfer = env.get("KV_P2P_TRANSFER")
    if kv_p2p_transfer:
        agg["kv_p2p_transfer"] = kv_p2p_transfer

    metadata = aggregate.get("metadata")
    if isinstance(metadata, dict):
        dataset = metadata.get("dataset")
        if isinstance(dataset, dict):
            agg["dataset"] = dataset

    request_flat, request_nested = compute_request_metrics(records, aggregate, traces=traces)
    _, server_nested, warnings = compute_server_metrics(
        server_metrics,
        framework=framework,
        records=records,
        server_logs=server_logs,
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
