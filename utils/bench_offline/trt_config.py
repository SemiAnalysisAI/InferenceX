"""Fixed configuration for Huawei-style TRT offline benchmarks."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


INPUT_TOKENS = 8192
MTP_DRAFT_TOKENS = 3
HUAWEI_WARMUP_DECODE_ROUNDS = 2
HUAWEI_MEASURED_DECODE_ROUNDS = 256
MAX_OUTPUT_TOKENS_PER_ROUND = MTP_DRAFT_TOKENS + 1
# The measured output cap can require 1024 decode iterations at zero draft
# acceptance, plus prefill. Keep enough TRT history for the complete pass.
ITERATION_STATS_CAPACITY = 2048
# Five decode tokens cannot fit in one MTP3 round, so this guarantees at
# least two warmup rounds while minimizing any extra low-acceptance rounds.
WARMUP_OUTPUT_TOKENS = (
    2
    + (HUAWEI_WARMUP_DECODE_ROUNDS - 1)
    * MAX_OUTPUT_TOKENS_PER_ROUND
)
MEASURED_OUTPUT_TOKENS = (
    1 + HUAWEI_MEASURED_DECODE_ROUNDS * MAX_OUTPUT_TOKENS_PER_ROUND
)
# Huawei allocates 8192 + 256 * 4 + 2 positions for MTP3, then aligns the
# paged-KV capacity to its 128-token block size.
MAX_SEQ_LEN = 9344
# Keep fused-MoE prefill/autotune tensors bounded without changing the
# executor's one-iteration full-batch token budget. Decode is far below this
# fixed cap for every supported GBS.
MOE_MAX_NUM_TOKENS = 65536
# TRT's synthetic engine warmup profiles tunable ops with a max-shape pure
# context request. Cap only that request shape; engine.max_num_tokens must stay
# at runtime capacity because DeepSeek-V4 attention metadata allocates lazily
# from it on the first forward pass.
ENGINE_WARMUP_MAX_TOKENS = 65536
# TRT's capacity estimator does not exercise the 131072-token real prefill.
# Leave room for its 8 GiB FP8 Q projection and 2 GiB BF16 RoPE projection,
# plus allocator headroom, by reducing only the final GBS128 KV budget.
GIB = 1024**3
GBS128_PREFILL_TRANSIENT_RESERVE_BYTES = 12 * GIB
KV_PREFILL_RESERVE_ENV = "TRTLLM_BENCH_KV_PREFILL_RESERVE_BYTES"
MIN_RUNTIME_KV_TOKENS_ENV = "TRTLLM_BENCH_MIN_RUNTIME_KV_TOKENS"
# The pinned TRT MLA context workspace grows by about 192 KiB per scheduled
# context token. Reserve a slightly larger linear budget plus fixed headroom
# before the first forward so the C++ attention op never resizes its live CUDA
# tensor during GBS128 capacity calibration.
ATTENTION_WORKSPACE_BYTES_PER_TOKEN = 200 * 1024
ATTENTION_WORKSPACE_HEADROOM_BYTES = 256 * 1024 * 1024
ATTENTION_WORKSPACE_ENV = "TRTLLM_BENCH_ATTENTION_WORKSPACE_BYTES"
# The pinned fused packed-FP8 quantizer fails its CUDA launch for the
# 65536-row MTP projection used by GBS64 prefill. Keep the fused decode path,
# but select TRT's existing Triton quantizer for larger prefill matrices.
FP8_FUSED_QUANT_MAX_ROWS = 32768
# The pinned Triton-quantize + DeepGemm runner also fails when the real
# GBS128 prefill presents 131072 rows at once. Process only that oversized
# internal GEMM in known-good 65536-row pieces while retaining one executor
# prefill iteration and the existing transformed DeepGemm weights.
FP8_DEEP_GEMM_MAX_ROWS = 65536
FP8_DEEP_GEMM_MAX_ROWS_ENV = (
    "TRTLLM_BENCH_FP8_DEEP_GEMM_MAX_ROWS"
)
SAMPLING_TEMPERATURE = 1.0
SAMPLING_TOP_P = 1.0
SAMPLING_TOP_K = 0
PINNED_TRT_GLOBAL_SEED = 42
ALLOWED_GLOBAL_BATCH_SIZES = (16, 64, 128)
FIXED_BATCH_ARM_ENV = "TRTLLM_BENCH_FIXED_BATCH_ARM_FILE"
FIXED_BATCH_ARM_FILENAME = "fixed_batch_barrier.armed.json"
CONTROLLED_ENVIRONMENT_VARIABLES = {
    "ENABLE_CONFIGURABLE_MOE",
    "TLLM_METRICS_ALL_RANKS",
    "TLLM_PROFILE_LOG_RANKS",
    "TLLM_PROFILE_START_STOP",
    "TLLM_TORCH_PROFILE_TRACE",
    "TRTLLM_BENCH_DSV4_PATCHED_SHA256",
    "TRTLLM_BENCH_ENABLE_CONFIGURABLE_MOE",
    ATTENTION_WORKSPACE_ENV,
    "TRTLLM_BENCH_ENGINE_WARMUP_MAX_TOKENS",
    FP8_DEEP_GEMM_MAX_ROWS_ENV,
    "TRTLLM_BENCH_FP8_FUSED_QUANT_MAX_ROWS",
    FIXED_BATCH_ARM_ENV,
    "TRTLLM_BENCH_FIXED_BATCH_TIMEOUT_SECONDS",
    "TRTLLM_BENCH_GLOBAL_BATCH_SIZE",
    KV_PREFILL_RESERVE_ENV,
    MIN_RUNTIME_KV_TOKENS_ENV,
    "TRTLLM_DSV4_SKIP_PREMOE_ALLREDUCE",
    "TRTLLM_ENABLE_PDL",
    "TRTLLM_FORCE_COMM_METHOD",
    "TRTLLM_GEN_MOE_AUTOTUNE_DUMMY_DISTRIBUTION",
    "TRTLLM_MEGAMOE_FUSED_PREPARE",
    "TRTLLM_MOE_POST_QUANT_ALLTOALLV",
}


@dataclass(frozen=True)
class FixedBenchmarkConfig:
    """One fixed TRT recipe used for every global batch on a profile."""

    name: str
    parallelism: str
    active_gpu_count: int
    tensor_parallel_size: int
    moe_expert_parallel_size: int
    moe_tensor_parallel_size: int = 1
    enable_attention_dp: bool = True
    enable_lm_head_tp_in_adp: bool = True
    batching_wait_iters: int = 0
    attention_dp_balance: bool = True
    overlap_scheduler: bool = False
    cuda_graph: bool = True
    enable_heuristic_topk: bool = True
    moe_backend: str = "TRTLLM"
    moe_load_balancer: str | None = None
    moe_load_balancer_slots: int | None = None
    moe_max_num_tokens: int | None = MOE_MAX_NUM_TOKENS
    use_low_precision_moe_combine: bool = True
    kv_cache_free_gpu_memory_fraction: float = 0.60
    enable_configurable_moe: bool = True
    moe_autotune_dummy_distribution: str = "random"
    print_iter_log: bool = False
    enable_pdl: bool = False
    fp8_fused_quant_max_rows: int | None = FP8_FUSED_QUANT_MAX_ROWS
    fp8_deep_gemm_max_rows: int | None = FP8_DEEP_GEMM_MAX_ROWS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HardwareProfile:
    """Runtime, topology, and renderer identity for one hardware target."""

    key: str
    hardware: str
    renderer_hw: str
    physical_nodes: int
    gpus_per_node: int
    image: str
    trt_source_commit: str
    cache_name: str
    config: FixedBenchmarkConfig
    reference_pr: str | None = None

    @property
    def world_size(self) -> int:
        return self.config.active_gpu_count

    @property
    def is_multinode(self) -> bool:
        return self.physical_nodes > 1


B300_PROFILE = HardwareProfile(
    key="b300",
    hardware="B300",
    renderer_hw="b300",
    physical_nodes=1,
    gpus_per_node=8,
    image=(
        "ghcr.io#semianalysisai/"
        "trtllm-deepseek-v4:feat-deepseek_v4-c185066"
    ),
    trt_source_commit="c185066",
    cache_name="dsv4-c185066-sm100a",
    config=FixedBenchmarkConfig(
        name="huawei-fixed-gbs-dep8",
        parallelism="DEP8",
        active_gpu_count=8,
        tensor_parallel_size=8,
        moe_expert_parallel_size=8,
    ),
)

GB300_PROFILE = HardwareProfile(
    key="gb300",
    hardware="GB300 NVL16",
    renderer_hw="gb300-nv",
    physical_nodes=4,
    gpus_per_node=4,
    image=(
        "nvcr.io#nvidia/ai-dynamo/"
        "tensorrtllm-runtime:1.3.0-deepseek-v4-dev.1"
    ),
    trt_source_commit="34a563ac6d8cc0ca7068c7f619e869fb8a625333",
    cache_name="dsv4-1.3.0-deepseek-v4-dev.1-sm100a",
    config=FixedBenchmarkConfig(
        name="huawei-fixed-gbs-dep16-gb300",
        parallelism="DEP16",
        active_gpu_count=16,
        tensor_parallel_size=16,
        moe_expert_parallel_size=16,
        moe_backend="MEGAMOE_DEEPGEMM",
        moe_load_balancer=(
            "/dsv4-eplb-configs/"
            "moe_load_balancer_gen_ep16_slots384.yaml"
        ),
        moe_load_balancer_slots=384,
        moe_max_num_tokens=None,
        kv_cache_free_gpu_memory_fraction=0.70,
        enable_configurable_moe=False,
        enable_pdl=True,
        fp8_fused_quant_max_rows=None,
        fp8_deep_gemm_max_rows=None,
    ),
    reference_pr="https://github.com/SemiAnalysisAI/InferenceX/pull/1689",
)

HARDWARE_PROFILES = {
    profile.key: profile
    for profile in (B300_PROFILE, GB300_PROFILE)
}
HARDWARE_PROFILE_ENV = "TRT_BENCH_HARDWARE_PROFILE"


def hardware_profile(name: str | None = None) -> HardwareProfile:
    """Resolve one supported hardware profile."""
    key = (name or os.getenv(HARDWARE_PROFILE_ENV, "b300")).lower()
    try:
        return HARDWARE_PROFILES[key]
    except KeyError as error:
        raise ValueError(
            f"Unsupported {HARDWARE_PROFILE_ENV}={key!r}; "
            f"expected one of {tuple(HARDWARE_PROFILES)}"
        ) from error


HARDWARE_PROFILE = hardware_profile()
WORLD_SIZE = HARDWARE_PROFILE.world_size
FIXED_BENCHMARK_CONFIG = HARDWARE_PROFILE.config


def validate_global_batch_size(
    global_batch_size: int,
    profile: HardwareProfile | None = None,
) -> None:
    selected = profile or HARDWARE_PROFILE
    if global_batch_size not in ALLOWED_GLOBAL_BATCH_SIZES:
        raise ValueError(
            "Global batch size must be one of "
            f"{ALLOWED_GLOBAL_BATCH_SIZES}, got {global_batch_size}"
        )
    if global_batch_size % selected.world_size != 0:
        raise ValueError(
            f"Global batch size {global_batch_size} is not divisible by "
            f"the {selected.world_size} attention-DP ranks"
        )


def local_batch_size(
    global_batch_size: int,
    profile: HardwareProfile | None = None,
) -> int:
    selected = profile or HARDWARE_PROFILE
    validate_global_batch_size(global_batch_size, selected)
    return global_batch_size // selected.world_size


def max_num_tokens(
    global_batch_size: int,
    profile: HardwareProfile | None = None,
) -> int:
    """Allow every local-rank prompt to prefill in the same iteration."""
    return local_batch_size(global_batch_size, profile) * INPUT_TOKENS


def attention_workspace_target_bytes(
    global_batch_size: int,
    profile: HardwareProfile | None = None,
) -> int:
    """Return the eager attention workspace reservation for this batch."""
    runtime_tokens = max_num_tokens(global_batch_size, profile)
    if runtime_tokens <= ENGINE_WARMUP_MAX_TOKENS:
        return 0
    return (
        runtime_tokens * ATTENTION_WORKSPACE_BYTES_PER_TOKEN
        + ATTENTION_WORKSPACE_HEADROOM_BYTES
    )


def kv_prefill_reserve_bytes(
    global_batch_size: int,
    profile: HardwareProfile | None = None,
) -> int:
    """Reserve transient memory only when runtime prefill exceeds warmup."""
    if (
        max_num_tokens(global_batch_size, profile)
        <= ENGINE_WARMUP_MAX_TOKENS
    ):
        return 0
    return GBS128_PREFILL_TRANSIENT_RESERVE_BYTES


def minimum_runtime_kv_tokens(
    global_batch_size: int,
    profile: HardwareProfile | None = None,
) -> int:
    """Return the KV capacity required for every fixed-batch sequence."""
    return local_batch_size(global_batch_size, profile) * MAX_SEQ_LEN


def build_llm_kwargs(
    model_path: str,
    global_batch_size: int,
    profile: HardwareProfile | None = None,
) -> dict[str, Any]:
    """Build TRT arguments from one authoritative global batch size."""
    selected = profile or HARDWARE_PROFILE
    config = selected.config
    local_batch = local_batch_size(global_batch_size, selected)
    kwargs: dict[str, Any] = {
        "model": model_path,
        "backend": "pytorch",
        "trust_remote_code": True,
        "tensor_parallel_size": config.tensor_parallel_size,
        "moe_expert_parallel_size": config.moe_expert_parallel_size,
        "moe_tensor_parallel_size": config.moe_tensor_parallel_size,
        "enable_attention_dp": config.enable_attention_dp,
        "enable_lm_head_tp_in_adp": config.enable_lm_head_tp_in_adp,
        # TRT defines max_batch_size per local attention-DP rank.
        "max_batch_size": local_batch,
        # This is intentionally local_batch * 8192. The old harness admitted
        # only one prompt's prefill tokens and therefore staggered the batch.
        "max_num_tokens": max_num_tokens(global_batch_size, selected),
        "max_seq_len": MAX_SEQ_LEN,
        "custom_tokenizer": "deepseek_v4",
        "return_perf_metrics": True,
        "enable_iter_perf_stats": True,
        "max_stats_len": ITERATION_STATS_CAPACITY,
        "print_iter_log": config.print_iter_log,
        "stream_interval": 100,
        "num_postprocess_workers": 4,
        # Huawei synchronizes each decode round before timing the next one.
        "disable_overlap_scheduler": not config.overlap_scheduler,
        "kv_cache_config": {
            "tokens_per_block": 128,
            "dtype": "fp8",
            "free_gpu_memory_fraction": (
                config.kv_cache_free_gpu_memory_fraction
            ),
            "enable_block_reuse": False,
        },
        "moe_config": {
            "backend": config.moe_backend,
            "use_low_precision_moe_combine": (
                config.use_low_precision_moe_combine
            ),
        },
        "sparse_attention_config": {
            "algorithm": "deepseek_v4",
            "enable_heuristic_topk": config.enable_heuristic_topk,
        },
        "speculative_config": {
            "decoding_type": "MTP",
            "max_draft_len": MTP_DRAFT_TOKENS,
        },
        "attention_dp_config": {
            "batching_wait_iters": config.batching_wait_iters,
            "enable_balance": config.attention_dp_balance,
        },
        "cuda_graph_config": {
            "batch_sizes": [local_batch],
            "max_batch_size": local_batch,
            "enable_padding": True,
        },
    }
    if config.moe_load_balancer is not None:
        kwargs["moe_config"]["load_balancer"] = (
            config.moe_load_balancer
        )
    if config.moe_max_num_tokens is not None:
        kwargs["moe_config"]["max_num_tokens"] = (
            config.moe_max_num_tokens
        )
    if selected.is_multinode:
        kwargs["gpus_per_node"] = selected.gpus_per_node
    return kwargs


def fixed_environment(
    global_batch_size: int,
    profile: HardwareProfile | None = None,
) -> dict[str, str]:
    selected = profile or HARDWARE_PROFILE
    validate_global_batch_size(global_batch_size, selected)
    config = selected.config
    configurable = "1" if config.enable_configurable_moe else "0"
    environment = {
        "ENABLE_CONFIGURABLE_MOE": configurable,
        "TRTLLM_BENCH_ENABLE_CONFIGURABLE_MOE": configurable,
        ATTENTION_WORKSPACE_ENV: str(
            attention_workspace_target_bytes(
                global_batch_size,
                selected,
            )
        ),
        "TRTLLM_BENCH_ENGINE_WARMUP_MAX_TOKENS": str(
            ENGINE_WARMUP_MAX_TOKENS
        ),
        FP8_DEEP_GEMM_MAX_ROWS_ENV: str(
            config.fp8_deep_gemm_max_rows or 0
        ),
        "TRTLLM_BENCH_FP8_FUSED_QUANT_MAX_ROWS": str(
            config.fp8_fused_quant_max_rows or 0
        ),
        "TRTLLM_BENCH_FIXED_BATCH_TIMEOUT_SECONDS": "120",
        "TRTLLM_BENCH_GLOBAL_BATCH_SIZE": str(global_batch_size),
        KV_PREFILL_RESERVE_ENV: str(
            kv_prefill_reserve_bytes(global_batch_size, selected)
        ),
        MIN_RUNTIME_KV_TOKENS_ENV: str(
            minimum_runtime_kv_tokens(global_batch_size, selected)
        ),
        "TRTLLM_GEN_MOE_AUTOTUNE_DUMMY_DISTRIBUTION": (
            config.moe_autotune_dummy_distribution
        ),
    }
    if config.enable_pdl:
        environment["TRTLLM_ENABLE_PDL"] = "1"
    return environment


def benchmark_environment(
    global_batch_size: int,
    fixed_batch_arm_file: str | Path,
    profile: HardwareProfile | None = None,
) -> dict[str, str]:
    """Build the complete environment inherited by every TRT rank."""
    arm_path = Path(fixed_batch_arm_file)
    if not arm_path.is_absolute():
        raise ValueError(
            "Fixed-batch barrier arm file must use an absolute path"
        )
    return {
        **fixed_environment(global_batch_size, profile),
        FIXED_BATCH_ARM_ENV: str(arm_path),
    }


def external_mpi_rank_environment(
    global_batch_size: int,
    fixed_batch_arm_file: str | Path,
    marker_file: str | Path,
    cute_cache_dir: str | Path,
    profile: HardwareProfile | None = None,
) -> dict[str, str]:
    """Build the environment required before external MPI ranks start."""
    marker_path = Path(marker_file)
    cache_path = Path(cute_cache_dir)
    for label, path in (
        ("Perfect-router marker", marker_path),
        ("CuTe cache directory", cache_path),
    ):
        if not path.is_absolute():
            raise ValueError(f"{label} must use an absolute path")

    configured = benchmark_environment(
        global_batch_size,
        fixed_batch_arm_file,
        profile,
    )
    return {
        **configured,
        "ENABLE_PERFECT_ROUTER": "1",
        "TRTLLM_ENABLE_PERFECT_ROUTER": "1",
        "TRTLLM_PERFECT_ROUTER_MARKER": str(marker_path),
        "CUTE_DSL_CACHE_DIR": str(cache_path),
        "TRTLLM_BENCH_CUTE_DSL_CACHE_DIR": str(cache_path),
        "TRTLLM_BENCH_EXPECTED_RANK_ENV": json.dumps(
            configured,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def resolved_parallelism(
    llm_args: Any,
    global_batch_size: int,
    profile: HardwareProfile | None = None,
) -> dict[str, Any]:
    """Validate that TRT resolved the fixed recipe without silent changes."""
    selected = profile or HARDWARE_PROFILE
    config = selected.config
    parallel = llm_args.parallel_config
    kv_cache = llm_args.kv_cache_config
    resolved = {
        "world_size": int(parallel.world_size),
        "tensor_parallel_size": int(parallel.tp_size),
        "moe_expert_parallel_size": int(parallel.moe_ep_size),
        "moe_tensor_parallel_size": int(parallel.moe_tp_size),
        "enable_attention_dp": bool(parallel.enable_attention_dp),
        "enable_lm_head_tp_in_adp": bool(
            parallel.enable_lm_head_tp_in_adp
        ),
        "effective_parallelism": config.parallelism,
        "global_batch_size": global_batch_size,
        "local_batch_size": local_batch_size(
            global_batch_size,
            selected,
        ),
        "max_batch_size": int(llm_args.max_batch_size),
        "max_num_tokens": int(llm_args.max_num_tokens),
        "max_seq_len": int(llm_args.max_seq_len),
        "kv_cache_free_gpu_memory_fraction": float(
            kv_cache.free_gpu_memory_fraction
        ),
        "enable_iter_perf_stats": bool(llm_args.enable_iter_perf_stats),
        "max_stats_len": int(llm_args.max_stats_len),
        "print_iter_log": bool(llm_args.print_iter_log),
        "disable_overlap_scheduler": bool(
            llm_args.disable_overlap_scheduler
        ),
    }
    expected = {
        "world_size": config.active_gpu_count,
        "tensor_parallel_size": config.tensor_parallel_size,
        "moe_expert_parallel_size": config.moe_expert_parallel_size,
        "moe_tensor_parallel_size": config.moe_tensor_parallel_size,
        "enable_attention_dp": config.enable_attention_dp,
        "enable_lm_head_tp_in_adp": config.enable_lm_head_tp_in_adp,
        "max_batch_size": local_batch_size(global_batch_size, selected),
        "max_num_tokens": max_num_tokens(global_batch_size, selected),
        "max_seq_len": MAX_SEQ_LEN,
        "kv_cache_free_gpu_memory_fraction": (
            config.kv_cache_free_gpu_memory_fraction
        ),
        "enable_iter_perf_stats": True,
        "max_stats_len": ITERATION_STATS_CAPACITY,
        "print_iter_log": config.print_iter_log,
        "disable_overlap_scheduler": True,
    }
    for key, expected_value in expected.items():
        if resolved[key] != expected_value:
            raise RuntimeError(
                f"Resolved TRT setting mismatch for {key}: "
                f"{resolved[key]!r} != {expected_value!r}"
            )

    speculative = llm_args.speculative_config
    resolved["mtp_max_draft_len"] = int(speculative.max_draft_len)
    if resolved["mtp_max_draft_len"] != MTP_DRAFT_TOKENS:
        raise RuntimeError(
            "Resolved TRT MTP draft length mismatch: "
            f"{resolved['mtp_max_draft_len']} != {MTP_DRAFT_TOKENS}"
        )

    moe = llm_args.moe_config
    resolved["moe_backend"] = str(
        getattr(moe.backend, "value", moe.backend)
    )
    resolved["moe_max_num_tokens"] = (
        int(moe.max_num_tokens)
        if moe.max_num_tokens is not None
        else None
    )
    resolved["use_low_precision_moe_combine"] = bool(
        moe.use_low_precision_moe_combine
    )
    if resolved["moe_backend"] != config.moe_backend:
        raise RuntimeError(
            f"Resolved TRT MoE backend {resolved['moe_backend']!r} != "
            f"{config.moe_backend!r}"
        )
    if resolved["moe_max_num_tokens"] != config.moe_max_num_tokens:
        raise RuntimeError(
            "Resolved TRT MoE token cap mismatch: "
            f"{resolved['moe_max_num_tokens']} != "
            f"{config.moe_max_num_tokens}"
        )
    if (
        resolved["use_low_precision_moe_combine"]
        != config.use_low_precision_moe_combine
    ):
        raise RuntimeError("Resolved TRT low-precision MoE combine mismatch")
    if config.moe_load_balancer is not None:
        load_balancer = moe.load_balancer
        resolved["moe_load_balancer"] = str(load_balancer)
        resolved_slots = getattr(load_balancer, "num_slots", None)
        resolved["moe_load_balancer_slots"] = (
            int(resolved_slots)
            if resolved_slots is not None
            else None
        )
        path_is_unresolved = isinstance(load_balancer, str)
        if path_is_unresolved and load_balancer != config.moe_load_balancer:
            raise RuntimeError(
                "Resolved TRT MoE load balancer mismatch: "
                f"{resolved['moe_load_balancer']!r} != "
                f"{config.moe_load_balancer!r}"
            )
        if (
            not path_is_unresolved
            and config.moe_load_balancer_slots is not None
            and resolved["moe_load_balancer_slots"]
            != config.moe_load_balancer_slots
        ):
            raise RuntimeError(
                "Resolved TRT MoE load-balancer slot mismatch: "
                f"{resolved['moe_load_balancer_slots']} != "
                f"{config.moe_load_balancer_slots}"
            )

    sparse = llm_args.sparse_attention_config
    resolved["enable_heuristic_topk"] = bool(
        sparse.enable_heuristic_topk
    )
    if resolved["enable_heuristic_topk"] != config.enable_heuristic_topk:
        raise RuntimeError("Resolved TRT heuristic top-k mismatch")

    graph = llm_args.cuda_graph_config
    if graph is None:
        raise RuntimeError("Resolved TRT CUDA graph config is disabled")
    graph_batch_sizes = (
        graph["batch_sizes"] if isinstance(graph, dict) else graph.batch_sizes
    )
    graph_max_batch_size = (
        graph["max_batch_size"]
        if isinstance(graph, dict)
        else graph.max_batch_size
    )
    resolved["cuda_graph_batch_sizes"] = [
        int(value) for value in graph_batch_sizes
    ]
    resolved["cuda_graph_max_batch_size"] = int(graph_max_batch_size)
    expected_graph = [local_batch_size(global_batch_size, selected)]
    if resolved["cuda_graph_batch_sizes"] != expected_graph:
        raise RuntimeError(
            "Resolved TRT CUDA graph batch sizes mismatch: "
            f"{resolved['cuda_graph_batch_sizes']} != {expected_graph}"
        )
    if resolved["cuda_graph_max_batch_size"] != expected_graph[0]:
        raise RuntimeError(
            "Resolved TRT CUDA graph max batch size mismatch: "
            f"{resolved['cuda_graph_max_batch_size']} != "
            f"{expected_graph[0]}"
        )
    if selected.is_multinode:
        resolved["gpus_per_node"] = int(llm_args.gpus_per_node)
        if resolved["gpus_per_node"] != selected.gpus_per_node:
            raise RuntimeError(
                "Resolved TRT GPUs per node mismatch: "
                f"{resolved['gpus_per_node']} != "
                f"{selected.gpus_per_node}"
            )
    return resolved
