"""Fixed configuration for Huawei-style TRT offline benchmarks."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, replace
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
# GB300 GBS128 reached the full 65536-token runtime shape during synthetic
# warmup, then exhausted memory while TRT created temporary KV-estimation
# resources. GBS64 proves the same image can initialize the 32768-token tuning
# shape. Limit only the synthetic pure-context request; runtime capacity and
# the real fixed-batch prefill remain 65536 tokens.
GB300_GBS128_ENGINE_WARMUP_MAX_TOKENS = 32768
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
# The pinned fused packed-FP8 quantizer fails its CUDA launch for 65536-row
# MTP projections used by B300 GBS64 and GB300 GBS128. Keep the fused decode
# path, but select TRT's existing Triton quantizer for larger prefill matrices.
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
PR_TP32_GRAPH_BATCH_SIZES = (1, 2, 4, 8)
PR_TP16_GRAPH_BATCH_SIZES = (1, 2, 4, 8, 16, 24, 32)
PR_TP8_GRAPH_BATCH_SIZES = (
    1,
    2,
    4,
    8,
    16,
    24,
    32,
    *range(40, 513, 8),
)
RACK_TP8_ENGINE_GLOBAL_BATCH_SIZES = (8, 32, 64, 3440, 4096)
FIXED_BATCH_ARM_ENV = "TRTLLM_BENCH_FIXED_BATCH_ARM_FILE"
FIXED_BATCH_ARM_FILENAME = "fixed_batch_barrier.armed.json"
BENCHMARK_PROFILE_ENV = "TRT_BENCH_CONFIG_PROFILE"
CONTROLLED_ENVIRONMENT_VARIABLES = {
    "ENABLE_CONFIGURABLE_MOE",
    "ENABLE_PERFECT_ROUTER",
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
    BENCHMARK_PROFILE_ENV,
    KV_PREFILL_RESERVE_ENV,
    MIN_RUNTIME_KV_TOKENS_ENV,
    "TRTLLM_DSV4_SKIP_PREMOE_ALLREDUCE",
    "TRTLLM_ENABLE_PDL",
    "TRTLLM_FORCE_COMM_METHOD",
    "TRTLLM_GEN_MOE_AUTOTUNE_DUMMY_DISTRIBUTION",
    "TRTLLM_MEGAMOE_FUSED_PREPARE",
    "TRTLLM_MOE_POST_QUANT_ALLTOALLV",
    "TRTLLM_ENABLE_PERFECT_ROUTER",
}


@dataclass(frozen=True)
class FixedBenchmarkConfig:
    """One fixed TRT recipe used for every global batch on a profile."""

    name: str
    parallelism: str
    active_gpu_count: int
    tensor_parallel_size: int
    moe_expert_parallel_size: int
    profile_key: str = "huawei"
    benchmark_mode: str = "huawei"
    allowed_global_batch_sizes: tuple[int, ...] = (
        ALLOWED_GLOBAL_BATCH_SIZES
    )
    mtp_draft_tokens: int = MTP_DRAFT_TOKENS
    max_seq_len: int = MAX_SEQ_LEN
    measured_decode_rounds: int = HUAWEI_MEASURED_DECODE_ROUNDS
    warmup_decode_rounds: int = HUAWEI_WARMUP_DECODE_ROUNDS
    run_request_warmup: bool = True
    latency_rounds_to_skip: int = 1
    timing_source: str = "iter_latency_ms"
    collect_request_perf_metrics: bool = True
    engine_max_batch_size: int | None = None
    runtime_max_num_tokens: int | None = None
    reference_decode_max_num_tokens: int | None = None
    cuda_graph_batch_sizes: tuple[int, ...] | None = None
    enable_sparse_attention: bool = True
    fixed_batch_timeout_seconds: int = 120
    reference_concurrency: int | None = None
    reference_active_global_batch: int | None = None
    reference_prefill_gpu_count: int = 0
    reference_output_tput_per_decode_gpu: float | None = None
    reference_output_tput_per_total_gpu: float | None = None
    reference_recipe_url: str | None = None
    moe_tensor_parallel_size: int = 1
    enable_attention_dp: bool = True
    enable_lm_head_tp_in_adp: bool = True
    batching_wait_iters: int | None = 0
    attention_dp_balance: bool | None = True
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
    enable_perfect_router: bool = True
    moe_autotune_dummy_distribution: str | None = "random"
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
        enable_configurable_moe=True,
        enable_pdl=True,
        fp8_fused_quant_max_rows=FP8_FUSED_QUANT_MAX_ROWS,
        fp8_deep_gemm_max_rows=None,
    ),
    reference_pr="https://github.com/SemiAnalysisAI/InferenceX/pull/1689",
)

GB300_HUAWEI_CONFIG = GB300_PROFILE.config
GB300_PR_CONFIGS = {
    "pr-tp32-mtp3": FixedBenchmarkConfig(
        name="pr-1689-tp32-ep32-batch8-mtp3",
        profile_key="pr-tp32-mtp3",
        benchmark_mode="pr_max_decode",
        parallelism="DEP32",
        active_gpu_count=32,
        tensor_parallel_size=32,
        moe_expert_parallel_size=32,
        allowed_global_batch_sizes=(192, 256),
        mtp_draft_tokens=3,
        max_seq_len=9256,
        warmup_decode_rounds=0,
        run_request_warmup=False,
        latency_rounds_to_skip=8,
        timing_source="trt_print_iter_log_host_step_time",
        collect_request_perf_metrics=False,
        engine_max_batch_size=8,
        runtime_max_num_tokens=32768,
        reference_decode_max_num_tokens=32,
        cuda_graph_batch_sizes=PR_TP32_GRAPH_BATCH_SIZES,
        enable_sparse_attention=False,
        fixed_batch_timeout_seconds=600,
        reference_concurrency=333,
        reference_active_global_batch=192,
        reference_prefill_gpu_count=16,
        reference_output_tput_per_decode_gpu=676.763,
        reference_output_tput_per_total_gpu=451.175,
        reference_recipe_url=(
            "https://github.com/NVIDIA/srt-slurm/blob/"
            "sa-submission-q2-2026/recipes/DeepSeek-V4-Pro/disagg/"
            "trtllm_dynamo/gb300_mxfp4/ISL8K_OSL1K/MTP/"
            "ctx4dep4_gen1dep32_batch8_eplb384_mtp3.yaml"
        ),
        overlap_scheduler=True,
        batching_wait_iters=None,
        attention_dp_balance=None,
        cuda_graph=True,
        enable_heuristic_topk=False,
        moe_backend="MEGAMOE_DEEPGEMM",
        moe_load_balancer=(
            "/dsv4-eplb-configs/"
            "moe_load_balancer_gen_ep32_slots384.yaml"
        ),
        moe_load_balancer_slots=384,
        moe_max_num_tokens=None,
        kv_cache_free_gpu_memory_fraction=0.70,
        enable_configurable_moe=True,
        enable_perfect_router=False,
        moe_autotune_dummy_distribution=None,
        print_iter_log=True,
        enable_pdl=True,
        fp8_fused_quant_max_rows=None,
        fp8_deep_gemm_max_rows=None,
    ),
    "pr-tp16-mtp3": FixedBenchmarkConfig(
        name="pr-1689-tp16-ep16-batch32-mtp3",
        profile_key="pr-tp16-mtp3",
        benchmark_mode="pr_max_decode",
        parallelism="DEP16",
        active_gpu_count=16,
        tensor_parallel_size=16,
        moe_expert_parallel_size=16,
        allowed_global_batch_sizes=(400, 512),
        mtp_draft_tokens=3,
        max_seq_len=9256,
        warmup_decode_rounds=0,
        run_request_warmup=False,
        latency_rounds_to_skip=8,
        timing_source="trt_print_iter_log_host_step_time",
        collect_request_perf_metrics=False,
        engine_max_batch_size=32,
        runtime_max_num_tokens=32768,
        reference_decode_max_num_tokens=128,
        cuda_graph_batch_sizes=PR_TP16_GRAPH_BATCH_SIZES,
        enable_sparse_attention=False,
        fixed_batch_timeout_seconds=600,
        reference_concurrency=666,
        reference_active_global_batch=400,
        reference_prefill_gpu_count=24,
        reference_output_tput_per_decode_gpu=2072.1585,
        reference_output_tput_per_total_gpu=828.8634,
        reference_recipe_url=(
            "https://github.com/NVIDIA/srt-slurm/blob/"
            "sa-submission-q2-2026/recipes/DeepSeek-V4-Pro/disagg/"
            "trtllm_dynamo/gb300_mxfp4/ISL8K_OSL1K/MTP/"
            "ctx6dep4_gen1dep16_batch32_eplb384_mtp3.yaml"
        ),
        overlap_scheduler=True,
        batching_wait_iters=None,
        attention_dp_balance=None,
        cuda_graph=True,
        enable_heuristic_topk=False,
        moe_backend="MEGAMOE_DEEPGEMM",
        moe_load_balancer=(
            "/dsv4-eplb-configs/"
            "moe_load_balancer_gen_ep16_slots384.yaml"
        ),
        moe_load_balancer_slots=384,
        moe_max_num_tokens=None,
        kv_cache_free_gpu_memory_fraction=0.70,
        enable_configurable_moe=True,
        enable_perfect_router=False,
        moe_autotune_dummy_distribution=None,
        print_iter_log=True,
        enable_pdl=True,
        fp8_fused_quant_max_rows=None,
        fp8_deep_gemm_max_rows=None,
    ),
    "pr-tp8-mtp1": FixedBenchmarkConfig(
        name="pr-1689-tp8-ep8-batch512-mtp1",
        profile_key="pr-tp8-mtp1",
        benchmark_mode="pr_max_decode",
        parallelism="DEP8",
        active_gpu_count=8,
        tensor_parallel_size=8,
        moe_expert_parallel_size=8,
        allowed_global_batch_sizes=(3440, 4096),
        mtp_draft_tokens=1,
        max_seq_len=9256,
        warmup_decode_rounds=0,
        run_request_warmup=False,
        latency_rounds_to_skip=8,
        timing_source="trt_print_iter_log_host_step_time",
        collect_request_perf_metrics=False,
        engine_max_batch_size=512,
        runtime_max_num_tokens=32768,
        reference_decode_max_num_tokens=1024,
        cuda_graph_batch_sizes=PR_TP8_GRAPH_BATCH_SIZES,
        enable_sparse_attention=False,
        fixed_batch_timeout_seconds=600,
        reference_concurrency=4301,
        reference_active_global_batch=3440,
        reference_prefill_gpu_count=48,
        reference_output_tput_per_decode_gpu=9686.735,
        reference_output_tput_per_total_gpu=1383.8193,
        reference_recipe_url=(
            "https://github.com/NVIDIA/srt-slurm/blob/"
            "sa-submission-q2-2026/recipes/DeepSeek-V4-Pro/disagg/"
            "trtllm_dynamo/gb300_mxfp4/ISL8K_OSL1K/MTP/"
            "ctx12dep4_gen1dep8_batch512_eplb384_mtp1.yaml"
        ),
        overlap_scheduler=True,
        batching_wait_iters=None,
        attention_dp_balance=None,
        cuda_graph=True,
        enable_heuristic_topk=False,
        moe_backend="MEGAMOE_DEEPGEMM",
        moe_load_balancer=(
            "/dsv4-eplb-configs/"
            "moe_load_balancer_gen_ep8_slots384.yaml"
        ),
        moe_load_balancer_slots=384,
        moe_max_num_tokens=None,
        kv_cache_free_gpu_memory_fraction=0.80,
        enable_configurable_moe=True,
        enable_perfect_router=False,
        moe_autotune_dummy_distribution=None,
        print_iter_log=True,
        enable_pdl=True,
        fp8_fused_quant_max_rows=None,
        fp8_deep_gemm_max_rows=None,
    ),
}
GB300_RACK_ENGINE_CONFIG = replace(
    GB300_PR_CONFIGS["pr-tp8-mtp1"],
    name="pr-1689-rack-replica-tp8-ep8-batch512-mtp1",
    profile_key="rack-tp8-mtp1-engine",
    allowed_global_batch_sizes=RACK_TP8_ENGINE_GLOBAL_BATCH_SIZES,
)
GB300_PR_CONFIGS[GB300_RACK_ENGINE_CONFIG.profile_key] = (
    GB300_RACK_ENGINE_CONFIG
)
BENCHMARK_PROFILES = {
    "huawei": GB300_HUAWEI_CONFIG,
    **GB300_PR_CONFIGS,
}


def benchmark_profile_key(
    profile: HardwareProfile | None = None,
) -> str:
    selected = profile or HARDWARE_PROFILE
    return selected.config.profile_key

HARDWARE_PROFILES = {
    profile.key: profile
    for profile in (B300_PROFILE, GB300_PROFILE)
}
HARDWARE_PROFILE_ENV = "TRT_BENCH_HARDWARE_PROFILE"


def hardware_profile(
    name: str | None = None,
    benchmark_profile_name: str | None = None,
) -> HardwareProfile:
    """Resolve one supported hardware profile."""
    key = (name or os.getenv(HARDWARE_PROFILE_ENV, "b300")).lower()
    try:
        base = HARDWARE_PROFILES[key]
    except KeyError as error:
        raise ValueError(
            f"Unsupported {HARDWARE_PROFILE_ENV}={key!r}; "
            f"expected one of {tuple(HARDWARE_PROFILES)}"
        ) from error
    config_key = (
        benchmark_profile_name
        or os.getenv(BENCHMARK_PROFILE_ENV, "huawei")
    ).lower()
    if config_key == "huawei":
        return base
    if key != "gb300":
        raise ValueError(
            f"{BENCHMARK_PROFILE_ENV}={config_key!r} requires gb300"
        )
    try:
        config = GB300_PR_CONFIGS[config_key]
    except KeyError as error:
        raise ValueError(
            f"Unsupported {BENCHMARK_PROFILE_ENV}={config_key!r}; "
            f"expected one of {tuple(BENCHMARK_PROFILES)}"
        ) from error
    physical_nodes = math.ceil(
        config.active_gpu_count / base.gpus_per_node
    )
    return replace(
        base,
        hardware=f"GB300 NVL{config.active_gpu_count}",
        physical_nodes=physical_nodes,
        config=config,
    )


HARDWARE_PROFILE = hardware_profile()
WORLD_SIZE = HARDWARE_PROFILE.world_size
FIXED_BENCHMARK_CONFIG = HARDWARE_PROFILE.config


def validate_global_batch_size(
    global_batch_size: int,
    profile: HardwareProfile | None = None,
) -> None:
    selected = profile or HARDWARE_PROFILE
    allowed = selected.config.allowed_global_batch_sizes
    if global_batch_size not in allowed:
        raise ValueError(
            "Global batch size must be one of "
            f"{allowed}, got {global_batch_size}"
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


def engine_max_batch_size(
    global_batch_size: int,
    profile: HardwareProfile | None = None,
) -> int:
    selected = profile or HARDWARE_PROFILE
    validate_global_batch_size(global_batch_size, selected)
    configured = selected.config.engine_max_batch_size
    if configured is not None:
        return configured
    return local_batch_size(global_batch_size, selected)


def max_num_tokens(
    global_batch_size: int,
    profile: HardwareProfile | None = None,
) -> int:
    """Return the monolithic offline prefill/decode token budget."""
    selected = profile or HARDWARE_PROFILE
    configured = selected.config.runtime_max_num_tokens
    if configured is not None:
        return configured
    return local_batch_size(global_batch_size, selected) * INPUT_TOKENS


def setup_prefill_iterations(
    global_batch_size: int,
    profile: HardwareProfile | None = None,
) -> int:
    """Conservative number of iterations needed to admit every 8K prompt."""
    selected = profile or HARDWARE_PROFILE
    local_tokens = (
        local_batch_size(global_batch_size, selected) * INPUT_TOKENS
    )
    return math.ceil(
        local_tokens / max_num_tokens(global_batch_size, selected)
    )


def warmup_output_tokens(
    global_batch_size: int,
    profile: HardwareProfile | None = None,
) -> int:
    selected = profile or HARDWARE_PROFILE
    if not selected.config.run_request_warmup:
        return 0
    return WARMUP_OUTPUT_TOKENS


def measured_output_tokens(
    global_batch_size: int,
    profile: HardwareProfile | None = None,
) -> int:
    selected = profile or HARDWARE_PROFILE
    config = selected.config
    if config.benchmark_mode == "huawei":
        return MEASURED_OUTPUT_TOKENS
    per_round = config.mtp_draft_tokens + 1
    output_tokens = 1 + (
        setup_prefill_iterations(global_batch_size, selected)
        - 1
        + config.measured_decode_rounds
    ) * per_round
    available = config.max_seq_len - INPUT_TOKENS
    if output_tokens > available:
        raise ValueError(
            f"Profile {config.name} needs {output_tokens} output tokens "
            f"to preserve its decode window, but max_seq_len leaves "
            f"only {available}"
        )
    return max(1024, output_tokens)


def engine_warmup_max_tokens(
    global_batch_size: int,
    profile: HardwareProfile | None = None,
) -> int:
    """Bound TRT's synthetic context-only tuning shape."""
    selected = profile or HARDWARE_PROFILE
    validate_global_batch_size(global_batch_size, selected)
    if (
        selected.config.benchmark_mode == "huawei"
        and selected.key == "gb300"
        and global_batch_size == 128
    ):
        return GB300_GBS128_ENGINE_WARMUP_MAX_TOKENS
    if selected.config.benchmark_mode != "huawei":
        return max_num_tokens(global_batch_size, selected)
    return ENGINE_WARMUP_MAX_TOKENS


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
    selected = profile or HARDWARE_PROFILE
    return (
        local_batch_size(global_batch_size, selected)
        * selected.config.max_seq_len
    )


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
        # TRT defines max_batch_size per local attention-DP rank. PR-active
        # points retain the copied engine capacity above their active batch.
        "max_batch_size": engine_max_batch_size(
            global_batch_size,
            selected,
        ),
        # Huawei admits every local 8K prompt together. PR-max intentionally
        # uses its recorded 32K monolithic-offline adaptation.
        "max_num_tokens": max_num_tokens(global_batch_size, selected),
        "max_seq_len": config.max_seq_len,
        "custom_tokenizer": "deepseek_v4",
        "return_perf_metrics": config.collect_request_perf_metrics,
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
        "speculative_config": {
            "decoding_type": "MTP",
            "max_draft_len": config.mtp_draft_tokens,
        },
        "cuda_graph_config": {
            "batch_sizes": list(
                config.cuda_graph_batch_sizes or (local_batch,)
            ),
            "enable_padding": True,
        },
    }
    if (
        config.batching_wait_iters is None
        or config.attention_dp_balance is None
    ):
        if not (
            config.batching_wait_iters is None
            and config.attention_dp_balance is None
        ):
            raise ValueError(
                "attention-DP batching and balance must both be set or "
                "both be omitted"
            )
    else:
        kwargs["attention_dp_config"] = {
            "batching_wait_iters": config.batching_wait_iters,
            "enable_balance": config.attention_dp_balance,
        }
    if config.enable_sparse_attention:
        kwargs["sparse_attention_config"] = {
            "algorithm": "deepseek_v4",
            "enable_heuristic_topk": config.enable_heuristic_topk,
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
            engine_warmup_max_tokens(global_batch_size, selected)
        ),
        FP8_DEEP_GEMM_MAX_ROWS_ENV: str(
            config.fp8_deep_gemm_max_rows or 0
        ),
        "TRTLLM_BENCH_FP8_FUSED_QUANT_MAX_ROWS": str(
            config.fp8_fused_quant_max_rows or 0
        ),
        "TRTLLM_BENCH_FIXED_BATCH_TIMEOUT_SECONDS": str(
            config.fixed_batch_timeout_seconds
        ),
        "TRTLLM_BENCH_GLOBAL_BATCH_SIZE": str(global_batch_size),
        BENCHMARK_PROFILE_ENV: benchmark_profile_key(selected),
        KV_PREFILL_RESERVE_ENV: str(
            kv_prefill_reserve_bytes(global_batch_size, selected)
        ),
        MIN_RUNTIME_KV_TOKENS_ENV: str(
            minimum_runtime_kv_tokens(global_batch_size, selected)
        ),
    }
    if config.moe_autotune_dummy_distribution is not None:
        environment[
            "TRTLLM_GEN_MOE_AUTOTUNE_DUMMY_DISTRIBUTION"
        ] = config.moe_autotune_dummy_distribution
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
    selected = profile or HARDWARE_PROFILE
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
        selected,
    )
    environment = {
        **configured,
        "TRTLLM_PERFECT_ROUTER_MARKER": str(marker_path),
        "CUTE_DSL_CACHE_DIR": str(cache_path),
        "TRTLLM_BENCH_CUTE_DSL_CACHE_DIR": str(cache_path),
        "TRTLLM_BENCH_EXPECTED_RANK_ENV": json.dumps(
            configured,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }
    if selected.config.enable_perfect_router:
        environment.update(
            {
                "ENABLE_PERFECT_ROUTER": "1",
                "TRTLLM_ENABLE_PERFECT_ROUTER": "1",
            }
        )
    return environment


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
        "max_batch_size": engine_max_batch_size(
            global_batch_size,
            selected,
        ),
        "max_num_tokens": max_num_tokens(global_batch_size, selected),
        "max_seq_len": config.max_seq_len,
        "kv_cache_free_gpu_memory_fraction": (
            config.kv_cache_free_gpu_memory_fraction
        ),
        "enable_iter_perf_stats": True,
        "max_stats_len": ITERATION_STATS_CAPACITY,
        "print_iter_log": config.print_iter_log,
        "disable_overlap_scheduler": not config.overlap_scheduler,
    }
    for key, expected_value in expected.items():
        if resolved[key] != expected_value:
            raise RuntimeError(
                f"Resolved TRT setting mismatch for {key}: "
                f"{resolved[key]!r} != {expected_value!r}"
            )

    attention_dp = llm_args.attention_dp_config
    resolved["attention_dp_configured"] = attention_dp is not None
    if config.batching_wait_iters is None:
        if attention_dp is not None:
            raise RuntimeError(
                "Resolved TRT attention-DP config should use the PR "
                "default None"
            )
        resolved["attention_dp_batching_wait_iters"] = None
        resolved["attention_dp_enable_balance"] = None
    else:
        if attention_dp is None:
            raise RuntimeError(
                "Resolved TRT attention-DP config is unexpectedly absent"
            )
        resolved["attention_dp_batching_wait_iters"] = int(
            attention_dp.batching_wait_iters
        )
        resolved["attention_dp_enable_balance"] = bool(
            attention_dp.enable_balance
        )
        if (
            resolved["attention_dp_batching_wait_iters"]
            != config.batching_wait_iters
        ):
            raise RuntimeError(
                "Resolved TRT attention-DP batching wait mismatch: "
                f"{resolved['attention_dp_batching_wait_iters']} != "
                f"{config.batching_wait_iters}"
            )
        if (
            resolved["attention_dp_enable_balance"]
            != config.attention_dp_balance
        ):
            raise RuntimeError(
                "Resolved TRT attention-DP balance mismatch: "
                f"{resolved['attention_dp_enable_balance']} != "
                f"{config.attention_dp_balance}"
            )

    speculative = llm_args.speculative_config
    resolved["mtp_max_draft_len"] = int(speculative.max_draft_len)
    if resolved["mtp_max_draft_len"] != config.mtp_draft_tokens:
        raise RuntimeError(
            "Resolved TRT MTP draft length mismatch: "
            f"{resolved['mtp_max_draft_len']} != "
            f"{config.mtp_draft_tokens}"
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
    resolved["enable_sparse_attention"] = sparse is not None
    if resolved["enable_sparse_attention"] != config.enable_sparse_attention:
        raise RuntimeError("Resolved TRT sparse-attention mode mismatch")
    if sparse is not None:
        resolved["enable_heuristic_topk"] = bool(
            sparse.enable_heuristic_topk
        )
        if (
            resolved["enable_heuristic_topk"]
            != config.enable_heuristic_topk
        ):
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
    expected_graph = list(
        config.cuda_graph_batch_sizes
        or (local_batch_size(global_batch_size, selected),)
    )
    if resolved["cuda_graph_batch_sizes"] != expected_graph:
        raise RuntimeError(
            "Resolved TRT CUDA graph batch sizes mismatch: "
            f"{resolved['cuda_graph_batch_sizes']} != {expected_graph}"
        )
    if resolved["cuda_graph_max_batch_size"] != max(expected_graph):
        raise RuntimeError(
            "Resolved TRT CUDA graph max batch size mismatch: "
            f"{resolved['cuda_graph_max_batch_size']} != "
            f"{max(expected_graph)}"
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
