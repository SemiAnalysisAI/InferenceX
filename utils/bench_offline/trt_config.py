"""Fixed configuration for the Huawei-style B300 TRT offline benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


WORLD_SIZE = 8
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
MOE_MAX_NUM_TOKENS = WORLD_SIZE * INPUT_TOKENS
# TRT's synthetic engine warmup profiles every tunable op at max_num_tokens.
# GBS128 would therefore spend hours profiling a 131072-token prefill shape
# that never enters the decode metric. Real warmup and measured generations
# retain the full runtime token limit and validate the complete fixed batch.
ENGINE_WARMUP_MAX_TOKENS = WORLD_SIZE * INPUT_TOKENS
# The pinned fused packed-FP8 quantizer fails its CUDA launch for the
# 65536-row MTP projection used by GBS64 prefill. Keep the fused decode path,
# but select TRT's existing Triton quantizer for larger prefill matrices.
FP8_FUSED_QUANT_MAX_ROWS = 32768
SAMPLING_TEMPERATURE = 1.0
SAMPLING_TOP_P = 1.0
SAMPLING_TOP_K = 0
PINNED_TRT_GLOBAL_SEED = 42
ALLOWED_GLOBAL_BATCH_SIZES = (16, 64, 128)
CONTROLLED_ENVIRONMENT_VARIABLES = {
    "ENABLE_CONFIGURABLE_MOE",
    "TLLM_METRICS_ALL_RANKS",
    "TLLM_PROFILE_LOG_RANKS",
    "TLLM_PROFILE_START_STOP",
    "TLLM_TORCH_PROFILE_TRACE",
    "TRTLLM_BENCH_DSV4_PATCHED_SHA256",
    "TRTLLM_BENCH_ENABLE_CONFIGURABLE_MOE",
    "TRTLLM_BENCH_ENGINE_WARMUP_MAX_TOKENS",
    "TRTLLM_BENCH_FP8_FUSED_QUANT_MAX_ROWS",
    "TRTLLM_BENCH_FIXED_BATCH_TIMEOUT_SECONDS",
    "TRTLLM_BENCH_GLOBAL_BATCH_SIZE",
    "TRTLLM_DSV4_SKIP_PREMOE_ALLREDUCE",
    "TRTLLM_ENABLE_PDL",
    "TRTLLM_FORCE_COMM_METHOD",
    "TRTLLM_GEN_MOE_AUTOTUNE_DUMMY_DISTRIBUTION",
    "TRTLLM_MEGAMOE_FUSED_PREPARE",
    "TRTLLM_MOE_POST_QUANT_ALLTOALLV",
}


@dataclass(frozen=True)
class FixedBenchmarkConfig:
    """The single branch-only TRT recipe used for every global batch."""

    name: str = "huawei-fixed-gbs-dep8"
    parallelism: str = "DEP8"
    active_gpu_count: int = WORLD_SIZE
    tensor_parallel_size: int = WORLD_SIZE
    moe_expert_parallel_size: int = WORLD_SIZE
    moe_tensor_parallel_size: int = 1
    enable_attention_dp: bool = True
    enable_lm_head_tp_in_adp: bool = True
    batching_wait_iters: int = 0
    attention_dp_balance: bool = True
    overlap_scheduler: bool = False
    cuda_graph: bool = True
    enable_heuristic_topk: bool = True
    moe_backend: str = "TRTLLM"
    moe_max_num_tokens: int = MOE_MAX_NUM_TOKENS
    use_low_precision_moe_combine: bool = True
    kv_cache_free_gpu_memory_fraction: float = 0.60
    enable_configurable_moe: bool = True
    moe_autotune_dummy_distribution: str = "random"
    print_iter_log: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


FIXED_BENCHMARK_CONFIG = FixedBenchmarkConfig()


def validate_global_batch_size(global_batch_size: int) -> None:
    if global_batch_size not in ALLOWED_GLOBAL_BATCH_SIZES:
        raise ValueError(
            "Global batch size must be one of "
            f"{ALLOWED_GLOBAL_BATCH_SIZES}, got {global_batch_size}"
        )
    if global_batch_size % WORLD_SIZE != 0:
        raise ValueError(
            f"Global batch size {global_batch_size} is not divisible by "
            f"the {WORLD_SIZE} attention-DP ranks"
        )


def local_batch_size(global_batch_size: int) -> int:
    validate_global_batch_size(global_batch_size)
    return global_batch_size // WORLD_SIZE


def max_num_tokens(global_batch_size: int) -> int:
    """Allow every local-rank prompt to prefill in the same iteration."""
    return local_batch_size(global_batch_size) * INPUT_TOKENS


def build_llm_kwargs(
    model_path: str,
    global_batch_size: int,
) -> dict[str, Any]:
    """Build TRT arguments from one authoritative global batch size."""
    config = FIXED_BENCHMARK_CONFIG
    local_batch = local_batch_size(global_batch_size)
    return {
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
        "max_num_tokens": max_num_tokens(global_batch_size),
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
            "max_num_tokens": config.moe_max_num_tokens,
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


def fixed_environment(global_batch_size: int) -> dict[str, str]:
    validate_global_batch_size(global_batch_size)
    config = FIXED_BENCHMARK_CONFIG
    configurable = "1" if config.enable_configurable_moe else "0"
    return {
        "ENABLE_CONFIGURABLE_MOE": configurable,
        "TRTLLM_BENCH_ENABLE_CONFIGURABLE_MOE": configurable,
        "TRTLLM_BENCH_ENGINE_WARMUP_MAX_TOKENS": str(
            ENGINE_WARMUP_MAX_TOKENS
        ),
        "TRTLLM_BENCH_FP8_FUSED_QUANT_MAX_ROWS": str(
            FP8_FUSED_QUANT_MAX_ROWS
        ),
        "TRTLLM_BENCH_FIXED_BATCH_TIMEOUT_SECONDS": "120",
        "TRTLLM_BENCH_GLOBAL_BATCH_SIZE": str(global_batch_size),
        "TRTLLM_GEN_MOE_AUTOTUNE_DUMMY_DISTRIBUTION": (
            config.moe_autotune_dummy_distribution
        ),
    }


def resolved_parallelism(
    llm_args: Any,
    global_batch_size: int,
) -> dict[str, Any]:
    """Validate that TRT resolved the fixed recipe without silent changes."""
    config = FIXED_BENCHMARK_CONFIG
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
        "local_batch_size": local_batch_size(global_batch_size),
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
        "max_batch_size": local_batch_size(global_batch_size),
        "max_num_tokens": max_num_tokens(global_batch_size),
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
    resolved["moe_backend"] = str(moe.backend)
    resolved["moe_max_num_tokens"] = int(moe.max_num_tokens)
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
    expected_graph = [local_batch_size(global_batch_size)]
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
    return resolved
