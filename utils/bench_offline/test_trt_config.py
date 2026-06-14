from types import SimpleNamespace

import pytest

from trt_config import (
    ALLOWED_GLOBAL_BATCH_SIZES,
    ATTENTION_WORKSPACE_BYTES_PER_TOKEN,
    ATTENTION_WORKSPACE_ENV,
    ATTENTION_WORKSPACE_HEADROOM_BYTES,
    CONTROLLED_ENVIRONMENT_VARIABLES,
    ENGINE_WARMUP_MAX_TOKENS,
    FIXED_BATCH_ARM_ENV,
    FP8_DEEP_GEMM_MAX_ROWS,
    FP8_DEEP_GEMM_MAX_ROWS_ENV,
    FP8_FUSED_QUANT_MAX_ROWS,
    GBS128_PREFILL_TRANSIENT_RESERVE_BYTES,
    GB300_PROFILE,
    HUAWEI_MEASURED_DECODE_ROUNDS,
    HUAWEI_WARMUP_DECODE_ROUNDS,
    ITERATION_STATS_CAPACITY,
    KV_PREFILL_RESERVE_ENV,
    MAX_SEQ_LEN,
    MEASURED_OUTPUT_TOKENS,
    MIN_RUNTIME_KV_TOKENS_ENV,
    MOE_MAX_NUM_TOKENS,
    WARMUP_OUTPUT_TOKENS,
    attention_workspace_target_bytes,
    benchmark_environment,
    build_llm_kwargs,
    fixed_environment,
    kv_prefill_reserve_bytes,
    local_batch_size,
    max_num_tokens,
    minimum_runtime_kv_tokens,
    resolved_parallelism,
    validate_global_batch_size,
)


@pytest.mark.parametrize(
    (
        "global_batch_size",
        "expected_local",
        "expected_tokens",
    ),
    (
        (16, 2, 16384),
        (64, 8, 65536),
        (128, 16, 131072),
    ),
)
def test_one_global_batch_derives_every_local_capacity(
    global_batch_size,
    expected_local,
    expected_tokens,
):
    kwargs = build_llm_kwargs("/model", global_batch_size)
    assert local_batch_size(global_batch_size) == expected_local
    assert max_num_tokens(global_batch_size) == expected_tokens
    assert kwargs["max_batch_size"] == expected_local
    assert kwargs["max_num_tokens"] == expected_tokens
    assert "max_tokens" not in kwargs["kv_cache_config"]
    assert kwargs["kv_cache_config"]["free_gpu_memory_fraction"] == 0.60
    assert kwargs["cuda_graph_config"]["batch_sizes"] == [expected_local]
    assert kwargs["cuda_graph_config"]["max_batch_size"] == expected_local


@pytest.mark.parametrize(
    ("global_batch_size", "expected_local", "expected_tokens"),
    (
        (16, 1, 8192),
        (64, 4, 32768),
        (128, 8, 65536),
    ),
)
def test_gb300_derives_dep16_local_capacity(
    global_batch_size,
    expected_local,
    expected_tokens,
):
    kwargs = build_llm_kwargs(
        "/model",
        global_batch_size,
        GB300_PROFILE,
    )
    assert (
        local_batch_size(global_batch_size, GB300_PROFILE)
        == expected_local
    )
    assert (
        max_num_tokens(global_batch_size, GB300_PROFILE)
        == expected_tokens
    )
    assert kwargs["tensor_parallel_size"] == 16
    assert kwargs["moe_expert_parallel_size"] == 16
    assert kwargs["max_batch_size"] == expected_local
    assert kwargs["max_num_tokens"] == expected_tokens
    assert kwargs["gpus_per_node"] == 4
    assert kwargs["kv_cache_config"]["free_gpu_memory_fraction"] == 0.70
    assert kwargs["moe_config"]["backend"] == "MEGAMOE_DEEPGEMM"
    assert "max_num_tokens" not in kwargs["moe_config"]
    assert kwargs["moe_config"]["load_balancer"].endswith(
        "moe_load_balancer_gen_ep16_slots384.yaml"
    )


def test_gb300_fixed_batches_do_not_need_b300_large_prefill_reserves():
    for global_batch_size in ALLOWED_GLOBAL_BATCH_SIZES:
        assert (
            attention_workspace_target_bytes(
                global_batch_size,
                GB300_PROFILE,
            )
            == 0
        )
        assert (
            kv_prefill_reserve_bytes(
                global_batch_size,
                GB300_PROFILE,
            )
            == 0
        )
    assert minimum_runtime_kv_tokens(128, GB300_PROFILE) == 74752


def test_gb300_rank_environment_enables_only_profile_runtime_flags():
    environment = fixed_environment(128, GB300_PROFILE)
    assert environment["ENABLE_CONFIGURABLE_MOE"] == "0"
    assert environment["TRTLLM_BENCH_ENABLE_CONFIGURABLE_MOE"] == "0"
    assert environment["TRTLLM_ENABLE_PDL"] == "1"
    assert environment[ATTENTION_WORKSPACE_ENV] == "0"
    assert environment[KV_PREFILL_RESERVE_ENV] == "0"
    assert environment[MIN_RUNTIME_KV_TOKENS_ENV] == "74752"
    assert environment[
        "TRTLLM_BENCH_FP8_FUSED_QUANT_MAX_ROWS"
    ] == "0"
    assert environment[FP8_DEEP_GEMM_MAX_ROWS_ENV] == "0"


def test_huawei_round_and_sequence_capacity_is_fixed():
    assert HUAWEI_WARMUP_DECODE_ROUNDS == 2
    assert HUAWEI_MEASURED_DECODE_ROUNDS == 256
    assert WARMUP_OUTPUT_TOKENS == 6
    assert MEASURED_OUTPUT_TOKENS == 1025
    assert MAX_SEQ_LEN == 9344
    assert FP8_DEEP_GEMM_MAX_ROWS == 65536


def test_attention_workspace_reservation_only_applies_above_warmup_cap():
    assert attention_workspace_target_bytes(16) == 0
    assert attention_workspace_target_bytes(64) == 0
    assert attention_workspace_target_bytes(128) == (
        131072 * ATTENTION_WORKSPACE_BYTES_PER_TOKEN
        + ATTENTION_WORKSPACE_HEADROOM_BYTES
    )
    assert attention_workspace_target_bytes(128) == 27_111_981_056


def test_prefill_reserve_preserves_full_fixed_batch_kv_capacity():
    assert kv_prefill_reserve_bytes(16) == 0
    assert kv_prefill_reserve_bytes(64) == 0
    assert kv_prefill_reserve_bytes(128) == (
        GBS128_PREFILL_TRANSIENT_RESERVE_BYTES
    )
    assert kv_prefill_reserve_bytes(128) == 12_884_901_888
    assert minimum_runtime_kv_tokens(16) == 18_688
    assert minimum_runtime_kv_tokens(64) == 74_752
    assert minimum_runtime_kv_tokens(128) == 149_504


def test_llm_kwargs_force_synchronized_dep8_iteration_stats():
    kwargs = build_llm_kwargs("/model", 64)
    assert kwargs["tensor_parallel_size"] == 8
    assert kwargs["moe_expert_parallel_size"] == 8
    assert kwargs["moe_tensor_parallel_size"] == 1
    assert kwargs["enable_attention_dp"] is True
    assert kwargs["enable_lm_head_tp_in_adp"] is True
    assert kwargs["disable_overlap_scheduler"] is True
    assert kwargs["enable_iter_perf_stats"] is True
    assert kwargs["max_stats_len"] == ITERATION_STATS_CAPACITY
    assert kwargs["print_iter_log"] is False
    assert kwargs["speculative_config"]["max_draft_len"] == 3
    assert kwargs["sparse_attention_config"]["enable_heuristic_topk"] is True
    assert kwargs["moe_config"]["max_num_tokens"] == MOE_MAX_NUM_TOKENS


def test_fixed_rank_environment_is_explicit():
    assert fixed_environment(64) == {
        "ENABLE_CONFIGURABLE_MOE": "1",
        "TRTLLM_BENCH_ENABLE_CONFIGURABLE_MOE": "1",
        ATTENTION_WORKSPACE_ENV: "0",
        "TRTLLM_BENCH_ENGINE_WARMUP_MAX_TOKENS": str(
            ENGINE_WARMUP_MAX_TOKENS
        ),
        FP8_DEEP_GEMM_MAX_ROWS_ENV: str(FP8_DEEP_GEMM_MAX_ROWS),
        "TRTLLM_BENCH_FP8_FUSED_QUANT_MAX_ROWS": str(
            FP8_FUSED_QUANT_MAX_ROWS
        ),
        "TRTLLM_BENCH_FIXED_BATCH_TIMEOUT_SECONDS": "120",
        "TRTLLM_BENCH_GLOBAL_BATCH_SIZE": "64",
        KV_PREFILL_RESERVE_ENV: "0",
        MIN_RUNTIME_KV_TOKENS_ENV: "74752",
        "TRTLLM_GEN_MOE_AUTOTUNE_DUMMY_DISTRIBUTION": "random",
    }


def test_benchmark_environment_adds_absolute_barrier_arm_path(tmp_path):
    arm_file = tmp_path / "fixed-batch.armed.json"
    environment = benchmark_environment(64, arm_file)

    assert environment[FIXED_BATCH_ARM_ENV] == str(arm_file)
    with pytest.raises(ValueError, match="absolute path"):
        benchmark_environment(64, "relative-arm-file")


def test_old_tuning_environment_is_always_cleared():
    assert {
        "TLLM_METRICS_ALL_RANKS",
        ATTENTION_WORKSPACE_ENV,
        FIXED_BATCH_ARM_ENV,
        FP8_DEEP_GEMM_MAX_ROWS_ENV,
        KV_PREFILL_RESERVE_ENV,
        MIN_RUNTIME_KV_TOKENS_ENV,
        "TRTLLM_DSV4_SKIP_PREMOE_ALLREDUCE",
        "TRTLLM_ENABLE_PDL",
        "TRTLLM_FORCE_COMM_METHOD",
        "TRTLLM_MOE_POST_QUANT_ALLTOALLV",
    } <= CONTROLLED_ENVIRONMENT_VARIABLES


def test_only_huawei_global_batches_are_allowed():
    assert ALLOWED_GLOBAL_BATCH_SIZES == (16, 64, 128)
    for value in ALLOWED_GLOBAL_BATCH_SIZES:
        validate_global_batch_size(value)
    with pytest.raises(ValueError, match="must be one of"):
        validate_global_batch_size(32)


def test_resolved_parallelism_validates_fixed_capacity(monkeypatch):
    monkeypatch.setenv("ENABLE_CONFIGURABLE_MOE", "1")
    llm_args = SimpleNamespace(
        parallel_config=SimpleNamespace(
            world_size=8,
            tp_size=8,
            moe_ep_size=8,
            moe_tp_size=1,
            enable_attention_dp=True,
            enable_lm_head_tp_in_adp=True,
        ),
        speculative_config=SimpleNamespace(max_draft_len=3),
        moe_config=SimpleNamespace(
            backend="TRTLLM",
            max_num_tokens=MOE_MAX_NUM_TOKENS,
            use_low_precision_moe_combine=True,
        ),
        kv_cache_config=SimpleNamespace(
            free_gpu_memory_fraction=0.60,
        ),
        sparse_attention_config=SimpleNamespace(
            enable_heuristic_topk=True,
        ),
        max_batch_size=8,
        max_num_tokens=65536,
        max_seq_len=9344,
        enable_iter_perf_stats=True,
        max_stats_len=ITERATION_STATS_CAPACITY,
        print_iter_log=False,
        disable_overlap_scheduler=True,
        cuda_graph_config=SimpleNamespace(
            batch_sizes=[8],
            max_batch_size=8,
        ),
    )
    resolved = resolved_parallelism(llm_args, 64)
    assert resolved["global_batch_size"] == 64
    assert resolved["local_batch_size"] == 8
    assert resolved["max_num_tokens"] == 65536
    assert resolved["kv_cache_free_gpu_memory_fraction"] == 0.60
    assert resolved["moe_max_num_tokens"] == MOE_MAX_NUM_TOKENS
    assert resolved["cuda_graph_batch_sizes"] == [8]


def test_resolved_parallelism_rejects_staggered_prefill_capacity():
    llm_args = SimpleNamespace(
        parallel_config=SimpleNamespace(
            world_size=8,
            tp_size=8,
            moe_ep_size=8,
            moe_tp_size=1,
            enable_attention_dp=True,
            enable_lm_head_tp_in_adp=True,
        ),
        speculative_config=SimpleNamespace(max_draft_len=3),
        moe_config=SimpleNamespace(
            backend="TRTLLM",
            max_num_tokens=MOE_MAX_NUM_TOKENS,
            use_low_precision_moe_combine=True,
        ),
        kv_cache_config=SimpleNamespace(
            free_gpu_memory_fraction=0.60,
        ),
        sparse_attention_config=SimpleNamespace(
            enable_heuristic_topk=True,
        ),
        max_batch_size=8,
        max_num_tokens=8480,
        max_seq_len=9344,
        enable_iter_perf_stats=True,
        max_stats_len=ITERATION_STATS_CAPACITY,
        print_iter_log=False,
        disable_overlap_scheduler=True,
        cuda_graph_config=SimpleNamespace(
            batch_sizes=[8],
            max_batch_size=8,
        ),
    )
    with pytest.raises(RuntimeError, match="max_num_tokens"):
        resolved_parallelism(llm_args, 64)


def test_resolved_parallelism_validates_gb300_dep16():
    kwargs = build_llm_kwargs("/model", 64, GB300_PROFILE)
    llm_args = SimpleNamespace(
        parallel_config=SimpleNamespace(
            world_size=16,
            tp_size=16,
            moe_ep_size=16,
            moe_tp_size=1,
            enable_attention_dp=True,
            enable_lm_head_tp_in_adp=True,
        ),
        speculative_config=SimpleNamespace(max_draft_len=3),
        moe_config=SimpleNamespace(
            backend="MEGAMOE_DEEPGEMM",
            load_balancer=kwargs["moe_config"]["load_balancer"],
            max_num_tokens=None,
            use_low_precision_moe_combine=True,
        ),
        kv_cache_config=SimpleNamespace(
            free_gpu_memory_fraction=0.70,
        ),
        sparse_attention_config=SimpleNamespace(
            enable_heuristic_topk=True,
        ),
        max_batch_size=4,
        max_num_tokens=32768,
        max_seq_len=9344,
        enable_iter_perf_stats=True,
        max_stats_len=ITERATION_STATS_CAPACITY,
        print_iter_log=False,
        disable_overlap_scheduler=True,
        cuda_graph_config=SimpleNamespace(
            batch_sizes=[4],
            max_batch_size=4,
        ),
        gpus_per_node=4,
    )
    resolved = resolved_parallelism(llm_args, 64, GB300_PROFILE)
    assert resolved["world_size"] == 16
    assert resolved["local_batch_size"] == 4
    assert resolved["gpus_per_node"] == 4
    assert resolved["moe_backend"] == "MEGAMOE_DEEPGEMM"


def test_resolved_parallelism_accepts_resolved_gb300_load_balancer():
    kwargs = build_llm_kwargs("/model", 64, GB300_PROFILE)
    llm_args = SimpleNamespace(
        parallel_config=SimpleNamespace(
            world_size=16,
            tp_size=16,
            moe_ep_size=16,
            moe_tp_size=1,
            enable_attention_dp=True,
            enable_lm_head_tp_in_adp=True,
        ),
        speculative_config=SimpleNamespace(max_draft_len=3),
        moe_config=SimpleNamespace(
            backend="MEGAMOE_DEEPGEMM",
            load_balancer=SimpleNamespace(num_slots=384),
            max_num_tokens=None,
            use_low_precision_moe_combine=True,
        ),
        kv_cache_config=SimpleNamespace(
            free_gpu_memory_fraction=0.70,
        ),
        sparse_attention_config=SimpleNamespace(
            enable_heuristic_topk=True,
        ),
        max_batch_size=4,
        max_num_tokens=32768,
        max_seq_len=9344,
        enable_iter_perf_stats=True,
        max_stats_len=ITERATION_STATS_CAPACITY,
        print_iter_log=False,
        disable_overlap_scheduler=True,
        cuda_graph_config=SimpleNamespace(
            batch_sizes=[4],
            max_batch_size=4,
        ),
        gpus_per_node=4,
    )
    resolved = resolved_parallelism(llm_args, 64, GB300_PROFILE)
    assert resolved["moe_load_balancer_slots"] == 384
    assert kwargs["moe_config"]["load_balancer"].endswith("slots384.yaml")
