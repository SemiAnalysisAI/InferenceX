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
    FP8_FUSED_QUANT_MAX_ROWS,
    HUAWEI_MEASURED_DECODE_ROUNDS,
    HUAWEI_WARMUP_DECODE_ROUNDS,
    ITERATION_STATS_CAPACITY,
    MAX_SEQ_LEN,
    MEASURED_OUTPUT_TOKENS,
    MOE_MAX_NUM_TOKENS,
    WARMUP_OUTPUT_TOKENS,
    attention_workspace_target_bytes,
    benchmark_environment,
    build_llm_kwargs,
    fixed_environment,
    local_batch_size,
    max_num_tokens,
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


def test_huawei_round_and_sequence_capacity_is_fixed():
    assert HUAWEI_WARMUP_DECODE_ROUNDS == 2
    assert HUAWEI_MEASURED_DECODE_ROUNDS == 256
    assert WARMUP_OUTPUT_TOKENS == 6
    assert MEASURED_OUTPUT_TOKENS == 1025
    assert MAX_SEQ_LEN == 9344


def test_attention_workspace_reservation_only_applies_above_warmup_cap():
    assert attention_workspace_target_bytes(16) == 0
    assert attention_workspace_target_bytes(64) == 0
    assert attention_workspace_target_bytes(128) == (
        131072 * ATTENTION_WORKSPACE_BYTES_PER_TOKEN
        + ATTENTION_WORKSPACE_HEADROOM_BYTES
    )
    assert attention_workspace_target_bytes(128) == 27_111_981_056


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
        "TRTLLM_BENCH_FP8_FUSED_QUANT_MAX_ROWS": str(
            FP8_FUSED_QUANT_MAX_ROWS
        ),
        "TRTLLM_BENCH_FIXED_BATCH_TIMEOUT_SECONDS": "120",
        "TRTLLM_BENCH_GLOBAL_BATCH_SIZE": "64",
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
