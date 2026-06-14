from types import SimpleNamespace

import pytest

from trt_config import (
    ALLOWED_GLOBAL_BATCH_SIZES,
    CONTROLLED_ENVIRONMENT_VARIABLES,
    HUAWEI_MEASURED_DECODE_ROUNDS,
    HUAWEI_WARMUP_DECODE_ROUNDS,
    ITERATION_STATS_CAPACITY,
    MAX_SEQ_LEN,
    MEASURED_OUTPUT_TOKENS,
    MOE_MAX_NUM_TOKENS,
    WARMUP_OUTPUT_TOKENS,
    build_llm_kwargs,
    fixed_environment,
    kv_cache_max_tokens,
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
        "expected_kv_tokens",
    ),
    (
        (16, 2, 16384, 18688),
        (64, 8, 65536, 74752),
        (128, 16, 131072, 149504),
    ),
)
def test_one_global_batch_derives_every_local_capacity(
    global_batch_size,
    expected_local,
    expected_tokens,
    expected_kv_tokens,
):
    kwargs = build_llm_kwargs("/model", global_batch_size)
    assert local_batch_size(global_batch_size) == expected_local
    assert max_num_tokens(global_batch_size) == expected_tokens
    assert kv_cache_max_tokens(global_batch_size) == expected_kv_tokens
    assert kwargs["max_batch_size"] == expected_local
    assert kwargs["max_num_tokens"] == expected_tokens
    assert kwargs["kv_cache_config"]["max_tokens"] == expected_kv_tokens
    assert kwargs["cuda_graph_config"]["batch_sizes"] == [expected_local]
    assert kwargs["cuda_graph_config"]["max_batch_size"] == expected_local


def test_huawei_round_and_sequence_capacity_is_fixed():
    assert HUAWEI_WARMUP_DECODE_ROUNDS == 2
    assert HUAWEI_MEASURED_DECODE_ROUNDS == 256
    assert WARMUP_OUTPUT_TOKENS == 6
    assert MEASURED_OUTPUT_TOKENS == 1025
    assert MAX_SEQ_LEN == 9344


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
        "TRTLLM_BENCH_FIXED_BATCH_TIMEOUT_SECONDS": "120",
        "TRTLLM_BENCH_GLOBAL_BATCH_SIZE": "64",
        "TRTLLM_GEN_MOE_AUTOTUNE_DUMMY_DISTRIBUTION": "random",
    }


def test_old_tuning_environment_is_always_cleared():
    assert {
        "TLLM_METRICS_ALL_RANKS",
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
            max_tokens=74752,
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
    assert resolved["kv_cache_max_tokens"] == 74752
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
            max_tokens=74752,
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
