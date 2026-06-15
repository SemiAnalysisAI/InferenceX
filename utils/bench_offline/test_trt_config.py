import json
from types import SimpleNamespace

import pytest

from trt_config import (
    ALLOWED_GLOBAL_BATCH_SIZES,
    ATTENTION_WORKSPACE_BYTES_PER_TOKEN,
    ATTENTION_WORKSPACE_ENV,
    ATTENTION_WORKSPACE_HEADROOM_BYTES,
    BENCHMARK_PROFILE_ENV,
    CONTROLLED_ENVIRONMENT_VARIABLES,
    ENGINE_WARMUP_MAX_TOKENS,
    FIXED_BATCH_ARM_ENV,
    FP8_DEEP_GEMM_MAX_ROWS,
    FP8_DEEP_GEMM_MAX_ROWS_ENV,
    FP8_FUSED_QUANT_MAX_ROWS,
    GBS128_PREFILL_TRANSIENT_RESERVE_BYTES,
    GB300_GBS128_ENGINE_WARMUP_MAX_TOKENS,
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
    engine_max_batch_size,
    engine_warmup_max_tokens,
    external_mpi_rank_environment,
    fixed_environment,
    hardware_profile,
    kv_prefill_reserve_bytes,
    local_batch_size,
    max_num_tokens,
    measured_output_tokens,
    minimum_runtime_kv_tokens,
    resolved_parallelism,
    setup_prefill_iterations,
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
    assert "max_batch_size" not in kwargs["cuda_graph_config"]


@pytest.mark.parametrize(
    (
        "global_batch_size",
        "expected_local",
        "expected_tokens",
        "expected_warmup_tokens",
    ),
    (
        (16, 1, 8192, 65536),
        (64, 4, 32768, 65536),
        (128, 8, 65536, 32768),
    ),
)
def test_gb300_derives_dep16_local_capacity(
    global_batch_size,
    expected_local,
    expected_tokens,
    expected_warmup_tokens,
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
    assert (
        engine_warmup_max_tokens(global_batch_size, GB300_PROFILE)
        == expected_warmup_tokens
    )
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


def test_gb300_rank_environment_enables_required_runtime_flags():
    environment = fixed_environment(128, GB300_PROFILE)
    assert environment["ENABLE_CONFIGURABLE_MOE"] == "1"
    assert environment["TRTLLM_BENCH_ENABLE_CONFIGURABLE_MOE"] == "1"
    assert environment["TRTLLM_ENABLE_PDL"] == "1"
    assert environment[ATTENTION_WORKSPACE_ENV] == "0"
    assert environment[KV_PREFILL_RESERVE_ENV] == "0"
    assert environment[MIN_RUNTIME_KV_TOKENS_ENV] == "74752"
    assert environment[
        "TRTLLM_BENCH_ENGINE_WARMUP_MAX_TOKENS"
    ] == str(GB300_GBS128_ENGINE_WARMUP_MAX_TOKENS)
    assert environment[
        "TRTLLM_BENCH_FP8_FUSED_QUANT_MAX_ROWS"
    ] == "32768"
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
    assert kwargs["attention_dp_config"] == {
        "batching_wait_iters": 0,
        "enable_balance": True,
    }
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
        BENCHMARK_PROFILE_ENV: "huawei",
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


def test_external_mpi_rank_environment_is_complete(tmp_path):
    arm_file = tmp_path / "fixed-batch.armed.json"
    marker_file = tmp_path / "perfect-router.jsonl"
    cache_dir = tmp_path / "cute-cache"
    environment = external_mpi_rank_environment(
        16,
        arm_file,
        marker_file,
        cache_dir,
        GB300_PROFILE,
    )
    configured = benchmark_environment(16, arm_file, GB300_PROFILE)

    assert environment["ENABLE_PERFECT_ROUTER"] == "1"
    assert environment["TRTLLM_ENABLE_PERFECT_ROUTER"] == "1"
    assert environment["TRTLLM_PERFECT_ROUTER_MARKER"] == str(marker_file)
    assert environment["CUTE_DSL_CACHE_DIR"] == str(cache_dir)
    assert environment["TRTLLM_BENCH_CUTE_DSL_CACHE_DIR"] == str(cache_dir)
    assert (
        environment["TRTLLM_BENCH_EXPECTED_RANK_ENV"]
        == json.dumps(
            configured,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    for name, value in configured.items():
        assert environment[name] == value


def test_pr_rank_environment_uses_learned_router(tmp_path):
    profile = hardware_profile("gb300", "pr-tp16-mtp3")
    environment = external_mpi_rank_environment(
        400,
        tmp_path / "fixed-batch.armed.json",
        tmp_path / "rank-marker.jsonl",
        tmp_path / "cute-cache",
        profile,
    )

    assert "ENABLE_PERFECT_ROUTER" not in environment
    assert "TRTLLM_ENABLE_PERFECT_ROUTER" not in environment


@pytest.mark.parametrize(
    ("keyword", "value"),
    (
        ("fixed_batch_arm_file", "relative-arm"),
        ("marker_file", "relative-marker"),
        ("cute_cache_dir", "relative-cache"),
    ),
)
def test_external_mpi_rank_environment_requires_absolute_paths(
    tmp_path,
    keyword,
    value,
):
    arguments = {
        "fixed_batch_arm_file": tmp_path / "arm",
        "marker_file": tmp_path / "marker",
        "cute_cache_dir": tmp_path / "cache",
    }
    arguments[keyword] = value
    with pytest.raises(ValueError, match="absolute path"):
        external_mpi_rank_environment(
            16,
            profile=GB300_PROFILE,
            **arguments,
        )


def test_old_tuning_environment_is_always_cleared():
    assert {
        "TLLM_METRICS_ALL_RANKS",
        "ENABLE_PERFECT_ROUTER",
        ATTENTION_WORKSPACE_ENV,
        FIXED_BATCH_ARM_ENV,
        FP8_DEEP_GEMM_MAX_ROWS_ENV,
        KV_PREFILL_RESERVE_ENV,
        MIN_RUNTIME_KV_TOKENS_ENV,
        "TRTLLM_DSV4_SKIP_PREMOE_ALLREDUCE",
        "TRTLLM_ENABLE_PDL",
        "TRTLLM_FORCE_COMM_METHOD",
        "TRTLLM_ENABLE_PERFECT_ROUTER",
        "TRTLLM_MOE_POST_QUANT_ALLTOALLV",
    } <= CONTROLLED_ENVIRONMENT_VARIABLES


def test_only_huawei_global_batches_are_allowed():
    assert ALLOWED_GLOBAL_BATCH_SIZES == (16, 64, 128)
    for value in ALLOWED_GLOBAL_BATCH_SIZES:
        validate_global_batch_size(value)
    with pytest.raises(ValueError, match="must be one of"):
        validate_global_batch_size(32)


@pytest.mark.parametrize(
    (
        "profile_name",
        "global_batch_size",
        "world_size",
        "physical_nodes",
        "local_batch",
        "draft_tokens",
        "kv_fraction",
        "output_tokens",
        "setup_iterations",
    ),
    (
        (
            "pr-tp32-mtp3",
            256,
            32,
            8,
            8,
            3,
            0.70,
            1029,
            2,
        ),
        (
            "pr-tp16-mtp3",
            512,
            16,
            4,
            32,
            3,
            0.70,
            1053,
            8,
        ),
        (
            "pr-tp8-mtp1",
            4096,
            8,
            2,
            512,
            1,
            0.80,
            1024,
            128,
        ),
    ),
)
def test_pr_profiles_copy_decode_shapes_with_offline_prefill_adaptation(
    profile_name,
    global_batch_size,
    world_size,
    physical_nodes,
    local_batch,
    draft_tokens,
    kv_fraction,
    output_tokens,
    setup_iterations,
):
    profile = hardware_profile("gb300", profile_name)
    kwargs = build_llm_kwargs(
        "/model",
        global_batch_size,
        profile,
    )

    assert profile.world_size == world_size
    assert profile.physical_nodes == physical_nodes
    assert local_batch_size(global_batch_size, profile) == local_batch
    assert max_num_tokens(global_batch_size, profile) == 32768
    assert (
        setup_prefill_iterations(global_batch_size, profile)
        == setup_iterations
    )
    assert (
        measured_output_tokens(global_batch_size, profile)
        == output_tokens
    )
    assert kwargs["max_batch_size"] == local_batch
    assert kwargs["max_num_tokens"] == 32768
    assert kwargs["max_seq_len"] == 9256
    assert kwargs["disable_overlap_scheduler"] is False
    assert kwargs["print_iter_log"] is True
    assert kwargs["return_perf_metrics"] is False
    assert kwargs["enable_iter_perf_stats"] is True
    assert "attention_dp_config" not in kwargs
    assert kwargs["speculative_config"]["max_draft_len"] == draft_tokens
    assert (
        kwargs["kv_cache_config"]["free_gpu_memory_fraction"]
        == kv_fraction
    )
    assert "sparse_attention_config" not in kwargs
    assert max(kwargs["cuda_graph_config"]["batch_sizes"]) == local_batch
    assert profile.config.enable_perfect_router is False


def test_pr_profile_rank_environment_drops_huawei_only_autotune_knob():
    profile = hardware_profile("gb300", "pr-tp8-mtp1")
    environment = fixed_environment(4096, profile)
    assert environment[BENCHMARK_PROFILE_ENV] == "pr-tp8-mtp1"
    assert environment[
        "TRTLLM_BENCH_ENGINE_WARMUP_MAX_TOKENS"
    ] == "32768"
    assert environment[
        "TRTLLM_BENCH_FIXED_BATCH_TIMEOUT_SECONDS"
    ] == "600"
    assert (
        "TRTLLM_GEN_MOE_AUTOTUNE_DUMMY_DISTRIBUTION"
        not in environment
    )


def test_pr_profile_rejects_non_capacity_global_batch():
    profile = hardware_profile("gb300", "pr-tp16-mtp3")
    validate_global_batch_size(400, profile)
    validate_global_batch_size(512, profile)
    with pytest.raises(ValueError, match="must be one of"):
        validate_global_batch_size(256, profile)


def test_rack_replica_profile_keeps_attempt14_tp8_engine_capacity():
    profile = hardware_profile("gb300", "rack-tp8-mtp1-engine")
    for global_batch_size in (8, 32, 64, 3440, 4096):
        validate_global_batch_size(global_batch_size, profile)
    kwargs = build_llm_kwargs("/model", 8, profile)
    assert profile.world_size == 8
    assert profile.physical_nodes == 2
    assert local_batch_size(8, profile) == 1
    assert kwargs["max_batch_size"] == 512
    assert max(kwargs["cuda_graph_config"]["batch_sizes"]) == 512
    assert kwargs["speculative_config"]["max_draft_len"] == 1
    assert kwargs["disable_overlap_scheduler"] is False
    assert kwargs["kv_cache_config"]["free_gpu_memory_fraction"] == 0.80


@pytest.mark.parametrize(
    ("profile_name", "global_batch_size", "local_batch", "engine_batch"),
    (
        ("pr-tp32-mtp3", 192, 6, 8),
        ("pr-tp16-mtp3", 400, 25, 32),
        ("pr-tp8-mtp1", 3440, 430, 512),
    ),
)
def test_pr_active_points_keep_full_copied_engine_capacity(
    profile_name,
    global_batch_size,
    local_batch,
    engine_batch,
):
    profile = hardware_profile("gb300", profile_name)
    kwargs = build_llm_kwargs(
        "/model",
        global_batch_size,
        profile,
    )
    assert local_batch_size(global_batch_size, profile) == local_batch
    assert (
        engine_max_batch_size(global_batch_size, profile)
        == engine_batch
    )
    assert kwargs["max_batch_size"] == engine_batch
    assert max(kwargs["cuda_graph_config"]["batch_sizes"]) == engine_batch


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
        attention_dp_config=SimpleNamespace(
            batching_wait_iters=0,
            enable_balance=True,
        ),
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


def test_resolved_parallelism_validates_pr_active_point():
    profile = hardware_profile("gb300", "pr-tp16-mtp3")
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
            max_num_tokens=None,
            use_low_precision_moe_combine=True,
            load_balancer=(
                "/dsv4-eplb-configs/"
                "moe_load_balancer_gen_ep16_slots384.yaml"
            ),
        ),
        kv_cache_config=SimpleNamespace(
            free_gpu_memory_fraction=0.70,
        ),
        sparse_attention_config=None,
        max_batch_size=32,
        max_num_tokens=32768,
        max_seq_len=9256,
        enable_iter_perf_stats=True,
        max_stats_len=ITERATION_STATS_CAPACITY,
        print_iter_log=True,
        disable_overlap_scheduler=False,
        attention_dp_config=None,
        cuda_graph_config=SimpleNamespace(
            batch_sizes=[1, 2, 4, 8, 16, 24, 32],
            max_batch_size=32,
        ),
        gpus_per_node=4,
    )
    resolved = resolved_parallelism(llm_args, 400, profile)
    assert resolved["local_batch_size"] == 25
    assert resolved["max_batch_size"] == 32
    assert resolved["enable_sparse_attention"] is False
    assert resolved["disable_overlap_scheduler"] is False
    assert resolved["cuda_graph_max_batch_size"] == 32


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
        attention_dp_config=SimpleNamespace(
            batching_wait_iters=0,
            enable_balance=True,
        ),
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
        attention_dp_config=SimpleNamespace(
            batching_wait_iters=0,
            enable_balance=True,
        ),
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
        attention_dp_config=SimpleNamespace(
            batching_wait_iters=0,
            enable_balance=True,
        ),
        cuda_graph_config=SimpleNamespace(
            batch_sizes=[4],
            max_batch_size=4,
        ),
        gpus_per_node=4,
    )
    resolved = resolved_parallelism(llm_args, 64, GB300_PROFILE)
    assert resolved["moe_load_balancer_slots"] == 384
    assert kwargs["moe_config"]["load_balancer"].endswith("slots384.yaml")
