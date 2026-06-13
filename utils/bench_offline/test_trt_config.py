import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from trt_config import (
    FINAL_MEASURED_PASSES,
    TUNING_MEASURED_PASSES,
    CandidateConfig,
    build_llm_kwargs,
    choose_winner,
    resolved_parallelism,
)


def test_benchmark_uses_one_measured_pass():
    assert TUNING_MEASURED_PASSES == 1
    assert FINAL_MEASURED_PASSES == 1


def result(value: float, name: str) -> dict:
    return {
        "status": "success",
        "candidate": {"name": name},
        "aggregate": {"derived_output_tput_per_gpu": value},
    }


def test_later_candidate_requires_three_percent_improvement():
    baseline = result(100.0, "baseline")
    too_small = result(102.99, "too-small")
    enough = result(103.0, "enough")
    assert choose_winner([baseline, too_small]) is baseline
    assert choose_winner([baseline, too_small, enough]) is enough


def test_llm_kwargs_are_fixed_dep8_mtp3():
    kwargs = build_llm_kwargs(
        "/model",
        64,
        CandidateConfig(name="wait30", batching_wait_iters=30),
    )
    assert kwargs["tensor_parallel_size"] == 8
    assert kwargs["moe_expert_parallel_size"] == 8
    assert kwargs["moe_tensor_parallel_size"] == 1
    assert kwargs["enable_attention_dp"] is True
    assert kwargs["enable_lm_head_tp_in_adp"] is False
    assert kwargs["speculative_config"]["max_draft_len"] == 3
    assert kwargs["cuda_graph_config"]["batch_sizes"] == [64]
    assert kwargs["max_seq_len"] == 9216
    assert kwargs["print_iter_log"] is True
    assert "sparse_attention_config" not in kwargs


def test_candidate_can_enable_cute_dsl_mqa_and_tighten_capacity():
    candidate = CandidateConfig(
        name="dsl-tight",
        batching_wait_iters=0,
        use_cute_dsl_paged_mqa_logits=True,
        print_iter_log=False,
        max_seq_len=8832,
    )
    kwargs = build_llm_kwargs("/model", 64, candidate)
    assert kwargs["sparse_attention_config"] == {
        "algorithm": "deepseek_v4",
        "use_cute_dsl_paged_mqa_logits": True,
    }
    assert kwargs["print_iter_log"] is False
    assert kwargs["max_seq_len"] == 8832


def test_candidate_can_enable_sparse_indexer_optimizations():
    candidate = CandidateConfig(
        name="sparse-optimizations",
        batching_wait_iters=0,
        use_cute_dsl_topk=True,
        use_cute_dsl_paged_mqa_logits=True,
        enable_heuristic_topk=True,
        indexer_k_dtype="fp8",
    )
    kwargs = build_llm_kwargs("/model", 128, candidate)
    assert kwargs["sparse_attention_config"] == {
        "algorithm": "deepseek_v4",
        "use_cute_dsl_topk": True,
        "use_cute_dsl_paged_mqa_logits": True,
        "enable_heuristic_topk": True,
        "indexer_k_dtype": "fp8",
    }


def test_candidate_rejects_invalid_indexer_dtype():
    with pytest.raises(ValueError, match="indexer_k_dtype"):
        CandidateConfig(
            name="invalid-indexer-dtype",
            batching_wait_iters=0,
            indexer_k_dtype="bf16",
        )


@pytest.mark.parametrize("max_seq_len", (8819, True, 8832.0))
def test_candidate_rejects_invalid_max_seq_len(max_seq_len):
    with pytest.raises(ValueError, match="max_seq_len"):
        CandidateConfig(
            name="invalid-seq",
            batching_wait_iters=0,
            max_seq_len=max_seq_len,
        )


def test_global_batch_mode_preserves_legacy_c8_capacity():
    kwargs = build_llm_kwargs(
        "/model",
        8,
        CandidateConfig(name="global", batching_wait_iters=30),
    )
    assert kwargs["max_batch_size"] == 16
    assert kwargs["max_num_tokens"] == 8512
    assert kwargs["cuda_graph_config"]["batch_sizes"] == [8]
    assert kwargs["cuda_graph_config"]["max_batch_size"] == 8


@pytest.mark.parametrize(
    ("concurrency", "expected_batch_size", "expected_max_num_tokens"),
    (
        (8, 1, 8452),
        (32, 4, 8464),
        (64, 8, 8480),
    ),
)
def test_local_rank_batch_mode_sizes_dep8_per_rank(
    concurrency,
    expected_batch_size,
    expected_max_num_tokens,
):
    candidate = CandidateConfig(
        name=f"local-{concurrency}",
        batching_wait_iters=30,
        attention_dp_batch_mode="local-rank",
    )
    kwargs = build_llm_kwargs("/model", concurrency, candidate)
    assert kwargs["max_batch_size"] == expected_batch_size
    assert kwargs["max_num_tokens"] == expected_max_num_tokens
    assert kwargs["cuda_graph_config"]["batch_sizes"] == [
        expected_batch_size
    ]
    assert kwargs["cuda_graph_config"]["max_batch_size"] == (
        expected_batch_size
    )


def test_candidate_can_enable_production_lm_head_tp():
    candidate = CandidateConfig(
        name="wait30-lmtp",
        batching_wait_iters=30,
        attention_dp_timeout_iters=50,
        enable_lm_head_tp_in_adp=True,
    )
    kwargs = build_llm_kwargs("/model", 32, candidate)
    assert kwargs["enable_lm_head_tp_in_adp"] is True
    assert kwargs["attention_dp_config"]["timeout_iters"] == 50
    assert kwargs["speculative_config"]["max_draft_len"] == 3


def test_candidate_can_use_trt_default_attention_dp_timeout():
    candidate = CandidateConfig(
        name="production",
        batching_wait_iters=30,
        attention_dp_timeout_iters=None,
        enable_lm_head_tp_in_adp=True,
    )
    kwargs = build_llm_kwargs("/model", 64, candidate)
    assert "timeout_iters" not in kwargs["attention_dp_config"]


def test_tp4_uses_four_active_gpus_without_attention_dp():
    candidate = CandidateConfig(
        name="tp4",
        batching_wait_iters=0,
        parallelism="tp4",
    )
    kwargs = build_llm_kwargs("/model", 32, candidate)
    assert candidate.active_gpu_count == 4
    assert candidate.effective_parallelism == "TP4"
    assert kwargs["tensor_parallel_size"] == 4
    assert kwargs["moe_expert_parallel_size"] == 1
    assert kwargs["moe_tensor_parallel_size"] == 4
    assert kwargs["enable_attention_dp"] is False
    assert kwargs["enable_lm_head_tp_in_adp"] is False
    assert "attention_dp_config" not in kwargs
    assert kwargs["kv_cache_config"]["free_gpu_memory_fraction"] == 0.90


def test_dep4_uses_four_active_attention_dp_ranks():
    candidate = CandidateConfig(
        name="dep4",
        batching_wait_iters=30,
        enable_lm_head_tp_in_adp=True,
        parallelism="dep4",
    )
    kwargs = build_llm_kwargs("/model", 64, candidate)
    assert candidate.active_gpu_count == 4
    assert candidate.effective_parallelism == "DEP4"
    assert kwargs["tensor_parallel_size"] == 4
    assert kwargs["moe_expert_parallel_size"] == 4
    assert kwargs["moe_tensor_parallel_size"] == 1
    assert kwargs["enable_attention_dp"] is True
    assert kwargs["enable_lm_head_tp_in_adp"] is True


def test_lm_head_tp_requires_attention_dp():
    with pytest.raises(ValueError, match="requires attention DP"):
        CandidateConfig(
            name="invalid",
            batching_wait_iters=0,
            enable_lm_head_tp_in_adp=True,
            parallelism="tp4",
        )


def test_local_rank_batch_mode_requires_attention_dp():
    with pytest.raises(ValueError, match="requires attention DP"):
        CandidateConfig(
            name="invalid-local",
            batching_wait_iters=0,
            attention_dp_batch_mode="local-rank",
            parallelism="tp4",
        )


def test_attention_dp_batch_mode_is_validated():
    with pytest.raises(ValueError, match="must be global or local-rank"):
        CandidateConfig(
            name="invalid-mode",
            batching_wait_iters=0,
            attention_dp_batch_mode="rank",
        )


def test_candidate_name_must_be_filename_safe():
    with pytest.raises(ValueError, match="Invalid candidate name"):
        CandidateConfig(name="../escape", batching_wait_iters=30)


def test_resolved_parallelism_rejects_no_expected_values():
    llm_args = SimpleNamespace(
        parallel_config=SimpleNamespace(
            world_size=8,
            tp_size=8,
            moe_ep_size=8,
            moe_tp_size=1,
            enable_attention_dp=True,
            enable_lm_head_tp_in_adp=False,
        ),
        speculative_config=SimpleNamespace(max_draft_len=3),
        max_seq_len=9216,
        print_iter_log=True,
        sparse_attention_config=None,
    )
    assert resolved_parallelism(llm_args)["effective_parallelism"] == "DEP8"


def test_resolved_parallelism_accepts_candidate_lm_head_tp():
    candidate = CandidateConfig(
        name="wait30-lmtp",
        batching_wait_iters=30,
        enable_lm_head_tp_in_adp=True,
    )
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
        max_seq_len=9216,
        print_iter_log=True,
        sparse_attention_config=None,
    )
    resolved = resolved_parallelism(llm_args, candidate)
    assert resolved["enable_lm_head_tp_in_adp"] is True


def test_resolved_parallelism_validates_local_rank_runtime_capacity():
    candidate = CandidateConfig(
        name="local",
        batching_wait_iters=30,
        attention_dp_batch_mode="local-rank",
    )
    llm_args = SimpleNamespace(
        parallel_config=SimpleNamespace(
            world_size=8,
            tp_size=8,
            moe_ep_size=8,
            moe_tp_size=1,
            enable_attention_dp=True,
            enable_lm_head_tp_in_adp=False,
        ),
        speculative_config=SimpleNamespace(max_draft_len=3),
        max_seq_len=9216,
        print_iter_log=True,
        sparse_attention_config=None,
        max_batch_size=4,
        max_num_tokens=8464,
        cuda_graph_config=SimpleNamespace(
            batch_sizes=[4],
            max_batch_size=4,
        ),
    )
    resolved = resolved_parallelism(llm_args, candidate, concurrency=32)
    assert resolved["attention_dp_batch_mode"] == "local-rank"
    assert resolved["global_concurrency"] == 32
    assert resolved["max_batch_size"] == 4
    assert resolved["max_num_tokens"] == 8464
    assert resolved["cuda_graph_batch_sizes"] == [4]
    assert resolved["cuda_graph_max_batch_size"] == 4


def test_resolved_parallelism_rejects_wrong_local_graph_capacity():
    candidate = CandidateConfig(
        name="local",
        batching_wait_iters=30,
        attention_dp_batch_mode="local-rank",
    )
    llm_args = SimpleNamespace(
        parallel_config=SimpleNamespace(
            world_size=8,
            tp_size=8,
            moe_ep_size=8,
            moe_tp_size=1,
            enable_attention_dp=True,
            enable_lm_head_tp_in_adp=False,
        ),
        speculative_config=SimpleNamespace(max_draft_len=3),
        max_seq_len=9216,
        print_iter_log=True,
        sparse_attention_config=None,
        max_batch_size=4,
        max_num_tokens=8464,
        cuda_graph_config=SimpleNamespace(
            batch_sizes=[32],
            max_batch_size=32,
        ),
    )
    with pytest.raises(
        RuntimeError,
        match="CUDA graph batch sizes mismatch",
    ):
        resolved_parallelism(llm_args, candidate, concurrency=32)


def test_resolved_parallelism_accepts_tp4():
    candidate = CandidateConfig(
        name="tp4",
        batching_wait_iters=0,
        parallelism="tp4",
    )
    llm_args = SimpleNamespace(
        parallel_config=SimpleNamespace(
            world_size=4,
            tp_size=4,
            moe_ep_size=1,
            moe_tp_size=4,
            enable_attention_dp=False,
            enable_lm_head_tp_in_adp=False,
        ),
        speculative_config=SimpleNamespace(max_draft_len=3),
        max_seq_len=9216,
        print_iter_log=True,
        sparse_attention_config=None,
    )
    resolved = resolved_parallelism(llm_args, candidate)
    assert resolved["effective_parallelism"] == "TP4"


def test_resolved_parallelism_validates_runtime_tuning_switches():
    candidate = CandidateConfig(
        name="runtime-switches",
        batching_wait_iters=0,
        use_cute_dsl_paged_mqa_logits=True,
        print_iter_log=False,
        max_seq_len=8832,
    )
    llm_args = SimpleNamespace(
        parallel_config=SimpleNamespace(
            world_size=8,
            tp_size=8,
            moe_ep_size=8,
            moe_tp_size=1,
            enable_attention_dp=True,
            enable_lm_head_tp_in_adp=False,
        ),
        speculative_config=SimpleNamespace(max_draft_len=3),
        max_seq_len=8832,
        print_iter_log=False,
        sparse_attention_config=SimpleNamespace(
            use_cute_dsl_paged_mqa_logits=True,
        ),
    )
    resolved = resolved_parallelism(llm_args, candidate)
    assert resolved["max_seq_len"] == 8832
    assert resolved["print_iter_log"] is False
    assert resolved["use_cute_dsl_paged_mqa_logits"] is True


def test_resolved_parallelism_validates_sparse_indexer_switches():
    candidate = CandidateConfig(
        name="sparse-runtime-switches",
        batching_wait_iters=0,
        use_cute_dsl_topk=True,
        use_cute_dsl_paged_mqa_logits=True,
        enable_heuristic_topk=True,
        indexer_k_dtype="fp8",
    )
    llm_args = SimpleNamespace(
        parallel_config=SimpleNamespace(
            world_size=8,
            tp_size=8,
            moe_ep_size=8,
            moe_tp_size=1,
            enable_attention_dp=True,
            enable_lm_head_tp_in_adp=False,
        ),
        speculative_config=SimpleNamespace(max_draft_len=3),
        max_seq_len=9216,
        print_iter_log=True,
        sparse_attention_config=SimpleNamespace(
            use_cute_dsl_topk=True,
            use_cute_dsl_paged_mqa_logits=True,
            enable_heuristic_topk=True,
            indexer_k_dtype="fp8",
            index_topk=1024,
        ),
    )
    resolved = resolved_parallelism(llm_args, candidate)
    assert resolved["use_cute_dsl_topk"] is True
    assert resolved["use_cute_dsl_paged_mqa_logits"] is True
    assert resolved["enable_heuristic_topk"] is True
    assert resolved["indexer_k_dtype"] == "fp8"
    assert resolved["index_topk"] == 1024


@pytest.mark.parametrize(
    "filename",
    (
        "b300_huawei_experiments.json",
        "b300_stage2_experiments.json",
        "b300_stage3_local_batch_experiments.json",
        "b300_stage4_kernel_experiments.json",
        "b300_huawei_global_batch_experiments.json",
    ),
)
def test_checked_in_experiment_matrices_are_valid(filename):
    path = Path(__file__).with_name(filename)
    experiments = json.loads(path.read_text(encoding="utf-8"))
    assert 1 <= len(experiments) <= 10
    assert len({item["id"] for item in experiments}) == len(experiments)
    for experiment in experiments:
        assert experiment["concurrency"] in {
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
        }
        CandidateConfig.from_dict(experiment["candidate"])
