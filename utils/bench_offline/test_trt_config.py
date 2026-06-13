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
    )
    resolved = resolved_parallelism(llm_args, candidate)
    assert resolved["enable_lm_head_tp_in_adp"] is True


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
    )
    resolved = resolved_parallelism(llm_args, candidate)
    assert resolved["effective_parallelism"] == "TP4"


@pytest.mark.parametrize(
    "filename",
    (
        "b300_huawei_experiments.json",
        "b300_stage2_experiments.json",
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
