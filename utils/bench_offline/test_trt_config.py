from types import SimpleNamespace

from trt_config import (
    CandidateConfig,
    build_llm_kwargs,
    choose_winner,
    resolved_parallelism,
)


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
