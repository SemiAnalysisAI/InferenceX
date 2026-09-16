"""Behavioral checks for the CI-to-srt-slurm adapter; no GPU or Slurm required."""

from pathlib import Path

import pytest

from utils.srt_slurm import prepare_recipe


def prepare(tmp_path: Path, *, dp: bool, concurrency: str, eval_only: bool = False):
    role = {"tp-size": 8, "enable-dp-attention": dp}
    recipe = {
        "model": {"container": "engine:image"},
        "frontend": {"args": {}},
        "backend": {
            "sglang_config": {
                "prefill": {**role, "enable-hierarchical-cache": True},
                "decode": {**role, "cuda-graph-bs-decode": []},
            },
            "decode_environment": {
                "SGLANG_SIMULATE_ACC_LEN": "2.3",
                "SGLANG_SIMULATE_ACC_METHOD": "match-expected",
                "SGLANG_SIMULATE_ACC_TOKEN_MODE": "real-draft-token",
                "ENGINE_OTHER_SETTING": "keep",
            },
        },
        "benchmark": {"type": "custom", "command": "benchmark", "env": {}},
    }
    env = {
        "IMAGE": "engine:image",
        "MODEL": "example/model",
        "CLIENT_IMAGE": "client:image",
        "CONC_LIST": concurrency,
        "PREFILL_TP": "8",
        "DECODE_TP": "8",
        "PREFILL_DP_ATTN": str(dp).lower(),
        "DECODE_DP_ATTN": str(dp).lower(),
        "DISABLE_CUSTOM_ALL_REDUCE": "0",
        "HICACHE_RATIO": "1.25",
        "PREFILL_ROUTER_POLICY": "consistent_hashing",
        "EVAL_ONLY": str(eval_only).lower(),
        "INFERENCEX_RUNTIME_ENV_VARS": "CUSTOM_CLIENT_SETTING",
        "CUSTOM_CLIENT_SETTING": "from-caller",
    }
    return prepare_recipe(
        recipe,
        {},
        env,
        workspace=tmp_path,
        results_root=tmp_path / "results",
        aiperf_cache=tmp_path / "cache",
        image_cache=tmp_path / "images",
    )[0]


@pytest.mark.parametrize(
    ("dp", "concurrency", "requests", "graphs"),
    [(False, "2 4", 8, [1, 2, 3, 4, 5, 6, 7, 8]), (True, "8 12", 24, [1, 2, 3])],
)
def test_concurrency_sizes_admission_and_phase_graphs(
    tmp_path, dp, concurrency, requests, graphs
):
    result = prepare(tmp_path, dp=dp, concurrency=concurrency)
    prefill = result["backend"]["sglang_config"]["prefill"]
    decode = result["backend"]["sglang_config"]["decode"]
    assert prefill["max-running-requests"] == requests
    assert decode["max-running-requests"] == requests
    assert decode["cuda-graph-bs-decode"] == graphs
    assert prefill["hicache-ratio"] == 1.25
    assert result["benchmark"]["container_image"] == "client:image"
    assert result["benchmark"]["env"]["CUSTOM_CLIENT_SETTING"] == "from-caller"


def test_eval_removes_synthetic_acceptance_but_keeps_other_engine_environment(tmp_path):
    result = prepare(tmp_path, dp=False, concurrency="4", eval_only=True)
    assert result["backend"]["decode_environment"] == {"ENGINE_OTHER_SETTING": "keep"}
    assert result["benchmark"]["env"]["DECODE_TP"] == "8"


def test_rejects_dp_concurrency_that_cannot_capture_one_decode_batch(tmp_path):
    with pytest.raises(ValueError, match="too small"):
        prepare(tmp_path, dp=True, concurrency="2")
