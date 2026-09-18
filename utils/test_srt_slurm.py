"""Behavioral checks for the CI-to-srt-slurm adapter; no GPU or Slurm required."""

import json
from pathlib import Path

import pytest

from infx.workflows.srt_slurm import collect_results, prepare_recipe


def prepare(tmp_path: Path, *, dp: bool, concurrency: str, eval_only: bool = False):
    role = {"tp-size": 8, "enable-dp-attention": dp}
    recipe = {
        "model": {"container": "engine:image"},
        "frontend": {"args": {}},
        "schema": 2,
        "engine": "sglang",
        "roles": {
            "prefill": {"args": {**role, "enable-hierarchical-cache": True}},
            "decode": {
                "args": {**role, "cuda-graph-bs-decode": []},
                "env": {
                    "SGLANG_SIMULATE_ACC_LEN": "2.3",
                    "SGLANG_SIMULATE_ACC_METHOD": "match-expected",
                    "SGLANG_SIMULATE_ACC_TOKEN_MODE": "real-draft-token",
                    "ENGINE_OTHER_SETTING": "keep",
                },
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
        "AIPERF_EXPERIMENTAL_FAST": "0",
        "REQUIRE_POWER": "0",
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
    prefill = result["roles"]["prefill"]["args"]
    decode = result["roles"]["decode"]["args"]
    assert prefill["max-running-requests"] == requests
    assert decode["max-running-requests"] == requests
    assert decode["cuda-graph-bs-decode"] == graphs
    assert prefill["hicache-ratio"] == 1.25
    assert result["benchmark"]["container_image"] == "client:image"
    assert result["benchmark"]["env"]["CUSTOM_CLIENT_SETTING"] == "from-caller"
    assert result["benchmark"]["env"]["AIPERF_EXPERIMENTAL_FAST"] == "0"
    assert result["benchmark"]["env"]["REQUIRE_POWER"] == "0"


def test_eval_removes_synthetic_acceptance_but_keeps_other_engine_environment(tmp_path):
    result = prepare(tmp_path, dp=False, concurrency="4", eval_only=True)
    assert result["roles"]["decode"]["env"] == {"ENGINE_OTHER_SETTING": "keep"}
    assert result["benchmark"]["env"]["DECODE_TP"] == "8"


def test_rejects_dp_concurrency_that_cannot_capture_one_decode_batch(tmp_path):
    with pytest.raises(ValueError, match="too small"):
        prepare(tmp_path, dp=True, concurrency="2")


def test_fixed_sequence_preserves_dispatch_pin_and_sizes_admission(tmp_path):
    recipe = {
        "schema": 2,
        "engine": "sglang",
        "model": {"container": "engine:image"},
        "roles": {
            "prefill": {"args": {"enable-dp-attention": True}},
            "decode": {
                "args": {"cuda-graph-bs": [1, 2, 4]},
                "env": {
                    "SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
                    "MORI_MAX_DISPATCH_TOKENS_DECODE": "1",
                },
            },
        },
    }
    result, profile = prepare_recipe(
        recipe,
        {"output_dir": "${RESULTS}"},
        {
            "IMAGE": "engine:image",
            "CONC_LIST": "16 48",
            "PREFILL_EP": "8",
            "DECODE_TP": "8",
            "DECODE_MTP_SIZE": "1",
            "RESULTS": "/shared/results",
        },
        workspace=tmp_path,
        results_root=tmp_path / "results",
        aiperf_cache=tmp_path / "cache",
        image_cache=tmp_path / "images",
    )
    assert result["roles"]["prefill"]["args"]["max-running-requests"] == 48
    assert result["roles"]["decode"]["args"] == {
        "cuda-graph-bs": [1, 2, 4],
        "max-running-requests": 48,
    }
    assert result["roles"]["decode"]["env"] == {
        "SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
        "MORI_MAX_DISPATCH_TOKENS_DECODE": "12",
        "SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD": "12",
    }
    assert profile["output_dir"] == "/shared/results"


@pytest.mark.parametrize("has_metadata", [True, False])
def test_collect_eval_from_native_job_logs(tmp_path, has_metadata):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    output = tmp_path / "job-42"
    eval_dir = output / "logs" / "eval_results"
    eval_dir.mkdir(parents=True)
    (eval_dir / "results.json").write_text('{"score": 0.9}')
    if has_metadata:
        (eval_dir / "meta_env.json").write_text('{"eval_concurrency": 48}')
    submission = {"slurm_job_id": "42", "output_dir": str(output)}
    env = {"RESULT_FILENAME": "example", "EVAL_ONLY": "true"}
    if not has_metadata:
        with pytest.raises(ValueError, match="No eval metadata"):
            collect_results(
                submission, env, workspace=workspace, results_root=tmp_path / "results"
            )
        return
    collect_results(
        submission, env, workspace=workspace, results_root=tmp_path / "results"
    )
    assert json.loads((workspace / "results.json").read_text()) == {"score": 0.9}
    assert json.loads((workspace / "meta_env.json").read_text()) == {
        "eval_concurrency": 48
    }
