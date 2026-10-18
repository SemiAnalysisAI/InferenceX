"""Selected YAML workloads bind to the native video service and client."""

import pytest

from infx.config import repository_root
from infx.srt_slurm.generate import point_environment
from infx.srt_slurm import video

ROOT = repository_root()
RECIPES = ROOT / "benchmarks/single_node/srt-slurm-recipes/h3"
ENV = {
    "IMAGE": "vllm/vllm-omni:v0.30.0",
    "MODEL": "MiniMaxAI/MiniMax-H3",
    "PRECISION": "bf16",
    "GPU_COUNT": "2",
    "CONC_LIST": "1",
    "H3_SPEC": "h3.yaml",
    "H3_SET": "release-v1",
    "FRAMEWORK": "vllm-omni",
    "HF_HUB_CACHE": "/mnt/hf_hub_cache",
    "RESULT_FILENAME": "h3-native-release",
}


def test_selected_set_reaches_the_video_binder():
    point = {
        "scenario-type": "video-generation", "image": ENV["IMAGE"],
        "model": ENV["MODEL"], "model-prefix": "MiniMax-H3", "precision": "bf16",
        "framework": "vllm-omni", "spec": "h3.yaml", "set": "release-v1",
        "gpus": 2, "conc-list": [1], "exp-name": "h3-native-release",
    }
    environment = point_environment(point)
    assert environment["H3_SET"] == "release-v1"
    recipe = video.prepare(
        RECIPES / "vllm-omni/h200-bf16/serving-client.yaml", environment, root=ROOT
    )
    assert recipe["benchmark"]["env"]["H3_SET"] == "release-v1"


def test_both_h3_engines_bind_the_same_selected_plan():
    vllm = video.prepare(RECIPES / "vllm-omni/h200-bf16/serving-client.yaml", ENV, root=ROOT)
    sglang = video.prepare(
        RECIPES / "sglang/h200-bf16/serving-client.yaml",
        {**ENV, "FRAMEWORK": "sglang", "IMAGE": "hongccc/sglang-omni:h3-runtime-71de97b"},
        root=ROOT,
    )
    for recipe, runtime in ((vllm, "vllm-omni"), (sglang, "sglang")):
        [service] = recipe["services"]
        client = recipe["benchmark"]["env"]
        assert service["env"]["H3_PLAN_SHA256"] == client["H3_PLAN_SHA256"]
        assert service["env"]["H3_SET"] == client["H3_SET"] == "release-v1"
        assert service["env"]["H3_RUNTIME"] == client["H3_RUNTIME"] == runtime
        assert service["env"]["H3_RUNTIME_REVISION"] == client["H3_RUNTIME_REVISION"]
        assert client["H3_SPEC_PATH"] == "/h3-package/specs/h3.yaml"
        assert client["GPU_COUNT"] == "2"
    assert vllm["benchmark"]["env"]["H3_PLAN_SHA256"] == sglang["benchmark"]["env"]["H3_PLAN_SHA256"]


@pytest.mark.parametrize("edits", [
    {"H3_SET": "missing"},
    {"H3_SET": "../release-v1"},
    {"MODEL": "Other/model"},
    {"FRAMEWORK": "other"},
])
def test_invalid_selected_plan_or_runtime_fails_before_submission(edits):
    with pytest.raises(ValueError):
        video.prepare(
            RECIPES / "vllm-omni/h200-bf16/serving-client.yaml",
            {**ENV, **edits}, root=ROOT,
        )
