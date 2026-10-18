"""The native YAML route preserves the selected request and model-view contract."""

import json
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

from evaluator import mvp_gpu_job, mvp_srt_selected  # noqa: E402
from evaluator.mvp_runner import preview_plan  # noqa: E402
from specs import h3_spec  # noqa: E402


def test_h3_selected_plan_drives_both_request_mappings():
    path = PACKAGE / "specs/h3.yaml"
    spec = h3_spec.load_spec(path)
    expected = h3_spec.plan_sha256(h3_spec.plan_for(spec, "release-v1"))
    plan = mvp_srt_selected.selected_plan(path, "release-v1", expected)
    sglang = preview_plan(plan, runtime="sglang")
    vllm = preview_plan(plan, runtime="vllm-omni")
    assert sglang["plan_sha256"] == vllm["plan_sha256"] == expected
    assert sglang["measurement_count"] == vllm["measurement_count"] == 20
    assert sglang["total_requests"] == vllm["total_requests"] == 21
    assert sglang["slots"][1]["request"]["prompt"] == vllm["slots"][1]["request"]["prompt"]
    with pytest.raises(ValueError, match="plan_sha256"):
        mvp_srt_selected.selected_plan(path, "release-v1", "0" * 64)


def test_h3_vllm_model_view_exposes_only_tested_partition(tmp_path):
    snapshot = tmp_path / "snapshot"
    (snapshot / "FL2VA").mkdir(parents=True)
    (snapshot / "Ref2VA").mkdir()
    for name in ("model_index.json", "LICENSE", "README.md", "Ref2VA/model_index.json", "modular_model_index.json"):
        (snapshot / name).write_text(name)
    view = mvp_srt_selected.h3_vllm_model_view(snapshot, tmp_path / "view")
    assert (view / "FL2VA").resolve() == (snapshot / "FL2VA").resolve()
    assert (view / "Ref2VA/model_index.json").read_text() == "Ref2VA/model_index.json"
    assert not (view / "modular_model_index.json").exists()


def test_h3_vllm_command_uses_the_tested_two_gpu_fl2va_profile():
    plan = h3_spec.plan_for(h3_spec.load_spec(PACKAGE / "specs/h3.yaml"), "release-v1")
    argv = mvp_srt_selected.server_argv(plan, "vllm-omni", Path("/model"), 30283, 2)
    assert argv[:3] == ["vllm", "serve", "/model"]
    assert argv[argv.index("--task-type") + 1] == "fl2va"
    assert argv[argv.index("--tensor-parallel-size") + 1] == "2"
    assert argv[argv.index("--num-gpus") + 1] == "2"
    with pytest.raises(ValueError, match="two GPUs"):
        mvp_srt_selected.server_argv(plan, "vllm-omni", Path("/model"), 30283, 4)


def test_wan_selected_plan_uses_the_same_video_request_on_both_engines():
    path = PACKAGE / "specs/wan.yaml"
    spec = h3_spec.load_spec(path)
    expected = h3_spec.plan_sha256(h3_spec.plan_for(spec, "release-v1"))
    plan = mvp_srt_selected.selected_plan(path, "release-v1", expected)
    sglang = preview_plan(plan, runtime="sglang")
    vllm = preview_plan(plan, runtime="vllm-omni")
    assert sglang["plan_sha256"] == vllm["plan_sha256"] == expected
    assert sglang["measurement_count"] == vllm["measurement_count"] == 20
    assert sglang["total_requests"] == vllm["total_requests"] == 21
    assert sglang["slots"][1]["request"] == vllm["slots"][1]["request"]


def test_wan_commands_use_the_recorded_eight_gpu_profiles(tmp_path):
    plan = h3_spec.plan_for(h3_spec.load_spec(PACKAGE / "specs/wan.yaml"), "release-v1")
    sglang = mvp_srt_selected.server_argv(
        plan, "sglang", Path("/model"), 30283, 8, output=tmp_path
    )
    assert sglang[sglang.index("--num-gpus") + 1] == "8"
    assert sglang[sglang.index("--ulysses-degree") + 1] == "8"
    assert sglang[sglang.index("--tp-size") + 1] == "1"
    assert sglang[sglang.index("--output-path") + 1] == str(tmp_path / "generated")
    vllm = mvp_srt_selected.server_argv(plan, "vllm-omni", Path("/model"), 30283, 8)
    assert vllm[:3] == ["vllm", "serve", "/model"]
    assert vllm[vllm.index("--cfg-parallel-size") + 1] == "2"
    assert vllm[vllm.index("--usp") + 1] == "4"
    assert vllm[vllm.index("--stage-init-timeout") + 1] == "3600"
    with pytest.raises(ValueError, match="eight Wan GPUs"):
        mvp_srt_selected.server_argv(plan, "vllm-omni", Path("/model"), 30283, 4)
@pytest.mark.parametrize("runtime", ["sglang", "vllm-omni"])
@pytest.mark.parametrize("model_name,gpus", [("h3", 2), ("wan", 8)])
def test_selected_server_seals_its_bound_execution_spec(tmp_path, monkeypatch, runtime, model_name, gpus):
    plan = h3_spec.plan_for(h3_spec.load_spec(PACKAGE / f"specs/{model_name}.yaml"), "release-v1")
    model = tmp_path / "model"
    model.mkdir()
    if model_name == "h3":
        (model / "FL2VA").mkdir()
        (model / "Ref2VA").mkdir()
        for name in ("model_index.json", "LICENSE", "README.md", "Ref2VA/model_index.json"):
            (model / name).write_text(name)
    uuids = [f"GPU-{index}" for index in range(gpus)]
    monkeypatch.setattr(mvp_gpu_job, "cuda_devices", lambda: uuids)
    executed = []
    monkeypatch.setattr(mvp_srt_selected.os, "execvpe", lambda file, argv, env: executed.append((file, argv)))
    output = tmp_path / "server"
    mvp_srt_selected.serve(
        plan, runtime=runtime, runtime_revision="a" * 40,
        model_path=model, port=30283, gpus=gpus, output=output,
    )
    spec = json.loads((output / "spec.json").read_text())
    launch = json.loads((output / "launch.json").read_text())
    assert spec["plan"] == plan and spec["gpu_uuids"] == launch["gpu_uuids"] == uuids
    assert spec["serving"] == {"mode": "closed_loop"}
    assert spec["baseline"]["name"] == launch["runtime"] == runtime
    assert spec["baseline"]["revision"] == launch["runtime_revision"] == "a" * 40
    assert spec["baseline"]["argv"] == launch["argv"] == executed[0][1]
    if runtime == "vllm-omni" and model_name == "h3":
        assert spec["baseline"]["model_path"] == executed[0][1][2]
    else:
        assert spec["baseline"]["model_path"] == str(model)
