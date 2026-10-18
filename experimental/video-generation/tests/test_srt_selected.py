"""The native YAML route preserves the selected request and model-view contract."""

import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

from evaluator import mvp_srt_selected  # noqa: E402
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
