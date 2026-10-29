"""The native YAML route preserves the selected request and model-view contract."""

import json
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

from evaluator import cli, gpu_job, srt_selected  # noqa: E402
from evaluator.runner import preview_plan  # noqa: E402
from specs import h3_spec  # noqa: E402


def test_h3_selected_plan_drives_both_request_mappings():
    path = PACKAGE / "specs/h3.yaml"
    spec = h3_spec.load_spec(path)
    expected = h3_spec.plan_sha256(h3_spec.plan_for(spec, "release-v1"))
    plan = srt_selected.selected_plan(path, "release-v1", expected)
    sglang = preview_plan(plan, runtime="sglang")
    vllm = preview_plan(plan, runtime="vllm-omni")
    assert sglang["plan_sha256"] == vllm["plan_sha256"] == expected
    assert sglang["measurement_count"] == vllm["measurement_count"] == 20
    assert sglang["total_requests"] == vllm["total_requests"] == 21
    assert sglang["slots"][1]["request"]["prompt"] == vllm["slots"][1]["request"]["prompt"]
    with pytest.raises(ValueError, match="plan_sha256"):
        srt_selected.selected_plan(path, "release-v1", "0" * 64)


def test_h3_vllm_model_view_exposes_only_tested_partition(tmp_path):
    snapshot = tmp_path / "snapshot"
    (snapshot / "FL2VA").mkdir(parents=True)
    (snapshot / "Ref2VA").mkdir()
    for name in ("model_index.json", "LICENSE", "README.md", "Ref2VA/model_index.json", "modular_model_index.json"):
        (snapshot / name).write_text(name)
    view = srt_selected.h3_vllm_model_view(snapshot, tmp_path / "view")
    assert (view / "FL2VA").resolve() == (snapshot / "FL2VA").resolve()
    assert (view / "Ref2VA/model_index.json").read_text() == "Ref2VA/model_index.json"
    assert not (view / "modular_model_index.json").exists()


@pytest.mark.parametrize(("runtime", "model_view"), [("sglang", "snapshot"), ("vllm-omni", "h3-fl2va")])
@pytest.mark.parametrize("recipe_metadata", [True, False])
def test_selected_server_executes_recipe_command_and_records_selected_environment(tmp_path, monkeypatch, runtime, model_view, recipe_metadata):
    plan = h3_spec.plan_for(h3_spec.load_spec(PACKAGE / "specs/h3.yaml"), "release-v1")
    model = tmp_path / "model"
    (model / "FL2VA").mkdir(parents=True)
    (model / "Ref2VA").mkdir()
    for name in ("model_index.json", "LICENSE", "README.md", "Ref2VA/model_index.json"):
        (model / name).write_text(name)
    uuids = ["GPU-1", "GPU-2"]
    monkeypatch.setattr(gpu_job, "cuda_devices", lambda: uuids)
    executed = []
    monkeypatch.setattr(srt_selected.os, "execvpe", lambda file, argv, env: executed.append((file, argv, env)))
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("VLLM_OMNI_VIDEO_SYNC_TIMEOUT", "777")
    if recipe_metadata:
        monkeypatch.setenv("H3_RECIPE_ENV_KEYS", '["HF_HUB_OFFLINE", "VLLM_OMNI_VIDEO_SYNC_TIMEOUT"]')
    else:
        monkeypatch.delenv("H3_RECIPE_ENV_KEYS", raising=False)
    monkeypatch.setenv("VLLM_OMNI_SERVER_STORAGE__PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("SECRET_TOKEN", "never-record")
    output = tmp_path / "server"
    command = (["/usr/bin/python3", "-m", "evaluator.srt_selected", "serve", "--model-path", str(model), "--tp-size", "3"]
               if runtime == "sglang" else ["vllm", "serve", str(model), "--tensor-parallel-size", "3"])
    srt_selected.serve(
        plan, runtime=runtime, runtime_revision="a" * 40,
        model_path=model, port=30283, gpus=2, output=output,
        command=command, model_view=model_view,
    )
    spec = json.loads((output / "spec.json").read_text())
    launch = json.loads((output / "launch.json").read_text())
    assert spec["plan"] == plan and spec["gpu_uuids"] == launch["gpu_uuids"] == uuids
    assert spec["serving"] == {"mode": "closed_loop"}
    assert spec["baseline"]["name"] == launch["runtime"] == runtime
    assert spec["baseline"]["revision"] == launch["runtime_revision"] == "a" * 40
    assert spec["baseline"]["argv"] == launch["argv"] == executed[0][1]
    assert executed[0][2]["HF_HUB_OFFLINE"] == "0"
    assert executed[0][2]["VLLM_OMNI_VIDEO_SYNC_TIMEOUT"] == "777"
    assert launch["environment"]["HF_HUB_OFFLINE"] == "0"
    assert launch["environment"]["VLLM_OMNI_VIDEO_SYNC_TIMEOUT"] == "777"
    assert "H3_RECIPE_ENV_KEYS" not in executed[0][2]
    assert "SECRET_TOKEN" not in launch["environment"]
    assert executed[0][2]["VGBENCH_GPU_UUIDS"] == "GPU-1,GPU-2"
    if runtime == "vllm-omni":
        assert spec["baseline"]["model_path"] == executed[0][1][2]
        assert (tmp_path / "storage").is_dir()
        assert executed[0][1][2] != str(model)
    else:
        assert spec["baseline"]["model_path"] == str(model)
        assert executed[0][1] == command


def test_selected_server_requires_snapshot_argument_and_explicit_view(tmp_path, monkeypatch):
    plan = h3_spec.plan_for(h3_spec.load_spec(PACKAGE / "specs/h3.yaml"), "release-v1")
    model = tmp_path / "model"
    model.mkdir()
    monkeypatch.setattr(gpu_job, "cuda_devices", lambda: ["GPU-1", "GPU-2"])
    with pytest.raises(ValueError, match="model snapshot"):
        srt_selected.serve(plan, runtime="sglang", runtime_revision="a" * 40,
            model_path=model, port=30283, gpus=2, output=tmp_path / "server",
            command=["python3", "serve"], model_view="snapshot")
    with pytest.raises(ValueError, match="model view"):
        srt_selected.serve(plan, runtime="sglang", runtime_revision="a" * 40,
            model_path=model, port=30283, gpus=2, output=tmp_path / "server",
            command=["python3", str(model)], model_view="h3-fl2va")


def test_selected_server_rejects_wrong_model_and_gpu_shape(tmp_path):
    plan = h3_spec.plan_for(h3_spec.load_spec(PACKAGE / "specs/h3.yaml"), "release-v1")
    model = tmp_path / "model"
    model.mkdir()
    kwargs = {"runtime_revision": "a" * 40, "model_path": model, "port": 30283,
              "output": tmp_path / "server", "command": ["vllm", "serve", str(model)],
              "model_view": "snapshot"}
    with pytest.raises(ValueError, match="requires MiniMaxAI/MiniMax-H3"):
        srt_selected.serve({**plan, "model_id": "other/model"}, runtime="sglang", gpus=2, **kwargs)
    with pytest.raises(ValueError, match="two GPUs"):
        srt_selected.serve(plan, runtime="vllm-omni", gpus=4, **kwargs)
    with pytest.raises(ValueError, match="two or four GPUs"):
        srt_selected.serve(plan, runtime="sglang", gpus=1, **kwargs)


def test_vllm_view_checks_snapshot_before_creating_runtime_directories(tmp_path, monkeypatch):
    plan = h3_spec.plan_for(h3_spec.load_spec(PACKAGE / "specs/h3.yaml"), "release-v1")
    model = tmp_path / "model"
    model.mkdir()
    storage = tmp_path / "storage"
    output = tmp_path / "server"
    monkeypatch.setenv("VLLM_OMNI_SERVER_STORAGE__PATH", str(storage))
    monkeypatch.setenv("H3_RECIPE_ENV_KEYS", '["VLLM_OMNI_SERVER_STORAGE__PATH"]')
    monkeypatch.setattr(gpu_job, "cuda_devices", lambda: ["GPU-1", "GPU-2"])
    with pytest.raises(ValueError, match="snapshot lacks"):
        srt_selected.serve(plan, runtime="vllm-omni", runtime_revision="a" * 40,
            model_path=model, port=30283, gpus=2, output=output,
            command=["vllm", "serve", str(model)], model_view="h3-fl2va")
    assert not storage.exists() and not output.exists()


def test_selected_cli_forwards_engine_command_after_separator(tmp_path, monkeypatch):
    path = PACKAGE / "specs/h3.yaml"
    plan = h3_spec.plan_for(h3_spec.load_spec(path), "release-v1")
    digest = h3_spec.plan_sha256(plan)
    seen = []
    monkeypatch.setattr(srt_selected, "serve", lambda *args, **kwargs: seen.append(kwargs))
    command = ["python3", "-m", "evaluator.srt_selected", "serve", "--tp-size", "3"]
    assert cli.main(["srt-yaml-serve", "--spec", str(path), "--set", "release-v1",
        "--plan-sha256", digest, "--runtime", "sglang", "--runtime-revision", "a" * 40,
        "--model-path", "/model", "--model-view", "snapshot", "--port", "30283",
        "--gpus", "2", "--record", str(tmp_path), "--", *command]) == 0
    assert seen[0]["command"] == command and seen[0]["model_view"] == "snapshot"


def test_selected_client_cli_passes_explicit_request_timeout(tmp_path, monkeypatch):
    path = PACKAGE / "specs/h3.yaml"
    plan = h3_spec.plan_for(h3_spec.load_spec(path), "release-v1")
    seen = []
    monkeypatch.setattr(srt_selected, "run_matrix", lambda *args, **kwargs: (
        seen.append(kwargs) or {1: {"summary": {"valid": 1, "scheduled": 1}, "status": "complete"}}))
    assert cli.main(["srt-yaml-client", "--spec", str(path), "--set", "release-v1",
        "--plan-sha256", h3_spec.plan_sha256(plan), "--runtime", "sglang",
        "--runtime-revision", "a" * 40, "--endpoint", "http://localhost:30283",
        "--output", str(tmp_path), "--concurrencies", "1", "--gpus", "2",
        "--window-dir", str(tmp_path / "windows"), "--request-seconds", "1200"]) == 0
    assert seen[0]["request_seconds"] == 1200


@pytest.mark.parametrize("seconds", [0, -1, float("nan")])
def test_selected_client_rejects_invalid_request_timeout(tmp_path, seconds):
    with pytest.raises(ValueError, match="positive finite"):
        srt_selected.run_matrix({}, runtime="sglang", runtime_revision="a" * 40,
            endpoint="http://localhost:30283", output=tmp_path, concurrencies=[1],
            gpus=2, window_dir=tmp_path / "windows", request_seconds=seconds)


def test_sglang_child_rechecks_uuid_order_before_import(monkeypatch):
    monkeypatch.setenv("VGBENCH_GPU_UUIDS", "GPU-1,GPU-2")
    monkeypatch.setattr(gpu_job, "cuda_devices", lambda: ["GPU-2", "GPU-1"])
    with pytest.raises(RuntimeError, match="UUIDs differ"):
        srt_selected._sglang_main()
