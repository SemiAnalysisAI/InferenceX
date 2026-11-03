"""The native YAML route preserves the selected request and model-view contract."""

import json
import sys
from pathlib import Path

import pytest
import yaml

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

from evaluator import cli, srt_selected  # noqa: E402
from specs import h3_spec  # noqa: E402


def _snapshot(path):
    (path / "FL2VA").mkdir(parents=True)
    (path / "Ref2VA").mkdir()
    for name in ("model_index.json", "LICENSE", "README.md", "Ref2VA/model_index.json"):
        (path / name).write_text(name)
    return path


def test_selected_plan_binds_requested_set_and_digest(tmp_path):
    spec = h3_spec.load_spec(PACKAGE / "specs/h3.yaml")
    selected = spec["sets"]["release-v1"]
    selected.update(plan_id="selected-control", repetitions=2, warmup_runs=0)
    selected["cases"] = selected["cases"][:1]
    selected["seed_sources"] = {
        case["seed"]: selected["seed_sources"][case["seed"]] for case in selected["cases"]
    }
    path = tmp_path / "controlled.yaml"
    path.write_text(yaml.safe_dump(spec))
    expected = h3_spec.plan_sha256(h3_spec.plan_for(spec, "release-v1"))
    plan = srt_selected.selected_plan(path, "release-v1", expected)
    assert plan["plan_id"] == "selected-control"
    assert (len(plan["cases"]), plan["repetitions"], plan["warmup_runs"]) == (1, 2, 0)
    with pytest.raises(ValueError, match="plan_sha256"):
        srt_selected.selected_plan(path, "release-v1", "0" * 64)


def test_h3_vllm_model_view_exposes_only_tested_partition(tmp_path):
    snapshot = _snapshot(tmp_path / "snapshot")
    (snapshot / "modular_model_index.json").write_text("modular_model_index.json")
    view = srt_selected.h3_vllm_model_view(snapshot, tmp_path / "view")
    assert (view / "FL2VA").resolve() == (snapshot / "FL2VA").resolve()
    assert (view / "Ref2VA/model_index.json").read_text() == "Ref2VA/model_index.json"
    assert not (view / "modular_model_index.json").exists()


@pytest.mark.parametrize(("runtime", "model_view"), [("sglang", "snapshot"), ("vllm-omni", "h3-fl2va")])
def test_selected_server_executes_recipe_command_and_records_selected_environment(tmp_path, monkeypatch, runtime, model_view):
    plan = h3_spec.plan_for(h3_spec.load_spec(PACKAGE / "specs/h3.yaml"), "release-v1")
    model = _snapshot(tmp_path / "model")
    uuids = ["GPU-1", "GPU-2"]
    monkeypatch.setattr(srt_selected, "cuda_devices", lambda: uuids)
    executed = []
    monkeypatch.setattr(srt_selected.os, "execvpe", lambda file, argv, env: executed.append((file, argv, env)))
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("VLLM_OMNI_VIDEO_SYNC_TIMEOUT", "777")
    monkeypatch.setenv("H3_RECIPE_ENV_KEYS", '["HF_HUB_OFFLINE", "VLLM_OMNI_VIDEO_SYNC_TIMEOUT"]')
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
    monkeypatch.setattr(srt_selected, "cuda_devices", lambda: ["GPU-1", "GPU-2"])
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


def test_v01_vllm_four_gpu_server_records_sealed_launch(tmp_path, monkeypatch):
    plan = h3_spec.plan_for(h3_spec.load_spec(PACKAGE / "specs/h3-v0.1.yaml"), "release-v1")
    model = _snapshot(tmp_path / "model")
    uuids = [f"GPU-{index}" for index in range(4)]
    monkeypatch.setattr(srt_selected, "cuda_devices", lambda: uuids)
    monkeypatch.setenv("VLLM_OMNI_SERVER_STORAGE__PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("H3_CONTAINER_IMAGE", "example/vllm:v0.30.0@sha256:" + "a" * 64)
    monkeypatch.setenv("H3_RECIPE_ENV_KEYS", '["VLLM_OMNI_SERVER_STORAGE__PATH"]')
    executed = []
    monkeypatch.setattr(srt_selected.os, "execvpe", lambda file, argv, env: executed.append(argv))
    command = ["vllm", "serve", str(model), "--num-gpus", "4", "--usp", "4", "--vae-patch-parallel-size", "4"]
    srt_selected.serve(plan, runtime="vllm-omni", runtime_revision="a" * 40,
        model_path=model, port=30283, gpus=4, output=tmp_path / "server",
        command=command, model_view="h3-fl2va")
    launch = json.loads((tmp_path / "server/launch.json").read_text())
    assert launch["gpu_uuids"] == uuids
    assert launch["image_digest"] == "a" * 64
    assert launch["argv"] == executed[0]
    with pytest.raises(ValueError, match="two GPUs or four for v0.1"):
        srt_selected.serve(plan, runtime="vllm-omni", runtime_revision="a" * 40,
            model_path=model, port=30283, gpus=3, output=tmp_path / "unused",
            command=command, model_view="h3-fl2va")
    monkeypatch.setenv("H3_CONTAINER_IMAGE", "example/vllm:v0.30.0")
    with pytest.raises(ValueError, match="pinned by a SHA256 digest"):
        srt_selected.serve(plan, runtime="vllm-omni", runtime_revision="a" * 40,
            model_path=model, port=30283, gpus=4, output=tmp_path / "unsealed",
            command=command, model_view="h3-fl2va")


def test_v01_client_requires_matching_sealed_launch_before_requests(tmp_path, monkeypatch):
    plan = h3_spec.plan_for(h3_spec.load_spec(PACKAGE / "specs/h3-v0.1.yaml"), "release-v1")
    seen = []

    def stop_after_preflight(*args, **kwargs):
        seen.append(kwargs)
        raise RuntimeError("preflight reached runner")

    monkeypatch.setattr(srt_selected, "run_plan", stop_after_preflight)
    kwargs = {"runtime": "vllm-omni", "runtime_revision": "a" * 40,
              "endpoint": "http://localhost:30283", "output": tmp_path,
              "concurrencies": [1], "gpus": 4, "window_dir": tmp_path / "window",
              "request_seconds": 1200}
    with pytest.raises(ValueError, match="sealed server launch"):
        srt_selected.run_matrix(plan, **kwargs)
    assert not seen
    (tmp_path / "server").mkdir()
    launch = {"runtime": "vllm-omni", "runtime_revision": "a" * 40,
              "model_revision": plan["model_revision"], "plan_sha256": h3_spec.plan_sha256(plan),
              "image_digest": "a" * 64,
              "gpu_uuids": [f"GPU-{index}" for index in range(4)],
              "argv": ["vllm", "serve", "/model", "--usp", "4"],
              "environment": {"OMNI_DIFFUSION_PROMPT_EMBED_CACHE": "0"}}
    (tmp_path / "server/launch.json").write_text(json.dumps(launch))
    with pytest.raises(RuntimeError, match="preflight reached runner"):
        srt_selected.run_matrix(plan, **kwargs)
    identity = seen[0]["numerical_identity"]
    assert identity["server_argv"] == launch["argv"]
    assert identity["image_digest"] == "a" * 64
    assert identity["server_environment"] == launch["environment"]
    assert identity["gpu_topology"] == {"gpu_uuids": launch["gpu_uuids"]}
    launch["plan_sha256"] = "0" * 64
    (tmp_path / "server/launch.json").write_text(json.dumps(launch))
    with pytest.raises(ValueError, match="differs from the selected"):
        srt_selected.run_matrix(plan, **kwargs)
    assert len(seen) == 1
    launch["plan_sha256"] = h3_spec.plan_sha256(plan)
    launch.pop("image_digest")
    (tmp_path / "server/launch.json").write_text(json.dumps(launch))
    seen.clear()
    with pytest.raises(RuntimeError, match="preflight reached runner"):
        srt_selected.run_matrix(plan, **kwargs)
    assert seen[0]["numerical_identity"] is None


def test_vllm_view_checks_snapshot_before_creating_runtime_directories(tmp_path, monkeypatch):
    plan = h3_spec.plan_for(h3_spec.load_spec(PACKAGE / "specs/h3.yaml"), "release-v1")
    model = tmp_path / "model"
    model.mkdir()
    storage = tmp_path / "storage"
    output = tmp_path / "server"
    monkeypatch.setenv("VLLM_OMNI_SERVER_STORAGE__PATH", str(storage))
    monkeypatch.setenv("H3_RECIPE_ENV_KEYS", '["VLLM_OMNI_SERVER_STORAGE__PATH"]')
    monkeypatch.setattr(srt_selected, "cuda_devices", lambda: ["GPU-1", "GPU-2"])
    with pytest.raises(ValueError, match="snapshot lacks"):
        srt_selected.serve(plan, runtime="vllm-omni", runtime_revision="a" * 40,
            model_path=model, port=30283, gpus=2, output=output,
            command=["vllm", "serve", str(model)], model_view="h3-fl2va")
    assert not storage.exists() and not output.exists()


def test_selected_cli_forwards_engine_command_after_separator(tmp_path, monkeypatch):
    path = PACKAGE / "specs/h3.yaml"
    plan = h3_spec.plan_for(h3_spec.load_spec(path), "release-v1")
    digest = h3_spec.plan_sha256(plan)
    model = tmp_path / "model"
    model.mkdir()
    output = tmp_path / "server"
    monkeypatch.setattr(srt_selected, "cuda_devices", lambda: ["GPU-1", "GPU-2"])
    monkeypatch.setattr(srt_selected.os, "execvpe", lambda *args: None)
    command = ["python3", "-m", "engine", "--model-path", str(model), "--tp-size", "3"]
    assert cli.main(["srt-yaml-serve", "--spec", str(path), "--set", "release-v1",
        "--plan-sha256", digest, "--runtime", "sglang", "--runtime-revision", "a" * 40,
        "--model-path", str(model), "--model-view", "snapshot", "--port", "30283",
        "--gpus", "2", "--record", str(output), "--", *command]) == 0
    launch = json.loads((output / "launch.json").read_text())
    assert launch["argv"] == command
    assert launch["plan_sha256"] == digest


def test_sglang_child_rechecks_uuid_order_before_import(monkeypatch):
    monkeypatch.setenv("VGBENCH_GPU_UUIDS", "GPU-1,GPU-2")
    monkeypatch.setattr(srt_selected, "cuda_devices", lambda: ["GPU-2", "GPU-1"])
    with pytest.raises(RuntimeError, match="UUIDs differ"):
        srt_selected._sglang_main()
