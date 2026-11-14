"""The native YAML route preserves the selected request and model-view contract."""

import json
import sys
from pathlib import Path

import pytest
import yaml

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

from evaluator import srt_selected  # noqa: E402
from specs import video_spec  # noqa: E402


def _snapshot(path):
    (path / "FL2VA").mkdir(parents=True)
    (path / "Ref2VA").mkdir()
    for name in ("model_index.json", "LICENSE", "README.md", "Ref2VA/model_index.json"):
        (path / name).write_text(name)
    return path


def test_selected_plan_binds_requested_set_and_digest(tmp_path):
    spec = video_spec.load_spec(PACKAGE / "specs/h3.yaml")
    selected = spec["sets"]["release-v1"]
    selected.update(plan_id="selected-control", repetitions=2, warmup_runs=0)
    selected["cases"] = selected["cases"][:1]
    path = tmp_path / "controlled.yaml"
    path.write_text(yaml.safe_dump(spec))
    expected = video_spec.plan_sha256(video_spec.plan_for(spec, "release-v1"))
    plan = srt_selected.selected_plan(path, "release-v1", expected)
    assert plan["plan_id"] == "selected-control"
    assert (len(plan["cases"]), plan["repetitions"], plan["warmup_runs"]) == (1, 2, 0)
    with pytest.raises(ValueError, match="plan_sha256"):
        srt_selected.selected_plan(path, "release-v1", "0" * 64)


def test_vllm_model_view_exposes_only_tested_partition(tmp_path):
    snapshot = _snapshot(tmp_path / "snapshot")
    (snapshot / "modular_model_index.json").write_text("modular_model_index.json")
    view = srt_selected.vllm_model_view(snapshot, tmp_path / "view")
    assert (view / "FL2VA").resolve() == (snapshot / "FL2VA").resolve()
    assert (view / "Ref2VA/model_index.json").read_text() == "Ref2VA/model_index.json"
    assert not (view / "modular_model_index.json").exists()


@pytest.mark.parametrize(
    ("runtime", "model_view"), [("sglang", "snapshot"), ("vllm-omni", "h3-fl2va")]
)
def test_selected_server_executes_recipe_command_and_records_selected_environment(
    tmp_path, monkeypatch, runtime, model_view
):
    plan = video_spec.plan_for(video_spec.load_spec(PACKAGE / "specs/h3.yaml"), "release-v1")
    model = _snapshot(tmp_path / "model")
    uuids = ["GPU-1", "GPU-2"]
    monkeypatch.setattr(srt_selected, "cuda_devices", lambda: uuids)
    monkeypatch.setattr(srt_selected, "KFD", tmp_path / "no-kfd")
    executed = []
    monkeypatch.setattr(
        srt_selected.os, "execvpe", lambda file, argv, env: executed.append((file, argv, env))
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("VLLM_OMNI_VIDEO_SYNC_TIMEOUT", "777")
    monkeypatch.setenv(
        "VIDEO_RECIPE_ENV_KEYS", '["HF_HUB_OFFLINE", "VLLM_OMNI_VIDEO_SYNC_TIMEOUT"]'
    )
    monkeypatch.setenv("VLLM_OMNI_SERVER_STORAGE__PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("SECRET_TOKEN", "never-record")
    monkeypatch.setenv("VIDEO_CONTAINER_IMAGE", "example/engine:v1@sha256:" + "b" * 64)
    output = tmp_path / "server"
    command = (
        [
            "/usr/bin/python3",
            "-m",
            "evaluator.srt_selected",
            "serve",
            "--model-path",
            str(model),
            "--tp-size",
            "1",
            "--ulysses-degree",
            "2",
            "--disable-conditioning-cache",
            "--dit-cpu-offload",
            "false",
        ]
        if runtime == "sglang"
        else [
            "vllm",
            "serve",
            str(model),
            "--tensor-parallel-size",
            "1",
            "--usp",
            "2",
            "--ring",
            "1",
        ]
    )
    srt_selected.serve(
        plan,
        runtime=runtime,
        runtime_revision="a" * 40,
        model_path=model,
        gpus=2,
        output=output,
        command=command,
        model_view=model_view,
    )
    spec = json.loads((output / "spec.json").read_text())
    launch = json.loads((output / "launch.json").read_text())
    assert spec["plan"] == plan and spec["gpu_uuids"] == launch["gpu_uuids"] == uuids
    assert spec["serving"] == {"mode": "closed_loop"}
    assert spec["server"] == (
        {"tp_size": 1, "ulysses_degree": 2, "dit_cpu_offload": False}
        if runtime == "sglang"
        else {"tp_size": 1, "ulysses_degree": 2}
    )
    assert spec["baseline"]["name"] == launch["runtime"] == runtime
    assert spec["baseline"]["revision"] == launch["runtime_revision"] == "a" * 40
    assert spec["baseline"]["argv"] == launch["argv"] == executed[0][1]
    assert executed[0][2]["HF_HUB_OFFLINE"] == "0"
    assert executed[0][2]["VLLM_OMNI_VIDEO_SYNC_TIMEOUT"] == "777"
    assert launch["environment"]["HF_HUB_OFFLINE"] == "0"
    assert launch["environment"]["VLLM_OMNI_VIDEO_SYNC_TIMEOUT"] == "777"
    assert "VIDEO_RECIPE_ENV_KEYS" not in executed[0][2]
    assert "SECRET_TOKEN" not in launch["environment"]
    assert executed[0][2]["VGBENCH_GPU_UUIDS"] == "GPU-1,GPU-2"
    assert launch["image_digest"] == "b" * 64
    if runtime == "vllm-omni":
        assert spec["baseline"]["model_path"] == executed[0][1][2]
        assert (tmp_path / "storage").is_dir()
        assert executed[0][1][2] != str(model)
    else:
        assert spec["baseline"]["model_path"] == str(model)
        assert executed[0][1] == command


def test_selected_server_requires_the_recipe_gpus_and_runtime(tmp_path, monkeypatch):
    plan = video_spec.plan_for(video_spec.load_spec(PACKAGE / "specs/h3.yaml"), "release-v1")
    model = _snapshot(tmp_path / "model")
    monkeypatch.setattr(srt_selected, "cuda_devices", lambda: ["GPU-1", "GPU-2"])
    monkeypatch.setattr(srt_selected, "KFD", tmp_path / "no-kfd")
    kwargs = {
        "runtime_revision": "a" * 40,
        "model_path": model,
        "output": tmp_path / "server",
        "command": ["python3", "-m", "engine", str(model)],
        "model_view": "snapshot",
    }
    with pytest.raises(RuntimeError, match="visible GPU count"):
        srt_selected.serve(plan, runtime="sglang", gpus=4, **kwargs)
    with pytest.raises(ValueError, match="unsupported native video runtime"):
        srt_selected.serve(plan, runtime="tgi", gpus=2, **kwargs)
    with pytest.raises(ValueError, match="model snapshot path exactly once"):
        srt_selected.serve(plan, runtime="sglang", gpus=2, **{**kwargs, "command": ["python3"]})
    assert not (tmp_path / "server").exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (None, None, "requires the sealed server launch"),
        ("runtime", "sglang", "runtime differs"),
        ("model_revision", "c" * 40, "model_revision differs"),
        ("plan_sha256", "0" * 64, "plan_sha256 differs"),
        ("gpu_uuids", ["GPU-0", "GPU-1"], "must record 4 GPU UUIDs"),
        ("image_digest", None, "complete matching server configuration"),
    ],
)
def test_client_rejects_a_missing_or_mismatched_sealed_launch_before_requests(
    tmp_path, field, value, message
):
    plan = video_spec.plan_for(video_spec.load_spec(PACKAGE / "specs/h3-v2.yaml"), "release-v1")
    (tmp_path / "server").mkdir()
    if field is not None:
        launch = {
            "runtime": "vllm-omni",
            "runtime_revision": "a" * 40,
            "model_revision": plan["model_revision"],
            "plan_sha256": video_spec.plan_sha256(plan),
            "image_digest": "a" * 64,
            "gpu_uuids": [f"GPU-{index}" for index in range(4)],
            "argv": ["vllm", "serve", "/model"],
            "environment": {"OMNI_DIFFUSION_PROMPT_EMBED_CACHE": "0"},
            field: value,
        }
        (tmp_path / "server/launch.json").write_text(json.dumps(launch))
    with pytest.raises(ValueError, match=message):
        srt_selected.run_matrix(
            plan,
            runtime="vllm-omni",
            runtime_revision="a" * 40,
            endpoint="http://localhost:30283",
            output=tmp_path,
            concurrencies=[1],
            gpus=4,
            window_dir=tmp_path / "window",
            request_seconds=1200,
        )
    assert sorted(path.name for path in tmp_path.iterdir()) == ["server"]


def test_rocm_server_names_hip_devices_by_board_serial(tmp_path, monkeypatch):
    plan = video_spec.plan_for(video_spec.load_spec(PACKAGE / "specs/h3-v2.yaml"), "release-v1")
    model = _snapshot(tmp_path / "model")
    kfd = tmp_path / "kfd"
    kfd.touch()
    buses = ["0000:f5:00.0", "0000:05:00.0"]
    for bus, serial in zip(buses, ["PCB-F5", "PCB-05"], strict=True):
        (tmp_path / "pci" / bus).mkdir(parents=True)
        (tmp_path / "pci" / bus / "serial_number").write_text(serial + "\n")

    class Hip:
        def hipGetDeviceCount(self, count):
            count._obj.value = len(buses)
            return 0

        def hipDeviceGetPCIBusId(self, bus, length, ordinal):
            bus.value = buses[ordinal].encode()
            return 0

    monkeypatch.setattr(srt_selected, "KFD", kfd)
    monkeypatch.setattr(srt_selected, "PCI_DEVICES", tmp_path / "pci")
    monkeypatch.setattr(srt_selected.ctypes, "CDLL", {"libamdhip64.so": Hip()}.__getitem__)
    executed = []
    monkeypatch.setattr(
        srt_selected.os, "execvpe", lambda file, argv, env: executed.append((argv, env))
    )
    monkeypatch.setenv("VIDEO_CONTAINER_IMAGE", "example/sglang:rocm@sha256:" + "c" * 64)
    output = tmp_path / "server"
    srt_selected.serve(
        plan,
        runtime="sglang",
        runtime_revision="a" * 40,
        model_path=model,
        gpus=2,
        output=output,
        command=["/opt/venv/bin/python3", "-m", "evaluator.srt_selected", "serve", str(model)],
        model_view="snapshot",
    )
    spec = json.loads((output / "spec.json").read_text())
    launch = json.loads((output / "launch.json").read_text())
    assert spec["gpu_uuids"] == launch["gpu_uuids"] == ["PCB-F5", "PCB-05"]
    assert executed[0][1]["VGBENCH_GPU_UUIDS"] == "PCB-F5,PCB-05"


def test_sol_engine_serves_the_snapshot_with_its_recipe_command(tmp_path, monkeypatch):
    plan = video_spec.plan_for(video_spec.load_spec(PACKAGE / "specs/h3-v2.yaml"), "release-v1")
    model = _snapshot(tmp_path / "model")
    monkeypatch.setattr(srt_selected, "cuda_devices", lambda: ["GPU-1", "GPU-2"])
    monkeypatch.setattr(srt_selected, "KFD", tmp_path / "no-kfd")
    executed = []
    monkeypatch.setattr(
        srt_selected.os, "execvpe", lambda file, argv, env: executed.append((file, argv, env))
    )
    monkeypatch.setenv("VIDEO_CONTAINER_IMAGE", "example/sol-engine:v1@sha256:" + "c" * 64)
    command = [
        "/opt/sol-engine/venv/bin/python3",
        "/opt/sol-engine/docker/gb200/serve.py",
        "--model-path",
        str(model),
        "--num-gpus",
        "2",
        "--attention",
        "sol",
        "--disable-conditioning-cache",
    ]
    output = tmp_path / "server"
    srt_selected.serve(
        plan,
        runtime="sol-engine",
        runtime_revision="d" * 40,
        model_path=model,
        gpus=2,
        output=output,
        command=command,
        model_view="snapshot",
    )
    spec = json.loads((output / "spec.json").read_text())
    launch = json.loads((output / "launch.json").read_text())
    assert spec["server"] == {"attention_backend": "sol"}
    assert spec["baseline"]["name"] == launch["runtime"] == "sol-engine"
    assert spec["baseline"]["model_path"] == str(model)
    assert executed[0][1] == launch["argv"] == command
    with pytest.raises(ValueError, match="for vllm-omni"):
        srt_selected.serve(
            plan,
            runtime="sol-engine",
            runtime_revision="d" * 40,
            model_path=model,
            gpus=2,
            output=tmp_path / "fl2va-view",
            command=command,
            model_view="h3-fl2va",
        )
