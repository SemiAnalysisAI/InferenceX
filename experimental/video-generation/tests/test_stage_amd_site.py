"""Synthetic CPU preparation tests; no AMD compatibility evidence."""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

import ci
import stage_amd_site as stage
from test_mvp_gpu_job import spec  # noqa: F401


def inspection():
    return {"status": "inspected", "source_revision": stage.REVISION,
            "probe": {"python": "/usr/bin/python3", "torch_hip": "synthetic",
                      "imports": {name: {"path": "/synthetic/" + name} for name in
                                  ("torch", "torchvision", "av", "numpy", "diffusers", "transformers", "sglang", "aiter", "triton", "amdsmi")},
                      "torch_devices": [{"name": "AMD Instinct MI355X"}] * 8,
                      "hip_devices": [{"hip_ordinal": i} for i in range(8)]}}


@pytest.mark.parametrize("fault", ["import", "hip", "devices", "revision"])
def test_unready_runtime_is_rejected(fault):
    record = inspection()
    if fault == "import":
        record["probe"]["imports"]["aiter"] = {"error": "missing"}
    elif fault == "hip":
        record["probe"]["torch_hip"] = None
    elif fault == "devices":
        record["probe"]["hip_devices"].pop()
    else:
        record["source_revision"] = "wrong"
    with pytest.raises(ValueError):
        stage.runtime_probe(record)


def test_seals_matched_workload_and_full_billed_allocation_without_gpu_calls(spec, tmp_path, monkeypatch):
    workspace = tmp_path
    monkeypatch.setattr(stage, "WORKSPACE", workspace)
    control = workspace / "campaigns/h3-cross-hardware"
    control.mkdir(parents=True)
    rootfs = workspace.parent / "enroot-data" / stage.CONTAINER
    rootfs.mkdir(parents=True)
    record = inspection()
    record.update(rootfs=str(rootfs), entrypoint='exec bash "$@"')
    ci.write(control / "runtime-inspected.json", record)
    entries = spec["model"]["files"]
    ci.write(control / "model-ready.json", {"status": "complete", "model_path": spec["model"]["path"],
             "model_revision": spec["model"]["revision"],
             "manifest_sha256": hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
    source = workspace / ("runtime-sglang-" + stage.REVISION)
    source.mkdir()
    monkeypatch.setattr(stage, "source_file_manifest", lambda p: {"revision": stage.REVISION, "source_sha256": "b" * 64})
    monkeypatch.setattr(ci, "allocate", lambda *a: pytest.fail("CPU preparation must not allocate"))
    output = tmp_path / "output"
    output.mkdir()
    result = stage.stage(spec, output)
    config = ci.read(output / "site.json")
    frozen = ci.read(output / "gpu-spec.json")
    assert result["generation_executed"] is False
    assert config["resources"]["gpus"] == 4
    assert config["resources"]["allocated_gpus"] == 8
    assert config["concurrencies"] == [1]
    assert len(frozen["plan"]["cases"]) == 20
    assert frozen["plan"]["generation"]["duration_seconds"] == 8
    assert frozen["server"]["attention_backend"] == "aiter"
    assert frozen["limits"]["job_seconds"] + 600 == config["resources"]["minutes"] * 60
    subprocess.run(["bash", "-n", str(output / "entry-only.sh")], check=True)
    with pytest.raises(ValueError, match="already exist"):
        stage.stage(spec, output)
