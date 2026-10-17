"""CPU-only download tests; fixture bytes are not model or benchmark evidence."""
from io import BytesIO
import json
from pathlib import Path

import pytest

import ci
import stage_model_ci as stage
from test_mvp_gpu_job import spec  # noqa: F401


def test_reuses_verified_model_without_downloading(spec, tmp_path, monkeypatch):
    original = Path(spec["model"]["path"])
    monkeypatch.setattr(stage, "urlopen", lambda *a, **kw: pytest.fail("verified weights must be reused"))
    workspace = tmp_path / "work"
    result = stage.stage_model(spec, workspace, [original])
    assert result["model_path"] == str(original)
    assert result["verified_files"] == 1
    assert not workspace.exists()


@pytest.mark.parametrize("content", [b"", b"wrong bytes", b"x" * 100])
def test_incomplete_or_wrong_download_never_becomes_ready(spec, tmp_path, monkeypatch, content):
    monkeypatch.setattr(stage, "urlopen", lambda *a, **kw: BytesIO(content))
    target = tmp_path / "destination"
    with pytest.raises(ValueError, match="frozen size|frozen size or SHA256"):
        stage.fetch_weight(target, spec["model"]["revision"], spec["model"]["files"][0])
    assert not (target / "weights.safetensors").exists()


def test_preparation_retains_source_and_writes_persistent_receipt(spec, tmp_path, monkeypatch):
    content = (Path(spec["model"]["path"]) / "weights.safetensors").read_bytes()
    monkeypatch.setattr(stage, "urlopen", lambda *a, **kw: BytesIO(content))
    monkeypatch.setattr(stage, "verified_execution", lambda run: ({"databaseId": 123}, {"name": "h3-video-123-1"}))
    commands = []
    def download(argv, **kwargs):
        commands.append(argv)
        root = Path(argv[argv.index("--dir") + 1])
        path = root / "gpu/c1/spec.json"
        path.parent.mkdir(parents=True)
        ci.write(path, spec)
        (root / "SHA256SUMS").write_text(ci.digest(path) + "  gpu/c1/spec.json\n")
    monkeypatch.setattr(stage.subprocess, "run", download)
    workspace, output = tmp_path / "work", tmp_path / "output"
    stage.prepare("123", workspace, output)
    receipt = json.loads((workspace / "campaigns/h3-cross-hardware/model-ready.json").read_text())
    assert receipt["status"] == "complete"
    assert receipt["gpu_allocation"] is False
    assert receipt["source_ci"]["databaseId"] == 123
    assert receipt["total_bytes"] == len(content)
    assert (Path(receipt["model_path"]) / "weights.safetensors").read_bytes() == content
    assert [command[0] for command in commands] == ["gh"]
