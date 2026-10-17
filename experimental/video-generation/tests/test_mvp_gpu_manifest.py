"""CPU-only file inventories; fake weight bytes are never model evidence."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from evaluator.cli import main
from evaluator import mvp_gpu_manifest
from evaluator.mvp_gpu_manifest import build_gpu_manifest, write_gpu_manifest
from evaluator.mvp_gpu_job import validate_gpu_job


REVISION = "a" * 40
ROOT = Path(__file__).resolve().parents[1]


def test_unconfigured_job_template_is_not_approved_or_executable_and_plan_does_not_drift():
    template = json.loads((ROOT / "mvp" / "h3-gpu-job.template.json").read_text())
    plan = json.loads((ROOT / "mvp" / "h3-smoke.plan.json").read_text())
    assert template["plan"] == plan
    assert template["authorization"] == {
        "compute_approved": False, "model_license_reviewed": False, "approval_reference": "",
    }
    assert template["model"]["files"] == []
    assert template["policy"]["calibration_status"] == "uncalibrated"
    with pytest.raises(ValueError):
        validate_gpu_job(template)


def model_tree(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "weights.safetensors").write_bytes(b"CPU fixture bytes, not real weights")
    (root / "config.json").write_text("{}")
    return root


def test_model_inventory_is_sorted_complete_and_has_no_execution_claim(tmp_path):
    root = model_tree(tmp_path)
    result = build_gpu_manifest("model", root, model_revision=REVISION)
    assert result["evidence_kind"] == "staged_files_only_no_gpu_execution"
    assert [item["path"] for item in result["files"]] == ["config.json", "weights.safetensors"]
    assert result["files"][0]["sha256"] == hashlib.sha256(b"{}").hexdigest()
    assert result["total_bytes"] == sum(item["size_bytes"] for item in result["files"])
    assert len(result["manifest_sha256"]) == 64
    assert len(list(root.iterdir())) == 2


def test_hf_file_symlink_to_own_blob_is_supported_but_external_links_are_refused(tmp_path):
    hub = tmp_path / "models--fixture"
    root = hub / "snapshots" / REVISION
    root.mkdir(parents=True)
    blobs = hub / "blobs"
    blobs.mkdir()
    blob = blobs / "blob"
    blob.write_bytes(b"fake fixture")
    (root / "weights.safetensors").symlink_to(blob)
    result = build_gpu_manifest("model", root, model_revision=REVISION)
    assert result["files"][0]["sha256"] == hashlib.sha256(b"fake fixture").hexdigest()
    (root / "bad.bin").symlink_to(tmp_path / "outside.bin")
    (tmp_path / "outside.bin").write_bytes(b"outside")
    with pytest.raises(ValueError, match="escapes"):
        build_gpu_manifest("model", root, model_revision=REVISION)


def test_directory_symlinks_are_not_silently_omitted(tmp_path):
    root = model_tree(tmp_path)
    nested = tmp_path / "nested"
    nested.mkdir()
    (root / "linked").symlink_to(nested, target_is_directory=True)
    with pytest.raises(ValueError, match="directory symlinks"):
        build_gpu_manifest("model", root, model_revision=REVISION)


def test_symlinked_hf_blob_directory_cannot_expand_the_allowed_cache(tmp_path):
    cache = tmp_path / "cache"
    root = cache / "snapshots" / REVISION
    root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "weight").write_bytes(b"outside fixture")
    (cache / "blobs").symlink_to(outside, target_is_directory=True)
    (root / "weights.safetensors").symlink_to(cache / "blobs" / "weight")
    with pytest.raises(ValueError, match="blob directory must not be a symlink"):
        build_gpu_manifest("model", root, model_revision=REVISION)


def test_named_pipe_is_refused_before_open(tmp_path):
    root = model_tree(tmp_path)
    os.mkfifo(root / "stream.bin")
    with pytest.raises(ValueError, match="regular files"):
        build_gpu_manifest("model", root, model_revision=REVISION)


@pytest.mark.parametrize("replacement", ["symlink", "fifo", "file"])
def test_file_replacement_is_rejected_before_reading_bytes(tmp_path, monkeypatch, replacement):
    root = model_tree(tmp_path)
    target = root / "weights.safetensors"
    outside = tmp_path / "outside"
    outside.write_bytes(b"must not read outside file")
    original_open = os.open
    replaced = False

    def changed_open(path, flags, *args, **kwargs):
        nonlocal replaced
        if path == target.name and not replaced:
            replaced = True
            target.unlink()
            if replacement == "symlink":
                target.symlink_to(outside)
            elif replacement == "fifo":
                os.mkfifo(target)
            else:
                target.write_bytes(b"changed regular-file content")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", changed_open)
    with pytest.raises((ValueError, OSError)):
        build_gpu_manifest("model", root, model_revision=REVISION)
    assert replaced


@pytest.mark.parametrize("revision", [None, "main", "a" * 39, "A" * 40])
def test_model_revision_is_explicit_and_immutable(tmp_path, revision):
    root = model_tree(tmp_path)
    with pytest.raises(ValueError, match="immutable"):
        build_gpu_manifest("model", root, model_revision=revision)


@pytest.mark.parametrize("timeout", [True, 0, -1, float("nan"), float("inf"), 7201])
def test_invalid_deadline_is_refused(tmp_path, timeout):
    with pytest.raises(ValueError, match="timeout"):
        build_gpu_manifest("model", tmp_path, model_revision=REVISION, timeout_seconds=timeout)


def test_expired_deadline_stops_before_hashing(tmp_path, monkeypatch):
    root = model_tree(tmp_path)
    ticks = iter([0.0, 1.0])
    monkeypatch.setattr(mvp_gpu_manifest.time, "monotonic", lambda: next(ticks))
    with pytest.raises(TimeoutError, match="deadline"):
        build_gpu_manifest("model", root, model_revision=REVISION, timeout_seconds=0.1)


def test_output_cannot_pollute_source_or_overwrite_evidence(tmp_path):
    root = model_tree(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        write_gpu_manifest("model", root, root / "manifest.json", model_revision=REVISION)
    existing = tmp_path / "existing.json"
    existing.write_text("keep me")
    with pytest.raises(FileExistsError):
        write_gpu_manifest("model", root, existing, model_revision=REVISION)
    assert existing.read_text() == "keep me"


def test_cli_writes_new_model_manifest_and_prints_compact_summary(tmp_path, capsys):
    root = model_tree(tmp_path)
    output = tmp_path / "manifest.json"
    assert main(["gpu-manifest", "model", str(root), "--model-revision", REVISION,
                 "--output", str(output)]) == 0
    summary = json.loads(capsys.readouterr().out)
    result = json.loads(output.read_text())
    assert "files" not in summary
    assert len(result["files"]) == 2
    assert summary["manifest_sha256"] == result["manifest_sha256"]


def test_runtime_inventory_uses_committed_content_and_rejects_dirty_source(tmp_path):
    root = tmp_path / "runtime"
    package = root / "python" / "sglang"
    (package / "cli").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "cli" / "main.py").write_text("# CPU fixture only\n")
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "-c", "commit.gpgsign=false", "commit", "-m", "CPU-only fixture"], check=True, capture_output=True)
    result = build_gpu_manifest("runtime", root)
    assert len(result["revision"]) == 40
    assert len(result["source_sha256"]) == 64
    assert len(result["files"]) == 2
    (package / "cli" / "main.py").write_text("# modified fixture\n")
    with pytest.raises(ValueError, match="clean committed"):
        build_gpu_manifest("runtime", root)
