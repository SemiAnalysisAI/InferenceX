"""CPU-only file inventories; fake weight bytes are never model evidence."""

import hashlib
import os

import pytest

from evaluator.gpu_manifest import build_gpu_manifest, write_gpu_manifest


REVISION = "a" * 40


def model_tree(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "weights.safetensors").write_bytes(b"CPU fixture bytes, not real weights")
    (root / "config.json").write_text("{}")
    return root


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


def test_output_cannot_pollute_source_or_overwrite_evidence(tmp_path):
    root = model_tree(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        write_gpu_manifest("model", root, root / "manifest.json", model_revision=REVISION)
    existing = tmp_path / "existing.json"
    existing.write_text("keep me")
    with pytest.raises(FileExistsError):
        write_gpu_manifest("model", root, existing, model_revision=REVISION)
    assert existing.read_text() == "keep me"
