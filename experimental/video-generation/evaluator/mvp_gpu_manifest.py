"""Read-only file inventory preparation for a controlled GPU job.

This module never imports a model, starts a server, downloads files, or probes a
GPU. Hashing a large staged snapshot can still cause substantial disk I/O. The
manifest establishes file identity, not model permission or runtime correctness.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .mvp_gpu_job import source_file_manifest
from .mvp_runner import canonical_json_bytes


def _deadline_check(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("file inventory exceeded its read-only preparation deadline")


@contextmanager
def _open_regular(path: Path):
    """Resolve the already-approved absolute path without following new links."""
    descriptors = []
    try:
        current = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        descriptors.append(current)
        for part in path.parts[1:-1]:
            current = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
            descriptors.append(current)
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=current)
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("model inventory accepts only regular files")
            yield stream
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _model_inventory(root: Path, revision: str, deadline: float) -> dict[str, Any]:
    blob_root = None
    if root.parent.name == "snapshots" and root.name == revision:
        possible = root.parent.parent / "blobs"
        if possible.is_symlink():
            raise ValueError("HF blob directory must not be a symlink outside its cache")
        if possible.is_dir():
            blob_root = possible.resolve(strict=True)

    names: list[str] = []

    def walk_error(error: OSError) -> None:
        raise error

    for directory, directories, files in os.walk(root, followlinks=False, onerror=walk_error):
        _deadline_check(deadline)
        if any((Path(directory) / name).is_symlink() for name in directories):
            raise ValueError("model inventory does not follow directory symlinks")
        names.extend((Path(directory) / name).relative_to(root).as_posix() for name in files)
        if len(names) > 20000:
            raise ValueError("model inventory exceeds 20000 files")
    if not names or not any(name.endswith((".safetensors", ".pt", ".bin")) for name in names):
        raise ValueError("staged model directory contains no supported weight files")

    entries = []
    for name in sorted(names):
        _deadline_check(deadline)
        if "\\" in name or "\x00" in name:
            raise ValueError("model file name is not supported by the GPU job contract")
        target = (root / name).resolve(strict=True)
        if not (target.is_relative_to(root) or (blob_root and target.is_relative_to(blob_root))):
            raise ValueError("model file escapes the staged snapshot and its HF blob directory")
        before = target.stat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("model inventory accepts only regular files")
        digest = hashlib.sha256()
        count = 0
        with _open_regular(target) as stream:
            opened = os.fstat(stream.fileno())
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns
            ):
                raise ValueError("model file changed before its inventory read")
            while True:
                _deadline_check(deadline)
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                count += len(chunk)
                digest.update(chunk)
            after = os.fstat(stream.fileno())
        final_path = target.lstat()
        if count != before.st_size or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ) or (final_path.st_dev, final_path.st_ino) != (after.st_dev, after.st_ino):
            raise ValueError("model file changed while preparing its inventory")
        entries.append({"path": name, "size_bytes": count, "sha256": digest.hexdigest()})
    return {
        "path": str(root), "revision": revision, "files": entries,
        "manifest_sha256": hashlib.sha256(canonical_json_bytes(entries)).hexdigest(),
        "total_bytes": sum(item["size_bytes"] for item in entries),
    }


def build_gpu_manifest(kind: str, directory: Path, *, model_revision: str | None = None,
                       timeout_seconds: float = 600) -> dict[str, Any]:
    """Hash an operator-selected staged tree; inputs are never modified."""
    if kind not in {"runtime", "model"}:
        raise ValueError("manifest kind must be runtime or model")
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or not 0.1 <= timeout_seconds <= 7200):
        raise ValueError("manifest timeout must be finite and between 0.1 and 7200 seconds")
    if kind == "model" and (not isinstance(model_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", model_revision)):
        raise ValueError("model inventory requires an explicit immutable --model-revision")
    if kind == "runtime" and model_revision is not None:
        raise ValueError("--model-revision applies only to model inventories")
    deadline = time.monotonic() + timeout_seconds
    root = Path(directory).resolve(strict=True)
    if not root.is_dir() or root == Path(root.anchor):
        raise ValueError("inventory path must name a specific staged directory, not the filesystem root")
    if kind == "runtime":
        inventory = source_file_manifest(root, timeout=min(10, timeout_seconds), deadline=deadline)
    else:
        inventory = _model_inventory(root, model_revision, deadline)
    _deadline_check(deadline)
    return {
        "schema_version": "0.1.0", "bundle_type": "gpu_preparation_manifest",
        "evidence_kind": "staged_files_only_no_gpu_execution", "kind": kind,
        "created_at": datetime.now(timezone.utc).isoformat(), **inventory,
    }


def write_gpu_manifest(kind: str, directory: Path, output: Path, *, model_revision: str | None = None,
                       timeout_seconds: float = 600) -> dict[str, Any]:
    """Write a new manifest outside the frozen input; refuse existing targets."""
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("manifest output must be a new file")
    source = Path(directory).resolve(strict=True)
    resolved_output = output.resolve(strict=False)
    if resolved_output.is_relative_to(source):
        raise ValueError("write the manifest outside the frozen input directory")
    for component in (output.parent, *output.parent.parents):
        if component.is_symlink():
            raise ValueError("manifest output parents cannot be symlinks")
    result = build_gpu_manifest(kind, source, model_revision=model_revision, timeout_seconds=timeout_seconds)
    # Creating a new, exclusive file never replaces existing evidence.
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return result
