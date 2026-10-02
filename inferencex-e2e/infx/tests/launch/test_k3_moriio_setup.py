"""The temporary worker backport must not execute in a standalone router image."""

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "benchmarks/multi_node/srt-slurm-recipes/configs/k3-moriio-debug.sh"
)


@pytest.mark.parametrize(
    ("router", "worker"), [(True, False), (False, True), (True, True), (False, False)]
)
def test_only_standalone_router_skips_engine_setup(tmp_path, router, worker):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name, present in (("vllm-router", router), ("vllm", worker)):
        if present:
            binary = binaries / name
            binary.write_text("#!/bin/bash\nexit 0\n")
            binary.chmod(0o755)
    # A failed package lookup must remain fatal for workers or an unknown image.
    python = binaries / "python3"
    python.write_text("#!/bin/bash\necho 'package lookup failed' >&2\nexit 7\n")
    python.chmod(0o755)
    completed = subprocess.run(
        ["/bin/bash", str(SCRIPT)],
        env={"PATH": str(binaries), "LANG": os.environ.get("LANG", "C")},
        capture_output=True,
        text=True,
        timeout=10,
    )
    if router and not worker:
        assert completed.returncode == 0
        assert "no engine backport required" in completed.stdout
        assert completed.stderr == ""
    else:
        assert completed.returncode == 7
        assert "package lookup failed" in completed.stderr
        assert "no engine backport required" not in completed.stdout


@pytest.mark.parametrize("failure", [None, "checksum", "context"])
def test_verified_patch_changes_only_vllm_and_rejects_invalid_input(tmp_path, failure):
    package_root = tmp_path / "site-packages"
    package = package_root / "vllm"
    package.mkdir(parents=True)
    worker = package / "worker.py"
    initial = "VALUE = 9\n" if failure == "context" else "VALUE = 1\n"
    worker.write_text(initial)
    unrelated = package_root / "other.txt"
    unrelated.write_text("untouched\n")
    patch = tmp_path / "backport.patch"
    patch.write_text(
        "diff --git a/vllm/worker.py b/vllm/worker.py\n"
        "--- a/vllm/worker.py\n+++ b/vllm/worker.py\n@@ -1 +1 @@\n"
        "-VALUE = 1\n+VALUE = 2\n"
        "diff --git a/other.txt b/other.txt\n"
        "--- a/other.txt\n+++ b/other.txt\n@@ -1 +1 @@\n"
        "-untouched\n+changed\n"
    )
    checksum = hashlib.sha256(patch.read_bytes()).hexdigest()
    if failure == "checksum":
        patch.write_text(patch.read_text() + "corrupted\n")
    completed = subprocess.run(
        [
            "/bin/bash",
            "-c",
            'source "$1"; apply_verified_patch "$2" "$3" "$4"',
            "test",
            str(SCRIPT),
            str(package_root),
            str(patch),
            checksum,
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert unrelated.read_text() == "untouched\n"
    if failure is None:
        assert completed.returncode == 0, completed.stderr
        assert worker.read_text() == "VALUE = 2\n"
    else:
        assert completed.returncode != 0
        assert worker.read_text() == initial
