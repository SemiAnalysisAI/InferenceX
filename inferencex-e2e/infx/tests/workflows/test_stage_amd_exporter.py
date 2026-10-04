"""Prepared exporter bytes reach shared caches or the allocated host without a pull."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from infx.clusters.slurm import SquashPolicy
from infx.workflows.stage_amd_exporter import install_exporter, stage_exporter

ROOT = Path(__file__).resolve().parents[3]
IMAGE = "example.test/amd@sha256:abc"
SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


@pytest.fixture
def artifact(tmp_path):
    source = tmp_path / "prepared exporter"
    source.mkdir()
    (source / "amd-exporter.sqsh").write_bytes(b"abc")
    (source / "provenance.json").write_text(json.dumps({
        "image": {"upstream_reported_registry_digest": "sha256:abc"},
        "preparation": "retained evidence",
    }))
    (source / "SHA256SUMS").write_text(f"{SHA256}  amd-exporter.sqsh\n")
    return source


def test_shared_cache_publishes_verified_bytes_and_reuses_them(artifact, tmp_path):
    policy = SquashPolicy(tmp_path / "cache", "compute")
    receipt = tmp_path / "power-exporter-source.json"
    staged = stage_exporter(artifact, IMAGE, policy, SHA256, receipt)

    assert staged.destination.read_bytes() == b"abc"
    assert staged.node_local is False
    assert receipt.read_bytes() == (artifact / "provenance.json").read_bytes()
    original_inode = staged.destination.stat().st_ino
    stage_exporter(artifact, IMAGE, policy, SHA256, receipt)
    assert staged.destination.stat().st_ino == original_inode


def test_conflicting_cache_is_not_replaced(artifact, tmp_path):
    destination = tmp_path / "existing.sqsh"
    destination.write_bytes(b"another workload")
    with pytest.raises(ValueError, match="Refusing to replace"):
        install_exporter(artifact / "amd-exporter.sqsh", destination, SHA256)
    assert destination.read_bytes() == b"another workload"


@pytest.mark.parametrize("corruption", ["manifest", "pin", "registry-digest"])
def test_untrusted_preparation_fails_before_populating_cache(artifact, tmp_path, corruption):
    expected = SHA256
    image = IMAGE
    if corruption == "manifest":
        (artifact / "amd-exporter.sqsh").write_bytes(b"changed")
    elif corruption == "pin":
        expected = "0" * 64
    else:
        image = "example.test/amd@sha256:other"
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        stage_exporter(artifact, image, SquashPolicy(tmp_path / "cache", "compute"),
                       expected, tmp_path / "receipt.json")
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "receipt.json").exists()


@pytest.mark.parametrize("corruption", [None, "source", "destination", "missing-source"])
def test_node_local_hook_checks_source_and_target_before_exporter_start(
    artifact, tmp_path, corruption
):
    policy = SquashPolicy(tmp_path / "node cache", "all-nodes", visibility="node-local")
    staged = stage_exporter(artifact, IMAGE, policy, SHA256, tmp_path / "receipt.json")
    assert staged.node_local is True
    assert not staged.destination.exists()
    if corruption == "source":
        staged.source.write_bytes(b"changed after staging")
    elif corruption == "missing-source":
        staged.source.unlink()
    elif corruption == "destination":
        staged.destination.parent.mkdir()
        staged.destination.write_bytes(b"another workload")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (binaries / "python3").symlink_to(sys.executable)
    smi = binaries / "rocm-smi"
    smi.write_text("#!/bin/sh\necho 'MEC firmware version 177'\n")
    smi.chmod(0o755)
    result = subprocess.run(
        ["bash", str(ROOT / "runners/srt-slurm/hooks/mi300x-amd/setup.sh")],
        env={**os.environ, "PATH": f"{binaries}{os.pathsep}{os.environ['PATH']}",
             "AMD_DME_SOURCE": str(staged.source),
             "AMD_DME_DESTINATION": str(staged.destination),
             "AMD_DME_SHA256": SHA256,
             "AMD_DME_STAGE_SCRIPT": str(ROOT / "infx/workflows/stage_amd_exporter.py")},
        text=True, capture_output=True,
    )
    if corruption is None:
        assert result.returncode == 0, result.stderr
        assert staged.destination.read_bytes() == b"abc"
        assert "MEC firmware 177" in result.stdout
    else:
        assert result.returncode != 0
        assert "MEC firmware 177" not in result.stdout
        if corruption == "destination":
            assert staged.destination.read_bytes() == b"another workload"
        else:
            assert not staged.destination.exists()
