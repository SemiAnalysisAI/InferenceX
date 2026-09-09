"""Site inspection must not acquire resources or query GPU devices."""

from pathlib import Path
from types import SimpleNamespace

import pytest

import site_preflight


@pytest.mark.parametrize("available", [False, True])
def test_preflight_reads_only_public_keys_and_keeps_missing_data_explicit(monkeypatch, available):
    key = Path("/etc/ssh/ssh_host_ed25519_key.pub")
    monkeypatch.setattr(Path, "is_file", lambda path: available and path == key)
    reads = []
    def read(path):
        reads.append(path)
        return "ssh-ed25519 public-test-key\n"
    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(site_preflight.shutil, "which", lambda name: f"/bin/{name}" if available else None)
    commands = []
    def run(argv, **kwargs):
        commands.append(argv)
        return SimpleNamespace(stdout="", returncode=1)
    monkeypatch.setattr(site_preflight.subprocess, "run", run)
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    result = site_preflight.inspect_site()
    assert result["gpu_execution"] is False
    assert result["runtime_compatibility"] == "not_tested"
    assert result["ci"]["GITHUB_SHA"] == "a" * 40
    assert result["previously_known_amd_jumpbox"] is None
    assert result["ssh_host_public_keys"] == ({"ed25519": "ssh-ed25519 public-test-key"} if available else {})
    assert reads == ([key] if available else [])
    assert commands == ([["ssh-keygen", "-F", "64.139.223.123"]] if available else [])
