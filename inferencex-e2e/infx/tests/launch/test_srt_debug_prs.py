"""The temporary upstream backport changes only a matching disposable checkout."""

import subprocess

import pytest

from infx.launch.context import LaunchError
from infx.launch.drivers.srt.checkout import SRT_DEBUG_PRS, apply_debug_prs


def test_pinned_debug_backport_applies_real_commit_and_rejects_conflict(tmp_path, monkeypatch):
    upstream = tmp_path / "upstream"
    upstream.mkdir()

    def git(*args, cwd=upstream):
        return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()

    git("init", "--quiet")
    git("config", "user.name", "Fixture")
    git("config", "user.email", "fixture@example.invalid")
    source = upstream / "feature.txt"
    source.write_text("binding: missing\n")
    git("add", "feature.txt")
    git("commit", "--quiet", "-m", "base")
    base = git("rev-parse", "HEAD")
    source.write_text("binding: resolved\n")
    git("commit", "--quiet", "-am", "upstream feature")
    patch = git("rev-parse", "HEAD")
    job = tmp_path / "job"
    git("clone", "--quiet", str(upstream), str(job))
    git("checkout", "--quiet", "--detach", base, cwd=job)
    monkeypatch.setitem(SRT_DEBUG_PRS, ("fixture", "vllm"), ((str(upstream), patch),))

    apply_debug_prs(job, "unrelated", "vllm")
    assert (job / "feature.txt").read_text() == "binding: missing\n"
    apply_debug_prs(job, "fixture", "vllm")
    assert (job / "feature.txt").read_text() == "binding: resolved\n"
    assert git("rev-parse", "HEAD", cwd=job) == base

    conflicting = tmp_path / "conflicting"
    git("clone", "--quiet", str(job), str(conflicting))
    (conflicting / "feature.txt").write_text("binding: incompatible\n")
    with pytest.raises(LaunchError, match="cherry-pick"):
        apply_debug_prs(conflicting, "fixture", "vllm")
    assert (conflicting / "feature.txt").read_text() == "binding: incompatible\n"
    assert source.read_text() == "binding: resolved\n"
