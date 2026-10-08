"""Job-local patches leave the shared source and recorded submodule revision intact."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from infx.launch.context import LaunchError
from infx.launch.drivers.srt.checkout import PATCHES, SUBMODULE, prepare_checkout
from infx.launch.drivers.srt.recipe import RECIPES_MIRROR

ROOT = Path(__file__).resolve().parents[3]


def test_job_checkout_renders_discovery_templates(tmp_path, monkeypatch):
    """Run upstream's real P/D rendering regressions after applying the carried patches.

    A subprocess keeps the job-local backend separate from the unpatched srtctl
    that other tests may already have imported. No cluster or install is needed.
    """
    workspace = tmp_path / "workspace"
    patches = workspace / PATCHES
    patches.parent.mkdir(parents=True)
    patches.symlink_to(ROOT / PATCHES, target_is_directory=True)
    recipes = workspace / RECIPES_MIRROR
    (recipes / "configs").mkdir(parents=True)
    (recipes / "recipe.yaml").write_text("name: test\n")
    monkeypatch.setenv("INFERENCEX_REPOSITORY_ROOT", str(ROOT))

    checkout = prepare_checkout(SimpleNamespace(workspace=workspace), tmp_path / "job", power=False)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_discovery_connector_templates.py"],
        cwd=checkout.root,
        env={**os.environ, "PYTHONPATH": str(checkout.root / "src")},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("conflict", [False, True])
def test_checkout_applies_local_patch_or_stops_before_staging(tmp_path, monkeypatch, conflict):
    workspace = tmp_path / "workspace"
    source = workspace / SUBMODULE
    source.mkdir(parents=True)

    def git(*args, cwd=source):
        return subprocess.run(
            ["git", *args], cwd=cwd, check=True, capture_output=True, text=True,
        ).stdout.strip()

    git("init", "--quiet")
    (source / "feature.txt").write_text("binding: original\n")
    git("add", "feature.txt")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "--quiet", "-m", "base")
    base = git("rev-parse", "HEAD")
    patches = workspace / PATCHES
    patches.mkdir(parents=True)
    before = "incompatible" if conflict else "original"
    (patches / "binding.patch").write_text(
        "--- a/feature.txt\n+++ b/feature.txt\n@@ -1 +1 @@\n"
        f"-binding: {before}\n+binding: resolved\n"
    )
    recipes = workspace / RECIPES_MIRROR
    (recipes / "configs").mkdir(parents=True)
    (recipes / "recipe.yaml").write_text("name: test\n")
    monkeypatch.setenv("INFERENCEX_REPOSITORY_ROOT", str(workspace))
    destination = tmp_path / "job"
    run = SimpleNamespace(workspace=workspace)

    if conflict:
        with pytest.raises(LaunchError, match="git -C failed"):
            prepare_checkout(run, destination, power=True)
        assert (destination / "feature.txt").read_text() == "binding: original\n"
        assert not (destination / "recipes").exists()
        assert not (workspace / "srt-slurm-sha.txt").exists()
    else:
        checkout = prepare_checkout(run, destination, power=True)
        assert checkout.commit == base
        assert (destination / "feature.txt").read_text() == "binding: resolved\n"
        assert (destination / "recipes/recipe.yaml").read_text() == "name: test\n"
        assert (workspace / "srt-slurm-sha.txt").read_text() == f"{base}\n"
        assert (workspace / "power-producer-sha.txt").read_text() == f"{base}\n"

    assert git("rev-parse", "HEAD", cwd=destination) == base
    assert (source / "feature.txt").read_text() == "binding: original\n"
    assert git("status", "--porcelain") == ""
