"""The carried discovery patch works with the current srt-slurm checkout."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from infx.launch.drivers.srt.checkout import PATCHES, prepare_checkout
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
