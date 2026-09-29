"""Execute the e2e get-jobs step against a measured checkout whose configs predate current code."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from infx.tests.historical_revision import POINTS, commit_history

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT.parent / ".github/workflows/e2e-tests.yml"


def run_get_jobs(tmp_path, *, launcher=True, **inputs):
    """Run the step from a dispatch whose measured ref predates this workflow's tooling."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    base, head = commit_history(workspace, "inferencex-e2e", launcher=launcher)
    (workspace / ".ci-priority").mkdir()
    (workspace / ".ci-priority/inferencex-e2e").symlink_to(ROOT, target_is_directory=True)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    # uv provisions the interpreter; the test environment already has its dependencies.
    (fake_bin / "uv").write_text(
        f'#!/bin/sh\nwhile [ "$1" != python ]; do shift; done\nshift\nexec "{sys.executable}" "$@"\n'
    )
    (fake_bin / "uv").chmod(0o755)
    outputs = tmp_path / "github-output"
    step = next(
        step for step in yaml.safe_load(WORKFLOW.read_text())["jobs"]["get-jobs"]["steps"]
        if step.get("id") == "get-jobs"
    )
    env = {
        **os.environ,
        "PATH": os.pathsep.join([str(fake_bin), str(Path(sys.executable).parent), os.environ["PATH"]]),
        "GITHUB_WORKSPACE": str(workspace),
        "GITHUB_OUTPUT": str(outputs),
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_RUN_ID": "1",
        "GITHUB_RUN_ATTEMPT": "1",
        # The workflow-level import path names the measured checkout, never the tooling.
        "PYTHONPATH": str(workspace / "inferencex-e2e"),
        "GENERATE_COMMAND": "",
        "PR_LABELS": "[]",
        "CHANGELOG_BASE_REF": "",
        "CHANGELOG_HEAD_REF": "",
        "TRIM_CONC": "false",
        "ALL_EVALS": "false",
        "EVALS_ONLY": "false",
        **{key: value.format(base=base, head=head) for key, value in inputs.items()},
    }
    result = subprocess.run(
        ["bash", "-e", "-c", step["run"]], cwd=workspace, env=env,
        capture_output=True, text=True, timeout=120,
    )
    jobs = {
        key: json.loads(value)
        for key, _, value in (
            line.partition("=") for line in (outputs.read_text() if outputs.exists() else "").splitlines()
        )
    }
    return result, jobs


@pytest.mark.parametrize("inputs", [
    {"CHANGELOG_BASE_REF": "{base}", "CHANGELOG_HEAD_REF": "{head}"},
    {"GENERATE_COMMAND": "test-config --config-keys fixture --config-files "
                         "configs/amd-master.yaml configs/nvidia-master.yaml"},
], ids=["changelog-dispatch", "generate-command"])
def test_get_jobs_runs_the_measured_revisions_own_tooling(tmp_path, inputs):
    result, jobs = run_get_jobs(tmp_path, **inputs)

    assert result.returncode == 0, result.stderr
    assert [(row["model"], row["conc"], row["image"]) for row in jobs["single-node-config"]] == POINTS
    assert {key: rows for key, rows in jobs.items() if key != "single-node-config"} == {
        key: [] for key in (
            "agentic-config", "agentic-eval-config", "multi-node-agentic-config",
            "multi-node-agentic-eval-config", "multi-node-config", "eval-config",
            "multi-node-eval-config",
        )
    }


def test_get_jobs_rejects_a_revision_its_gpu_jobs_could_not_launch(tmp_path):
    result, jobs = run_get_jobs(tmp_path, launcher=False, CHANGELOG_BASE_REF="{base}", CHANGELOG_HEAD_REF="{head}")

    assert result.returncode == 1
    assert "predates the Python launcher (infx/launch)" in result.stderr
    assert jobs == {}
