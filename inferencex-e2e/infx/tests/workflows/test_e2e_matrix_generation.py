"""Execute the e2e get-jobs step against a measured checkout whose configs predate current code."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from infx.tests.historical_revision import POINTS, commit_history

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT.parent / ".github/workflows/e2e-tests.yml"
GENERATE = "test-config --config-keys fixture --config-files configs/amd-master.yaml configs/nvidia-master.yaml"
# Shadows the standard json package from the measured checkout and records which process loaded it.
SHADOW_JSON = '''\
import importlib.util, os, sys, sysconfig
if os.environ.get("MEASURED_JSON_MARKER"):
    with open(os.environ["MEASURED_JSON_MARKER"], "a") as handle:
        handle.write(" ".join(sys.argv) + "\\n")
package = os.path.join(sysconfig.get_paths()["stdlib"], "json")
spec = importlib.util.spec_from_file_location(
    "json", os.path.join(package, "__init__.py"), submodule_search_locations=[package])
real = importlib.util.module_from_spec(spec)
sys.modules["json"] = real
spec.loader.exec_module(real)
'''


def run_get_jobs(tmp_path, *, launcher=True, measured_files=None, **inputs):
    """Run the step from a dispatch whose measured ref predates this workflow's tooling."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    base, head = commit_history(workspace, "inferencex-e2e", launcher=launcher)
    for relative, text in (measured_files or {}).items():
        (workspace / relative).write_text(text)
    (workspace / ".ci-priority").mkdir()
    (workspace / ".ci-priority/inferencex-e2e").symlink_to(ROOT, target_is_directory=True)
    (tmp_path / "runner-temp").mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "uv").write_text(
        f'#!/bin/sh\nprintf \'%s\\t%s\\n\' "$PWD" "$*" >> "{tmp_path / "uv-calls.log"}"\n'
        f'while [ "$1" != python ]; do shift; done\nshift\nexec "{sys.executable}" "$@"\n'
    )
    (fake_bin / "uv").chmod(0o755)
    # The trusted tree is this checkout, whose srt-slurm the test job already initialized;
    # parallel tests must not race on its submodule config.
    (fake_bin / "git").write_text(
        f'#!/bin/sh\ncase " $* " in *" submodule update "*) exit 0 ;; esac\nexec "{shutil.which("git")}" "$@"\n'
    )
    (fake_bin / "git").chmod(0o755)
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
        "RUNNER_TEMP": str(tmp_path / "runner-temp"),
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_RUN_ID": "1",
        "GITHUB_RUN_ATTEMPT": "1",
        "PYTHONPATH": str(workspace / "inferencex-e2e"),
        "GENERATE_COMMAND": "",
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


DISPATCHES = pytest.mark.parametrize("inputs", [
    {"CHANGELOG_BASE_REF": "{base}", "CHANGELOG_HEAD_REF": "{head}"},
    {"GENERATE_COMMAND": GENERATE},
], ids=["changelog-dispatch", "generate-command"])


@DISPATCHES
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


@DISPATCHES
def test_trusted_tools_run_outside_the_measured_checkout_without_uv_config(tmp_path, inputs):
    result, _ = run_get_jobs(tmp_path, **inputs)

    assert result.returncode == 0, result.stderr
    calls = [line.split("\t", 1) for line in (tmp_path / "uv-calls.log").read_text().splitlines()]
    workspace = (tmp_path / "workspace").resolve()
    assert calls
    assert [(cwd, args) for cwd, args in calls
            if Path(cwd).resolve().is_relative_to(workspace) or "--no-config" not in args.split()] == []


def test_a_measured_json_module_cannot_rewrite_the_dispatched_matrix(tmp_path):
    marker = tmp_path / "measured-json-imports"
    result, jobs = run_get_jobs(
        tmp_path, measured_files={"inferencex-e2e/json.py": SHADOW_JSON},
        GENERATE_COMMAND=GENERATE, MEASURED_JSON_MARKER=str(marker),
    )

    assert result.returncode == 0, result.stderr
    assert not marker.exists(), marker.read_text()
    assert [(row["model"], row["conc"], row["image"]) for row in jobs["single-node-config"]] == POINTS


def test_get_jobs_rejects_a_revision_its_gpu_jobs_could_not_launch(tmp_path):
    result, jobs = run_get_jobs(tmp_path, launcher=False, CHANGELOG_BASE_REF="{base}", CHANGELOG_HEAD_REF="{head}")

    assert result.returncode == 1
    assert "predates the Python launcher (infx/launch)" in result.stderr
    assert jobs == {}
