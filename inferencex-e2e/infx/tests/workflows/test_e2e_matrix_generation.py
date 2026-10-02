"""Execute the e2e and profile matrix steps against a measured checkout whose configs predate current code."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from infx.tests.historical_revision import KEY, MASTER, POINTS, commit_history

ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT.parent / ".github/workflows"
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
# Names a recipe the preflight cannot find, so no matrix with this point may be dispatched.
STALE_MASTER = yaml.safe_dump({KEY: {**MASTER[KEY], "srt-recipe": "benchmarks/single_node/srt-slurm-recipes/absent.yaml"}})
# How the runner executes a step's script for each `shell:` value.
SHELLS = {None: "bash -e {0}", "bash": "bash --noprofile --norc -eo pipefail {0}", "python": "python {0}"}


def measured_checkout(tmp_path, *, launcher=True, measured_files=None):
    """A measured workspace whose trusted tooling is this checkout, with uv and git calls recorded."""
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
    # The trusted tree is this checkout, whose srt-slurm the test job already initialized;
    # parallel tests must not race on its submodule config.
    (fake_bin / "git").write_text(
        f'#!/bin/sh\ncase " $* " in *" submodule update "*) echo "$*" >> "{tmp_path / "git-submodules.log"}"; exit 0 ;; esac\n'
        f'exec "{shutil.which("git")}" "$@"\n'
    )
    for tool in ("uv", "git"):
        (fake_bin / tool).chmod(0o755)
    env = {
        **os.environ,
        "PATH": os.pathsep.join([str(fake_bin), str(Path(sys.executable).parent), os.environ["PATH"]]),
        "GITHUB_WORKSPACE": str(workspace),
        "RUNNER_TEMP": str(tmp_path / "runner-temp"),
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_RUN_ID": "1",
        "GITHUB_RUN_ATTEMPT": "1",
        "PYTHONPATH": str(workspace / "inferencex-e2e"),
    }
    return workspace, base, head, env


def workflow_step(workflow, job, step_id):
    return next(
        step for step in yaml.safe_load((WORKFLOWS / workflow).read_text())["jobs"][job]["steps"]
        if step.get("id") == step_id
    )


def run_step(step, workspace, env, tmp_path):
    """Run ``step``'s script with its shell, from the workspace, as the runner does."""
    script = tmp_path / f"{step['id']}.step"
    script.write_text(step["run"])
    template = SHELLS.get(step.get("shell"), step.get("shell"))
    command = [sys.executable if word == "python" else word for word in template.format(script).split()]
    return subprocess.run(command, cwd=workspace, env=env, capture_output=True, text=True, timeout=120)


def step_outputs(path):
    return dict(line.partition("=")[::2] for line in (path.read_text() if path.exists() else "").splitlines())


def uv_calls(tmp_path):
    """``(cwd, module, arguments)`` of each recorded uv call, in order."""
    calls = []
    for line in (tmp_path / "uv-calls.log").read_text().splitlines():
        cwd, _, args = line.partition("\t")
        words = args.split()
        calls.append((cwd, words[words.index("-m") + 1], words))
    return calls


def outside_without_config(calls, workspace):
    """The calls that run in the measured checkout or let uv read configuration."""
    return [(cwd, words) for cwd, _, words in calls
            if Path(cwd).resolve().is_relative_to(workspace.resolve()) or "--no-config" not in words]


def trusted_submodule_update(workspace):
    return [f"-C {workspace}/.ci-priority/inferencex-e2e submodule update --init utils/srt-slurm"]


def run_get_jobs(tmp_path, *, launcher=True, measured_files=None, **inputs):
    """Run e2e's get-jobs step from a dispatch whose measured ref predates this workflow's tooling."""
    workspace, base, head, env = measured_checkout(tmp_path, launcher=launcher, measured_files=measured_files)
    outputs = tmp_path / "github-output"
    env = {
        **env,
        "GITHUB_OUTPUT": str(outputs),
        "GENERATE_COMMAND": "",
        "CHANGELOG_BASE_REF": "",
        "CHANGELOG_HEAD_REF": "",
        "TRIM_CONC": "false",
        "ALL_EVALS": "false",
        "EVALS_ONLY": "false",
        **{key: value.format(base=base, head=head) for key, value in inputs.items()},
    }
    result = run_step(workflow_step("e2e-tests.yml", "get-jobs", "get-jobs"), workspace, env, tmp_path)
    return result, {key: json.loads(value) for key, value in step_outputs(outputs).items()}


def run_profile_matrix(tmp_path, *, measured_files=None, filter_files=None, **inputs):
    """Run profile's gen step, then its filter step on gen's output unless gen failed."""
    workspace, _, _, env = measured_checkout(tmp_path, measured_files=measured_files)
    env = {**env, "INPUTS_CONFIG_FILE": "configs/nvidia-master.yaml", "INPUTS_CONFIG_KEY": KEY,
           "INPUTS_CONC": "2", "PRIORITY_CRITERIA": "", **inputs}
    gen_outputs = tmp_path / "gen-output"
    gen = run_step(workflow_step("profile.yml", "get-jobs", "gen"), workspace,
                   {**env, "GITHUB_OUTPUT": str(gen_outputs)}, tmp_path)
    if gen.returncode:
        return gen, None, {}
    for relative, text in (filter_files or {}).items():
        (workspace / relative).write_text(text)
    filter_outputs = tmp_path / "filter-output"
    raw = step_outputs(gen_outputs)["raw"]
    filtered = run_step(workflow_step("profile.yml", "get-jobs", "filter"), workspace,
                        {**env, "GITHUB_OUTPUT": str(filter_outputs), "STEPS_GEN_OUTPUTS_RAW": raw}, tmp_path)
    return gen, filtered, step_outputs(filter_outputs)


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
    calls = uv_calls(tmp_path)
    assert [module for _, module, _ in calls[:3]] == [
        "infx.matrix.revision", "infx.workflows.benchmark_schema", "infx.srt_slurm.preflight",
    ]
    assert {module for _, module, _ in calls[3:]} == {"infx.workflows.ci_priority"}
    assert outside_without_config(calls, tmp_path / "workspace") == []
    assert (tmp_path / "git-submodules.log").read_text().splitlines() == trusted_submodule_update(tmp_path / "workspace")


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


def test_profile_dispatches_the_first_point_that_passed_its_preflight(tmp_path):
    gen, filtered, outputs = run_profile_matrix(tmp_path)

    assert gen.returncode == 0, gen.stderr
    assert "srt-slurm recipe preflight passed" in gen.stderr
    assert filtered.returncode == 0, filtered.stderr
    assert ([(row["model"], row["conc"], row["image"]) for row in json.loads(outputs["filtered"])],
            outputs["count"]) == (POINTS[:1], "1")
    calls = uv_calls(tmp_path)
    assert [module for _, module, _ in calls] == [
        "infx.matrix.generate", "infx.srt_slurm.preflight", "infx.workflows.ci_priority",
    ]
    # The measured revision's own generator runs in its checkout by design; the trusted tools do not.
    assert outside_without_config(calls[1:], tmp_path / "workspace") == []
    assert (tmp_path / "git-submodules.log").read_text().splitlines() == trusted_submodule_update(tmp_path / "workspace")


def test_profile_filter_cannot_import_a_measured_json_module(tmp_path):
    marker = tmp_path / "measured-json-imports"
    # Added after gen, whose measured generator loads its own tree by design.
    gen, filtered, outputs = run_profile_matrix(
        tmp_path, filter_files={"inferencex-e2e/json.py": SHADOW_JSON}, MEASURED_JSON_MARKER=str(marker),
    )

    assert gen.returncode == 0, gen.stderr
    assert filtered.returncode == 0, filtered.stderr
    assert not marker.exists(), marker.read_text()
    assert [(row["model"], row["conc"], row["image"]) for row in json.loads(outputs["filtered"])] == POINTS[:1]


@pytest.mark.parametrize("workflow", ["e2e-tests", "profile"])
def test_a_failed_preflight_leaves_no_matrix_to_dispatch(tmp_path, workflow):
    files = {"inferencex-e2e/configs/nvidia-master.yaml": STALE_MASTER}
    if workflow == "e2e-tests":
        result, outputs = run_get_jobs(tmp_path, measured_files=files, GENERATE_COMMAND=GENERATE)
    else:
        result, filtered, outputs = run_profile_matrix(tmp_path, measured_files=files)
        assert filtered is None

    assert result.returncode == 1
    assert "srt-slurm recipe preflight found 1 problem(s)" in result.stderr
    assert outputs == {}
