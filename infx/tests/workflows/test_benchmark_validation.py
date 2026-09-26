"""Execute benchmark validation steps against historical measured checkouts."""

import json
import os
import shutil
import subprocess
import venv
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def validation_python(tmp_path_factory):
    root = tmp_path_factory.mktemp("validation-python")
    venv.EnvBuilder(with_pip=False).create(root)
    return root / "bin" / "python"


def run_step(tmp_path, validation_python, workflow, name, *, hostile=True, **extra_env):
    tooling = tmp_path / ".result-tooling"
    evals = tooling / "infx" / "evals"
    shutil.copytree(
        ROOT / "infx",
        tooling / "infx",
        ignore=shutil.ignore_patterns("tests", "__pycache__", "*.pyc"),
    )
    (tooling / "benchmarks").symlink_to(ROOT / "benchmarks", target_is_directory=True)
    (evals / "thresholds.yaml").write_text(json.dumps({"default": {"gsm8k": 0.90}}))
    if hostile:
        (tmp_path / "infx").mkdir()
        (tmp_path / "infx" / "__init__.py").write_text(
            "raise RuntimeError('measured package must not be imported')\n"
        )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ambient_python = bin_dir / "python3"
    ambient_python.write_text("#!/bin/sh\nexit 73\n")
    ambient_python.chmod(0o755)
    config = yaml.safe_load((ROOT / ".github" / "workflows" / workflow).read_text())
    step = next(
        step
        for job in config["jobs"].values()
        for step in job.get("steps", [])
        if step.get("name") == name
    )
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "INFERENCEX_RESULTS_PYTHON": str(validation_python),
        "MODEL_PREFIX": "fixture",
        **{
            key: value.replace("${{ github.workspace }}", str(tmp_path))
            for key, value in step.get("env", {}).items()
        },
        **extra_env,
    }
    env = {key: value for key, value in env.items() if value is not None}
    return subprocess.run(
        ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", step["run"]],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )


@pytest.mark.parametrize(
    "workflow", ["benchmark-tmpl.yml", "benchmark-multinode-tmpl.yml"]
)
@pytest.mark.parametrize("hostile", [False, True])
@pytest.mark.parametrize("score,returncode,label", [(0.95, 0, "PASS"), (0.89, 1, "FAIL")])
def test_eval_step_uses_canonical_validator(
    tmp_path, validation_python, workflow, hostile, score, returncode, label
):
    result_file = tmp_path / "results_fixture.json"
    result_file.write_text(
        json.dumps({"results": {"gsm8k": {"exact_match,strict-match": score}}})
    )
    (tmp_path / "meta_env.json").write_text(
        json.dumps({"conc": 32, "infmax_model_prefix": "fixture"})
    )
    completed = run_step(
        tmp_path,
        validation_python,
        workflow,
        "Verify eval scores",
        hostile=hostile,
        EVAL_CONC="",
        CONC_LIST="32 8 16",
    )
    assert completed.returncode == returncode, completed.stdout + completed.stderr
    assert f"{label}: gsm8k exact_match,strict-match = {score:.4f}" in (
        completed.stdout + completed.stderr
    )
    assert result_file.is_file()


@pytest.mark.parametrize(
    "eval_conc,measured_conc,returncode", [("16", 16, 0), ("16", 32, 1), ("", 16, 1)]
)
def test_multinode_eval_step_honors_explicit_concurrency(
    tmp_path, validation_python, eval_conc, measured_conc, returncode
):
    (tmp_path / "results_fixture.json").write_text(
        json.dumps({"results": {"gsm8k": {"exact_match,strict-match": 0.95}}})
    )
    (tmp_path / "meta_env.json").write_text(json.dumps({"conc": measured_conc}))
    completed = run_step(
        tmp_path,
        validation_python,
        "benchmark-multinode-tmpl.yml",
        "Verify eval scores",
        EVAL_CONC=eval_conc,
        CONC_LIST="32 8 16",
    )
    assert completed.returncode == returncode, completed.stdout + completed.stderr
    if returncode:
        assert "eval metadata concurrency does not match workflow request" in (
            completed.stdout + completed.stderr
        )
    else:
        assert "PASS: gsm8k exact_match,strict-match = 0.9500" in completed.stdout


@pytest.mark.parametrize("hostile", [False, True])
@pytest.mark.parametrize("errors,returncode", [(10, 0), (11, 1)])
def test_agentic_step_preserves_failure_gate(
    tmp_path, validation_python, hostile, errors, returncode
):
    artifacts = tmp_path / "results" / "aiperf_artifacts"
    artifacts.mkdir(parents=True)
    aggregate = artifacts / "profile_export_aiperf.json"
    data = {
        "request_count": {"avg": 100 - errors},
        "error_request_count": {"avg": errors},
        "completed_request_count": {"avg": 100},
    }
    aggregate.write_text(json.dumps(data))
    completed = run_step(
        tmp_path,
        validation_python,
        "benchmark-tmpl.yml",
        "Validate agentic result",
        hostile=hostile,
        AIPERF_FAILED_REQUEST_THRESHOLD="0.10",
    )
    assert completed.returncode == returncode, completed.stdout + completed.stderr
    assert f"{errors}/100 = {errors:.3f}%" in completed.stdout + completed.stderr
    assert json.loads(aggregate.read_text()) == data


@pytest.mark.parametrize(
    "variable", ["INFERENCEX_RESULTS_PYTHON", "AIPERF_FAILED_REQUEST_THRESHOLD"]
)
@pytest.mark.parametrize("value", [None, ""])
def test_agentic_step_rejects_missing_required_environment(
    tmp_path, validation_python, variable, value
):
    env = {"AIPERF_FAILED_REQUEST_THRESHOLD": "0.10", variable: value}
    completed = run_step(
        tmp_path,
        validation_python,
        "benchmark-tmpl.yml",
        "Validate agentic result",
        **env,
    )
    assert completed.returncode != 0
    assert variable in completed.stdout + completed.stderr
