"""GB200 llm-d vLLM launches through infx.launch instead of runners/launch_gb200-nv.sh."""

import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from infx.tests.launch.fake_slurm import (
    base_env,
    install_fakes,
    launch,
    make_workspace,
    runner_for,
    sandbox_runner_config,
)

LLMD_SUBMIT = """#!/usr/bin/env bash
set -e
env > "$GITHUB_WORKSPACE/submitted.env"
logs="$BENCHMARK_LOGS_DIR"
job="$logs/slurm_job-4299"
mkdir -p "$logs/agentic/conc_128" "$job/eval_results"
echo '{"conc": 128}' > "$logs/point-identity_conc128.json"
echo trace > "$logs/agentic/conc_128/profile.json"
echo '{"score": 1}' > "$job/eval_results/results_gsm8k.json"
echo 'server log' > "$logs/server.log"
echo 'benchmark done' > "$logs/slurm_job-4299.out"
echo 'worker warning' > "$logs/slurm_job-4299.err"
echo 'submitting' >&2
[[ "${NO_JOB_ID:-}" == 1 ]] && exit 1
echo 4299
"""


@pytest.fixture
def harness(tmp_path):
    """Sandboxed gb200-nv cluster, fake Slurm binaries, and a workspace with the llm-d wrapper."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    config = sandbox_runner_config(sandbox)
    workspace = make_workspace(tmp_path / "workspace")
    wrapper = workspace / "benchmarks/multi_node/dsv4_fp4_gb200_llmd-vllm-disagg.sh"
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(LLMD_SUBMIT)
    wrapper.chmod(0o755)

    model_root = sandbox / "mnt/lustre01/users-public/sa-shared/models/DeepSeek-V4-Pro-0813"
    model_root.mkdir(parents=True)
    (model_root / "config.json").write_text("{}\n")

    logs = tmp_path / "logs"
    env = base_env(
        fakes=install_fakes(tmp_path / "bin"), logs=logs, workspace=workspace, sandbox=sandbox
    )
    env.update(
        RUNNER_NAME=runner_for("gb200-nv"),
        IS_MULTINODE="true",
        IS_AGENTIC="1",
        RUN_EVAL="true",
        EVAL_ONLY="false",
        FRAMEWORK="llmd-vllm",
        MODEL="deepseek-ai/DeepSeek-V4-Pro-0813",
        MODEL_PREFIX="dsv4",
        PRECISION="fp4",
        SPEC_DECODING="mtp",
        THINKING_MODE="thinking_on",
        EXP_NAME="dsv4_agentic_p1x8",
        DISAGG="true",
        IMAGE="vllm/vllm-openai:v0.21.0",
        RESULT_FILENAME="point-identity",
        ENROOT_IMPORT_TIME_LIMIT="10",
    )
    return config, workspace, env


def run_launch(harness) -> subprocess.CompletedProcess[str]:
    config, workspace, env = harness
    return launch(env, config, workspace)


def test_llmd_driver_submits_the_wrapper_and_stages_artifacts(harness):
    result = run_launch(harness)
    config, workspace, env = harness

    assert result.returncode == 0, result.stdout + result.stderr
    submitted = dict(
        line.split("=", 1) for line in (workspace / "submitted.env").read_text().splitlines() if "=" in line
    )
    model_path = f"{config.parent}/mnt/lustre01/users-public/sa-shared/models/DeepSeek-V4-Pro-0813"
    assert submitted["MODEL_PATH"] == model_path
    assert submitted["MODEL_NAME"] == env["MODEL"]
    assert submitted["LLMD_CONTAINER_ENGINE"] == "pyxis"
    assert submitted["LLMD_SQUASH_FILE"]
    assert submitted["BENCHMARK_LOGS_DIR"] == f"{workspace}/benchmark_logs"
    assert submitted["SLURM_PARTITION"] == "batch"
    assert submitted["SLURM_ACCOUNT"] == "benchmark"

    assert json.loads((workspace / "point-identity_conc128.json").read_text()) == {"conc": 128}
    assert (workspace / "LOGS/agentic/conc_128/profile.json").read_text() == "trace\n"
    assert json.loads((workspace / "results_gsm8k.json").read_text()) == {"score": 1}
    with tarfile.open(workspace / "multinode_server_logs.tar.gz") as bundle:
        assert "./server.log" in bundle.getnames()
    assert "submitting" in result.stderr


def test_llmd_driver_fails_when_the_wrapper_prints_no_job_id(harness):
    _, workspace, env = harness
    env["NO_JOB_ID"] = "1"

    result = launch(env, harness[0], workspace)

    assert result.returncode == 1
    assert "failed before returning a Slurm job id" in result.stderr
