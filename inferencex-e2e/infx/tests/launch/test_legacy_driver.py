"""MI355X amd_utils AgentX launch lanes with fake Slurm binaries."""

import json
import os
import sys
import tarfile
from pathlib import Path

import pytest

from infx.clusters import load_inventory
from infx.launch.__main__ import launch
from infx.launch.drivers import legacy
from infx.launch.request import LaunchRequest

RECORD = f"""#!{sys.executable}
import json, os, sys
with open(os.environ["FAKE_CALLS"], "a") as log:
    log.write(json.dumps([os.path.basename(sys.argv[0]), *sys.argv[1:]]) + "\\n")
"""
TAIL = f"""#!{sys.executable}
import os, sys
for arg in sys.argv[1:]:
    if os.path.isfile(arg):
        sys.stdout.write(open(arg, errors="replace").read())
"""

AMD_RECIPE = "benchmarks/multi_node/agentic/dsv4_fp4_mi355x_atom-disagg.sh"
AMD_SUBMIT = """#!/usr/bin/env bash
set -e
env > "$GITHUB_WORKSPACE/submitted.env"
job="$BENCHMARK_LOGS_DIR/logs/slurm_job-4242"
mkdir -p "$job/agentic/conc_1/aiperf_artifacts" "$job/eval_results"
echo trace > "$job/agentic/conc_1/aiperf_artifacts/profile.json"
echo '{"score": 1}' > "$job/eval_results/results_gsm8k.json"
echo 'server log' > "$job/server.log"
[[ "${NO_JOB_LOG:-}" == 1 ]] || echo 'benchmark done' > "$BENCHMARK_LOGS_DIR/slurm_job-4242.out"
echo 'worker warning' > "$BENCHMARK_LOGS_DIR/slurm_job-4242.err"
echo 'submitting' >&2
[[ "${NO_JOB_ID:-}" == 1 ]] && exit 1
echo 4242
"""


def inventory(tmp_path: Path) -> dict:
    """Runner config for the remaining amd_utils launch lane."""
    return {
        "labels": {
            "cluster:mi355x-amds": ["mi355x-amds_00"],
        },
        "clusters": {
            "mi355x-amds": {
                "gpus-per-node": 8, "arch": "x86_64", "scheduler": "slurm",
                "slurm": {
                    "partition": "compute", "exclusive": True,
                    "volumes": {
                        "it-share-data": {"path": str(tmp_path / "it-share")},
                        "aiperf-cache": {"path": str(tmp_path / "aiperf-cache")},
                        "shared-hf-hub-cache": {"path": str(tmp_path / "shared-hf-hub-cache")},
                    },
                    "srt-slurm": {"network-interface": "", "host-setup": {
                        "script": "runners/srt-slurm/hooks/mi355x-amds/setup.sh", "env": {"IBDEVICES": "rdma0,rdma1"},
                    }},
                },
            },
        },
    }  # fmt: skip


@pytest.fixture
def fakes(tmp_path, monkeypatch):
    """Fake sudo/scancel/squeue/scontrol/tail on PATH; returns a reader of Slurm calls."""
    binaries = tmp_path / "bin"
    binaries.mkdir()
    scripts = {"sudo": '#!/bin/bash\nexec "$@"\n', "tail": TAIL}
    scripts.update(dict.fromkeys(("scancel", "squeue", "scontrol"), RECORD))
    for name, body in scripts.items():
        (binaries / name).write_text(body)
        (binaries / name).chmod(0o755)
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("PATH", f"{binaries}:/usr/bin:/bin")
    monkeypatch.setenv("FAKE_CALLS", str(log))

    def calls() -> list[list[str]]:
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    return calls


@pytest.fixture
def amd_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    recipe = workspace / AMD_RECIPE
    recipe.parent.mkdir(parents=True)
    recipe.write_text(AMD_SUBMIT)
    return workspace


def amd_launch(tmp_path: Path, workspace: Path, monkeypatch, **overrides: str) -> int:
    """Dispatch a dsv4-fp4-mi355x-atom-disagg AgentX point through the real driver selection."""
    monkeypatch.setattr(legacy, "SUBMIT_SETTLE_S", 0)
    cluster = load_inventory(inventory(tmp_path)).clusters["mi355x-amds"]
    env = {
        **os.environ,
        "RUNNER_NAME": "mi355x-amds_00",
        "GITHUB_WORKSPACE": str(workspace),
        "GITHUB_ACTIONS": "true",
        "USER": "runner",
        "IS_MULTINODE": "true",
        "IS_AGENTIC": "1",
        "RUN_EVAL": "true",
        "KEEP_LOGS": "0",
        "MODEL": "deepseek-ai/DeepSeek-V4-Pro-0813",
        "MODEL_PREFIX": "dsv4",
        "EXP_NAME": "dsv4_agentic_p1x8",
        "PRECISION": "fp4",
        "FRAMEWORK": "atom-disagg",
        **overrides,
    }
    return launch(cluster, LaunchRequest.from_env(env))


def test_amd_utils_lane_follows_the_job_and_stages_its_artifacts(
    fakes, tmp_path, amd_workspace, monkeypatch, capfd
):
    assert amd_launch(tmp_path, amd_workspace, monkeypatch) == 0

    submitted = dict(
        line.split("=", 1) for line in (amd_workspace / "submitted.env").read_text().splitlines()
        if "=" in line
    )
    assert {key: submitted[key] for key in (
        "SLURM_ACCOUNT", "SLURM_PARTITION", "MODEL_NAME", "MODEL_PATH", "MODEL_DIR",
        "GPUS_PER_NODE", "BENCHMARK_LOGS_DIR", "IBDEVICES",
    )} == {
        "SLURM_ACCOUNT": "runner", "SLURM_PARTITION": "compute",
        "MODEL_NAME": "DeepSeek-V4-Pro-0813", "MODEL_PATH": f"{tmp_path}/it-share",
        "MODEL_DIR": f"{tmp_path}/it-share", "GPUS_PER_NODE": "8",
        "BENCHMARK_LOGS_DIR": f"{amd_workspace}/benchmark_logs",
        "IBDEVICES": "rdma0,rdma1",
    }
    assert json.loads((amd_workspace / "results_gsm8k.json").read_text()) == {"score": 1}
    assert (amd_workspace / "LOGS/agentic/conc_1/aiperf_artifacts/profile.json").exists()
    with tarfile.open(amd_workspace / "multinode_server_logs.tar.gz") as bundle:
        assert "./server.log" in bundle.getnames()
    artifacts = amd_workspace / "benchmark_artifacts"
    assert (artifacts / "slurm_job-4242.out").read_text() == "benchmark done\n"
    assert (artifacts / "slurm_job-4242.err").read_text() == "worker warning\n"
    assert not (amd_workspace / "benchmark_logs").exists()
    assert "worker warning" in capfd.readouterr().out
    assert not any(call[0] == "scancel" for call in fakes())


@pytest.mark.parametrize(("overrides", "submitted"), [
    ({"NO_JOB_ID": "1"}, True),
    ({"NO_JOB_LOG": "1"}, True),
    ({"IS_AGENTIC": "0"}, False),
], ids=["no-job-id", "no-job-log", "fixed-sequence"])  # fmt: skip
def test_a_failed_amd_utils_launch_still_removes_the_log_tree(
    fakes, tmp_path, amd_workspace, monkeypatch, overrides, submitted
):
    assert amd_launch(tmp_path, amd_workspace, monkeypatch, **overrides) == 1

    assert (amd_workspace / "submitted.env").exists() == submitted
    assert not (amd_workspace / "benchmark_logs").exists()
    assert not any(call[0] == "scancel" for call in fakes())


def test_an_inherited_log_dir_cannot_point_the_cleanup_at_the_checkout(
    fakes, tmp_path, amd_workspace, monkeypatch
):
    keep = amd_workspace / "results.json"
    keep.write_text("{}")

    assert amd_launch(tmp_path, amd_workspace, monkeypatch, BENCHMARK_LOGS_DIR=str(amd_workspace)) == 0

    assert keep.read_text() == "{}"
    submitted = (amd_workspace / "submitted.env").read_text().splitlines()
    assert f"BENCHMARK_LOGS_DIR={amd_workspace}/benchmark_logs" in submitted
    assert not (amd_workspace / "benchmark_logs").exists()


def test_amd_utils_lane_keeps_the_log_tree_with_keep_logs(
    fakes, tmp_path, amd_workspace, monkeypatch
):
    assert amd_launch(tmp_path, amd_workspace, monkeypatch, KEEP_LOGS="1") == 0

    logs = amd_workspace / "benchmark_logs"
    assert (logs / "slurm_job-4242.out").exists()
    assert not (amd_workspace / "benchmark_artifacts").exists()
