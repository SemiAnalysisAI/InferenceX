"""Legacy launch lanes (B200 TileRT disagg, MI355X amd_utils AgentX) with fake Slurm binaries."""

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import yaml

from infx.clusters import load_inventory
from infx.launch.__main__ import launch
from infx.launch.drivers import legacy
from infx.launch.request import LaunchRequest
from infx.tests.launch.fake_slurm import sandbox_runner_config

ROOT = Path(__file__).resolve().parents[3]

RECORD = f"""#!{sys.executable}
import json, os, sys
with open(os.environ["FAKE_CALLS"], "a") as log:
    log.write(json.dumps([os.path.basename(sys.argv[0]), *sys.argv[1:]]) + "\\n")
"""
# GNU and BSD tail differ (--pid); print the named files instead of following them.
TAIL = f"""#!{sys.executable}
import os, sys
for arg in sys.argv[1:]:
    if os.path.isfile(arg):
        sys.stdout.write(open(arg, errors="replace").read())
"""

AMD_RECIPE = "benchmarks/multi_node/agentic/dsv4_fp4_mi355x_atom-disagg.sh"
# Submits "job 4242": writes what amd_utils/job.slurm leaves behind and prints the id.
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
    """Runner config with records for the two clusters that have legacy lanes.

    A launch checks every table row keyed by its cluster, and b200-nscale's rows name most
    of its staged checkpoints, so b200-nscale is its sandboxed checked-in record with the
    Slurm facts the TileRT script reads replaced by known values.
    """
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir(exist_ok=True)
    checked_in = yaml.safe_load(sandbox_runner_config(sandbox).read_text())["clusters"]
    b200 = checked_in["b200-nscale"]["slurm"]
    b200.update(partition="tilert-partition", account="tilert-account")
    b200["squash"]["dir"] = str(tmp_path / "squash")
    return {
        "labels": {
            "cluster:b200-nscale": ["b200-nscale-slurm_00"],
            "cluster:mi355x-amds": ["mi355x-amds_00"],
        },
        "clusters": {
            "b200-nscale": checked_in["b200-nscale"],
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


# --------------------------------------------------------------------------
# B200 TileRT disagg: exec of the direct script
# --------------------------------------------------------------------------


def tilert_env(workspace: Path) -> dict[str, str]:
    """glm5.1-fp8-b200-tilert as the multi-node workflow exports it."""
    return {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "RUNNER_NAME": "b200-nscale-slurm_00",
        "GITHUB_WORKSPACE": str(workspace),
        "IS_MULTINODE": "true",
        "IS_AGENTIC": "0",
        "FRAMEWORK": "tilert",
        "MODEL_PREFIX": "glm5.1",
        "PRECISION": "fp8",
        "SPEC_DECODING": "mtp",
        "EXP_NAME": "glm5.1_1k1k",
        "SCENARIO_SUBDIR": "fixed_seq_len/",
        # Master-config additional-settings; MODEL_PATH wins over the cluster default.
        "MODEL_PATH": "/hf/snapshots/glm-5.1",
        "TILERT_WEIGHTS_DIR": "/tilert-cache/glm5.1-fp8-8shard",
    }


def run_cli(tmp_path: Path, env: dict[str, str], cwd: Path) -> subprocess.Popen:
    """Start ``python -m infx.launch run`` against the fixture runner config."""
    config = tmp_path / "runners.yaml"
    config.write_text(yaml.safe_dump(inventory(tmp_path)))
    return subprocess.Popen(
        [sys.executable, "-m", "infx.launch", "--runner-config", str(config), "run"],
        env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def disagg_script(workspace: Path) -> Path:
    """The TileRT disagg script of the glm5.1 point; it records what it was handed and exits 5."""
    script = workspace / "benchmarks/multi_node/glm5.1_fp8_b200_tilert-disagg.sh"
    script.parent.mkdir(parents=True)
    script.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$$" "$SLURM_PARTITION" "$SLURM_ACCOUNT" "$MODEL_PATH" '
        '"$TILERT_WEIGHTS_DIR" "$B200_SQUASH_DIR" > "$GITHUB_WORKSPACE/seen.txt"\n'
        "exit 5\n"
    )
    return script


def test_tilert_lane_replaces_the_launcher_with_the_disagg_script(tmp_path):
    workspace = tmp_path / "workspace"
    disagg_script(workspace)

    launcher = run_cli(tmp_path, tilert_env(workspace), workspace)
    out, err = launcher.communicate(timeout=60)

    # exec: the script runs as the launcher process and its exit code is the job's.
    assert launcher.returncode == 5, out + err
    assert (workspace / "seen.txt").read_text().splitlines() == [
        str(launcher.pid), "tilert-partition", "tilert-account", "/hf/snapshots/glm-5.1",
        "/tilert-cache/glm5.1-fp8-8shard", str(tmp_path / "squash"),
    ]


def test_tilert_lane_requires_its_weights_dir(tmp_path):
    workspace = tmp_path / "workspace"
    disagg_script(workspace)

    # The master config names the weights cache; there is no default to fall back to.
    launcher = run_cli(tmp_path, {**tilert_env(workspace), "TILERT_WEIGHTS_DIR": ""}, workspace)
    out, err = launcher.communicate(timeout=60)

    assert launcher.returncode == 1, out + err
    assert "TILERT_WEIGHTS_DIR" in err
    assert not (workspace / "seen.txt").exists()


def test_tilert_lane_fails_when_the_disagg_script_is_missing(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    launcher = run_cli(tmp_path, tilert_env(workspace), workspace)
    out, err = launcher.communicate(timeout=60)

    assert launcher.returncode == 1, out + err
    assert "tilert disagg script not found" in out


# --------------------------------------------------------------------------
# MI355X amd_utils AgentX
# --------------------------------------------------------------------------


@pytest.fixture
def amd_workspace(tmp_path):
    """Measured checkout with a fake amd_utils AgentX recipe."""
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
        # The fabric the cluster's srt-slurm host setup is given.
        "IBDEVICES": "rdma0,rdma1",
    }
    # Uploaded by benchmark-multinode-tmpl.yml: eval results, LOGS/agentic, server logs.
    assert json.loads((amd_workspace / "results_gsm8k.json").read_text()) == {"score": 1}
    assert (amd_workspace / "LOGS/agentic/conc_1/aiperf_artifacts/profile.json").exists()
    with tarfile.open(amd_workspace / "multinode_server_logs.tar.gz") as bundle:
        assert "./server.log" in bundle.getnames()
    # The exit cleanup keeps the Slurm output, shows its stderr, then drops the log tree.
    artifacts = amd_workspace / "benchmark_artifacts"
    assert (artifacts / "slurm_job-4242.out").read_text() == "benchmark done\n"
    assert (artifacts / "slurm_job-4242.err").read_text() == "worker warning\n"
    assert not (amd_workspace / "benchmark_logs").exists()
    assert "worker warning" in capfd.readouterr().out
    # The job had left the queue when its log ended, so nothing is left to cancel.
    assert not any(call[0] == "scancel" for call in fakes())


def test_amd_utils_lane_without_a_job_id_fails_and_still_cleans_up(
    fakes, tmp_path, amd_workspace, monkeypatch
):
    assert amd_launch(tmp_path, amd_workspace, monkeypatch, NO_JOB_ID="1") == 1

    assert not (amd_workspace / "benchmark_logs").exists()
    assert not any(call[0] == "scancel" for call in fakes())


def test_amd_utils_lane_fails_when_the_job_ends_before_its_log(
    fakes, tmp_path, amd_workspace, monkeypatch
):
    assert amd_launch(tmp_path, amd_workspace, monkeypatch, NO_JOB_LOG="1") == 1

    assert ["scontrol", "show", "job", "4242"] in fakes()
    assert not any(call[0] == "scancel" for call in fakes())
    assert not (amd_workspace / "benchmark_logs").exists()


def test_an_inherited_log_dir_cannot_point_the_cleanup_at_the_checkout(
    fakes, tmp_path, amd_workspace, monkeypatch
):
    keep = amd_workspace / "results.json"
    keep.write_text("{}")

    # The B200 TileRT runtime settings export the checkout itself; the exit cleanup
    # removes the log directory wholesale, so the lane uses its own.
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


def test_amd_utils_lane_rejects_fixed_sequence_points(fakes, tmp_path, amd_workspace, monkeypatch):
    assert amd_launch(tmp_path, amd_workspace, monkeypatch, IS_AGENTIC="0") == 1

    assert not (amd_workspace / "submitted.env").exists()
