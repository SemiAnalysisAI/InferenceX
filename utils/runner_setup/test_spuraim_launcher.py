"""Integration-style tests for the SPUR GitHub Actions launcher."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "runners" / "launch_spuraim.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def _launcher_env(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    bin_dir = tmp_path / "bin"
    capture_dir = tmp_path / "capture"
    workspace = tmp_path / "workspace"
    bin_dir.mkdir()
    capture_dir.mkdir()
    workspace.mkdir()

    _write_executable(
        bin_dir / "squeue",
        """#!/usr/bin/env bash
if [[ "$*" == *"%.200j"* ]]; then
    [[ "${TEST_STALE_JOB:-0}" == 1 ]] && echo '456 ix-spuraim_00-stale-job'
    exit 0
fi
echo '123|RUNNING|00:01|test-node'
""",
    )
    _write_executable(
        bin_dir / "scancel",
        """#!/usr/bin/env bash
printf '%s\n' "$*" >> "$TEST_CAPTURE_DIR/scancel.log"
""",
    )
    _write_executable(
        bin_dir / "srun",
        """#!/usr/bin/env bash
printf '%s\n' "$*" >> "$TEST_CAPTURE_DIR/srun.log"
inner="${!#}"
bash "$inner"
""",
    )
    _write_executable(
        bin_dir / "docker",
        """#!/usr/bin/env bash
set -u
command_name="${1:-}"
shift || true
printf '%s %s\n' "$command_name" "$*" >> "$TEST_CAPTURE_DIR/docker.log"
case "$command_name" in
    info|rm|pull)
        exit 0
        ;;
    image)
        exit 0
        ;;
    inspect)
        echo running
        exit 0
        ;;
    run)
        env_file=''
        is_chown=0
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --env-file)
                    env_file="$2"
                    shift 2
                    ;;
                --entrypoint)
                    [[ "$2" == chown ]] && is_chown=1
                    shift 2
                    ;;
                *)
                    shift
                    ;;
            esac
        done
        [[ "$is_chown" == 1 ]] && exit 0
        cp "$env_file" "$TEST_CAPTURE_DIR/env.list"
        stat -c '%a' "$env_file" > "$TEST_CAPTURE_DIR/env.mode"
        sleep "${TEST_DOCKER_RUN_SECONDS:-2}"
        exit "${TEST_DOCKER_RC:-0}"
        ;;
    *)
        echo "unexpected docker command: $command_name" >&2
        exit 2
        ;;
esac
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "TEST_CAPTURE_DIR": str(capture_dir),
            "TEST_STALE_JOB": "1",
            "SPUR_CONTROLLER_ADDR": "test-controller",
            "SPUR_SHARED_HF_ROOT": str(tmp_path / "shared-hf"),
            "SPUR_NODE_SCRATCH": str(tmp_path / "scratch"),
            "SPUR_HEARTBEAT_SECONDS": "1",
            "GITHUB_WORKSPACE": str(workspace),
            "RUNNER_NAME": "spuraim_00",
            "RUNNER_TYPE": "cluster:spuraim",
            "IMAGE": "test/image:latest",
            "MODEL": "test/model",
            "MODEL_PREFIX": "kimik3",
            "EXP_NAME": "kimik3_tp8_conc8_kvdram-lmcache-k3_spec-mtp",
            "PRECISION": "fp4",
            "FRAMEWORK": "vllm",
            "TP": "8",
            "GPU_COUNT": "8",
            "CONC": "8",
            "SCENARIO_SUBDIR": "agentic/",
            "SPEC_DECODING": "mtp",
            "KV_OFFLOADING": "dram",
            "KV_OFFLOAD_BACKEND": "lmcache-k3",
            "KV_OFFLOAD_BACKEND_METADATA": (
                '{\n  "name": "lmcache-k3",\n  "version": "0.5.1"\n}'
            ),
            "ROUTER_METADATA": '{\n  "name": "test-router",\n  "version": "1"\n}',
            "HF_TOKEN": "test-hf-secret",
            "MODAL_TOKEN_ID": "test-modal-id",
            "MODAL_TOKEN_SECRET": "test-modal-secret",
        }
    )
    return env, capture_dir, workspace


def test_launcher_compacts_metadata_and_emits_heartbeats(tmp_path: Path) -> None:
    env, capture_dir, workspace = _launcher_env(tmp_path)

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    combined_output = result.stdout + result.stderr
    assert result.returncode == 0, combined_output
    assert "[spuraim/heartbeat]" in combined_output
    assert "test-hf-secret" not in combined_output
    assert "test-modal-id" not in combined_output
    assert "test-modal-secret" not in combined_output

    env_lines = (capture_dir / "env.list").read_text().splitlines()
    assert (
        'KV_OFFLOAD_BACKEND_METADATA={"name":"lmcache-k3","version":"0.5.1"}'
        in env_lines
    )
    assert 'ROUTER_METADATA={"name":"test-router","version":"1"}' in env_lines
    assert (capture_dir / "env.mode").read_text().strip() == "600"
    assert len((capture_dir / "srun.log").read_text().splitlines()) == 1
    assert "456" in (capture_dir / "scancel.log").read_text().splitlines()

    stage_dir = workspace / ".spuraim"
    assert stage_dir.stat().st_mode & 0o777 == 0o700
    assert list(stage_dir.iterdir()) == []


def test_launcher_rejects_invalid_metadata_before_srun(tmp_path: Path) -> None:
    env, capture_dir, workspace = _launcher_env(tmp_path)
    env["KV_OFFLOAD_BACKEND_METADATA"] = "{not-json"

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 65
    assert "KV_OFFLOAD_BACKEND_METADATA must contain valid JSON" in result.stderr
    assert not (capture_dir / "srun.log").exists()
    assert list((workspace / ".spuraim").iterdir()) == []


def test_launcher_cancels_allocation_and_cleans_files_on_signal(
    tmp_path: Path,
) -> None:
    env, capture_dir, workspace = _launcher_env(tmp_path)
    env["TEST_DOCKER_RUN_SECONDS"] = "30"

    process = subprocess.Popen(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not (capture_dir / "srun.log").exists():
            if time.monotonic() >= deadline:
                raise AssertionError("launcher did not invoke srun")
            time.sleep(0.05)
        time.sleep(0.25)
        os.killpg(process.pid, signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=10)
    except BaseException:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
        raise

    assert process.returncode == 143, stdout + stderr
    assert "received TERM" in stderr
    assert "-n ix-spuraim_00-" in (capture_dir / "scancel.log").read_text()
    assert "--entrypoint chown" in (capture_dir / "docker.log").read_text()
    assert list((workspace / ".spuraim").iterdir()) == []
