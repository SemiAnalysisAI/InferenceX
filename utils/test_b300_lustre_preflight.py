"""Exercise B300 storage checks before import or container I/O."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "runners/b300_lustre_preflight.sh"


def executable(path: Path, body: str) -> None:
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(0o755)


@pytest.fixture
def env(tmp_path: Path) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # macOS has no GNU timeout. Only lctl is mocked here; the live Slurm check
    # exercises the real timeout command on Linux.
    executable(bin_dir / "timeout", 'shift 2\nexec "$@"\n')
    executable(
        bin_dir / "lctl",
        'printf "%s\\n" "$IMPORTS"\nexit "${LCTL_STATUS:-0}"\n',
    )
    return {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "LUSTRE_PREFLIGHT_TIMEOUT": "0",
        "IMPORTS": "    name: test-OST0000\n    state: FULL\n"
        "    name: test-MDT0000\n    state: FULL\n",
    }


@pytest.mark.parametrize("state", ["FULL", "EVICTED", "CONNECTING", "IDLE"])
def test_all_clients_must_be_connected(env: dict[str, str], state: str) -> None:
    env["IMPORTS"] += f"    name: test-OST0001\n    state: {state}\n"
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; check_b300_lustre', "bash", str(PREFLIGHT)],
        env=env, capture_output=True, text=True, timeout=5,
    )
    assert (result.returncode == 0) == (state == "FULL")
    if state != "FULL":
        assert "test-OST0001" in result.stderr
        assert "refusing shared-filesystem I/O" in result.stderr


@pytest.mark.parametrize("output,status", [("", "0"), ("permission denied", "1")])
def test_missing_client_evidence_fails_closed(
    env: dict[str, str], output: str, status: str,
) -> None:
    env.update(IMPORTS=output, LCTL_STATUS=status)
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; check_b300_lustre', "bash", str(PREFLIGHT)],
        env=env, capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 1
    assert "Error:" in result.stderr


def test_client_can_recover_during_grace_period(
    tmp_path: Path, env: dict[str, str],
) -> None:
    marker = tmp_path / "checked"
    env.update(LUSTRE_PREFLIGHT_TIMEOUT="30", CHECK_MARKER=str(marker))
    executable(
        tmp_path / "bin/lctl",
        'if [[ -e "$CHECK_MARKER" ]]; then\n'
        '  printf "    state: FULL\\n"\n'
        'else\n'
        '  touch "$CHECK_MARKER"\n'
        '  printf "    state: CONNECTING\\n"\n'
        'fi\n',
    )
    executable(tmp_path / "bin/sleep", "exit 0\n")
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; check_b300_lustre', "bash", str(PREFLIGHT)],
        env=env, capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 0
    assert "Waiting for Lustre clients" in result.stderr


@pytest.mark.parametrize("state", ["FULL", "EVICTED"])
def test_import_checks_storage_before_opening_lock(
    tmp_path: Path, env: dict[str, str], state: str,
) -> None:
    bin_dir = tmp_path / "bin"
    env.update(
        IMPORTS=f"    name: test-OST0000\n    state: {state}\n",
        SRUN_LOG=str(tmp_path / "srun.log"),
    )
    executable(
        bin_dir / "srun",
        'printf "%s\\n" "$@" > "$SRUN_LOG"\n'
        'while [[ "$1" != bash ]]; do shift; done\nexec "$@"\n',
    )
    executable(bin_dir / "unsquashfs", '[[ -s "$2" ]]\n')
    executable(bin_dir / "flock", 'exit 0\n')
    executable(bin_dir / "enroot", 'printf valid-image > "$3"\n')
    launcher = (ROOT / "runners/launch_b300-dsxe.sh").read_text()
    start = launcher.index("import_squash_image() {")
    end = launcher.index('\nif [[ "$IS_MULTINODE"', start)
    script = (
        'source "$1"\n'
        'LUSTRE_PREFLIGHT="$(declare -f check_b300_lustre); check_b300_lustre"\n'
        'SLURM_ACCOUNT=benchmark\nSLURM_PARTITION=batch_1\n'
        'RUNNER_NAME=b300-test_01\nSALLOC_EXCLUDE=bad-node\n'
        + launcher[start:end]
        + '\nimport_squash_image example/test:tag "$2"\n'
    )
    squash = tmp_path / "test.sqsh"
    result = subprocess.run(
        ["bash", "-c", script, "bash", str(PREFLIGHT), str(squash)],
        env=env, capture_output=True, text=True, timeout=5,
    )
    args = (tmp_path / "srun.log").read_text().splitlines()
    assert "--job-name=b300-test_01" in args
    assert "--chdir=/tmp" in args
    assert "--exclude=bad-node" in args
    assert (result.returncode == 0) == (state == "FULL"), result.stderr
    assert squash.exists() == (state == "FULL")
    assert squash.with_suffix(".sqsh.lock").exists() == (state == "FULL")
