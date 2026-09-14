"""Keep a failed container probe from becoming a successful diagnostic job."""
import os
from pathlib import Path
import subprocess

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("probe_rc", [0, 23])
def test_batch_preserves_container_exit_status(tmp_path, probe_rc):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    srun = bindir / "srun"
    srun.write_text(f"#!/bin/sh\nexit {probe_rc}\n")
    srun.chmod(0o755)
    env = {**os.environ, "PATH": str(bindir) + os.pathsep + os.environ["PATH"],
           "GITHUB_WORKSPACE": str(tmp_path), "SLURM_JOB_ID": "123",
           "SQUASH_FILE": str(tmp_path / "runtime.sqsh")}
    result = subprocess.run(["bash", str(ROOT / "runners/b300_rdma_probe.sbatch")],
                            env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == probe_rc, result.stdout + result.stderr


@pytest.mark.parametrize("status,expected_rc", [("COMPLETED|0:0", 0), ("FAILED|23:0", 1)])
def test_workflow_rejects_failed_batch(tmp_path, status, expected_rc):
    workflow = yaml.safe_load((ROOT / ".github/workflows/speedbench-al.yml").read_text())
    command = next(step["run"] for step in workflow["jobs"]["collect-al"]["steps"]
                   if step.get("name") == "Probe the RDMA provider ABI in the cached runtime")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name, script in {"sbatch": "echo 123", "squeue": "exit 0", "sinfo": "exit 0",
                         "sacct": f"echo '{status}'"}.items():
        path = bindir / name
        path.write_text("#!/bin/sh\n" + script + "\n")
        path.chmod(0o755)
    (tmp_path / "diag").mkdir()
    (tmp_path / "diag/slurm-123.out").write_text("retained probe output\n")
    image = tmp_path / "runtime.sqsh"
    image.touch()
    env = {**os.environ, "PATH": str(bindir) + os.pathsep + os.environ["PATH"],
           "RUNNER_TEMP": str(tmp_path), "GITHUB_WORKSPACE": str(tmp_path),
           "TASK_JOB_NAME": "pr3088-test", "SQUASH_FILE": str(image)}
    result = subprocess.run(["bash", "-c", command.replace("date -u --iso-8601=seconds", "date -u")],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == expected_rc, result.stdout + result.stderr
    assert (tmp_path / "pr3088-rdma/slurm-123.out").read_text() == "retained probe output\n"
