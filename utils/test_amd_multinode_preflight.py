from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LIBRARY = REPO_ROOT / "benchmarks/benchmark_lib.sh"
PREFLIGHT = REPO_ROOT / "benchmarks/multi_node/amd_utils/preflight_node.sh"


@pytest.fixture
def cluster(tmp_path: Path):
    """Only Slurm, Docker, GPU telemetry, host naming, and the sleep clock are fake."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    scripts = {
        "srun": """#!/usr/bin/env python3
import os, subprocess, sys
args = sys.argv[1:]
while args and args[0].startswith("--"):
    args.pop(0)
children = [subprocess.Popen(args, env={**os.environ, "SLURM_PROCID": str(rank)})
            for rank in range(2)]
statuses = [child.wait() for child in children]
sys.exit(next((status for status in statuses if status), 0))
""",
        "docker": """#!/usr/bin/env python3
import os, pathlib, sys
with (pathlib.Path(os.environ["TEST_STATE"]) / ("docker-" + os.environ["SLURM_PROCID"])).open("a") as f:
    f.write(" ".join(sys.argv[1:]) + "\\n")
""",
        "hostname": '#!/bin/sh\necho "node-$SLURM_PROCID"\n',
        "sleep": "#!/bin/sh\nexit 0\n",
        "rocm-smi": """#!/usr/bin/env python3
import os, pathlib, time
state = pathlib.Path(os.environ["TEST_STATE"])
rank = os.environ["SLURM_PROCID"]
mode = os.environ["TEST_MODE"]
with (state / ("probes-" + rank)).open("a") as f:
    f.write("probe\\n")
if rank == "1" and mode == "delayed" and not (state / "release").exists():
    (state / "waiting").touch()
    time.sleep(0.02)
    used = 92
elif rank == "1" and mode == "failure":
    used = 92
else:
    used = 0
    (state / ("clean-" + rank)).touch()
print(f"GPU[0] : GPU Memory Allocated (VRAM%): {used}")
""",
    }
    for name, source in scripts.items():
        path = bin_dir / name
        path.write_text(source)
        path.chmod(0o755)
    job_id = f"infx-preflight-test-{uuid.uuid4().hex}"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "DOCKER_CMD_DETECT": "DOCKER_CMD=docker",
        "DI_REPO_DIR": str(REPO_ROOT),
        "SLURM_JOB_ID": job_id,
        "TEST_STATE": str(tmp_path),
        "TEST_MODE": "delayed",
    }
    yield tmp_path, env, Path("/tmp") / f"slurm_job-{job_id}"
    shutil.rmtree(Path("/tmp") / f"slurm_job-{job_id}", ignore_errors=True)


def _command(skip: str = "0", server_status: int = 0) -> list[str]:
    # The actual orchestration and node preflight run unchanged. The server is
    # an external stand-in that records both observed clean markers at launch.
    server = """import json, os, pathlib
p = pathlib.Path(os.environ["TEST_STATE"])
(p / ("server-" + os.environ["SLURM_PROCID"])).write_text(json.dumps([
    (p / "clean-0").exists(), (p / "clean-1").exists()]))
raise SystemExit(int(os.environ["TEST_SERVER_STATUS"]))
"""
    return [
        "bash",
        "-c",
        'source "$1" --validation-only; shift; run_amd_multinode_after_preflight "$@"',
        "test",
        str(LIBRARY),
        "node-0,node-1",
        "2",
        str(PREFLIGHT),
        "name=^container_test_",
        skip,
        "env",
        f"TEST_SERVER_STATUS={server_status}",
        "python3",
        "-c",
        server,
    ]


def test_slow_peer_finishes_gpu_preflight_before_any_server_starts(cluster) -> None:
    state, env, logs = cluster
    with (state / "output").open("w+") as output:
        proc = subprocess.Popen(_command(), env=env, stdout=output, stderr=output)
        try:
            deadline = time.monotonic() + 10
            while not ((state / "waiting").exists() and (state / "clean-0").exists()):
                assert proc.poll() is None, (state / "output").read_text()
                assert time.monotonic() < deadline
                time.sleep(0.02)
            assert not list(state.glob("server-*"))
            (state / "release").touch()
            assert proc.wait(timeout=10) == 0
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    for rank in range(2):
        assert json.loads((state / f"server-{rank}").read_text()) == [True, True]
        assert "GPUs clean" in (logs / f"preflight_node-{rank}.log").read_text()


def test_failed_gpu_guard_prevents_all_servers_and_retains_diagnostics(cluster) -> None:
    state, env, logs = cluster
    completed = subprocess.run(
        _command(),
        env={**env, "TEST_MODE": "failure"},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 1
    assert not list(state.glob("server-*"))
    assert len((state / "probes-1").read_text().splitlines()) == 90
    assert "GPUs still draining" in (logs / "preflight_node-1.log").read_text()
    assert "no server containers launched" in completed.stderr


def test_explicit_gpu_skip_still_precleans_and_preserves_server_failure(
    cluster,
) -> None:
    state, env, logs = cluster
    completed = subprocess.run(
        _command(skip="1", server_status=23),
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 23
    assert not list(state.glob("probes-*"))
    assert len(list(state.glob("server-*"))) == 2
    for rank in range(2):
        assert (state / f"docker-{rank}").read_text().splitlines() == [
            "ps -aq --filter name=^container_test_",
            "ps -aq",
            "ps -aq",
        ]
        assert (
            "skipping GPU pre-flight"
            in (logs / f"preflight_node-{rank}.log").read_text()
        )
