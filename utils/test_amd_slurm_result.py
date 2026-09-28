"""A completed log tail must not conceal a failed Slurm allocation."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("records", "expected"),
    [
        ("321|COMPLETED|0:0", 0),
        ("321|FAILED|1:0", 1),
        ("321|CANCELLED by 123|0:15", 1),
        ("321|TIMEOUT|0:0", 1),
        ("321|COMPLETED|0:9", 1),
        ("", 1),
        ("1321|COMPLETED|0:0\n321.batch|COMPLETED|0:0", 1),
        ("1321|FAILED|1:0\n321|COMPLETED|0:0", 0),
    ],
)
def test_exact_allocation_exit_status(records: str, expected: int) -> None:
    source = (ROOT / "runners/launch_mi355x-amds.sh").read_text()
    code = source.split("    SLURM_BENCHMARK_RC=1\n", 1)[1].split("\n    set -x", 1)[0]
    script = (
        """JOB_ID=321
sacct() { printf '%s\n' "$TEST_RECORDS"; }
sleep() { :; }
SLURM_BENCHMARK_RC=1
"""
        + code
        + '\nexit "$SLURM_BENCHMARK_RC"\n'
    )
    result = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "TEST_RECORDS": records},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == expected, result.stderr
    assert 'exit "$SLURM_BENCHMARK_RC"' in source


def test_job_uses_owned_wrapper_and_preserves_srun_status() -> None:
    source = (ROOT / "benchmarks/multi_node/amd_utils/job.slurm").read_text()
    assert (
        r"exec bash \"$DI_REPO_DIR/benchmarks/multi_node/amd_utils/run_owned_container.sh\""
        in source
    )
    assert "SERVING_RC=$?" in source
    assert 'exit "$SERVING_RC"' in source
    assert "docker ps -aq" not in source
