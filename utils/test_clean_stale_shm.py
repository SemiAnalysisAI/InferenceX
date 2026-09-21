"""Exercise shared-memory cleanup against isolated filesystem fixtures."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/multi_node/srt-slurm-recipes/configs/clean_stale_shm.sh"
)


def run_cleanup(directory: Path, job_start: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash", "-c", 'source "$1"; clean_stale_shm "$2" "$3"',
            "bash", str(SCRIPT), str(directory), job_start,
        ],
        env={**os.environ, "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=False,
    )


def segment(path: Path, timestamp: int) -> None:
    path.write_text("segment")
    os.utime(path, (timestamp, timestamp))


def test_cleanup_removes_only_matching_old_regular_files(tmp_path: Path) -> None:
    job_start = int(time.time())
    for name in ("vader_segment.old", "nccl-old", "sem.old", "psm3old", "fe80::old"):
        segment(tmp_path / name, job_start - 601)
    segment(tmp_path / "nccl-boundary", job_start - 600)
    segment(tmp_path / "nccl-current", job_start)
    segment(tmp_path / "unrelated", job_start - 3600)
    directory = tmp_path / "nccl-directory"
    directory.mkdir()
    segment(directory / "child", job_start - 3600)
    os.utime(directory, (job_start - 3600, job_start - 3600))
    (tmp_path / "nccl-link").symlink_to(tmp_path / "unrelated")

    result = run_cleanup(tmp_path, str(job_start))

    assert result.returncode == 0, result.stderr
    assert "removed 5 stale segments" in result.stdout
    assert {path.name for path in tmp_path.iterdir()} == {
        "nccl-boundary", "nccl-current", "unrelated", "nccl-directory", "nccl-link",
    }
    assert (directory / "child").read_text() == "segment"
    assert (tmp_path / "nccl-link").is_symlink()


def test_delayed_repeated_setup_keeps_current_allocation_segments(tmp_path: Path) -> None:
    job_start = int(time.time()) - 3600
    segment(tmp_path / "nccl-previous", job_start - 601)
    segment(tmp_path / "nccl-this-job", job_start + 1)

    first = run_cleanup(tmp_path, str(job_start))
    second = run_cleanup(tmp_path, str(job_start))

    assert first.returncode == second.returncode == 0, first.stderr + second.stderr
    assert "removed 1 stale segments" in first.stdout
    assert "removed 0 stale segments" in second.stdout
    assert (tmp_path / "nccl-this-job").read_text() == "segment"
    assert not (tmp_path / "nccl-previous").exists()


@pytest.mark.parametrize("job_start", ["invalid", "0", "9999999999"])
def test_invalid_allocation_start_leaves_files_untouched(tmp_path: Path, job_start: str) -> None:
    segment(tmp_path / "nccl-old", 1)

    result = run_cleanup(tmp_path, job_start)

    assert result.returncode != 0
    assert "ValueError" in result.stderr
    assert (tmp_path / "nccl-old").read_text() == "segment"
