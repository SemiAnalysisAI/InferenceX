"""Regression checks for Slurm benchmark exit-code propagation."""

from pathlib import Path


REPO_ROOT = Path(__file__).parents[1]
LAUNCHERS = (
    "runners/launch_h100-dgxc-slurm.sh",
    "runners/launch_h200-dgxc-slurm.sh",
    "runners/launch_mi300x-amds.sh",
    "runners/launch_mi325x-amds.sh",
)


def test_slurm_launchers_fail_fast_and_cleanup_without_overwriting_status():
    for relative_path in LAUNCHERS:
        source = (REPO_ROOT / relative_path).read_text()

        assert "set -e" in source
        assert "pipefail" in source
        assert "trap 'rc=$?; scancel" in source
        assert 'exit "$rc"' in source
