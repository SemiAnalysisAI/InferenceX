"""E2E tests for the GB200 Engram-on-disk recipe.

Two things are covered: that dsv41flashssd routes to its own single-node
recipe and gets the table directory mounted, and that the placement guard
refuses to benchmark a shared filesystem unless that was asked for.
"""
import json
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LIB = REPO_ROOT / "benchmarks/benchmark_lib.sh"


def run_bash(command: str, *args: Path | str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", command, "bash", *(str(arg) for arg in args)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_gb200_ssd_recipe_routes_and_mounts_the_table_dir(tmp_path: Path) -> None:
    log = tmp_path / "srun.jsonl"
    table_dir = tmp_path / "engram"
    launcher = tmp_path / "launch_gb200-nv.sh"
    source = (REPO_ROOT / "runners/launch_gb200-nv.sh").read_text()
    source = source.replace(
        'SQUASH_DIR="/mnt/lustre01/users-public/sa-shared"', f'SQUASH_DIR="{tmp_path}"'
    )
    launcher.write_text(source)
    (tmp_path / "slurm_utils.sh").symlink_to(REPO_ROOT / "runners/slurm_utils.sh")
    result = run_bash(
        '''
        flock() { :; }
        unsquashfs() { :; }
        srun() {
            python3 -c 'import json,sys; open(sys.argv[1], "a").write(json.dumps(sys.argv[2:])+"\\n")' "$SRUN_LOG" "$@"
            return 0
        }
        export MODEL_PREFIX=dsv41flashssd PRECISION=fp4 FRAMEWORK=vllm
        export MODEL=deepseek-ai/DeepSeek-V4.1-Flash IS_MULTINODE=false
        export SPEC_DECODING=mtp TP=4 RUNNER_NAME=gb200-ssd-test IS_AGENTIC=1
        export IMAGE=vllm/test:fixture GITHUB_WORKSPACE="$1"
        export SRUN_LOG="$2" ENGRAM_SSD_DIR="$3"
        cd "$GITHUB_WORKSPACE"
        source "$4"
        ''',
        REPO_ROOT, log, table_dir, launcher,
    )
    assert result.returncode == 0, result.stderr
    serve = [json.loads(line) for line in log.read_text().splitlines()][-1]

    # Routed to the SSD recipe, not the in-DRAM one.
    script = REPO_ROOT / serve[-1]
    assert serve[-2] == "bash"
    assert script.name == "dsv41flashssd_fp4_gb200_vllm_mtp.sh"
    assert script.is_file()

    # The table directory is mounted at the same path inside the container, so
    # the recipe's df-based guard inspects the real filesystem and not the
    # container overlay.
    mounts = next(arg for arg in serve if arg.startswith("--container-mounts="))
    assert f"{table_dir}:{table_dir}" in mounts
    assert "/hf-cache" in mounts
    assert table_dir.is_dir()
    assert "--nodes=1" in serve and "--ntasks=1" in serve and "--gpus=4" in serve


GUARD = f'''
    eval "$(awk '/^require_engram_table_placement\\(\\) \\{{/,/^\\}}/' {LIB})"
    df() {{ printf 'Filesystem Type\\n/dev/x %s\\n' "$FSTYPE"; }}
    require_engram_table_placement /tmp/engram "$SHARED"
'''


@pytest.mark.parametrize(
    "fstype,shared,expect_rc,expect_text",
    [
        ("xfs", "0", 0, "engram_placement=local"),
        ("ext4", "0", 0, "engram_placement=local"),
        # A shared mount must be opted into, or a network filesystem gets
        # silently benchmarked as if it were local disk.
        ("lustre", "0", 1, "not local disk"),
        ("nfs4", "0", 1, "not local disk"),
        ("tmpfs", "0", 1, "not local disk"),
        ("lustre", "1", 0, "engram_placement=shared"),
        ("nfs4", "1", 0, "engram_placement=shared"),
        # Claiming shared while pointing at local disk is also a mislabel.
        ("xfs", "1", 1, "which is local"),
        ("", "0", 1, "Could not determine"),
    ],
)
def test_placement_guard(fstype: str, shared: str, expect_rc: int, expect_text: str) -> None:
    result = subprocess.run(
        ["bash", "-c", GUARD],
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "FSTYPE": fstype, "SHARED": shared},
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == expect_rc, f"{result.stdout}\n{result.stderr}"
    assert expect_text in (result.stdout + result.stderr)
