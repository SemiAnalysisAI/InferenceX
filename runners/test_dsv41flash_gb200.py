import json
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_bash(command: str, *args: Path | str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", command, "bash", *(str(arg) for arg in args)],
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("serve_exit", [0, 42])
def test_gb200_direct_vllm_uses_one_tray_and_propagates_failure(
    tmp_path: Path, serve_exit: int,
) -> None:
    log = tmp_path / "srun.jsonl"
    # Use a temporary image cache while exercising the real import/launch path.
    launcher = tmp_path / "launch_gb200-nv.sh"
    source = (REPO_ROOT / "runners/launch_gb200-nv.sh").read_text()
    source = source.replace('SQUASH_DIR="/mnt/lustre01/users-public/sa-shared"', f'SQUASH_DIR="{tmp_path}"')
    launcher.write_text(source)
    (tmp_path / "slurm_utils.sh").symlink_to(REPO_ROOT / "runners/slurm_utils.sh")
    result = run_bash(
        '''
        mkdir() { :; }
        flock() { :; }
        unsquashfs() { :; }
        srun() {
            python3 -c 'import json,sys; open(sys.argv[1], "a").write(json.dumps(sys.argv[2:])+"\\n")' "$SRUN_LOG" "$@"
            case " $* " in
                *" --container-image="*) return "$SERVE_EXIT" ;;
            esac
        }
        export MODEL_PREFIX=dsv41flash PRECISION=fp4 FRAMEWORK=vllm
        export MODEL=deepseek-ai/DeepSeek-V4.1-Flash IS_MULTINODE=false
        export SPEC_DECODING=mtp TP=4 RUNNER_NAME=gb200-test IS_AGENTIC=1
        export IMAGE=vllm/test:fixture GITHUB_WORKSPACE="$1"
        export SRUN_LOG="$2" SERVE_EXIT="$3"
        cd "$GITHUB_WORKSPACE"
        source "$4"
        ''',
        REPO_ROOT, log, str(serve_exit), launcher,
    )
    assert result.returncode == serve_exit, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    serve = calls[-1]
    assert "--nodes=1" in serve
    assert "--ntasks=1" in serve
    assert "--gpus=4" in serve
    assert "--mem=0" in serve
    assert "--job-name=gb200-test" in serve
    mounts = next(arg for arg in serve if arg.startswith("--container-mounts="))
    assert f"{REPO_ROOT}:/ix," in mounts
    assert mounts.endswith(":/hf-cache")
    script = REPO_ROOT / serve[-1]
    assert serve[-2] == "bash" and script.is_file()
    assert all("nginx" not in " ".join(call) for call in calls)

