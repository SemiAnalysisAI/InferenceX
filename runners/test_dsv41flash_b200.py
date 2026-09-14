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


def test_b200_v41_uses_hf_cache_without_mounting_model_id(tmp_path: Path) -> None:
    log = tmp_path / "launch.jsonl"
    result = run_bash(
        '''
        mkdir() { :; }
        salloc() { :; }
        squeue() { echo 123; }
        srun() {
            python3 -c 'import json,os,sys; open(sys.argv[1], "a").write(json.dumps({"args":sys.argv[2:], "model":os.environ["MODEL"], "cache":os.environ["HF_HUB_CACHE"]})+"\\n")' "$SRUN_LOG" "$@"
        }
        export MODEL_PREFIX=dsv41flash PRECISION=fp4 FRAMEWORK=vllm
        export MODEL=deepseek-ai/DeepSeek-V4.1-Flash IS_MULTINODE=false
        export SPEC_DECODING=mtp TP=4 RUNNER_NAME=b200-test IS_AGENTIC=1
        export SCENARIO_SUBDIR=agentic/ EXP_NAME=dsv41flash_tp4_conc1
        export IMAGE=vllm/test:fixture GITHUB_WORKSPACE="$1" SRUN_LOG="$2"
        cd "$GITHUB_WORKSPACE"
        source runners/launch_b200-nscale-compat.sh
        ''',
        REPO_ROOT, log,
    )
    assert result.returncode == 0, result.stderr
    serve = json.loads(log.read_text().splitlines()[-1])
    assert serve["model"] == "deepseek-ai/DeepSeek-V4.1-Flash"
    assert serve["cache"] == "/hf-cache"
    mounts = next(arg for arg in serve["args"] if arg.startswith("--container-mounts="))
    assert f"{REPO_ROOT}:/ix," in mounts
    assert ":/hf-cache," in mounts
    assert "deepseek-ai/" not in mounts
    assert serve["args"][-2] == "bash"
    assert (REPO_ROOT / serve["args"][-1]).is_file()
