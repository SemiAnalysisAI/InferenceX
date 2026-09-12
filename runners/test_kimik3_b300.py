import json
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("model_prefix", ["kimik3", "dsv4"])
def test_b300_staged_target_keeps_kimi_draft_in_persistent_mount(
    tmp_path: Path, model_prefix: str,
) -> None:
    log = tmp_path / "launch.jsonl"
    result = subprocess.run(
        ["bash", "-c", '''
        mkdir() { :; }
        unsquashfs() { return 0; }
        salloc() { echo 'salloc: Granted job allocation 123' >&2; }
        scancel() { :; }
        srun() {
            python3 -c 'import json,os,sys; open(sys.argv[1], "a").write(json.dumps({"args":sys.argv[2:], "draft_root":os.environ.get("WRITABLE_MODELS_DIR")})+"\\n")' "$SRUN_LOG" "$@"
        }
        unset WRITABLE_MODELS_DIR
        export MODEL_PREFIX="$3" PRECISION=fp4 FRAMEWORK=vllm
        export MODEL=moonshotai/Kimi-K3 IS_MULTINODE=false
        export SPEC_DECODING=mtp TP=8 RUNNER_NAME=b300-test IS_AGENTIC=1
        export SCENARIO_SUBDIR=agentic/ EXP_NAME="${MODEL_PREFIX}_tp8_conc1"
        export IMAGE=vllm/test:fixture GITHUB_WORKSPACE="$1" SRUN_LOG="$2"
        cd "$GITHUB_WORKSPACE"
        source runners/launch_b300-dsxe.sh
        ''', "bash", str(REPO_ROOT), str(log), model_prefix],
        capture_output=True, text=True, timeout=10, check=False,
    )
    assert result.returncode == 0, result.stderr
    serve = json.loads(log.read_text().splitlines()[-1])
    mounts = next(arg for arg in serve["args"] if arg.startswith("--container-mounts="))
    assert "/scratch/models:/scratch/models" in mounts
    if model_prefix == "kimik3":
        draft_root = serve["draft_root"]
        assert draft_root
        assert f"{draft_root}:{draft_root}" in mounts
    else:
        assert serve["draft_root"] is None
        assert "/data/home/sa-gha-runner/models" not in mounts
