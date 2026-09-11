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
def test_gb300_direct_vllm_uses_one_tray_and_propagates_failure(
    tmp_path: Path, serve_exit: int,
) -> None:
    log = tmp_path / "srun.jsonl"
    result = run_bash(
        '''
        mkdir() { :; }
        srun() {
            python3 -c 'import json,sys; open(sys.argv[1], "a").write(json.dumps(sys.argv[2:])+"\\n")' "$SRUN_LOG" "$@"
            case " $* " in
                *" --container-image="*) return "$SERVE_EXIT" ;;
            esac
        }
        export MODEL_PREFIX=dsv41flash PRECISION=fp4 FRAMEWORK=vllm
        export MODEL=deepseek-ai/DeepSeek-V4.1-Flash IS_MULTINODE=false
        export SPEC_DECODING=mtp TP=4 RUNNER_NAME=gb300-test IS_AGENTIC=1
        export IMAGE=vllm/test:fixture GITHUB_WORKSPACE="$1"
        export SRUN_LOG="$2" SERVE_EXIT="$3"
        cd "$GITHUB_WORKSPACE"
        source runners/launch_gb300-nv.sh
        ''',
        REPO_ROOT, log, str(serve_exit),
    )
    assert result.returncode == serve_exit, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    serve = calls[-1]
    assert "--nodes=1" in serve
    assert "--ntasks=1" in serve
    assert "--gpus=4" in serve
    assert "--mem=0" in serve
    assert "--job-name=gb300-test" in serve
    mounts = next(arg for arg in serve if arg.startswith("--container-mounts="))
    assert f"{REPO_ROOT}:/ix," in mounts
    assert mounts.endswith(":/hf-cache")
    script = REPO_ROOT / serve[-1]
    assert serve[-2] == "bash" and script.is_file()
    assert all("nginx" not in " ".join(call) for call in calls)


@pytest.mark.parametrize("eval_only", ["false", "true"])
def test_upstream_gb300_command_changes_only_throughput_acceptance(tmp_path: Path, eval_only: str) -> None:
    import os
    result = subprocess.run(
        ["bash", "-c", r'''
        source() { :; }
        check_env_vars() { :; }
        require_agentic_kv_offload_none() { :; }
        hf() { :; }
        nvidia-smi() { :; }
        resolve_trace_source() { :; }
        install_agentic_deps() { :; }
        select_available_server_port() { export PORT=23456; }
        vllm() {
          python3 -c 'import json,os,sys; open(os.environ["RESULT_DIR"]+"/command.json","w").write(json.dumps({"args":sys.argv[1:],"timeout":os.environ["VLLM_ENGINE_READY_TIMEOUT_S"],"rust":os.environ["VLLM_USE_RUST_FRONTEND"]}))' "$@"
        }
        wait_for_server_ready() { wait "$SERVER_PID"; }
        run_eval() { :; }
        build_replay_cmd() { :; }
        run_agentic_replay_and_write_outputs() { :; }
        builtin source "$1"
        ''', "bash", str(REPO_ROOT / "benchmarks/single_node/agentic/dsv41flash_fp4_gb300_vllm_mtp.sh")],
        env={**os.environ, "MODEL": "deepseek-ai/DeepSeek-V4.1-Flash", "TP": "4", "CONC": "1", "RESULT_DIR": str(tmp_path), "EVAL_ONLY": eval_only, "VLLM_ENGINE_READY_TIMEOUT_S": "7200"},
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    command = json.loads((tmp_path / "command.json").read_text())
    args = command["args"]
    spec_index = args.index("--speculative-config") + 1
    spec = json.loads(args[spec_index])
    expected_spec = {"method": "dspark", "num_speculative_tokens": 5, "draft_sample_method": "probabilistic", "rejection_sample_method": "block", "enable_adaptive_verification": True}
    if eval_only == "false":
        expected_spec.update(rejection_sample_method="synthetic", synthetic_acceptance_length=3.51)
    assert spec == expected_spec
    args[spec_index] = "SPEC"
    assert args == ["serve", "deepseek-ai/DeepSeek-V4.1-Flash", "--tokenizer-mode", "deepseek_v41", "--tensor-parallel-size", "4", "--tool-call-parser", "deepseek_v41", "--enable-auto-tool-choice", "--reasoning-parser", "deepseek_v41", "--speculative-config", "SPEC", "--language-model-only", "--port", "23456"]
    assert command["timeout"] == "3600"
    assert command["rust"] == "1"
