import json
import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
H100_SCRIPT = (
    REPO_ROOT / "benchmarks/single_node/agentic/dsv41flash_fp4_h100_vllm_mtp.sh"
)

LAUNCH_HARNESS = '''
    salloc() { :; }
    squeue() { echo 123; }
    srun() {
        python3 -c 'import json,os,sys; open(sys.argv[1], "a").write(json.dumps({"args": sys.argv[2:], "result_dir": os.environ.get("RESULT_DIR", "")})+"\\n")' "$SRUN_LOG" "$@"
    }
    scancel() { :; }
    export IS_MULTINODE=false IS_AGENTIC=1 SCENARIO_SUBDIR=agentic/
    export IMAGE=vllm/vllm-openai:deepseekv41-flash-0909
    export HF_HUB_CACHE=/mnt/hf_hub_cache/ RESULT_DIR=/workspace/results
    export GITHUB_WORKSPACE="$1" SRUN_LOG="$2"
    cd "$GITHUB_WORKSPACE"
'''


def launch(log: Path, **env: str) -> dict:
    exports = " ".join(f"export {key}={value};" for key, value in env.items())
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"{LAUNCH_HARNESS}\n{exports}\nsource runners/launch_h100-dgxc-slurm.sh",
            "bash",
            str(REPO_ROOT),
            str(log),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(log.read_text().splitlines()[-1])


def test_h100_flash_runs_its_own_script_from_an_ix_mount(tmp_path: Path) -> None:
    serve = launch(
        tmp_path / "launch.jsonl",
        MODEL_PREFIX="dsv41flash",
        MODEL="deepseek-ai/DeepSeek-V4.1-Flash",
        PRECISION="fp4",
        FRAMEWORK="vllm",
        SPEC_DECODING="mtp",
        TP="8",
        RUNNER_NAME="h100-test",
        EXP_NAME="dsv41flash_tp8_conc1",
    )
    script = serve["args"][-1]
    assert script == "benchmarks/single_node/agentic/dsv41flash_fp4_h100_vllm_mtp.sh"
    assert (REPO_ROOT / script).is_file()

    mounts = next(arg for arg in serve["args"] if arg.startswith("--container-mounts="))
    assert f"{REPO_ROOT}:/ix/," in mounts
    # AgentX must not write runtime directories under /workspace.
    assert "/workspace" not in mounts
    assert serve["result_dir"] == "/ix/results"


def test_h100_still_resolves_scripts_without_a_framework_tag(tmp_path: Path) -> None:
    """The pre-framework h100 recipes keep working after the suffix change."""
    serve = launch(
        tmp_path / "launch.jsonl",
        MODEL_PREFIX="qwen3.5",
        MODEL="Qwen/Qwen3.5-397B-A17B-FP8",
        PRECISION="fp8",
        FRAMEWORK="sglang",
        SPEC_DECODING="mtp",
        TP="8",
        RUNNER_NAME="h100-test",
        EXP_NAME="qwen3.5_tp8_conc1",
    )
    assert serve["args"][-1] == "benchmarks/single_node/agentic/qwen3.5_fp8_h100_mtp.sh"
    mounts = next(arg for arg in serve["args"] if arg.startswith("--container-mounts="))
    assert f"{REPO_ROOT}:/workspace/," in mounts
    assert serve["result_dir"] == "/workspace/results"


def test_indexer_buffer_fits_an_80gb_card() -> None:
    """The batched-token cap is what keeps the sparse-attention indexer in VRAM.

    The indexer allocates [max-num-batched-tokens, max-model-len] at 2 bytes.
    Raising either past this budget reproduces the concurrency-1 OOM in run
    34467029236, so assert the product, not the literal flag value.
    """
    script = H100_SCRIPT.read_text()
    batched = int(re.search(r"^MAX_NUM_BATCHED_TOKENS=(\d+)$", script, re.M).group(1))
    context = int(re.search(r"--max-model-len (\d+)", script).group(1))

    indexer_gib = batched * context * 2 / 1024**3
    # ~35.9 GiB/GPU of resident weights out of 80 GB, and the KV cache still
    # needs its share of the rest.
    assert indexer_gib <= 8.0, f"indexer buffer is {indexer_gib:.1f} GiB"


def test_h100_caps_the_scheduler_batch_to_the_trajectory_concurrency() -> None:
    """vLLM's default max_num_seqs of 1024 oversizes buffers for AgentX."""
    script = H100_SCRIPT.read_text()
    assert "--max-num-seqs" in script
    assert re.search(r"^MAX_NUM_SEQS=\$\(\(2 \* CONC\)\)$", script, re.M)


@pytest.mark.parametrize("backend", ["mooncake", ""])
@pytest.mark.parametrize("eval_only", ["false", "true"])
def test_h100_allocator_environment_matches_kv_backend(
    tmp_path: Path, backend: str, eval_only: str,
) -> None:
    env = {
        **os.environ,
        "MODEL": "fixture", "TP": "8", "CONC": "1",
        "KV_OFFLOADING": "dram" if backend else "none",
        "KV_OFFLOAD_BACKEND": backend, "TOTAL_CPU_DRAM_GB": "1000",
        "RESULT_DIR": str(tmp_path), "DURATION": "1", "PORT": "18888",
        "EVAL_ONLY": eval_only,
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    result = subprocess.run(
        ["bash", "-c", r'''
source() { :; }
check_env_vars() { :; }
require_agentic_kv_offload_none() { :; }
require_agentic_kv_offload_backend() { :; }
hf() { :; }
nvidia-smi() { :; }
resolve_trace_source() { :; }
install_agentic_deps() { :; }
select_available_server_port() { :; }
setup_agentic_mooncake() { :; }
stop_background_process_tree() { :; }
wait_for_server_ready() { wait "$SERVER_PID"; }
build_replay_cmd() { :; }
run_agentic_replay_and_write_outputs() { :; }
run_eval() { :; }
vllm() {
    python3 -c 'import json,os; json.dump({key:os.environ.get(key) for key in ["PYTORCH_ALLOC_CONF","PYTORCH_CUDA_ALLOC_CONF"]},open(os.environ["RESULT_DIR"]+"/allocator.json","w"))'
}
builtin source "$1"
''', "bash", str(H100_SCRIPT)],
        env=env, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    allocator = json.loads((tmp_path / "allocator.json").read_text())
    if backend == "mooncake":
        assert allocator == {"PYTORCH_ALLOC_CONF": None, "PYTORCH_CUDA_ALLOC_CONF": None}
    else:
        assert allocator["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
