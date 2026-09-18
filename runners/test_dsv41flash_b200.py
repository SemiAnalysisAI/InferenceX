"""Run the B200 recipe entrypoints with GPU/network collaborators replaced."""

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "scenario,eval_only,concurrency,capture",
    [
        ("fixed_seq_len", False, 1, 64),
        # TP2 fixed-seq capture is capped at 512 (run 35316389982 OOMed at 1024);
        # the eval halves it to 256 and raises the memory fraction (run 35355746550).
        ("fixed_seq_len", True, 128, 256),
        ("agentic", False, 16, 128),
        ("agentic", True, 2, 64),
    ],
)
def test_b200_tp2_launch_and_workload(tmp_path, scenario, eval_only, concurrency, capture):
    env = {
        **os.environ,
        "MODEL": "fixture",
        "TP": "2",
        "CONC": str(concurrency),
        "ISL": "8192",
        "OSL": "1024",
        "RANDOM_RANGE_RATIO": "1",
        "RESULT_FILENAME": "fixture-result",
        "RESULT_DIR": str(tmp_path),
        "MAX_MODEL_LEN": "9472",
        "INFMAX_CONTAINER_WORKSPACE": str(tmp_path),
        "KV_OFFLOADING": "none",
        "TOTAL_CPU_DRAM_GB": "0",
        "DURATION": "3600",
        "DSV41_MIN_CUDAGRAPH_CAPTURE_SIZE": "64",
        "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
        "EVAL_ONLY": str(eval_only).lower(),
        "PORT": "18888",
    }
    script = ROOT / f"benchmarks/single_node/{scenario}/dsv41flash_fp4_b200_vllm_mtp.sh"
    result = subprocess.run(
        [
            "bash",
            "-c",
            r'''
source() { :; }
check_env_vars() { :; }
require_agentic_kv_offload_none() { :; }
hf() { :; }
nvidia-smi() { :; }
resolve_trace_source() { :; }
install_agentic_deps() { :; }
select_available_server_port() { :; }
start_gpu_monitor() { :; }
setup_eval_context() { export EVAL_MAX_MODEL_LEN=16384; }
wait_for_server_ready() { wait "$SERVER_PID"; }
build_replay_cmd() { :; }
record() {
    command python3 -c 'import json,os,sys; json.dump(sys.argv[2:],open(os.environ["RESULT_DIR"]+"/"+sys.argv[1]+".json","w"))' "$@"
}
vllm() { record serve "$@"; }
run_eval() { record workload eval "$@"; }
append_lm_eval_summary() { record staged lm-eval; }
run_benchmark_serving() { record workload fixed "$@"; }
run_agentic_replay_and_write_outputs() { record workload agentic "$@"; }
builtin source "$1"
''',
            "bash",
            str(script),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    args = json.loads((tmp_path / "serve.json").read_text())
    assert args[args.index("--tensor-parallel-size") + 1] == "2"
    assert json.loads(args[args.index("--engram-config") + 1]) == {"cpu_offload": True}
    assert args[args.index("--tokenizer-mode") + 1] == "deepseek_v41"
    assert int(args[args.index("--max-cudagraph-capture-size") + 1]) == capture
    if scenario == "fixed_seq_len":
        # The 8k1k arm serves the matrix context, and evals the eval context.
        assert args[args.index("--max-model-len") + 1] == ("16384" if eval_only else "9472")
        # Only the TP2 eval raises the memory fraction; throughput points keep the default.
        if eval_only:
            assert args[args.index("--gpu-memory-utilization") + 1] == "0.95"
        else:
            assert "--gpu-memory-utilization" not in args
    else:
        assert args[args.index("--max-model-len") + 1] == "1048576"
        # TP2 bounds the scheduler batch to the AgentX fan-out, floored at 16
        # (run 35320655804: smaller values broke FlashInfer autotune).
        assert args[args.index("--max-num-seqs") + 1] == str(max(16, 2 * concurrency))
    spec = json.loads(args[args.index("--speculative-config") + 1])
    assert spec["method"] == "dspark"
    assert spec["num_speculative_tokens"] == 5
    assert spec["rejection_sample_method"] == ("block" if eval_only else "synthetic")
    assert spec["enable_adaptive_verification"] is eval_only
    if eval_only:
        assert "synthetic_acceptance_length" not in spec
    else:
        assert spec["synthetic_acceptance_length"] == 3.51
    workload = json.loads((tmp_path / "workload.json").read_text())
    if eval_only:
        if scenario == "fixed_seq_len":
            # Non-agentic evals pick lm-eval explicitly and must stage their
            # artifacts into the workspace root; the agentic eval path stages itself.
            assert workload == ["eval", "--framework", "lm-eval", "--port", "18888"]
            assert json.loads((tmp_path / "staged.json").read_text()) == ["lm-eval"]
        else:
            assert workload == ["eval", "--port", "18888"]
    elif scenario == "agentic":
        assert workload == ["agentic", str(tmp_path)]
    else:
        assert workload[0] == "fixed"
        assert "--use-chat-template" in workload
        assert workload[workload.index("--tokenizer-mode") + 1] == "deepseek_v41"
        assert workload[workload.index("--input-len") + 1] == "8192"
        assert workload[workload.index("--output-len") + 1] == "1024"
        assert workload[workload.index("--num-prompts") + 1] == "10"
        # The result JSON goes to the container repository root, not RESULT_DIR.
        assert workload[workload.index("--result-dir") + 1] == f"{tmp_path}/"
