import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "tp,agentic,eval_only,rejection,offload",
    [
        (4, "1", "false", "synthetic", False),
        (2, "1", "false", "synthetic", True),
        (2, "1", "true", "block", True),
        (2, "0", "false", "block", True),
        (2, "0", "true", "block", True),
    ],
)
def test_mi355x_serving_and_client_modes(
    tmp_path, tp, agentic, eval_only, rejection, offload
):
    env = {
        **os.environ,
        "MODEL": "fixture",
        "TP": str(tp),
        "CONC": "2",
        "KV_OFFLOADING": "none",
        "TOTAL_CPU_DRAM_GB": "0",
        "DURATION": "3600",
        "RESULT_DIR": str(tmp_path),
        "EVAL_ONLY": eval_only,
        "IS_AGENTIC": agentic,
        "PORT": "18888",
        "ISL": "8192",
        "OSL": "1024",
        "RANDOM_RANGE_RATIO": "1",
        "RESULT_FILENAME": "fixture",
    }
    result = subprocess.run(
        [
            "bash",
            "-c",
            r"""
source() { :; }
check_env_vars() { :; }
require_agentic_kv_offload_none() { :; }
hf() { :; }
resolve_trace_source() { :; }
install_agentic_deps() { :; }
stop_background_process_tree() { :; }
wait_for_server_ready() { wait "$SERVER_PID"; }
build_replay_cmd() { :; }
start_gpu_monitor() { :; }
stop_gpu_monitor() { :; }
run_agentic_replay_and_write_outputs() { printf agentic > "$RESULT_DIR/client"; }
run_eval() { printf eval > "$RESULT_DIR/client"; }
bash() { printf '%s\n' "$@" > "$RESULT_DIR/patch_call"; }
run_benchmark_serving() {
    printf fixed > "$RESULT_DIR/client"
    printf '%s\n' "$@" > "$RESULT_DIR/client_args"
}
vllm() {
    python3 -c 'import json,os,sys; json.dump(sys.argv[1:],open(os.environ["RESULT_DIR"]+"/args.json","w"))' "$@"
}
builtin source "$1/benchmarks/single_node/agentic/dsv41flash_fp4_mi355x_vllm_mtp.sh"
""",
            "bash",
            str(ROOT),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    args = json.loads((tmp_path / "args.json").read_text())
    assert args[args.index("--tensor-parallel-size") + 1] == str(tp)
    spec = json.loads(args[args.index("--speculative-config") + 1])
    assert spec["rejection_sample_method"] == rejection
    assert spec["enable_adaptive_verification"] is False
    assert spec["num_speculative_tokens"] == 5
    assert ("synthetic_acceptance_length" in spec) == (rejection == "synthetic")
    assert (tmp_path / "patch_call").exists() is offload
    if offload:
        assert json.loads(args[args.index("--engram-config") + 1]) == {
            "cpu_offload": True
        }
    else:
        assert "--engram-config" not in args
    expected_client = (
        "eval" if eval_only == "true" else ("agentic" if agentic == "1" else "fixed")
    )
    assert (tmp_path / "client").read_text() == expected_client
    if expected_client == "fixed":
        client = (tmp_path / "client_args").read_text().splitlines()
        assert "--use-chat-template" in client
        assert client[client.index("--tokenizer-mode") + 1] == "deepseek_v41"
        assert client[client.index("--input-len") + 1] == "8192"
        assert client[client.index("--output-len") + 1] == "1024"
