import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "tp,eval_only,rejection,offload",
    [
        (4, "false", "synthetic", False),
        (2, "false", "synthetic", True),
        (2, "true", "block", True),
    ],
)
def test_mi355x_serving_and_client_modes(tmp_path, tp, eval_only, rejection, offload):
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
        "PORT": "18888",
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
run_agentic_replay_and_write_outputs() { printf agentic > "$RESULT_DIR/client"; }
run_eval() { printf eval > "$RESULT_DIR/client"; }
bash() { printf '%s\n' "$@" > "$RESULT_DIR/patch_call"; }
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
    expected_client = "eval" if eval_only == "true" else "agentic"
    assert (tmp_path / "client").read_text() == expected_client
