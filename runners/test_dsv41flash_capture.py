import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("concurrency,floor,expected", [(1, "", 8), (1, "64", 64), (2, "64", 64), (4, "64", 64), (8, "64", 64), (16, "64", 128)])
@pytest.mark.parametrize("eval_only", ["false", "true"])
def test_capture_floor_keeps_speculative_and_context_settings(
    tmp_path: Path, concurrency: int, floor: str, expected: int, eval_only: str
) -> None:
    env = {**os.environ, "MODEL": "fixture", "TP": "4", "CONC": str(concurrency),
           "KV_OFFLOADING": "none", "TOTAL_CPU_DRAM_GB": "0", "DURATION": "3600",
           "RESULT_DIR": str(tmp_path), "EVAL_ONLY": eval_only, "PORT": "18888",
           "DSV41_MIN_CUDAGRAPH_CAPTURE_SIZE": floor}
    result = subprocess.run(["bash", "-c", r'''
source() { :; }
check_env_vars() { :; }
require_agentic_kv_offload_none() { :; }
hf() { :; }
nvidia-smi() { :; }
resolve_trace_source() { :; }
install_agentic_deps() { :; }
select_available_server_port() { :; }
wait_for_server_ready() { wait "$SERVER_PID"; }
build_replay_cmd() { :; }
run_agentic_replay_and_write_outputs() { :; }
run_eval() { :; }
vllm() { command python3 -c 'import json,os,sys; json.dump(sys.argv[1:],open(os.environ["RESULT_DIR"]+"/args.json","w"))' "$@"; }
builtin source "$1/benchmarks/single_node/agentic/dsv41flash_fp4_vllm_mtp.sh"
''', "bash", str(ROOT)], env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    args = json.loads((tmp_path / "args.json").read_text())
    assert int(args[args.index("--max-cudagraph-capture-size") + 1]) == expected
    assert args[args.index("--max-model-len") + 1] == "1048576"
    spec = json.loads(args[args.index("--speculative-config") + 1])
    assert spec["method"] == "dspark" and spec["num_speculative_tokens"] == 5
    assert spec["rejection_sample_method"] == ("block" if eval_only == "true" else "synthetic")
    if eval_only == "false":
        assert spec["synthetic_acceptance_length"] == 3.51
