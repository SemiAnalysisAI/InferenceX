import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "tp,concurrency,expected_capture,expected_batched_tokens,expected_memory_utilization",
    [
        (4, 1, 2046, 2048, None),
        (4, 8, 8190, 8192, None),
        (2, 128, 2046, 2048, "0.97"),
    ],
)
def test_b300_capture_tiers_build_expected_vllm_command(
    tmp_path: Path,
    tp: int,
    concurrency: int,
    expected_capture: int,
    expected_batched_tokens: int,
    expected_memory_utilization: str | None,
) -> None:
    env = {
        **os.environ,
        "MODEL": "fixture",
        "TP": str(tp),
        "CONC": str(concurrency),
        "KV_OFFLOADING": "none",
        "TOTAL_CPU_DRAM_GB": "0",
        "DURATION": "3600",
        "RESULT_DIR": str(tmp_path),
        "EVAL_ONLY": "false",
        "PORT": "18888",
    }
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
wait_for_server_ready() { wait "$SERVER_PID"; }
build_replay_cmd() { :; }
run_agentic_replay_and_write_outputs() { :; }
run_eval() { :; }
vllm() { command python3 -c 'import json,os,sys; json.dump(sys.argv[1:],open(os.environ["RESULT_DIR"]+"/args.json","w"))' "$@"; }
builtin source "$1/benchmarks/single_node/agentic/dsv41flash_fp4_b300_vllm_mtp.sh"
''',
            "bash",
            str(ROOT),
        ],
        env=env,
        capture_output=True,
        check=False,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr

    args = json.loads((tmp_path / "args.json").read_text())
    assert int(args[args.index("--max-cudagraph-capture-size") + 1]) == expected_capture
    assert int(args[args.index("--max-num-batched-tokens") + 1]) == expected_batched_tokens
    assert args[args.index("--max-num-seqs") + 1] == "256"
    compilation = json.loads(args[args.index("--compilation-config") + 1])
    assert compilation["mode"] == "VLLM_COMPILE"
    assert compilation["cudagraph_mode"] == "FULL_AND_PIECEWISE"
    assert compilation["cudagraph_capture_sizes"][-1] == expected_capture

    if expected_memory_utilization is None:
        assert "--gpu-memory-utilization" not in args
    else:
        assert args[args.index("--gpu-memory-utilization") + 1] == expected_memory_utilization
