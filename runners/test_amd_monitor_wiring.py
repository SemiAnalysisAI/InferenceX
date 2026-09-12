import os
import subprocess
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]

def test_amd_stop_coverage_uses_slowest_gpu_and_normalizes_milliseconds(tmp_path):
    csv = tmp_path / "gpu_metrics.csv"
    csv.write_text('timestamp,gpu,vcn_activity,socket_power\n'
                   '1700000002000,0,"[0, 0]",250\n'
                   '1700000001000,1,"[0, 0]",250\n'
                   '1700000009000,1,"[0, 0]",N/A\n')
    result = subprocess.run(["bash", "-c", 'source "$1"; GPU_METRICS_CSV="$2"; _amd_monitor_min_covered_tick',
                             "test", str(REPO / "benchmarks/benchmark_lib.sh"), str(csv)],
                            capture_output=True, text=True, check=True, timeout=10)
    assert result.stdout.strip() == "1700000001"


def test_amd_shutdown_kills_both_pipeline_processes_when_stream_dies(tmp_path):
    binary = tmp_path / "bin"; binary.mkdir()
    fake = binary / "amd-smi"
    fake.write_text(f'''#!{sys.executable}
import json, sys, time
if "-w" in sys.argv:
    print("timestamp,gpu,socket_power", flush=True)
    while True:
        print(str(int(time.time())) + ",0,250", flush=True); time.sleep(.1)
else:
    print("[]")
''')
    fake.chmod(0o755)
    csv = tmp_path / "gpu_metrics.csv"
    script = '''source "$1"
start_gpu_monitor --output "$2"
source_pid=$GPU_MONITOR_SOURCE_PID
sink_pid=$GPU_MONITOR_PID
kill "$sink_pid"
wait "$sink_pid" 2>/dev/null || true
AMD_MONITOR_STOP_TIMEOUT_S=0
stop_gpu_monitor
if kill -0 "$source_pid" 2>/dev/null; then echo 'source leaked'; exit 1; fi
if kill -0 "$sink_pid" 2>/dev/null; then echo 'sink leaked'; exit 1; fi
[[ ! -p "$2.pipe.$$" ]]
'''
    subprocess.run(["bash", "-c", script, "test", str(REPO / "benchmarks/benchmark_lib.sh"), str(csv)],
                   env={**os.environ, "PATH": f"{binary}:{os.environ['PATH']}"},
                   capture_output=True, text=True, check=True, timeout=10)


def test_amd_multinode_selects_equal_local_tensor_ranks(tmp_path):
    arguments = tmp_path / "collector.args"
    # Mock only the downstream collector command; exercise actual topology routing.
    script = f'''source {str(REPO / 'benchmarks/multi_node/amd_utils/power.sh')!r}
bash() {{ printf '%s\\0' "$@" > {str(arguments)!r}; }}
start_amd_multinode_power
wait "$POWERX_COLLECTOR_PID"
'''
    subprocess.run(["bash", "-c", script], env={**os.environ, "BENCH_INPUT_LEN": "8192",
                   "BENCH_OUTPUT_LEN": "1024", "PREFILL_TP_SIZE": "12", "DECODE_TP_SIZE": "12",
                   "GPUS_PER_NODE": "8", "xP": "1", "yD": "1", "NNODES": "4", "NODE_RANK": "1",
                   "WS_PATH": str(tmp_path), "BENCHMARK_LOGS_DIR": str(tmp_path), "SLURM_JOB_ID": "123",
                   "IS_AGENTIC": "0", "EVAL_ONLY": "false", "DRY_RUN": "0"},
                   capture_output=True, text=True, timeout=10, check=True)
    args = arguments.read_bytes().decode().split("\0")
    assert args[-6:-1] == ["amd", "1", "prefill", "0,1,2,3,4,5", "4"]
