#!/usr/bin/env bash

# Usage: check_env_vars VAR1 VAR2 ...; exits 1 if any is unset.
check_env_vars() {
    local missing_vars=()

    for var_name in "$@"; do
        if [[ -z "${!var_name:-}" ]]; then
            missing_vars+=("$var_name")
        fi
    done

    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        echo "Error: The following required environment variables are not set:"
        for var in "${missing_vars[@]}"; do
            echo "  - $var"
        done
        exit 1
    fi
}

# Launchers may load only input validation, without benchmark initialization.
if [[ "${1-}" == "--validation-only" ]]; then
    return 0
fi

# Keep Python bytecode out of the mounted workspace. Benchmark jobs often run as
# root inside containers, and root-owned cache directories break future checkout
# cleanup on self-hosted runners.
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/inferencex-pycache}"
mkdir -p "$PYTHONPYCACHEPREFIX" 2>/dev/null || true
INFERENCEX_BENCHMARK_LIB_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
)"
INFERENCEX_REPO_ROOT="$(
    cd "$INFERENCEX_BENCHMARK_LIB_DIR/.." && pwd
)"

# Workflows supply PORT; launchers may select a cluster-specific port.

# Opt-in for recipes running in the host network namespace. Probe the preferred
# port on the compute node; fall back to an OS-selected port if it is occupied.
# Call immediately before server launch and construct client URLs afterward.
select_available_server_port() {
    check_env_vars PORT
    PORT=$(python3 - "${PORT}" <<'PYPORT'
import errno
import socket
import sys

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    try:
        sock.bind(("0.0.0.0", int(sys.argv[1])))
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        sock.bind(("0.0.0.0", 0))
    print(sock.getsockname()[1])
PYPORT
    ) || return $?
    export PORT
}

agentic_kv_offload_enabled() {
    if [[ -z "${KV_OFFLOADING+x}" || -z "$KV_OFFLOADING" ]]; then
        echo "Error: KV_OFFLOADING must be set for agentic benchmarks" >&2
        exit 1
    fi
    [[ "$KV_OFFLOADING" != "none" ]]
}

require_agentic_kv_offload_none() {
    if agentic_kv_offload_enabled; then
        echo "Error: expected KV_OFFLOADING=none, got '$KV_OFFLOADING'" >&2
        exit 1
    fi
    if [[ -n "${KV_OFFLOAD_BACKEND:-}" ]]; then
        echo "Error: KV_OFFLOAD_BACKEND must be empty when KV_OFFLOADING=none" >&2
        exit 1
    fi
}

require_agentic_kv_offload_backend() {
    local expected_backend="$1"
    if [[ -z "${KV_OFFLOADING+x}" || -z "$KV_OFFLOADING" ]]; then
        echo "Error: KV_OFFLOADING must be set for agentic benchmarks" >&2
        exit 1
    fi
    case "$KV_OFFLOADING" in
        none)
            if [[ -n "${KV_OFFLOAD_BACKEND:-}" ]]; then
                echo "Error: KV_OFFLOAD_BACKEND must be empty when KV_OFFLOADING=none" >&2
                exit 1
            fi
            return 1
            ;;
        dram)
            if [[ "${KV_OFFLOAD_BACKEND:-}" != "$expected_backend" ]]; then
                echo "Error: expected KV_OFFLOAD_BACKEND=$expected_backend when KV_OFFLOADING=dram, got '${KV_OFFLOAD_BACKEND:-}'" >&2
                exit 1
            fi
            if [[ ! "${TOTAL_CPU_DRAM_GB:-}" =~ ^[1-9][0-9]*$ ]]; then
                echo "Error: DRAM KV offloading requires a positive TOTAL_CPU_DRAM_GB capacity" >&2
                exit 1
            fi
            return 0
            ;;
        *)
            echo "Error: unsupported KV_OFFLOADING value '$KV_OFFLOADING' (expected one of: none, dram)" >&2
            exit 1
            ;;
    esac
}

# vLLM images before upstream #44244 need client-side template kwargs support.
# Keep the compatibility implementation shared until SPEED-Bench's default image
# includes it. Existing upstream support and repeated invocations are no-ops.
apply_chat_template_kwargs_shim() {
    echo "=== Checking vLLM benchmark for --chat-template-kwargs support ==="
    python3 - <<'PYEOF'
import inspect
import re
import sys

import vllm.benchmarks.serve as S
import vllm.benchmarks.datasets.datasets as D

serve_src = open(S.__file__).read()
ds_src = open(D.__file__).read()


def speed_bench_block(src):
    """Byte range of the speed_bench entry in get_samples' dataset_mapping."""
    i = src.find('"speed_bench": lambda:')
    if i == -1:
        return None, None
    end = src.find("\n        }", i)
    return i, (len(src) if end == -1 else end)


def dump(src, needle, path, what):
    print(f"ERROR: {what} not found in {path}", file=sys.stderr)
    i = src.find(needle)
    if i == -1:
        print(f"  (marker {needle!r} absent entirely)", file=sys.stderr)
    else:
        lo = src.rfind("\n", 0, max(0, i - 600)) + 1
        hi = src.find("\n", i + 600)
        print(
            "  surrounding source:\n" + src[lo : hi if hi != -1 else None],
            file=sys.stderr,
        )


# --- Detect the three halves of #44244 independently ------------------------
serve_ok = '"--chat-template-kwargs"' in serve_src

sb_start, sb_end = speed_bench_block(ds_src)
speed_bench_ok = (
    sb_start is not None and "chat_template_kwargs" in ds_src[sb_start:sb_end]
)

sample = D.CustomDataset.sample
sample_ok = (
    "chat_template_kwargs" in inspect.signature(sample).parameters
    or '_ctk = kwargs.get("chat_template_kwargs") or {}' in inspect.getsource(sample)
)

if serve_ok and speed_bench_ok and sample_ok:
    print(
        "upstream --chat-template-kwargs support detected "
        "(vllm-project/vllm#44244); skipping shim"
    )
    sys.exit(0)

print(
    f"patching: serve_cli={serve_ok} speed_bench_dispatch={speed_bench_ok} "
    f"custom_dataset_sample={sample_ok}"
)

# --- Edit 1: serve.py declares the --chat-template-kwargs argument ----------
if not serve_ok:
    serve_old = """    parser.add_argument(
        "--extra-body","""
    serve_new = """    parser.add_argument(
        "--chat-template-kwargs",
        type=json.loads,
        default=None,
        help="JSON dict forwarded to apply_chat_template during "
        "client-side prompt rendering, e.g. to enable reasoning mode.",
    )
    parser.add_argument(
        "--extra-body","""
    if serve_src.count(serve_old) != 1:
        dump(serve_src, '"--extra-body"', S.__file__, "--extra-body anchor")
        sys.exit(1)
    open(S.__file__, "w").write(serve_src.replace(serve_old, serve_new, 1))
    print("patched OK ->", S.__file__)

# --- Edit 2: forward the kwarg into the speed_bench .sample() call ----------
if not speed_bench_ok:
    if sb_start is None:
        dump(ds_src, '"speed_bench"', D.__file__, "speed_bench dispatch")
        sys.exit(1)
    block = ds_src[sb_start:sb_end]
    m = re.search(r"^([ \t]*)output_len=args\.speed_bench_output_len,\n", block, re.M)
    if not m:
        dump(
            ds_src,
            "speed_bench_output_len",
            D.__file__,
            "output_len anchor inside the speed_bench block",
        )
        sys.exit(1)
    line = (
        m.group(1)
        + 'chat_template_kwargs=getattr(args, "chat_template_kwargs", None),\n'
    )
    block = block[: m.end()] + line + block[m.end() :]
    ds_src = ds_src[:sb_start] + block + ds_src[sb_end:]

# --- Edit 3: apply the kwarg in CustomDataset.sample's template call --------
if not sample_ok:
    samp_old = """                # apply template
                if not skip_chat_template:
                    prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        add_generation_prompt=True,
                        tokenize=False,
                    )"""
    samp_new = """                # apply template
                if not skip_chat_template:
                    _ctk = kwargs.get("chat_template_kwargs") or {}
                    prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        add_generation_prompt=True,
                        tokenize=False,
                        **_ctk,
                    )"""
    if ds_src.count(samp_old) != 1:
        dump(
            ds_src,
            "apply_chat_template",
            D.__file__,
            "CustomDataset.sample apply_chat_template anchor",
        )
        sys.exit(1)
    ds_src = ds_src.replace(samp_old, samp_new, 1)

if not speed_bench_ok or not sample_ok:
    open(D.__file__, "w").write(ds_src)
    print("patched OK ->", D.__file__)
PYEOF
}

disable_trtllm_detailed_perf_metrics() {
    local trtllm_root
    local py_executor
    local detailed_metrics_gate="enabled=getattr(self.llm_args, 'return_perf_metrics', False))"

    trtllm_root=$(python3 -c 'from importlib.util import find_spec; from pathlib import Path; print(Path(find_spec("tensorrt_llm").origin).parent)')
    py_executor="$trtllm_root/_torch/pyexecutor/py_executor.py"
    if ! grep -Fq "$detailed_metrics_gate" "$py_executor"; then
        echo "Error: unsupported TensorRT-LLM detailed metrics implementation in $py_executor" >&2
        return 1
    fi
    sed -i "s/enabled=getattr(self.llm_args, 'return_perf_metrics', False))/enabled=False)/" "$py_executor"
    grep -Fq "self.perf_manager = PerfMetricsManager(" "$py_executor"
    grep -Fq "enabled=False)" "$py_executor"
}

# Agentic replays must use the model's native context limit. Ignore inherited
# workflow or shell overrides so neither the server nor AIPerf applies a cap.
_benchmark_caller="${BASH_SOURCE[1]:-}"
if [[ "$_benchmark_caller" == */agentic/* ||
      "$_benchmark_caller" == */agentic_*.sh ||
      "${IS_AGENTIC-}" == "1" ||
      "${SCENARIO_TYPE:-}" == "agentic-coding" ]]; then
    unset MAX_MODEL_LEN
    if [[ -z "${KV_OFFLOADING+x}" || -z "$KV_OFFLOADING" ]]; then
        echo "Error: KV_OFFLOADING must be set for agentic benchmarks" >&2
        exit 1
    fi
    case "$KV_OFFLOADING" in
        none)
            if [[ -n "${KV_OFFLOAD_BACKEND:-}" ]]; then
                echo "Error: KV_OFFLOAD_BACKEND must be empty when KV_OFFLOADING=none" >&2
                exit 1
            fi
            ;;
        dram)
            if [[ -z "${KV_OFFLOAD_BACKEND:-}" || "${KV_OFFLOAD_BACKEND:-}" == "none" ]]; then
                echo "Error: KV_OFFLOAD_BACKEND is required when KV_OFFLOADING=dram" >&2
                exit 1
            fi
            if [[ ! "${TOTAL_CPU_DRAM_GB:-}" =~ ^[1-9][0-9]*$ ]]; then
                echo "Error: DRAM KV offloading requires a positive configured TOTAL_CPU_DRAM_GB capacity" >&2
                exit 1
            fi
            ;;
        *)
            echo "Error: unsupported KV_OFFLOADING value '$KV_OFFLOADING' (expected one of: none, dram)" >&2
            exit 1
            ;;
    esac
fi
unset _benchmark_caller

# GPU monitoring helpers

GPU_MONITOR_PID=""
GPU_MONITOR_VENDOR=""
GPU_MONITOR_INTERVAL=1
GPU_METRICS_CSV="${GPU_METRICS_CSV:-gpu_metrics.csv}"
NVIDIA_GPU_MONITOR_QUERY="timestamp,index,power.draw,temperature.gpu,clocks.current.sm,clocks.current.memory,utilization.gpu,utilization.memory"
export GPU_METRICS_CSV

# Background nvidia-smi/amd-smi sampler writing CSV.
# Usage: start_gpu_monitor [--output /path/to/output.csv] [--interval 1]
start_gpu_monitor() {
    local output="$GPU_METRICS_CSV"
    local interval=1

    while [[ $# -gt 0 ]]; do
        case $1 in
            --output)   output="$2"; shift 2 ;;
            --interval) interval="$2"; shift 2 ;;
            *)          shift ;;
        esac
    done

    GPU_METRICS_CSV="$output"
    GPU_MONITOR_INTERVAL="$interval"
    export GPU_METRICS_CSV

    if command -v nvidia-smi &>/dev/null; then
        GPU_MONITOR_VENDOR="nvidia"
        if ! nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,driver_version \
            --format=csv > "${output%.csv}_identity.csv" 2>/dev/null; then
            rm -f "${output%.csv}_identity.csv"
            echo "[GPU Monitor] Warning: NVIDIA identity sidecar failed" >&2
        fi
        nvidia-smi --query-gpu="$NVIDIA_GPU_MONITOR_QUERY" \
            --format=csv -l "$interval" > "$output" 2>/dev/null &
        GPU_MONITOR_PID=$!
        echo "[GPU Monitor] Started NVIDIA (PID=$GPU_MONITOR_PID, interval=${interval}s, output=$output)"
    elif command -v amd-smi &>/dev/null; then
        GPU_MONITOR_VENDOR="amd"
        # amd-smi is Python and block-buffers stdout; without PYTHONUNBUFFERED the
        # trailing ticks were lost at kill (measured on MI355X). awk keeps the first
        # CSV header, drops repeated ones, and flushes every row for the same reason.
        PYTHONUNBUFFERED=1 amd-smi metric -p -c -t -u -w "$interval" --csv 2>/dev/null \
            | awk '/^timestamp,/{if(!h){print;h=1};next} h{print;fflush()}' > "$output" &
        GPU_MONITOR_PID=$!
        # Hardware energy-accumulator + identity snapshots; the end-side twin in
        # stop_gpu_monitor lets auditors cross-check the integrated energy
        # against the accumulator delta.
        _write_amd_smi_sidecar "${output%.csv}_energy_start.csv" metric -E --csv
        _write_amd_smi_sidecar "${output%.csv}_identity.json" static --json
        echo "[GPU Monitor] Started AMD (PID=$GPU_MONITOR_PID, interval=${interval}s, output=$output)"
    else
        GPU_MONITOR_VENDOR=""
        echo "[GPU Monitor] No GPU monitoring tool found (nvidia-smi or amd-smi), skipping"
        return 0
    fi
}

stop_gpu_monitor() {
    if [[ -n "$GPU_MONITOR_PID" ]] && kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then
        # The stream must cover one sample past benchmark_end_time_unix for
        # boundary interpolation. NVIDIA appends a one-shot sample below; amd-smi
        # one-shot CSV has no timestamp column, so the AMD watch stream must emit
        # final ticks before the kill. Two extra intervals because amd-smi stamps
        # integer seconds: a tick in the same second as the window end still
        # fails bracketing (MI355X: end=...153.325 vs last sample ...153.0).
        if [[ "$GPU_MONITOR_VENDOR" == "amd" ]]; then
            sleep $(( ${GPU_MONITOR_INTERVAL} + 2 ))
        fi
        kill "$GPU_MONITOR_PID" 2>/dev/null
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
        case "$GPU_MONITOR_VENDOR" in
            nvidia)
                if _repair_truncated_gpu_metrics_tail; then
                    nvidia-smi --query-gpu="$NVIDIA_GPU_MONITOR_QUERY" \
                        --format=csv,noheader >> "$GPU_METRICS_CSV" 2>/dev/null ||
                        echo "[GPU Monitor] Warning: final NVIDIA sample failed" >&2
                fi
                ;;
            amd)
                _repair_truncated_gpu_metrics_tail || true
                _write_amd_smi_sidecar "${GPU_METRICS_CSV%.csv}_energy_end.csv" metric -E --csv
                ;;
        esac
        echo "[GPU Monitor] Stopped (PID=$GPU_MONITOR_PID)"
        if [[ -f "$GPU_METRICS_CSV" ]]; then
            local lines
            lines=$(wc -l < "$GPU_METRICS_CSV")
            echo "[GPU Monitor] Collected $lines rows -> $GPU_METRICS_CSV"
        fi
    fi
    GPU_MONITOR_PID=""
    GPU_MONITOR_VENDOR=""
}

# Drop a partial trailing row left behind when the monitor dies mid-write.
# Returns non-zero when a truncated row was detected but could not be removed.
_repair_truncated_gpu_metrics_tail() {
    local repaired_metrics="${GPU_METRICS_CSV}.repair.$$"
    if [[ -s "$GPU_METRICS_CSV" ]] &&
        ! tail -c 1 "$GPU_METRICS_CSV" | grep -q '^$'; then
        if sed '$d' "$GPU_METRICS_CSV" > "$repaired_metrics" &&
            mv "$repaired_metrics" "$GPU_METRICS_CSV"; then
            echo "[GPU Monitor] Dropped truncated trailing sample"
        else
            rm -f "$repaired_metrics"
            echo "[GPU Monitor] Warning: could not repair truncated trailing sample" >&2
            return 1
        fi
    fi
    return 0
}

# Write one best-effort amd-smi snapshot; remove the file rather than keep a
# partial one when the invocation fails.
_write_amd_smi_sidecar() {
    local out="$1"
    shift
    if ! amd-smi "$@" > "$out" 2>/dev/null; then
        rm -f "$out"
        echo "[GPU Monitor] Warning: amd-smi $1 sidecar failed" >&2
    fi
}

# Poll rocm-smi VRAM% every 10s for up to 15 min until the busiest GPU is at or
# below the threshold percent (default 10); return 1 otherwise so the caller
# aborts instead of starting on GPUs still draining the previous job.
# Pass a stricter threshold when the run sizes its KV cache from device-wide free
# memory (torch.cuda.mem_get_info): on 288 GB parts the 10% gate admits ~28.8 GB
# of residual, which the engine folds into non_torch and subtracts from the KV
# pool, so the pool drifts run to run.
wait_for_amd_gpu_clean() {
    local threshold="${1:-10}"
    local gpu_clean=false vram_max i
    for i in $(seq 1 90); do
        vram_max=$(rocm-smi --showmemuse 2>/dev/null \
            | grep -oE "GPU Memory Allocated \(VRAM%\): [0-9]+" \
            | awk '{if ($NF > m) m = $NF} END {print m+0}')
        if [ "${vram_max:-0}" -le "$threshold" ]; then
            echo "GPUs clean (vram%max=$vram_max <= $threshold after $((i * 10))s)"
            gpu_clean=true
            break
        fi
        echo "waiting for prior-job GPU memory reclaim: vram%max=$vram_max (target <= $threshold)"
        sleep 10
    done
    if [ "$gpu_clean" != "true" ]; then
        echo "Error: GPUs still draining prior job's memory after 15min" >&2
        return 1
    fi
}

# Return success only while a PID exists and is not a zombie waiting to be
# reaped. `kill -0` alone treats zombies as live processes.
_background_process_is_running() {
    local pid="$1"
    local state
    kill -0 "$pid" 2>/dev/null || return 1
    state=$(ps -o stat= -p "$pid" 2>/dev/null) || return 1
    [[ -n "$state" && "${state:0:1}" != "Z" ]]
}

_background_process_descendants() {
    local parent_pid="$1"
    local child_pid
    while read -r child_pid; do
        [[ -n "$child_pid" ]] || continue
        echo "$child_pid"
        _background_process_descendants "$child_pid"
    done < <(pgrep -P "$parent_pid" 2>/dev/null || true)
}

# Stop a background service and every process that descended from it. Capture
# descendants before terminating the root because orphaned workers are
# reparented and can otherwise keep a Slurm step alive after the benchmark
# script exits.
stop_background_process_tree() {
    local root_pid="${1:-}"
    local label="${2:-background process}"
    local grace_seconds="${3:-30}"

    if [[ ! "$root_pid" =~ ^[1-9][0-9]*$ ]] || ! _background_process_is_running "$root_pid"; then
        return 0
    fi

    local descendants
    local child_pid
    descendants=$(_background_process_descendants "$root_pid")

    echo "Stopping $label (PID=$root_pid)..."
    kill -TERM "$root_pid" 2>/dev/null || true

    local deadline=$((SECONDS + grace_seconds))
    while _background_process_is_running "$root_pid" && [[ $SECONDS -lt $deadline ]]; do
        sleep 1
    done

    local forced_stop=false
    while read -r child_pid; do
        [[ -n "$child_pid" ]] || continue
        if _background_process_is_running "$child_pid"; then
            if [[ "$forced_stop" == "false" ]]; then
                echo "Force-stopping remaining $label processes."
                forced_stop=true
            fi
            echo "  PID=$child_pid"
            kill -KILL "$child_pid" 2>/dev/null || true
        fi
    done <<EOF
$root_pid
$descendants
EOF

    wait "$root_pid" 2>/dev/null || true
    echo "Stopped $label."
}


# Poll an HTTP endpoint while streaming the owning process log.
# Required: --endpoint, --log, --pid. A zero timeout waits indefinitely.
wait_for_ready() {
    set +x
    local endpoint=""
    local process_log=""
    local process_pid=""
    local sleep_interval=5
    local timeout=0

    while [[ $# -gt 0 ]]; do
        case $1 in
            --endpoint)
                endpoint="$2"
                shift 2
                ;;
            --log)
                process_log="$2"
                shift 2
                ;;
            --pid)
                process_pid="$2"
                shift 2
                ;;
            --sleep-interval)
                sleep_interval="$2"
                shift 2
                ;;
            --timeout)
                timeout="$2"
                shift 2
                ;;
            *)
                echo "Unknown parameter: $1"
                return 1
                ;;
        esac
    done

    if [[ -z "$endpoint" ]]; then
        echo "Error: --endpoint is required"
        return 1
    fi
    if [[ -z "$process_log" ]]; then
        echo "Error: --log is required"
        return 1
    fi
    if [[ -z "$process_pid" ]]; then
        echo "Error: --pid is required"
        return 1
    fi
    if [[ ! "$sleep_interval" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: --sleep-interval must be a positive integer"
        return 1
    fi
    if [[ ! "$timeout" =~ ^[0-9]+$ ]]; then
        echo "Error: --timeout must be a non-negative integer"
        return 1
    fi

    local deadline=0
    if [[ "$timeout" -gt 0 ]]; then
        deadline=$((SECONDS + timeout))
    fi

    while [[ ! -f "$process_log" ]]; do
        if ! kill -0 "$process_pid" 2>/dev/null; then
            echo "Process died before creating $process_log." >&2
            exit 1
        fi
        if [[ "$deadline" -gt 0 && "$SECONDS" -ge "$deadline" ]]; then
            echo "Timed out waiting for $endpoint." >&2
            exit 1
        fi
        sleep 1
    done

    tail -f -n +1 "$process_log" &
    local tail_pid=$!
    until curl --output /dev/null --silent --fail "$endpoint"; do
        if ! kill -0 "$process_pid" 2>/dev/null; then
            echo "Process died before $endpoint became ready." >&2
            kill "$tail_pid" 2>/dev/null || true
            exit 1
        fi
        if [[ "$deadline" -gt 0 && "$SECONDS" -ge "$deadline" ]]; then
            echo "Timed out waiting for $endpoint." >&2
            kill "$tail_pid" 2>/dev/null || true
            exit 1
        fi
        sleep "$sleep_interval"
    done
    kill "$tail_pid" 2>/dev/null || true
    wait "$tail_pid" 2>/dev/null || true
}

wait_for_server_ready() {
    local port=""
    local server_log=""
    local server_pid=""
    local sleep_interval=5

    while [[ $# -gt 0 ]]; do
        case $1 in
            --port) port="$2"; shift 2 ;;
            --server-log) server_log="$2"; shift 2 ;;
            --server-pid) server_pid="$2"; shift 2 ;;
            --sleep-interval) sleep_interval="$2"; shift 2 ;;
            *) echo "Unknown parameter: $1"; return 1 ;;
        esac
    done

    if [[ -z "$port" || -z "$server_log" || -z "$server_pid" ]]; then
        echo "Error: --port, --server-log, and --server-pid are required"
        return 1
    fi

    wait_for_ready \
        --endpoint "http://0.0.0.0:${port}/health" \
        --log "$server_log" \
        --pid "$server_pid" \
        --sleep-interval "$sleep_interval" || return $?
    INFERENCEX_SERVER_STATE=$(mktemp /tmp/inferencex-server-state.XXXXXX) || return 1
    PYTHONPATH="$INFERENCEX_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 -m infx.bench_serving.server_watch capture --pid "$server_pid" \
        > "$INFERENCEX_SERVER_STATE" || return 1
    INFERENCEX_SERVER_PID="$server_pid"
}

# Keep client process groups separate; on confirmed server/worker death only
# this client's descendants are stopped. There is no elapsed-time cutoff.
run_server_client() {
    if [[ -n "${INFERENCEX_SERVER_STATE:-}" ]]; then
        PYTHONPATH="$INFERENCEX_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 -m infx.bench_serving.server_watch run --state "$INFERENCEX_SERVER_STATE" -- "$@"
    else
        "$@"
    fi
}

# Persist an argv array in shell-replayable form.
write_command() {
    local output_file="$1"
    shift
    printf '%q ' "$@" | tee "$output_file"
    printf '\n' | tee -a "$output_file"
}

append_command() {
    local output_file="$1"
    shift
    printf '%q ' "$@" >> "$output_file"
    printf '\n' >> "$output_file"
}

# --dsv4 renders prompts with the DeepSeek-V4 template (encoding_dsv4.py) instead
# of the tokenizer's jinja template and implies --use-chat-template.
run_benchmark_serving() {
    if [ "${EVAL_ONLY}" = "true" ]; then
        echo "EVAL_ONLY mode: skipping throughput benchmark"
        return 0
    fi

    set +x
    local model=""
    local port=""
    local backend=""
    local endpoint=""
    local input_len=""
    local output_len=""
    local random_range_ratio=""
    local num_prompts=""
    local max_concurrency=""
    local result_filename=""
    local result_dir=""
    local workspace_dir=""
    local use_chat_template=false
    local dsv4=false
    local trust_remote_code=false
    local server_pid=""
    local tokenizer=""
    local tokenizer_mode=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            --model)
                model="$2"
                shift 2
                ;;
            --port)
                port="$2"
                shift 2
                ;;
            --backend)
                backend="$2"
                shift 2
                ;;
            --endpoint)
                endpoint="$2"
                shift 2
                ;;
            --input-len)
                input_len="$2"
                shift 2
                ;;
            --output-len)
                output_len="$2"
                shift 2
                ;;
            --random-range-ratio)
                random_range_ratio="$2"
                shift 2
                ;;
            --num-prompts)
                num_prompts="$2"
                shift 2
                ;;
            --max-concurrency)
                max_concurrency="$2"
                shift 2
                ;;
            --result-filename)
                result_filename="$2"
                shift 2
                ;;
            --result-dir)
                result_dir="$2"
                shift 2
                ;;
            --bench-serving-dir)
                workspace_dir="$2"
                shift 2
                ;;
            --use-chat-template)
                use_chat_template=true
                shift
                ;;
            --dsv4)
                dsv4=true
                use_chat_template=true
                shift
                ;;
            --trust-remote-code)
                trust_remote_code=true
                shift
                ;;
            --server-pid)
                server_pid="$2"
                shift 2
                ;;
            --tokenizer)
                tokenizer="$2"
                shift 2
                ;;
            --tokenizer-mode)
                tokenizer_mode="$2"
                shift 2
                ;;
            *)
                echo "Unknown parameter: $1"
                return 1
                ;;
        esac
    done
    
    if [[ -z "$model" ]]; then
        echo "Error: --model is required"
        return 1
    fi
    if [[ -z "$port" ]]; then
        echo "Error: --port is required"
        return 1
    fi
    if [[ -z "$backend" ]]; then
        echo "Error: --backend is required"
        return 1
    fi
    if [[ -z "$input_len" ]]; then
        echo "Error: --input-len is required"
        return 1
    fi
    if [[ -z "$output_len" ]]; then
        echo "Error: --output-len is required"
        return 1
    fi
    if [[ -z "$random_range_ratio" ]]; then
        echo "Error: --random-range-ratio is required"
        return 1
    fi
    if [[ -z "$num_prompts" ]]; then
        echo "Error: --num-prompts is required"
        return 1
    fi
    if [[ -z "$max_concurrency" ]]; then
        echo "Error: --max-concurrency is required"
        return 1
    fi
    if [[ -z "$result_filename" ]]; then
        echo "Error: --result-filename is required"
        return 1
    fi
    if [[ -z "$result_dir" ]]; then
        echo "Error: --result-dir is required"
        return 1
    fi

    if [[ -z "$workspace_dir" ]]; then
        workspace_dir=$(pwd)
    fi

    # PROFILE=1 caps num_prompts at max_concurrency to keep traces small.
    local profile_flag=()
    if [[ "${PROFILE:-}" == "1" ]]; then
        local _prof_dir="${SGLANG_TORCH_PROFILER_DIR:-${VLLM_TORCH_PROFILER_DIR:-}}"
        if [[ -n "$_prof_dir" ]]; then
            mkdir -p "$_prof_dir"
        fi
        profile_flag+=(--profile)
        num_prompts="$max_concurrency"
    fi

    local benchmark_cmd=(
        env PYTHONPATH="$workspace_dir${PYTHONPATH:+:$PYTHONPATH}"
        python3 -m infx.bench_serving.benchmark_serving
        --model "$model"
        --backend "$backend"
        --base-url "http://0.0.0.0:$port"
        --dataset-name random
        --random-input-len "$input_len"
        --random-output-len "$output_len"
        --random-range-ratio "$random_range_ratio"
        --num-prompts "$num_prompts"
        --max-concurrency "$max_concurrency"
        --request-rate inf
        --ignore-eos
        "${profile_flag[@]}"
        --save-result
        --num-warmups "$((2 * max_concurrency))" \
        --percentile-metrics 'ttft,tpot,itl,e2el'
        --result-dir "$result_dir"
        --result-filename "$result_filename.json"
    )

    if [[ -n "$endpoint" ]]; then
        benchmark_cmd+=(--endpoint "$endpoint")
    fi
    
    if [[ "$use_chat_template" == true ]]; then
        benchmark_cmd+=(--use-chat-template)
    fi

    if [[ "$dsv4" == true ]]; then
        benchmark_cmd+=(--dsv4)
    fi

    if [[ "$trust_remote_code" == true ]]; then
        benchmark_cmd+=(--trust-remote-code)
    fi

    if [[ -n "$tokenizer" ]]; then
        benchmark_cmd+=(--tokenizer "$tokenizer")
    fi

    if [[ -n "$tokenizer_mode" ]]; then
        benchmark_cmd+=(--tokenizer-mode "$tokenizer_mode")
    fi

    if [[ -n "$server_pid" && "$server_pid" != "${INFERENCEX_SERVER_PID:-}" ]]; then
        INFERENCEX_SERVER_STATE=$(mktemp /tmp/inferencex-server-state.XXXXXX) || return 1
        PYTHONPATH="$INFERENCEX_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 -m infx.bench_serving.server_watch capture --pid "$server_pid" \
            > "$INFERENCEX_SERVER_STATE" || return 1
        INFERENCEX_SERVER_PID="$server_pid"
    fi
    local benchmark_exit_code=0
    set -x
    run_server_client "${benchmark_cmd[@]}" || benchmark_exit_code=$?
    set +x

    if [[ "${PROFILE:-}" == "1" ]]; then
        move_profile_trace_for_relay
    fi

    return $benchmark_exit_code
}


# Profiling trace helpers

_find_latest_profile_trace() {
    local latest=""
    local dir="" candidate="" base=""
    local -a search_roots=()

    for dir in "$@"; do
        search_roots=()
        if [[ -d "$dir" ]]; then
            search_roots+=("$dir")
        fi
        if [[ -d "$dir/profiles" ]]; then
            search_roots+=("$dir/profiles")
        fi
        if [[ ${#search_roots[@]} -eq 0 ]]; then
            continue
        fi

        while IFS= read -r -d '' candidate; do
            base="$(basename "$candidate")"
            if [[ "$base" == profile_*.trace.json.gz ]]; then
                continue
            fi
            if [[ -z "$latest" || "$candidate" -nt "$latest" ]]; then
                latest="$candidate"
            fi
        done < <(
            find "${search_roots[@]}" -maxdepth 1 -type f \
                \( -name "*.trace.json" -o -name "*.trace.json.gz" -o -name "*trace*.json" -o -name "*trace*.json.gz" -o -name "*profile*.json" -o -name "*profile*.json.gz" \) \
                -print0 2>/dev/null
        )
    done

    printf '%s' "$latest"
}

# Move profiler trace into a stable workspace path for workflow relay/upload.
move_profile_trace_for_relay() {
    if [[ "${PROFILE:-}" != "1" ]]; then
        return 0
    fi

    if [[ -z "${RESULT_FILENAME:-}" ]]; then
        echo "[PROFILE] RESULT_FILENAME is not set; skipping relay trace staging." >&2
        return 0
    fi

    check_env_vars SGLANG_TORCH_PROFILER_DIR VLLM_TORCH_PROFILER_DIR
    local sglang_dir="${SGLANG_TORCH_PROFILER_DIR}"
    local vllm_dir="${VLLM_TORCH_PROFILER_DIR}"
    local -a search_dirs=()
    local dir="" existing=""
    local seen=0

    for dir in "$sglang_dir" "$vllm_dir" "/workspace"; do
        if [[ -z "$dir" ]]; then
            continue
        fi
        seen=0
        for existing in "${search_dirs[@]}"; do
            if [[ "$existing" == "$dir" ]]; then
                seen=1
                break
            fi
        done
        if [[ "$seen" -eq 0 ]]; then
            search_dirs+=("$dir")
        fi
    done

    local trace_file=""
    local wait_attempts=10
    for (( i=1; i<=wait_attempts; i++ )); do
        trace_file="$(_find_latest_profile_trace "${search_dirs[@]}")"
        if [[ -n "$trace_file" ]]; then
            break
        fi
        sleep 10
    done

    if [[ -z "$trace_file" ]]; then
        echo "[PROFILE] No trace found for relay under: ${search_dirs[*]}" >&2
        return 0
    fi

    local dest_trace="/workspace/profile_${RESULT_FILENAME}.trace.json.gz"
    if [[ "$trace_file" == *.gz ]]; then
        cp -f "$trace_file" "$dest_trace"
    else
        gzip -c "$trace_file" > "$dest_trace"
    fi

    echo "[PROFILE] Relay trace prepared: $dest_trace (source: $trace_file)"
}


# Eval (lm-eval-harness) helpers

_install_lm_eval_deps() {
    # torchvision causes circular imports in ATOM; TRT-LLM/SGLang need it at module level.
    if [[ "${IMAGE:-}" == *atom* ]]; then
        python3 -m pip uninstall -y torchvision 2>/dev/null || true
    fi
    python3 -m pip install -q --no-cache-dir --break-system-packages "lm-eval[api]" || true
    local lm_eval_ref="b315ef3b05176acc9732bb7fdec116abe1ecc476"
    if command -v git >/dev/null 2>&1; then
        if ! python3 -m pip install -q --no-cache-dir --no-deps --force-reinstall --break-system-packages \
            "git+https://github.com/EleutherAI/lm-evaluation-harness.git@${lm_eval_ref}"; then
            python3 -m pip install -q --no-cache-dir --no-deps --force-reinstall --break-system-packages \
                "https://github.com/EleutherAI/lm-evaluation-harness/archive/${lm_eval_ref}.tar.gz" || true
        fi
    else
        python3 -m pip install -q --no-cache-dir --no-deps --force-reinstall --break-system-packages \
            "https://github.com/EleutherAI/lm-evaluation-harness/archive/${lm_eval_ref}.tar.gz" || true
    fi
}

_prepare_vendor_verifier_python() {
    local verifier_name="$1"
    local runtime_prefix="$2"
    local use_system_site_packages="${3:-false}"
    local minimum_python_minor="${4:-12}"

    VENDOR_VERIFIER_PYTHON=python3
    VENDOR_VERIFIER_PYTHON_CLEANUP_DIR=""
    export VENDOR_VERIFIER_PYTHON VENDOR_VERIFIER_PYTHON_CLEANUP_DIR

    local system_python_is_compatible=false
    if python3 -c \
        'import sys; raise SystemExit(sys.version_info < (3, int(sys.argv[1])))' \
        "$minimum_python_minor"; then
        system_python_is_compatible=true
        if [ "$use_system_site_packages" != "true" ]; then
            return 0
        fi
    fi

    local python_dir uv_prefix uv_bin venv_dir prepare_rc=0
    python_dir="$(mktemp -d "/tmp/${runtime_prefix}-XXXXXX")" || {
        echo "ERROR: could not create a temporary Python directory for ${verifier_name}" >&2
        return 1
    }
    VENDOR_VERIFIER_PYTHON_CLEANUP_DIR="$python_dir"
    export VENDOR_VERIFIER_PYTHON_CLEANUP_DIR

    venv_dir="${python_dir}/venv"
    if [ "$system_python_is_compatible" = "true" ]; then
        python3 -m venv --system-site-packages "$venv_dir" || prepare_rc=$?
    else
        uv_prefix="${python_dir}/uv"
        uv_bin="${uv_prefix}/bin/uv"
        python3 -m pip install -q --no-cache-dir --break-system-packages \
            --prefix "$uv_prefix" "uv==0.11.33" || prepare_rc=$?
        if [ "$prepare_rc" -eq 0 ] && [ ! -x "$uv_bin" ]; then
            echo "ERROR: pinned uv installation did not create ${uv_bin}" >&2
            prepare_rc=1
        fi
        if [ "$prepare_rc" -eq 0 ]; then
            local system_site_packages_args=()
            if [ "$use_system_site_packages" = "true" ]; then
                system_site_packages_args+=(--system-site-packages)
            fi
            UV_CACHE_DIR="${python_dir}/uv-cache" \
            UV_PYTHON_INSTALL_DIR="${python_dir}/python" \
                "$uv_bin" venv --python "3.${minimum_python_minor}" --seed \
                    "${system_site_packages_args[@]}" "$venv_dir" \
                || prepare_rc=$?
        fi
    fi
    if [ "$prepare_rc" -eq 0 ] && [ ! -x "${venv_dir}/bin/python" ]; then
        echo "ERROR: pinned Python setup did not create the ${verifier_name} interpreter" >&2
        prepare_rc=1
    fi
    if [ "$prepare_rc" -ne 0 ]; then
        rm -rf "$python_dir" || true
        VENDOR_VERIFIER_PYTHON=python3
        VENDOR_VERIFIER_PYTHON_CLEANUP_DIR=""
        export VENDOR_VERIFIER_PYTHON VENDOR_VERIFIER_PYTHON_CLEANUP_DIR
        return "$prepare_rc"
    fi

    VENDOR_VERIFIER_PYTHON="${venv_dir}/bin/python"
    export VENDOR_VERIFIER_PYTHON
}

_install_kimi_vendor_eval_deps() {
    check_env_vars VENDOR_VERIFIER_PYTHON
    local target_dir="$1"
    local eval_suite="${2:-kimi_tool_call_schema}"
    local -a packages=(
        "httpx[http2]==0.28.1"
        "openai==2.14.0"
        "jsonschema==4.25.1"
        "pytest==8.4.2"
    )
    if [ "$eval_suite" = "kimi_tool_call_schema_full" ]; then
        packages+=("pytest-xdist==3.8.0")
    fi
    "${VENDOR_VERIFIER_PYTHON}" -m pip install -q --no-cache-dir \
        --target "$target_dir" "${packages[@]}"
}

_prepare_kimi_vendor_runtime() {
    local eval_suite="${1:-kimi_tool_call_schema}"
    local runtime_dir install_rc=0
    runtime_dir="$(mktemp -d /tmp/kimi-vendor-runtime-XXXXXX)" || return $?
    _install_kimi_vendor_eval_deps "$runtime_dir" "$eval_suite" >&2 || install_rc=$?
    if [ "$install_rc" -ne 0 ]; then
        rm -rf "$runtime_dir"
        return "$install_rc"
    fi
    printf '%s\n' "$runtime_dir"
}

_prepare_kimi_vendor_verifier() {
    check_env_vars VENDOR_VERIFIER_PYTHON
    local repo_url="$1"
    local verifier_ref="$2"
    local expected_archive_sha256="$3"
    local checkout_dir prepare_rc=0

    checkout_dir="$(mktemp -d /tmp/kimi-vendor-verifier-XXXXXX)" || {
        echo "ERROR: could not create a temporary directory for Kimi-Vendor-Verifier" >&2
        return 1
    }

    "${VENDOR_VERIFIER_PYTHON}" - \
        "$repo_url" "$verifier_ref" "$expected_archive_sha256" "$checkout_dir" \
        < "${INFERENCEX_REPO_ROOT}/infx/evals/_kimi_verifier_archive.py" || prepare_rc=$?

    if [ "$prepare_rc" -ne 0 ]; then
        if ! rm -rf "$checkout_dir"; then
            echo "ERROR: failed to remove partial Kimi-Vendor-Verifier directory ${checkout_dir}" >&2
        fi
        return "$prepare_rc"
    fi

    printf '%s\n' "$checkout_dir"
}

_cleanup_vendor_eval() {
    local path
    for path in "$@"; do
        [ -z "$path" ] || rm -rf "$path" || true
    done
}

_has_eval_result() {
    local results_dir="$1"
    local filename_prefix="$2"
    local matches=("${results_dir}/${filename_prefix}"*.json)
    [ -f "${matches[0]}" ]
}

_prepare_eval_artifact_family() {
    local results_dir="$1"
    local family="$2"
    local artifact rm_rc=0
    local artifacts=()

    export EVAL_RESULT_DIR=""
    case "$family" in
        kimi)
            artifacts=(
                "${results_dir}"/results_kimi_vendor_*.json
                "${results_dir}/kimi_vendor_report.json"
            )
            ;;
        minimax)
            artifacts=(
                "${results_dir}"/results_minimax_vendor_*.json
                "${results_dir}/minimax_vendor_report.json"
                "${results_dir}/minimax_vendor_results.jsonl"
            )
            ;;
        bfcl)
            artifacts=(
                "${results_dir}"/results_bfcl*.json
                "${results_dir}/bfcl_report.json"
                "${results_dir}/bfcl_upstream_artifacts.tar.gz"
            )
            ;;
        *)
            echo "ERROR: unsupported eval artifact family '${family}'" >&2
            return 2
            ;;
    esac

    for artifact in "${artifacts[@]}"; do
        if [ -e "$artifact" ] || [ -L "$artifact" ]; then
            rm -f -- "$artifact" || rm_rc=$?
            if [ "$rm_rc" -ne 0 ]; then
                echo "ERROR: failed to remove stale eval artifact ${artifact}" >&2
                return "$rm_rc"
            fi
        fi
    done
    export EVAL_RESULT_DIR="$results_dir"
}

_write_kimi_vendor_integration_error() {
    check_env_vars VENDOR_VERIFIER_PYTHON
    local adapter_path="$1"
    local model_name="$2"
    local results_dir="$3"
    local task_name="$4"
    local message="$5"

    "${VENDOR_VERIFIER_PYTHON}" "$adapter_path" \
        --model "$model_name" \
        --output-dir "$results_dir" \
        --task-name "$task_name" \
        --integration-error "$message"
}

_run_kimi_tool_call_schema_eval() {
    check_env_vars PORT
    local port="${PORT}"
    local results_dir="${EVAL_RESULT_DIR:-$(mktemp -d /tmp/eval_out-XXXXXX)}"
    local verifier_repo="https://github.com/MoonshotAI/Kimi-Vendor-Verifier.git"
    local verifier_ref="3dad65a760a8867cda72f6dd8848d876a4e851b4"
    local verifier_archive_sha256="ede9ea300c72ccfde9d8975ea4b1b54e423c7625690f6631ab1e65a715821e01"
    local eval_suite="${EVAL_SUITE:-kimi_tool_call_schema}"
    local timeout_seconds=900
    if [ "$eval_suite" = "kimi_tool_call_schema_full" ]; then
        timeout_seconds=7200
    fi

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --port|--results-dir)
                if [[ $# -lt 2 || -z "${2:-}" || "${2:-}" == --* ]]; then
                    echo "ERROR: $1 requires a value" >&2
                    return 2
                fi
                case "$1" in
                    --port)        port="$2" ;;
                    --results-dir) results_dir="$2" ;;
                esac
                shift 2
                ;;
            *)
                echo "Unknown parameter: $1" >&2
                return 2
                ;;
        esac
    done


    local model_name="${MODEL_NAME:-${MODEL:-}}"
    local adapter_path="${INFERENCEX_REPO_ROOT}/infx/evals/kimi_vendor_eval.py"
    local runtime_dir=""
    local checkout_dir=""

    mkdir -p "$results_dir" || return $?
    results_dir="$(cd "$results_dir" && pwd)" || return $?
    _prepare_eval_artifact_family "$results_dir" kimi || return $?

    local setup_rc=0 integration_error=""
    _prepare_vendor_verifier_python "Kimi-Vendor-Verifier" "kimi-vendor-python" || {
        setup_rc=$?
        integration_error="Kimi Vendor Verifier Python runtime preparation failed with exit code ${setup_rc}"
    }
    if [ "$setup_rc" -eq 0 ]; then
        runtime_dir=$(_prepare_kimi_vendor_runtime "$eval_suite") || {
            setup_rc=$?
            integration_error="Kimi Vendor Verifier dependency installation failed with exit code ${setup_rc}"
        }
    fi
    if [ "$setup_rc" -eq 0 ]; then
        checkout_dir=$(
            _prepare_kimi_vendor_verifier \
                "$verifier_repo" "$verifier_ref" "$verifier_archive_sha256"
        ) || {
            setup_rc=$?
            integration_error="Kimi Vendor Verifier checkout failed with exit code ${setup_rc}"
        }
    fi
    if [ "$setup_rc" -ne 0 ]; then
        echo "ERROR: ${integration_error}" >&2
        local artifact_rc=0
        _write_kimi_vendor_integration_error \
            "$adapter_path" "$model_name" "$results_dir" "$eval_suite" \
            "$integration_error" || artifact_rc=$?
        if [ "$artifact_rc" -ne 0 ]; then
            echo "ERROR: failed to write Kimi verifier failure artifact (exit code ${artifact_rc})" >&2
        fi
        _cleanup_vendor_eval \
            "$runtime_dir" "$checkout_dir" "${VENDOR_VERIFIER_PYTHON_CLEANUP_DIR:-}"
        return "$setup_rc"
    fi

    local eval_rc=0
    PYTHONPATH="${runtime_dir}${PYTHONPATH:+:${PYTHONPATH}}" \
        run_server_client "${VENDOR_VERIFIER_PYTHON}" "$adapter_path" \
            --verifier-dir "$checkout_dir" \
            --base-url "http://127.0.0.1:${port}/v1" \
            --api-key EMPTY \
            --model "$model_name" \
            --model-prefix "${MODEL_PREFIX:-}" \
            --output-dir "$results_dir" \
            --task-name "$eval_suite" \
            --timeout-seconds "$timeout_seconds" \
            || eval_rc=$?
    if [ "$eval_rc" -ne 0 ] \
        && ! _has_eval_result "$results_dir" "results_kimi_vendor_"; then
        integration_error="Kimi Vendor Verifier failed with exit code ${eval_rc}"
        local artifact_rc=0
        _write_kimi_vendor_integration_error \
            "$adapter_path" "$model_name" "$results_dir" "$eval_suite" \
            "$integration_error" || artifact_rc=$?
        if [ "$artifact_rc" -ne 0 ]; then
            echo "ERROR: failed to write Kimi verifier failure artifact (exit code ${artifact_rc})" >&2
        fi
    fi
    _cleanup_vendor_eval \
        "$runtime_dir" "$checkout_dir" "${VENDOR_VERIFIER_PYTHON_CLEANUP_DIR:-}"
    return "$eval_rc"
}

run_kimi_vendor_eval() {
    local eval_suite="${EVAL_SUITE:-kimi_tool_call_schema}"
    export EVAL_SUITE="$eval_suite"

    case "$eval_suite" in
        kimi_tool_call_schema|kimi_tool_call_schema_full)
            _run_kimi_tool_call_schema_eval "$@"
            ;;
        *)
            echo "ERROR: unsupported Kimi Vendor Verifier suite '${eval_suite}'" >&2
            export EVAL_RESULT_DIR=""
            return 2
            ;;
    esac
}

_install_bfcl_eval_deps() {
    check_env_vars VENDOR_VERIFIER_PYTHON
    local download_dir="$1"
    local wheel_url="https://files.pythonhosted.org/packages/ba/41/ed458527c770c50225b60bae3b0c3444b26804ee455fa2d8f187018d2cb2/bfcl_eval-2026.3.23-py3-none-any.whl"
    local wheel_sha256="3bb6dfa5f0c68ad403c9ec50b00db2bb3b4cc9b38ab1ff33f48fe30d853d3a0a"
    local wheel_path="${download_dir}/bfcl_eval-2026.3.23-py3-none-any.whl"

    "${VENDOR_VERIFIER_PYTHON}" - \
        "$wheel_url" "$wheel_sha256" "$wheel_path" <<'PY' || return $?
from hashlib import sha256
from pathlib import Path
import sys
from urllib.request import Request, urlopen

wheel_url, expected_sha256, wheel_path_arg = sys.argv[1:]
wheel_path = Path(wheel_path_arg)
digest = sha256()
downloaded = 0
request = Request(
    wheel_url,
    headers={"User-Agent": "InferenceX-BFCL-Smoke"},
)

try:
    with urlopen(request, timeout=180) as response, wheel_path.open("xb") as output:
        while chunk := response.read(1024 * 1024):
            downloaded += len(chunk)
            if downloaded > 512 * 1024 * 1024:
                raise ValueError("BFCL wheel exceeds the 512 MiB safety limit")
            digest.update(chunk)
            output.write(chunk)
    if downloaded == 0:
        raise ValueError("downloaded BFCL wheel is empty")
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"BFCL wheel SHA256 mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
except Exception as error:
    wheel_path.unlink(missing_ok=True)
    print(f"ERROR: failed to download and verify the pinned BFCL wheel: {error}", file=sys.stderr)
    raise SystemExit(1)
PY

    timeout 600 "${VENDOR_VERIFIER_PYTHON}" -m pip install \
        -q --no-cache-dir "$wheel_path" "soundfile==0.13.1"
}

_prepare_bfcl_runtime() {
    local runtime_dir install_rc=0
    runtime_dir="$(mktemp -d /tmp/bfcl-runtime-XXXXXX)" || return $?
    _install_bfcl_eval_deps "$runtime_dir" >&2 || install_rc=$?
    if [ "$install_rc" -ne 0 ]; then
        rm -rf "$runtime_dir"
        return "$install_rc"
    fi
    printf '%s\n' "$runtime_dir"
}

_archive_bfcl_upstream_artifacts() {
    check_env_vars VENDOR_VERIFIER_PYTHON
    local project_root="$1"
    local archive_path="$2"

    "${VENDOR_VERIFIER_PYTHON}" - "$project_root" "$archive_path" <<'PY'
import gzip
import os
from pathlib import Path
import tarfile
import sys

project_root = Path(sys.argv[1])
archive_path = Path(sys.argv[2])
temporary_path = archive_path.with_name(f".{archive_path.name}.tmp")
temporary_path.unlink(missing_ok=True)

try:
    with (
        temporary_path.open("xb") as raw_archive,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw_archive, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive,
    ):
        for path in sorted(
            project_root.rglob("*"),
            key=lambda candidate: candidate.relative_to(project_root).as_posix(),
        ):
            relative_path = path.relative_to(project_root).as_posix()
            if path.is_symlink():
                raise ValueError(f"refusing to archive symbolic link: {relative_path}")
            info = archive.gettarinfo(str(path), arcname=relative_path)
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            if info.isdir():
                archive.addfile(info)
            elif info.isfile():
                with path.open("rb") as source:
                    archive.addfile(info, source)
            else:
                raise ValueError(f"refusing to archive special file: {relative_path}")
    os.replace(temporary_path, archive_path)
except BaseException:
    temporary_path.unlink(missing_ok=True)
    raise
PY
}

_write_bfcl_integration_error() {
    check_env_vars VENDOR_VERIFIER_PYTHON
    local adapter_path="$1"
    local model_name="$2"
    local results_dir="$3"
    local message="$4"
    local suite="$5"
    local adapter_rc=0

    # Integration errors deliberately make the adapter exit nonzero after
    # publishing both score artifacts. Treat those artifacts, not that expected
    # status, as proof that failure reporting succeeded.
    "${VENDOR_VERIFIER_PYTHON}" "$adapter_path" \
        --model "$model_name" \
        --output-dir "$results_dir" \
        --suite "$suite" \
        --integration-error "$message" \
        || adapter_rc=$?
    if [ -f "${results_dir}/bfcl_report.json" ] \
        && [ -f "${results_dir}/results_bfcl.json" ]; then
        return 0
    fi
    if [ "$adapter_rc" -eq 0 ]; then
        return 1
    fi
    return "$adapter_rc"
}

_run_bfcl_suite_eval() {
    check_env_vars PORT
    local eval_suite="$1"
    local num_threads="$2"
    local process_timeout_seconds="$3"
    local archive_upstream="$4"
    shift 4

    local port="${PORT}"
    local results_dir="${EVAL_RESULT_DIR:-}"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --port|--results-dir)
                if [[ $# -lt 2 || -z "${2:-}" || "${2:-}" == --* ]]; then
                    echo "ERROR: $1 requires a value" >&2
                    return 2
                fi
                case "$1" in
                    --port)        port="$2" ;;
                    --results-dir) results_dir="$2" ;;
                esac
                shift 2
                ;;
            *)
                echo "Unknown parameter: $1" >&2
                return 2
                ;;
        esac
    done

    if [ -z "$results_dir" ]; then
        results_dir="$(mktemp -d /tmp/eval_out-XXXXXX)" || return $?
    fi

    local model_name="${MODEL_NAME:-${MODEL:-}}"
    local adapter_path="${INFERENCEX_REPO_ROOT}/infx/evals/bfcl_adapter.py"
    local runtime_dir=""
    local project_root=""

    mkdir -p "$results_dir" || return $?
    results_dir="$(cd "$results_dir" && pwd)" || return $?
    _prepare_eval_artifact_family "$results_dir" bfcl || return $?

    local setup_rc=0 integration_error=""
    _prepare_vendor_verifier_python "BFCL" "bfcl-python" true 10 || {
        setup_rc=$?
        integration_error="BFCL Python runtime preparation failed with exit code ${setup_rc}"
    }
    if [ "$setup_rc" -eq 0 ]; then
        runtime_dir=$(_prepare_bfcl_runtime) || {
            setup_rc=$?
            integration_error="BFCL dependency installation failed with exit code ${setup_rc}"
        }
    fi
    if [ "$setup_rc" -eq 0 ]; then
        project_root="$(mktemp -d /tmp/bfcl-project-root-XXXXXX)" || {
            setup_rc=$?
            integration_error="BFCL project root preparation failed with exit code ${setup_rc}"
        }
    fi
    if [ "$setup_rc" -ne 0 ]; then
        echo "ERROR: ${integration_error}" >&2
        local artifact_rc=0
        _write_bfcl_integration_error \
            "$adapter_path" "$model_name" "$results_dir" "$integration_error" \
            "$eval_suite" || artifact_rc=$?
        if [ "$artifact_rc" -ne 0 ]; then
            echo "ERROR: failed to write BFCL failure artifact (exit code ${artifact_rc})" >&2
        fi
        _cleanup_vendor_eval \
            "$runtime_dir" "$project_root" "${VENDOR_VERIFIER_PYTHON_CLEANUP_DIR:-}"
        return "$setup_rc"
    fi

    local eval_rc=0
    local -a suite_args=()
    if [ "$eval_suite" != "bfcl_smoke" ]; then
        suite_args=(--suite "$eval_suite")
    fi
    run_server_client timeout "$process_timeout_seconds" \
        "${VENDOR_VERIFIER_PYTHON}" "$adapter_path" \
        --base-url "http://127.0.0.1:${port}/v1" \
        --api-key EMPTY \
        --model "$model_name" \
        --output-dir "$results_dir" \
        --bfcl-project-root "$project_root" \
        "${suite_args[@]}" \
        --num-threads "$num_threads" \
        || eval_rc=$?
    local archive_rc=0
    if [ "$archive_upstream" = true ]; then
        _archive_bfcl_upstream_artifacts \
            "$project_root" "${results_dir}/bfcl_upstream_artifacts.tar.gz" \
            || archive_rc=$?
        if [ "$archive_rc" -ne 0 ]; then
            echo "ERROR: failed to archive BFCL upstream artifacts (exit code ${archive_rc})" >&2
        fi
    fi
    if [ "$eval_rc" -ne 0 ] \
        && { [ ! -f "${results_dir}/bfcl_report.json" ] \
            || [ ! -f "${results_dir}/results_bfcl.json" ]; }; then
        local integration_error="BFCL evaluation failed with exit code ${eval_rc}"
        local artifact_rc=0
        _write_bfcl_integration_error \
            "$adapter_path" "$model_name" "$results_dir" "$integration_error" \
            "$eval_suite" || artifact_rc=$?
        if [ "$artifact_rc" -ne 0 ]; then
            echo "ERROR: failed to write BFCL failure artifact (exit code ${artifact_rc})" >&2
        fi
    fi
    _cleanup_vendor_eval \
        "$runtime_dir" "$project_root" "${VENDOR_VERIFIER_PYTHON_CLEANUP_DIR:-}"
    if [ "$eval_rc" -ne 0 ]; then
        return "$eval_rc"
    fi
    return "$archive_rc"
}


_run_bfcl_smoke_eval() {
    _run_bfcl_suite_eval bfcl_smoke 4 900 false "$@"
}

run_bfcl_eval() {
    local eval_suite="${EVAL_SUITE:-bfcl_smoke}"
    export EVAL_SUITE="$eval_suite"

    case "$eval_suite" in
        bfcl_smoke)
            _run_bfcl_smoke_eval "$@"
            ;;
        bfcl_vllm_minimax_m3)
            _run_bfcl_suite_eval "$eval_suite" 8 7200 true "$@"
            ;;
        bfcl_vllm_kimi)
            _run_bfcl_suite_eval "$eval_suite" 16 14400 true "$@"
            ;;
        *)
            echo "ERROR: unsupported BFCL suite '${eval_suite}'" >&2
            export EVAL_RESULT_DIR=""
            return 2
            ;;
    esac
}


_write_minimax_vendor_integration_error() {
    check_env_vars VENDOR_VERIFIER_PYTHON
    local adapter_path="$1"
    local model_name="$2"
    local results_dir="$3"
    local message="$4"

    # The failure path is stdlib-only, so it remains usable when runtime
    # provisioning or dependency installation is what failed.
    "${VENDOR_VERIFIER_PYTHON}" "$adapter_path" failure \
        --model "$model_name" \
        --output-dir "$results_dir" \
        --message "$message"
}

_run_minimax_m3_smoke_eval() {
    check_env_vars PORT
    local port="${PORT}"
    local results_dir="${EVAL_RESULT_DIR:-}"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --port|--results-dir)
                if [[ $# -lt 2 || -z "${2:-}" || "${2:-}" == --* ]]; then
                    echo "ERROR: $1 requires a value" >&2
                    return 2
                fi
                case "$1" in
                    --port)        port="$2" ;;
                    --results-dir) results_dir="$2" ;;
                esac
                shift 2
                ;;
            *)
                echo "Unknown parameter: $1" >&2
                return 2
                ;;
        esac
    done

    if [ -z "$results_dir" ]; then
        results_dir="$(mktemp -d /tmp/eval_out-XXXXXX)" || return $?
    fi

    local model_name="${MODEL_NAME:-${MODEL:-}}"
    local adapter_path="${INFERENCEX_REPO_ROOT}/infx/evals/minimax_provider_eval.py"
    local fixture_path="${INFERENCEX_REPO_ROOT}/infx/evals/minimax_m3_smoke.json"
    local runtime_dir=""

    mkdir -p "$results_dir" || return $?
    results_dir="$(cd "$results_dir" && pwd)" || return $?
    _prepare_eval_artifact_family "$results_dir" minimax || return $?

    local setup_rc=0 integration_error=""
    _prepare_vendor_verifier_python "MiniMax Provider Verifier" "minimax-vendor-python" || {
        setup_rc=$?
        integration_error="MiniMax Provider Verifier Python runtime preparation failed with exit code ${setup_rc}"
    }
    if [ "$setup_rc" -eq 0 ]; then
        runtime_dir=$(_prepare_minimax_m3_full_runtime) || {
            setup_rc=$?
            integration_error="MiniMax Provider Verifier pinned runtime preparation failed with exit code ${setup_rc}"
        }
    fi
    if [ "$setup_rc" -ne 0 ]; then
        echo "ERROR: ${integration_error}" >&2
        local artifact_rc=0
        _write_minimax_vendor_integration_error \
            "$adapter_path" "$model_name" "$results_dir" "$integration_error" \
            || artifact_rc=$?
        if [ "$artifact_rc" -ne 0 ]; then
            echo "ERROR: failed to write MiniMax verifier failure artifact (exit code ${artifact_rc})" >&2
        fi
        _cleanup_vendor_eval \
            "$runtime_dir" "${VENDOR_VERIFIER_PYTHON_CLEANUP_DIR:-}"
        return "$setup_rc"
    fi

    local eval_rc=0
    run_server_client "${VENDOR_VERIFIER_PYTHON}" "$adapter_path" run \
        --python "${VENDOR_VERIFIER_PYTHON}" \
        --source-dir "${runtime_dir}/source" \
        --dependency-dir "${runtime_dir}/deps" \
        --base-url "http://127.0.0.1:${port}/v1" \
        --model "$model_name" \
        --output-dir "$results_dir" \
        --fixture "$fixture_path" \
        || eval_rc=$?
    if [ "$eval_rc" -ne 0 ] \
        && ! _has_eval_result "$results_dir" "results_minimax_vendor_"; then
        integration_error="MiniMax Provider Verifier failed with exit code ${eval_rc}"
        local artifact_rc=0
        _write_minimax_vendor_integration_error \
            "$adapter_path" "$model_name" "$results_dir" "$integration_error" \
            || artifact_rc=$?
        if [ "$artifact_rc" -ne 0 ]; then
            echo "ERROR: failed to write MiniMax verifier failure artifact (exit code ${artifact_rc})" >&2
        fi
    fi
    _cleanup_vendor_eval \
        "$runtime_dir" "${VENDOR_VERIFIER_PYTHON_CLEANUP_DIR:-}"
    return "$eval_rc"
}

_install_minimax_m3_full_deps() {
    check_env_vars VENDOR_VERIFIER_PYTHON
    local target_dir="$1"
    "${VENDOR_VERIFIER_PYTHON}" -m pip install -q --no-cache-dir --target "$target_dir" \
        "jsonschema==4.25.1" \
        "loguru==0.7.3" \
        "megfile==4.2.5" \
        "numpy==2.3.4" \
        "openai==2.7.1" \
        "tqdm==4.67.1"
}

_prepare_minimax_m3_full_runtime() {
    check_env_vars VENDOR_VERIFIER_PYTHON
    local source_adapter_path="${INFERENCEX_REPO_ROOT}/infx/evals/minimax_m3_full_eval.py"
    local runtime_dir prepare_rc=0
    runtime_dir="$(mktemp -d /tmp/minimax-m3-full-runtime-XXXXXX)" || return $?
    "${VENDOR_VERIFIER_PYTHON}" "$source_adapter_path" prepare-source \
        --source-dir "${runtime_dir}/source" >&2 || prepare_rc=$?
    if [ "$prepare_rc" -eq 0 ]; then
        _install_minimax_m3_full_deps "${runtime_dir}/deps" >&2 || prepare_rc=$?
    fi
    if [ "$prepare_rc" -ne 0 ]; then
        rm -rf "$runtime_dir"
        return "$prepare_rc"
    fi
    printf '%s\n' "$runtime_dir"
}

_run_minimax_m3_full_eval() {
    check_env_vars PORT
    local port="${PORT}"
    local results_dir="${EVAL_RESULT_DIR:-}"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --port|--results-dir)
                if [[ $# -lt 2 || -z "${2:-}" || "${2:-}" == --* ]]; then
                    echo "ERROR: $1 requires a value" >&2
                    return 2
                fi
                case "$1" in
                    --port)        port="$2" ;;
                    --results-dir) results_dir="$2" ;;
                esac
                shift 2
                ;;
            *)
                echo "Unknown parameter: $1" >&2
                return 2
                ;;
        esac
    done

    if [ -z "$results_dir" ]; then
        results_dir="$(mktemp -d /tmp/eval_out-XXXXXX)" || return $?
    fi

    local model_name="${MODEL_NAME:-${MODEL:-}}"
    local adapter_path="${INFERENCEX_REPO_ROOT}/infx/evals/minimax_m3_full_eval.py"
    local runtime_dir=""

    mkdir -p "$results_dir" || return $?
    results_dir="$(cd "$results_dir" && pwd)" || return $?
    _prepare_eval_artifact_family "$results_dir" minimax || return $?

    local setup_rc=0 integration_error=""
    _prepare_vendor_verifier_python "MiniMax M3 full verifier" "minimax-m3-full-python" || {
        setup_rc=$?
        integration_error="MiniMax M3 full Python runtime preparation failed with exit code ${setup_rc}"
    }
    if [ "$setup_rc" -eq 0 ]; then
        runtime_dir=$(_prepare_minimax_m3_full_runtime) || {
            setup_rc=$?
            integration_error="MiniMax M3 full pinned runtime preparation failed with exit code ${setup_rc}"
        }
    fi
    if [ "$setup_rc" -ne 0 ]; then
        echo "ERROR: ${integration_error}" >&2
        local artifact_rc=0
        _write_minimax_vendor_integration_error \
            "$adapter_path" "$model_name" "$results_dir" "$integration_error" \
            || artifact_rc=$?
        if [ "$artifact_rc" -ne 0 ]; then
            echo "ERROR: failed to write MiniMax full verifier failure artifact (exit code ${artifact_rc})" >&2
        fi
        _cleanup_vendor_eval \
            "$runtime_dir" "${VENDOR_VERIFIER_PYTHON_CLEANUP_DIR:-}"
        return "$setup_rc"
    fi

    local eval_rc=0
    run_server_client "${VENDOR_VERIFIER_PYTHON}" "$adapter_path" run \
        --python "${VENDOR_VERIFIER_PYTHON}" \
        --source-dir "${runtime_dir}/source" \
        --dependency-dir "${runtime_dir}/deps" \
        --base-url "http://127.0.0.1:${port}/v1" \
        --model "$model_name" \
        --output-dir "$results_dir" \
        || eval_rc=$?
    if [ "$eval_rc" -ne 0 ] \
        && ! _has_eval_result "$results_dir" "results_minimax_vendor_full_"; then
        integration_error="MiniMax M3 full verifier failed with exit code ${eval_rc}"
        local artifact_rc=0
        _write_minimax_vendor_integration_error \
            "$adapter_path" "$model_name" "$results_dir" "$integration_error" \
            || artifact_rc=$?
        if [ "$artifact_rc" -ne 0 ]; then
            echo "ERROR: failed to write MiniMax full verifier failure artifact (exit code ${artifact_rc})" >&2
        fi
    fi
    _cleanup_vendor_eval \
        "$runtime_dir" "${VENDOR_VERIFIER_PYTHON_CLEANUP_DIR:-}"
    return "$eval_rc"
}


run_minimax_vendor_eval() {
    local eval_suite="${EVAL_SUITE:-minimax_m3_smoke}"
    export EVAL_SUITE="$eval_suite"

    case "$eval_suite" in
        minimax_m3_smoke)
            _run_minimax_m3_smoke_eval "$@"
            ;;
        minimax_m3_full)
            _run_minimax_m3_full_eval "$@"
            ;;
        *)
            echo "ERROR: unsupported MiniMax Provider Verifier suite '${eval_suite}'" >&2
            export EVAL_RESULT_DIR=""
            return 2
            ;;
    esac
}

_eval_patches_dir() {
    printf '%s\n' "${INFERENCEX_REPO_ROOT}/infx/evals/patches"
}

_patch_lm_eval() {
    local patch_dir
    patch_dir="$(mktemp -d)"
    cp "$(_eval_patches_dir)/lm_eval_sitecustomize.py" "$patch_dir/sitecustomize.py"
    export PYTHONPATH="${patch_dir}${PYTHONPATH:+:${PYTHONPATH}}"
}

get_native_max_context_length() {
    local model_path="$1"
    # Prefer MODEL_PATH (local model directory) when available, since the
    # argument may be a served-model name that is neither a valid HF repo
    # ID nor a local path (e.g. "deepseek-r1-fp4" on the B300 cluster).
    if [ -n "${MODEL_PATH:-}" ] && [ -d "${MODEL_PATH}" ]; then
        model_path="${MODEL_PATH}"
    fi
    python3 -c "
try:
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained('${model_path}', trust_remote_code=True)
    for attr in ['max_position_embeddings', 'max_sequence_length', 'seq_length', 'n_positions']:
        if hasattr(config, attr):
            print(getattr(config, attr))
            break
    else:
        print(0)
except Exception:
    print(0)
"
}

# Requested benchmark context capped at the model's native max. Sets
# EVAL_MAX_MODEL_LEN (read by run_lm_eval) and echoes the value.
compute_eval_context_length() {
    local model="$1"
    local benchmark_ctx="${2:-0}"
    local native_max
    native_max=$(get_native_max_context_length "$model")
    native_max="${native_max:-0}"

    if [ "$benchmark_ctx" -eq 0 ] 2>/dev/null; then
        benchmark_ctx="${native_max:-0}"
    fi
    local eval_ctx=$(( benchmark_ctx * 1 ))
    if [ "$native_max" -gt 0 ] 2>/dev/null && [ "$eval_ctx" -gt "$native_max" ]; then
        eval_ctx="$native_max"
    fi
    if [ "$eval_ctx" -le 0 ] 2>/dev/null; then
        echo "WARN: compute_eval_context_length could not determine context length for $model" >&2
        eval_ctx="${MAX_MODEL_LEN:-16384}"
    fi
    EVAL_MAX_MODEL_LEN="$eval_ctx"
    echo "$eval_ctx"
}

# Call directly, not in a subshell, so the EVAL_MAX_MODEL_LEN export persists.
setup_eval_context() {
    EVAL_MAX_MODEL_LEN=$(compute_eval_context_length "$MODEL" "$((ISL + OSL + 256))")
    export EVAL_MAX_MODEL_LEN
}

run_lm_eval() {
    check_env_vars OPENAI_API_KEY PORT
    local port="${PORT}"
    local tasks_dir="${EVAL_TASKS_DIR:-infx/evals/gsm8k.yaml}"
    local results_dir="${EVAL_RESULT_DIR:-$(mktemp -d /tmp/eval_out-XXXXXX)}"
    local eval_context_len="${EVAL_MAX_MODEL_LEN}"
    local temperature=0
    local top_p=1
    local concurrent_requests="${EVAL_CONCURRENT_REQUESTS:-${CONC}}"
    check_env_vars concurrent_requests
    # SWE-bench adds a repo-local task YAML, hence --include_path. --limit is
    # passed only when EVAL_LIMIT requests a smoke-test slice.
    local eval_limit="${EVAL_LIMIT:-}"
    local include_path="${EVAL_INCLUDE_PATH:-}"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --port|--task|--results-dir|--gen-max-tokens|--temperature|--top-p)
                if [[ $# -lt 2 || -z "${2:-}" || "${2:-}" == --* ]]; then
                    echo "ERROR: $1 requires a value" >&2
                    return 2
                fi
                case "$1" in
                    --port)           port="$2" ;;
                    --task)           tasks_dir="$2" ;;
                    --results-dir)    results_dir="$2" ;;
                    --gen-max-tokens) eval_context_len="$2" ;;
                    --temperature)    temperature="$2" ;;
                    --top-p)          top_p="$2" ;;
                esac
                shift 2
                ;;
            *)
                echo "Unknown parameter: $1" >&2
                return 2
                ;;
        esac
    done

    check_env_vars eval_context_len

    # Serving images may use a different WORKDIR.
    local _repo_root="$INFERENCEX_REPO_ROOT"
    if [[ "$tasks_dir" == *.yaml && "$tasks_dir" != /* \
          && ! -f "$tasks_dir" && -f "$_repo_root/$tasks_dir" ]]; then
        echo "run_lm_eval: anchoring relative task '$tasks_dir' to repo root -> $_repo_root/$tasks_dir"
        tasks_dir="$_repo_root/$tasks_dir"
    fi

    export EVAL_TASKS_DIR="$tasks_dir"

    if [ "${INFERENCEX_LM_EVAL_RUNTIME_READY:-false}" != "true" ]; then
        _install_lm_eval_deps
        _patch_lm_eval
        export INFERENCEX_LM_EVAL_RUNTIME_READY=true
    fi

    local openai_server_base="http://0.0.0.0:${port}"
    local openai_chat_base="${openai_server_base}/v1/chat/completions"
    export OPENAI_API_KEY=${OPENAI_API_KEY}
    MODEL_NAME=${MODEL_NAME:-$MODEL} # Prefer MODEL_NAME, else MODEL

    # Leave room for input within the context window and avoid excessive
    # per-request KV cache reservation on TRT.
    local max_output_tokens=$(( eval_context_len > 4096 ? eval_context_len - 4096 : eval_context_len / 2 ))
    if [ "$max_output_tokens" -gt 16384 ]; then
        max_output_tokens=16384
    fi
    echo "Eval budget: eval_context_len=${eval_context_len}, max_output_tokens=${max_output_tokens}"

    # Read by append_lm_eval_summary.
    export EVAL_RESULT_DIR="$results_dir"
    set -x
    run_server_client python3 -m lm_eval --model local-chat-completions --apply_chat_template \
      ${include_path:+--include_path "$include_path"} \
      --tasks "${tasks_dir}" \
      --output_path "${results_dir}" \
      --log_samples \
      --model_args "model=${MODEL_NAME},base_url=${openai_chat_base},api_key=${OPENAI_API_KEY},eos_string=</s>,max_retries=5,num_concurrent=${concurrent_requests},timeout=1800,tokenized_requests=False,max_length=${eval_context_len}" \
      --gen_kwargs "max_tokens=${max_output_tokens},temperature=${temperature},top_p=${top_p}" \
      ${eval_limit:+--limit "$eval_limit"}
    local eval_exit=$?
    set +x
    return $eval_exit
}

_stage_lm_eval_artifacts() {
    local results_dir="$1"
    local eval_conc="$2"
    local moved=0
    local failed=0
    local jf base stem extension target suffix

    if [ ! -d "${results_dir}" ]; then
        echo "WARN: eval result directory '${results_dir}' does not exist" >&2
        return 1
    fi

    while IFS= read -r -d '' jf; do
        base=$(basename "$jf")
        case "$base" in
            meta_env.json)
                continue
                ;;
            *.jsonl)
                stem="${base%.jsonl}"
                extension=".jsonl"
                ;;
            *.json)
                stem="${base%.json}"
                extension=".json"
                ;;
            *)
                continue
                ;;
        esac

        target="./${stem}_conc${eval_conc}${extension}"
        suffix=2
        while [ -e "$target" ]; do
            target="./${stem}_conc${eval_conc}_${suffix}${extension}"
            suffix=$((suffix + 1))
        done

        if mv -f "$jf" "$target"; then
            moved=1
        else
            echo "WARN: failed to stage eval artifact ${jf}" >&2
            failed=1
        fi
    done < <(
        find "${results_dir}" -type f \
            \( -name "*.json" -o -name "*.jsonl" \) -print0 2>/dev/null
    )

    rm -rf --one-file-system "${results_dir}" 2>/dev/null \
        || rm -rf "${results_dir}" \
        || true

    if [ "$moved" -eq 0 ]; then
        echo "WARN: no eval artifacts were produced for concurrency ${eval_conc}" >&2
        return 1
    fi
    return "$failed"
}

_eval_concs_to_json() {
    local values="$1"
    local value
    local joined=""

    for value in $values; do
        if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
            echo "ERROR: invalid eval concurrency '${value}'" >&2
            return 1
        fi
        if [ -n "$joined" ]; then
            joined="${joined}, "
        fi
        joined="${joined}${value}"
    done

    printf '[%s]' "$joined"
}

_env_is_true() {
    case "${1:-}" in
        1|[Tt][Rr][Uu][Ee]|[Yy][Ee][Ss]|[Oo][Nn]) return 0 ;;
        *) return 1 ;;
    esac
}

_resolve_disagg_ep() {
    local ep="${1:-1}"
    local enable_flag="${2:-false}"
    local tp_size="${3:-1}"
    if [[ "$ep" == "1" ]] && _env_is_true "$enable_flag"; then
        echo "$tp_size"
    else
        echo "$ep"
    fi
}

_normalize_bool_json() {
    if _env_is_true "${1:-false}"; then
        echo "true"
    else
        echo "false"
    fi
}

# Export TP/EP/DP metadata for append_lm_eval_summary / meta_env.json.
# Prefer workflow PREFILL_EP/DECODE_EP and *_DP_ATTN (from job.slurm) over
# ENABLE_* launch booleans so DEP8/DPA arms record the correct topology.
bridge_disagg_eval_metadata() {
    export TP="${PREFILL_TP:-${PREFILL_TP_SIZE:-${TP:-1}}}"
    export PREFILL_TP="${PREFILL_TP:-${PREFILL_TP_SIZE:-${TP:-1}}}"
    export PREFILL_EP="$(_resolve_disagg_ep "${PREFILL_EP:-${EP_SIZE:-${EP:-1}}}" "${PREFILL_ENABLE_EP:-false}" "${PREFILL_TP_SIZE:-${PREFILL_TP:-1}}")"
    export EP_SIZE="${PREFILL_EP}"
    export PREFILL_NUM_WORKERS="${PREFILL_NUM_WORKERS:-${xP:-1}}"
    export DECODE_TP="${DECODE_TP:-${DECODE_TP_SIZE:-${TP:-1}}}"
    export DECODE_EP="$(_resolve_disagg_ep "${DECODE_EP:-${EP_SIZE:-${EP:-1}}}" "${DECODE_ENABLE_EP:-false}" "${DECODE_TP_SIZE:-${DECODE_TP:-1}}")"
    export DECODE_NUM_WORKERS="${DECODE_NUM_WORKERS:-${yD:-1}}"

    local prefill_dp="${PREFILL_DP_ATTN:-${PREFILL_DP_ATTENTION:-${PREFILL_ENABLE_DP:-false}}}"
    local decode_dp="${DECODE_DP_ATTN:-${DECODE_DP_ATTENTION:-${DECODE_ENABLE_DP:-false}}}"
    export DP_ATTENTION="$(_normalize_bool_json "$prefill_dp")"
    export PREFILL_DP_ATTENTION="$(_normalize_bool_json "$prefill_dp")"
    export DECODE_DP_ATTENTION="$(_normalize_bool_json "$decode_dp")"
}

_write_lm_eval_meta_json() {
    check_env_vars IS_MULTINODE
    local meta_json="$1"
    local batch_metadata="${2:-}"
    local metadata_conc="${3:-${CONC:-1}}"

    # Single-node jobs already export TP/EP/DP_ATTENTION. The disaggregated
    # bridge defaults missing per-phase DP flags to false, so applying it to
    # single-node jobs would overwrite their actual DP-attention setting.
    if [ "${IS_MULTINODE}" = "true" ]; then
        bridge_disagg_eval_metadata
    fi

    local model_name="${MODEL_NAME:-$MODEL}"
    local is_multinode_json="false"
    if [ "${IS_MULTINODE}" = "true" ]; then
        is_multinode_json="true"
    fi

    local prefill_tp="${PREFILL_TP:-${TP:-1}}"
    local prefill_pp="${PREFILL_PP_SIZE:-${PP_SIZE:-1}}"
    local prefill_dcp_size="${PREFILL_DCP_SIZE:-${DCP_SIZE:-1}}"
    local prefill_pcp_size="${PREFILL_PCP_SIZE:-${PCP_SIZE:-1}}"
    local prefill_ep="${PREFILL_EP:-${EP_SIZE:-1}}"
    local prefill_num_workers="${PREFILL_NUM_WORKERS:-1}"
    local decode_tp="${DECODE_TP:-${TP:-1}}"
    local decode_pp="${DECODE_PP_SIZE:-${PP_SIZE:-1}}"
    local decode_dcp_size="${DECODE_DCP_SIZE:-${DCP_SIZE:-1}}"
    local decode_pcp_size="${DECODE_PCP_SIZE:-${PCP_SIZE:-1}}"
    local decode_ep="${DECODE_EP:-${EP_SIZE:-1}}"
    local decode_num_workers="${DECODE_NUM_WORKERS:-1}"

    local dp_json
    dp_json="$(_normalize_bool_json "${DP_ATTENTION:-false}")"
    local prefill_dp_json
    prefill_dp_json="$(_normalize_bool_json "${PREFILL_DP_ATTENTION:-${DP_ATTENTION:-false}}")"
    local decode_dp_json
    decode_dp_json="$(_normalize_bool_json "${DECODE_DP_ATTENTION:-${DP_ATTENTION:-false}}")"

    local fw="${FRAMEWORK:-}"
    local prec="${PRECISION:-}"
    if [[ -z "$fw" || -z "$prec" ]]; then
        if [[ -n "${RESULT_FILENAME:-}" ]]; then
            local parsed
            parsed=$(echo "${RESULT_FILENAME}" | sed -n 's/.*_\([^_][^_]*\)_\([^_][^_]*\)_tp.*/\1 \2/p')
            local p1="${parsed%% *}"
            local p2="${parsed#* }"
            if [[ -z "$prec" && -n "$p1" && "$p1" != "$parsed" ]]; then
                prec="$p1"
            fi
            if [[ -z "$fw" && -n "$p2" && "$p2" != "$parsed" ]]; then
                fw="$p2"
            fi
        fi
    fi
    local eval_suite="${EVAL_COMPLETED_SUITE:-${EVAL_SUITE:-}}"
    if [ -z "$eval_suite" ] && [ -n "${EVAL_TASKS_DIR:-}" ]; then
        eval_suite="$(basename "${EVAL_TASKS_DIR}")"
        eval_suite="${eval_suite%.yaml}"
        eval_suite="${eval_suite%.yml}"
    fi
    eval_suite="${eval_suite:-gsm8k}"

    cat > "${meta_json}" <<META
{
  "is_multinode": ${is_multinode_json},
  "framework": "${fw:-unknown}",
  "precision": "${prec:-unknown}",
  "spec_decoding": "${SPEC_DECODING:-}",
  "eval_suite": "${eval_suite}",
  "recipe_fingerprint": "${RECIPE_FINGERPRINT:-}",
  "tp": ${TP:-1},
  "pp": ${PP_SIZE:-1},
  "dcp_size": ${DCP_SIZE:-1},
  "pcp_size": ${PCP_SIZE:-1},
  "conc": ${metadata_conc},
${batch_metadata}  "ep": ${EP_SIZE:-1},
  "dp_attention": ${dp_json},
  "prefill_tp": ${prefill_tp},
  "prefill_pp": ${prefill_pp},
  "prefill_dcp_size": ${prefill_dcp_size},
  "prefill_pcp_size": ${prefill_pcp_size},
  "prefill_ep": ${prefill_ep},
  "prefill_dp_attention": ${prefill_dp_json},
  "prefill_num_workers": ${prefill_num_workers},
  "decode_tp": ${decode_tp},
  "decode_pp": ${decode_pp},
  "decode_dcp_size": ${decode_dcp_size},
  "decode_pcp_size": ${decode_pcp_size},
  "decode_ep": ${decode_ep},
  "decode_dp_attention": ${decode_dp_json},
  "decode_num_workers": ${decode_num_workers},
  "model": "${model_name:-}",
  "infmax_model_prefix": "${MODEL_PREFIX:-unknown}",
  "hw": "${RUNNER_TYPE:-unknown}",
  "isl": "${ISL:-0}",
  "osl": "${OSL:-0}"
}
META
}

rewrite_lm_eval_meta_env() {
    if [ -n "${EVAL_BATCHED_CONCS:-}" ]; then
        append_lm_eval_summary
    else
        _write_lm_eval_meta_json "./meta_env.json" "" "${CONC:-1}"
    fi
}

append_lm_eval_summary() {
    local batch_concs="${EVAL_BATCHED_CONCS:-}"
    local results_dir="${EVAL_RESULT_DIR:-}"
    local out_dir="${results_dir}"
    local meta_json
    local metadata_conc="${CONC:-1}"
    local batch_metadata=""

    if [ -n "$batch_concs" ]; then
        meta_json="./meta_env.json"
        metadata_conc="${batch_concs%% *}"

        local eval_concs_json completed_concs_json failed_concs_json
        eval_concs_json=$(_eval_concs_to_json "$batch_concs") || return 1
        completed_concs_json=$(
            _eval_concs_to_json "${EVAL_BATCHED_COMPLETED_CONCS:-}"
        ) || return 1
        failed_concs_json=$(
            _eval_concs_to_json "${EVAL_BATCHED_FAILED_CONCS:-}"
        ) || return 1
        printf -v batch_metadata \
            '  "eval_concs": %s,\n  "completed_eval_concs": %s,\n  "failed_eval_concs": %s,\n' \
            "$eval_concs_json" \
            "$completed_concs_json" \
            "$failed_concs_json"
    else
        if [ -z "${results_dir}" ]; then
            echo "WARN: EVAL_RESULT_DIR is empty; skipping artifact collection" >&2
            return 1
        fi
        if [ ! -d "${out_dir}" ]; then
            echo "WARN: EVAL_RESULT_DIR='${out_dir}' does not exist; skipping artifact collection" >&2
            return 1
        fi
        meta_json="${out_dir}/meta_env.json"
    fi

    _write_lm_eval_meta_json "$meta_json" "$batch_metadata" "$metadata_conc"

    if [ -n "$batch_concs" ]; then
        echo "Prepared batched eval artifacts in: $(pwd)"
        return 0
    fi

    stage_eval_artifacts "$(pwd)" "$out_dir" || return $?

    if [ -n "${out_dir}" ] && [ -d "${out_dir}" ]; then
        rm -rf --one-file-system "${out_dir}" || rm -rf "${out_dir}" || true
    fi

    echo "Staged eval artifacts in: $(pwd)"
}

stage_eval_artifacts() {
    local destination="$1"
    shift

    mkdir -p "$destination" || return $?
    local source_dir artifact
    local copied=0
    local artifacts=()
    for source_dir in "$@"; do
        [ -d "$source_dir" ] || continue
        artifacts=(
            "$source_dir"/meta_env.json
            "$source_dir"/results*.json
            "$source_dir"/*_report.json
            "$source_dir"/*_results.jsonl
            "$source_dir"/*_artifacts.tar.gz
            "$source_dir"/sample*.jsonl
            "$source_dir"/agent_preds.json
            "$source_dir"/swebench_report_*.json
            "$source_dir"/predictions.jsonl
            "$source_dir"/*.traj*
        )
        for artifact in "${artifacts[@]}"; do
            [ -f "$artifact" ] || continue
            cp -f "$artifact" "$destination/" || return $?
            copied=$((copied + 1))
        done
    done
    if [ "$copied" -eq 0 ]; then
        echo "ERROR: no eval artifacts found to stage" >&2
        return 1
    fi
}


_install_swebench_agent_deps() {
    python3 -m pip install -q --no-cache-dir --break-system-packages \
        'mini-swe-agent==2.4.5' 'swe-rex[modal]==1.4.0' || true
    _patch_swebench_agent || \
        echo "WARN: mini-swe-agent/swe-rex patches failed; sandbox cleanup, submission fallback, or non-interactive stdin handling may be degraded" >&2
}

_patch_swebench_agent() {
    python3 "$(_eval_patches_dir)/patch_swebench_agent.py"
}

_install_swebench_deps() {
    check_env_vars SWEBENCH_USE_MODAL
    # Patch anchors depend on SWE-bench 4.1.0.
    python3 -m pip install -q --no-cache-dir --break-system-packages 'swebench==4.1.0' || true
    if [ "${SWEBENCH_USE_MODAL}" = "true" ]; then
        python3 -m pip install -q --no-cache-dir --break-system-packages modal || true
        _patch_swebench_scoring || \
            echo "WARN: scoring patches failed; eval sandboxes will reserve 4 CPUs and idle-bill to their timeout" >&2
    fi
}

_patch_swebench_scoring() {
    python3 "$(_eval_patches_dir)/patch_swebench_scoring.py"
}

# SWE-bench requires ~/.modal.toml despite env credentials.
_ensure_modal_credentials() {
    check_env_vars IS_AGENTIC SWEBENCH_USE_MODAL
    # Agentic generation uses swerex_modal sandboxes even when scoring is local.
    if [ "${SWEBENCH_USE_MODAL}" != "true" ] \
        && [ "${IS_AGENTIC}" != "1" ] \
        && [ "${SCENARIO_TYPE:-}" != "agentic-coding" ]; then
        return 0
    fi
    # CI secrets may include whitespace or quotes.
    if [ -n "${MODAL_TOKEN_ID:-}" ]; then
        MODAL_TOKEN_ID=$(printf %s "$MODAL_TOKEN_ID" | tr -d "[:space:]\"'")
        export MODAL_TOKEN_ID
    fi
    if [ -n "${MODAL_TOKEN_SECRET:-}" ]; then
        MODAL_TOKEN_SECRET=$(printf %s "$MODAL_TOKEN_SECRET" | tr -d "[:space:]\"'")
        export MODAL_TOKEN_SECRET
    fi
    if [ -f "${HOME:-}/.modal.toml" ]; then return 0; fi
    if [ -n "${MODAL_TOKEN_ID:-}" ] && [ -n "${MODAL_TOKEN_SECRET:-}" ]; then
        # Slurm may provide an unwritable HOME.
        if [ -z "${HOME:-}" ] || ! mkdir -p "$HOME" 2>/dev/null || [ ! -w "$HOME" ]; then
            export HOME=/tmp/inferencex-modal-home
            mkdir -p "$HOME"
            echo "[swebench] HOME remapped to $HOME for Modal credentials (original path missing or not writable)"
        fi
        printf '[default]\ntoken_id = "%s"\ntoken_secret = "%s"\nactive = true\n' \
            "$MODAL_TOKEN_ID" "$MODAL_TOKEN_SECRET" > "$HOME/.modal.toml"
        chmod 600 "$HOME/.modal.toml"
        echo "[swebench] wrote ~/.modal.toml from MODAL_TOKEN_ID/MODAL_TOKEN_SECRET env"
    else
        echo "WARN: Modal credentials required but no ~/.modal.toml and no MODAL_TOKEN_ID/MODAL_TOKEN_SECRET env; Modal sandboxes will fail authentication" >&2
    fi
}


_run_swebench_agentic_generation() {
    check_env_vars \
        PORT SWEBENCH_AGENT_EXIT_GRACE SWEBENCH_AGENT_STEP_LIMIT SWEBENCH_AGENT_TIMEOUT \
        SWEBENCH_EXPECTED_INSTANCES SWEBENCH_SANDBOX_SWEEP SWEBENCH_WATCHDOG_POLL
    local gen_dir="$1"; shift
    local port="${PORT}"
    while [[ $# -gt 0 ]]; do
        case $1 in
            --port) port="$2"; shift 2 ;;
            *)      shift ;;
        esac
    done

    _install_swebench_agent_deps
    _ensure_modal_credentials

    # minisweagent logs before its config path.
    local default_cfg
    default_cfg=$(python3 -c 'import minisweagent, os; print(os.path.join(os.path.dirname(minisweagent.__file__), "config/benchmarks/swebench.yaml"))' 2>/dev/null | tail -n 1)
    if [ ! -f "$default_cfg" ]; then
        echo "ERROR: could not locate mini-swe-agent default swebench config (got: '${default_cfg}')" >&2
        return 1
    fi

    local cfg="$gen_dir/mini_swebench_overrides.yaml"
    SWEBENCH_AGENT_PORT="$port" python3 - "$default_cfg" "$cfg" <<'PYGEN'
import os, sys, yaml
default_path, out_path = sys.argv[1], sys.argv[2]
d = yaml.safe_load(open(default_path)) or {}
d.setdefault("agent", {})
step_limit = int(os.environ.get("SWEBENCH_AGENT_STEP_LIMIT", "250"))
guidance = f"""

<additional_critical_guidance>
- You have a hard budget of {step_limit} commands total. Plan: reproduce -> fix -> verify -> submit, finishing the submission well before the budget runs out. A correct fix that is never submitted scores ZERO.
- BEFORE submitting you MUST run the test(s) that cover the issue and confirm your fix makes them pass. Identify the failing test from the issue/PR, run it (e.g. `python -m pytest <path>::<test>` or `python tests/runtests.py <label>`), and check the result. Do not submit a patch you have not verified unless running tests is impossible.
- The scoring harness re-runs tests in its own clean environment. If the package fails to BUILD or IMPORT after 2-3 attempts, do NOT keep fixing the environment -- apply your source-code fix and submit it. A local build is not required for your patch to score.
- `git diff` alone is NOT a submission. Submitting requires the exact final command sequence described above.
- When unsure how code behaves, write and RUN a short script instead of reasoning about it at length in prose.
</additional_critical_guidance>"""
it = d["agent"].get("instance_template", "")
d["agent"]["instance_template"] = it.rstrip() + guidance + "\n"
d["agent"]["step_limit"] = step_limit
d["agent"]["cost_limit"] = 0.0
env = d.get("environment") or {}
env.update({
    "environment_class": "swerex_modal",
    # Modal cold starts exceed the default timeout.
    "startup_timeout": float(os.environ.get("SWEBENCH_AGENT_STARTUP_TIMEOUT", "900")),
    "timeout": int(os.environ.get("SWEBENCH_AGENT_CMD_TIMEOUT", "300")),
    # Limit billing if cleanup misses a sandbox.
    "runtime_timeout": float(os.environ.get("SWEBENCH_AGENT_RUNTIME_TIMEOUT", "3600")),
})
agent_cpu = os.environ.get("SWEBENCH_AGENT_SANDBOX_CPU", "")
if agent_cpu:
    env["modal_sandbox_kwargs"] = {"cpu": float(agent_cpu)}
d["environment"] = env
model_name = os.environ.get("MODEL_NAME") or os.environ.get("MODEL", "")
d["model"] = {
    "model_name": f"openai/{model_name}",
    "cost_tracking": "ignore_errors",
    "model_kwargs": {
        "api_base": f"http://0.0.0.0:{os.environ['SWEBENCH_AGENT_PORT']}/v1",
        "api_key": "dummy",
        "custom_llm_provider": "openai",
        "temperature": 0.0,
    },
}
yaml.safe_dump(d, open(out_path, "w"), default_flow_style=False, sort_keys=False)
PYGEN

    case "${EVAL_LIMIT:-}" in
        full|FULL|0) EVAL_LIMIT="" ;;
    esac
    if [ -n "${EVAL_LIMIT:-}" ] && [[ ! "$EVAL_LIMIT" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: EVAL_LIMIT='${EVAL_LIMIT}' must be a positive integer, 'full', or 0" >&2
        return 1
    fi
    local slice_args=()
    if [ -n "${EVAL_LIMIT:-}" ]; then
        slice_args=(--slice "0:${EVAL_LIMIT}")
    fi

    export MSWEA_COST_TRACKING=ignore_errors
    local expected="${EVAL_LIMIT:-${SWEBENCH_EXPECTED_INSTANCES}}"
    local workers="${SWEBENCH_AGENT_WORKERS:-${CONC}}"
    check_env_vars workers
    echo "[swebench-agentic] mini-swe-agent: workers=${workers} step_limit=${SWEBENCH_AGENT_STEP_LIMIT} slice=${EVAL_LIMIT:-full} expected=$expected"
    local agen_rc=0
    mini-extra swebench \
        -c "$cfg" \
        --subset lite --split test \
        --environment-class swerex_modal \
        "${slice_args[@]}" \
        -w "${workers}" \
        -o "$gen_dir/agent_out" &
    local mini_pid=$!
    # preds.json detects completion despite teardown hangs.
    local preds_file="$gen_dir/agent_out/preds.json"
    local deadline=$(( $(date +%s) + ${SWEBENCH_AGENT_TIMEOUT} ))
    local grace_until=0
    local killed_after_complete=0
    while kill -0 "$mini_pid" 2>/dev/null; do
        if [ "$(date +%s)" -ge "$deadline" ]; then
            echo "ERROR: generation exceeded SWEBENCH_AGENT_TIMEOUT (${SWEBENCH_AGENT_TIMEOUT}s); killing mini-extra" >&2
            kill "$mini_pid" 2>/dev/null; sleep 5; kill -9 "$mini_pid" 2>/dev/null
            agen_rc=124
            break
        fi
        local done_count
        done_count=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$preds_file" 2>/dev/null || echo 0)
        if [ "${done_count:-0}" -ge "$expected" ]; then
            if [ "$grace_until" -eq 0 ]; then
                grace_until=$(( $(date +%s) + ${SWEBENCH_AGENT_EXIT_GRACE} ))
                echo "[swebench-agentic] all $expected predictions written; waiting ${SWEBENCH_AGENT_EXIT_GRACE}s for mini-extra to exit"
            elif [ "$(date +%s)" -ge "$grace_until" ]; then
                echo "WARN: mini-extra hung after completing all instances; killing (known hang-on-exit)" >&2
                kill "$mini_pid" 2>/dev/null; sleep 5; kill -9 "$mini_pid" 2>/dev/null
                killed_after_complete=1
                break
            fi
        fi
        sleep "${SWEBENCH_WATCHDOG_POLL}"
    done
    wait "$mini_pid" 2>/dev/null
    local wait_rc=$?
    if [ "$killed_after_complete" -eq 1 ]; then
        agen_rc=0
    elif [ "$agen_rc" -eq 0 ] && [ "$wait_rc" -ne 0 ]; then
        agen_rc=$wait_rc
    fi
    # Isolate sweeps to avoid killing unrelated sandboxes.
    [ "${SWEBENCH_SANDBOX_SWEEP}" = "1" ] && python3 - <<'PYSWEEP' || true
try:
    import os
    import modal
    name = os.environ.get("SWEBENCH_MODAL_APP_NAME", "infx-evals-swe")
    app = modal.App.lookup(name)
    n = 0
    for sb in modal.Sandbox.list(app_id=app.app_id):
        try:
            sb.terminate()
            n += 1
        except Exception as e:
            print(f"[swebench-agentic] sweep: could not terminate {sb.object_id}: {e}")
    print(f"[swebench-agentic] sandbox sweep ({name}): terminated {n} lingering sandbox(es)")
except Exception as e:
    print(f"[swebench-agentic] sandbox sweep skipped: {e}")
PYSWEEP
    if [ "$agen_rc" -ne 0 ]; then
        # Partial runs may still be scoreable.
        local salvage_count
        salvage_count=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$gen_dir/agent_out/preds.json" 2>/dev/null || echo 0)
        if [ "${salvage_count:-0}" -gt 0 ]; then
            echo "WARN: generation exited rc=$agen_rc but $salvage_count/$expected predictions exist; scoring the partial set" >&2
        else
            echo "ERROR: agentic generation (mini-swe-agent) failed with $agen_rc" >&2
            return "$agen_rc"
        fi
    fi
    if [ ! -s "$gen_dir/agent_out/preds.json" ]; then
        echo "ERROR: agentic generation produced no preds.json" >&2
        return 1
    fi
}

run_swebench_eval() {
    check_env_vars \
        SWEBENCH_EVAL_TIMEOUT SWEBENCH_MAX_WORKERS SWEBENCH_SCORE_TIMEOUT SWEBENCH_SKIP_SCORE \
        SWEBENCH_USE_MODAL SWEBENCH_GEN_MODE
    local out_dir="${EVAL_RESULT_DIR:-$(mktemp -d /tmp/eval_out-XXXXXX)}"
    local task_name="${SWEBENCH_TASK_NAME:-swebench_lite}"
    export EVAL_SUITE="${EVAL_SUITE:-$task_name}"
    local gen_dir
    gen_dir=$(mktemp -d /tmp/swebench_gen-XXXXXX)

    # Generation and scoring must share a dataset.
    local yaml_path="${EVAL_TASKS_DIR:-infx/evals/${task_name}.yaml}"
    local dataset
    dataset=$(awk '/^dataset_path:[[:space:]]/{print $2; exit}' "$yaml_path" 2>/dev/null)
    if [ -z "$dataset" ]; then
        echo "ERROR: could not read dataset_path from ${yaml_path}" >&2
        rm -rf "$gen_dir" 2>/dev/null || true
        return 1
    fi
    if [ -n "${SWEBENCH_DATASET:-}" ] && [ "${SWEBENCH_DATASET}" != "$dataset" ]; then
        echo "ERROR: SWEBENCH_DATASET='${SWEBENCH_DATASET}' disagrees with ${yaml_path} dataset_path='${dataset}'." >&2
        echo "       Generation and scoring must use the same dataset; edit the YAML or unset SWEBENCH_DATASET." >&2
        rm -rf "$gen_dir" 2>/dev/null || true
        return 1
    fi

    local gen_mode="$SWEBENCH_GEN_MODE"
    local score_input=()
    if [ "$gen_mode" = "agentic" ]; then
        # mini-extra supports only SWE-bench Lite.
        case "$dataset" in
            *SWE-bench_Lite|*SWE-bench_Lite/*) ;;
            *)
                echo "ERROR: agentic generation only produces SWE-bench_Lite instances, but ${yaml_path} dataset_path='${dataset}' is not Lite." >&2
                echo "       Use gen_mode=single-shot for other datasets, or point the YAML at SWE-bench_Lite." >&2
                rm -rf "$gen_dir" 2>/dev/null || true
                return 1
                ;;
        esac
        _run_swebench_agentic_generation "$gen_dir" "$@" || {
            local agen_rc=$?
            rm -rf "$gen_dir" 2>/dev/null || true
            return "$agen_rc"
        }
        score_input=(--predictions-file "$gen_dir/agent_out/preds.json")
        mkdir -p "$out_dir"
        cp -f "$gen_dir/agent_out/preds.json" "$out_dir/agent_preds.json" 2>/dev/null || true
        find "$gen_dir/agent_out" -name "*.traj*" -exec cp -f {} "$out_dir/" \; 2>/dev/null || true
    else
        local prev_tasks_dir="${EVAL_TASKS_DIR:-}"
        local prev_include_path="${EVAL_INCLUDE_PATH:-}"
        export EVAL_TASKS_DIR="$task_name"
        export EVAL_INCLUDE_PATH="$(dirname "$yaml_path")"
        local gen_rc=0
        run_lm_eval "$@" --results-dir "$gen_dir" || gen_rc=$?
        export EVAL_TASKS_DIR="$prev_tasks_dir"
        export EVAL_INCLUDE_PATH="$prev_include_path"
        if [ "$gen_rc" -ne 0 ]; then
            echo "ERROR: swebench generation (lm-eval) failed with $gen_rc" >&2
            rm -rf "$gen_dir" 2>/dev/null || true
            return "$gen_rc"
        fi

        mkdir -p "$out_dir"
        find "$gen_dir" -name 'samples_*.jsonl' -exec cp -f {} "$out_dir"/ \; 2>/dev/null || true
        score_input=(--samples-dir "$gen_dir")
    fi
    export EVAL_RESULT_DIR="$out_dir"

    local lm_eval_version
    lm_eval_version=$(python3 -c 'import lm_eval; print(lm_eval.__version__)' 2>/dev/null || echo unknown)

    if [ "${SWEBENCH_SKIP_SCORE}" = "true" ]; then
        local skip_rc=0
        python3 -m infx.evals.swebench_score \
            "${score_input[@]}" --out-dir "$out_dir" \
            --model-name "${MODEL_NAME:-$MODEL}" --task-name "$task_name" \
            --predictions-only || skip_rc=$?
        echo "SWEBENCH_SKIP_SCORE=true: staged predictions only (no resolved-rate)." >&2
        rm -rf "$gen_dir" 2>/dev/null || true
        return "$skip_rc"
    fi

    if [ "${INFERENCEX_SWEBENCH_RUNTIME_READY:-false}" != "true" ]; then
        _install_swebench_deps
        export INFERENCEX_SWEBENCH_RUNTIME_READY=true
    fi
    _ensure_modal_credentials
    local score_rc=0
    local ns_args=()
    if [ "${SWEBENCH_NAMESPACE+set}" = "set" ]; then ns_args=(--namespace "$SWEBENCH_NAMESPACE"); fi
    local modal_args=()
    if [ "${SWEBENCH_USE_MODAL}" = "true" ]; then modal_args=(--modal); fi
    local itimeout_args=(--instance-timeout "${SWEBENCH_EVAL_TIMEOUT}")
    # Avoid holding the GPU on scoring stalls.
    timeout "${SWEBENCH_SCORE_TIMEOUT}" \
    python3 -m infx.evals.swebench_score \
        "${score_input[@]}" \
        --out-dir "$out_dir" \
        --model-name "${MODEL_NAME:-$MODEL}" \
        --task-name "$task_name" \
        --dataset-name "$dataset" \
        --max-workers "${SWEBENCH_MAX_WORKERS}" \
        --lm-eval-version "$lm_eval_version" \
        "${modal_args[@]}" \
        "${itimeout_args[@]}" \
        "${ns_args[@]}" \
        || score_rc=$?
    rm -rf "$gen_dir" 2>/dev/null || true
    if [ "$score_rc" -ne 0 ]; then
        echo "ERROR: swebench scoring failed with $score_rc" >&2
        return "$score_rc"
    fi
}

_wait_for_openai_chat_route() {
    check_env_vars EVAL_ENDPOINT_READY_TIMEOUT_SECONDS EVAL_MODEL_STABILIZATION_SECONDS PORT
    local port="${PORT}"
    local timeout_seconds="${EVAL_ENDPOINT_READY_TIMEOUT_SECONDS}"
    local poll_seconds=5
    local stabilization_seconds="${EVAL_MODEL_STABILIZATION_SECONDS}"
    local start_seconds=$SECONDS
    local server_ready_since=-1
    local next_report=0
    local elapsed percent chat_status
    local served_model="${SERVED_MODEL_NAME:-${MODEL:-}}"
    local health_url models_url chat_url

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --port)
                if [[ $# -lt 2 || -z "${2:-}" || "${2:-}" == --* ]]; then
                    echo "ERROR: --port requires a value" >&2
                    return 2
                fi
                port="$2"
                shift 2
                ;;
            *) shift ;;
        esac
    done
    if ! [[ "$timeout_seconds" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: EVAL_ENDPOINT_READY_TIMEOUT_SECONDS must be a positive integer" >&2
        return 2
    fi
    if ! [[ "$stabilization_seconds" =~ ^[0-9]+$ ]]; then
        echo "ERROR: EVAL_MODEL_STABILIZATION_SECONDS must be a non-negative integer" >&2
        return 2
    fi
    if [ -z "$served_model" ]; then
        echo "ERROR: MODEL or SERVED_MODEL_NAME is required for chat endpoint readiness" >&2
        return 2
    fi
    health_url="http://localhost:${port}/health"
    models_url="http://localhost:${port}/v1/models"
    chat_url="http://localhost:${port}/v1/chat/completions"

    while true; do
        local model_ready=false
        local server_ready=false
        if curl -fsS --max-time 10 "$health_url" >/dev/null 2>&1; then
            server_ready=true
        fi
        if curl -fsS --max-time 10 "$models_url" 2>/dev/null \
            | python3 -c '
import json
import sys

expected = sys.argv[1]
payload = json.load(sys.stdin)
models = payload.get("data", [])
raise SystemExit(0 if any(model.get("id") == expected for model in models) else 1)
' "$served_model" >/dev/null 2>&1; then
            model_ready=true
        fi

        chat_status=""
        if [ "$model_ready" = true ]; then
            chat_status="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 10 \
                "$chat_url" 2>/dev/null)" || true
            case "$chat_status" in
                401|403|405) break ;;
            esac
        fi
        if [ "$server_ready" = true ]; then
            if [ "$server_ready_since" -lt 0 ]; then
                server_ready_since=$SECONDS
            fi
            if [ $((SECONDS - server_ready_since)) -ge "$stabilization_seconds" ]; then
                break
            fi
        else
            server_ready_since=-1
        fi

        elapsed=$((SECONDS - start_seconds))
        if [ "$elapsed" -ge "$timeout_seconds" ]; then
            echo "ERROR: chat endpoint for model '$served_model' did not become ready within ${timeout_seconds}s: $chat_url" >&2
            return 1
        fi
        if [ "$elapsed" -ge "$next_report" ]; then
            percent=$((elapsed * 100 / timeout_seconds))
            echo "Waiting for chat endpoint for model '$served_model': ${elapsed}/${timeout_seconds}s (${percent}%)"
            next_report=$((next_report + 60))
        fi
        sleep "$poll_seconds"
    done
    echo "OpenAI chat endpoint ready for model '$served_model': $chat_url"
}


# Unified eval entrypoint

run_eval() {
    check_env_vars EVAL_ONLY IS_AGENTIC
    local cli_framework=""
    local forwarded=()
    # Keep runner-selected suite identity scoped to this invocation.
    local EVAL_SUITE="${EVAL_SUITE:-}"
    unset EVAL_COMPLETED_SUITE

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --framework)
                if [[ $# -lt 2 || -z "${2:-}" || "${2:-}" == --* ]]; then
                    echo "ERROR: --framework requires a value" >&2
                    return 2
                fi
                cli_framework="$2"
                shift 2
                ;;
            *)
                forwarded+=("$1")
                shift
                ;;
        esac
    done

    local scenario_default="lm-eval"
    local scenario_is_agentic=0
    if [ "${IS_AGENTIC}" = "1" ] || [ "${SCENARIO_TYPE:-}" = "agentic-coding" ]; then
        scenario_is_agentic=1
    fi

    local framework="${EVAL_FRAMEWORK:-${cli_framework:-$scenario_default}}"
    case "$framework" in
        kimi-vendor)
            [ -n "${EVAL_SUITE:-}" ] || EVAL_SUITE="kimi_tool_call_schema"
            ;;
        minimax-vendor)
            [ -n "${EVAL_SUITE:-}" ] || EVAL_SUITE="minimax_m3_smoke"
            ;;
        bfcl)
            [ -n "${EVAL_SUITE:-}" ] || EVAL_SUITE="bfcl_smoke"
            ;;
    esac

    case "${EVAL_SUITE:-}" in
        "") ;;
        *[!A-Za-z0-9_.-]*)
            echo "ERROR: EVAL_SUITE may contain only letters, digits, '.', '_', and '-'" >&2
            return 2
            ;;
    esac

    if [ -n "${EVAL_SUITE:-}" ] \
        && [ "$framework" != "kimi-vendor" ] \
        && [ "$framework" != "minimax-vendor" ] \
        && [ "$framework" != "bfcl" ]; then
        echo "ERROR: EVAL_SUITE is only supported with kimi-vendor, minimax-vendor, or bfcl" >&2
        return 2
    fi

    if [ "${EVAL_ONLY}" = "true" ]; then
        case "$framework" in
            kimi-vendor|minimax-vendor|bfcl)
                _wait_for_openai_chat_route "${forwarded[@]}" || return $?
                ;;
        esac
    fi

    # Explicit verifier suites use fixed request budgets and do not consume
    # EVAL_MAX_MODEL_LEN, so avoid loading model configuration for those paths.
    if [ "$framework" != "kimi-vendor" ] \
        && [ "$framework" != "minimax-vendor" ] \
        && [ "$framework" != "bfcl" ] \
        && [ -z "${EVAL_MAX_MODEL_LEN:-}" ]; then
        compute_eval_context_length "$MODEL" "${MAX_MODEL_LEN:-0}" > /dev/null
    fi

    unset EVAL_BATCHED_CONCS
    unset EVAL_BATCHED_COMPLETED_CONCS
    unset EVAL_BATCHED_FAILED_CONCS

    local requested_concs="${EVAL_CONCURRENT_REQUESTS:-}"
    local eval_concs=()
    if [ -n "$requested_concs" ]; then
        read -r -a eval_concs <<< "$requested_concs"
    fi

    if [ "${#eval_concs[@]}" -gt 1 ]; then
        if [[ "$framework" != "lm-eval" && "$framework" != "lm_eval" ]]; then
            echo "ERROR: batched eval concurrency is only supported for lm-eval" >&2
            return 1
        fi

        local eval_conc results_dir eval_rc stage_rc
        local completed_concs=()
        local failed_concs=()

        for eval_conc in "${eval_concs[@]}"; do
            if [[ ! "$eval_conc" =~ ^[1-9][0-9]*$ ]]; then
                echo "ERROR: invalid eval concurrency '${eval_conc}'" >&2
                return 1
            fi

            if ! results_dir=$(mktemp -d /tmp/eval_out-conc"${eval_conc}"-XXXXXX); then
                echo "ERROR: failed to create eval output directory for concurrency ${eval_conc}" >&2
                failed_concs+=("$eval_conc")
                continue
            fi

            echo "Running lm-eval at concurrency ${eval_conc} using the existing engine"
            export EVAL_CONCURRENT_REQUESTS="$eval_conc"
            export CONC="$eval_conc"
            eval_rc=0
            stage_rc=0
            run_lm_eval "${forwarded[@]}" --results-dir "$results_dir" \
                || eval_rc=$?
            _stage_lm_eval_artifacts "$results_dir" "$eval_conc" \
                || stage_rc=$?

            if [ "$eval_rc" -eq 0 ] && [ "$stage_rc" -eq 0 ]; then
                completed_concs+=("$eval_conc")
            else
                echo "ERROR: lm-eval failed at concurrency ${eval_conc} (eval_rc=${eval_rc}, stage_rc=${stage_rc})" >&2
                failed_concs+=("$eval_conc")
            fi
        done

        export EVAL_CONCURRENT_REQUESTS="$requested_concs"
        export EVAL_RESULT_DIR=""
        export EVAL_BATCHED_CONCS="${eval_concs[*]}"
        export EVAL_BATCHED_COMPLETED_CONCS="${completed_concs[*]}"
        export EVAL_BATCHED_FAILED_CONCS="${failed_concs[*]}"

        if [ "${#failed_concs[@]}" -gt 0 ]; then
            echo "ERROR: batched eval failed for concurrency: ${failed_concs[*]}" >&2
            echo "Deferring failure until post-upload score validation preserves all artifacts" >&2
        fi
        return 0
    fi

    if [ -n "${EVAL_CONCURRENT_REQUESTS:-}" ]; then
        export CONC="$EVAL_CONCURRENT_REQUESTS"
    fi

    local eval_rc=0
    case "$framework" in
        lm-eval|lm_eval) run_lm_eval "${forwarded[@]}" || eval_rc=$? ;;
        swebench)        run_swebench_eval "${forwarded[@]}" || eval_rc=$? ;;
        kimi-vendor)     run_kimi_vendor_eval "${forwarded[@]}" || eval_rc=$? ;;
        minimax-vendor)  run_minimax_vendor_eval "${forwarded[@]}" || eval_rc=$? ;;
        bfcl)           run_bfcl_eval "${forwarded[@]}" || eval_rc=$? ;;
        *)               echo "Unknown framework '${framework}'"; eval_rc=1 ;;
    esac

    if [ -n "${EVAL_SUITE:-}" ]; then
        export EVAL_COMPLETED_SUITE="$EVAL_SUITE"
    fi

    local stage_rc=0
    # Agentic eval-only recipes have no separate staging step. Provider
    # failures are staged before returning so diagnostic artifacts survive.
    if { [ "${EVAL_ONLY}" = "true" ] && [ "$scenario_is_agentic" = "1" ]; } \
        || { { [ "$framework" = "kimi-vendor" ] \
            || [ "$framework" = "minimax-vendor" ] \
            || [ "$framework" = "bfcl" ]; } \
            && [ "$eval_rc" -ne 0 ]; }; then
        append_lm_eval_summary || stage_rc=$?
    fi
    if [ "$eval_rc" -ne 0 ]; then
        echo "ERROR: run_eval failed with exit code $eval_rc" >&2
        if [ "${EVAL_ONLY}" = "true" ]; then
            echo "Eval-only mode: failing after artifact collection" >&2
        fi
        return "$eval_rc"
    fi
    if [ "$stage_rc" -ne 0 ]; then
        echo "ERROR: eval artifact staging failed with exit code $stage_rc" >&2
        return "$stage_rc"
    fi
    return 0
}


# Agentic trace replay helpers (aiperf driver)

AGENTIC_DIR="${INFMAX_CONTAINER_WORKSPACE}/utils/agentic-benchmark"
AIPERF_DIR="${INFMAX_CONTAINER_WORKSPACE}/utils/aiperf"
AIPERF_RUNTIME_DIR="${AIPERF_RUNTIME_DIR:-${TMPDIR:-/tmp}/inferencex-agentic-${SLURM_JOB_ID:-$$}}"
AIPERF_VENV="${AIPERF_RUNTIME_DIR}/venv"
AIPERF_UV_INSTALL_DIR="${AIPERF_RUNTIME_DIR}/uv/bin"
AIPERF_UV_CACHE_DIR="${AIPERF_RUNTIME_DIR}/uv-cache"
AIPERF_PYTHON="${AIPERF_VENV}/bin/python"
AIPERF_CLI="${AIPERF_VENV}/bin/aiperf"
AIPERF_HF_CLI="${AIPERF_VENV}/bin/hf"
AIPERF_DEPS_READY=0

agentic_pip_install() {
    local pip_install=(python3 -m pip install)
    if python3 -m pip install --help 2>/dev/null | grep -q -- "--break-system-packages"; then
        pip_install+=(--break-system-packages)
    fi

    "${pip_install[@]}" "$@"
}

ensure_agentic_uv() {
    if command -v uv >/dev/null 2>&1; then
        AIPERF_UV_BIN="$(command -v uv)"
        return
    fi

    AIPERF_UV_BIN="${AIPERF_UV_INSTALL_DIR}/uv"
    if [ ! -x "$AIPERF_UV_BIN" ]; then
        mkdir -p "$AIPERF_UV_INSTALL_DIR"
        curl -LsSf https://astral.sh/uv/install.sh |
            UV_INSTALL_DIR="$AIPERF_UV_INSTALL_DIR" sh
    fi

    if [ ! -x "$AIPERF_UV_BIN" ]; then
        echo "ERROR: uv installation did not create $AIPERF_UV_BIN" >&2
        return 1
    fi
}

install_agentic_deps() {
    check_env_vars AIPERF_PYTHON_VERSION INFMAX_CONTAINER_WORKSPACE
    if [ "$AIPERF_DEPS_READY" = "1" ]; then
        return
    fi

    # uv install from the checked-out aiperf source: needs no git, and rootless
    # Enroot containers cannot mutate dpkg.

    ensure_agentic_uv || return $?
    rm -rf "$AIPERF_VENV"
    mkdir -p "$AIPERF_UV_CACHE_DIR"

    # Pin the interpreter version instead of the container's python3: aiperf
    # dropped Python 3.10 (SemiAnalysisAI/aiperf#1107) while sglang-rocm/vllm-rocm
    # images still default to 3.10.12, which left the venv without aiperf/hf.
    # uv downloads a standalone build when the system lacks one.
    UV_CACHE_DIR="$AIPERF_UV_CACHE_DIR" \
        "$AIPERF_UV_BIN" venv --python "${AIPERF_PYTHON_VERSION}" "$AIPERF_VENV" || return $?
    UV_CACHE_DIR="$AIPERF_UV_CACHE_DIR" UV_HTTP_TIMEOUT=120 UV_HTTP_RETRIES=3 \
        "$AIPERF_UV_BIN" pip install --python "$AIPERF_PYTHON" \
        -r "$AGENTIC_DIR/requirements.txt" \
        -e "$AIPERF_DIR" \
        "datasets>=4.7.0" \
        "huggingface_hub[cli]>=0.25.0" \
        urllib3 \
        requests || {
            echo "ERROR: benchmark client dependency bootstrap failed; inspect network/package resolution before recipe repairs" >&2
            return 1
        }

    if [ ! -x "$AIPERF_CLI" ] || [ ! -x "$AIPERF_HF_CLI" ]; then
        echo "ERROR: isolated AIPerf environment is incomplete at $AIPERF_VENV" >&2
        return 1
    fi
    AIPERF_DEPS_READY=1
}

ensure_hf_cli() {
    install_agentic_deps
}

resolve_trace_source() {
    # WEKA_LOADER_OVERRIDE picks an aiperf public-dataset loader for recipes
    # with non-default context caps (minimaxm2.5 at ~256k cannot replay the
    # unfiltered corpus) or to pin an older corpus. Default: the 062126 v7
    # corpus; 1M-context families take the unfiltered variant, others 256k.
    local default_loader
    case "${MODEL_PREFIX:-}" in
        dsv4*|glm5.2*|minimaxm3*|kimik3*)
            default_loader="semianalysis_cc_traces_weka_062126"
            ;;
        *)
            default_loader="semianalysis_cc_traces_weka_062126_256k"
            ;;
    esac
    local loader="${WEKA_LOADER_OVERRIDE:-$default_loader}"
    local dataset
    case "$loader" in
        semianalysis_cc_traces_weka_with_subagents)
            dataset="semianalysisai/cc-traces-weka-061526"
            ;;
        semianalysis_cc_traces_weka_with_subagents_256k)
            dataset="semianalysisai/cc-traces-weka-061526-256k"
            ;;
        semianalysis_cc_traces_weka_with_subagents_060226)
            dataset="semianalysisai/cc-traces-weka-with-subagents-060226"
            ;;
        semianalysis_cc_traces_weka_with_subagents_060226_256k)
            dataset="semianalysisai/cc-traces-weka-with-subagents-060226-256k"
            ;;
        semianalysis_cc_traces_weka_with_subagents_060526)
            dataset="semianalysisai/cc-traces-weka-with-subagents-060526"
            ;;
        semianalysis_cc_traces_weka_with_subagents_060526_256k)
            dataset="semianalysisai/cc-traces-weka-with-subagents-060526-256k"
            ;;
        semianalysis_cc_traces_weka_with_subagents_060826)
            dataset="semianalysisai/cc-traces-weka-with-subagents-060826"
            ;;
        semianalysis_cc_traces_weka_with_subagents_060826_256k)
            dataset="semianalysisai/cc-traces-weka-with-subagents-060826-256k"
            ;;
        semianalysis_cc_traces_weka_061326)
            dataset="semianalysisai/cc-traces-weka-061326"
            ;;
        semianalysis_cc_traces_weka_061326_256k)
            dataset="semianalysisai/cc-traces-weka-061326-256k"
            ;;
        semianalysis_cc_traces_weka_061526)
            dataset="semianalysisai/cc-traces-weka-061526"
            ;;
        semianalysis_cc_traces_weka_061526_256k)
            dataset="semianalysisai/cc-traces-weka-061526-256k"
            ;;
        semianalysis_cc_traces_weka_062126)
            dataset="semianalysisai/cc-traces-weka-062126"
            ;;
        semianalysis_cc_traces_weka_062126_256k)
            dataset="semianalysisai/cc-traces-weka-062126-256k"
            ;;
        *)
            echo "Error: unknown WEKA_LOADER_OVERRIDE='$loader'. Allowed: semianalysis_cc_traces_weka_with_subagents, semianalysis_cc_traces_weka_with_subagents_256k, semianalysis_cc_traces_weka_with_subagents_060226, semianalysis_cc_traces_weka_with_subagents_060226_256k, semianalysis_cc_traces_weka_with_subagents_060526, semianalysis_cc_traces_weka_with_subagents_060526_256k, semianalysis_cc_traces_weka_with_subagents_060826, semianalysis_cc_traces_weka_with_subagents_060826_256k, semianalysis_cc_traces_weka_061326, semianalysis_cc_traces_weka_061326_256k, semianalysis_cc_traces_weka_061526, semianalysis_cc_traces_weka_061526_256k, semianalysis_cc_traces_weka_062126, semianalysis_cc_traces_weka_062126_256k" >&2
            exit 1
            ;;
    esac
    TRACE_SOURCE_FLAG="--public-dataset $loader"
    echo "Loading traces via aiperf public-dataset: $loader ($dataset) [MODEL_PREFIX=${MODEL_PREFIX:-unset}]"
    # Pre-download into the shared HF_HUB_CACHE so later jobs hit cache.
    ensure_hf_cli
    "$AIPERF_HF_CLI" download --repo-type dataset "$dataset"
}

build_replay_cmd() {
    check_env_vars INFMAX_CONTAINER_WORKSPACE MODEL PORT CONC DURATION
    check_env_vars \
        AIPERF_FAILED_REQUEST_THRESHOLD AIPERF_LIVE_FAILED_REQUEST_THRESHOLD \
        AIPERF_TRACE_IDLE_GAP_CAP_SECONDS
    check_env_vars \
        AGENTIC_WARMUP_GRACE_PERIOD AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES \
        AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS AIPERF_EXPERIMENTAL_FAST \
        AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID AIPERF_UNSAFE_OVERRIDE \
        AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING AIPERF_WARMUP_REQUESTS_PER_LANE
    # Recorded assistant responses drive prompt construction by default;
    # AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=1 threads the live response
    # back into the session instead. The scenario plugin locks --cache-bust
    # first_turn_prefix and a 10s whole-system idle cap; the 300s per-trajectory
    # cap below is ours. See utils/aiperf/docs/tutorials/agentx-mvp.md.
    local result_dir="$1"
    local duration="$DURATION"
    local warmup_requests_per_lane="${AIPERF_WARMUP_REQUESTS_PER_LANE}"

    # Fast mode: one advance per lane and a 20-minute profile.
    if [[ "${AIPERF_EXPERIMENTAL_FAST}" == "1" ]]; then
        duration=1200
        warmup_requests_per_lane=1
    fi

    export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES="${AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES}"
    # Dataset configuration takes 4-5 min on fast /tmp (B300) but reached 14 min
    # on H200 when 14 parallel jobs hit aiperf's default 900s Configure Profiling
    # timeout; 1800s absorbs that without touching the measurement window.
    export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
    # aiperf requires SERVICE_PROFILE_CONFIGURE_TIMEOUT >= DATASET_CONFIGURATION_TIMEOUT.
    export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
    # Headless realtime metrics are opt-in on AIPerf main.
    export AIPERF_UI_REALTIME_METRICS_ENABLED=true
    REPLAY_CMD="$AIPERF_CLI profile --scenario inferencex-agentx-mvp"
    REPLAY_CMD+=" --url ${AIPERF_SERVER_URL:-http://localhost:$PORT}"
    REPLAY_CMD+=" --endpoint /v1/chat/completions"
    REPLAY_CMD+=" --endpoint-type chat"
    REPLAY_CMD+=" --streaming"
    # SERVED_MODEL_NAME covers frontends that register the model under a wire
    # name (dynamo-trt serves "DeepSeek-V4-Pro" while $MODEL is the HF id);
    # a mismatch 404s at warmup.
    REPLAY_CMD+=" --model ${SERVED_MODEL_NAME:-$MODEL}"
    # The tokenizer defaults to --model, and a wire name is not necessarily a
    # valid HF repo id, so pass the real id explicitly.
    REPLAY_CMD+=" --tokenizer $MODEL"
    REPLAY_CMD+=" --concurrency $CONC"
    REPLAY_CMD+=" --benchmark-duration $duration"
    REPLAY_CMD+=" --stats-interval 30"
    REPLAY_CMD+=" --random-seed 42"
    # Live abort threshold; recipes with correlated low-concurrency trajectories
    # may loosen it while AIPERF_FAILED_REQUEST_THRESHOLD stays the post-run gate.
    REPLAY_CMD+=" --failed-request-threshold $AIPERF_LIVE_FAILED_REQUEST_THRESHOLD"
    # AIPerf clamps the start ratio so at least one profile turn follows warmup.
    REPLAY_CMD+=" --trajectory-start-min-ratio 0.25"
    REPLAY_CMD+=" --trajectory-start-max-ratio 0.75"
    # Extra one-token requests per lane after the t* snapshot primers; profiling
    # resumes from the resulting live state. Do not pass --burst-phase-starts:
    # the spread default preserves each lane's recorded phase-start offset.
    REPLAY_CMD+=" --warmup-requests-per-lane $warmup_requests_per_lane"
    # Caps end-to-start idle time per trajectory tree (root plus subagents)
    # without reordering requests or bypassing spawn/join dependencies.
    REPLAY_CMD+=" --trace-idle-gap-cap-seconds $AIPERF_TRACE_IDLE_GAP_CAP_SECONDS"
    # Maximum wait for warmup to drain, not a fixed sleep; saturation arms with a
    # larger in-flight set can raise AGENTIC_WARMUP_GRACE_PERIOD.
    REPLAY_CMD+=" --warmup-grace-period ${AGENTIC_WARMUP_GRACE_PERIOD}"
    # Server usage fields for ISL/OSL instead of client-side tokenize; the
    # per-record tokenization was pinning CPU on minimax-m2.5 at high concurrency.
    REPLAY_CMD+=" --use-server-token-count"
    if [ -n "${AIPERF_EXTRA_INPUTS:-}" ]; then
        REPLAY_CMD+=" --extra-inputs $AIPERF_EXTRA_INPUTS"
    fi
    # Dynamo's KV router needs an explicit session binding to keep later turns on
    # the prefill worker owning their prefix blocks; X-Correlation-ID alone does
    # not establish it. This flag emits nvext.session_control, which dynamo builds
    # after #9920 (v1.3.0-dev) reject with 400; recipes on those builds set
    # AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true (header routing)
    # or AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING=0.
    if [[ "${FRAMEWORK:-}" == dynamo-* \
          && "${AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING}" != "0" \
          && "${AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID}" != "true" ]]; then
        REPLAY_CMD+=" --use-dynamo-conv-aware-routing"
        # The upstream 300s affinity TTL is shorter than an overloaded agentic
        # request; this is the router's inactivity lease, not an HTTP timeout.
        REPLAY_CMD+=" --dynamo-session-timeout-seconds ${AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS}"
    fi
    # aiperf's GpuMetricTimeSeries freezes its schema on the first DCGM scrape
    # and KeyErrors when an optional field (xid_errors, power_violation) first
    # appears mid-run. The gpu_telemetry artifact is unused downstream; the
    # Prometheus server-metrics path is unaffected.
    REPLAY_CMD+=" --no-gpu-telemetry"
    # The dataset manager loads the tokenizer regardless of
    # --use-server-token-count, and Kimi checkpoints ship a custom tokenizer
    # that needs trust_remote_code. Benign for other models.
    REPLAY_CMD+=" --tokenizer-trust-remote-code"
    # The WEKA corpus has a few traces longer than smaller-context servers
    # accept; replayed unfiltered they become deterministic 4xxs that still
    # pressure the engine while queued.
    if [ -n "${MAX_MODEL_LEN:-}" ] && [ "$MAX_MODEL_LEN" != "0" ]; then
        REPLAY_CMD+=" --max-context-length $MAX_MODEL_LEN"
    fi
    # Default is 100; the with-subagents corpus has 393 unique traces. The loader
    # treats this as min(cap, available), see semianalysis_cc_traces_weka.py.
    REPLAY_CMD+=" --num-dataset-entries 393"
    # Per-second server-metrics slices feed the post-run plotter; matches
    # kv-cache-tester's poll_interval=1.0 so metrics_plots.png is comparable.
    # Without it aiperf emits only aggregates and the panels are flat lines.
    REPLAY_CMD+=" --slice-duration 1.0"
    # Multi-node launchers pass every worker's Prometheus endpoint; AIPerf takes
    # several values after one --server-metrics flag and keeps endpoint_url per series.
    if [ -n "${AIPERF_SERVER_METRICS_URLS:-}" ]; then
        local metrics_url
        local -a metrics_urls
        IFS=',' read -r -a metrics_urls <<< "$AIPERF_SERVER_METRICS_URLS"
        REPLAY_CMD+=" --server-metrics"
        for metrics_url in "${metrics_urls[@]}"; do
            if [ -z "$metrics_url" ] || [[ "$metrics_url" == *[[:space:]]* ]]; then
                echo "ERROR: AIPERF_SERVER_METRICS_URLS must be a comma-separated list of non-empty URLs" >&2
                return 1
            fi
            REPLAY_CMD+=" $metrics_url"
        done
    fi
    REPLAY_CMD+=" --output-artifact-dir $result_dir/aiperf_artifacts"
    # The scenario enforces a 900s minimum duration; shorter smoke tests need
    # --unsafe-override and are flagged submission_valid=false.
    if [ "$duration" -lt 900 ] || [ "${AIPERF_UNSAFE_OVERRIDE}" = "true" ]; then
        REPLAY_CMD+=" --unsafe-override"
    fi
    REPLAY_CMD+=" $TRACE_SOURCE_FLAG"
}

write_agentic_result_json() {
    check_env_vars INFMAX_CONTAINER_WORKSPACE
    # Writes $AGENTIC_OUTPUT_DIR/$RESULT_FILENAME.json; the workflow checks that
    # file exists, and the caller separately rejects high error rates.
    local result_dir="$1"
    (
        cd "$INFMAX_CONTAINER_WORKSPACE"
        RESULT_DIR="$result_dir" AGENTIC_OUTPUT_DIR="${AGENTIC_OUTPUT_DIR:-$INFMAX_CONTAINER_WORKSPACE}" \
            "$AIPERF_PYTHON" -m infx.results.agentic.process_agentic_result
    )

    # Best-effort metrics_plots.png (matplotlib may be missing in stripped-down
    # images); the agg JSON above is the success gate.
    PYTHONPATH="$INFMAX_CONTAINER_WORKSPACE${PYTHONPATH:+:$PYTHONPATH}" "$AIPERF_PYTHON" -m infx.results.generate_aiperf_plots "$result_dir" 2>&1 || true
}

validate_required_agentic_server_metrics() {
    local result_dir="$1"
    local required_prefix="${AIPERF_REQUIRED_SERVER_METRIC_PREFIX:-}"
    local metrics_dir="$result_dir/aiperf_artifacts"
    local metrics_json="$metrics_dir/server_metrics_export.json"
    local metrics_csv="$metrics_dir/server_metrics_export.csv"

    # Opt-in: recipes that require trace charts set a metric prefix (for example
    # `sglang:`) and fail loudly instead of publishing a partial trace artifact.
    if [ -z "$required_prefix" ]; then
        return 0
    fi

    if [ ! -s "$metrics_json" ] || [ ! -s "$metrics_csv" ]; then
        echo "ERROR: required AIPerf server metrics artifacts are missing or empty in $metrics_dir" >&2
        return 1
    fi

    # The JSON can be multi-GiB, so scan for the key instead of parsing; metric
    # names are object keys, so a hit proves backend engine metrics were captured.
    if ! grep -F -m 1 -q "\"${required_prefix}" "$metrics_json"; then
        echo "ERROR: $metrics_json contains no metric with required prefix '$required_prefix'" >&2
        return 1
    fi

    echo "Validated required AIPerf server metrics prefix '$required_prefix'"
}

run_agentic_replay_and_write_outputs() (
    check_env_vars ENABLE_AGENTX_POWER IS_MULTINODE REQUIRE_POWER
    local result_dir="$1"
    local replay_rc
    local validation_rc
    local power_rc=0
    local agentx_power_enabled=0
    local agentx_multinode_power_enabled=0
    local agentx_multinode_contract_missing=0
    local agentx_monitor_stopped=1

    case "${ENABLE_AGENTX_POWER}" in
        1|true|TRUE|yes|YES)
            if [ "${IS_MULTINODE}" = "true" ]; then
                if [ -n "${SRT_MEASUREMENT_WINDOW_DIR:-}" ]; then
                    agentx_multinode_power_enabled=1
                else
                    agentx_multinode_contract_missing=1
                fi
            else
                agentx_power_enabled=1
            fi
            ;;
    esac

    _stop_agentx_power_monitor() {
        if [ "$agentx_monitor_stopped" = "0" ]; then
            agentx_monitor_stopped=1
            stop_gpu_monitor
        fi
    }

    _write_agentx_multinode_window() {
        local state="$1"
        local -a power_args
        power_args=(
            --result-dir "$result_dir"
            --concurrency "${CONC:?CONC must be set for multinode AgentX power}"
            --write-multinode-window "$state"
        )
        case "${REQUIRE_POWER}" in
            1|true|TRUE|yes|YES) power_args+=(--require-power) ;;
        esac
        (
            cd "$INFMAX_CONTAINER_WORKSPACE"
            "$AIPERF_PYTHON" -m infx.results.agentic.power_adapter "${power_args[@]}"
        )
    }

    if [ "$agentx_power_enabled" = "1" ] || [ "$agentx_multinode_power_enabled" = "1" ]; then
        # AIPerf exports naive local datetimes and SMI the same host wall clock;
        # the adapter needs the offset to normalize the profiling window.
        date +%z > "$result_dir/agentic_power_timezone_offset.txt"
    fi

    if [ "$agentx_multinode_power_enabled" = "1" ]; then
        set +e
        _write_agentx_multinode_window running
        power_rc=$?
        set -e
        if [ "$power_rc" -ne 0 ]; then
            echo "ERROR: failed to publish the AgentX formal running power window" >&2
            return "$power_rc"
        fi
    fi

    if [ "$agentx_power_enabled" = "1" ]; then
        start_gpu_monitor --output "$result_dir/gpu_metrics.csv"
        agentx_monitor_stopped=0
        # This function runs in a subshell, so these traps cannot clobber
        # launcher-owned ones; the stopped flag keeps cleanup idempotent.
        trap '_stop_agentx_power_monitor' EXIT
        trap '_stop_agentx_power_monitor; exit 130' INT
        trap '_stop_agentx_power_monitor; exit 143' TERM
    fi

    echo "$REPLAY_CMD" > "$result_dir/benchmark_command.txt"

    set +e
    set -x
    run_server_client $REPLAY_CMD 2>&1 | tee "$result_dir/benchmark.log"
    replay_rc=${PIPESTATUS[0]}
    set +x
    set -e

    if [ "$agentx_power_enabled" = "1" ]; then
        _stop_agentx_power_monitor
        trap - EXIT INT TERM
    fi

    write_agentic_result_json "$result_dir"

    if [ "$agentx_multinode_power_enabled" = "1" ] && [ "$replay_rc" -eq 0 ]; then
        set +e
        _write_agentx_multinode_window completed
        power_rc=$?
        set -e
    fi

    if [ "$agentx_power_enabled" = "1" ] || [ "$agentx_multinode_contract_missing" = "1" ]; then
        local expected_num_gpus
        local -a power_args
        power_args=(
            --result-dir "$result_dir"
            --agg-result "${AGENTIC_OUTPUT_DIR:-$INFMAX_CONTAINER_WORKSPACE}/$RESULT_FILENAME.json"
        )
        if [ "$agentx_multinode_contract_missing" = "1" ]; then
            power_args+=(--multinode-contract-missing)
        else
            check_env_vars TP PP_SIZE PCP_SIZE
            expected_num_gpus=$((TP * PP_SIZE * PCP_SIZE))
            power_args+=(--expected-num-gpus "$expected_num_gpus")
        fi
        case "${REQUIRE_POWER}" in
            1|true|TRUE|yes|YES) power_args+=(--require-power) ;;
        esac
        set +e
        (
            cd "$INFMAX_CONTAINER_WORKSPACE"
            "$AIPERF_PYTHON" -m infx.results.agentic.power_adapter "${power_args[@]}"
        )
        power_rc=$?
        set -e
    fi

    PYTHONPATH="$INFMAX_CONTAINER_WORKSPACE${PYTHONPATH:+:$PYTHONPATH}" "$AIPERF_PYTHON" -m infx.results.agentic.analyze_benchmark_distributions \
        "$result_dir/aiperf_artifacts" -o "$result_dir" 2>&1 || true

    set +e
    (
        cd "$INFMAX_CONTAINER_WORKSPACE"
        "$AIPERF_PYTHON" -m infx.results.agentic.validate_agentic_result \
            "$result_dir/aiperf_artifacts" \
            --failed-request-threshold "$AIPERF_FAILED_REQUEST_THRESHOLD"
    )
    validation_rc=$?
    set -e

    if [ "$replay_rc" -ne 0 ]; then
        echo "ERROR: agentic trace replay exited with code $replay_rc after writing available results" >&2
        return "$replay_rc"
    fi

    if [ "$validation_rc" -ne 0 ]; then
        echo "ERROR: agentic trace replay produced invalid results after writing available artifacts" >&2
        return "$validation_rc"
    fi

    if [ "$power_rc" -ne 0 ]; then
        echo "ERROR: AgentX power validation failed after writing audit artifacts" >&2
        return "$power_rc"
    fi

    validate_required_agentic_server_metrics "$result_dir"
)
