#!/usr/bin/env bash

# Shared benchmarking utilities for InferenceMAX

# Keep Python bytecode out of the mounted workspace. Benchmark jobs often run as
# root inside containers, and root-owned cache directories break future checkout
# cleanup on self-hosted runners.
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/inferencex-pycache}"
mkdir -p "$PYTHONPYCACHEPREFIX" 2>/dev/null || true

# --------------------------------
# GPU monitoring helpers
# --------------------------------

GPU_MONITOR_PID=""
GPU_METRICS_CSV="/workspace/gpu_metrics.csv"

# Start background GPU monitoring that logs metrics every second to CSV.
# Auto-detects NVIDIA (nvidia-smi) or AMD (amd-smi) GPUs.
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

    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=timestamp,index,power.draw,temperature.gpu,clocks.current.sm,clocks.current.memory,utilization.gpu,utilization.memory \
            --format=csv -l "$interval" > "$output" 2>/dev/null &
        GPU_MONITOR_PID=$!
        echo "[GPU Monitor] Started NVIDIA (PID=$GPU_MONITOR_PID, interval=${interval}s, output=$output)"
    elif command -v amd-smi &>/dev/null; then
        # Use amd-smi native watch mode (-w) which includes timestamps automatically.
        # Pipe through awk to: skip preamble lines, keep first CSV header, skip repeated headers.
        amd-smi metric -p -c -t -u -w "$interval" --csv 2>/dev/null \
            | awk '/^timestamp,/{if(!h){print;h=1};next} h{print}' > "$output" &
        GPU_MONITOR_PID=$!
        echo "[GPU Monitor] Started AMD (PID=$GPU_MONITOR_PID, interval=${interval}s, output=$output)"
    else
        echo "[GPU Monitor] No GPU monitoring tool found (nvidia-smi or amd-smi), skipping"
        return 0
    fi
}

# Stop the background GPU monitor and report file size.
stop_gpu_monitor() {
    if [[ -n "$GPU_MONITOR_PID" ]] && kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then
        kill "$GPU_MONITOR_PID" 2>/dev/null
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
        echo "[GPU Monitor] Stopped (PID=$GPU_MONITOR_PID)"
        if [[ -f "$GPU_METRICS_CSV" ]]; then
            local lines
            lines=$(wc -l < "$GPU_METRICS_CSV")
            echo "[GPU Monitor] Collected $lines rows -> $GPU_METRICS_CSV"
        fi
    fi
    GPU_MONITOR_PID=""
}

KV_METRICS_PID=""
KV_METRICS_CSV="/workspace/kv_metrics.csv"
VLLM_OFFLOAD_EXTRA_ARGS=""
VLLM_EXTRA_ARGS=""
SGLANG_EXTRA_ARGS=""

build_yarn_override_json() {
    local max_model_len="${1:?}"
    local factor="2.0"
    if (( max_model_len > 600000 )); then
        factor="4.0"
    fi
    echo "{\"text_config\":{\"rope_parameters\":{\"mrope_interleaved\":true,\"mrope_section\":[11,11,10],\"rope_type\":\"yarn\",\"rope_theta\":10000000,\"partial_rotary_factor\":0.25,\"factor\":${factor},\"original_max_position_embeddings\":262144}}}"
}

apply_yarn_config_if_needed() {
    local model="${1:?}"
    local max_model_len="${2:?}"
    if [[ "$model" == *"Qwen3.5"* || "$model" == *"qwen3.5"* || "$model" == *"Qwen3_5"* ]] && (( max_model_len > 262144 )); then
        YARN_OVERRIDE_JSON=$(build_yarn_override_json "$max_model_len")
        export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
        export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
        echo "YaRN enabled: factor=$(echo "$YARN_OVERRIDE_JSON" | grep -o '"factor":[0-9.]*' | cut -d: -f2) for max-model-len=$max_model_len"
    fi
}

_append_config_kv_once() {
    local key="$1"
    local value="$2"

    if [[ ! -f config.yaml ]]; then
        return 0
    fi

    if ! grep -Eq "^${key}:" config.yaml; then
        echo "${key}: ${value}" >> config.yaml
    fi
}

_remove_config_kv() {
    local key="$1"

    if [[ ! -f config.yaml ]]; then
        return 0
    fi

    local tmp_file
    tmp_file=$(mktemp)
    grep -Ev "^${key}:" config.yaml > "$tmp_file"
    mv "$tmp_file" config.yaml
}

_detect_total_cpu_dram_gb() {
    if [[ -n "${TOTAL_CPU_DRAM_GB:-}" ]]; then
        echo "${TOTAL_CPU_DRAM_GB}"
        return 0
    fi

    if [[ -f /proc/meminfo ]]; then
        awk '/MemTotal/{printf "%.0f", $2/1048576}' /proc/meminfo
        return 0
    fi

    if command -v sysctl >/dev/null 2>&1; then
        local mem_bytes
        mem_bytes=$(sysctl -n hw.memsize 2>/dev/null || echo "")
        if [[ -n "$mem_bytes" ]]; then
            awk -v bytes="$mem_bytes" 'BEGIN {printf "%.0f", bytes/1073741824}'
            return 0
        fi
    fi

    echo "64"
}

apply_vllm_offload_config() {
    local mode="${OFFLOAD_MODE:-legacy}"
    local detected_dram_gb=""

    VLLM_OFFLOAD_EXTRA_ARGS=""
    VLLM_EXTRA_ARGS=""

    case "$mode" in
        on)
            PREFIX_CACHING_CONFIG=""
            _remove_config_kv "no-enable-prefix-caching"
            _remove_config_kv "cpu-offload-gb"
            _remove_config_kv "swap-space"
            detected_dram_gb="$(_detect_total_cpu_dram_gb)"
            VLLM_OFFLOAD_EXTRA_ARGS="--kv_offloading_backend native --kv_offloading_size ${detected_dram_gb} --disable-hybrid-kv-cache-manager"
            ;;
        off)
            PREFIX_CACHING_CONFIG=""
            _remove_config_kv "no-enable-prefix-caching"
            _remove_config_kv "cpu-offload-gb"
            _remove_config_kv "swap-space"
            ;;
        noprefix)
            PREFIX_CACHING_CONFIG="no-enable-prefix-caching: true"
            _remove_config_kv "cpu-offload-gb"
            _remove_config_kv "swap-space"
            _append_config_kv_once "no-enable-prefix-caching" "true"
            ;;
        legacy|"")
            if [[ -n "${VLLM_CPU_OFFLOAD_GB:-}" ]]; then
                _append_config_kv_once "cpu-offload-gb" "${VLLM_CPU_OFFLOAD_GB}"
            fi
            if [[ -n "${VLLM_SWAP_SPACE_GB:-}" ]]; then
                _append_config_kv_once "swap-space" "${VLLM_SWAP_SPACE_GB}"
            fi
            ;;
        *)
            echo "WARN: Unknown OFFLOAD_MODE='${mode}', falling back to legacy behavior" >&2
            if [[ -n "${VLLM_CPU_OFFLOAD_GB:-}" ]]; then
                _append_config_kv_once "cpu-offload-gb" "${VLLM_CPU_OFFLOAD_GB}"
            fi
            if [[ -n "${VLLM_SWAP_SPACE_GB:-}" ]]; then
                _append_config_kv_once "swap-space" "${VLLM_SWAP_SPACE_GB}"
            fi
            ;;
    esac

    if [[ "${DISABLE_PREFIX_CACHING:-false}" == "true" ]]; then
        PREFIX_CACHING_CONFIG="no-enable-prefix-caching: true"
        _append_config_kv_once "no-enable-prefix-caching" "true"
    fi

    if [[ "${KV_CACHE_DTYPE:-}" == "fp8" ]]; then
        _append_config_kv_once "kv-cache-dtype" "fp8"
    fi

    if [[ -n "${YARN_OVERRIDE_JSON:-}" ]]; then
        VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-} --hf-overrides '${YARN_OVERRIDE_JSON}'"
    fi
}

apply_sglang_offload_config() {
    local mode="${OFFLOAD_MODE:-legacy}"

    SGLANG_EXTRA_ARGS=""

    case "$mode" in
        on)
            echo "WARN: OFFLOAD_MODE=on requested for SGLang, but native KV offload is not supported. Leaving cache mode unchanged." >&2
            ;;
        off)
            RADIX_CACHE_ARGS=""
            ;;
        noprefix)
            RADIX_CACHE_ARGS="--disable-radix-cache"
            ;;
        legacy|"")
            ;;
        *)
            echo "WARN: Unknown OFFLOAD_MODE='${mode}' for SGLang; leaving radix cache args unchanged." >&2
            ;;
    esac

    if [[ "${DISABLE_PREFIX_CACHING:-false}" == "true" ]]; then
        RADIX_CACHE_ARGS="--disable-radix-cache"
    fi

    if [[ -n "${YARN_OVERRIDE_JSON:-}" ]]; then
        SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-} --json-model-override-args '${YARN_OVERRIDE_JSON}'"
    fi
}

# launch_vllm_server <model> <port> <config_yaml_path> [extra args...]
# Sets: SERVER_PID, SERVER_LOG
launch_vllm_server() {
    local model="$1"
    local port="$2"
    local config_yaml_path="$3"
    shift 3 || true
    local extra_args=("$@")

    if [[ -z "$model" || -z "$port" || -z "$config_yaml_path" ]]; then
        echo "launch_vllm_server requires: model port config_yaml_path" >&2
        return 1
    fi

    hf download "$model"
    apply_vllm_offload_config

    SERVER_LOG="${SERVER_LOG:-/workspace/server.log}"

    local vllm_max_num_seqs="${VLLM_MAX_NUM_SEQS:-}"
    if [[ -z "$vllm_max_num_seqs" ]]; then
        local conc_value="${CONC:-256}"
        if [[ "$conc_value" =~ ^[0-9]+$ ]] && (( conc_value > 256 )); then
            vllm_max_num_seqs="$conc_value"
        else
            vllm_max_num_seqs="256"
        fi
    fi

    local vllm_tp="${TP:-1}"
    local vllm_gpu_mem_util="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"

    local offload_args=()
    if [[ -n "$VLLM_OFFLOAD_EXTRA_ARGS" ]]; then
        # shellcheck disable=SC2206
        offload_args=($VLLM_OFFLOAD_EXTRA_ARGS)
    fi

    PYTHONNOUSERSITE=1 vllm serve "$model" --host 0.0.0.0 --port "$port" \
        --config "$config_yaml_path" \
        --gpu-memory-utilization "$vllm_gpu_mem_util" \
        --tensor-parallel-size "$vllm_tp" \
        --max-num-seqs "$vllm_max_num_seqs" \
        "${extra_args[@]}" \
        "${offload_args[@]}" \
        > "$SERVER_LOG" 2>&1 &

    SERVER_PID=$!
    export SERVER_PID
    export SERVER_LOG
}

# launch_sglang_server <model> <port> [extra args...]
# Sets: SERVER_PID, SERVER_LOG
launch_sglang_server() {
    local model="$1"
    local port="$2"
    shift 2 || true
    local extra_args=("$@")

    if [[ -z "$model" || -z "$port" ]]; then
        echo "launch_sglang_server requires: model port" >&2
        return 1
    fi

    hf download "$model"
    if [[ -n "${OFFLOAD_MODE:-}" || "${DISABLE_PREFIX_CACHING:-false}" == "true" ]]; then
        apply_sglang_offload_config
    fi

    SERVER_LOG="${SERVER_LOG:-/workspace/server.log}"

    local sglang_tp="${TP:-1}"
    local sglang_dp="${DP_SIZE:-1}"

    PYTHONNOUSERSITE=1 python3 -m sglang.launch_server \
        --model-path "$model" \
        --host 0.0.0.0 \
        --port "$port" \
        --tensor-parallel-size "$sglang_tp" \
        --data-parallel-size "$sglang_dp" \
        "${extra_args[@]}" \
        > "$SERVER_LOG" 2>&1 &

    SERVER_PID=$!
    export SERVER_PID
    export SERVER_LOG
}

start_kv_metrics_collector() {
    local port="${1:-8888}"
    local output="${2:-$KV_METRICS_CSV}"
    local interval="${3:-2.0}"
    local collector_script

    collector_script="$(cd "$(dirname "${BASH_SOURCE[0]}")/../datasets/isb1/scripts" && pwd)/metrics_collector.py"

    if [[ ! -f "$collector_script" ]]; then
        echo "[KV Metrics] Collector script not found at $collector_script, skipping"
        return 0
    fi

    if [[ -n "$KV_METRICS_PID" ]] && kill -0 "$KV_METRICS_PID" 2>/dev/null; then
        echo "[KV Metrics] Collector already running (PID=$KV_METRICS_PID)"
        return 0
    fi

    KV_METRICS_CSV="$output"
    python3 "$collector_script" \
        --metrics-url "http://0.0.0.0:${port}/metrics" \
        --output "$output" \
        --interval "$interval" >/tmp/kv_metrics_collector.log 2>&1 &
    KV_METRICS_PID=$!

    echo "[KV Metrics] Started (PID=$KV_METRICS_PID, interval=${interval}s, output=$output)"
}

stop_kv_metrics_collector() {
    if [[ -n "$KV_METRICS_PID" ]] && kill -0 "$KV_METRICS_PID" 2>/dev/null; then
        kill "$KV_METRICS_PID" 2>/dev/null || true
        wait "$KV_METRICS_PID" 2>/dev/null || true
        echo "[KV Metrics] Stopped (PID=$KV_METRICS_PID)"
        if [[ -f "$KV_METRICS_CSV" ]]; then
            local lines
            lines=$(wc -l < "$KV_METRICS_CSV")
            echo "[KV Metrics] Collected $lines rows -> $KV_METRICS_CSV"
        fi
    fi
    KV_METRICS_PID=""
}

# Check if required environment variables are set
# Usage: check_env_vars VAR1 VAR2 VAR3 ...
# Exits with code 1 if any variable is not set
check_env_vars() {
    local missing_vars=()

    for var_name in "$@"; do
        if [[ -z "${!var_name}" ]]; then
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

# Wait for server to be ready by polling the health endpoint
# All parameters are required
# Parameters:
#   --port: Server port
#   --server-log: Path to server log file
#   --server-pid: Server process ID (required)
#   --sleep-interval: Sleep interval between health checks (optional, default: 5)
wait_for_server_ready() {
    set +x
    local port=""
    local server_log=""
    local server_pid=""
    local sleep_interval=5

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --port)
                port="$2"
                shift 2
                ;;
            --server-log)
                server_log="$2"
                shift 2
                ;;
            --server-pid)
                server_pid="$2"
                shift 2
                ;;
            --sleep-interval)
                sleep_interval="$2"
                shift 2
                ;;
            *)
                echo "Unknown parameter: $1"
                return 1
                ;;
        esac
    done

    # Validate required parameters
    if [[ -z "$port" ]]; then
        echo "Error: --port is required"
        return 1
    fi
    if [[ -z "$server_log" ]]; then
        echo "Error: --server-log is required"
        return 1
    fi
    if [[ -z "$server_pid" ]]; then
        echo "Error: --server-pid is required"
        return 1
    fi

    # Wait for server log file to be created (container startup may delay this)
    while [ ! -f "$server_log" ]; do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "Server died before creating log file. Exiting."
            exit 1
        fi
        sleep 1
    done

    # Show logs until server is ready
    tail -f -n +1 "$server_log" &
    local TAIL_PID=$!
    until curl --output /dev/null --silent --fail http://0.0.0.0:$port/health; do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "Server died before becoming healthy. Exiting."
            kill $TAIL_PID
            exit 1
        fi
        sleep "$sleep_interval"
    done
    kill $TAIL_PID
}

# Run benchmark serving with standardized parameters
# All parameters are required except --use-chat-template and --trust-remote-code
# Parameters:
#   --model: Model name
#   --port: Server port
#   --backend: Backend type - e.g., 'vllm' or 'openai'
#   --input-len: Random input sequence length
#   --output-len: Random output sequence length
#   --random-range-ratio: Random range ratio
#   --num-prompts: Number of prompts
#   --max-concurrency: Max concurrency
#   --result-filename: Result filename without extension
#   --result-dir: Result directory
#   --use-chat-template: Optional flag to enable chat template
#   --trust-remote-code: Optional flag to trust remote code from HuggingFace
#   --server-pid: Optional server process ID to monitor during benchmark
run_benchmark_serving() {
    # In eval-only mode, skip the throughput benchmark entirely.
    if [ "${EVAL_ONLY}" = "true" ]; then
        echo "EVAL_ONLY mode: skipping throughput benchmark"
        return 0
    fi

    set +x
    local model=""
    local port=""
    local backend=""
    local input_len=""
    local output_len=""
    local random_range_ratio=""
    local num_prompts=""
    local max_concurrency=""
    local result_filename=""
    local result_dir=""
    local workspace_dir=""
    local use_chat_template=false
    local trust_remote_code=false
    local server_pid=""

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
            --trust-remote-code)
                trust_remote_code=true
                shift
                ;;
            --server-pid)
                server_pid="$2"
                shift 2
                ;;
            *)
                echo "Unknown parameter: $1"
                return 1
                ;;
        esac
    done
    
    # Validate all required parameters
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

    # Profiling support: when PROFILE=1, ensure profiler dir exists, add --profile flag,
    # and cap num_prompts to keep traces small.
    local profile_flag=()
    if [[ "${PROFILE:-}" == "1" ]]; then
        local _prof_dir="${SGLANG_TORCH_PROFILER_DIR:-${VLLM_TORCH_PROFILER_DIR:-}}"
        if [[ -n "$_prof_dir" ]]; then
            mkdir -p "$_prof_dir"
        fi
        profile_flag+=(--profile)
        num_prompts="$max_concurrency"
    fi

    # Build benchmark command
    local benchmark_cmd=(
        python3 "$workspace_dir/utils/bench_serving/benchmark_serving.py"
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
    
    # Add --use-chat-template if requested
    if [[ "$use_chat_template" == true ]]; then
        benchmark_cmd+=(--use-chat-template)
    fi

    # Add --trust-remote-code if requested
    if [[ "$trust_remote_code" == true ]]; then
        benchmark_cmd+=(--trust-remote-code)
    fi

    # Run benchmark with optional server monitoring
    set -x
    if [[ -n "$server_pid" ]]; then
        # Run benchmark in background and monitor server health
        "${benchmark_cmd[@]}" &
        local benchmark_pid=$!

        # Monitor loop: check both benchmark and server status
        while kill -0 "$benchmark_pid" 2>/dev/null; do
            if ! kill -0 "$server_pid" 2>/dev/null; then
                echo "ERROR: Server process $server_pid died during benchmark"
                kill "$benchmark_pid" 2>/dev/null
                wait "$benchmark_pid" 2>/dev/null
                set +x
                return 1
            fi
            sleep 2
        done

        # Benchmark finished, get its exit code
        wait "$benchmark_pid"
        local benchmark_exit_code=$?
    else
        # No server monitoring, run benchmark directly
        "${benchmark_cmd[@]}"
        local benchmark_exit_code=$?
    fi
    set +x

    # If profiling, move trace to relay-upload location
    if [[ "${PROFILE:-}" == "1" ]]; then
        move_profile_trace_for_relay
    fi

    return $benchmark_exit_code
}

is_isb1_replay_benchmark() {
    [[ "${BENCHMARK_TYPE:-}" == "isb1_replay" ]]
}

is_isb1_kv_stress_benchmark() {
    [[ "${BENCHMARK_TYPE:-}" == "isb1_kv_stress" ]]
}

resolve_replay_request_mode_for_harness() {
    local requested_mode="${1:-auto}"

    case "$requested_mode" in
        ""|auto|chat|completions)
            printf '%s' "${requested_mode:-auto}"
            ;;
        multi-turn|multi_turn|multiturn)
            printf 'auto'
            ;;
        *)
            echo "WARN: Unsupported replay request mode '$requested_mode'; using 'auto' for the harness boundary" >&2
            printf 'auto'
            ;;
    esac
}

run_isb1_kv_stress_campaign_cell() {
    check_env_vars \
        BENCHMARK_TYPE \
        EXPORT_FILE \
        MAX_CONCURRENCY \
        OFFLOAD_MODE \
        BENCHMARK_DURATION_S \
        KV_CACHE_DTYPE \
        WORKLOAD_TYPE

    if ! is_isb1_kv_stress_benchmark; then
        echo "Error: run_isb1_kv_stress_campaign_cell called with BENCHMARK_TYPE='${BENCHMARK_TYPE:-}'" >&2
        return 1
    fi

    local port="${PORT:-8888}"
    local kv_metrics_output="/workspace/kv_metrics.csv"
    local metadata_path="/workspace/kv_stress_campaign_metadata.json"
    local replay_exit_code=0

    start_gpu_monitor
    start_kv_metrics_collector "$port" "$kv_metrics_output" 2.0

    run_benchmark_export_replay "$@" || replay_exit_code=$?

    stop_kv_metrics_collector
    stop_gpu_monitor

    python3 - <<'PY'
import json
import os
import time

metadata = {
    "benchmark_type": os.getenv("BENCHMARK_TYPE", ""),
    "export_file": os.getenv("EXPORT_FILE", ""),
    "runtime_stack_id": os.getenv("RUNTIME_STACK_ID", ""),
    "hardware_profile_id": os.getenv("HARDWARE_PROFILE_ID", ""),
    "canonical_model_id": os.getenv("CANONICAL_MODEL_ID", ""),
    "request_mode": os.getenv("REQUEST_MODE", ""),
    "max_concurrency": os.getenv("MAX_CONCURRENCY", ""),
    "offload_mode": os.getenv("OFFLOAD_MODE", ""),
    "disable_prefix_caching": os.getenv("DISABLE_PREFIX_CACHING", ""),
    "kv_cache_dtype": os.getenv("KV_CACHE_DTYPE", ""),
    "benchmark_duration_s": os.getenv("BENCHMARK_DURATION_S", ""),
    "workload_type": os.getenv("WORKLOAD_TYPE", ""),
    "metrics_files": {
        "gpu": "/workspace/gpu_metrics.csv",
        "kv": "/workspace/kv_metrics.csv",
    },
    "captured_at_epoch_s": int(time.time()),
}
with open("/workspace/kv_stress_campaign_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, sort_keys=True)
PY

    echo "[KV Stress] Campaign metadata written to $metadata_path"
    return "$replay_exit_code"
}

run_single_node_benchmark() {
    if ! is_isb1_replay_benchmark && ! is_isb1_kv_stress_benchmark; then
        run_benchmark_serving "$@"
        return $?
    fi

    set +x
    local model=""
    local port=""
    local result_filename=""
    local result_dir=""
    local workspace_dir=""
    local trust_remote_code=false
    local server_pid=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            --model)               model="$2";           shift 2 ;;
            --port)                port="$2";            shift 2 ;;
            --result-filename)     result_filename="$2"; shift 2 ;;
            --result-dir)          result_dir="$2";      shift 2 ;;
            --bench-serving-dir)   workspace_dir="$2";   shift 2 ;;
            --trust-remote-code)   trust_remote_code=true; shift ;;
            --server-pid)          server_pid="$2";      shift 2 ;;
            --backend|--input-len|--output-len|--random-range-ratio|--num-prompts|--max-concurrency)
                shift 2
                ;;
            --use-chat-template)
                shift
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
    if [[ -z "$result_filename" ]]; then
        echo "Error: --result-filename is required"
        return 1
    fi
    if [[ -z "$result_dir" ]]; then
        echo "Error: --result-dir is required"
        return 1
    fi

    local replay_args=(
        --model "$model"
        --port "$port"
        --export-file "${EXPORT_FILE}"
        --runtime-stack-id "${RUNTIME_STACK_ID}"
        --hardware-profile-id "${HARDWARE_PROFILE_ID}"
        --canonical-model-id "${CANONICAL_MODEL_ID}"
        --request-mode "${REQUEST_MODE:-auto}"
        --max-concurrency "${MAX_CONCURRENCY}"
        --num-warmup-sessions "${NUM_WARMUP_SESSIONS:-0}"
        --result-filename "$result_filename"
        --result-dir "$result_dir"
    )

    if [[ -n "$workspace_dir" ]]; then
        replay_args+=(--bench-serving-dir "$workspace_dir")
    fi
    if [[ -n "${MAX_SESSIONS:-}" ]]; then
        replay_args+=(--max-sessions "${MAX_SESSIONS}")
    fi
    if [[ -n "${SUPPORT_STATUS:-}" ]]; then
        replay_args+=(--support-status "${SUPPORT_STATUS}")
    fi
    if [[ -n "${MAX_TURNS_PER_SESSION:-}" ]]; then
        replay_args+=(--max-turns-per-session "${MAX_TURNS_PER_SESSION}")
    fi
    if [[ -n "${MAX_OUTPUT_LEN:-}" ]]; then
        replay_args+=(--max-output-len "${MAX_OUTPUT_LEN}")
    fi
    if [[ "${IGNORE_WAITS:-false}" == "true" ]]; then
        replay_args+=(--ignore-waits)
    fi
    if [[ "${IGNORE_EOS:-false}" == "true" ]]; then
        replay_args+=(--ignore-eos)
    fi
    if [[ "$trust_remote_code" == true ]]; then
        replay_args+=(--trust-remote-code)
    fi
    if [[ -n "$server_pid" ]]; then
        replay_args+=(--server-pid "$server_pid")
    fi

    if is_isb1_kv_stress_benchmark; then
        run_isb1_kv_stress_campaign_cell "${replay_args[@]}"
    else
        run_benchmark_export_replay "${replay_args[@]}"
    fi
}


# --------------------------------
# Profiling trace helpers
# --------------------------------

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

    local sglang_dir="${SGLANG_TORCH_PROFILER_DIR:-/workspace}"
    local vllm_dir="${VLLM_TORCH_PROFILER_DIR:-/workspace}"
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


# ------------------------------
# Eval (lm-eval-harness) helpers
# ------------------------------

_install_lm_eval_deps() {
    # torchvision causes circular imports in ATOM; TRT-LLM/SGLang need it at module level.
    if [[ "${IMAGE:-}" == *atom* ]]; then
        python3 -m pip uninstall -y torchvision 2>/dev/null || true
    fi
    python3 -m pip install -q --no-cache-dir --break-system-packages "lm-eval[api]" || true
    local lm_eval_ref="b315ef3b05176acc9732bb7fdec116abe1ecc476"
    if command -v git >/dev/null 2>&1; then
        if ! python3 -m pip install -q --no-cache-dir --no-deps --break-system-packages \
            "git+https://github.com/EleutherAI/lm-evaluation-harness.git@${lm_eval_ref}"; then
            python3 -m pip install -q --no-cache-dir --no-deps --break-system-packages \
                "https://github.com/EleutherAI/lm-evaluation-harness/archive/${lm_eval_ref}.tar.gz" || true
        fi
    else
        python3 -m pip install -q --no-cache-dir --no-deps --break-system-packages \
            "https://github.com/EleutherAI/lm-evaluation-harness/archive/${lm_eval_ref}.tar.gz" || true
    fi
}

# Patch lm-eval filters to be robust to empty strings via sitecustomize
_patch_lm_eval() {
    local patch_dir
    patch_dir="$(mktemp -d)"
    cat > "$patch_dir/sitecustomize.py" <<'PY'
# --- Patch LocalChatCompletion.parse_generations to handle empty content with reasoning_content ---
import re, sys, unicodedata, json
from lm_eval.filters import extraction as ex
from lm_eval.models.openai_completions import LocalChatCompletion as _LCC

def _le_parse_generations(outputs, **kwargs):
      res = []
      if not isinstance(outputs, list):
          outputs = [outputs]
      for out in (outputs or []):
          try:
              choices = out.get("choices", [])
              tmp = ["" for _ in choices]
              for choice in choices:
                  idx = choice.get("index", 0)
                  msg = (choice.get("message") or {})
                  content = msg.get("content")
                  if content in (None, "", []):
                      content = msg.get("reasoning_content") or ""
                  tmp[idx] = content
          except Exception:
              tmp = [""]
          res.extend(tmp)
      return res

# Keep staticmethod semantics
_LCC.parse_generations = staticmethod(_le_parse_generations)

# --- Patch TemplateAPI.apply_chat_template to avoid injecting "type": "text" for TRT ---
try:
    from lm_eval.models import api_models as _api_models
    _TemplateAPI = _api_models.TemplateAPI
    _JsonChatStr = _api_models.JsonChatStr
except Exception:
    _TemplateAPI = None
    _JsonChatStr = None

if _TemplateAPI is not None and _JsonChatStr is not None:
    _orig_apply_chat_template = _TemplateAPI.apply_chat_template

    def _patched_apply_chat_template(
        self,
        chat_history,
        add_generation_prompt: bool = True,
    ):
        """Applies a chat template to a list of chat history between user and model."""
        if self.tokenizer_backend == "huggingface" and self.tokenized_requests:
            return self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
            )
        elif self.tokenizer_backend == "remote" and self.tokenized_requests:
            return chat_history
        else:
            # NOTE: we no longer inject `"type": "text"` when tokenizer is None / non-HF
            return _JsonChatStr(
                json.dumps(
                    [{**item} for item in chat_history],
                    ensure_ascii=False,
                )
            )

    _TemplateAPI.apply_chat_template = _patched_apply_chat_template
PY
    export PYTHONPATH="${patch_dir}:${PYTHONPATH:-}"
}

get_native_max_context_length() {
    local model_path="$1"
    python3 -c "
from transformers import AutoConfig
config = AutoConfig.from_pretrained('${model_path}', trust_remote_code=True)
for attr in ['max_position_embeddings', 'max_sequence_length', 'seq_length', 'n_positions']:
    if hasattr(config, attr):
        print(getattr(config, attr))
        break
else:
    print(0)
"
}

# Compute the context length for eval-only mode.
# Uses 5x the benchmark context capped at the model's native max.
# Sets EVAL_MAX_MODEL_LEN (needed by run_lm_eval).
# Echoes the computed value for scripts to capture.
#
# Usage: local ctx=$(compute_eval_context_length "$MODEL" "${current_ctx}")
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
    # If eval_ctx is still 0 (both benchmark_ctx and native_max were 0), fall back
    if [ "$eval_ctx" -le 0 ] 2>/dev/null; then
        echo "WARN: compute_eval_context_length could not determine context length for $model" >&2
        eval_ctx="${MAX_MODEL_LEN:-16384}"
    fi
    EVAL_MAX_MODEL_LEN="$eval_ctx"
    echo "$eval_ctx"
}

# Convenience wrapper: compute eval context from ISL/OSL and export EVAL_MAX_MODEL_LEN.
# Call directly (not in a subshell) so the export persists.
# Scripts then wire $EVAL_MAX_MODEL_LEN into whichever server variable they need.
setup_eval_context() {
    EVAL_MAX_MODEL_LEN=$(compute_eval_context_length "$MODEL" "$((ISL + OSL + 200))")
    export EVAL_MAX_MODEL_LEN
}

run_lm_eval() {
    local port="${PORT:-8888}"
    local tasks_dir="${EVAL_TASKS_DIR:-utils/evals/gsm8k.yaml}"
    local results_dir="${EVAL_RESULT_DIR:-$(mktemp -d /tmp/eval_out-XXXXXX)}"
    local eval_context_len="${EVAL_MAX_MODEL_LEN:-16384}"
    local temperature=0
    local top_p=1
    local concurrent_requests="${EVAL_CONCURRENT_REQUESTS:-64}"

    while [[ $# -gt 0 ]]; do
        case $1 in
            --port)           port="$2"; shift 2 ;;
            --task)           tasks_dir="$2"; shift 2 ;;
            --results-dir)    results_dir="$2"; shift 2 ;;
            --gen-max-tokens) eval_context_len="$2"; shift 2 ;;
            --temperature)    temperature="$2"; shift 2 ;;
            --top-p)          top_p="$2"; shift 2 ;;
            *)                echo "Unknown parameter: $1"; return 1 ;;
        esac
    done

    _install_lm_eval_deps
    _patch_lm_eval

    local openai_server_base="http://0.0.0.0:${port}"
    local openai_chat_base="${openai_server_base}/v1/chat/completions"
    export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}
    MODEL_NAME=${MODEL_NAME:-$MODEL} # Prefer MODEL_NAME, else MODEL

    # Cap output tokens: must fit within context window (leave room for input),
    # and avoid excessive KV cache reservation per request on TRT.
    local max_output_tokens=$(( eval_context_len > 4096 ? eval_context_len - 4096 : eval_context_len / 2 ))
    if [ "$max_output_tokens" -gt 16384 ]; then
        max_output_tokens=16384
    fi
    echo "Eval budget: eval_context_len=${eval_context_len}, max_output_tokens=${max_output_tokens}"

    # Export for append_lm_eval_summary to pick up
    export EVAL_RESULT_DIR="$results_dir"
    set -x
    python3 -m lm_eval --model local-chat-completions --apply_chat_template \
      --tasks "${tasks_dir}" \
      --output_path "${results_dir}" \
      --log_samples \
      --model_args "model=${MODEL_NAME},base_url=${openai_chat_base},api_key=${OPENAI_API_KEY},eos_string=</s>,max_retries=5,num_concurrent=${concurrent_requests},timeout=1800,tokenized_requests=False,max_length=${eval_context_len}" \
      --gen_kwargs "max_tokens=${max_output_tokens},temperature=${temperature},top_p=${top_p}"
    local eval_exit=$?
    set +x
    return $eval_exit
}

append_lm_eval_summary() {
    local results_dir="${EVAL_RESULT_DIR}"
    if [ -z "${results_dir}" ]; then
        echo "WARN: EVAL_RESULT_DIR is empty; skipping artifact collection" >&2
        return 1
    fi
    local out_dir="${results_dir}"
    if [ ! -d "${out_dir}" ]; then
        echo "WARN: EVAL_RESULT_DIR='${out_dir}' does not exist; skipping artifact collection" >&2
        return 1
    fi

    # Write minimal meta for collectors that expect it
    local meta_json="${out_dir}/meta_env.json"
    local model_name="${MODEL_NAME:-$MODEL}"
    local dp_json="false"
    if [ "${DP_ATTENTION}" = "true" ]; then dp_json="true"; fi

    # Derive framework/precision from env, fallback to parsing RESULT_FILENAME
    # RESULT_FILENAME format (from workflow):
    #   <exp_name>_<precision>_<framework>_tp<...>_ep<...>_dpa_<...>_conc<...>_<runner>
    local fw="${FRAMEWORK:-}"
    local prec="${PRECISION:-}"
    if [[ -z "$fw" || -z "$prec" ]]; then
        if [[ -n "${RESULT_FILENAME}" ]]; then
            # Extract the two fields immediately before "_tp"
            # Handles arbitrary underscores in exp_name by matching from the end
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
    cat > "${meta_json}" <<META
{
  "framework": "${fw:-unknown}",
  "precision": "${prec:-unknown}",
  "spec_decoding": "${SPEC_DECODING}",
  "tp": ${TP:-1},
  "conc": ${CONC:-1},
  "ep": ${EP_SIZE:-1},
  "dp_attention": ${dp_json},
  "model": "${model_name:-}",
  "infmax_model_prefix": "${MODEL_PREFIX:-unknown}",
  "hw": "${RUNNER_TYPE:-unknown}",
  "isl": "${ISL:-0}",
  "osl": "${OSL:-0}"
}
META

    # Move eval artifacts into PWD (no new directories in workspace)
    if [ -f "${meta_json}" ]; then
        mv -f "${meta_json}" ./ || echo "WARN: failed to move ${meta_json}" >&2
    fi
    if [ -d "${out_dir}" ]; then
        while IFS= read -r -d '' jf; do
            base=$(basename "$jf")
            if [ "$base" != "meta_env.json" ]; then
                mv -f "$jf" ./ || echo "WARN: failed to move ${jf}" >&2
            fi
        done < <(find "${out_dir}" -type f -name "*.json*" -print0 2>/dev/null)
    fi

    # Best-effort cleanup of the temp directory
    if [ -n "${out_dir}" ] && [ -d "${out_dir}" ]; then
        rm -rf --one-file-system "${out_dir}" || rm -rf "${out_dir}" || true
    fi

    echo "Moved eval artifacts to: $(pwd)"
}

# ------------------------------
# Unified eval entrypoint
# ------------------------------

run_eval() {
    local framework="${EVAL_FRAMEWORK:-lm-eval}"
    local forwarded=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --framework) framework="$2"; shift 2 ;;
            *)           forwarded+=("$1"); shift ;;
        esac
    done

    # Compute EVAL_MAX_MODEL_LEN if not already set by the calling script
    if [ -z "${EVAL_MAX_MODEL_LEN:-}" ]; then
        compute_eval_context_length "$MODEL" "${MAX_MODEL_LEN:-0}" > /dev/null
    fi

    local eval_rc=0
    case "$framework" in
        lm-eval|lm_eval) run_lm_eval "${forwarded[@]}" || eval_rc=$? ;;
        *)               echo "Unknown framework '${framework}'"; eval_rc=1 ;;
    esac

    if [ "$eval_rc" -ne 0 ]; then
        echo "ERROR: run_eval failed with exit code $eval_rc" >&2
        if [ "${EVAL_ONLY}" = "true" ]; then
            echo "Eval-only mode: failing after artifact collection" >&2
            return "$eval_rc"
        fi
    fi
    return $eval_rc
}


# ---------------------------------------------------------------------------
# Multi-turn benchmark wrapper
# ---------------------------------------------------------------------------

# Run multi-turn chat benchmark with standardized parameters.
# Exercises growing KV cache across conversation turns via /v1/chat/completions.
#
# IMPORTANT: The server MUST be started with prefix/radix caching ENABLED
# for meaningful multi-turn results.  Do NOT use --disable-radix-cache or
# --no-enable-prefix-caching with multi-turn benchmarks.
# Replay ISB1 export sessions/events against a running server.
#
# Supports:
#   - inferencex_multiturn exports via /v1/chat/completions (standalone vLLM/SGLang)
#   - inferencex_trace_replay exports via either chat or projected completions
#     mode (useful for TRT / Dynamo-style cells)
#
# Parameters:
#   --model: Model name sent to the target server
#   --port: Server port
#   --export-file: Path to export JSON
#   --runtime-stack-id: Filter selected export cells to one runtime stack
#   --hardware-profile-id: Filter selected export cells to one hardware row
#   --canonical-model-id: Filter selected export cells to one canonical model row
#   --request-mode: auto|chat|completions (default: auto)
#   --max-concurrency: Max concurrent replay sessions
#   --num-warmup-sessions: Warmup sessions before measurement
#   --result-filename: Result filename without extension
#   --result-dir: Result directory
#   --max-sessions: Optional session limit for smoke runs
#   --max-turns-per-session: Optional turn cap for smoke runs
#   --max-output-len: Optional per-turn output cap
#   --ignore-waits: Ignore inter-turn wait gaps from export metadata
#   --trust-remote-code: Optional flag
#   --server-pid: Optional server process ID to monitor
run_benchmark_export_replay() {
    set +x
    local model=""
    local port=""
    local export_file=""
    local runtime_stack_id=""
    local hardware_profile_id=""
    local canonical_model_id=""
    local trace_id=""
    local support_status=""
    local request_mode="auto"
    local max_concurrency="8"
    local num_warmup_sessions="1"
    local result_filename=""
    local result_dir=""
    local workspace_dir=""
    local max_sessions=""
    local max_turns_per_session=""
    local max_output_len=""
    local ignore_waits=false
    local trust_remote_code=false
    local ignore_eos=false
    local server_pid=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            --model)                  model="$2";                  shift 2 ;;
            --port)                   port="$2";                   shift 2 ;;
            --export-file)            export_file="$2";            shift 2 ;;
            --runtime-stack-id)       runtime_stack_id="$2";       shift 2 ;;
            --hardware-profile-id)    hardware_profile_id="$2";    shift 2 ;;
            --canonical-model-id)     canonical_model_id="$2";     shift 2 ;;
            --trace-id)               trace_id="$2";               shift 2 ;;
            --support-status)         support_status="$2";         shift 2 ;;
            --request-mode)           request_mode="$2";           shift 2 ;;
            --max-concurrency)        max_concurrency="$2";        shift 2 ;;
            --num-warmup-sessions)    num_warmup_sessions="$2";    shift 2 ;;
            --result-filename)        result_filename="$2";        shift 2 ;;
            --result-dir)             result_dir="$2";            shift 2 ;;
            --bench-serving-dir)      workspace_dir="$2";          shift 2 ;;
            --max-sessions)           max_sessions="$2";           shift 2 ;;
            --max-turns-per-session)  max_turns_per_session="$2";  shift 2 ;;
            --max-output-len)         max_output_len="$2";         shift 2 ;;
            --ignore-waits)           ignore_waits=true;           shift   ;;
            --trust-remote-code)      trust_remote_code=true;      shift   ;;
            --ignore-eos)             ignore_eos=true;             shift   ;;
            --server-pid)             server_pid="$2";             shift 2 ;;
            *)                        echo "Unknown parameter: $1"; return 1 ;;
        esac
    done

    if [[ -z "$model" ]]; then echo "Error: --model is required"; return 1; fi
    if [[ -z "$port" ]]; then echo "Error: --port is required"; return 1; fi
    if [[ -z "$export_file" ]]; then echo "Error: --export-file is required"; return 1; fi
    if [[ -z "$result_filename" ]]; then echo "Error: --result-filename is required"; return 1; fi
    if [[ -z "$result_dir" ]]; then echo "Error: --result-dir is required"; return 1; fi

    if [[ -z "$workspace_dir" ]]; then
        workspace_dir=$(pwd)
    fi

    local requested_request_mode="$request_mode"
    local harness_request_mode
    harness_request_mode=$(resolve_replay_request_mode_for_harness "$request_mode")

    local benchmark_cmd=(
        python3 "$workspace_dir/utils/bench_serving/benchmark_export_replay.py"
        --model "$model"
        --base-url "http://0.0.0.0:$port"
        --export-file "$export_file"
        --request-mode "$harness_request_mode"
        --max-concurrency "$max_concurrency"
        --num-warmup-sessions "$num_warmup_sessions"
        --save-result
        --result-dir "$result_dir"
        --result-filename "$result_filename.json"
        --metadata
        "benchmark_type=${BENCHMARK_TYPE:-isb1_replay}"
        "export_file=$export_file"
        "runtime_stack_id=$runtime_stack_id"
        "hardware_profile_id=$hardware_profile_id"
        "canonical_model_id=$canonical_model_id"
        "request_mode=$requested_request_mode"
        "harness_request_mode=$harness_request_mode"
    )

    if [[ -n "${WORKLOAD_TYPE:-}" ]]; then
        benchmark_cmd+=(--metadata "workload_type=${WORKLOAD_TYPE}")
    fi
    if [[ -n "${BENCHMARK_DURATION_S:-}" ]]; then
        benchmark_cmd+=(--metadata "benchmark_duration_s=${BENCHMARK_DURATION_S}")
    fi
    if [[ -n "${OFFLOAD_MODE:-}" ]]; then
        benchmark_cmd+=(--metadata "offload_mode=${OFFLOAD_MODE}")
    fi
    if [[ -n "${KV_CACHE_DTYPE:-}" ]]; then
        benchmark_cmd+=(--metadata "kv_cache_dtype=${KV_CACHE_DTYPE}")
    fi
    if [[ -n "${DISABLE_PREFIX_CACHING:-}" ]]; then
        benchmark_cmd+=(--metadata "disable_prefix_caching=${DISABLE_PREFIX_CACHING}")
    fi

    if [[ -n "${VLLM_CPU_OFFLOAD_GB:-}" ]]; then
        benchmark_cmd+=(--metadata "vllm_cpu_offload_gb=${VLLM_CPU_OFFLOAD_GB}")
    fi
    if [[ -n "${VLLM_SWAP_SPACE_GB:-}" ]]; then
        benchmark_cmd+=(--metadata "vllm_swap_space_gb=${VLLM_SWAP_SPACE_GB}")
    fi
    if [[ -n "${SGLANG_MEM_FRACTION_OVERRIDE:-}" ]]; then
        benchmark_cmd+=(--metadata "sglang_mem_fraction_override=${SGLANG_MEM_FRACTION_OVERRIDE}")
    fi
    if [[ -n "${SGLANG_CHUNKED_PREFILL_OVERRIDE:-}" ]]; then
        benchmark_cmd+=(--metadata "sglang_chunked_prefill_override=${SGLANG_CHUNKED_PREFILL_OVERRIDE}")
    fi

    if [[ -n "$runtime_stack_id" ]]; then
        benchmark_cmd+=(--runtime-stack-id "$runtime_stack_id")
    fi
    if [[ -n "$hardware_profile_id" ]]; then
        benchmark_cmd+=(--hardware-profile-id "$hardware_profile_id")
    fi
    if [[ -n "$canonical_model_id" ]]; then
        benchmark_cmd+=(--canonical-model-id "$canonical_model_id")
    fi
    if [[ -n "$trace_id" ]]; then
        benchmark_cmd+=(--trace-id "$trace_id")
    fi
    if [[ -n "$support_status" ]]; then
        benchmark_cmd+=(--support-status "$support_status")
    fi
    if [[ -n "$max_sessions" ]]; then
        benchmark_cmd+=(--max-sessions "$max_sessions")
    fi
    if [[ -n "$max_turns_per_session" ]]; then
        benchmark_cmd+=(--max-turns-per-session "$max_turns_per_session")
    fi
    if [[ -n "$max_output_len" ]]; then
        benchmark_cmd+=(--max-output-len "$max_output_len")
    fi
    if [[ "$ignore_waits" == true ]]; then
        benchmark_cmd+=(--ignore-waits)
    fi
    if [[ "$trust_remote_code" == true ]]; then
        benchmark_cmd+=(--trust-remote-code)
    fi
    if [[ "$ignore_eos" == true ]]; then
        benchmark_cmd+=(--ignore-eos)
    fi

    set -x
    if [[ -n "$server_pid" ]]; then
        "${benchmark_cmd[@]}" &
        local benchmark_pid=$!

        while kill -0 "$benchmark_pid" 2>/dev/null; do
            if ! kill -0 "$server_pid" 2>/dev/null; then
                echo "ERROR: Server process $server_pid died during export replay benchmark"
                kill "$benchmark_pid" 2>/dev/null
                wait "$benchmark_pid" 2>/dev/null
                set +x
                return 1
            fi
            sleep 2
        done

        wait "$benchmark_pid"
        local benchmark_exit_code=$?
    else
        "${benchmark_cmd[@]}"
        local benchmark_exit_code=$?
    fi
    set +x

    return $benchmark_exit_code
}
