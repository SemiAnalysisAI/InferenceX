#!/usr/bin/env bash

# Shared benchmarking utilities for InferenceX

# Keep Python bytecode out of the mounted workspace. Benchmark jobs often run as
# root inside containers, and root-owned cache directories break future checkout
# cleanup on self-hosted runners.
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/inferencex-pycache}"
mkdir -p "$PYTHONPYCACHEPREFIX" 2>/dev/null || true

# Inference server port shared by every benchmark recipe. Launchers that need
# a non-default value (e.g. launch_mi355x-amds.sh derives PORT from RUNNER_NAME
# to avoid collisions across concurrent gh-runners on a shared host) set PORT
# themselves before sourcing this file; the `:-` fallback only kicks in when
# nothing upstream set it.
export PORT="${PORT:-8888}"

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

# Agentic replays must use the model's native context limit. Ignore inherited
# workflow or shell overrides so neither the server nor AIPerf applies a cap.
_benchmark_caller="${BASH_SOURCE[1]:-}"
if [[ "$_benchmark_caller" == */agentic/* ||
      "$_benchmark_caller" == */agentic_*.sh ||
      "${IS_AGENTIC:-0}" == "1" ||
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

# --------------------------------
# GPU monitoring helpers
# --------------------------------

GPU_MONITOR_PID=""
GPU_METRICS_CSV="/workspace/gpu_metrics.csv"
export GPU_METRICS_CSV

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
    export GPU_METRICS_CSV

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

# Check if required environment variables are set
# Usage: check_env_vars VAR1 VAR2 VAR3 ...
# Exits with code 1 if any variable is not set
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
# All parameters are required except --endpoint, --use-chat-template, --dsv4, and --trust-remote-code
# Parameters:
#   --model: Model name
#   --port: Server port
#   --backend: Backend type - e.g., 'vllm' or 'openai'
#   --endpoint: Optional API endpoint override
#   --input-len: Random input sequence length
#   --output-len: Random output sequence length
#   --random-range-ratio: Random range ratio
#   --num-prompts: Number of prompts
#   --max-concurrency: Max concurrency
#   --result-filename: Result filename without extension
#   --result-dir: Result directory
#   --use-chat-template: Optional flag to enable chat template
#   --dsv4: Optional flag to use the DeepSeek-V4 chat template
#           (encoding_dsv4.py) instead of the tokenizer's built-in jinja
#           template. Implies --use-chat-template.
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

    if [[ -n "$endpoint" ]]; then
        benchmark_cmd+=(--endpoint "$endpoint")
    fi
    
    # Add --use-chat-template if requested
    if [[ "$use_chat_template" == true ]]; then
        benchmark_cmd+=(--use-chat-template)
    fi

    # Add --dsv4 if requested (requires --use-chat-template, which we
    # auto-enable when --dsv4 is passed in).
    if [[ "$dsv4" == true ]]; then
        benchmark_cmd+=(--dsv4)
    fi

    # Add --trust-remote-code if requested
    if [[ "$trust_remote_code" == true ]]; then
        benchmark_cmd+=(--trust-remote-code)
    fi

    if [[ -n "$tokenizer" ]]; then
        benchmark_cmd+=(--tokenizer "$tokenizer")
    fi

    if [[ -n "$tokenizer_mode" ]]; then
        benchmark_cmd+=(--tokenizer-mode "$tokenizer_mode")
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

_eval_patches_dir() {
    cd "$(dirname "${BASH_SOURCE[0]}")/../utils/evals/patches" && pwd
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

# Compute the context length for eval-only mode.
# Eval-only runs use at least EVAL_MIN_MODEL_LEN, capped at the model's native max.
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

    # Fixed-sequence benchmark windows are often too short for reasoning evals.
    # In eval-only mode, raise the context budget without changing recipe configs
    # or throughput runs.
    local eval_min="${EVAL_MIN_MODEL_LEN:-32768}"
    if [ "${EVAL_ONLY:-false}" = "true" \
        ] && [ "$benchmark_ctx" -lt "$eval_min" ] 2>/dev/null; then
        benchmark_ctx="$eval_min"
    fi

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
    EVAL_MAX_MODEL_LEN=$(compute_eval_context_length "$MODEL" "$((ISL + OSL + 256))")
    export EVAL_MAX_MODEL_LEN
}

run_lm_eval() {
    local port="${PORT:-8888}"
    local tasks_dir="${EVAL_TASKS_DIR:-gsm8k,gpqa_diamond_cot_n_shot,aime26}"
    local results_dir="${EVAL_RESULT_DIR:-$(mktemp -d /tmp/eval_out-XXXXXX)}"
    local eval_context_len="${EVAL_MAX_MODEL_LEN:-16384}"
    local temperature=0
    local top_p=1
    local concurrent_requests="${EVAL_CONCURRENT_REQUESTS:-${CONC:-64}}"
    # Repo-local task YAMLs are discovered through --include_path. Full-dataset
    # runs remain the default; --limit is passed only when EVAL_LIMIT explicitly
    # requests a smaller smoke-test slice.
    local eval_limit="${EVAL_LIMIT:-}"
    local include_path="${EVAL_INCLUDE_PATH:-utils/evals}"

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

    # Serving images may use a different WORKDIR.
    local _repo_root
    _repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    if [[ "$tasks_dir" == *.yaml && "$tasks_dir" != /* \
          && ! -f "$tasks_dir" && -f "$_repo_root/$tasks_dir" ]]; then
        echo "run_lm_eval: anchoring relative task '$tasks_dir' to repo root -> $_repo_root/$tasks_dir"
        tasks_dir="$_repo_root/$tasks_dir"
    fi
    if [[ "$include_path" != /* \
          && ! -d "$include_path" && -d "$_repo_root/$include_path" ]]; then
        echo "run_lm_eval: anchoring relative include path '$include_path' to repo root -> $_repo_root/$include_path"
        include_path="$_repo_root/$include_path"
    fi

    local eval_tasks=()
    IFS=',' read -r -a eval_tasks <<< "$tasks_dir"

    if [ "${INFERENCEX_LM_EVAL_RUNTIME_READY:-false}" != "true" ]; then
        _install_lm_eval_deps
        _patch_lm_eval
        export INFERENCEX_LM_EVAL_RUNTIME_READY=true
    fi

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
      ${include_path:+--include_path "$include_path"} \
      --tasks "${eval_tasks[@]}" \
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

    # Write minimal meta for collectors that expect it
    local model_name="${MODEL_NAME:-$MODEL}"
    local is_multinode_json="false"
    if [ "${IS_MULTINODE:-false}" = "true" ]; then
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

    local dp_json="false"
    if [ "${DP_ATTENTION:-false}" = "true" ]; then dp_json="true"; fi
    local prefill_dp_json="$dp_json"
    if [ "${PREFILL_DP_ATTENTION:-${DP_ATTENTION:-false}}" = "true" ]; then
        prefill_dp_json="true"
    else
        prefill_dp_json="false"
    fi
    local decode_dp_json="$dp_json"
    if [ "${DECODE_DP_ATTENTION:-${DP_ATTENTION:-false}}" = "true" ]; then
        decode_dp_json="true"
    else
        decode_dp_json="false"
    fi

    # Derive framework/precision from env, fallback to parsing RESULT_FILENAME
    # RESULT_FILENAME format (from workflow):
    #   <exp_name>_<precision>_<framework>_tp<...>_ep<...>_dpa_<...>_conc<...>_<runner>
    local fw="${FRAMEWORK:-}"
    local prec="${PRECISION:-}"
    if [[ -z "$fw" || -z "$prec" ]]; then
        if [[ -n "${RESULT_FILENAME:-}" ]]; then
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
  "is_multinode": ${is_multinode_json},
  "framework": "${fw:-unknown}",
  "precision": "${prec:-unknown}",
  "spec_decoding": "${SPEC_DECODING:-}",
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

    if [ -n "$batch_concs" ]; then
        echo "Prepared batched eval artifacts in: $(pwd)"
        return 0
    fi

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


_install_swebench_agent_deps() {
    python3 -m pip install -q --no-cache-dir --break-system-packages \
        'mini-swe-agent==2.4.5' 'swe-rex[modal]==1.4.0' || true
    _patch_swebench_agent || \
        echo "WARN: mini-swe-agent/swe-rex patches failed; sandboxes will leak until runtime_timeout and budget-exhausted instances will submit nothing" >&2
}

_patch_swebench_agent() {
    python3 "$(_eval_patches_dir)/patch_swebench_agent.py"
}

_install_swebench_deps() {
    # Patch anchors depend on SWE-bench 4.1.0.
    python3 -m pip install -q --no-cache-dir --break-system-packages 'swebench==4.1.0' || true
    if [ "${SWEBENCH_USE_MODAL:-false}" = "true" ]; then
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
    if [ "${SWEBENCH_USE_MODAL:-false}" != "true" ]; then return 0; fi
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
        echo "WARN: SWEBENCH_USE_MODAL=true but no ~/.modal.toml and no MODAL_TOKEN_ID/MODAL_TOKEN_SECRET env; Modal scoring will fail credential validation" >&2
    fi
}


_run_swebench_agentic_generation() {
    local gen_dir="$1"; shift
    local port="${PORT:-8888}"
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
step_limit = int(os.environ.get("SWEBENCH_AGENT_STEP_LIMIT", "75"))
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
    local expected="${EVAL_LIMIT:-${SWEBENCH_EXPECTED_INSTANCES:-300}}"
    echo "[swebench-agentic] mini-swe-agent: workers=${SWEBENCH_AGENT_WORKERS:-${CONC:-64}} step_limit=${SWEBENCH_AGENT_STEP_LIMIT:-75} slice=${EVAL_LIMIT:-full} expected=$expected"
    local agen_rc=0
    mini-extra swebench \
        -c "$cfg" \
        --subset lite --split test \
        --environment-class swerex_modal \
        "${slice_args[@]}" \
        -w "${SWEBENCH_AGENT_WORKERS:-${CONC:-64}}" \
        -o "$gen_dir/agent_out" &
    local mini_pid=$!
    # preds.json detects completion despite teardown hangs.
    local preds_file="$gen_dir/agent_out/preds.json"
    local deadline=$(( $(date +%s) + ${SWEBENCH_AGENT_TIMEOUT:-14400} ))
    local grace_until=0
    local killed_after_complete=0
    while kill -0 "$mini_pid" 2>/dev/null; do
        if [ "$(date +%s)" -ge "$deadline" ]; then
            echo "ERROR: generation exceeded SWEBENCH_AGENT_TIMEOUT (${SWEBENCH_AGENT_TIMEOUT:-14400}s); killing mini-extra" >&2
            kill "$mini_pid" 2>/dev/null; sleep 5; kill -9 "$mini_pid" 2>/dev/null
            agen_rc=124
            break
        fi
        local done_count
        done_count=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$preds_file" 2>/dev/null || echo 0)
        if [ "${done_count:-0}" -ge "$expected" ]; then
            if [ "$grace_until" -eq 0 ]; then
                grace_until=$(( $(date +%s) + ${SWEBENCH_AGENT_EXIT_GRACE:-300} ))
                echo "[swebench-agentic] all $expected predictions written; waiting ${SWEBENCH_AGENT_EXIT_GRACE:-300}s for mini-extra to exit"
            elif [ "$(date +%s)" -ge "$grace_until" ]; then
                echo "WARN: mini-extra hung after completing all instances; killing (known hang-on-exit)" >&2
                kill "$mini_pid" 2>/dev/null; sleep 5; kill -9 "$mini_pid" 2>/dev/null
                killed_after_complete=1
                break
            fi
        fi
        sleep "${SWEBENCH_WATCHDOG_POLL:-30}"
    done
    wait "$mini_pid" 2>/dev/null
    local wait_rc=$?
    if [ "$killed_after_complete" -eq 1 ]; then
        agen_rc=0
    elif [ "$agen_rc" -eq 0 ] && [ "$wait_rc" -ne 0 ]; then
        agen_rc=$wait_rc
    fi
    # Isolate sweeps to avoid killing unrelated sandboxes.
    [ "${SWEBENCH_SANDBOX_SWEEP:-1}" = "1" ] && python3 - <<'PYSWEEP' || true
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
    local out_dir="${EVAL_RESULT_DIR:-$(mktemp -d /tmp/eval_out-XXXXXX)}"
    local task_name="${SWEBENCH_TASK_NAME:-swebench_lite}"
    local gen_dir
    gen_dir=$(mktemp -d /tmp/swebench_gen-XXXXXX)

    # Generation and scoring must share a dataset.
    local yaml_path="${EVAL_TASKS_DIR:-utils/evals/${task_name}.yaml}"
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

    local gen_mode="${SWEBENCH_GEN_MODE:-agentic}"
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

    if [ "${SWEBENCH_SKIP_SCORE:-false}" = "true" ]; then
        local skip_rc=0
        python3 utils/evals/swebench_score.py \
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
    if [ "${SWEBENCH_USE_MODAL:-false}" = "true" ]; then modal_args=(--modal); fi
    local itimeout_args=(--instance-timeout "${SWEBENCH_EVAL_TIMEOUT:-900}")
    # Avoid holding the GPU on scoring stalls.
    timeout "${SWEBENCH_SCORE_TIMEOUT:-7200}" \
    python3 utils/evals/swebench_score.py \
        "${score_input[@]}" \
        --out-dir "$out_dir" \
        --model-name "${MODEL_NAME:-$MODEL}" \
        --task-name "$task_name" \
        --dataset-name "$dataset" \
        --max-workers "${SWEBENCH_MAX_WORKERS:-4}" \
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

# ------------------------------
# Unified eval entrypoint
# ------------------------------

run_eval() {
    local cli_framework=""
    local forwarded=()

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
    if [ "${IS_AGENTIC:-0}" = "1" ] || [ "${SCENARIO_TYPE:-}" = "agentic-coding" ]; then
        scenario_default="swebench"
        scenario_is_agentic=1
    fi

    local framework="${EVAL_FRAMEWORK:-${cli_framework:-$scenario_default}}"

    # Compute EVAL_MAX_MODEL_LEN if not already set by the calling script
    if [ -z "${EVAL_MAX_MODEL_LEN:-}" ]; then
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

    local eval_rc=0
    case "$framework" in
        lm-eval|lm_eval) run_lm_eval "${forwarded[@]}" || eval_rc=$? ;;
        swebench)        run_swebench_eval "${forwarded[@]}" || eval_rc=$? ;;
        *)               echo "Unknown framework '${framework}'"; eval_rc=1 ;;
    esac

    # Agentic eval-only recipes have no separate staging step.
    if [ "${EVAL_ONLY:-false}" = "true" ] && [ "$scenario_is_agentic" = "1" ]; then
        append_lm_eval_summary || true
    fi

    if [ "$eval_rc" -ne 0 ]; then
        echo "ERROR: run_eval failed with exit code $eval_rc" >&2
        if [ "${EVAL_ONLY}" = "true" ]; then
            echo "Eval-only mode: failing after artifact collection" >&2
            return "$eval_rc"
        fi
    fi
    return $eval_rc
}


# --------------------------------
# Agentic trace replay helpers (aiperf driver)
# --------------------------------

INFMAX_CONTAINER_WORKSPACE="${INFMAX_CONTAINER_WORKSPACE:-/workspace}"
AGENTIC_DIR="${AGENTIC_DIR:-${INFMAX_CONTAINER_WORKSPACE}/utils/agentic-benchmark}"
AIPERF_DIR="${AIPERF_DIR:-${INFMAX_CONTAINER_WORKSPACE}/utils/aiperf}"
AIPERF_RUNTIME_DIR="${AIPERF_RUNTIME_DIR:-${TMPDIR:-/tmp}/inferencex-agentic-${SLURM_JOB_ID:-$$}}"
AIPERF_VENV="${AIPERF_VENV:-${AIPERF_RUNTIME_DIR}/venv}"
AIPERF_UV_INSTALL_DIR="${AIPERF_UV_INSTALL_DIR:-${AIPERF_RUNTIME_DIR}/uv/bin}"
AIPERF_UV_CACHE_DIR="${AIPERF_UV_CACHE_DIR:-${AIPERF_RUNTIME_DIR}/uv-cache}"
AIPERF_PYTHON="${AIPERF_VENV}/bin/python"
AIPERF_CLI="${AIPERF_VENV}/bin/aiperf"
AIPERF_HF_CLI="${AIPERF_VENV}/bin/hf"
AIPERF_DEPS_READY=0
AIPERF_FAILED_REQUEST_THRESHOLD="${AIPERF_FAILED_REQUEST_THRESHOLD:-0.10}"

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
    if [ "$AIPERF_DEPS_READY" = "1" ]; then
        return
    fi

    # AIPerf must not share site-packages with the inference server. Installing
    # it into vLLM/SGLang's system Python can upgrade FastAPI, Starlette,
    # transformers, or other packages while the server imports from that same
    # environment.
    if ! command -v git >/dev/null 2>&1; then
        apt-get update && apt-get install -y git
    fi

    ensure_agentic_uv
    rm -rf "$AIPERF_VENV"
    mkdir -p "$AIPERF_UV_CACHE_DIR"

    UV_CACHE_DIR="$AIPERF_UV_CACHE_DIR" \
        "$AIPERF_UV_BIN" venv --python "$(command -v python3)" "$AIPERF_VENV"
    UV_CACHE_DIR="$AIPERF_UV_CACHE_DIR" \
        "$AIPERF_UV_BIN" pip install --python "$AIPERF_PYTHON" \
        -r "$AGENTIC_DIR/requirements.txt" \
        -e "$AIPERF_DIR" \
        "datasets>=4.7.0" \
        "huggingface_hub[cli]>=0.25.0" \
        urllib3 \
        requests

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
    # Per-recipe override: set WEKA_LOADER_OVERRIDE to one of the aiperf
    # public-dataset loader names allowed by the inferencex-agentx-mvp
    # scenario. Used by recipes whose servers have non-default context
    # caps (e.g. minimaxm2.5 at max_model_len ~256k can't replay the
    # unfiltered corpus and switches to the 256k-capped variant), or
    # by recipes that want to pin an older corpus generation.
    #
    # Default (no override): the 062126 v7 corpus, selected by the model
    # family's native context length. Models with a 1M-token default context
    # use the unfiltered corpus; shorter-context families use the 256k-capped
    # variant. Any recipe can still pin a specific corpus via
    # WEKA_LOADER_OVERRIDE.
    local default_loader
    case "${MODEL_PREFIX:-}" in
        dsv4*|glm5.2*|minimaxm3*)
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
    # Pre-download the dataset into the shared HF_HUB_CACHE (same mount used
    # for model weights) so subsequent runs read from cache instead of
    # re-downloading every job.
    ensure_hf_cli
    "$AIPERF_HF_CLI" download --repo-type dataset "$dataset"
}

build_replay_cmd() {
    # aiperf invocation for the inferencex-agentx-mvp scenario.
    #
    # Pre-canned assistant replay is the default: recorded assistant responses
    # are used for future prompt construction, and live server responses are
    # discarded. Set AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=1 explicitly
    # to use live-assistant mode, where the loader emits user-only deltas and
    # the worker threads the server's live assistant response back into the
    # session.
    #
    # The scenario plugin locks: --cache-bust first_turn_prefix and
    # --trace-idle-gap-cap-seconds 10 (per-trace idle-gap compression
    # against parent + subagent request-start timestamps; supersedes the
    # legacy --use-think-time-only / --inter-turn-delay-cap-seconds path),
    # and auto-injects them — so we do not pass them. See
    # utils/aiperf/docs/tutorials/agentx-mvp.md.
    local result_dir="$1"
    local duration="$DURATION"

    export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES="${AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES:-0}"
    # Dataset configuration (load + reconstruct + inputs.json + mmap)
    # routinely takes 4-5 min for the Weka corpus on fast /tmp
    # (B300) but can stretch to 14 min on slower /tmp + parallel contention
    # (observed on H200 where all 14 R3 jobs hit aiperf's 900s Configure
    # Profiling timeout simultaneously). Bump to 1800s to absorb 3x
    # worst-case slowdown — the post-setup measurement window is unaffected.
    export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
    # aiperf validates that SERVICE_PROFILE_CONFIGURE_TIMEOUT >=
    # DATASET_CONFIGURATION_TIMEOUT at startup. Bump it in lockstep.
    export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
    REPLAY_CMD="$AIPERF_CLI profile --scenario inferencex-agentx-mvp"
    REPLAY_CMD+=" --url http://localhost:$PORT"
    REPLAY_CMD+=" --endpoint /v1/chat/completions"
    REPLAY_CMD+=" --endpoint-type chat"
    REPLAY_CMD+=" --streaming"
    REPLAY_CMD+=" --model $MODEL"
    REPLAY_CMD+=" --concurrency $CONC"
    REPLAY_CMD+=" --benchmark-duration $duration"
    REPLAY_CMD+=" --random-seed 42"
    # Fail runs once more than 10% of requests error. This keeps known
    # transient low-rate failures from killing long sweeps while still
    # catching malformed payloads or server crashes before they get aggregated
    # as benchmarkable data.
    REPLAY_CMD+=" --failed-request-threshold $AIPERF_FAILED_REQUEST_THRESHOLD"
    # Sample each trajectory's warmup start position uniformly from
    # [25%, 75%] of the trace's turn count, clamped by AIPerf to leave at
    # least one profile turn after warmup.
    REPLAY_CMD+=" --trajectory-start-min-ratio 0.25"
    REPLAY_CMD+=" --trajectory-start-max-ratio 0.75"
    # After the normal t* snapshot warmup, continue those exact trajectories
    # with one-token outputs and no idle delays for 10 minutes. Profiling begins
    # only after those requests drain and resumes from the resulting live state.
    REPLAY_CMD+=" --agentic-cache-warmup-duration 600"
    # Give long-context warmup requests up to 30 minutes to drain before
    # declaring warmup failed. Recipes whose saturation arms carry a larger
    # in-flight working set may override via AGENTIC_WARMUP_GRACE_PERIOD
    # (grace is a maximum wait, not a fixed sleep — drain exits when done).
    # cancelling any remaining requests and starting profiling.
    REPLAY_CMD+=" --warmup-grace-period ${AGENTIC_WARMUP_GRACE_PERIOD:-1800}"
    # Use server-reported usage fields (prompt_tokens / completion_tokens) for
    # ISL/OSL instead of client-side tokenizer.encode(). Auto-enables
    # stream_options.include_usage on the OpenAI chat endpoint. Skips the
    # heavy per-record tokenization in the records pipeline that was pinning
    # CPU on minimax-m2.5 at high concurrency. Lossless for vLLM (server
    # usage is authoritative).
    REPLAY_CMD+=" --use-server-token-count"
    # Dynamo's KV router needs an explicit conversation session binding to
    # keep later turns on the prefill worker that owns their prefix blocks.
    # X-Correlation-ID is useful tracing metadata but does not establish that
    # binding by itself. AIPerf emits nvext.session_control bind/close actions
    # keyed by the stable conversation correlation ID when this flag is set.
    if [[ "${FRAMEWORK:-}" == dynamo-* ]]; then
        REPLAY_CMD+=" --use-dynamo-conv-aware-routing"
        # The upstream 300s affinity TTL is shorter than an overloaded
        # high-concurrency agentic request. Keep bindings alive across long
        # prefills, generation, and capped inter-turn delay. This controls the
        # router's inactivity lease; it does not relax HTTP/request failures.
        REPLAY_CMD+=" --dynamo-session-timeout-seconds ${AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS:-3600}"
    fi
    # Disable DCGM GPU telemetry collection. aiperf's GpuMetricTimeSeries
    # freezes its metric schema on the first DCGM scrape, then KeyErrors when
    # an optional field (xid_errors, power_violation, encoder_utilization)
    # first appears mid-run. We don't consume the gpu_telemetry artifact in
    # downstream processing, and the server-metrics path (Prometheus /metrics
    # from vLLM) is unaffected by this flag and still gives us KV usage,
    # prefix cache hit rate, etc.
    REPLAY_CMD+=" --no-gpu-telemetry"
    # aiperf's dataset manager (separate from the inference parser) loads
    # the model's tokenizer for trace-prompt tokenization regardless of
    # --use-server-token-count. Models like kimi (amd/Kimi-K2.5-MXFP4,
    # moonshotai/Kimi-K2.5) ship a custom tokenizer in their HF repo and
    # need trust_remote_code=True to load. Benign for models without
    # custom tokenizer code, so we set it unconditionally.
    REPLAY_CMD+=" --tokenizer-trust-remote-code"
    # Keep replay inputs inside the same context window used to launch the
    # server. The WEKA corpus contains a few very long parent/subagent traces;
    # if we mmap and replay them against a smaller-context server they become
    # deterministic 4xxs and can still pressure the engine while queued.
    if [ -n "${MAX_MODEL_LEN:-}" ] && [ "$MAX_MODEL_LEN" != "0" ]; then
        REPLAY_CMD+=" --max-context-length $MAX_MODEL_LEN"
    fi
    # Default --num-dataset-entries is 100; the with-subagents Weka corpus
    # has 393. Cap at 393 so all unique traces are loaded (the loader treats
    # this as a ``min(cap, available)`` ceiling, not a target — see
    # semianalysis_cc_traces_weka.py).
    REPLAY_CMD+=" --num-dataset-entries 393"
    # 1-second timeslices on the server-metrics scrape so the post-run
    # plotter has per-window time series (KV usage, cache hit rate,
    # throughput, etc.). Matches kv-cache-tester's poll_interval=1.0
    # snapshot cadence so metrics_plots.png is visually comparable.
    # Without this, aiperf only emits aggregate stats and the 6x2 panels
    # collapse to flat lines.
    REPLAY_CMD+=" --slice-duration 1.0"
    # Multi-node launchers can provide the Prometheus endpoints for every
    # inference worker as a comma-separated list. AIPerf accepts multiple
    # values after one --server-metrics flag and preserves endpoint_url on
    # every exported series. The inference frontend's automatically detected
    # /metrics endpoint remains enabled as well.
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
    # The inferencex-agentx-mvp scenario enforces a 900s minimum
    # benchmark duration. For smoke tests with shorter durations, opt
    # into --unsafe-override (the run's submission_valid will be flagged
    # false; that's expected for non-canonical runs).
    if [ "$duration" -lt 900 ] || [ "${AIPERF_UNSAFE_OVERRIDE:-false}" = "true" ]; then
        REPLAY_CMD+=" --unsafe-override"
    fi
    REPLAY_CMD+=" $TRACE_SOURCE_FLAG"
}

write_agentic_result_json() {
    # Aggregate aiperf's profile_export.{json,jsonl} + server_metrics_export.json
    # into $AGENTIC_OUTPUT_DIR/$RESULT_FILENAME.json. The workflow checks that
    # this file exists; run_agentic_replay_and_write_outputs separately rejects
    # aggregates whose request error rate exceeds the configured limit.
    local result_dir="$1"
    (
        cd "$INFMAX_CONTAINER_WORKSPACE"
        RESULT_DIR="$result_dir" AGENTIC_OUTPUT_DIR="${AGENTIC_OUTPUT_DIR:-$INFMAX_CONTAINER_WORKSPACE}" \
            "$AIPERF_PYTHON" -m utils.agentic.aggregation.process_agentic_result
    )

    # Generate metrics_plots.png from the same aiperf artifacts. Best-effort:
    # don't fail the launcher if plot generation has trouble (e.g. matplotlib
    # missing in a stripped-down image). The agg JSON is the success gate.
    "$AIPERF_PYTHON" "$INFMAX_CONTAINER_WORKSPACE/utils/generate_aiperf_plots.py" "$result_dir" 2>&1 || true
}

run_agentic_replay_and_write_outputs() {
    local result_dir="$1"
    local replay_rc
    local validation_rc

    echo "$REPLAY_CMD" > "$result_dir/benchmark_command.txt"

    set +e
    set -x
    $REPLAY_CMD 2>&1 | tee "$result_dir/benchmark.log"
    replay_rc=${PIPESTATUS[0]}
    set +x
    set -e

    write_agentic_result_json "$result_dir"

    "$AIPERF_PYTHON" "$AGENTIC_DIR/scripts/analyze_benchmark_distributions.py" \
        "$result_dir/aiperf_artifacts" -o "$result_dir" 2>&1 || true

    set +e
    (
        cd "$INFMAX_CONTAINER_WORKSPACE"
        "$AIPERF_PYTHON" -m utils.agentic.validation.validate_agentic_result \
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
}
