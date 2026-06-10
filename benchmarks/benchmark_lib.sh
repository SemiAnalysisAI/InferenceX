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
# Uses the requested benchmark context capped at the model's native max.
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
    EVAL_MAX_MODEL_LEN=$(compute_eval_context_length "$MODEL" "$((ISL + OSL + 256))")
    export EVAL_MAX_MODEL_LEN
}

# ------------------------------
# SpeedBench acceptance-length eval helpers
# ------------------------------

_prometheus_metric_values_from_text() {
    local name="$1"
    awk -v name="$name" '
        /^#/ { next }
        {
            metric = $1
            sub(/\{.*/, "", metric)
            if (metric == name && $NF ~ /^-?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][-+]?[0-9]+)?$/) {
                print $NF
                found = 1
            }
        }
        END {
            if (!found) {
                exit 1
            }
        }
    '
}

_prometheus_metric_values_url() {
    local url="$1"
    local name="$2"
    local metrics
    metrics=$(curl -fsS --max-time "${SPEEDBENCH_METRICS_CURL_TIMEOUT:-10}" "$url" 2>/dev/null) || return 1
    _prometheus_metric_values_from_text "$name" <<< "$metrics"
}

_prometheus_metric_sum_url() {
    local url="$1"
    local name="$2"
    local values
    values=$(_prometheus_metric_values_url "$url" "$name") || return 1
    awk '
        { sum += $1; found = 1 }
        END {
            if (!found) {
                exit 1
            }
            printf "%.10f\n", sum
        }
    ' <<< "$values"
}

_prometheus_metric_avg_url() {
    local url="$1"
    local name="$2"
    local values
    values=$(_prometheus_metric_values_url "$url" "$name") || return 1
    awk '
        { sum += $1; count += 1 }
        END {
            if (count == 0) {
                exit 1
            }
            printf "%.10f\n", sum / count
        }
    ' <<< "$values"
}

_prometheus_metric_sum() {
    local port="$1"
    local name="$2"
    _prometheus_metric_sum_url "http://0.0.0.0:${port}/metrics" "$name"
}

_speedbench_normalize_metrics_url() {
    local endpoint="$1"
    endpoint="${endpoint%,}"
    endpoint="${endpoint%/}"
    [[ -z "$endpoint" ]] && return 0

    if [[ "$endpoint" =~ ^https?:// ]]; then
        if [[ "$endpoint" == */metrics || "$endpoint" == */metrics\?* ]]; then
            echo "$endpoint"
        else
            echo "${endpoint}/metrics"
        fi
    elif [[ "$endpoint" =~ ^[0-9]+$ ]]; then
        echo "http://0.0.0.0:${endpoint}/metrics"
    elif [[ "$endpoint" =~ ^:[0-9]+$ ]]; then
        echo "http://0.0.0.0${endpoint}/metrics"
    elif [[ "$endpoint" == */metrics || "$endpoint" == */metrics\?* ]]; then
        echo "http://${endpoint}"
    else
        echo "http://${endpoint}/metrics"
    fi
}

_speedbench_metric_urls() {
    local port="$1"
    local raw="${SPEEDBENCH_DECODE_METRICS_URLS:-${SPEEDBENCH_METRICS_URLS:-}}"
    local endpoint

    if [[ -n "$raw" ]]; then
        for endpoint in ${raw//,/ }; do
            _speedbench_normalize_metrics_url "$endpoint"
        done
        return 0
    fi

    raw="${SPEEDBENCH_METRICS_PORTS:-}"
    if [[ -n "$raw" ]]; then
        for endpoint in ${raw//,/ }; do
            _speedbench_normalize_metrics_url "$endpoint"
        done
        return 0
    fi

    echo "http://0.0.0.0:${port}/metrics"
}

_speedbench_metric_sum() {
    local port="$1"
    local name="$2"
    local url value
    local total="0"
    local found=0

    while IFS= read -r url; do
        [[ -z "$url" ]] && continue
        value=$(_prometheus_metric_sum_url "$url" "$name" 2>/dev/null || true)
        if [[ -n "$value" ]]; then
            total=$(awk -v a="$total" -v b="$value" 'BEGIN { printf "%.10f", a + b }')
            found=1
        fi
    done < <(_speedbench_metric_urls "$port")

    [[ "$found" -eq 1 ]] || return 1
    awk -v total="$total" 'BEGIN { printf "%.10f\n", total }'
}

_speedbench_metric_avg() {
    local port="$1"
    local name="$2"
    local url value
    local total="0"
    local count=0

    while IFS= read -r url; do
        [[ -z "$url" ]] && continue
        while IFS= read -r value; do
            [[ -z "$value" ]] && continue
            total=$(awk -v a="$total" -v b="$value" 'BEGIN { printf "%.10f", a + b }')
            count=$((count + 1))
        done < <(_prometheus_metric_values_url "$url" "$name" 2>/dev/null || true)
    done < <(_speedbench_metric_urls "$port")

    [[ "$count" -gt 0 ]] || return 1
    awk -v total="$total" -v count="$count" 'BEGIN { printf "%.10f\n", total / count }'
}

_speedbench_metric_endpoint_count() {
    local port="$1"
    local url count=0
    while IFS= read -r url; do
        [[ -n "$url" ]] && count=$((count + 1))
    done < <(_speedbench_metric_urls "$port")
    echo "$count"
}

_speedbench_trtllm_json_metrics_urls() {
    local port="$1"
    local raw="${SPEEDBENCH_TRTLLM_JSON_METRICS_URLS:-}"
    local endpoint url

    if [[ -n "$raw" ]]; then
        for endpoint in ${raw//,/ }; do
            _speedbench_normalize_metrics_url "$endpoint"
        done
        return 0
    fi

    while IFS= read -r url; do
        [[ -z "$url" ]] && continue
        echo "$url" | sed -E 's#/prometheus/metrics([?].*)?$#/metrics#'
    done < <(_speedbench_metric_urls "$port")
}

_speedbench_trtllm_json_spec_metrics() {
    local port="$1"
    local mtp="$2"
    local urls=()
    local url

    while IFS= read -r url; do
        [[ -n "$url" ]] && urls+=("$url")
    done < <(_speedbench_trtllm_json_metrics_urls "$port")

    [[ "${#urls[@]}" -gt 0 ]] || return 1

    python3 - "$mtp" "${urls[@]}" <<'PY'
import json
import os
import sys
import urllib.request


def number(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def stats_from_payload(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    return []


try:
    mtp = float(sys.argv[1])
except (IndexError, ValueError):
    mtp = 0.0

timeout = float(os.environ.get("SPEEDBENCH_METRICS_CURL_TIMEOUT", "10"))
total_draft = 0.0
total_accepted = 0.0
total_requests = 0.0
weighted_acceptance_length = 0.0
unweighted_acceptance_length = 0.0
unweighted_count = 0
used_endpoints = 0

for url in sys.argv[2:]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.load(response)
    except Exception as exc:  # noqa: BLE001 - diagnostics for CI logs
        print(f"SpeedBench AL eval: TRT-LLM JSON metrics fetch failed for {url}: {exc}", file=sys.stderr)
        continue

    endpoint_had_spec = False
    for stat in stats_from_payload(payload):
        if not isinstance(stat, dict):
            continue
        spec = stat.get("specDecodingStats")
        if not isinstance(spec, dict):
            continue

        draft = number(spec.get("numDraftTokens"))
        if draft <= 0:
            continue

        accepted = number(spec.get("numAcceptedTokens"))
        requests = number(spec.get("numRequestsWithDraftTokens"))
        acceptance_length = number(spec.get("acceptanceLength"), default=-1.0)

        total_draft += draft
        total_accepted += accepted
        endpoint_had_spec = True

        if acceptance_length > 0:
            if requests > 0:
                total_requests += requests
                weighted_acceptance_length += acceptance_length * requests
            else:
                unweighted_acceptance_length += acceptance_length
                unweighted_count += 1

    if endpoint_had_spec:
        used_endpoints += 1

if total_requests > 0:
    acceptance_length = weighted_acceptance_length / total_requests
elif unweighted_count > 0:
    acceptance_length = unweighted_acceptance_length / unweighted_count
elif total_draft > 0 and mtp > 0:
    acceptance_length = 1.0 + (total_accepted / (total_draft / mtp))
else:
    sys.exit(1)

verify_steps = round(total_draft / mtp) if total_draft > 0 and mtp > 0 else 0
print(
    f"{acceptance_length:.4f}\t"
    f"{int(round(total_accepted))}\t"
    f"{int(verify_steps)}\t"
    f"{int(round(total_draft))}\t"
    f"{used_endpoints}"
)
PY
}

_speedbench_trtllm_avg_decoded_al() {
    local port="$1"
    local value
    value=$(_speedbench_metric_avg "$port" "trtllm_avg_decoded_tokens_per_iter" 2>/dev/null || true)
    [[ -n "$value" ]] || return 1
    awk -v value="$value" '
        BEGIN {
            if (value < 1.0) {
                exit 1
            }
            printf "%.4f\n", value
        }
    '
}

_speedbench_trtllm_json_avg_decoded_al() {
    local port="$1"
    local urls=()
    local url

    while IFS= read -r url; do
        [[ -n "$url" ]] && urls+=("$url")
    done < <(_speedbench_trtllm_json_metrics_urls "$port")

    [[ "${#urls[@]}" -gt 0 ]] || return 1

    python3 - "${urls[@]}" <<'PY'
import json
import os
import sys
import urllib.request


def number(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def stats_from_payload(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    return []


timeout = float(os.environ.get("SPEEDBENCH_METRICS_CURL_TIMEOUT", "10"))
weighted_total = 0.0
total_requests = 0.0
unweighted_total = 0.0
unweighted_count = 0
used_endpoints = 0

for url in sys.argv[1:]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.load(response)
    except Exception as exc:  # noqa: BLE001 - diagnostics for CI logs
        print(f"SpeedBench AL eval: TRT-LLM JSON metrics fetch failed for {url}: {exc}", file=sys.stderr)
        continue

    endpoint_had_avg = False
    for stat in stats_from_payload(payload):
        if not isinstance(stat, dict):
            continue
        ifb = stat.get("inflightBatchingStats")
        if not isinstance(ifb, dict):
            continue

        avg_decoded = number(ifb.get("avgNumDecodedTokensPerIter"), default=-1.0)
        if avg_decoded < 1.0:
            continue

        gen_requests = number(ifb.get("numGenRequests"))
        endpoint_had_avg = True
        if gen_requests > 0:
            weighted_total += avg_decoded * gen_requests
            total_requests += gen_requests
        else:
            unweighted_total += avg_decoded
            unweighted_count += 1

    if endpoint_had_avg:
        used_endpoints += 1

if total_requests > 0:
    acceptance_length = weighted_total / total_requests
elif unweighted_count > 0:
    acceptance_length = unweighted_total / unweighted_count
else:
    sys.exit(1)

print(f"{acceptance_length:.4f}\t{used_endpoints}")
PY
}

_speedbench_metric_delta() {
    local before="$1"
    local after="$2"
    [[ -n "$before" && -n "$after" ]] || return 1
    awk -v before="$before" -v after="$after" '
        BEGIN {
            delta = after - before
            if (delta < 0) {
                delta = after
            }
            printf "%.10f\n", delta
        }
    '
}

_speedbench_round_metric() {
    local value="$1"
    [[ -n "$value" ]] || return 1
    awk -v value="$value" 'BEGIN { printf "%.0f\n", value }'
}

_speedbench_metrics_framework() {
    local fw="${SPEEDBENCH_METRICS_FRAMEWORK:-${FRAMEWORK:-vllm}}"
    fw="${fw,,}"
    if [[ "$fw" == "dynamo" ]]; then
        local inner="${SPEEDBENCH_DYNAMO_BACKEND_FRAMEWORK:-${DYNAMO_BACKEND_FRAMEWORK:-${DYNAMO_BACKEND:-}}}"
        [[ -n "$inner" ]] && fw="dynamo-${inner,,}"
    fi

    case "$fw" in
        vllm|dynamo-vllm)
            echo "vllm"
            ;;
        sglang|dynamo-sglang)
            echo "sglang"
            ;;
        trt|trtllm|tensorrt-llm|tensorrt_llm|dynamo-trt|dynamo-trtllm|dynamo-tensorrt-llm|dynamo-tensorrt_llm)
            echo "trtllm"
            ;;
        *)
            echo "$fw"
            ;;
    esac
}

_speedbench_metric_source_base() {
    local framework="$1"
    local configured="${SPEEDBENCH_METRICS_FRAMEWORK:-${FRAMEWORK:-$framework}}"
    configured="${configured,,}"
    if [[ "$configured" == dynamo* ]]; then
        echo "dynamo-${framework}-prometheus"
    else
        echo "${framework}-prometheus"
    fi
}

_speedbench_spec_counter_metric() {
    local framework="$1"
    local kind="$2"
    case "${framework}:${kind}" in
        vllm:accepted)
            echo "vllm:spec_decode_num_accepted_tokens_total"
            ;;
        vllm:proposed)
            echo "vllm:spec_decode_num_draft_tokens_total"
            ;;
        vllm:verify)
            echo "vllm:spec_decode_num_drafts_total"
            ;;
        trtllm:accepted)
            echo "trtllm_spec_decode_num_accepted_tokens_total"
            ;;
        trtllm:proposed)
            echo "trtllm_spec_decode_num_draft_tokens_total"
            ;;
        sglang:verify)
            echo "sglang:spec_verify_calls_total"
            ;;
        *)
            return 1
            ;;
    esac
}

_speedbench_spec_gauge_metric() {
    local framework="$1"
    local kind="$2"
    case "${framework}:${kind}" in
        trtllm:acceptance_length)
            echo "trtllm_spec_decode_acceptance_length"
            ;;
        sglang:acceptance_length)
            echo "sglang:spec_accept_length"
            ;;
        sglang:draft_tokens_per_step)
            echo "sglang:spec_num_draft_tokens"
            ;;
        *)
            return 1
            ;;
    esac
}

_speedbench_spec_counter_sum() {
    local framework="$1"
    local port="$2"
    local kind="$3"
    local metric
    metric=$(_speedbench_spec_counter_metric "$framework" "$kind") || return 1
    _speedbench_metric_sum "$port" "$metric"
}

_speedbench_spec_gauge_avg() {
    local framework="$1"
    local port="$2"
    local kind="$3"
    local metric
    metric=$(_speedbench_spec_gauge_metric "$framework" "$kind") || return 1
    _speedbench_metric_avg "$port" "$metric"
}

_speedbench_write_eval_result() {
    local output="$1"
    local mode="$2"
    local mtp="$3"
    local al="${4:-}"
    local accepted="${5:-}"
    local verify_steps="${6:-}"
    local proposed_drafts="${7:-}"
    local framework="${8:-${SPEEDBENCH_METRICS_FRAMEWORK:-${FRAMEWORK:-}}}"
    local metric_source="${9:-}"
    local error="${10:-}"
    local speedbench_model="${MODEL_NAME:-${MODEL:-}}"

    local record_cmd=(
        python3 "$(pwd)/utils/evals/speedbench_al.py"
        record
        --output "$output"
        --reference-yaml "benchmarks/speedbench-reference-al.yaml"
        --model "$speedbench_model"
        --model-prefix "${MODEL_PREFIX:-}"
        --thinking-mode "$mode"
        --num-speculative-tokens "$mtp"
        --category "coding"
        --output-len "4096"
        --temperature "1.0"
        --threshold-ratio "0.90"
    )
    if [[ -n "$framework" ]]; then
        record_cmd+=(--framework "$framework")
    fi
    if [[ -n "$metric_source" ]]; then
        record_cmd+=(--metric-source "$metric_source")
    fi
    if [[ -n "$al" ]]; then
        record_cmd+=(--acceptance-length "$al")
    fi
    if [[ -n "$accepted" ]]; then
        record_cmd+=(--accepted-tokens "$accepted")
    fi
    if [[ -n "$verify_steps" ]]; then
        record_cmd+=(--draft-tokens "$verify_steps")
        record_cmd+=(--verify-steps "$verify_steps")
    fi
    if [[ -n "$proposed_drafts" ]]; then
        record_cmd+=(--proposed-draft-tokens "$proposed_drafts")
    fi
    if [[ -n "$error" ]]; then
        record_cmd+=(--error "$error")
    fi
    "${record_cmd[@]}" || true
}

_speedbench_reference_available() {
    local mode="$1"
    local mtp="$2"
    local reference="benchmarks/speedbench-reference-al.yaml"
    local speedbench_model="${MODEL_NAME:-${MODEL:-}}"
    [[ -f "$reference" ]] || return 1
    python3 "$(pwd)/utils/evals/speedbench_al.py" resolve \
        --reference-yaml "$reference" \
        --model "$speedbench_model" \
        --model-prefix "${MODEL_PREFIX:-}" \
        --thinking-mode "$mode" \
        --num-speculative-tokens "$mtp" \
        --threshold-ratio "0.90" >/dev/null
}

_speedbench_prepare_dataset() {
    local speedbench_dir="$1"
    if [[ -f "$speedbench_dir/qualitative.jsonl" ]]; then
        return 0
    fi
    mkdir -p "$speedbench_dir"
    python3 -m pip install -q datasets tiktoken
    curl -LsSf https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/refs/heads/main/nemo_skills/dataset/speed-bench/prepare.py \
      | python3 - --config qualitative --output_dir "$speedbench_dir"
    [[ -f "$speedbench_dir/qualitative.jsonl" ]]
}

_speedbench_apply_chat_template_kwargs_shim() {
    echo "SpeedBench AL eval: patching vLLM benchmark --chat-template-kwargs support if needed"
    python3 - <<'PYEOF'
import vllm.benchmarks.serve as S
import vllm.benchmarks.datasets.datasets as D


def patch(mod, edits, marker):
    f = mod.__file__
    with open(f) as handle:
        src = handle.read()
    if marker in src:
        print("already patched:", f)
        return
    for old, new in edits:
        n = src.count(old)
        assert n == 1, f"anchor matched {n} times in {f}, aborting:\n{old[:80]}..."
        src = src.replace(old, new, 1)
    with open(f, "w") as handle:
        handle.write(src)
    print("patched OK ->", f)


serve_old = '''    parser.add_argument(
        "--extra-body",'''
serve_new = '''    parser.add_argument(
        "--chat-template-kwargs",
        type=json.loads,
        default=None,
        help="JSON dict forwarded to apply_chat_template during "
        "client-side prompt rendering, e.g. to enable reasoning mode.",
    )
    parser.add_argument(
        "--extra-body",'''
patch(S, [(serve_old, serve_new)], marker='"--chat-template-kwargs"')

disp_old = '''                output_len=args.speed_bench_output_len,
                enable_multimodal_chat=args.enable_multimodal_chat,'''
disp_new = '''                output_len=args.speed_bench_output_len,
                chat_template_kwargs=args.chat_template_kwargs,
                enable_multimodal_chat=args.enable_multimodal_chat,'''

samp_old = '''                # apply template
                if not skip_chat_template:
                    prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        add_generation_prompt=True,
                        tokenize=False,
                    )

                prompt_len = len(tokenizer(prompt).input_ids)'''
samp_new = '''                # apply template
                if not skip_chat_template:
                    _ctk = kwargs.get("chat_template_kwargs") or {}
                    prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        add_generation_prompt=True,
                        tokenize=False,
                        **_ctk,
                    )

                prompt_len = len(tokenizer(prompt).input_ids)'''
patch(D, [(disp_old, disp_new), (samp_old, samp_new)],
      marker="chat_template_kwargs=args.chat_template_kwargs")
PYEOF
}

run_speedbench_al_eval() {
    local port="${PORT:-8888}"
    while [[ $# -gt 0 ]]; do
        case $1 in
            --port) port="$2"; shift 2 ;;
            *)
                if [[ $# -gt 1 && "$2" != --* ]]; then
                    shift 2
                else
                    shift
                fi
                ;;
        esac
    done

    local mtp="${SPEEDBENCH_NUM_SPEC_TOKENS:-${NUM_SPEC_TOKENS:-${SPECULATIVE_DRAFT_TOKENS:-2}}}"
    local default_thinking_mode="off"
    if [[ "${MODEL_PREFIX:-}" == "dsv4" ]]; then
        default_thinking_mode="on"
    fi
    local mode="$default_thinking_mode"

    if [[ "${SPEC_DECODING:-none}" != "mtp" ]]; then
        echo "SpeedBench AL eval: skipping non-MTP config (SPEC_DECODING=${SPEC_DECODING:-none})"
        return 0
    fi

    if [[ -z "${EVAL_RESULT_DIR:-}" ]]; then
        EVAL_RESULT_DIR="$(mktemp -d /tmp/eval_out-XXXXXX)"
        export EVAL_RESULT_DIR
    fi

    local output="${EVAL_RESULT_DIR}/results_speedbench_al_${mode}_mtp${mtp}.json"
    local metrics_framework result_framework metric_source_base metrics_endpoint_count
    metrics_framework=$(_speedbench_metrics_framework)
    result_framework="${SPEEDBENCH_METRICS_FRAMEWORK:-${FRAMEWORK:-$metrics_framework}}"
    metric_source_base=$(_speedbench_metric_source_base "$metrics_framework")
    if [[ "$metrics_framework" == "trtllm" && -z "${SPEEDBENCH_DECODE_METRICS_URLS:-}${SPEEDBENCH_METRICS_URLS:-}${SPEEDBENCH_METRICS_PORTS:-}" ]]; then
        export SPEEDBENCH_METRICS_URLS="http://0.0.0.0:${port}/prometheus/metrics"
    fi
    metrics_endpoint_count=$(_speedbench_metric_endpoint_count "$port")

    case "$metrics_framework" in
        vllm|sglang|trtllm)
            ;;
        *)
            echo "SpeedBench AL eval: unsupported speculative metrics framework=${metrics_framework}" >&2
            _speedbench_write_eval_result "$output" "$mode" "$mtp" "" "" "" "" "$result_framework" "$metric_source_base" "Unsupported speculative metrics framework: ${metrics_framework}"
            return 0
            ;;
    esac

    echo "SpeedBench AL eval: metrics framework=${metrics_framework}, endpoints=${metrics_endpoint_count}"

    local speedbench_dir="${SPEEDBENCH_DIR:-$(pwd)/speed_bench_data}"
    if ! _speedbench_prepare_dataset "$speedbench_dir"; then
        echo "SpeedBench AL eval: SpeedBench dataset download failed" >&2
        _speedbench_write_eval_result "$output" "$mode" "$mtp" "" "" "" "" "$result_framework" "$metric_source_base" "SpeedBench dataset download failed"
        return 0
    fi

    if ! _speedbench_reference_available "$mode" "$mtp"; then
        echo "SpeedBench AL eval: no reference for mode=${mode} mtp=${mtp}" >&2
        _speedbench_write_eval_result "$output" "$mode" "$mtp" "" "" "" "" "$result_framework" "$metric_source_base" "No SpeedBench AL reference for this eval cell"
        return 0
    fi

    local thinking_kwargs='{"thinking": true, "reasoning_effort": "high"}'
    local client="${SPEEDBENCH_CLIENT:-auto}"
    local use_vllm_client=0
    if [[ "$client" != "openai" && "$client" != "native" ]] && command -v vllm >/dev/null 2>&1; then
        use_vllm_client=1
    fi

    local think_args=()
    if [[ "$mode" == "on" ]]; then
        if [[ "$use_vllm_client" -eq 1 ]]; then
            if ! _speedbench_apply_chat_template_kwargs_shim; then
                echo "SpeedBench AL eval: --chat-template-kwargs shim failed" >&2
                _speedbench_write_eval_result "$output" "$mode" "$mtp" "" "" "" "" "$result_framework" "$metric_source_base" "--chat-template-kwargs shim failed"
                return 0
            fi
            think_args=(--chat-template-kwargs "$thinking_kwargs")
        fi
    fi

    local accepted_before="" proposed_before="" verify_before=""
    accepted_before=$(_speedbench_spec_counter_sum "$metrics_framework" "$port" "accepted" 2>/dev/null || true)
    proposed_before=$(_speedbench_spec_counter_sum "$metrics_framework" "$port" "proposed" 2>/dev/null || true)
    verify_before=$(_speedbench_spec_counter_sum "$metrics_framework" "$port" "verify" 2>/dev/null || true)
    accepted_before="${accepted_before:-0}"
    proposed_before="${proposed_before:-0}"
    verify_before="${verify_before:-0}"

    local bench_rc=0
    local speedbench_model="${MODEL_NAME:-${MODEL:-}}"
    echo "SpeedBench AL eval: running mode=${mode} mtp=${mtp}"
    if [[ "$use_vllm_client" -eq 1 ]]; then
        local raw_result_dir
        raw_result_dir="$(mktemp -d /tmp/speedbench_al_raw-XXXXXX)"
        local bench_cmd=(
            vllm bench serve
            --model "$speedbench_model"
            --port "$port"
            --dataset-name speed_bench
            --dataset-path "$speedbench_dir"
            --speed-bench-category coding
            --speed-bench-output-len 4096
            --num-prompts -1
            --max-concurrency 1
            --save-result
            --result-dir "$raw_result_dir"
            --result-filename "speedbench_al_${mode}_mtp${mtp}"
            --trust-remote-code
            --tokenizer-mode deepseek_v4
            --temperature 1.0
            "${think_args[@]}"
        )
        "${bench_cmd[@]}" || bench_rc=$?
        rm -rf "$raw_result_dir" || true
    else
        export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
        local native_cmd=(
            python3 "$(pwd)/utils/evals/speedbench_client.py"
            --model "$speedbench_model"
            --base-url "http://0.0.0.0:${port}"
            --dataset-path "$speedbench_dir"
            --category coding
            --output-len 4096
            --temperature 1.0
            --thinking-mode "$mode"
            --timeout "${SPEEDBENCH_CLIENT_TIMEOUT:-1800}"
            --retries "${SPEEDBENCH_CLIENT_RETRIES:-2}"
        )
        if [[ -n "${SPEEDBENCH_CLIENT_ENDPOINT:-}" ]]; then
            native_cmd+=(--endpoint "$SPEEDBENCH_CLIENT_ENDPOINT")
        elif [[ "${MODEL_PREFIX:-}" == "dsv4" ]]; then
            native_cmd+=(--endpoint completions)
        fi
        if [[ "$mode" == "on" ]]; then
            native_cmd+=(--thinking-kwargs "$thinking_kwargs")
        fi
        if [[ "${MODEL_PREFIX:-}" == "dsv4" ]]; then
            native_cmd+=(--dsv4)
        fi
        "${native_cmd[@]}" || bench_rc=$?
    fi
    if [[ "$bench_rc" -ne 0 ]]; then
        echo "SpeedBench AL eval: client failed with exit code ${bench_rc}" >&2
        _speedbench_write_eval_result "$output" "$mode" "$mtp" "" "" "" "" "$result_framework" "$metric_source_base" "SpeedBench client failed with exit code ${bench_rc}"
        return 0
    fi

    local accepted_after="" proposed_after="" verify_after=""
    local al="" delta_acc="" delta_proposed="" delta_verify="" metric_source=""
    accepted_after=$(_speedbench_spec_counter_sum "$metrics_framework" "$port" "accepted" 2>/dev/null || true)
    proposed_after=$(_speedbench_spec_counter_sum "$metrics_framework" "$port" "proposed" 2>/dev/null || true)
    verify_after=$(_speedbench_spec_counter_sum "$metrics_framework" "$port" "verify" 2>/dev/null || true)

    if [[ -n "$accepted_after" ]]; then
        delta_acc=$(_speedbench_round_metric "$(_speedbench_metric_delta "$accepted_before" "$accepted_after")")
    fi
    if [[ -n "$proposed_after" ]]; then
        delta_proposed=$(_speedbench_round_metric "$(_speedbench_metric_delta "$proposed_before" "$proposed_after")")
    fi
    if [[ -n "$verify_after" ]]; then
        delta_verify=$(_speedbench_round_metric "$(_speedbench_metric_delta "$verify_before" "$verify_after")")
    fi

    if [[ "$metrics_framework" == "vllm" && -n "$delta_acc" && -n "$delta_verify" && "$delta_verify" -gt 0 ]]; then
        al=$(awk -v accepted="$delta_acc" -v verify="$delta_verify" 'BEGIN { printf "%.4f", 1 + (accepted / verify) }')
        metric_source="${metric_source_base}-counters-endpoints${metrics_endpoint_count}"
    elif [[ "$metrics_framework" == "trtllm" ]]; then
        al=$(_speedbench_spec_gauge_avg "$metrics_framework" "$port" "acceptance_length" 2>/dev/null | awk '{ printf "%.4f", $1 }' || true)
        if [[ -n "$al" ]]; then
            metric_source="${metric_source_base}-gauge-endpoints${metrics_endpoint_count}"
            if [[ -n "$delta_acc" || -n "$delta_proposed" ]]; then
                metric_source="${metric_source}+token-counters"
            fi
        else
            local trt_json_metrics="" trt_json_endpoints=""
            trt_json_metrics=$(_speedbench_trtllm_json_spec_metrics "$port" "$mtp" || true)
            if [[ -n "$trt_json_metrics" ]]; then
                IFS=$'\t' read -r al delta_acc delta_verify delta_proposed trt_json_endpoints <<< "$trt_json_metrics"
                metric_source="trtllm-json-iteration-stats-endpoints${trt_json_endpoints}"
            fi
            if [[ -z "$al" ]]; then
                al=$(_speedbench_trtllm_avg_decoded_al "$port" || true)
                if [[ -n "$al" ]]; then
                    metric_source="${metric_source_base}-avg-decoded-tokens-endpoints${metrics_endpoint_count}"
                fi
            fi
            if [[ -z "$al" ]]; then
                local trt_json_avg_metrics="" trt_json_avg_endpoints=""
                trt_json_avg_metrics=$(_speedbench_trtllm_json_avg_decoded_al "$port" || true)
                if [[ -n "$trt_json_avg_metrics" ]]; then
                    IFS=$'\t' read -r al trt_json_avg_endpoints <<< "$trt_json_avg_metrics"
                    metric_source="trtllm-json-avg-decoded-tokens-endpoints${trt_json_avg_endpoints}"
                fi
            fi
        fi
    elif [[ "$metrics_framework" == "sglang" ]]; then
        al=$(_speedbench_spec_gauge_avg "$metrics_framework" "$port" "acceptance_length" 2>/dev/null | awk '{ printf "%.4f", $1 }' || true)
        if [[ -n "$al" ]]; then
            metric_source="${metric_source_base}-gauge-endpoints${metrics_endpoint_count}"
        fi
        if [[ -n "$delta_verify" && "$delta_verify" -gt 0 ]]; then
            local draft_depth=""
            draft_depth=$(_speedbench_spec_gauge_avg "$metrics_framework" "$port" "draft_tokens_per_step" 2>/dev/null || true)
            if [[ -n "$draft_depth" ]]; then
                delta_proposed=$(_speedbench_round_metric "$(awk -v verify="$delta_verify" -v depth="$draft_depth" 'BEGIN { value = verify * (depth - 1); if (value < 0) value = 0; printf "%.10f\n", value }')")
            fi
            if [[ -n "$al" ]]; then
                delta_acc=$(_speedbench_round_metric "$(awk -v verify="$delta_verify" -v al="$al" 'BEGIN { value = verify * (al - 1); if (value < 0) value = 0; printf "%.10f\n", value }')")
                metric_source="${metric_source:-${metric_source_base}-gauge-endpoints${metrics_endpoint_count}}+derived-token-counters"
            fi
        fi
    fi

    if [[ -z "$al" ]]; then
        echo "SpeedBench AL eval: could not collect speculative acceptance metrics from server" >&2
        local metric_error="Could not collect speculative acceptance metrics from server"
        if [[ "${FRAMEWORK:-}" == dynamo* && -z "${SPEEDBENCH_DECODE_METRICS_URLS:-}${SPEEDBENCH_METRICS_URLS:-}${SPEEDBENCH_METRICS_PORTS:-}" ]]; then
            metric_error="${metric_error}; for Dynamo/disagg set SPEEDBENCH_DECODE_METRICS_URLS or SPEEDBENCH_METRICS_PORTS to decode-worker /metrics endpoints"
        fi
        _speedbench_write_eval_result "$output" "$mode" "$mtp" "" "$delta_acc" "$delta_verify" "$delta_proposed" "$result_framework" "$metric_source_base" "$metric_error"
    else
        _speedbench_write_eval_result "$output" "$mode" "$mtp" "$al" "$delta_acc" "$delta_verify" "$delta_proposed" "$result_framework" "$metric_source"
    fi
}

run_lm_eval() {
    local port="${PORT:-8888}"
    local tasks_dir="${EVAL_TASKS_DIR:-utils/evals/gsm8k.yaml}"
    local results_dir="${EVAL_RESULT_DIR:-$(mktemp -d /tmp/eval_out-XXXXXX)}"
    local eval_context_len="${EVAL_MAX_MODEL_LEN:-16384}"
    local temperature=0
    local top_p=1
    local concurrent_requests="${EVAL_CONCURRENT_REQUESTS:-${CONC:-64}}"

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
    local is_multinode_json="false"
    if [ "${IS_MULTINODE:-false}" = "true" ]; then
        is_multinode_json="true"
    fi

    local prefill_tp="${PREFILL_TP:-${TP:-1}}"
    local prefill_ep="${PREFILL_EP:-${EP_SIZE:-1}}"
    local prefill_num_workers="${PREFILL_NUM_WORKERS:-1}"
    local decode_tp="${DECODE_TP:-${TP:-1}}"
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
  "is_multinode": ${is_multinode_json},
  "framework": "${fw:-unknown}",
  "precision": "${prec:-unknown}",
  "spec_decoding": "${SPEC_DECODING}",
  "tp": ${TP:-1},
  "conc": ${CONC:-1},
  "ep": ${EP_SIZE:-1},
  "dp_attention": ${dp_json},
  "prefill_tp": ${prefill_tp},
  "prefill_ep": ${prefill_ep},
  "prefill_dp_attention": ${prefill_dp_json},
  "prefill_num_workers": ${prefill_num_workers},
  "decode_tp": ${decode_tp},
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
    run_speedbench_al_eval "${forwarded[@]}" || true
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


# --------------------------------
# Agentic trace replay helpers (aiperf driver)
# --------------------------------

INFMAX_CONTAINER_WORKSPACE="${INFMAX_CONTAINER_WORKSPACE:-/workspace}"
AGENTIC_DIR="${AGENTIC_DIR:-${INFMAX_CONTAINER_WORKSPACE}/utils/agentic-benchmark}"
AIPERF_DIR="${AIPERF_DIR:-${INFMAX_CONTAINER_WORKSPACE}/utils/aiperf}"

agentic_pip_install() {
    local pip_install=(python3 -m pip install)
    if python3 -m pip install --help 2>/dev/null | grep -q -- "--break-system-packages"; then
        pip_install+=(--break-system-packages)
    fi

    "${pip_install[@]}" "$@"
}

ensure_hf_cli() {
    if command -v hf >/dev/null 2>&1; then
        return 0
    fi

    # Some lean runtime images used by multinode SGLang include Python but not
    # the Hugging Face CLI. Install just the hub CLI before prefetching traces.
    agentic_pip_install --quiet "huggingface_hub[cli]>=0.25.0"
}

resolve_trace_source() {
    # Per-recipe override: set WEKA_LOADER_OVERRIDE to one of the aiperf
    # public-dataset loader names allowed by the inferencex-agentx-mvp
    # scenario. Used by recipes whose servers have non-default context
    # caps (e.g. minimaxm2.5 at max_model_len ~256k can't replay the
    # unfiltered 052726 corpus and switches to the 256k-capped variant).
    local loader="${WEKA_LOADER_OVERRIDE:-semianalysis_cc_traces_weka_with_subagents}"
    local dataset
    case "$loader" in
        semianalysis_cc_traces_weka_with_subagents)
            dataset="semianalysisai/cc-traces-weka-with-subagents-052726"
            ;;
        semianalysis_cc_traces_weka_with_subagents_256k)
            dataset="semianalysisai/cc-traces-weka-with-subagents-052726-256k"
            ;;
        *)
            echo "Error: unknown WEKA_LOADER_OVERRIDE='$loader'. Allowed: semianalysis_cc_traces_weka_with_subagents, semianalysis_cc_traces_weka_with_subagents_256k" >&2
            exit 1
            ;;
    esac
    TRACE_SOURCE_FLAG="--public-dataset $loader"
    echo "Loading traces via aiperf public-dataset: $loader ($dataset)"
    # Pre-download the dataset into the shared HF_HUB_CACHE (same mount used
    # for model weights) so subsequent runs read from cache instead of
    # re-downloading every job.
    ensure_hf_cli
    hf download --repo-type dataset "$dataset"
}

install_agentic_deps() {
    # vllm/vllm-openai container ships without git. pip needs git to
    # introspect the aiperf source tree on install. Install on demand;
    # no-op when git is already present (e.g. AMD images that ship it).
    if ! command -v git >/dev/null 2>&1; then
        apt-get update && apt-get install -y git
    fi
    agentic_pip_install --quiet urllib3 requests 2>/dev/null || true
    agentic_pip_install -q -r "$AGENTIC_DIR/requirements.txt"
    # Editable install of aiperf from the submodule — gives us the
    # `aiperf` CLI plus the inferencex-agentx-mvp scenario plugin.
    #
    # `--ignore-installed` sidesteps the distutils-uninstall error that
    # vLLM containers hit on apt-managed system packages (blinker, etc.)
    # when pip's resolver tries to upgrade one of aiperf's transitive
    # deps. Installing fresh into the user/site location is safe — the
    # system package stays in place and pip's import order picks up our
    # newer copy first.
    agentic_pip_install -q --ignore-installed -e "$AIPERF_DIR"
    # Force-upgrade datasets: containers often ship an older version without
    # the `Json` feature type used by the HF traces dataset. `Json` was added
    # in datasets 4.7.0 (March 2025). Unpinned installs won't upgrade an
    # already-present package.
    agentic_pip_install --upgrade "datasets>=4.7.0"
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
    # --trace-idle-gap-cap-seconds 60 (per-trace idle-gap compression
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
    REPLAY_CMD="aiperf profile --scenario inferencex-agentx-mvp"
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
    REPLAY_CMD+=" --failed-request-threshold 0.10"
    # Sample each trajectory's warmup start position uniformly from
    # [25%, 75%] of the trace's turn count (was hardcoded 0%-70% upstream).
    # Avoids starting trajectories right at turn 0 where the KV cache is
    # cold and skews early steady-state samples.
    REPLAY_CMD+=" --trajectory-start-min-ratio 0.25"
    REPLAY_CMD+=" --trajectory-start-max-ratio 0.75"
    # Use server-reported usage fields (prompt_tokens / completion_tokens) for
    # ISL/OSL instead of client-side tokenizer.encode(). Auto-enables
    # stream_options.include_usage on the OpenAI chat endpoint. Skips the
    # heavy per-record tokenization in the records pipeline that was pinning
    # CPU on minimax-m2.5 at high concurrency. Lossless for vLLM (server
    # usage is authoritative).
    REPLAY_CMD+=" --use-server-token-count"
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
    # has 472. Cap at 472 so all unique traces are loaded (the loader treats
    # this as a ``min(cap, available)`` ceiling, not a target — see
    # semianalysis_cc_traces_weka.py).
    REPLAY_CMD+=" --num-dataset-entries 472"
    # 1-second timeslices on the server-metrics scrape so the post-run
    # plotter has per-window time series (KV usage, cache hit rate,
    # throughput, etc.). Matches kv-cache-tester's poll_interval=1.0
    # snapshot cadence so metrics_plots.png is visually comparable.
    # Without this, aiperf only emits aggregate stats and the 6x2 panels
    # collapse to flat lines.
    REPLAY_CMD+=" --slice-duration 1.0"
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
    # into $AGENTIC_OUTPUT_DIR/$RESULT_FILENAME.json. The workflow's existing
    # retry-based existence check is the single success gate.
    local result_dir="$1"
    RESULT_DIR="$result_dir" AGENTIC_OUTPUT_DIR="${AGENTIC_OUTPUT_DIR:-$INFMAX_CONTAINER_WORKSPACE}" \
        python3 "$INFMAX_CONTAINER_WORKSPACE/utils/process_agentic_result.py"

    # Generate metrics_plots.png from the same aiperf artifacts. Best-effort:
    # don't fail the launcher if plot generation has trouble (e.g. matplotlib
    # missing in a stripped-down image). The agg JSON is the success gate.
    python3 "$INFMAX_CONTAINER_WORKSPACE/utils/generate_aiperf_plots.py" "$result_dir" 2>&1 || true
}

run_agentic_replay_and_write_outputs() {
    local result_dir="$1"
    local replay_rc

    echo "$REPLAY_CMD" > "$result_dir/benchmark_command.txt"

    set +e
    set -x
    $REPLAY_CMD 2>&1 | tee "$result_dir/benchmark.log"
    replay_rc=${PIPESTATUS[0]}
    set +x
    set -e

    write_agentic_result_json "$result_dir"

    python3 "$AGENTIC_DIR/scripts/analyze_benchmark_distributions.py" \
        "$result_dir/aiperf_artifacts" -o "$result_dir" 2>&1 || true

    if [ "$replay_rc" -ne 0 ]; then
        echo "ERROR: agentic trace replay exited with code $replay_rc after writing available results" >&2
        return "$replay_rc"
    fi
}
