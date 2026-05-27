#!/usr/bin/env bash

# Shared benchmarking utilities for InferenceX

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
# TRACE_REPLAY_DIR retained for any out-of-tree consumer that still
# imports the kv-cache-tester scripts. Not used by the helpers below.
TRACE_REPLAY_DIR="${TRACE_REPLAY_DIR:-${INFMAX_CONTAINER_WORKSPACE}/utils/trace-replay}"

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
    local dataset="semianalysisai/cc-traces-weka-no-subagents-051226"
    # aiperf reads the corpus via its public-dataset registry; the loader
    # under the hood pulls from semianalysisai/cc-traces-weka-no-subagents-051226
    # (949 traces, no-subagents variant — see plugins.yaml).
    TRACE_SOURCE_FLAG="--public-dataset semianalysis_cc_traces_weka"
    echo "Loading traces via aiperf public-dataset: semianalysis_cc_traces_weka ($dataset)"
    # Pre-download the dataset into the shared HF_HUB_CACHE (same mount used
    # for model weights) so subsequent runs read from cache instead of
    # re-downloading every job.
    ensure_hf_cli
    hf download --repo-type dataset "$dataset"
}

install_agentic_deps() {
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
    # Live-assistant mode is on by default
    # (AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=1): the loader emits
    # user-only deltas and the worker threads the server's live assistant
    # response back into the session. This preserves cache-hit reuse on
    # the just-generated KV blocks at the cost of hash-id fidelity past
    # turn 0 — which is exactly what we want for benchmark numbers.
    #
    # The scenario plugin locks: --cache-bust first_turn_prefix,
    # --inter-turn-delay-cap-seconds 60, etc., and auto-injects them — so
    # we do not pass them. See utils/aiperf/docs/tutorials/agentx-mvp.md.
    local result_dir="$1"
    local duration="${DURATION:-1800}"

    export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=1
    # Dataset configuration (load + reconstruct + inputs.json + mmap)
    # routinely takes 4-5 min for the 949-trace weka corpus on fast /tmp
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
    # Abort the run if real-failure rate exceeds 5% after a grace floor of
    # max(CONC, 10) records. Context-overflow records are dropped from the
    # failure tally in AGENTIC_REPLAY scenarios (see record_processor_service
    # in the aiperf submodule), so this threshold measures only real failures
    # (server 5xx, parse errors, malformed responses).
    REPLAY_CMD+=" --failed-request-threshold 0.05"
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
    # Default --num-dataset-entries is 100; the weka corpus has 949. Cap
    # at 949 so all unique traces are loaded (the loader treats this as a
    # ``min(cap, available)`` ceiling, not a target — see
    # semianalysis_cc_traces_weka.py).
    REPLAY_CMD+=" --num-dataset-entries 949"
    # 1-second timeslices on the server-metrics scrape so the post-run
    # plotter has per-window time series (KV usage, cache hit rate,
    # throughput, etc.). Matches kv-cache-tester's poll_interval=1.0
    # snapshot cadence so metrics_plots.png is visually comparable.
    # Without this, aiperf only emits aggregate stats and the 6x2 panels
    # collapse to flat lines.
    REPLAY_CMD+=" --slice-duration 1.0"
    REPLAY_CMD+=" --output-artifact-dir $result_dir/trace_replay"
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
