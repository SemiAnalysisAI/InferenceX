#!/usr/bin/env bash

# Shared benchmarking utilities for InferenceMAX

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

    # Show logs until server is ready
    tail -f "$server_log" &
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
# All parameters are required
# Parameters:
#   --model: Model name
#   --port: Server port
#   --backend: Backend type - 'vllm' or 'openai'
#   --input-len: Random input sequence length
#   --output-len: Random output sequence length
#   --random-range-ratio: Random range ratio
#   --num-prompts: Number of prompts
#   --max-concurrency: Max concurrency
#   --result-filename: Result filename without extension
#   --result-dir: Result directory
run_benchmark_serving() {
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

    # Parse arguments
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

    # Clone benchmark serving repo
    local BENCH_SERVING_DIR=$(mktemp -d /tmp/bmk-XXXXXX)
    git clone https://github.com/kimbochen/bench_serving.git "$BENCH_SERVING_DIR"

    # Run benchmark
    set -x
    python3 "$BENCH_SERVING_DIR/benchmark_serving.py" \
        --model "$model" \
        --backend "$backend" \
        --base-url "http://0.0.0.0:$port" \
        --dataset-name random \
        --random-input-len "$input_len" \
        --random-output-len "$output_len" \
        --random-range-ratio "$random_range_ratio" \
        --num-prompts "$num_prompts" \
        --max-concurrency "$max_concurrency" \
        --request-rate inf \
        --ignore-eos \
        --save-result \
        --percentile-metrics 'ttft,tpot,itl,e2el' \
        --result-dir "$result_dir" \
        --result-filename "$result_filename.json"
    set +x
}


# ------------------------------
# Eval (lm-eval-harness) helpers
# ------------------------------

# Install or update lm-eval dependencies
_install_lm_eval_deps() {
    set +x
    python3 -m pip install -q --no-cache-dir "lm-eval[api]" || true
    # Temporary: workaround known harness issue by using main
    python3 -m pip install -q --no-cache-dir --no-deps \
        "git+https://github.com/EleutherAI/lm-evaluation-harness.git@main" || true
}

# Patch lm-eval filters to be robust to empty strings via sitecustomize
_patch_lm_eval_filters() {
    set +x
    local patch_dir
    patch_dir="$(mktemp -d)"
    cat > "$patch_dir/sitecustomize.py" <<'PY'
import re, sys, unicodedata
from lm_eval.filters import extraction as ex

def _s(x):  # coerce to str
    return x if isinstance(x, str) else ""

# --- Patch RegexFilter.apply (used by many datasets) ---
_orig_regex_apply = ex.RegexFilter.apply
def _safe_regex_apply(self, resps, docs):
    out = []
    for inst in resps:  # inst is a list of candidate responses for one doc
        filtered = []
        for resp in inst:
            txt = _s(resp)
            m = self.regex.findall(txt)
            if m:
                m = m[self.group_select]
                if isinstance(m, tuple):
                    m = [t for t in m if t]
                    m = m[0] if m else self.fallback
                m = m.strip()
            else:
                m = self.fallback
            filtered.append(m)
        out.append(filtered)
    return out
ex.RegexFilter.apply = _safe_regex_apply

# --- Patch MultiChoiceRegexFilter.apply (used by GSM8K flexible-extract) ---
_orig_mc_apply = ex.MultiChoiceRegexFilter.apply
def _safe_mc_apply(self, resps, docs):
    def find_match(regex, resp, convert_dict={}):
        txt = _s(resp)
        match = regex.findall(txt)
        if match:
            match = match[self.group_select]
            if isinstance(match, tuple):
                match = [m for m in match if m]
                if match:
                    match = match[0]
        if match:
            match = match.strip()
            if match in convert_dict:
                return convert_dict[match]
            return match
        return None

    punct_tbl = dict.fromkeys(
        i for i in range(sys.maxunicode)
        if unicodedata.category(chr(i)).startswith("P")
    )

    def filter_ignores(st):
        st = _s(st)
        if self.regexes_to_ignore is not None:
            for s in self.regexes_to_ignore:
                st = re.sub(s, "", st)
        if self.ignore_case:
            st = st.lower()
        if self.ignore_punctuation:
            st = st.translate(punct_tbl)
        return st

    out = []
    for r, doc in zip(resps, docs):
        # Build fallback regexes from choices (A, B, C, ...) as in upstream
        fallback_regexes, choice_to_alpha = [], {}
        next_alpha = "A"
        without_paren, without_paren_to_target = [], {}
        for c in doc.get("choices", []):
            m = filter_ignores(c.strip())
            fallback_regexes.append(re.escape(m))
            choice_to_alpha[m] = f"({next_alpha})"
            without_paren.append(next_alpha)
            without_paren_to_target[next_alpha] = f"({next_alpha})"
            next_alpha = chr(ord(next_alpha) + 1)

        fallback_regex = re.compile("|".join(fallback_regexes)) if fallback_regexes else None
        without_paren_regex = re.compile(rf":[\s]*({'|'.join(without_paren)})") if without_paren else None

        filtered = []
        for resp in r:
            m = find_match(self.regex, resp)
            if not m and fallback_regex:
                m = find_match(fallback_regex, filter_ignores(resp), choice_to_alpha)
            if not m and without_paren_regex:
                m = find_match(without_paren_regex, resp, without_paren_to_target)
            if not m:
                m = self.fallback
            filtered.append(m)
        out.append(filtered)
    return out

ex.MultiChoiceRegexFilter.apply = _safe_mc_apply
PY
    export PYTHONPATH="${patch_dir}:${PYTHONPATH:-}"
}

# Run an lm-eval-harness task against a local OpenAI-compatible server
# Parameters:
#   --port:              Server port (default: $PORT or 8888)
#   --task:              Eval task (default: $EVAL_TASK or gsm8k)
#   --num-fewshot:       Fewshot k (default: $NUM_FEWSHOT or 5)
#   --results-dir:       Output dir (default: $EVAL_RESULT_DIR or eval_out)
#   --batch-size:        Harness batch size (default: 2)
#   --gen-max-tokens:    Max tokens for generation (default: 8192)
#   --temperature:       Temperature (default: 0)
#   --top-p:             Top-p (default: 1)
run_lm_eval() {
    set +x
    local port="${PORT:-8888}"
    local task="${EVAL_TASK:-gsm8k}"
    local num_fewshot="${NUM_FEWSHOT:-5}"
    local results_dir="${EVAL_RESULT_DIR:-eval_out}"
    local batch_size=2
    local gen_max_tokens=8192
    local temperature=0
    local top_p=1

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --port)
                port="$2"; shift 2;;
            --task)
                task="$2"; shift 2;;
            --num-fewshot)
                num_fewshot="$2"; shift 2;;
            --results-dir)
                results_dir="$2"; shift 2;;
            --batch-size)
                batch_size="$2"; shift 2;;
            --gen-max-tokens)
                gen_max_tokens="$2"; shift 2;;
            --temperature)
                temperature="$2"; shift 2;;
            --top-p)
                top_p="$2"; shift 2;;
            *)
                echo "Unknown parameter: $1"; return 1;;
        esac
    done
 
    _install_lm_eval_deps
    _patch_lm_eval_filters

    local openai_server_base="http://0.0.0.0:${port}"
    local openai_chat_base="$openai_server_base/v1/chat/completions"
    export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}

    set -x
    python3 -m lm_eval --model local-chat-completions --apply_chat_template \
      --tasks "${task}" \
      --num_fewshot "${num_fewshot}" \
      --batch_size "${batch_size}" \
      --output_path "/workspace/${results_dir}" \
      --model_args "model=${MODEL},base_url=${openai_chat_base},api_key=${OPENAI_API_KEY},eos_string=</s>,max_retries=3,num_concurrent=32,tokenized_requests=False" \
      --gen_kwargs "max_tokens=${gen_max_tokens},temperature=${temperature},top_p=${top_p}"
    set +x
}

# Append a Markdown summary to GitHub step summary (no-op if not in GH Actions)
append_lm_eval_summary() {
    set +x
    local results_dir="${EVAL_RESULT_DIR:-eval_out}"
    local task="${EVAL_TASK:-gsm8k}"
    # Render markdown once, then decide where to write it to avoid redirection errors
    local md_out
    md_out=$(python3 utils/lm_eval_to_md.py \
            --results-dir "/workspace/${results_dir}" \
            --task "${task}" \
            --framework "${FRAMEWORK}" \
            --precision "${PRECISION}" \
            --tp "${TP:-1}" \
            --ep "${EP_SIZE:-1}" \
            --dp-attention "${DP_ATTENTION:-false}" 2>/dev/null || true)

    # If nothing was produced, nothing to append
    if [ -z "${md_out}" ]; then
        return 0
    fi

    # Prefer GitHub step summary when available and path is valid; otherwise fallback to workspace file
    if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
        local _gh_path="$GITHUB_STEP_SUMMARY"
        local _gh_dir
        _gh_dir="$(dirname "$_gh_path")"
        if [ -d "$_gh_dir" ]; then
            printf "%s\n" "${md_out}" >> "$_gh_path" || true
            return 0
        fi
    fi

    # Fallback: write to a summary file alongside results
    mkdir -p "/workspace/${results_dir}" 2>/dev/null || true
    printf "%s\n" "${md_out}" >> "/workspace/${results_dir}/SUMMARY.md" || true
}
