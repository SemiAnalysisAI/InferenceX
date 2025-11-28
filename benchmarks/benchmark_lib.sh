#!/usr/bin/env bash

# Shared benchmarking + evaluation utilities for InferenceMAX

# ---------------------------------
# Server readiness / benchmarks
# ---------------------------------

# Wait for server to be ready by polling the health endpoint
# Parameters:
#   --port: Server port (required)
#   --server-log: Path to server log file (required)
#   --server-pid: Server process ID (required)
#   --sleep-interval: Sleep interval between health checks (optional, default: 5)
wait_for_server_ready() {
    set +x
    local port=""
    local server_log=""
    local server_pid=""
    local sleep_interval=5

    while [[ $# -gt 0 ]]; do
        case $1 in
            --port)           port="$2"; shift 2 ;;
            --server-log)     server_log="$2"; shift 2 ;;
            --server-pid)     server_pid="$2"; shift 2 ;;
            --sleep-interval) sleep_interval="$2"; shift 2 ;;
            *)                echo "Unknown parameter: $1"; return 1 ;;
        esac
    done

    if [[ -z "$port" ]]; then echo "Error: --port is required"; return 1; fi
    if [[ -z "$server_log" ]]; then echo "Error: --server-log is required"; return 1; fi
    if [[ -z "$server_pid" ]]; then echo "Error: --server-pid is required"; return 1; fi

    # Show logs until server is ready
    tail -f "$server_log" &
    local TAIL_PID=$!
    
    until curl --output /dev/null --silent --fail "http://0.0.0.0:$port/health"; do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "Server died before becoming healthy. Exiting."
            kill "$TAIL_PID"
            exit 1
        fi
        sleep "$sleep_interval"
    done
    kill "$TAIL_PID"
}

# Run benchmark serving with standardized parameters
# All parameters are required unless otherwise noted
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
    local tokenizer=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            --model)              model="$2"; shift 2 ;;
            --port)               port="$2"; shift 2 ;;
            --backend)            backend="$2"; shift 2 ;;
            --input-len)          input_len="$2"; shift 2 ;;
            --output-len)         output_len="$2"; shift 2 ;;
            --random-range-ratio) random_range_ratio="$2"; shift 2 ;;
            --num-prompts)        num_prompts="$2"; shift 2 ;;
            --max-concurrency)    max_concurrency="$2"; shift 2 ;;
            --result-filename)    result_filename="$2"; shift 2 ;;
            --result-dir)         result_dir="$2"; shift 2 ;;
            --tokenizer)          tokenizer="$2"; shift 2 ;;
            *)                    echo "Unknown parameter: $1"; return 1 ;;
        esac
    done

    # Validation
    local vars=(model port backend input_len output_len random_range_ratio num_prompts max_concurrency result_filename result_dir)
    for var in "${vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            echo "Error: --${var//_/-} is required"
            return 1
        fi
    done

    local BENCH_SERVING_DIR
    BENCH_SERVING_DIR=$(mktemp -d /tmp/bmk-XXXXXX)
    git clone https://github.com/kimbochen/bench_serving.git "$BENCH_SERVING_DIR"

    local extra_tokenizer_args=()
    if [[ -n "$tokenizer" ]]; then
        extra_tokenizer_args=(--tokenizer "$tokenizer")
    fi

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
        "${extra_tokenizer_args[@]}" \
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

# --- Patch RegexFilter.apply ---
_orig_regex_apply = ex.RegexFilter.apply
def _safe_regex_apply(self, resps, docs):
    out = []
    for inst in resps:
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

# --- Patch MultiChoiceRegexFilter.apply ---
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
PY
    export PYTHONPATH="${patch_dir}:${PYTHONPATH:-}"
}

run_lm_eval() {
    set +x
    local port="${PORT:-8888}"
    local task="${EVAL_TASK:-gsm8k}"
    local num_fewshot="${NUM_FEWSHOT:-5}"
    local results_dir="${EVAL_RESULT_DIR:-eval_out}"
    local gen_max_tokens=4096
    local temperature=0
    local top_p=1
    local concurrent_requests=32

    while [[ $# -gt 0 ]]; do
        case $1 in
            --port)           port="$2"; shift 2 ;;
            --task)           task="$2"; shift 2 ;;
            --num-fewshot)    num_fewshot="$2"; shift 2 ;;
            --results-dir)    results_dir="$2"; shift 2 ;;
            --gen-max-tokens) gen_max_tokens="$2"; shift 2 ;;
            --temperature)    temperature="$2"; shift 2 ;;
            --top-p)          top_p="$2"; shift 2 ;;
            --concurrent-requests) concurrent_requests="$2"; shift 2 ;;
            *)                echo "Unknown parameter: $1"; return 1 ;;
        esac
    done

    _install_lm_eval_deps
    _patch_lm_eval_filters

    local openai_server_base="http://0.0.0.0:${port}"
    local openai_chat_base="${openai_server_base}/v1/chat/completions"
    export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}
    MODEL_NAME=${MODEL_NAME:-$MODEL} # Prefer MODEL_NAME, else MODEL

    set -x
    python3 -m lm_eval --model local-chat-completions --apply_chat_template \
      --tasks "utils/evals/${task}.yaml" \
      --num_fewshot "${num_fewshot}" \
      --output_path "/workspace/${results_dir}" \
      --model_args "model=${MODEL_NAME},base_url=${openai_chat_base},api_key=${OPENAI_API_KEY},eos_string=</s>,max_retries=2,num_concurrent=${concurrent_requests},tokenized_requests=False" \
      --gen_kwargs "max_tokens=${gen_max_tokens},temperature=${temperature},top_p=${top_p}"
    set +x
}

append_lm_eval_summary() {
    set +x
    local results_dir="${EVAL_RESULT_DIR:-eval_out}"
    local task="${EVAL_TASK:-gsm8k}"
    # Always render a local summary so the runner can pick it up
    local out_dir="/workspace/${results_dir}"
    local summary_md="${out_dir}/SUMMARY.md"
    mkdir -p "$out_dir" || true

    python3 utils/lm_eval_to_md.py \
        --results-dir "$out_dir" \
        --task "${task}" \
        --framework "${FRAMEWORK}" \
        --precision "${PRECISION}" \
        --tp "${TP:-1}" \
        --ep "${EP_SIZE:-1}" \
        --dp-attention "${DP_ATTENTION:-false}" \
        > "$summary_md" || true

    # If running inside a GitHub Actions step on this same machine, append there too
    if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
        local GH_SUM_DIR
        GH_SUM_DIR="$(dirname "$GITHUB_STEP_SUMMARY")"
        if [ -d "$GH_SUM_DIR" ] && [ -w "$GH_SUM_DIR" ]; then
            cat "$summary_md" >> "$GITHUB_STEP_SUMMARY" || true
        fi
    fi
}


# ------------------------------
# Lighteval + LiteLLM patching
# ------------------------------

_install_lighteval_deps() {
    set +x
    python3 -m pip install -q --no-cache-dir "lighteval[api]" "litellm" || true
}

# Patch lighteval's LiteLLMClient to handle reasoning content and Python name mangling
# 1. Removed "response_format": {"type": "text"}, as it interferred with vllm endpoint
# 2. Concat reasoning with output tokens as sometimes the output is empty.
_patch_lighteval_litellm() {
    set +x
    local patch_dir
    patch_dir="$(mktemp -d)"
    cat > "$patch_dir/sitecustomize.py" <<'PY'
import logging
import os
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import litellm
from tqdm import tqdm

litellm.suppress_debug_info = True
litellm.drop_params = True

# Remove sglang import that crashes
try:
    # This is where lighteval's is_package_available lives
    from lighteval.utils import imports as le_imports
except Exception:
    le_imports = None
else:
    _orig_is_package_available = le_imports.is_package_available

    def _patched_is_package_available(pkg: str) -> bool:
        # Force "sglang" to look unavailable so that
        # lighteval.models.sglang.sglang_model never imports `sglang`
        if pkg == "sglang":
            return False
        return _orig_is_package_available(pkg)

    le_imports.is_package_available = _patched_is_package_available

from lighteval.models.endpoints.litellm_model import LiteLLMClient
from lighteval.data import GenerativeTaskDataset
from lighteval.tasks.requests import Doc
from lighteval.models.model_output import ModelResponse

logger = logging.getLogger(__name__)

def _patched___call_api(self, prompt, return_logits, max_new_tokens, num_samples, stop_sequence):  # noqa: C901, N802
    from lighteval.models.endpoints.litellm_model import LitellmModelResponse
    response = LitellmModelResponse()
    # Keep dataset-provided stop sequences to cut early
    max_new_tokens = self._prepare_max_new_tokens(max_new_tokens)

    if return_logits and not self.provider == "openai":
        logger.warning("Returning logits is not supported for this provider, ignoring.")

    kwargs = {
        "model": self.model,
        "messages": prompt,
        "max_tokens": max_new_tokens,
        "logprobs": return_logits if self.provider == "openai" else None,
        "stop": stop_sequence,
        "base_url": self.base_url,
        "api_key": self.api_key,
        "n": num_samples,
        "timeout": self.timeout,
    }

    # vLLM/SGLang OpenAI servers: apply chat template and start assistant turn
    if (
        self.provider == "openai"
        and isinstance(self.base_url, str)
        and self.base_url
        and ("api.openai.com" not in self.base_url)
    ):
        kwargs["extra_body"] = {"use_chat_template": True, "add_generation_prompt": True}

    if "o1" in self.model:
        logger.warning("O1 models do not support temperature, top_p, stop sequence. Disabling.")
    else:
        kwargs.update(self.generation_parameters.to_litellm_dict())

    if kwargs.get("max_completion_tokens", None) is None:
        kwargs["max_completion_tokens"] = max_new_tokens

    for attempt in range(self.API_MAX_RETRY):
        try:
            response = litellm.completion(**kwargs)
            msg = response.choices[0].message
            content = getattr(msg, "content", None)
            reasoning = getattr(msg, "reasoning_content", None)

            # Accept reasoning-only replies
            if (not content) and reasoning:
                return response

            if not content and LITELLM_CACHE:
                logger.info("Empty content with caching on; retrying uncached once")
                kwargs["caching"] = False
                response = litellm.completion(**kwargs)

            return response
        except litellm.BadRequestError as e:
            if "message" in e.__dict__ and "policy" in e.__dict__["message"]:
                logger.warning("Content filtered. Returning empty response.")
                return LitellmModelResponse()
        except Exception as e:
            wait_time = min(64, self.API_RETRY_SLEEP * (self.API_RETRY_MULTIPLIER**attempt))
            logger.warning(f"Error: {e}, waiting {wait_time}s before retry {attempt + 1}/{self.API_MAX_RETRY}")
            time.sleep(wait_time)

    logger.error(f"API call failed after {self.API_MAX_RETRY} attempts.")
    return LitellmModelResponse()


def _patched___call_api_parallel(self, prompts, return_logits, max_new_tokens, num_samples, stop_sequence):  # noqa: N802
    # Build per-item args
    return_logitss = [return_logits for _ in prompts] if not isinstance(return_logits, list) else return_logits
    max_new_tokenss = [max_new_tokens for _ in prompts] if not isinstance(max_new_tokens, list) else max_new_tokens
    num_sampless = [num_samples for _ in prompts] if not isinstance(num_samples, list) else num_samples
    stop_sequencess = [stop_sequence for _ in prompts]

    n = len(prompts)
    assert n == len(return_logitss) == len(max_new_tokenss) == len(num_sampless) == len(stop_sequencess), (
        f"Length mismatch: {len(prompts)}, {len(return_logitss)}, {len(max_new_tokenss)}, "
        f"{len(num_sampless)}, {len(stop_sequencess)}"
    )

    results = [None] * n
    with ThreadPoolExecutor(self.concurrent_requests) as executor:
        futures = []
        for idx in range(n):
            fut = executor.submit(
                self._LiteLLMClient__call_api,
                prompts[idx],
                return_logitss[idx],
                max_new_tokenss[idx],
                num_sampless[idx],
                stop_sequencess[idx],
            )
            fut._le_idx = idx  # attach index for order restoration
            futures.append(fut)

        for fut in tqdm(as_completed(futures), total=n, disable=self.disable_tqdm):
            idx = getattr(fut, "_le_idx", None)
            try:
                res = fut.result()
            except Exception:
                res = None
            if idx is not None:
                results[idx] = res

    if any(r is None for r in results):
        raise ValueError("Some entries are not annotated due to errors in __call_api_parallel, please inspect and retry.")

    return results


def _greedy_until_impl(self, docs: list[Doc]) -> list[ModelResponse]:
    dataset = GenerativeTaskDataset(requests=docs, num_dataset_splits=self.DATASET_SPLITS)
    results: list[ModelResponse] = []

    for split in tqdm(
        dataset.splits_iterator(),
        total=dataset.num_dataset_splits,
        desc="Splits",
        position=0,
        disable=self.disable_tqdm,
    ):
        contexts = [self.prompt_manager.prepare_prompt_api(doc) for doc in split]

        max_new_tokens = split[0].generation_size
        return_logits = split[0].use_logits
        num_samples = split[0].num_samples
        stop_sequence = split[0].stop_sequences

        if num_samples > 1 and self.generation_parameters.temperature == 0:
            raise ValueError("num_samples > 1 requires temperature > 0")

        responses = self._LiteLLMClient__call_api_parallel(
            contexts,
            return_logits,
            max_new_tokens,
            num_samples,
            stop_sequence,
        )

        for response, context in zip(responses, contexts):
            merged_texts: list[str] = []
            reasonings: list[str | None] = []

            for choice in response.choices:
                msg = choice.message
                raw_content = getattr(msg, "content", None) or ""
                reasoning = getattr(msg, "reasoning_content", None)

                # For answer extraction, use only the content field
                # The reasoning is stored separately for logging/debugging
                merged_texts.append(raw_content.strip() if raw_content else "")
                reasonings.append(reasoning if reasoning else None)

            if not merged_texts or merged_texts[0] is None:
                merged_texts = [""]

            results.append(
                ModelResponse(
                    text=merged_texts,
                    reasonings=reasonings,
                    input=context,
                )
            )

    if len(results) != len(dataset):
        raise RuntimeError(f"Internal mismatch: {len(results)} outputs vs {len(dataset)} docs.")

    return dataset.get_original_order(results)

# Bind patches
LiteLLMClient._LiteLLMClient__call_api = _patched___call_api
LiteLLMClient._LiteLLMClient__call_api_parallel = _patched___call_api_parallel
#LiteLLMClient.greedy_until = _greedy_until_impl
PY
    export PYTHONPATH="${patch_dir}:${PYTHONPATH:-}"
}

run_lighteval_eval() {
    set +x
    local port="${PORT:-8888}"
    local task="${EVAL_TASK:-gsm8k}"
    local num_fewshot="${NUM_FEWSHOT:-5}"
    local results_dir="${EVAL_RESULT_DIR:-eval_out_lighteval}"
    local max_samples=0
    local concurrent_requests=32

    while [[ $# -gt 0 ]]; do
        case $1 in
            --port)        port="$2"; shift 2 ;;
            --task)        task="$2"; shift 2 ;;
            --num-fewshot) num_fewshot="$2"; shift 2 ;;
            --results-dir) results_dir="$2"; shift 2 ;;
            --max-samples) max_samples="$2"; shift 2 ;;
            --concurrent-requests) concurrent_requests="$2"; shift 2 ;;
            *)             echo "Unknown parameter: $1"; return 1 ;;
        esac
    done

    _install_lighteval_deps
    _patch_lighteval_litellm

    # Prefer OPENAI_MODEL_NAME, then EVAL_MODEL_NAME, then MODEL
    local model_name="${MODEL_NAME}"
    if [[ -z "$model_name" ]]; then
        echo "Error: MODEL not set for lighteval." >&2
        return 1
    fi

    # LiteLLM provider prefix logic
    local lite_model="$model_name"
    if [[ "$lite_model" != openai/* ]]; then
        lite_model="openai/${lite_model}"
    fi

    local base_url="http://0.0.0.0:${port}/v1"
    export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

    local MODEL_ARGS="model_name=${lite_model},base_url=${base_url},api_key=${OPENAI_API_KEY},generation_parameters={temperature:0.0,top_p=1,max_new_tokens:2048},concurrent_requests=${concurrent_requests}"
    local TASK_SPEC="${task}|${num_fewshot}"

    set -x
    lighteval endpoint litellm \
        "${MODEL_ARGS}" \
        "${TASK_SPEC}" \
        --output-dir "/workspace/${results_dir}" \
        --custom-tasks utils/evals/custom_gsm8k.py \
        --max-samples "${max_samples}"
    set +x
}


# ------------------------------
# Unified eval entrypoint
# ------------------------------

run_eval() {
    set +x
    local framework="${EVAL_FRAMEWORK:-lm-eval}"
    local forwarded=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --framework) framework="$2"; shift 2 ;;
            *)           forwarded+=("$1"); shift ;;
        esac
    done

    case "$framework" in
        lm-eval|lm_eval) run_lm_eval "${forwarded[@]}" ;;
        lighteval)       run_lighteval_eval "${forwarded[@]}" ;;
        *)               echo "Unknown framework '${framework}'"; return 1 ;;
    esac
}
