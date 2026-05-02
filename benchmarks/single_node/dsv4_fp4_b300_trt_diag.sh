#!/usr/bin/env bash

# Temporary B300/TRTLLM DeepSeek-V4 diagnostic.
#
# This isolates the current garbage-output failure by checking completion vs
# chat endpoint behavior with strict final-answer scoring, then comparing the
# baseline path against targeted config/topology ablations:
#   1. Default/auto KV cache, same CUDA graph config.
#   2. Auto KV cache with TRTLLM autotuner disabled.
#   3. Auto KV cache with the vanilla MoE backend.
#   4. FP8 KV cache with CUDA graph disabled.
#   5. num_postprocess_workers=1 for postprocess/scatter race isolation.
#   6. TP/EP/DPA topology controls.
#   7. Explicit KV dtype controls if the branch accepts them.
#
# Each live server gets a queued probe batch that records local prompt token IDs,
# decoded prompts with and without special tokens, one-token/logprob probes, and
# manual DeepSeek-V4 prompt variants from utils/bench_serving/encoding_dsv4.py.
#
# The runner routes only the representative B300 DeepSeek-V4 TRT job here.

set -euo pipefail

source "$(dirname "$0")/../benchmark_lib.sh"
source "$(dirname "$0")/trtllm_dsv4_bootstrap.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    MAX_MODEL_LEN \
    RESULT_FILENAME \
    DP_ATTENTION \
    EP_SIZE

export TRTLLM_DSV4_USE_MPIRUN="${TRTLLM_DSV4_USE_MPIRUN:-1}"
export TRTLLM_DSV4_SANITIZE_SLURM_MPI_ENV="${TRTLLM_DSV4_SANITIZE_SLURM_MPI_ENV:-1}"
export TRTLLM_DSV4_BOOTSTRAP="${TRTLLM_DSV4_BOOTSTRAP:-0}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

SERVER_LOG="$PWD/server.log"
DIAG_JSONL="$PWD/dsv4_trt_b300_diag.jsonl"
DIAG_SUMMARY_JSON="$PWD/dsv4_trt_b300_diag_summary.json"
PORT_BASE="${PORT:-8888}"
MAX_BATCH_SIZE="${TRTLLM_DSV4_DIAG_MAX_BATCH_SIZE:-$(( CONC > 16 ? CONC : 16 ))}"
KV_CACHE_FREE_MEM_FRACTION="${KV_CACHE_FREE_MEM_FRACTION:-0.50}"
DIAG_MAX_MODEL_LEN="${TRTLLM_DSV4_DIAG_MAX_MODEL_LEN:-$MAX_MODEL_LEN}"
DIAG_MAX_NUM_TOKENS="${TRTLLM_DSV4_DIAG_MAX_NUM_TOKENS:-$MAX_MODEL_LEN}"
DIAG_LOG_LEVEL="${TRTLLM_DSV4_DIAG_LOG_LEVEL:-debug}"

if (( DIAG_MAX_MODEL_LEN < 9472 )); then
    DIAG_MAX_MODEL_LEN=9472
fi
if (( DIAG_MAX_NUM_TOKENS < DIAG_MAX_MODEL_LEN )); then
    DIAG_MAX_NUM_TOKENS="$DIAG_MAX_MODEL_LEN"
fi

: > "$SERVER_LOG"
: > "$DIAG_JSONL"

log() {
    echo "$@" | tee -a "$SERVER_LOG"
}

sanitize_slurm_mpi_env_for_trtllm() {
    if [[ "${TRTLLM_DSV4_SANITIZE_SLURM_MPI_ENV:-0}" != "1" ]]; then
        return 0
    fi

    log "Sanitizing Slurm/PMI environment for TensorRT-LLM launch"
    while IFS='=' read -r name _; do
        case "$name" in
            SLURM_*|PMIX*|PMI*|OMPI_*|ORTE_*)
                unset "$name"
                ;;
        esac
    done < <(env)
}

write_config() {
    local config_file="$1"
    local kv_dtype="$2"
    local graph_mode="$3"
    local moe_backend="$4"
    local autotuner="$5"
    local dp_attention="$6"
    local postprocess_workers="$7"

    local attention_dp_config=""
    if [[ "$dp_attention" == "true" ]]; then
        attention_dp_config="
attention_dp_config:
    batching_wait_iters: 0
    enable_balance: true
    timeout_iters: 60"
    fi

    {
        if [[ "$graph_mode" == "on" ]]; then
            cat <<EOF
cuda_graph_config:
    enable_padding: true
    max_batch_size: $MAX_BATCH_SIZE
EOF
        else
            cat <<'EOF'
cuda_graph_config: null
EOF
        fi

        if [[ "$autotuner" != "default" ]]; then
            printf 'enable_autotuner: %s\n' "$autotuner"
        fi

        cat <<EOF
enable_attention_dp: $dp_attention$attention_dp_config
print_iter_log: true
kv_cache_config:
    tokens_per_block: 128
EOF
        if [[ "$kv_dtype" != "unset" ]]; then
            printf '    dtype: %s\n' "$kv_dtype"
        fi
        cat <<EOF
    free_gpu_memory_fraction: $KV_CACHE_FREE_MEM_FRACTION
    enable_block_reuse: false
stream_interval: 10
num_postprocess_workers: $postprocess_workers
moe_config:
    backend: $moe_backend
EOF
    } > "$config_file"
}

write_placeholder_outputs() {
    local pass_metric="$1"
    python3 - "$RESULT_FILENAME" "$MODEL" "$pass_metric" <<'PY'
import json
import sys

result_filename, model, pass_metric = sys.argv[1], sys.argv[2], float(sys.argv[3])

benchmark = {
    "model_id": model,
    "max_concurrency": 1,
    "total_token_throughput": 0.0,
    "output_throughput": 0.0,
    "mean_ttft_ms": 0.0,
    "p50_ttft_ms": 0.0,
    "p90_ttft_ms": 0.0,
    "p99_ttft_ms": 0.0,
    "mean_e2el_ms": 0.0,
    "p50_e2el_ms": 0.0,
    "p90_e2el_ms": 0.0,
    "p99_e2el_ms": 0.0,
}
with open(f"{result_filename}.json", "w") as f:
    json.dump(benchmark, f, indent=2)

eval_result = {
    "results": {
        "gsm8k": {
            "exact_match,strict-match": pass_metric,
            "exact_match,flexible-extract": pass_metric,
        }
    },
    "versions": {"gsm8k": 0},
    "config": {
        "note": "temporary TRTLLM DeepSeek-V4 B300 diagnostic placeholder; inspect dsv4_trt_b300_diag_summary.json for probe results"
    },
}
with open("results_dsv4_trt_b300_diag.json", "w") as f:
    json.dump(eval_result, f, indent=2)

with open("meta_env.json", "w") as f:
    json.dump({
        "diagnostic": "dsv4_trt_b300",
        "placeholder_eval_metric": pass_metric,
        "summary_json": "dsv4_trt_b300_diag_summary.json",
    }, f, indent=2)
PY
}

cleanup_server() {
    local server_pid="${1:-}"
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid" 2>/dev/null || true
        for _ in {1..20}; do
            if ! kill -0 "$server_pid" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        kill -9 "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}

ensure_fast_hadamard_transform() {
    if [[ "${TRTLLM_DSV4_DIAG_INSTALL_FHT:-1}" != "1" ]]; then
        log "TRTLLM_DSV4_DIAG_INSTALL_FHT!=1; not installing fast_hadamard_transform"
        return 0
    fi

    if python3 - <<'PY' >/dev/null 2>&1
import fast_hadamard_transform  # noqa: F401
PY
    then
        log "fast_hadamard_transform already importable"
        return 0
    fi

    log "fast_hadamard_transform missing; attempting runtime install"
    set +e
    python3 -m pip install --no-cache-dir --no-build-isolation \
        "git+https://github.com/Dao-AILab/fast-hadamard-transform.git" \
        2>&1 | tee -a "$SERVER_LOG"
    local install_status=${PIPESTATUS[0]}
    set -e

    if [[ "$install_status" != "0" ]]; then
        log "WARNING: fast_hadamard_transform install failed with status $install_status; continuing without it"
        return 0
    fi

    if python3 - <<'PY' >/dev/null 2>&1
import fast_hadamard_transform  # noqa: F401
PY
    then
        log "fast_hadamard_transform import succeeded after install"
    else
        log "WARNING: fast_hadamard_transform still not importable after install"
    fi
}

run_client_probe() {
    local variant="$1"
    local port="$2"
    local output_json="$3"

    VARIANT="$variant" PORT="$port" MODEL="$MODEL" OUTPUT_JSON="$output_json" python3 - <<'PY'
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import sys
import urllib.error
import urllib.request

variant = os.environ["VARIANT"]
port = os.environ["PORT"]
model = os.environ["MODEL"]
output_json = os.environ["OUTPUT_JSON"]
padding_lines = int(os.environ.get("TRTLLM_DSV4_DIAG_PADDING_LINES", "550"))
probe_workers = int(os.environ.get("TRTLLM_DSV4_DIAG_PROBE_WORKERS", "4"))
completion_logprobs = int(os.environ.get("TRTLLM_DSV4_DIAG_COMPLETION_LOGPROBS", "20"))
chat_top_logprobs = int(os.environ.get("TRTLLM_DSV4_DIAG_CHAT_TOP_LOGPROBS", "20"))

bench_utils = Path.cwd() / "utils" / "bench_serving"
if bench_utils.exists():
    sys.path.insert(0, str(bench_utils))

filler = (
    "This line is padding context for a deterministic math probe and should be ignored.\n"
    * padding_lines
)

short_math = "Answer with the final integer only. What is 2 + 2?"
gsm8k_like = (
    "Answer math word problems. Put the final answer as #### <number>.\n\n"
    "Q: Sarah has 3 boxes with 4 pencils in each box. How many pencils does she have?\n"
    "A: Sarah has 3 * 4 = 12 pencils. #### 12\n\n"
    "Q: A store had 20 oranges and sold 7. How many oranges remain?\n"
    "A: The store has 20 - 7 = 13 oranges left. #### 13\n\n"
    "Q: James has 6 apples, buys 7 more, and gives away 5. How many apples does James have left?\n"
    "A:"
)
long_prefill_math = (
    filler
    + "\nIgnore the padding above. Answer with the final integer only. "
    + "James has 6 apples, buys 7 more, and gives away 5. How many apples does James have left?"
)

probes = [
    {
        "name": "tiny_completion",
        "endpoint": "completion",
        "expected": 4,
        "max_tokens": 4,
        "content": "2+2=",
    },
    {
        "name": "tiny_completion_first_token",
        "endpoint": "completion",
        "expected": None,
        "max_tokens": 1,
        "content": "2+2=",
        "diagnostic_only": True,
    },
    {
        "name": "short_math_completion",
        "endpoint": "completion",
        "expected": 4,
        "max_tokens": 96,
        "content": short_math,
    },
    {
        "name": "short_math_completion_first_token",
        "endpoint": "completion",
        "expected": None,
        "max_tokens": 1,
        "content": short_math,
        "diagnostic_only": True,
    },
    {
        "name": "short_math_dsv4_thinking_completion",
        "endpoint": "dsv4_thinking_completion",
        "expected": 4,
        "max_tokens": 96,
        "content": short_math,
    },
    {
        "name": "short_math_dsv4_thinking_first_token",
        "endpoint": "dsv4_thinking_completion",
        "expected": None,
        "max_tokens": 1,
        "content": short_math,
        "diagnostic_only": True,
    },
    {
        "name": "short_math_dsv4_chat_completion",
        "endpoint": "dsv4_chat_completion",
        "expected": 4,
        "max_tokens": 96,
        "content": short_math,
    },
    {
        "name": "short_math_manual_eot_thinking_completion",
        "endpoint": "manual_eot_thinking_completion",
        "expected": 4,
        "max_tokens": 96,
        "content": short_math,
    },
    {
        "name": "short_math_chat",
        "endpoint": "chat",
        "expected": 4,
        "max_tokens": 96,
        "content": short_math,
    },
    {
        "name": "short_math_hf_template_completion",
        "endpoint": "hf_template_completion",
        "expected": 4,
        "max_tokens": 96,
        "content": short_math,
    },
    {
        "name": "gsm8k_like_chat",
        "endpoint": "chat",
        "expected": 8,
        "max_tokens": 96,
        "content": gsm8k_like,
    },
    {
        "name": "gsm8k_like_dsv4_thinking_completion",
        "endpoint": "dsv4_thinking_completion",
        "expected": 8,
        "max_tokens": 128,
        "content": gsm8k_like,
    },
    {
        "name": "gsm8k_like_hf_template_completion",
        "endpoint": "hf_template_completion",
        "expected": 8,
        "max_tokens": 96,
        "content": gsm8k_like,
    },
    {
        "name": "long_prefill_math_chat",
        "endpoint": "chat",
        "expected": 8,
        "max_tokens": 96,
        "content": long_prefill_math,
    },
    {
        "name": "long_prefill_math_dsv4_thinking_completion",
        "endpoint": "dsv4_thinking_completion",
        "expected": 8,
        "max_tokens": 96,
        "content": long_prefill_math,
    },
]

def exact_final_number(text: str, expected: int) -> bool:
    text = (text or "").strip()
    match = re.search(r"(?<![\d.])(?:####\s*)?(-?\d+(?:\.0+)?)\s*$", text)
    if not match:
        return False
    try:
        return float(match.group(1)) == float(expected)
    except ValueError:
        return False

def token_list(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    return list(value or [])

tokenizer = None
tokenizer_info = {"model": model}
try:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    tokenizer_info.update({
        "loaded": True,
        "class": type(tokenizer).__name__,
        "chat_template_preview": (getattr(tokenizer, "chat_template", None) or "")[:1000],
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "special_tokens_map": {
            str(key): str(value)
            for key, value in (getattr(tokenizer, "special_tokens_map", {}) or {}).items()
        },
        "additional_special_tokens": [
            str(token) for token in (getattr(tokenizer, "additional_special_tokens", []) or [])
        ],
        "eos_token": getattr(tokenizer, "eos_token", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token": getattr(tokenizer, "pad_token", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
    })
except Exception as exc:
    tokenizer_info.update({"loaded": False, "error": repr(exc)})

try:
    from encoding_dsv4 import encode_messages as dsv4_encode_messages
    tokenizer_info["encoding_dsv4_imported"] = True
except Exception as exc:
    dsv4_encode_messages = None
    tokenizer_info["encoding_dsv4_imported"] = False
    tokenizer_info["encoding_dsv4_error"] = repr(exc)

def render_hf_chat_prompt(content: str) -> str:
    if tokenizer is None:
        raise RuntimeError(f"HF tokenizer unavailable: {tokenizer_info.get('error')}")
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )

def render_dsv4_prompt(content: str, thinking_mode: str) -> str:
    if dsv4_encode_messages is None:
        raise RuntimeError(f"encoding_dsv4 unavailable: {tokenizer_info.get('encoding_dsv4_error')}")
    return dsv4_encode_messages(
        [{"role": "user", "content": content}],
        thinking_mode=thinking_mode,
    )

def render_manual_eot_thinking_prompt(content: str) -> str:
    return (
        "<｜begin▁of▁sentence｜><｜User｜>"
        + content
        + "<|EOT|><｜Assistant｜><think>"
    )

def analyze_text(label: str, text: str) -> dict:
    result = {
        "label": label,
        "repr": repr(text),
        "char_len": len(text or ""),
        "preview": (text or "")[:1000],
    }
    if tokenizer is None:
        result["tokenizer_error"] = tokenizer_info.get("error") or "tokenizer unavailable"
        return result
    try:
        ids = tokenizer.encode(text or "", add_special_tokens=False)
        result.update({
            "token_count": len(ids),
            "token_ids_head": ids[:80],
            "token_ids_tail": ids[-80:],
            "tokens_head": tokenizer.convert_ids_to_tokens(ids[:40]),
            "tokens_tail": tokenizer.convert_ids_to_tokens(ids[-40:]),
            "decoded_skip_special_false_preview": tokenizer.decode(ids, skip_special_tokens=False)[:1000],
            "decoded_skip_special_true_preview": tokenizer.decode(ids, skip_special_tokens=True)[:1000],
        })
    except Exception as exc:
        result["tokenize_error"] = repr(exc)
    return result

def compact_logprobs(logprobs):
    if not logprobs:
        return None
    if isinstance(logprobs, dict):
        compact = {}
        if "tokens" in logprobs:
            tokens = logprobs.get("tokens") or []
            compact["tokens_head"] = tokens[:32]
            compact["tokens_tail"] = tokens[-32:]
            compact["num_tokens"] = len(tokens)
        if "token_logprobs" in logprobs:
            vals = logprobs.get("token_logprobs") or []
            compact["token_logprobs_head"] = vals[:32]
            compact["token_logprobs_tail"] = vals[-32:]
        if "top_logprobs" in logprobs:
            vals = logprobs.get("top_logprobs") or []
            compact["top_logprobs_head"] = vals[:4]
            compact["top_logprobs_tail"] = vals[-4:]
        if "content" in logprobs:
            content = logprobs.get("content") or []
            compact["content_head"] = content[:8]
            compact["content_tail"] = content[-8:]
            compact["num_content_tokens"] = len(content)
        return compact or {"raw_preview": json.dumps(logprobs, ensure_ascii=False)[:4000]}
    return {"raw_preview": json.dumps(logprobs, ensure_ascii=False)[:4000]}

def summarize_body(body: dict) -> dict:
    summary = {
        "id": body.get("id"),
        "object": body.get("object"),
        "created": body.get("created"),
        "model": body.get("model"),
        "usage": body.get("usage"),
    }
    choices = body.get("choices") or []
    if choices:
        choice = choices[0]
        summary["choice"] = {
            "index": choice.get("index"),
            "finish_reason": choice.get("finish_reason"),
            "logprobs": compact_logprobs(choice.get("logprobs")),
        }
        if "message" in choice:
            msg = choice.get("message") or {}
            summary["choice"]["message_keys"] = sorted(msg.keys())
            summary["choice"]["message_content_repr"] = repr(msg.get("content"))
            summary["choice"]["message_reasoning_content_repr"] = repr(msg.get("reasoning_content"))
        if "text" in choice:
            summary["choice"]["text_repr"] = repr(choice.get("text"))
    return summary

def post_json(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://0.0.0.0:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {err_body[:2000]}") from exc

def complete_chat(content: str, max_tokens: int) -> tuple[str, str, dict, str, dict, dict]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
    }
    diagnostics = {}
    if chat_top_logprobs > 0:
        payload["logprobs"] = True
        payload["top_logprobs"] = chat_top_logprobs
    try:
        body = post_json("/v1/chat/completions", payload)
    except Exception as exc:
        diagnostics["logprobs_request_error"] = repr(exc)
        payload.pop("logprobs", None)
        payload.pop("top_logprobs", None)
        body = post_json("/v1/chat/completions", payload)
    choice = body["choices"][0]
    message = choice.get("message") or {}
    return (
        message.get("content") or "",
        message.get("reasoning_content") or "",
        body.get("usage") or {},
        choice.get("finish_reason"),
        body,
        diagnostics,
    )

def complete_text(content: str, max_tokens: int) -> tuple[str, str, dict, str, dict, dict]:
    payload = {
        "model": model,
        "prompt": content,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
    }
    diagnostics = {}
    if completion_logprobs > 0:
        payload["logprobs"] = completion_logprobs
    try:
        body = post_json("/v1/completions", payload)
    except Exception as exc:
        diagnostics["logprobs_request_error"] = repr(exc)
        payload.pop("logprobs", None)
        body = post_json("/v1/completions", payload)
    choice = body["choices"][0]
    return choice.get("text") or "", "", body.get("usage") or {}, choice.get("finish_reason"), body, diagnostics

def render_probe_prompt(probe: dict) -> tuple[str, str]:
    endpoint = probe["endpoint"]
    content = probe["content"]
    if endpoint == "hf_template_completion":
        return "completion", render_hf_chat_prompt(content)
    if endpoint == "dsv4_thinking_completion":
        return "completion", render_dsv4_prompt(content, "thinking")
    if endpoint == "dsv4_chat_completion":
        return "completion", render_dsv4_prompt(content, "chat")
    if endpoint == "manual_eot_thinking_completion":
        return "completion", render_manual_eot_thinking_prompt(content)
    return endpoint, content

def run_one_probe(probe: dict) -> dict:
    try:
        request_endpoint, prompt = render_probe_prompt(probe)
        if request_endpoint == "completion":
            text, reasoning_text, usage, finish_reason, body, request_diagnostics = complete_text(prompt, probe["max_tokens"])
        elif request_endpoint == "chat":
            text, reasoning_text, usage, finish_reason, body, request_diagnostics = complete_chat(prompt, probe["max_tokens"])
        else:
            raise RuntimeError(f"unknown endpoint type {probe['endpoint']} rendered to {request_endpoint}")
        expected = probe.get("expected")
        scored_text = text if text else reasoning_text
        exact_ok = False if expected is None else exact_final_number(scored_text, expected)
        return {
            "name": probe["name"],
            "endpoint": probe["endpoint"],
            "request_endpoint": request_endpoint,
            "expected": expected,
            "diagnostic_only": bool(probe.get("diagnostic_only", False)),
            "expected_found": exact_ok,
            "exact_final_answer": exact_ok,
            "prompt_preview": prompt[:500],
            "prompt_analysis": analyze_text("prompt", prompt),
            "output_analysis": analyze_text("output_text", text),
            "reasoning_output_analysis": analyze_text("reasoning_text", reasoning_text),
            "usage": usage,
            "finish_reason": finish_reason,
            "raw": text,
            "raw_repr": repr(text),
            "raw_preview": text[:500],
            "reasoning_raw": reasoning_text,
            "reasoning_raw_repr": repr(reasoning_text),
            "request_diagnostics": request_diagnostics,
            "response_summary": summarize_body(body),
        }
    except Exception as exc:
        return {
            "name": probe["name"],
            "endpoint": probe["endpoint"],
            "expected": probe.get("expected"),
            "diagnostic_only": bool(probe.get("diagnostic_only", False)),
            "expected_found": False,
            "exact_final_answer": False,
            "prompt_analysis": analyze_text("raw_user_content", probe.get("content") or ""),
            "error": repr(exc),
        }

results_by_name = {}
with ThreadPoolExecutor(max_workers=max(1, probe_workers)) as executor:
    future_to_probe = {executor.submit(run_one_probe, probe): probe for probe in probes}
    for future in as_completed(future_to_probe):
        result = future.result()
        results_by_name[result["name"]] = result

results = [results_by_name[probe["name"]] for probe in probes if probe["name"] in results_by_name]
ok_count = sum(
    int(result.get("exact_final_answer", False))
    for result in results
    if not result.get("diagnostic_only", False)
)

by_name = {result["name"]: result for result in results}
def probe_ok(name: str) -> bool:
    return bool(by_name.get(name, {}).get("exact_final_answer", False))

tiny_completion_ok = probe_ok("tiny_completion")
short_completion_ok = probe_ok("short_math_completion")
short_chat_ok = probe_ok("short_math_chat")
gsm8k_chat_ok = probe_ok("gsm8k_like_chat")
short_hf_template_completion_ok = probe_ok("short_math_hf_template_completion")
gsm8k_hf_template_completion_ok = probe_ok("gsm8k_like_hf_template_completion")
short_dsv4_thinking_completion_ok = probe_ok("short_math_dsv4_thinking_completion")
short_dsv4_chat_completion_ok = probe_ok("short_math_dsv4_chat_completion")
short_manual_eot_thinking_completion_ok = probe_ok("short_math_manual_eot_thinking_completion")
gsm8k_dsv4_thinking_completion_ok = probe_ok("gsm8k_like_dsv4_thinking_completion")
long_prefill_chat_ok = probe_ok("long_prefill_math_chat")
long_prefill_dsv4_thinking_completion_ok = probe_ok("long_prefill_math_dsv4_thinking_completion")

completion_ok = tiny_completion_ok and short_completion_ok
chat_ok = short_chat_ok and gsm8k_chat_ok
hf_template_completion_ok = (
    short_hf_template_completion_ok and gsm8k_hf_template_completion_ok
)
dsv4_template_completion_ok = (
    short_dsv4_thinking_completion_ok or short_dsv4_chat_completion_ok
)
nontrivial_completion_ok = (
    short_completion_ok
    or short_dsv4_thinking_completion_ok
    or short_dsv4_chat_completion_ok
    or short_manual_eot_thinking_completion_ok
)
gsm8k_semantic_ok = gsm8k_chat_ok or gsm8k_dsv4_thinking_completion_ok

summary = {
    "variant": variant,
    "ok": tiny_completion_ok and nontrivial_completion_ok and gsm8k_semantic_ok,
    "ok_count": ok_count,
    "num_probes": len(probes),
    "probe_workers": probe_workers,
    "completion_ok": completion_ok,
    "nontrivial_completion_ok": nontrivial_completion_ok,
    "chat_ok": chat_ok,
    "hf_template_completion_ok": hf_template_completion_ok,
    "dsv4_template_completion_ok": dsv4_template_completion_ok,
    "endpoint_split_suspect": nontrivial_completion_ok and not chat_ok,
    "long_prefill_chat_ok": long_prefill_chat_ok,
    "long_prefill_dsv4_thinking_completion_ok": long_prefill_dsv4_thinking_completion_ok,
    "required_probe_status": {
        "tiny_completion_ok": tiny_completion_ok,
        "short_completion_ok": short_completion_ok,
        "short_chat_ok": short_chat_ok,
        "gsm8k_chat_ok": gsm8k_chat_ok,
        "short_hf_template_completion_ok": short_hf_template_completion_ok,
        "gsm8k_hf_template_completion_ok": gsm8k_hf_template_completion_ok,
        "short_dsv4_thinking_completion_ok": short_dsv4_thinking_completion_ok,
        "short_dsv4_chat_completion_ok": short_dsv4_chat_completion_ok,
        "short_manual_eot_thinking_completion_ok": short_manual_eot_thinking_completion_ok,
        "gsm8k_dsv4_thinking_completion_ok": gsm8k_dsv4_thinking_completion_ok,
        "long_prefill_chat_ok": long_prefill_chat_ok,
        "long_prefill_dsv4_thinking_completion_ok": long_prefill_dsv4_thinking_completion_ok,
    },
    "tokenizer": tokenizer_info,
    "probes": results,
}

with open(output_json, "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(json.dumps(summary, ensure_ascii=False))
sys.exit(0 if summary["ok"] else 3)
PY
}

run_variant() {
    local variant="$1"
    local kv_dtype="$2"
    local graph_mode="$3"
    local port="$4"
    local moe_backend="$5"
    local autotuner="$6"
    local variant_tp="${7:-$TP}"
    local variant_ep_size="${8:-$EP_SIZE}"
    local variant_dp_attention="${9:-$DP_ATTENTION}"
    local postprocess_workers="${10:-4}"
    local config_file="dsv4-fp4-trt-${variant}.yml"
    local variant_log="/tmp/dsv4_trt_${variant}_server.log"
    local probe_json="/tmp/dsv4_trt_${variant}_probe.json"
    local server_pid=""
    local ready=0
    local probe_status=1

    log
    log "===== TRTLLM DSV4 DIAGNOSTIC VARIANT: $variant ====="
    log "kv_dtype=$kv_dtype cuda_graph=$graph_mode moe_backend=$moe_backend autotuner=$autotuner tp=$variant_tp ep_size=$variant_ep_size dp_attention=$variant_dp_attention num_postprocess_workers=$postprocess_workers port=$port"

    write_config "$config_file" "$kv_dtype" "$graph_mode" "$moe_backend" "$autotuner" "$variant_dp_attention" "$postprocess_workers"
    log "Generated config $config_file:"
    sed 's/^/[config] /' "$config_file" | tee -a "$SERVER_LOG"

    SERVE_CMD=(
        trtllm-serve "$MODEL"
        --host 0.0.0.0
        --port "$port"
        --trust_remote_code
        --backend pytorch
        --log_level "$DIAG_LOG_LEVEL"
        --max_batch_size "$MAX_BATCH_SIZE"
        --max_seq_len "$DIAG_MAX_MODEL_LEN"
        --max_num_tokens "$DIAG_MAX_NUM_TOKENS"
        --tp_size "$variant_tp"
        --ep_size "$variant_ep_size"
        --custom_tokenizer deepseek_v4
        --config "$config_file"
    )

    if [[ "${TRTLLM_DSV4_USE_MPIRUN:-1}" == "0" ]]; then
        "${SERVE_CMD[@]}" > "$variant_log" 2>&1 &
    else
        mpirun -n 1 --oversubscribe --allow-run-as-root \
            "${SERVE_CMD[@]}" \
            > "$variant_log" 2>&1 &
    fi
    server_pid=$!

    if ( wait_for_server_ready --port "$port" --server-log "$variant_log" --server-pid "$server_pid" --sleep-interval 5 ); then
        ready=1
        log "Variant $variant became healthy; sending deterministic probes"
        set +e
        run_client_probe "$variant" "$port" "$probe_json" | tee -a "$SERVER_LOG"
        probe_status=${PIPESTATUS[0]}
        set -e
    else
        log "Variant $variant failed before readiness"
    fi

    cleanup_server "$server_pid"

    log "----- server log for $variant -----"
    sed "s/^/[$variant] /" "$variant_log" | tee -a "$SERVER_LOG" >/dev/null || true
    log "----- end server log for $variant -----"

    local kvcache_nan=0
    local hadamard_missing=0
    if grep -q "NaNs/Infs have been introduced to KVCache" "$variant_log"; then
        kvcache_nan=1
    fi
    if grep -qi "fast-hadamard-transform not available\\|skip hadamard" "$variant_log"; then
        hadamard_missing=1
    fi

    python3 - "$variant" "$kv_dtype" "$graph_mode" "$moe_backend" "$autotuner" "$variant_tp" "$variant_ep_size" "$variant_dp_attention" "$postprocess_workers" "$ready" "$probe_status" "$kvcache_nan" "$hadamard_missing" "$probe_json" "$DIAG_JSONL" "$variant_log" <<'PY'
import json
import os
import re
import sys

variant, kv_dtype, graph_mode = sys.argv[1], sys.argv[2], sys.argv[3]
moe_backend, autotuner = sys.argv[4], sys.argv[5]
variant_tp, variant_ep_size = sys.argv[6], sys.argv[7]
variant_dp_attention = sys.argv[8]
postprocess_workers = sys.argv[9]
ready = bool(int(sys.argv[10]))
probe_status = int(sys.argv[11])
kvcache_nan = bool(int(sys.argv[12]))
hadamard_missing = bool(int(sys.argv[13]))
probe_json = sys.argv[14]
diag_jsonl = sys.argv[15]
variant_log = sys.argv[16]

probe = {}
if os.path.exists(probe_json):
    with open(probe_json) as f:
        probe = json.load(f)

def matching_log_lines(pattern: str, limit: int = 30):
    if not os.path.exists(variant_log):
        return []
    expr = re.compile(pattern, re.IGNORECASE)
    matches = []
    with open(variant_log, errors="replace") as f:
        for line in f:
            if expr.search(line):
                matches.append(line.rstrip()[:1200])
                if len(matches) >= limit:
                    break
    return matches

kv_dtype_lines = matching_log_lines(r"kv.*dtype|cache.*dtype|KVCache", 40)
moe_warning_lines = matching_log_lines(r"mxe4m3_mxe2m1_block_scale_moe_runner|no valid tactic|moe", 40)
mhc_warning_lines = matching_log_lines(r"mhc_fused_hc|fused_hc|hyper.?connection|\\bmhc\\b", 40)

row = {
    "variant": variant,
    "kv_dtype": kv_dtype,
    "cuda_graph": graph_mode,
    "moe_backend": moe_backend,
    "autotuner": autotuner,
    "tp": variant_tp,
    "ep_size": variant_ep_size,
    "dp_attention": variant_dp_attention == "true",
    "num_postprocess_workers": int(postprocess_workers),
    "ready": ready,
    "probe_status": probe_status,
    "probe_ok": bool(probe.get("ok", False)),
    "ok_count": probe.get("ok_count", 0),
    "completion_ok": bool(probe.get("completion_ok", False)),
    "chat_ok": bool(probe.get("chat_ok", False)),
    "hf_template_completion_ok": bool(probe.get("hf_template_completion_ok", False)),
    "endpoint_split_suspect": bool(probe.get("endpoint_split_suspect", False)),
    "long_prefill_chat_ok": bool(probe.get("long_prefill_chat_ok", False)),
    "long_prefill_dsv4_thinking_completion_ok": bool(probe.get("long_prefill_dsv4_thinking_completion_ok", False)),
    "nontrivial_completion_ok": bool(probe.get("nontrivial_completion_ok", False)),
    "dsv4_template_completion_ok": bool(probe.get("dsv4_template_completion_ok", False)),
    "kvcache_nan_or_inf_warning": kvcache_nan,
    "hadamard_missing_or_skipped_warning": hadamard_missing,
    "kv_dtype_or_cache_log_lines": kv_dtype_lines,
    "moe_warning": bool(moe_warning_lines),
    "moe_warning_lines": moe_warning_lines,
    "mhc_warning": bool(mhc_warning_lines),
    "mhc_warning_lines": mhc_warning_lines,
    "probe": probe,
}
with open(diag_jsonl, "a") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
print("DIAG_ROW " + json.dumps(row, ensure_ascii=False))
PY
}

log "Starting TRTLLM DeepSeek-V4 B300 diagnostic"
log "MODEL=$MODEL TP=$TP EP_SIZE=$EP_SIZE DP_ATTENTION=$DP_ATTENTION ISL=$ISL OSL=$OSL CONC=$CONC"
log "MAX_BATCH_SIZE=$MAX_BATCH_SIZE DIAG_MAX_MODEL_LEN=$DIAG_MAX_MODEL_LEN DIAG_MAX_NUM_TOKENS=$DIAG_MAX_NUM_TOKENS"
log "DIAG_LOG_LEVEL=$DIAG_LOG_LEVEL"
log "NCCL_NVLS_ENABLE=$NCCL_NVLS_ENABLE"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    log "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

sanitize_slurm_mpi_env_for_trtllm
bootstrap_trtllm_dsv4 || exit 1
ensure_fast_hadamard_transform

if [[ "$MODEL" != /* ]]; then
    hf download "$MODEL"
fi

{
    echo "===== environment probe ====="
    nvidia-smi
    python3 - <<'PY'
import importlib
import json
from pathlib import Path
import sys

info = {}
try:
    import tensorrt_llm
    info["tensorrt_llm_version"] = getattr(tensorrt_llm, "__version__", "unknown")
except Exception as exc:
    info["tensorrt_llm_import_error"] = repr(exc)

try:
    import torch
    info["torch_version"] = torch.__version__
    info["compressor_ops"] = {
        name: hasattr(torch.ops.trtllm, name)
        for name in [
            "compressor_prefill_reduction",
            "compressor_paged_kv_compress",
            "compressor_postprocess_scatter",
        ]
    }
except Exception as exc:
    info["torch_or_ops_error"] = repr(exc)

for module in [
    "fast_hadamard_transform",
    "tensorrt_llm._torch.models.modeling_deepseekv4",
    "tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.deepseek_v4",
]:
    try:
        importlib.import_module(module)
        info[f"import:{module}"] = True
    except Exception as exc:
        info[f"import:{module}"] = repr(exc)

try:
    import tensorrt_llm
    trtllm_root = Path(tensorrt_llm.__file__).resolve().parent
    info["trtllm_deepseek_tokenizer_files"] = [
        str(path.relative_to(trtllm_root))
        for path in trtllm_root.rglob("*.py")
        if "deepseek" in str(path).lower() and "token" in str(path).lower()
    ][:50]
except Exception as exc:
    info["trtllm_deepseek_tokenizer_file_scan_error"] = repr(exc)

print(json.dumps(info, indent=2))
PY
    echo "===== end environment probe ====="
} | tee -a "$SERVER_LOG"

start_gpu_monitor --output "$PWD/gpu_metrics.csv"
trap 'stop_gpu_monitor' EXIT

run_variant "baseline_fp8_graph" "fp8" "on" "$PORT_BASE" "TRTLLM" "default" "$TP" "$EP_SIZE" "$DP_ATTENTION" "4"
run_variant "auto_kv_graph" "unset" "on" "$((PORT_BASE + 1))" "TRTLLM" "default" "$TP" "$EP_SIZE" "$DP_ATTENTION" "4"
run_variant "auto_kv_no_autotune" "unset" "on" "$((PORT_BASE + 2))" "TRTLLM" "false" "$TP" "$EP_SIZE" "$DP_ATTENTION" "4"
run_variant "auto_kv_vanilla_moe" "unset" "on" "$((PORT_BASE + 3))" "VANILLA" "false" "$TP" "$EP_SIZE" "$DP_ATTENTION" "4"
run_variant "fp8_no_cuda_graph" "fp8" "off" "$((PORT_BASE + 4))" "TRTLLM" "default" "$TP" "$EP_SIZE" "$DP_ATTENTION" "4"

if [[ "${TRTLLM_DSV4_DIAG_ENABLE_PP1:-1}" == "1" ]]; then
    run_variant "baseline_fp8_graph_pp1" "fp8" "on" "$((PORT_BASE + 5))" "TRTLLM" "default" "$TP" "$EP_SIZE" "$DP_ATTENTION" "1"
    run_variant "auto_kv_graph_pp1" "unset" "on" "$((PORT_BASE + 6))" "TRTLLM" "default" "$TP" "$EP_SIZE" "$DP_ATTENTION" "1"
fi

if [[ "${TRTLLM_DSV4_DIAG_ENABLE_TP_EP_MATRIX:-1}" == "1" ]]; then
    run_variant "tp4_ep1_dpa_false_fp8_graph" "fp8" "on" "$((PORT_BASE + 7))" "TRTLLM" "default" "4" "1" "false" "4"
    run_variant "tp8_ep8_dpa_true_fp8_graph" "fp8" "on" "$((PORT_BASE + 8))" "TRTLLM" "default" "8" "8" "true" "4"
    run_variant "tp4_ep4_dpa_true_fp8_graph" "fp8" "on" "$((PORT_BASE + 9))" "TRTLLM" "default" "4" "4" "true" "4"
fi

if [[ "${TRTLLM_DSV4_DIAG_ENABLE_EXPLICIT_KV_DTYPES:-1}" == "1" ]]; then
    run_variant "bf16_kv_graph" "bf16" "on" "$((PORT_BASE + 10))" "TRTLLM" "default" "$TP" "$EP_SIZE" "$DP_ATTENTION" "4"
    run_variant "default_kv_graph" "default" "on" "$((PORT_BASE + 11))" "TRTLLM" "default" "$TP" "$EP_SIZE" "$DP_ATTENTION" "4"
fi

stop_gpu_monitor
trap - EXIT

python3 - "$DIAG_JSONL" "$DIAG_SUMMARY_JSON" <<'PY' | tee -a "$SERVER_LOG"
import json
import sys

jsonl, summary_path = sys.argv[1], sys.argv[2]
rows = []
with open(jsonl) as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

by_name = {row["variant"]: row for row in rows}
baseline = by_name.get("baseline_fp8_graph", {})
auto_kv = by_name.get("auto_kv_graph", {})
no_graph = by_name.get("fp8_no_cuda_graph", {})
no_autotune = by_name.get("auto_kv_no_autotune", {})
vanilla_moe = by_name.get("auto_kv_vanilla_moe", {})
baseline_pp1 = by_name.get("baseline_fp8_graph_pp1", {})
auto_kv_pp1 = by_name.get("auto_kv_graph_pp1", {})
tp4_ep1 = by_name.get("tp4_ep1_dpa_false_fp8_graph", {})
tp8_ep8_dpa = by_name.get("tp8_ep8_dpa_true_fp8_graph", {})
tp4_ep4_dpa = by_name.get("tp4_ep4_dpa_true_fp8_graph", {})
bf16_kv = by_name.get("bf16_kv_graph", {})
default_kv = by_name.get("default_kv_graph", {})

endpoint_split_variants = [
    row["variant"] for row in rows if row.get("endpoint_split_suspect")
]
chat_ok_variants = [row["variant"] for row in rows if row.get("chat_ok")]
completion_ok_variants = [
    row["variant"] for row in rows if row.get("completion_ok")
]
nontrivial_completion_ok_variants = [
    row["variant"] for row in rows if row.get("nontrivial_completion_ok")
]
dsv4_template_completion_ok_variants = [
    row["variant"] for row in rows if row.get("dsv4_template_completion_ok")
]
moe_warning_variants = [row["variant"] for row in rows if row.get("moe_warning")]
mhc_warning_variants = [row["variant"] for row in rows if row.get("mhc_warning")]

summary = {
    "variants": rows,
    "baseline_ok": bool(baseline.get("probe_ok", False)),
    "baseline_completion_ok": bool(baseline.get("completion_ok", False)),
    "baseline_chat_ok": bool(baseline.get("chat_ok", False)),
    "baseline_hf_template_completion_ok": bool(baseline.get("hf_template_completion_ok", False)),
    "auto_kv_ok": bool(auto_kv.get("probe_ok", False)),
    "auto_kv_no_autotune_ok": bool(no_autotune.get("probe_ok", False)),
    "auto_kv_vanilla_moe_ok": bool(vanilla_moe.get("probe_ok", False)),
    "fp8_no_cuda_graph_ok": bool(no_graph.get("probe_ok", False)),
    "baseline_pp1_ok": bool(baseline_pp1.get("probe_ok", False)),
    "auto_kv_pp1_ok": bool(auto_kv_pp1.get("probe_ok", False)),
    "tp4_ep1_dpa_false_ok": bool(tp4_ep1.get("probe_ok", False)),
    "tp8_ep8_dpa_true_ok": bool(tp8_ep8_dpa.get("probe_ok", False)),
    "tp4_ep4_dpa_true_ok": bool(tp4_ep4_dpa.get("probe_ok", False)),
    "bf16_kv_ok": bool(bf16_kv.get("probe_ok", False)),
    "default_kv_ok": bool(default_kv.get("probe_ok", False)),
    "completion_ok_variants": completion_ok_variants,
    "nontrivial_completion_ok_variants": nontrivial_completion_ok_variants,
    "dsv4_template_completion_ok_variants": dsv4_template_completion_ok_variants,
    "chat_ok_variants": chat_ok_variants,
    "endpoint_split_variants": endpoint_split_variants,
    "moe_warning_variants": moe_warning_variants,
    "mhc_warning_variants": mhc_warning_variants,
    "supports_explicit_fp8_override_suspect": (
        baseline.get("probe_ok") is False and auto_kv.get("probe_ok") is True
    ),
    "supports_autotuner_suspect": (
        auto_kv.get("probe_ok") is False and no_autotune.get("probe_ok") is True
    ),
    "supports_moe_backend_suspect": (
        auto_kv.get("probe_ok") is False and vanilla_moe.get("probe_ok") is True
    ),
    "supports_cuda_graph_stale_metadata_suspect": (
        baseline.get("probe_ok") is False and no_graph.get("probe_ok") is True
    ),
    "supports_postprocess_worker_race_suspect": (
        baseline.get("probe_ok") is False and baseline_pp1.get("probe_ok") is True
    ),
    "supports_tp_ep_topology_suspect": (
        baseline.get("probe_ok") is False
        and any(row.get("probe_ok") for row in [tp4_ep1, tp8_ep8_dpa, tp4_ep4_dpa])
    ),
    "supports_chat_endpoint_or_serialization_suspect": bool(
        baseline.get("endpoint_split_suspect", False)
    ),
    "supports_hf_template_mismatch_suspect": (
        baseline.get("completion_ok") is True
        and baseline.get("hf_template_completion_ok") is False
    ),
    "supports_explicit_bf16_kv_suspect": (
        baseline.get("probe_ok") is False and bf16_kv.get("probe_ok") is True
    ),
    "supports_explicit_default_kv_suspect": (
        baseline.get("probe_ok") is False and default_kv.get("probe_ok") is True
    ),
    "any_variant_ok": any(row.get("probe_ok") for row in rows),
    "any_completion_ok": any(row.get("completion_ok") for row in rows),
    "any_nontrivial_completion_ok": any(row.get("nontrivial_completion_ok") for row in rows),
    "any_dsv4_template_completion_ok": any(row.get("dsv4_template_completion_ok") for row in rows),
    "any_chat_ok": any(row.get("chat_ok") for row in rows),
    "any_endpoint_split_suspect": bool(endpoint_split_variants),
    "any_kvcache_nan_warning": any(row.get("kvcache_nan_or_inf_warning") for row in rows),
    "any_hadamard_warning": any(row.get("hadamard_missing_or_skipped_warning") for row in rows),
    "any_moe_warning": bool(moe_warning_variants),
    "any_mhc_warning": bool(mhc_warning_variants),
}

with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("===== TRTLLM DSV4 DIAGNOSTIC SUMMARY =====")
print(json.dumps(summary, indent=2, ensure_ascii=False))
print("===== END TRTLLM DSV4 DIAGNOSTIC SUMMARY =====")
PY

strict_any_ok="$(python3 - "$DIAG_SUMMARY_JSON" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    summary = json.load(f)
print("1" if summary.get("any_variant_ok") else "0")
PY
)"
pass_metric="0.0"
write_placeholder_outputs "$pass_metric"

if [[ "${TRTLLM_DSV4_DIAG_FAIL_AFTER:-1}" == "1" ]]; then
    log "TRTLLM_DSV4_DIAG_FAIL_AFTER=1; failing intentionally after diagnostics so this temporary run is not mistaken for a benchmark."
    exit 1
fi

if [[ "$strict_any_ok" == "1" ]]; then
    exit 0
fi

exit 1
