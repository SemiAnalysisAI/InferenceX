#!/usr/bin/env bash

# InferenceX benchmark_serving adapter for srt-slurm's custom benchmark runner.
# The recipe converter in runners/srt_slurm_benchmark_config.py supplies the
# environment below from the selected recipe's SA-Bench fields.

set -euo pipefail

INFMAX_WS="${INFMAX_CONTAINER_WORKSPACE:-/infmax-workspace}"
# shellcheck disable=SC1091
source "$INFMAX_WS/benchmarks/benchmark_lib.sh"

check_env_vars MODEL TOKENIZER ISL OSL CONC_LIST REQ_RATE DISAGG \
               PREFILL_GPUS DECODE_GPUS TOTAL_GPUS

BENCH_HOST="${SRT_FRONTEND_HOST:-127.0.0.1}"
BENCH_PORT="${SRT_FRONTEND_PORT:-8000}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"
NUM_PROMPTS_MULT="${NUM_PROMPTS_MULT:-10}"
NUM_WARMUP_MULT="${NUM_WARMUP_MULT:-2}"
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-true}"
DSV4="${DSV4:-false}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
RESULT_DIR="/logs/sa-bench_isl_${ISL}_osl_${OSL}"

ensure_bench_serving_deps() {
    local deps=(aiohttp numpy tqdm transformers huggingface_hub)
    if python3 -c \
        "import aiohttp, numpy, tqdm, transformers, huggingface_hub" \
        2>/dev/null; then
        return
    fi

    local venv="/tmp/inferencex-srt-bench-venv"
    [[ -d "$venv" ]] || python3 -m venv --system-site-packages "$venv"
    # shellcheck disable=SC1091
    source "$venv/bin/activate"
    pip install --quiet "${deps[@]}"
}

ensure_bench_serving_deps
mkdir -p "$RESULT_DIR"

curl -fsS "http://${BENCH_HOST}:${BENCH_PORT}/v1/models" >/dev/null || {
    echo "InferenceX benchmark could not reach the srt-slurm frontend" >&2
    exit 66
}
ulimit -n 65536 2>/dev/null || true

read -r -a concurrencies <<< "${CONC_LIST//x/ }"
for concurrency in "${concurrencies[@]}"; do
    if [[ "$DISAGG" == "true" ]]; then
        result_filename="results_concurrency_${concurrency}_gpus_${TOTAL_GPUS}_ctx_${PREFILL_GPUS}_gen_${DECODE_GPUS}"
    else
        result_filename="results_concurrency_${concurrency}_gpus_${TOTAL_GPUS}"
    fi

    args=(
        --model "$MODEL"
        --tokenizer "$TOKENIZER"
        --host "$BENCH_HOST"
        --port "$BENCH_PORT"
        --backend openai
        --input-len "$ISL"
        --output-len "$OSL"
        --random-range-ratio "$RANDOM_RANGE_RATIO"
        --num-prompts "$((concurrency * NUM_PROMPTS_MULT))"
        --num-warmups "$((concurrency * NUM_WARMUP_MULT))"
        --request-rate "$REQ_RATE"
        --max-concurrency "$concurrency"
        --result-filename "$result_filename"
        --result-dir "$RESULT_DIR"
        --bench-serving-dir "$INFMAX_WS"
    )
    [[ "$USE_CHAT_TEMPLATE" == "true" ]] && args+=(--use-chat-template)
    [[ "$DSV4" == "true" ]] && args+=(--dsv4)
    [[ "$TRUST_REMOTE_CODE" == "true" ]] && args+=(--trust-remote-code)

    run_benchmark_serving "${args[@]}"
done
