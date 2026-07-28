#!/usr/bin/env bash

# InferenceX benchmark_serving adapter for srt-slurm's custom benchmark runner.

set -eo pipefail

if [[ $# -ne 15 ]]; then
    echo "Expected 15 benchmark arguments, received $#" >&2
    exit 64
fi

readonly MODEL=$1
readonly TOKENIZER=$2
readonly ISL=$3
readonly OSL=$4
readonly CONC_LIST=$5
readonly REQ_RATE=$6
readonly DISAGG=$7
readonly PREFILL_GPUS=$8
readonly DECODE_GPUS=$9
readonly TOTAL_GPUS=${10}
readonly RANDOM_RANGE_RATIO=${11}
readonly NUM_PROMPTS_MULT=${12}
readonly NUM_WARMUP_MULT=${13}
readonly USE_CHAT_TEMPLATE=${14}
readonly DSV4=${15}

readonly INFERENCEX_WORKSPACE=/infmax-workspace
readonly BENCH_HOST=127.0.0.1
readonly BENCH_PORT=8000
readonly BENCH_BACKEND=openai
readonly TRUST_REMOTE_CODE=true
readonly RESULT_DIR="/logs/inferencex-bench_isl_${ISL}_osl_${OSL}"
export EVAL_ONLY=false
export PROFILE=0

# shellcheck disable=SC1091
source "$INFERENCEX_WORKSPACE/benchmarks/benchmark_lib.sh"

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

    optional_args=()
    [[ "$USE_CHAT_TEMPLATE" == "true" ]] && optional_args+=(--use-chat-template)
    [[ "$DSV4" == "true" ]] && optional_args+=(--dsv4)
    [[ "$TRUST_REMOTE_CODE" == "true" ]] && optional_args+=(--trust-remote-code)

    run_benchmark_serving \
        --model "$MODEL" \
        --tokenizer "$TOKENIZER" \
        --host "$BENCH_HOST" \
        --port "$BENCH_PORT" \
        --backend "$BENCH_BACKEND" \
        --input-len "$ISL" \
        --output-len "$OSL" \
        --random-range-ratio "$RANDOM_RANGE_RATIO" \
        --num-prompts "$((concurrency * NUM_PROMPTS_MULT))" \
        --num-warmups "$((concurrency * NUM_WARMUP_MULT))" \
        --request-rate "$REQ_RATE" \
        --max-concurrency "$concurrency" \
        --result-filename "$result_filename" \
        --result-dir "$RESULT_DIR" \
        --bench-serving-dir "$INFERENCEX_WORKSPACE" \
        "${optional_args[@]}"
done
