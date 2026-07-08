#!/usr/bin/env bash
# InferenceX serving benchmark adapter for srt-slurm benchmark.type=custom.
set -euo pipefail

INFMAX_CONTAINER_WORKSPACE="${INFMAX_CONTAINER_WORKSPACE:-/infmax-workspace}"
RESULT_ROOT="${RESULT_ROOT:-/logs}"
SERVER_HOST="${SRT_FRONTEND_HOST:-0.0.0.0}"
PORT="${SRT_FRONTEND_PORT:-${PORT:-8000}}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/model}"

source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh"

ensure_benchmark_deps() {
    local venv="/tmp/inferencex-benchmark-venv"
    if python3 -c "import aiohttp, numpy, pandas, datasets, PIL, tqdm, transformers, huggingface_hub" 2>/dev/null; then
        return
    fi
    if [[ ! -d "$venv" ]]; then
        python3 -m venv --system-site-packages "$venv"
    fi
    source "$venv/bin/activate"
    python -m pip install aiohttp numpy pandas datasets Pillow tqdm transformers huggingface_hub
}

ensure_benchmark_deps

check_env_vars \
    ISL \
    OSL \
    CONC_LIST \
    REQUEST_RATE \
    RANDOM_RANGE_RATIO \
    NUM_PROMPTS_MULTIPLIER \
    NUM_WARMUP_MULTIPLIER \
    MODEL_NAME \
    TOKENIZER_MODE \
    TOTAL_GPUS \
    PREFILL_GPUS \
    DECODE_GPUS

read -r -a CONCURRENCIES <<< "${CONC_LIST//x/ }"
if [[ "${#CONCURRENCIES[@]}" -eq 0 ]]; then
    echo "Error: CONC_LIST must contain at least one concurrency" >&2
    exit 1
fi

RESULT_DIR="$RESULT_ROOT/inferencex_isl_${ISL}_osl_${OSL}"
mkdir -p "$RESULT_DIR"

extra_args=(
    --trust-remote-code
    --tokenizer "$TOKENIZER_PATH"
    --tokenizer-mode "$TOKENIZER_MODE"
)
if [[ "${USE_CHAT_TEMPLATE:-true}" == "true" ]]; then
    extra_args+=(--use-chat-template)
fi
if [[ "${DSV4_CHAT_TEMPLATE:-false}" == "true" ]]; then
    extra_args+=(--dsv4)
fi

for concurrency in "${CONCURRENCIES[@]}"; do
    if ! [[ "$concurrency" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: invalid concurrency '$concurrency'" >&2
        exit 1
    fi

    num_prompts=$((concurrency * NUM_PROMPTS_MULTIPLIER))
    num_warmups=$((concurrency * NUM_WARMUP_MULTIPLIER))
    result_filename="results_concurrency_${concurrency}_gpus_${TOTAL_GPUS}"
    if (( PREFILL_GPUS > 0 && DECODE_GPUS > 0 )); then
        result_filename+="_ctx_${PREFILL_GPUS}_gen_${DECODE_GPUS}"
    fi

    run_benchmark_serving \
        --bench-serving-dir "$INFMAX_CONTAINER_WORKSPACE" \
        --model "$MODEL_NAME" \
        --host "$SERVER_HOST" \
        --port "$PORT" \
        --backend openai \
        --endpoint /v1/completions \
        --input-len "$ISL" \
        --output-len "$OSL" \
        --random-range-ratio "$RANDOM_RANGE_RATIO" \
        --num-prompts "$num_prompts" \
        --num-warmups "$num_warmups" \
        --max-concurrency "$concurrency" \
        --request-rate "$REQUEST_RATE" \
        --result-filename "$result_filename" \
        --result-dir "$RESULT_DIR" \
        "${extra_args[@]}"
done
