#!/usr/bin/env bash
# Drop-in replacement for srt-slurm's bundled `sa-bench` benchmark, wired to
# this repo's utils/bench_serving/benchmark_serving.py via srt-slurm's
# `benchmark.type: custom` feature. srt-slurm owns server bring-up; this
# script runs against the already-ready frontend on the head node, then
# writes one results JSON per concurrency to a path the launcher's
# result-harvester recognizes.
#
# Required env (set via `benchmark.env` in the recipe yaml):
#   ISL OSL CONCURRENCIES MODEL_NAME
#   IS_DISAGGREGATED TOTAL_GPUS PREFILL_GPUS DECODE_GPUS
#
# Optional env (defaults shown):
#   PORT=8000                  frontend port reachable at localhost
#   REQ_RATE=inf
#   RANDOM_RANGE_RATIO=0.8
#   NUM_PROMPTS_MULT=10        prompts per conc = NUM_PROMPTS_MULT * conc
#   NUM_WARMUP_MULT=2          warmup prompts per conc = NUM_WARMUP_MULT * conc
#   USE_CHAT_TEMPLATE=true
#   CUSTOM_TOKENIZER=          (empty: skip --custom-tokenizer)
#   DATASET_NAME=random
#   DATASET_PATH=              (only used when DATASET_NAME != random)
#   TOKENIZER_PATH=$MODEL_PATH (or container path; falls back to $MODEL_NAME)
#   PORT_HEALTH_PATH=/v1/models
#
# The InferenceX repo is bind-mounted into the container at /infmax-workspace
# (configured by the recipe's `container_mounts` block). This script lives at
# /infmax-workspace/benchmarks/multi_node/srt_bench.sh and shells out to
# /infmax-workspace/utils/bench_serving/benchmark_serving.py.
set -euo pipefail

INFMAX_WS="${INFMAX_CONTAINER_WORKSPACE:-/infmax-workspace}"

require() {
    for v in "$@"; do
        if [[ -z "${!v:-}" ]]; then
            echo "ERROR: required env var '$v' is unset" >&2
            exit 64
        fi
    done
}
require ISL OSL CONCURRENCIES MODEL_NAME IS_DISAGGREGATED TOTAL_GPUS

PORT="${PORT:-8000}"
REQ_RATE="${REQ_RATE:-inf}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"
NUM_PROMPTS_MULT="${NUM_PROMPTS_MULT:-10}"
NUM_WARMUP_MULT="${NUM_WARMUP_MULT:-2}"
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-true}"
CUSTOM_TOKENIZER="${CUSTOM_TOKENIZER:-}"
DATASET_NAME="${DATASET_NAME:-random}"
DATASET_PATH="${DATASET_PATH:-}"
PREFILL_GPUS="${PREFILL_GPUS:-0}"
DECODE_GPUS="${DECODE_GPUS:-0}"

ENDPOINT="http://localhost:${PORT}"
RESULT_DIR="/logs/sa-bench_isl_${ISL}_osl_${OSL}"
mkdir -p "$RESULT_DIR"

BENCH_PY="${INFMAX_WS}/utils/bench_serving/benchmark_serving.py"
[[ -f "$BENCH_PY" ]] || { echo "ERROR: benchmark_serving.py not found at $BENCH_PY (mount $INFMAX_WS missing?)" >&2; exit 65; }

# Bench-serving deps. The srt-slurm worker container ships most of these but
# not all (datasets in particular). Reuse system-site-packages so we don't
# rebuild what's already there.
ensure_deps() {
    local deps=(aiohttp numpy pandas datasets Pillow tqdm transformers huggingface_hub)
    if python3 -c "import aiohttp, numpy, pandas, datasets, PIL, tqdm, transformers, huggingface_hub" 2>/dev/null; then
        return
    fi
    local venv="/tmp/srt-bench-venv"
    [[ -d "$venv" ]] || python3 -m venv --system-site-packages "$venv"
    # shellcheck disable=SC1091
    source "$venv/bin/activate"
    pip install --quiet "${deps[@]}"
}
ensure_deps

# Verify endpoint
echo "Verifying endpoint at $ENDPOINT ..."
curl -fsS "${ENDPOINT}/v1/models" >/dev/null || {
    echo "ERROR: endpoint $ENDPOINT did not respond on /v1/models" >&2
    exit 66
}

ulimit -n 65536 2>/dev/null || true

DATASET_ARGS=(--dataset-name "$DATASET_NAME")
[[ -n "$DATASET_PATH" ]] && DATASET_ARGS+=(--dataset-path "$DATASET_PATH")

RANDOM_LEN_ARGS=()
if [[ "$DATASET_NAME" == "random" ]]; then
    RANDOM_LEN_ARGS=(
        --random-input-len "$ISL"
        --random-output-len "$OSL"
        --random-range-ratio "$RANDOM_RANGE_RATIO"
    )
fi

CHAT_TEMPLATE_ARGS=()
[[ "$USE_CHAT_TEMPLATE" == "true" ]] && CHAT_TEMPLATE_ARGS+=(--use-chat-template)

CUSTOM_TOKENIZER_ARGS=()
[[ -n "$CUSTOM_TOKENIZER" ]] && CUSTOM_TOKENIZER_ARGS+=(--custom-tokenizer "$CUSTOM_TOKENIZER")

# `tokenizer` is required by benchmark_serving.py; pass MODEL_NAME by default
# (HF will fetch). Recipe can override via TOKENIZER_PATH for a local path.
TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_NAME}"

# Concurrency list is "x"-separated for parity with sa-bench.
IFS='x' read -r -a CONC_LIST <<< "$CONCURRENCIES"

run_bench() {
    local conc=$1
    local n_prompts=$2
    local request_rate=$3
    shift 3
    python3 -u "$BENCH_PY" \
        --model "$MODEL_NAME" --tokenizer "$TOKENIZER_PATH" \
        --host localhost --port "$PORT" \
        --backend dynamo --endpoint /v1/completions \
        --disable-tqdm \
        "${DATASET_ARGS[@]}" \
        --num-prompts "$n_prompts" \
        "${RANDOM_LEN_ARGS[@]}" \
        --ignore-eos \
        --request-rate "$request_rate" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --max-concurrency "$conc" \
        --trust-remote-code \
        "${CHAT_TEMPLATE_ARGS[@]}" \
        "${CUSTOM_TOKENIZER_ARGS[@]}" \
        "$@"
}

for conc in "${CONC_LIST[@]}"; do
    echo "=== conc=$conc warmup ==="
    run_bench "$conc" "$((conc * NUM_WARMUP_MULT))" 250 || true

    if [[ "$IS_DISAGGREGATED" == "true" ]]; then
        result_filename="results_concurrency_${conc}_gpus_${TOTAL_GPUS}_ctx_${PREFILL_GPUS}_gen_${DECODE_GPUS}.json"
    else
        result_filename="results_concurrency_${conc}_gpus_${TOTAL_GPUS}.json"
    fi

    echo "=== conc=$conc bench → $RESULT_DIR/$result_filename ==="
    run_bench "$conc" "$((conc * NUM_PROMPTS_MULT))" "$REQ_RATE" \
        --result-dir "$RESULT_DIR" \
        --result-filename "$result_filename"
done

echo "Done. Results in $RESULT_DIR."
