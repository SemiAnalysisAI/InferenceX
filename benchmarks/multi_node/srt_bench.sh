#!/usr/bin/env bash
# Multi-node bench-serving wrapper invoked by srt-slurm via
# `benchmark.type: custom`. srt-slurm owns server bring-up; this script runs
# inside the same job's benchmark container against the already-ready
# frontend on the head node, then writes one results JSON per concurrency to
# /logs/sa-bench_isl_<ISL>_osl_<OSL>/ — the same path the launcher's existing
# result-harvesters glob.
#
# This is a thin loop on top of run_benchmark_serving() in benchmark_lib.sh
# (the same shim every single-node bench script uses), so any future change
# to bench-serving CLI conventions, profiling, server-health monitoring, etc.
# applies here automatically.
#
# Reads from env. Most of these are *already* exported by
# .github/workflows/benchmark-multinode-tmpl.yml at the workflow step level
# and propagate down through the launcher → srtctl → srun (default
# --export=ALL) → pyxis → bench container, so recipes do not need to
# re-declare them in `benchmark.env`:
#
#   $MODEL              served-model-name; matches workflow `inputs.model`
#   $ISL $OSL           sequence lengths
#   $CONC_LIST          space-separated concurrency list
#   $DISAGG             "true" / "false" — disagg vs aggregated
#   $RANDOM_RANGE_RATIO 0.8 (workflow default)
#
# Per-recipe knobs that *do* live in `benchmark.env` (no workflow equivalent):
#   PREFILL_GPUS        per-prefill-worker GPU count (filename component)
#   DECODE_GPUS         per-decode-worker GPU count (filename component)
#   TOTAL_GPUS          sum across all workers (filename component)
#
# Optional per-recipe overrides (defaults shown):
#   MODEL_NAME=$MODEL          override when server's served-model-name differs
#                              from the master-yaml `model:` field
#   PORT=8000                  frontend port reachable at localhost
#   BACKEND=dynamo
#   ENDPOINT=/v1/completions
#   NUM_PROMPTS_MULT=10        prompts per conc = NUM_PROMPTS_MULT * conc
#   USE_CHAT_TEMPLATE=true
#   DSV4=false                 sets the --dsv4 flag (auto-enables chat template)
#   TRUST_REMOTE_CODE=true
#   DATASET_NAME=random
#   DATASET_PATH=              (only meaningful when DATASET_NAME != random)
#
# The InferenceX repo is bind-mounted at /infmax-workspace via each recipe's
# `container_mounts` block. Model files are auto-mounted at /model by srtctl
# (RuntimeContext.create unconditionally adds the mount when model.path is a
# local path), so we point --tokenizer at /model to load the tokenizer from
# the same files the engine is serving — no HF Hub dependency.
set -euo pipefail

INFMAX_WS="${INFMAX_CONTAINER_WORKSPACE:-/infmax-workspace}"
# shellcheck disable=SC1091
source "$INFMAX_WS/benchmarks/benchmark_lib.sh"

check_env_vars MODEL ISL OSL CONC_LIST DISAGG \
               PREFILL_GPUS DECODE_GPUS TOTAL_GPUS

MODEL_NAME="${MODEL_NAME:-$MODEL}"
PORT="${PORT:-8000}"
BACKEND="${BACKEND:-dynamo}"
ENDPOINT="${ENDPOINT:-/v1/completions}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"
NUM_PROMPTS_MULT="${NUM_PROMPTS_MULT:-10}"
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-true}"
DSV4="${DSV4:-false}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
DATASET_NAME="${DATASET_NAME:-random}"
DATASET_PATH="${DATASET_PATH:-}"

RESULT_DIR="/logs/sa-bench_isl_${ISL}_osl_${OSL}"
mkdir -p "$RESULT_DIR"

# srt-slurm worker containers don't always ship bench_serving.py's runtime
# deps (datasets in particular). Install missing ones into a system-site-
# packages venv so we don't perturb the framework's own packages.
ensure_bench_serving_deps() {
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
ensure_bench_serving_deps

curl -fsS "http://localhost:${PORT}/v1/models" >/dev/null || {
    echo "ERROR: frontend at http://localhost:${PORT} did not respond on /v1/models" >&2
    exit 66
}
ulimit -n 65536 2>/dev/null || true

# CONC_LIST from the workflow is space-separated; bench loops one run per value.
read -r -a CONC_LIST_ARR <<< "$CONC_LIST"

for conc in "${CONC_LIST_ARR[@]}"; do
    if [[ "$DISAGG" == "true" ]]; then
        result_filename="results_concurrency_${conc}_gpus_${TOTAL_GPUS}_ctx_${PREFILL_GPUS}_gen_${DECODE_GPUS}"
    else
        result_filename="results_concurrency_${conc}_gpus_${TOTAL_GPUS}"
    fi
    echo "=== conc=$conc → $RESULT_DIR/${result_filename}.json ==="

    args=(
        --model "$MODEL_NAME"
        --tokenizer /model
        --port "$PORT"
        --backend "$BACKEND"
        --endpoint "$ENDPOINT"
        --input-len "$ISL"
        --output-len "$OSL"
        --random-range-ratio "$RANDOM_RANGE_RATIO"
        --num-prompts "$((conc * NUM_PROMPTS_MULT))"
        --max-concurrency "$conc"
        --dataset-name "$DATASET_NAME"
        --result-filename "$result_filename"
        --result-dir "$RESULT_DIR"
        --bench-serving-dir "$INFMAX_WS"
    )
    [[ -n "$DATASET_PATH" ]]                && args+=(--dataset-path "$DATASET_PATH")
    [[ "$USE_CHAT_TEMPLATE" == "true" ]]    && args+=(--use-chat-template)
    [[ "$DSV4" == "true" ]]                 && args+=(--dsv4)
    [[ "$TRUST_REMOTE_CODE" == "true" ]]    && args+=(--trust-remote-code)

    run_benchmark_serving "${args[@]}"
done

echo "Done. Results in $RESULT_DIR."
