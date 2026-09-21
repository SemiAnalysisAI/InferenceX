#!/usr/bin/env bash

# SRT owns the server lifecycle; retain the existing InferenceX client and sampler.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../benchmark_lib.sh" --validation-only
check_env_vars MODEL CONC ISL OSL RANDOM_RANGE_RATIO RESULT_FILENAME RESULT_DIR \
    SRT_FRONTEND_HOST SRT_FRONTEND_PORT RUN_EVAL EVAL_ONLY GPU_MONITOR_INTERVAL
SRT_MONITOR_INTERVAL="$GPU_MONITOR_INTERVAL"

for name in CONC ISL OSL SRT_FRONTEND_PORT GPU_MONITOR_INTERVAL; do
    if [[ ! "${!name}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: $name must be a positive integer" >&2
        exit 1
    fi
done

# The initial parallel port supports throughput only. Eval context and artifact
# forwarding must be connected before production cutover.
if [[ "$RUN_EVAL" != false || "$EVAL_ONLY" != false ]]; then
    echo "ERROR: the single-node SRT pilot does not support evals yet" >&2
    exit 1
fi

if [[ ! -d "$RESULT_DIR" ]]; then
    echo "ERROR: RESULT_DIR must be an existing runtime-provided directory" >&2
    exit 1
fi

source "$(dirname "${BASH_SOURCE[0]}")/../benchmark_lib.sh"
cd "$INFERENCEX_REPO_ROOT"
pip3 install --user --break-system-packages sentencepiece

start_gpu_monitor --output "$RESULT_DIR/gpu_metrics.csv" --interval "$SRT_MONITOR_INTERVAL"
trap 'rc=$?; stop_gpu_monitor; exit "$rc"' EXIT

run_benchmark_serving \
    --model "$MODEL" \
    --port "$SRT_FRONTEND_PORT" \
    --base-url "http://${SRT_FRONTEND_HOST}:${SRT_FRONTEND_PORT}" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$((CONC * 10))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir "$RESULT_DIR"
