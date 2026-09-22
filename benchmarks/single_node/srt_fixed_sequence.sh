#!/usr/bin/env bash

# SRT owns the server lifecycle; retain the existing InferenceX client and sampler.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../benchmark_lib.sh" --validation-only
check_env_vars MODEL CONC ISL OSL RANDOM_RANGE_RATIO RESULT_FILENAME RESULT_DIR \
    SRT_FRONTEND_HOST SRT_FRONTEND_PORT RUN_EVAL EVAL_ONLY GPU_MONITOR_INTERVAL USE_CHAT_TEMPLATE FRAMEWORK
for name in RUN_EVAL EVAL_ONLY; do
    if [[ "${!name}" != true && "${!name}" != false ]]; then
        echo "ERROR: $name must be true or false" >&2
        exit 1
    fi
done
case "$FRAMEWORK" in
    sglang|atom) CLIENT_BACKEND=vllm ;;
    trt) CLIENT_BACKEND=openai ;;
    *) echo "ERROR: unsupported fixed-sequence FRAMEWORK: $FRAMEWORK" >&2; exit 1 ;;
esac
SRT_MONITOR_INTERVAL="$GPU_MONITOR_INTERVAL"
CLIENT_ARGS=()
for argument in "$@"; do
    case "$argument" in
        --trust-remote-code) CLIENT_ARGS+=("$argument") ;;
        *) echo "ERROR: unsupported fixed-sequence argument: $argument" >&2; exit 1 ;;
    esac
done
case "$USE_CHAT_TEMPLATE" in
    true) CLIENT_ARGS+=(--use-chat-template) ;;
    false) ;;
    *) echo "ERROR: USE_CHAT_TEMPLATE must be true or false" >&2; exit 1 ;;
esac

for name in CONC ISL OSL SRT_FRONTEND_PORT GPU_MONITOR_INTERVAL; do
    if [[ ! "${!name}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: $name must be a positive integer" >&2
        exit 1
    fi
done

if [[ ! -d "$RESULT_DIR" ]]; then
    echo "ERROR: RESULT_DIR must be an existing runtime-provided directory" >&2
    exit 1
fi

source "$(dirname "${BASH_SOURCE[0]}")/../benchmark_lib.sh"
cd "$INFERENCEX_REPO_ROOT"
pip3 install --user --break-system-packages sentencepiece datasets pandas

start_gpu_monitor --output "$RESULT_DIR/gpu_metrics.csv" --interval "$SRT_MONITOR_INTERVAL"
trap 'rc=$?; stop_gpu_monitor; exit "$rc"' EXIT

run_benchmark_serving \
    --model "$MODEL" \
    --port "$SRT_FRONTEND_PORT" \
    --base-url "http://${SRT_FRONTEND_HOST}:${SRT_FRONTEND_PORT}" \
    --backend "$CLIENT_BACKEND" \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$((CONC * 10))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir "$RESULT_DIR" \
    "${CLIENT_ARGS[@]}"
