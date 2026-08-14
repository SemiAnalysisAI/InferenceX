#!/usr/bin/env bash
set -euo pipefail
set -x

SCENARIO="${1:?usage: client.sh fixed-seq-len|agentic-coding}"
INFMAX_CONTAINER_WORKSPACE="${INFMAX_CONTAINER_WORKSPACE:-/workspace}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen/Qwen3.8-2.4T-A95B-FP8}"
PORT="${PORT:-8000}"

source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh"

check_env_vars MODEL MODEL_PREFIX FRAMEWORK PRECISION CONC_LIST RESULT_FILENAME

export INFMAX_CONTAINER_WORKSPACE
export MODEL_PATH
export PORT

case "$SCENARIO" in
    fixed-seq-len)
        check_env_vars ISL OSL RANDOM_RANGE_RATIO
        RESULT_DIR="${RESULT_DIR:-/run_logs/${FRAMEWORK}_isl_${ISL}_osl_${OSL}}"
        mkdir -p "$RESULT_DIR"

        for concurrency in $CONC_LIST; do
            if ! [[ "$concurrency" =~ ^[1-9][0-9]*$ ]]; then
                echo "ERROR: invalid fixed-sequence concurrency: $concurrency" >&2
                exit 1
            fi
            num_prompts=$((concurrency * 10))
            if ((num_prompts < 16)); then
                num_prompts=16
            fi
            result_stem="isl_${ISL}_osl_${OSL}_concurrency_${concurrency}_req_rate_inf_gpus_16"
            run_benchmark_serving \
                --bench-serving-dir "$INFMAX_CONTAINER_WORKSPACE" \
                --model "$MODEL" \
                --port "$PORT" \
                --backend openai-chat \
                --endpoint /v1/chat/completions \
                --input-len "$ISL" \
                --output-len "$OSL" \
                --random-range-ratio "$RANDOM_RANGE_RATIO" \
                --num-prompts "$num_prompts" \
                --max-concurrency "$concurrency" \
                --use-chat-template \
                --tokenizer "$MODEL_PATH" \
                --trust-remote-code \
                --result-filename "$result_stem" \
                --result-dir "$RESULT_DIR"
        done

        if [[ "${RUN_EVAL:-false}" == "true" ]]; then
            export CONC="${EVAL_CONC:-${CONC_LIST##* }}"
            run_eval --framework lm-eval --port "$PORT"
            append_lm_eval_summary
        fi
        ;;
    agentic-coding)
        check_env_vars DURATION
        export RESULT_DIR="${RESULT_DIR:-/run_logs/agentic}"
        export AGENTIC_OUTPUT_DIR="${AGENTIC_OUTPUT_DIR:-$INFMAX_CONTAINER_WORKSPACE}"
        export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
        export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="vllm:"
        exec bash "$INFMAX_CONTAINER_WORKSPACE/benchmarks/multi_node/agentic_srt.sh"
        ;;
    *)
        echo "ERROR: unsupported scenario: $SCENARIO" >&2
        exit 2
        ;;
esac
