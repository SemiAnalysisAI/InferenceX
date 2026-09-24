#!/usr/bin/env bash

# Fixed-sequence client for multi-node srt-slurm recipes: SRT owns the servers;
# this runs the InferenceX client once per concurrency and writes the result
# layout that copy_fixed_sequence_results collects.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../benchmark_lib.sh" --validation-only
check_env_vars MODEL ISL OSL SRT_FRONTEND_HOST SRT_FRONTEND_PORT CONC_LIST \
    PREFILL_NUM_WORKERS PREFILL_TP DECODE_NUM_WORKERS DECODE_TP
CLIENT_ARGS=()
for argument in "$@"; do
    case "$argument" in
        --trust-remote-code) CLIENT_ARGS+=("$argument") ;;
        *) echo "ERROR: unsupported fixed-sequence argument: $argument" >&2; exit 1 ;;
    esac
done
case "${CLIENT_BACKEND:=openai}" in
    openai) endpoint=/v1/completions ;;
    openai-chat) endpoint=/v1/chat/completions ;;
    *) echo "ERROR: unsupported CLIENT_BACKEND: $CLIENT_BACKEND" >&2; exit 1 ;;
esac
case "${USE_CHAT_TEMPLATE:=false}" in
    true) CLIENT_ARGS+=(--use-chat-template) ;;
    false) ;;
    *) echo "ERROR: USE_CHAT_TEMPLATE must be true or false" >&2; exit 1 ;;
esac

result_dir="/logs/sa-bench_isl_${ISL}_osl_${OSL}"
mkdir -p "$result_dir"
ctx=$((PREFILL_NUM_WORKERS * PREFILL_TP))
gen=$((DECODE_NUM_WORKERS * DECODE_TP))
for concurrency in $CONC_LIST; do
    num_prompts=$((concurrency * 10 < 16 ? 16 : concurrency * 10))
    python3 "$(dirname "${BASH_SOURCE[0]}")/../../utils/bench_serving/benchmark_serving.py" \
        --backend "$CLIENT_BACKEND" \
        --base-url "http://${SRT_FRONTEND_HOST}:${SRT_FRONTEND_PORT}" \
        --endpoint "$endpoint" \
        --model "$MODEL" \
        --tokenizer "${TOKENIZER:-$MODEL}" \
        --dataset-name random \
        --random-input-len "$ISL" \
        --random-output-len "$OSL" \
        --random-range-ratio "${RANDOM_RANGE_RATIO:-1.0}" \
        --random-num-workers 1 \
        --num-warmups "$((concurrency * 2))" \
        --num-prompts "$num_prompts" \
        --max-concurrency "$concurrency" \
        --request-rate inf \
        --ignore-eos \
        --disable-tqdm \
        --save-result \
        --result-dir "$result_dir" \
        --result-filename "results_concurrency_${concurrency}_gpus_$((ctx + gen))_ctx_${ctx}_gen_${gen}.json" \
        "${CLIENT_ARGS[@]}"
done
