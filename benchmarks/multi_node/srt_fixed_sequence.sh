#!/usr/bin/env bash

# Fixed-sequence client for multi-node srt-slurm recipes: SRT owns the servers;
# this runs the InferenceX client once per concurrency and writes the result
# layout that copy_fixed_sequence_results collects.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../benchmark_lib.sh" --validation-only
check_env_vars ISL OSL SRT_FRONTEND_HOST SRT_FRONTEND_PORT CONC_LIST \
    PREFILL_NUM_WORKERS PREFILL_TP DECODE_NUM_WORKERS DECODE_TP
CLIENT_ARGS=(--trust-remote-code)
case "${CLIENT_BACKEND:=openai}" in
    openai) endpoint=/v1/completions ;;
    openai-chat) endpoint=/v1/chat/completions ;;
    *) echo "ERROR: unsupported CLIENT_BACKEND: $CLIENT_BACKEND" >&2; exit 1 ;;
esac
case "${USE_CHAT_TEMPLATE:=true}" in
    true) CLIENT_ARGS+=(--use-chat-template) ;;
    false) ;;
    *) echo "ERROR: USE_CHAT_TEMPLATE must be true or false" >&2; exit 1 ;;
esac

repo_root="$(dirname "${BASH_SOURCE[0]}")/../.."
# Request the name the workers registered; the workflow's MODEL is the HF id, which can differ.
model=$(curl -sf "http://${SRT_FRONTEND_HOST}:${SRT_FRONTEND_PORT}/v1/models" |
    python3 -c 'import json, sys; print(json.load(sys.stdin)["data"][0]["id"])')
result_dir="/logs/sa-bench_isl_${ISL}_osl_${OSL}"
mkdir -p "$result_dir"
ctx=$((PREFILL_NUM_WORKERS * PREFILL_TP))
gen=$((DECODE_NUM_WORKERS * DECODE_TP))
for concurrency in $CONC_LIST; do
    result="results_concurrency_${concurrency}_gpus_$((ctx + gen))_ctx_${ctx}_gen_${gen}.json"
    PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" python3 -P -m infx.bench_serving.benchmark_serving \
        --backend "$CLIENT_BACKEND" \
        --base-url "http://${SRT_FRONTEND_HOST}:${SRT_FRONTEND_PORT}" \
        --endpoint "$endpoint" \
        --model "$model" \
        --tokenizer "${TOKENIZER:-$model}" \
        --dataset-name random \
        --random-input-len "$ISL" \
        --random-output-len "$OSL" \
        --random-range-ratio "${RANDOM_RANGE_RATIO:-0.8}" \
        --random-num-workers 1 \
        --num-warmups "$((concurrency * 2))" \
        --num-prompts "$((concurrency * 10))" \
        --max-concurrency "$concurrency" \
        --request-rate inf \
        --ignore-eos \
        --disable-tqdm \
        --save-result \
        --result-dir "$result_dir" \
        --result-filename "$result" \
        "${CLIENT_ARGS[@]}"
    # Power lanes: tell srt-slurm which interval this concurrency's result measured.
    if [[ -n "${SRT_MEASUREMENT_WINDOW_DIR:-}" ]]; then
        PYTHONPATH="$repo_root" python3 -m infx.results.power.window "$result_dir/$result" "$concurrency"
    fi
done
