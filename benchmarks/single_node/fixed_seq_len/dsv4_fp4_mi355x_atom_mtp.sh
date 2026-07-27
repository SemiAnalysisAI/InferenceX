#!/usr/bin/env bash

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME \
    EP_SIZE \
    DP_ATTENTION

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

echo "TP: $TP, CONC: $CONC, ISL: $ISL, OSL: $OSL, EP_SIZE: $EP_SIZE, DP_ATTENTION: $DP_ATTENTION"

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

PARALLEL_ARGS=(-tp "$TP") #TP
if [ "$DP_ATTENTION" = "true" ]; then
    if [ "$EP_SIZE" -gt 1 ]; then #DP+EP
        PARALLEL_ARGS=(-tp "$TP" --enable-expert-parallel --enable-dp-attention )
    else #DPA+TP
        PARALLEL_ARGS=(-tp "$TP" --enable-dp-attention )
    fi
fi

SPEC_ARGS=(--method mtp --num-speculative-tokens 3)

# VERIFY (throwaway branch): TBO + MTP at conc>=128. ATOM owner says TBO now works
# with MTP at high concurrency. dp-on cells at conc>=128 add --enable-tbo +
# GPU_MAX_HW_QUEUES=5. MUST CONFIRM MTP is not silently dropped (ubatch_wrapper
# sets spec_decode_metadata=None) — check server log for spec/MTP init + eval
# accept-rate. If MTP is dropped, this image doesn't support the combo.
if [ "$DP_ATTENTION" = "true" ] && [ "$CONC" -ge 128 ]; then
    PARALLEL_ARGS+=(--enable-tbo)
    export GPU_MAX_HW_QUEUES=5
fi

# max_num_seqs=conc for dp-on cells and conc>=64 (avoid OOM; MTP reserves q=mtp_k+1)
if [ "$DP_ATTENTION" = "true" ] || [ "$CONC" -ge 64 ]; then
    PARALLEL_ARGS+=(--max-num-seqs "$CONC")
fi

BENCHMARK_MAX_MODEL_LEN="$MAX_MODEL_LEN"

if [ "${EVAL_ONLY}" = "true" ]; then
    EVAL_MAX_MODEL_LEN=$(compute_eval_context_length "$MODEL" "$BENCHMARK_MAX_MODEL_LEN")
    export EVAL_MAX_MODEL_LEN
fi

start_gpu_monitor

set -x
export ATOM_DISABLE_MMAP=true
export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1

python3 -m atom.entrypoints.openai_server \
    --model $MODEL \
    --server-port $PORT \
    "${PARALLEL_ARGS[@]}" \
    "${SPEC_ARGS[@]}" \
    --kv_cache_dtype fp8 \
    --trust-remote-code \
    --no-enable_prefix_caching \
    > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# --dsv4: InferenceX bench (utils/bench_serving) uses encoding_dsv4.py; DSv4-Pro has
# no jinja chat_template so plain --use-chat-template yields no result.
run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$((CONC * 10))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/ \
    --trust-remote-code \
    --dsv4

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
