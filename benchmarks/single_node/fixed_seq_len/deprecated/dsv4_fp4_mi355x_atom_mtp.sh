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

# MTP speculative decoding (ATOM self-draft).
SPEC_ARGS=(--method mtp --num-speculative-tokens 3)

# max_req=conc for dp-on cells (dp-attention keeps a full KV pool per rank, and MTP
# reserves q=num_speculative_tokens+1 per request, so the large default max_num_seqs
# OOMs) and for conc>=64. dp-off low conc uses the ATOM default.
if [ "$DP_ATTENTION" = "true" ] || [ "$CONC" -ge 64 ]; then
    PARALLEL_ARGS+=(--max-num-seqs "$CONC")
fi

# prefill-only TBO (--enable-tbo -> argparse const=prefill -> enable_tbo_decode=False):
# MTP-compatible because it never touches decode. (--enable-tbo all / decode-TBO WOULD
# drop MTP's spec_decode_metadata in UBatchWrapper -- that is the real incompatibility;
# prefill-only does not.) Threshold is conc>=256 for MTP (NOT the non-MTP #2327 conc>=64):
# measured crossover (TBO run 30257759947 vs non-TBO run 30238071409, same image/harness)
# shows TBO HURTS at c64/c128 (-14%/-10% output tput) but HELPS at c256+ (+11/+8/+14%).
# MTP already removes low-conc latency, so TBO overlap there is pure overhead; the win
# only appears once high conc turns compute-bound. Matches ATOM models.json non-MTP
# DPA-TBO 256-2048 band. Together with GPU_MAX_HW_QUEUES=5.
if [ "$DP_ATTENTION" = "true" ] && [ "$CONC" -ge 256 ]; then
    PARALLEL_ARGS+=(--enable-tbo)
    export GPU_MAX_HW_QUEUES=5
fi

BENCHMARK_MAX_MODEL_LEN="$MAX_MODEL_LEN"

if [ "${EVAL_ONLY}" = "true" ]; then
    EVAL_MAX_MODEL_LEN=$(compute_eval_context_length "$MODEL" "$BENCHMARK_MAX_MODEL_LEN")
    export EVAL_MAX_MODEL_LEN
fi

# Start GPU monitoring (power, temperature, clocks every second)
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

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# --dsv4: InferenceX's bench (utils/bench_serving/benchmark_serving.py) uses the
# DeepSeek-V4 chat encoder (encoding_dsv4.py) instead of the tokenizer's jinja
# template. DSv4-Pro has NO jinja chat_template, so a plain --use-chat-template
# produces empty/broken prompts (bench yields no result). --dsv4 implies
# --use-chat-template. Chat-formatted inputs are required for MTP: EAGLE/MTP
# acceptance silently regresses on raw random tokens (AGENTS.md).
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

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
