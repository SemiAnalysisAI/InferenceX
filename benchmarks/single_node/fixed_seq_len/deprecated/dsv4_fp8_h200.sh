#!/usr/bin/env bash

# Per https://vllm.ai/blog/deepseek-v4 the DeepSeek-V4-Pro H200 recipe uses
# the cu129 image and omits the FP4 indexer cache flag (H200 has no FP4
# path). Max-model-len is pinned at 800k per the recipe.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    DP_ATTENTION \
    CONC \
    ISL \
    OSL \
    MAX_MODEL_LEN \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

nvidia-smi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

SERVER_LOG=/workspace/server.log

# DeepSeek-V4-Pro weights are large; engine startup can exceed the default
# 600s. Give it an hour to load.
export VLLM_ENGINE_READY_TIMEOUT_S=3600

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN_ARG="--max-model-len $EVAL_MAX_MODEL_LEN"
else
    MAX_MODEL_LEN_ARG="--max-model-len $MAX_MODEL_LEN"
fi

# DP_ATTENTION=true runs DP-attention with expert parallel (DP size = TP);
# DP_ATTENTION=false runs pure tensor parallel.
PARALLEL_ARGS=(--tensor-parallel-size "$TP" --data-parallel-size 1)
if [ "${DP_ATTENTION}" = "true" ]; then
    PARALLEL_ARGS=(--tensor-parallel-size 1 --data-parallel-size "$TP")
fi

EP_ARGS=()
if [ "${EP_SIZE:-1}" -gt 1 ]; then
    EP_ARGS=(--enable-expert-parallel)
fi

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x
vllm serve $MODEL --host 0.0.0.0 --port $PORT \
--trust-remote-code \
--kv-cache-dtype fp8 \
--block-size 256 \
--no-enable-prefix-caching \
"${PARALLEL_ARGS[@]}" \
"${EP_ARGS[@]}" \
$MAX_MODEL_LEN_ARG \
--quantization deepseek_v4_fp8 \
--gpu-memory-utilization 0.95 \
--max-num-seqs 512 \
--max-num-batched-tokens 512 \
--no-enable-flashinfer-autotune \
--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
--tokenizer-mode deepseek_v4 \
--tool-call-parser deepseek_v4 \
--enable-auto-tool-choice \
--reasoning-parser deepseek_v4 > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

pip install -q datasets pandas

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
    --trust-remote-code

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
