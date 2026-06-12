#!/usr/bin/env bash

# Day-zero MiniMax-M3 MXFP8 H100 recipe
# (https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3). --block-size 128 is
# mandatory: MSA sparse attention's block size is 128 and the KV cache
# block size must match. TP mode (dp-attn=false) optionally enables expert
# parallel; DP mode (dp-attn=true) is the recipe's "DP + Expert Parallel"
# serve mode (--data-parallel-size $TP --enable-expert-parallel).

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    EP_SIZE \
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

export PYTHONNOUSERSITE=1
export SAFETENSORS_FAST_GPU=1
# ~427 GB of MXFP8 weights; engine startup can exceed the default 600s.
export VLLM_ENGINE_READY_TIMEOUT_S=3600

if [ "${DP_ATTENTION}" = "true" ]; then
  PARALLEL_ARGS=(--tensor-parallel-size 1 --data-parallel-size "$TP" --enable-expert-parallel)
elif [ "$EP_SIZE" -gt 1 ]; then
  PARALLEL_ARGS=(--tensor-parallel-size "$TP" --enable-expert-parallel)
else
  PARALLEL_ARGS=(--tensor-parallel-size "$TP")
fi

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi

# Size the running batch and graph capture to the benchmark concurrency
# rather than the engine defaults. Under DP each rank only sees its share
# of the requests; 2x slack covers uneven load balancing.
if [ "${DP_ATTENTION}" = "true" ]; then
  MAX_NUM_SEQS=$(( ((CONC + TP - 1) / TP) * 2 ))
  if [ "$MAX_NUM_SEQS" -gt "$CONC" ]; then MAX_NUM_SEQS=$CONC; fi
else
  MAX_NUM_SEQS=$CONC
fi
CUDAGRAPH_CAPTURE_SIZE=$(( MAX_NUM_SEQS < 512 ? MAX_NUM_SEQS : 512 ))

if [ "$ISL" = "8192" ]; then
  MAX_NUM_BATCHED_TOKENS=16384
else
  MAX_NUM_BATCHED_TOKENS=8192
fi

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x
vllm serve "$MODEL" --host 0.0.0.0 --port "$PORT" \
--block-size 128 \
"${PARALLEL_ARGS[@]}" \
--gpu-memory-utilization 0.90 \
--max-model-len "$MAX_MODEL_LEN" \
--max-num-seqs "$MAX_NUM_SEQS" \
--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
--max-cudagraph-capture-size "$CUDAGRAPH_CAPTURE_SIZE" \
--language-model-only \
--tool-call-parser minimax_m3 \
--reasoning-parser minimax_m3 \
--enable-auto-tool-choice \
--no-enable-prefix-caching \
--trust-remote-code > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

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
