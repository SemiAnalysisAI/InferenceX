#!/usr/bin/env bash

source "$(dirname "$0")/benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME \
    EP_SIZE

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

nvidia-smi

hf download "$MODEL"

export SGLANG_ENABLE_JIT_DEEPGEMM=false

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

# MTP only supports TP=8 for now
if [[ $TP -ne 8 ]]; then
  echo "MTP only supports TP=8, got TP=$TP!"
  exit 1
fi

# Default: recv every ~10 requests; if CONC >= 16, relax to ~30 requests between scheduler recv polls.
if [[ $CONC -ge 16 ]]; then
  SCHEDULER_RECV_INTERVAL=30
else
  SCHEDULER_RECV_INTERVAL=10
fi

# Setting these values (passed in to --cuda-graph-max-bs and --max-running-requests) as the maximum concurrency
# this will help us save memory from being unnecessary used.
MAX_RUNNING_REQUESTS=512
CUDA_GRAPH_MAX_BATCH_SIZE=512

MEM_FRAC_STATIC=0.82
CHUNKED_PREFILL_SIZE=16384
MAX_PREFILL_TOKENS=16384

echo "SCHEDULER_RECV_INTERVAL: $SCHEDULER_RECV_INTERVAL, CONC: $CONC, ISL: $ISL, OSL: $OSL"

# MTP (Multi-Token Prediction) Config - EAGLE speculative decoding
SPECULATIVE_NUM_STEPS=2
SPECULATIVE_DRAFT_TOKENS=3
SPECULATIVE_EAGLE_TOPK=1

SGLANG_ENABLE_SPEC_V2=1

set -x
PYTHONNOUSERSITE=1 python3 -m sglang.launch_server \
    --model-path=$MODEL \
    --host=0.0.0.0 \
    --port=$PORT \
    --tensor-parallel-size=$TP \
    --data-parallel-size=1 \
    --cuda-graph-max-bs $CUDA_GRAPH_MAX_BATCH_SIZE \
    --max-running-requests $MAX_RUNNING_REQUESTS \
    --mem-fraction-static $MEM_FRAC_STATIC \
    --kv-cache-dtype fp8_e4m3 \
    --chunked-prefill-size $CHUNKED_PREFILL_SIZE \
    --max-prefill-tokens $MAX_PREFILL_TOKENS \
    --enable-flashinfer-allreduce-fusion \
    --scheduler-recv-interval $SCHEDULER_RECV_INTERVAL \
    --disable-radix-cache \
    --fp8-gemm-backend=flashinfer_trtllm \
    --attention-backend trtllm_mla \
    --stream-interval 30 \
    --ep-size $EP_SIZE \
    --moe-runner-backend flashinfer_trtllm \
    --quantization fp8 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps $SPECULATIVE_NUM_STEPS \
    --speculative-num-draft-tokens $SPECULATIVE_DRAFT_TOKENS \
    --speculative-eagle-topk $SPECULATIVE_EAGLE_TOPK \
    > $SERVER_LOG 2>&1 &

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
    --use-chat-template

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC
    append_lm_eval_summary
fi
set +x