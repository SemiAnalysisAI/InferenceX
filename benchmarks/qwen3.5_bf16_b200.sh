#!/usr/bin/env bash

source "$(dirname "$0")/benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

nvidia-smi

hf download "$MODEL"

export NCCL_NVLS_ENABLE=1
export SGL_ENABLE_JIT_DEEPGEMM=false
export SGLANG_ENABLE_FLASHINFER_GEMM=true
export PYTHONUNBUFFERED=1

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

# Low latency (conc 4,8): recv interval 10; max throughput (conc 16+): recv interval 30
if [[ $CONC -ge 16 ]]; then
  SCHEDULER_RECV_INTERVAL=30
else
  SCHEDULER_RECV_INTERVAL=10
fi

MEM_FRAC_STATIC=0.82
CHUNKED_PREFILL_SIZE=32768
MAX_PREFILL_TOKENS=32768
CUDA_GRAPH_MAX_BATCH_SIZE=$CONC
MAX_RUNNING_REQUESTS=128
CONTEXT_LENGTH=$((ISL + OSL + 20))

echo "SCHEDULER_RECV_INTERVAL: $SCHEDULER_RECV_INTERVAL, CONC: $CONC, ISL: $ISL, OSL: $OSL"

ps aux

set -x
PYTHONNOUSERSITE=1 python3 -m sglang.launch_server \
    --model-path=$MODEL \
    --served-model-name "Qwen/Qwen3.5-397B-A17B" \
    --host=0.0.0.0 \
    --port=$PORT \
    --trust-remote-code \
    --tensor-parallel-size=$TP \
    --disable-radix-cache \
    --mem-fraction-static $MEM_FRAC_STATIC \
    --chunked-prefill-size $CHUNKED_PREFILL_SIZE \
    --max-prefill-tokens $MAX_PREFILL_TOKENS \
    --cuda-graph-max-bs $CUDA_GRAPH_MAX_BATCH_SIZE \
    --max-running-requests $MAX_RUNNING_REQUESTS \
    --context-length $CONTEXT_LENGTH \
    --attention-backend trtllm_mha \
    --moe-runner-backend flashinfer_trtllm \
    --tokenizer-worker-num 6 \
    --stream-interval 30 \
    --scheduler-recv-interval $SCHEDULER_RECV_INTERVAL \
    --enable-flashinfer-allreduce-fusion \
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
    --result-dir /workspace/

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC
    append_lm_eval_summary
fi
set +x
