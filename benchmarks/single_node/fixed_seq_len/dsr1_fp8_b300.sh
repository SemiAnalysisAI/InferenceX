#!/usr/bin/env bash

# https://cookbook.sglang.io/autoregressive/DeepSeek/DeepSeek-R1 has no B300-specific recipe; this reuses the B200 SGLang tuning.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME \
    EP_SIZE

if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

nvidia-smi


export SGL_ENABLE_JIT_DEEPGEMM=false
export SGLANG_ENABLE_FLASHINFER_GEMM=true
SERVER_LOG=/workspace/server.log

if [[ $TP -eq 8 ]]; then
  if [[ $CONC -ge 16 ]]; then
    SCHEDULER_RECV_INTERVAL=30
  else
    SCHEDULER_RECV_INTERVAL=10
  fi

  # Capped so KV memory is not reserved for requests that never run.
  MAX_RUNNING_REQUESTS=128
  CUDA_GRAPH_MAX_BATCH_SIZE=128

  MEM_FRAC_STATIC=0.82
  CHUNKED_PREFILL_SIZE=32768
  MAX_PREFILL_TOKENS=32768
elif [[ $TP -eq 4 ]]; then
  if [[ $ISL -ne 8192 ]] || [[ $OSL -ne 1024 ]]; then
    echo "TP=4 not yet supported for ISL=$ISL OSL=$OSL!"
    exit 1
  fi

  # Capped so KV memory is not reserved for requests that never run.
  MAX_RUNNING_REQUESTS=32
  CUDA_GRAPH_MAX_BATCH_SIZE=32

  MEM_FRAC_STATIC=0.95
  CHUNKED_PREFILL_SIZE=8192
  MAX_PREFILL_TOKENS=8192

  SCHEDULER_RECV_INTERVAL=10
else
  echo "Unrecognized TP size $TP!"
  exit 1
fi
echo "SCHEDULER_RECV_INTERVAL: $SCHEDULER_RECV_INTERVAL, CONC: $CONC, ISL: $ISL, OSL: $OSL"

EVAL_CONTEXT_ARGS=""
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    EVAL_CONTEXT_ARGS="--context-length $EVAL_MAX_MODEL_LEN"
fi
start_gpu_monitor

set -x
PYTHONNOUSERSITE=1 python3 -m sglang.launch_server --model-path $MODEL_PATH --served-model-name $MODEL --host 0.0.0.0 --port $PORT \
--tensor-parallel-size $TP --data-parallel-size 1 \
--cuda-graph-max-bs $CUDA_GRAPH_MAX_BATCH_SIZE --max-running-requests $MAX_RUNNING_REQUESTS \
--mem-fraction-static $MEM_FRAC_STATIC --kv-cache-dtype fp8_e4m3 --chunked-prefill-size $CHUNKED_PREFILL_SIZE --max-prefill-tokens $MAX_PREFILL_TOKENS \
--enable-flashinfer-allreduce-fusion --scheduler-recv-interval $SCHEDULER_RECV_INTERVAL --disable-radix-cache \
--attention-backend trtllm_mla --stream-interval 30 --ep-size $EP_SIZE --moe-runner-backend flashinfer_trtllm --quantization fp8 $EVAL_CONTEXT_ARGS > $SERVER_LOG 2>&1 &

SERVER_PID=$!

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

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
