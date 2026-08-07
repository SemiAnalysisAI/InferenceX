#!/usr/bin/env bash

# NOTE: At the time of submission, https://docs.vllm.ai/projects/recipes/en/latest/moonshotai/Kimi-K2.5.html
# does not have a B300-specific recipe, so this script reuses the existing
# Kimi-K2.5 FP4 B200 vLLM recipe as-is until B300-specific tuning is available.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    MAX_MODEL_LEN \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

PARALLEL_ARGS=(--tensor-parallel-size "$TP" --data-parallel-size 1)
GMU=0.90
PREFILL_SCHEDULE_ARGS=()
if [ "${DP_ATTENTION:-false}" = "true" ]; then
    PARALLEL_ARGS=(--tensor-parallel-size 1 --data-parallel-size "$TP")
    GMU=0.85
    PREFILL_SCHEDULE_ARGS=(--prefill-schedule-interval 4)
fi

EP_ARGS=()
if [ "${EP_SIZE:-1}" -gt 1 ]; then
    EP_ARGS=(--enable-expert-parallel)
fi


# `hf download` creates the target dir if missing and is itself idempotent. 
# When MODEL_PATH is unset (stand-alone runs), fall back to the HF_HUB_CACHE
# Either way, MODEL_PATH is what the server is launched with.
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

export TORCH_CUDA_ARCH_LIST="10.0"
export PYTHONNOUSERSITE=1
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS=""
export VLLM_RPC_TIMEOUT=600000

SERVER_LOG=/workspace/server.log

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi
# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x
vllm serve $MODEL_PATH --served-model-name $MODEL --host 0.0.0.0 --port $PORT \
"${PARALLEL_ARGS[@]}" \
"${EP_ARGS[@]}" \
"${PREFILL_SCHEDULE_ARGS[@]}" \
--gpu-memory-utilization "$GMU" \
--max-model-len $MAX_MODEL_LEN \
--max-num-seqs $CONC \
--reasoning-parser kimi_k2 \
--tool-call-parser kimi_k2 \
--compilation_config.pass_config.fuse_allreduce_rms true \
--kv-cache-dtype fp8 \
--max-cudagraph-capture-size "$((CONC * 2))" \
--stream-interval 32 \
--attention-config '{"mla_prefill_backend":"FLASHINFER","use_prefill_query_quantization":true}' \
--linear-backend flashinfer_cutlass \
--no-enable-prefix-caching \
--trust-remote-code > $SERVER_LOG 2>&1 &

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
    --num-prompts $(( CONC * 10 )) \
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
