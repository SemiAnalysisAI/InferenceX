#!/usr/bin/env bash

# MiniMax-M3 MXFP8 MI325X (Vultr) single-node vLLM recipe
# (https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3?hardware=mi325x). ROCm sibling of
# minimaxm3_fp8_b300.sh: same M3 essentials (--block-size 128 for MSA sparse/index
# cache alignment, --language-model-only for text-only throughput, conc-scaled
# cudagraph capture) with the MI325X recipe's --attention-backend TRITON_ATTN and the
# gfx942 ROCm idioms from minimaxm2.5_fp8_mi325x.sh (HIP_VISIBLE_DEVICES for vLLM 0.14+
# Ray, VLLM_ROCM_USE_AITER). The vultr launcher bind-mounts the staged HF cache over
# HF_HUB_CACHE, so `hf download` reuses staged weights (or pulls ~444 GB MXFP8 on first
# run); the server is launched with MODEL directly, no MODEL_PATH split.

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

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

# Set HIP_VISIBLE_DEVICES to match ROCR_VISIBLE_DEVICES for Ray compatibility in vLLM 0.14+
if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

export VLLM_ROCM_USE_AITER=1

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

rocm-smi || true
amd-smi || true

SERVER_LOG=/workspace/server.log

# ~444 GB of MXFP8 weights off shared FS; engine startup can exceed the
# default 600s readiness window.
export VLLM_ENGINE_READY_TIMEOUT_S=3600

if [ "${DP_ATTENTION}" = "true" ]; then
  PARALLEL_ARGS="--tensor-parallel-size=1 --data-parallel-size=$TP --enable-expert-parallel"
elif [ "$EP_SIZE" -gt 1 ]; then
  PARALLEL_ARGS="--tensor-parallel-size=$TP --enable-expert-parallel"
else
  PARALLEL_ARGS="--tensor-parallel-size=$TP"
fi

# Fixed-seq-len runs don't need graphs past the request concurrency: capture
# up to the next power of two >= CONC, capped at vLLM's 2048 ceiling.
CAPTURE_SIZE=4
while (( CAPTURE_SIZE < CONC )); do CAPTURE_SIZE=$((CAPTURE_SIZE * 2)); done
(( CAPTURE_SIZE > 2048 )) && CAPTURE_SIZE=2048

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi
# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x
vllm serve "$MODEL" --host 0.0.0.0 --port $PORT \
$PARALLEL_ARGS \
--gpu-memory-utilization 0.90 \
--max-model-len $MAX_MODEL_LEN \
--block-size 128 \
--attention-backend TRITON_ATTN \
--language-model-only \
--max-cudagraph-capture-size $CAPTURE_SIZE \
--max-num-batched-tokens "$((ISL * 2 ))" \
--no-enable-prefix-caching \
--trust-remote-code > $SERVER_LOG 2>&1 &

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
