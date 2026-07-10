#!/usr/bin/env bash

# MiniMax-M3 NVFP4 B300 single-node vLLM recipe.
# Same shape as minimaxm3_fp8_b300.sh but uses the nvidia/MiniMax-M3-NVFP4
# checkpoint. MiniMax-M3 modelopt NVFP4 support (vllm-project/vllm PR #46380) is
# baked into the perf container image, so no runtime patch is needed.

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

# Test the BF16 MiniMax-M3 routing gate against the main branch runtime.
VLLM_GATE_PATCH="$(dirname "$0")/patches/vllm-minimaxm3-gate-bf16.patch"
if ! command -v patch >/dev/null 2>&1; then
    apt-get update -y && apt-get install -y --no-install-recommends patch \
        || { echo "Failed to install patch(1)" >&2; exit 1; }
fi
VLLM_PACKAGE_DIR=$(python3 -c "import importlib.util; print(importlib.util.find_spec('vllm').submodule_search_locations[0])") \
    || { echo "Could not locate the installed vllm package" >&2; exit 1; }
VLLM_SITE_PACKAGES=$(dirname "$VLLM_PACKAGE_DIR")
patch --dry-run -p1 -d "$VLLM_SITE_PACKAGES" < "$VLLM_GATE_PATCH" >/dev/null \
    || { echo "vLLM MiniMax-M3 GateLinear BF16 patch does not apply" >&2; exit 1; }
patch -p1 -d "$VLLM_SITE_PACKAGES" < "$VLLM_GATE_PATCH" \
    || { echo "vLLM MiniMax-M3 GateLinear BF16 patch failed" >&2; exit 1; }

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

SERVER_LOG=/workspace/server.log

export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_FLOAT32_MATMUL_PRECISION=high
export VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm

if [ "${DP_ATTENTION}" = "true" ]; then
  PARALLEL_ARGS="--tensor-parallel-size=1 --data-parallel-size=$TP --enable-expert-parallel"
elif [ "$EP_SIZE" -gt 1 ]; then
  PARALLEL_ARGS="--tensor-parallel-size=$TP --enable-expert-parallel"
else
  PARALLEL_ARGS="--tensor-parallel-size=$TP"
fi

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi
start_gpu_monitor

set -x
vllm serve "$MODEL_PATH" --served-model-name "$MODEL" --host 0.0.0.0 --port $PORT \
$PARALLEL_ARGS \
--gpu-memory-utilization 0.95 \
--max-model-len $MAX_MODEL_LEN \
--kv-cache-dtype fp8 \
--block-size 128 \
--language-model-only \
--max-cudagraph-capture-size 2048 \
--max-num-batched-tokens "$((ISL * 2 ))" \
--stream-interval 20 --no-enable-prefix-caching \
--trust-remote-code > $SERVER_LOG 2>&1 &

SERVER_PID=$!

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

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
