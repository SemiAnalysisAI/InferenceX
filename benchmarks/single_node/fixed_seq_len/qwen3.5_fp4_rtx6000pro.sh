#!/usr/bin/env bash

# Qwen3.5-397B-A17B NVFP4 on four RTX PRO 6000 Blackwell GPUs.
# SM120 uses Marlin for the routed experts and Triton for attention. The node is
# PCIe-only, so collectives use NCCL rather than NVLink-tuned custom all-reduce.

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

if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
    SERVE_MODEL="$MODEL_PATH"
else
    hf download "$MODEL"
    SERVE_MODEL="$MODEL"
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

nvidia-smi

SERVER_LOG=/workspace/server.log
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

export VLLM_ENGINE_READY_TIMEOUT_S=3600

if [[ "$DP_ATTENTION" == "true" ]]; then
    PARALLEL_ARGS=(
        --tensor-parallel-size 1
        --data-parallel-size "$TP"
        --enable-expert-parallel
    )
elif [[ "$EP_SIZE" -gt 1 ]]; then
    PARALLEL_ARGS=(
        --tensor-parallel-size "$TP"
        --enable-expert-parallel
    )
else
    PARALLEL_ARGS=(--tensor-parallel-size "$TP")
fi

if [[ "$EVAL_ONLY" == "true" ]]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi

start_gpu_monitor

set -x
vllm serve "$SERVE_MODEL" \
    --served-model-name "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    "${PARALLEL_ARGS[@]}" \
    --disable-custom-all-reduce \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$CONC" \
    --max-num-batched-tokens "$((ISL * 2))" \
    --max-cudagraph-capture-size "$CONC" \
    --kv-cache-dtype fp8 \
    --language-model-only \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --enable-auto-tool-choice \
    --attention-backend TRITON_ATTN \
    --moe-backend marlin \
    --stream-interval 20 \
    --no-enable-prefix-caching \
    --trust-remote-code > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

wait_for_server_ready \
    --port "$PORT" \
    --server-log "$SERVER_LOG" \
    --server-pid "$SERVER_PID"

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

if [[ "$RUN_EVAL" == "true" ]]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
