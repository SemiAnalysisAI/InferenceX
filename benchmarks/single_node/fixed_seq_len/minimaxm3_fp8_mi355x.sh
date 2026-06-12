#!/usr/bin/env bash

# MiniMax-M3 MXFP8 MI355X (gfx950) single-node vLLM recipe
# (https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3). MXFP8 uses native MXFP8
# matrix cores on CDNA4 and runs from TP=4 per the recipe; main-model
# attention on AMD is the Triton backend (TRITON_ATTN). --block-size 128 is
# mandatory (MSA sparse_block_size; default 16 fails with "No common block
# size for 16" on AMD).
#
# Day-zero caveat: no public ROCm image carries M3 support yet
# (vllm-project/vllm#45381 unmerged; the recipe's AMD image is a placeholder).
# The M3 AMD path is pure Python/Triton (vllm/models/minimax_m3/{amd,common}),
# so we overlay the m3_release python tree onto the installed nightly wheel —
# the image's base commit (6fbfdd18) is ~6 commits behind the PR merge-base
# (0cd9b7af), keeping drift minimal. Compiled .so artifacts from the wheel are
# preserved; the new csrc kernels in the PR are NVIDIA-only.

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

# Overlay the m3_release python tree if this vllm build doesn't know M3 yet.
if ! python3 -c "import vllm.models.minimax_m3" 2>/dev/null; then
    echo "Overlaying vLLM m3_release python tree onto installed package"
    git clone --depth 1 --branch m3_release https://github.com/vllm-project/vllm.git /tmp/vllm-m3
    VLLM_PKG_DIR=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
    cp -r /tmp/vllm-m3/vllm/* "$VLLM_PKG_DIR/"
    python3 -c "import vllm.models.minimax_m3; print('m3 overlay OK')"
fi

# Weights live on the NFS hub cache (/it-share/hf-hub-cache mounted as
# HF_HUB_CACHE by launch_mi355x-amds.sh) — pre-downloaded; this is a no-op
# when the snapshot is complete.
if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

# Set HIP_VISIBLE_DEVICES to match ROCR_VISIBLE_DEVICES for Ray compatibility in vLLM 0.14+
if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

SERVER_LOG=/workspace/server.log

# 444 GB of MXFP8 weights off NFS; engine startup can exceed the default
# 600s readiness window.
export VLLM_ENGINE_READY_TIMEOUT_S=3600

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

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x
vllm serve $MODEL --port $PORT \
$PARALLEL_ARGS \
--gpu-memory-utilization 0.92 \
--max-model-len $MAX_MODEL_LEN \
--block-size 128 \
--language-model-only \
--attention-backend TRITON_ATTN \
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
