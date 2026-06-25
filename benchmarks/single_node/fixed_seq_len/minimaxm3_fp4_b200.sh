#!/usr/bin/env bash

# MiniMax-M3 NVFP4 B200 single-node vLLM recipe.
# Same shape as minimaxm3_fp8_b200.sh but uses the nvidia/MiniMax-M3-NVFP4
# checkpoint. Applies vllm-project/vllm PR #46380 (MiniMax-M3 modelopt NVFP4
# support) from commit 6c08558 by overwriting the 3 changed files in the
# installed vLLM package before the server starts.

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

# Apply vllm-project/vllm PR #46380 (Add MiniMax-M3 modelopt NVFP4 support, commit 6c08558).
# This patch is required for nvidia/MiniMax-M3-NVFP4: without it vLLM does not
# recognise the NVFP4 quant config and falls back to an unsupported path.
VLLM_DIR=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
for f in \
  model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py \
  model_executor/layers/quantization/modelopt.py \
  model_executor/layers/quantization/utils/flashinfer_utils.py
do
  curl -fsSL "https://raw.githubusercontent.com/vllm-project/vllm/6c08558/vllm/${f}" -o "${VLLM_DIR}/${f}"
done
python3 -c "from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import TrtLlmNvFp4ExpertsModular; print('[nvfp4-patch] OK')"

# launch_b200-dgxc.sh rewrites MODEL to the pre-downloaded path; only download
# when handed a bare HF id (b200-cw / b200-nb runners).
if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

nvidia-smi

SERVER_LOG=/workspace/server.log

export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_FLOAT32_MATMUL_PRECISION=high

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
vllm serve $MODEL --port $PORT \
$PARALLEL_ARGS \
--gpu-memory-utilization 0.90 \
--max-model-len $MAX_MODEL_LEN \
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
