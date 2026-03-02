#!/usr/bin/env bash

source "$(dirname "$0")/../benchmark_lib.sh"

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

# GLM-5 requires transformers with glm_moe_dsa model type support.
export DEBIAN_FRONTEND=noninteractive
apt-get update && \
  apt-get install -y --no-install-recommends git build-essential && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

python3 -m pip install -U --no-cache-dir \
  "git+https://github.com/huggingface/transformers.git@6ed9ee36f608fd145168377345bfc4a5de12e1e2"

hf download "$MODEL"

# ROCm / SGLang performance tuning for MI355X
export SGLANG_ROCM_FUSED_DECODE_MLA=0
export SGLANG_USE_AITER=1
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export HSA_NO_SCRATCH_RECLAIM=1
export SAFETENSORS_FAST_GPU=1

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

python3 -m sglang.launch_server \
    --model-path $MODEL \
    --host=0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size $TP \
    --trust-remote-code \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --mem-fraction-static 0.85 \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' \
    --log-level info \
    --nsa-prefill-backend tilelang \
    --nsa-decode-backend tilelang > $SERVER_LOG 2>&1 &

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
    --result-dir /workspace/

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC
    append_lm_eval_summary
fi
set +x
