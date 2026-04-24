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

# DSv4 requires transformers with deepseek_v4 model type support (huggingface/transformers#45616)
python3 -m pip install --no-cache-dir \
  "git+https://github.com/ArthurZucker/transformers.git@add-deepseek-v4"

# Patch rope_theta type: config.json has int but the dataclass declares float.
# Must happen before any transformers import that triggers strict validation.
python3 << 'PYEOF'
from transformers.models.deepseek_v4 import configuration_deepseek_v4 as m
path = m.__file__
with open(path) as f:
    src = f.read()
if "rope_theta: float" in src and "Union[int, float]" not in src:
    src = src.replace("rope_theta: float", "rope_theta: Union[int, float]", 1)
    if "from typing import" in src and "Union" not in src.split("from typing import")[1].split("\n")[0]:
        src = src.replace("from typing import ", "from typing import Union, ", 1)
    with open(path, "w") as f:
        f.write(src)
    print(f"Patched rope_theta type in {path}")
PYEOF

# hf CLI breaks after huggingface_hub upgrade (typer incompatibility in container);
# use Python API directly instead.
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')"

# DSv4-specific SGLang env vars (from sgl-project/sglang#23608)
export SGLANG_OPT_USE_FUSED_COMPRESS=false
export SGLANG_OPT_USE_OLD_COMPRESSOR=true
export SGLANG_OPT_USE_TILELANG_SWA_PREPARE=false
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=false
export SGLANG_OPT_USE_FUSED_HASH_TOPK=false
export SGLANG_HACK_FLASHMLA_BACKEND=torch
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=false
export SGLANG_OPT_USE_TILELANG_MHC_PRE=false
export SGLANG_OPT_USE_TILELANG_MHC_POST=false
export SGLANG_ENABLE_THINKING=1
export SGLANG_USE_AITER=1
export SGLANG_USE_ROCM700A=1
export SGLANG_TOPK_TRANSFORM_512_TORCH=1
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
export SGLANG_DSV4_FP4_EXPERTS=false
export SGLANG_OPT_DPSK_V4_RADIX=0
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=false
export SGLANG_OPT_USE_FUSED_STORE_CACHE=false
export SGLANG_FORCE_TRITON_MOE_FP8=1

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

EVAL_CONTEXT_ARGS=""
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    EVAL_CONTEXT_ARGS="--context-length $EVAL_MAX_MODEL_LEN"
fi
# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

python3 -m sglang.launch_server \
    --model-path $MODEL \
    --host=0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size $TP \
    --dp $TP \
    --enable-dp-attention \
    --trust-remote-code \
    --disable-radix-cache \
    --attention-backend compressed \
    --max-running-request 256 \
    --page-size 256 \
    --chunked-prefill-size 8192 \
    --disable-shared-experts-fusion \
    --disable-cuda-graph \
    --tool-call-parser deepseekv4 \
    --reasoning-parser deepseek-v4 $EVAL_CONTEXT_ARGS > $SERVER_LOG 2>&1 &

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
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
