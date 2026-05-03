#!/usr/bin/env bash

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    DP_ATTENTION \
    EP_SIZE \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

hf download "$MODEL"

# Overlay sglang from the amd/deepseek_v4 branch on top of whatever the
# rocm/sgl-dev:rocm720-mi35x-583b1b6-20260501-DSv4 image ships with. The
# branch is moving fast and we want a reproducible pin per benchmark run.
# Bump SGL_PR_SHA when the branch advances.
SGL_PR_SHA="a8410de6fba3c3d44d9c7e49af1868b3940ce654"
SGL_PR_DIR="/tmp/sglang-amd-dsv4"

if [ ! -d "$SGL_PR_DIR/.git" ]; then
    git clone --filter=blob:none https://github.com/sgl-project/sglang.git "$SGL_PR_DIR"
fi
(
    cd "$SGL_PR_DIR"
    git fetch --depth=1 origin "$SGL_PR_SHA" 2>/dev/null \
        || git fetch --depth=1 origin amd/deepseek_v4
    git checkout --force "$SGL_PR_SHA"
    test "$(git rev-parse HEAD)" = "$SGL_PR_SHA"

    # Reinstall just the Python package; the image already has the ROCm
    # kernel deps (aiter, triton, tilelang, torch) at versions matched to
    # this branch, so --no-deps avoids pip resolving them against PyPI.
    pip install --no-build-isolation --no-deps --force-reinstall -e python/
)

# Transformers in the container doesn't recognize the `deepseek_v4` model_type.
# PR #23608's fallback in hf_transformers_utils.get_config tries to handle this
# by writing a patched config to /tmp, but in practice isn't catching the error
# in this image. Patch the cached config.json directly instead: set model_type
# to `deepseek_v3` so AutoConfig.from_pretrained succeeds, and keep
# architectures=['DeepseekV4ForCausalLM'] so SGLang dispatches to its native
# DSv4 model class (python/sglang/srt/models/deepseek_v4.py).
python3 << PYEOF
import json
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id="$MODEL", filename="config.json")
with open(path) as f:
    config = json.load(f)
if config.get("model_type") == "deepseek_v4":
    config["model_type"] = "deepseek_v3"
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Patched {path}: model_type deepseek_v4 -> deepseek_v3")
else:
    print(f"No patch needed: model_type is {config.get('model_type')!r}")
PYEOF

# DSv4 FP4-experts path. Tracks the env block in python/run_dsv4.sh on the
# amd/deepseek_v4 branch (HEAD's active block is FP8; we override the two
# FP4-specific flags below):
#   SGLANG_DSV4_FP4_EXPERTS=True   -> route experts through the FP4 kernels
#   SGLANG_FORCE_TRITON_MOE_FP8=0  -> dispatch MoE through aiter and apply
#                                    the swiglu_limit clamp in the triton
#                                    MoE fallback path.
export SGLANG_REASONING_EFFORT=max
export SGLANG_OPT_USE_FUSED_COMPRESS=false
export SGLANG_OPT_USE_OLD_COMPRESSOR=true
export SGLANG_OPT_USE_TILELANG_SWA_PREPARE=false
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=false
export SGLANG_OPT_USE_FUSED_HASH_TOPK=false
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=false
export SGLANG_OPT_USE_TILELANG_MHC_PRE=false
export SGLANG_OPT_USE_TILELANG_MHC_POST=false
export SGLANG_ENABLE_THINKING=1
export SGLANG_USE_AITER=1
export SGLANG_USE_ROCM700A=1
export SGLANG_TOPK_TRANSFORM_512_TORCH=1
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
export SGLANG_DSV4_FP4_EXPERTS=True
export SGLANG_OPT_DPSK_V4_RADIX=0
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=false
export SGLANG_OPT_USE_FUSED_STORE_CACHE=false
export SGLANG_FORCE_TRITON_MOE_FP8=0
export SGLANG_HACK_FLASHMLA_BACKEND=tilelang
export SGLANG_OPT_USE_TILELANG_INDEXER=true

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

EVAL_CONTEXT_ARGS=""
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    EVAL_CONTEXT_ARGS="--context-length $EVAL_MAX_MODEL_LEN"
fi
# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

PARALLEL_ARGS=(
    --tensor-parallel-size "$TP"
)
if [ "${DP_ATTENTION}" = "true" ]; then
    PARALLEL_ARGS+=(
        --dp "$TP"
        --enable-dp-attention
    )
fi
if [ "${EP_SIZE:-1}" -gt 1 ]; then
    PARALLEL_ARGS+=(--ep-size "$EP_SIZE")
fi

python3 -m sglang.launch_server \
    --model-path $MODEL \
    --host=0.0.0.0 \
    --port $PORT \
    "${PARALLEL_ARGS[@]}" \
    --trust-remote-code \
    --disable-radix-cache \
    --attention-backend compressed \
    --max-running-request 256 \
    --page-size 256 \
    --chunked-prefill-size 8192 \
    --disable-shared-experts-fusion \
    --tool-call-parser deepseekv4 \
    --reasoning-parser deepseek-v4 \
    --chat-template "$(dirname "$0")/chat_templates/deepseek_v4_thinking.jinja" \
    --watchdog-timeout 1800 $EVAL_CONTEXT_ARGS > $SERVER_LOG 2>&1 &

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
