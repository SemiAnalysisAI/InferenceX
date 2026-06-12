#!/usr/bin/env bash

# MiniMax-M3 MXFP8 MI355X (gfx950) single-node vLLM recipe
# (https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3). MXFP8 uses native MXFP8
# matrix cores on CDNA4 and runs from TP=4 per the recipe; main-model
# attention on AMD is the Triton backend (TRITON_ATTN). --block-size 128 is
# mandatory (MSA sparse_block_size; default 16 fails with "No common block
# size for 16" on AMD).
#
# Day-zero enablement: no public ROCm image carries M3 support yet
# (vllm-project/vllm#45381 unmerged; the recipe's AMD image is a placeholder).
# Two pieces are missing from the installed nightly wheel and we add them at
# job start:
#   1) The M3 python tree (vllm/models/minimax_m3/{amd,common,nvidia}) — overlaid
#      from the PR's m3_release branch onto the installed package.
#   2) The fused attention pre-processing kernel
#      (fused_minimax_m3_qknorm_rope_kv_insert) — the AMD model path calls this
#      compiled op on EVERY forward (Gemma QK-norm + partial-NeoX RoPE +
#      bf16 KV/index-cache scatter). It is NOT NVIDIA-only; it lives in
#      csrc/libtorch_stable/ and the kernel author guards the ROCm path with
#      USE_ROCM. The wheel's _C extension (base commit 6fbfdd18) predates it,
#      so we compile just that one .cu as a supplemental libtorch-stable .so for
#      gfx950 and load it into the _C namespace. Cached on the shared HF mount so
#      only the first job per image pays the (~1-2 min) build cost.

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

M3_BRANCH=m3_release
M3_SRC=/tmp/vllm-m3

# 1) Overlay the m3_release python tree if this vllm build doesn't know M3 yet.
if ! python3 -c "import vllm.models.minimax_m3" 2>/dev/null; then
    echo "Overlaying vLLM ${M3_BRANCH} python tree onto installed package"
    rm -rf "$M3_SRC"
    git clone --depth 1 --branch "$M3_BRANCH" https://github.com/vllm-project/vllm.git "$M3_SRC"
    VLLM_PKG_DIR=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
    cp -r "$M3_SRC"/vllm/* "$VLLM_PKG_DIR/"
    python3 -c "import vllm.models.minimax_m3; print('m3 overlay OK')"
fi

# 2) Build + load the fused _C op if the installed wheel lacks the symbol.
if ! python3 -c "import torch,sys; sys.exit(0 if hasattr(torch.ops._C,'fused_minimax_m3_qknorm_rope_kv_insert') else 1)" 2>/dev/null; then
    M3_BUILD_DIR="${HF_HUB_CACHE%/}/m3build-wave32"
    M3_SO="$M3_BUILD_DIR/ext/minimax_m3_fused_op.so"
    mkdir -p "$M3_BUILD_DIR/ext"
    [ -d "$M3_SRC" ] || git clone --depth 1 --branch "$M3_BRANCH" https://github.com/vllm-project/vllm.git "$M3_SRC"

    # The kernel operates on 32-lane logical warps (`width=32`) on every
    # platform. Upstream currently uses a 64-lane active mask under USE_ROCM,
    # which faults on gfx950 at the first shuffle. Keep the mask aligned with
    # the explicit shuffle width until the upstream HIP fix lands.
    sed -i 's/#define FINAL_MASK 0xffffffffffffffffULL/#define FINAL_MASK 0xffffffffULL/' \
        "$M3_SRC/csrc/libtorch_stable/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu"

    # Minimal stable-ABI registration for the single op, mirroring
    # csrc/libtorch_stable/torch_bindings.cpp (def schema + CUDA impl).
    cat > "$M3_BUILD_DIR/binding.cpp" <<'CPPEOF'
// Day-zero: register fused_minimax_m3_qknorm_rope_kv_insert into the _C
// namespace from a supplemental .so (the nightly wheel's _C predates it).
#include "ops.h"
#include "core/registration.h"
#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(
      "fused_minimax_m3_qknorm_rope_kv_insert("
      "Tensor! qkv, Tensor q_norm_weight, Tensor k_norm_weight, "
      "Tensor cos_sin_cache, Tensor positions, int num_heads, "
      "int num_kv_heads, int rotary_dim, float eps, "
      "Tensor? index_q_norm_weight, Tensor? index_k_norm_weight, "
      "int num_index_heads, "
      "Tensor? slot_mapping, Tensor? index_slot_mapping, "
      "Tensor!? kv_cache, Tensor!? index_cache, "
      "int block_size, Tensor!? q_out, Tensor!? index_q_out) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl("fused_minimax_m3_qknorm_rope_kv_insert",
           TORCH_BOX(&fused_minimax_m3_qknorm_rope_kv_insert));
}
CPPEOF

    # Use the cached .so if it loads on this image; otherwise build it. The
    # launcher gives one job per node (--exclusive), so node-local /tmp is a
    # safe build scratch dir; the result is published to the shared mount via an
    # atomic rename so concurrent jobs on other nodes reuse it.
    if ! python3 -c "import torch; torch.ops.load_library('$M3_SO')" 2>/dev/null; then
        echo "Building MiniMax-M3 fused op for ROCm (gfx950)"
        LOCAL_EXT=/tmp/m3ext
        rm -rf "$LOCAL_EXT"; mkdir -p "$LOCAL_EXT"
        M3_KERNEL_SRC="$M3_SRC/csrc" M3_BIND="$M3_BUILD_DIR/binding.cpp" M3_OUT="$LOCAL_EXT" python3 <<'PYEOF'
import os, torch
from torch.utils.cpp_extension import load
src = os.environ["M3_KERNEL_SRC"]
stable = src + "/libtorch_stable"
flags = ["-DUSE_ROCM=1", "-O3", "-std=c++17"]
load(
    name="minimax_m3_fused_op",
    sources=[stable + "/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu",
             os.environ["M3_BIND"]],
    extra_include_paths=[stable, src],   # libtorch_stable first: stable ops.h wins
    extra_cflags=flags,
    extra_cuda_cflags=flags,
    build_directory=os.environ["M3_OUT"],
    with_cuda=True,
    is_python_module=False,
    verbose=True,
)
assert hasattr(torch.ops._C, "fused_minimax_m3_qknorm_rope_kv_insert"), "op did not register"
print("m3 fused op build OK")
PYEOF
        cp -f "$LOCAL_EXT/minimax_m3_fused_op.so" "$M3_SO.tmp.$$" && mv -f "$M3_SO.tmp.$$" "$M3_SO"
    fi

    # Have _custom_ops load the prebuilt op into _C in every (worker) process.
    VLLM_PKG_DIR=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
    if ! grep -q "MiniMax-M3 day-zero" "$VLLM_PKG_DIR/_custom_ops.py"; then
        cat >> "$VLLM_PKG_DIR/_custom_ops.py" <<PYEOF2

# --- MiniMax-M3 day-zero overlay: load prebuilt fused _C op if missing ---
import os as _m3_os
_m3_so = _m3_os.environ.get("M3_FUSED_OP_SO", "$M3_SO")
if _m3_so and _m3_os.path.exists(_m3_so) and not hasattr(torch.ops._C, "fused_minimax_m3_qknorm_rope_kv_insert"):
    torch.ops.load_library(_m3_so)
PYEOF2
    fi
    export M3_FUSED_OP_SO="$M3_SO"
    python3 -c "import vllm._custom_ops, torch; assert hasattr(torch.ops._C,'fused_minimax_m3_qknorm_rope_kv_insert'); print('m3 fused op load OK')"
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

# ROCm graph capture raises HSA_STATUS_ERROR_EXCEPTION during the first batch.
set -x
vllm serve $MODEL --port $PORT \
$PARALLEL_ARGS \
--gpu-memory-utilization 0.92 \
--max-model-len $MAX_MODEL_LEN \
--block-size 128 \
--language-model-only \
--attention-backend TRITON_ATTN \
--enforce-eager \
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
