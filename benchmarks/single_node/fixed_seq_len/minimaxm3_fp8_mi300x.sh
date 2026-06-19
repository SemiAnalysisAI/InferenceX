#!/usr/bin/env bash

# MiniMax-M3 MXFP8 MI300X (gfx942) single-node vLLM recipe.
# Reuses the dedicated ROCm image and converts MXFP8 MoE weights to 128x128
# block FP8 at load time. Block size 128 is mandatory for MSA sparse attention.
# The second runtime patch carries the profiled sparse-attention, indexer, MoE,
# router, and collective changes. Only TP8 enables the pinned AITER Gemma fusion;
# EP keeps the faster native collectives.
# Keep the default BF16 KV cache on gfx942: the checkpoint has no calibrated
# q/prob scales for ROCm FP8 attention, and vLLM's fallback scale of 1.0
# corrupts model accuracy.
# Target image vLLM revision: 4a560dd8db67c270f5e2afb614558271b76f2294.

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

if ! VLLM_PACKAGE_ROOT="$(
    python3 - <<'PY'
from pathlib import Path

import vllm

print(Path(vllm.__file__).resolve().parent.parent)
PY
)"; then
    echo "Failed to locate the installed vLLM package" >&2
    exit 1
fi
if [[ -z "$VLLM_PACKAGE_ROOT" || ! -d "$VLLM_PACKAGE_ROOT/vllm" ]]; then
    echo "Invalid installed vLLM package root: $VLLM_PACKAGE_ROOT" >&2
    exit 1
fi

apply_vllm_patch() {
    local patch_label="$1"
    local patch_path="$2"
    local -a patch_check_args=(
        --batch
        --silent
        -d "$VLLM_PACKAGE_ROOT"
        -p1
        --dry-run
    )

    if [[ ! -f "$patch_path" ]]; then
        echo "$patch_label patch is missing: $patch_path" >&2
        exit 1
    fi
    if patch "${patch_check_args[@]}" --reverse --forward < "$patch_path"; then
        echo "$patch_label patch is already fully applied"
    elif patch "${patch_check_args[@]}" --forward < "$patch_path"; then
        if ! patch --batch --forward -d "$VLLM_PACKAGE_ROOT" -p1 < "$patch_path"; then
            echo "Failed to apply the $patch_label patch" >&2
            exit 1
        fi
    else
        echo "Installed vLLM cannot cleanly apply the $patch_label patch" >&2
        exit 1
    fi
    if ! patch "${patch_check_args[@]}" --reverse --forward < "$patch_path"; then
        echo "$patch_label patch verification failed" >&2
        exit 1
    fi
}

download_verified() {
    local url="$1"
    local sha256="$2"
    local output="$3"
    local temporary="${output}.tmp.$$"

    if [[ -f "$output" ]] \
        && printf '%s  %s\n' "$sha256" "$output" \
            | sha256sum --check --status; then
        return 0
    fi
    rm -f "$output" "$temporary"
    if ! curl \
        --fail \
        --location \
        --retry 5 \
        --retry-delay 2 \
        --output "$temporary" \
        "$url"; then
        echo "Failed to download $url" >&2
        return 1
    fi
    if ! printf '%s  %s\n' "$sha256" "$temporary" | sha256sum --check --status; then
        echo "SHA256 verification failed for $url" >&2
        rm -f "$temporary"
        return 1
    fi
    mv "$temporary" "$output"
}

setup_tp_aiter_gemma_fusion() {
    export VLLM_ROCM_USE_AITER=0
    export VLLM_ROCM_USE_AITER_MOE=0
    export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0
    export VLLM_ROCM_USE_AITER_FUSED_ALLREDUCE_GEMMA_RMSNORM=0

    # The fused collective wins on the profiled TP8 decode shapes, but loses
    # at both EP boundaries. Keep every other AITER backend disabled.
    if [[ "$DP_ATTENTION" == "true" || "$EP_SIZE" -gt 1 || "$TP" -ne 8 ]]; then
        echo "Using native collectives for MiniMax M3 EP/non-TP8"
        return 0
    fi

    local aiter_commit="a40c487b3c01dc03fd3872d65b1f7404f669471f"
    local cache_root="${XDG_CACHE_HOME:-$HOME/.cache}/inferencex/minimax-m3-aiter"
    local aiter_archive="$cache_root/aiter-${aiter_commit}.tar.gz"
    local aiter_root="/tmp/aiter-${aiter_commit}"
    local flydsl_wheel="$cache_root/flydsl-0.2.1-cp312-cp312-manylinux_2_27_x86_64.whl"

    mkdir -p "$cache_root" || return 1
    download_verified \
        "https://codeload.github.com/ROCm/aiter/tar.gz/${aiter_commit}" \
        "8cf142a4210e7a6fb88211b1a521c789f652e9f819ac6a0218cdeebc18f4808d" \
        "$aiter_archive" || return 1
    download_verified \
        "https://files.pythonhosted.org/packages/59/16/c87972f06b8f9a9b6ab08b598d706b687a969750df7131fc27aebae1a87a/flydsl-0.2.1-cp312-cp312-manylinux_2_27_x86_64.whl" \
        "98aa84678a515535283bf1a4b3e491c6f38de1fe16452dc8bfa44e9bd28ca99c" \
        "$flydsl_wheel" || return 1

    rm -rf "$aiter_root"
    mkdir -p "$aiter_root" || return 1
    tar \
        --extract \
        --gzip \
        --file "$aiter_archive" \
        --directory "$aiter_root" \
        --strip-components 1 || return 1
    printf 'develop\n' > "$aiter_root/aiter/install_mode" || return 1
    python3 -m pip install \
        --disable-pip-version-check \
        --no-index \
        --no-deps \
        "$flydsl_wheel" || return 1

    export PYTHONPATH="$aiter_root${PYTHONPATH:+:$PYTHONPATH}"
    export AITER_JIT_DIR="$cache_root/jit"
    export TORCH_EXTENSIONS_DIR="$cache_root/torch-extensions"
    export AITER_REBUILD=0
    export MAX_JOBS=32
    export VLLM_ROCM_USE_AITER_FUSED_ALLREDUCE_GEMMA_RMSNORM=1
    mkdir -p "$AITER_JIT_DIR" "$TORCH_EXTENSIONS_DIR"

    (
        flock -w 1800 9 || {
            echo "Timed out waiting for the MiniMax M3 AITER build lock" >&2
            exit 1
        }
        python3 - <<'PY'
import inspect

from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

assert "gemma_norm" in inspect.signature(CustomAllreduce.fused_ar_rms).parameters
PY
    ) 9> "$cache_root/build.lock" || return 1
}

PATCH_DIR="$(dirname "$0")"
apply_vllm_patch \
    "MI300X block-FP8 conversion" \
    "$PATCH_DIR/minimaxm3_mi300x_mxfp8.patch"
apply_vllm_patch \
    "MI300X profile-guided kernels and collectives" \
    "$PATCH_DIR/minimaxm3_mi300x_profiled.patch"
if ! setup_tp_aiter_gemma_fusion; then
    echo "Failed to install the pinned TP-only AITER collective" >&2
    exit 1
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

SERVER_LOG=/workspace/server.log
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_BREAKABLE_CUDAGRAPH=0

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
fi

PARALLEL_ARGS=(--tensor-parallel-size "$TP")
if [ "${DP_ATTENTION}" = "true" ]; then
    PARALLEL_ARGS=(
        --tensor-parallel-size 1
        --data-parallel-size "$TP"
        --enable-expert-parallel
    )
elif [ "$EP_SIZE" -gt 1 ]; then
    PARALLEL_ARGS+=(--enable-expert-parallel)
fi

SCHEDULER_ARGS=()
if (( ISL >= 8192 && CONC >= 16 )); then
    # The 32K budget keeps long-prefill chunks large enough to avoid starving
    # decode at the measured 8k1k c16/c128/c256 and 32k1k c16 points.
    SCHEDULER_ARGS+=(--max-num-batched-tokens 32768)
fi

start_gpu_monitor

set -x
vllm serve "$MODEL" --port "$PORT" \
    "${PARALLEL_ARGS[@]}" \
    "${SCHEDULER_ARGS[@]}" \
    --block-size 128 \
    --no-enable-prefix-caching \
    --language-model-only \
    --max-model-len "$MAX_MODEL_LEN" \
    --attention-backend TRITON_ATTN \
    --tool-call-parser minimax_m3 \
    --reasoning-parser minimax_m3 \
    --enable-auto-tool-choice > "$SERVER_LOG" 2>&1 &

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
