#!/usr/bin/env bash

# Build and install the TensorRT-LLM DeepSeek-V4 feature branch at runtime.
# This avoids relying on a custom prebuilt image while still picking up the
# branch's required C++/CUDA kernels and Python model/tokenizer code.

trtllm_dsv4_supported() {
    python3 - <<'PY'
import importlib
import sys

try:
    import tensorrt_llm  # noqa: F401
    import torch

    importlib.import_module("tensorrt_llm._torch.models.modeling_deepseekv4")
    importlib.import_module(
        "tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.deepseek_v4"
    )
    getattr(torch.ops.trtllm, "compressor_prefill_reduction")
    getattr(torch.ops.trtllm, "compressor_paged_kv_compress")
    getattr(torch.ops.trtllm, "compressor_postprocess_scatter")
except Exception as exc:
    print(f"TensorRT-LLM DeepSeek-V4 support check failed: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
}

bootstrap_trtllm_dsv4() {
    if [[ "${TRTLLM_DSV4_BOOTSTRAP:-auto}" == "0" ]]; then
        echo "TRTLLM_DSV4_BOOTSTRAP=0; skipping TensorRT-LLM DeepSeek-V4 bootstrap"
        return 0
    fi

    if [[ "${TRTLLM_DSV4_BOOTSTRAP:-auto}" != "force" ]] && trtllm_dsv4_supported; then
        echo "TensorRT-LLM DeepSeek-V4 support already available"
        return 0
    fi

    local repo="${TRTLLM_DSV4_REPO:-https://github.com/NVIDIA/TensorRT-LLM.git}"
    local branch="${TRTLLM_DSV4_BRANCH:-feat/deepseek_v4}"
    local ref="${TRTLLM_DSV4_REF:-f1c5fe143febb70cd74f0fb4ccca1516206268d7}"
    local src="${TRTLLM_DSV4_SRC:-/tmp/trtllm-dsv4-src}"
    local build_dir="${TRTLLM_DSV4_BUILD_DIR:-/tmp/trtllm-dsv4-build}"
    local dist_dir="${TRTLLM_DSV4_DIST_DIR:-/tmp/trtllm-dsv4-wheel}"
    local archs="${TRTLLM_DSV4_CUDA_ARCHITECTURES:-100-real;103-real}"
    local lock_file="${TRTLLM_DSV4_LOCK_FILE:-/tmp/trtllm-dsv4-bootstrap.lock}"

    echo "Bootstrapping TensorRT-LLM DeepSeek-V4 support"
    echo "  repo:   $repo"
    echo "  branch: $branch"
    echo "  ref:    $ref"
    echo "  archs:  $archs"

    if ! command -v git >/dev/null 2>&1; then
        if command -v apt-get >/dev/null 2>&1; then
            apt-get update
            apt-get install -y git
        else
            echo "git is required to bootstrap TensorRT-LLM DeepSeek-V4 support" >&2
            return 1
        fi
    fi

    (
        set -euo pipefail
        flock 9

        if [[ "${TRTLLM_DSV4_BOOTSTRAP:-auto}" != "force" ]] && trtllm_dsv4_supported; then
            echo "TensorRT-LLM DeepSeek-V4 support became available while waiting for bootstrap lock"
            exit 0
        fi

        if [[ ! -d "$src/.git" ]]; then
            rm -rf "$src"
            git clone \
                --filter=blob:none \
                --single-branch \
                --branch "$branch" \
                "$repo" "$src"
        fi

        cd "$src"
        git fetch origin "$branch" --depth 1
        git fetch origin "$ref" --depth 1 || true
        git checkout "$ref"
        git submodule update --init --recursive --depth 1

        if command -v git-lfs >/dev/null 2>&1; then
            git lfs install --local
            git lfs pull
        else
            echo "git-lfs not found; continuing without LFS pull"
        fi

        rm -rf "$dist_dir"
        mkdir -p "$dist_dir"

        python3 scripts/build_wheel.py \
            --cuda_architectures "$archs" \
            --build_dir "$build_dir" \
            --dist_dir "$dist_dir" \
            --clean \
            --skip-stubs \
            ${TRTLLM_DSV4_BUILD_ARGS:-}

        local wheel
        wheel="$(ls -t "$dist_dir"/tensorrt_llm*.whl | head -1)"
        python3 -m pip install --force-reinstall --no-deps "$wheel"
    ) 9>"$lock_file"

    trtllm_dsv4_supported
}
