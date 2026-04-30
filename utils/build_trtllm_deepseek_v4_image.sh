#!/usr/bin/env bash

set -euo pipefail

TRTLLM_REPO="${TRTLLM_REPO:-https://github.com/NVIDIA/TensorRT-LLM.git}"
TRTLLM_REF="${TRTLLM_REF:-feat/deepseek_v4}"
TRTLLM_COMMIT="${TRTLLM_COMMIT:-f1c5fe143febb70cd74f0fb4ccca1516206268d7}"
IMAGE_WITH_TAG="${IMAGE_WITH_TAG:-ghcr.io/semianalysiswork/trtllm-deepseek-v4:feat-deepseek_v4-f1c5fe1}"
CUDA_ARCHS="${CUDA_ARCHS:-100-real;103-real}"
PUSH="${PUSH:-0}"
KEEP_SRC="${KEEP_SRC:-0}"

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1" >&2
        exit 1
    fi
}

to_enroot_image() {
    local image="$1"
    local registry="${image%%/*}"
    local rest="${image#*/}"

    if [[ "$image" == "$rest" ]]; then
        printf '%s\n' "$image"
    elif [[ "$registry" == *.* || "$registry" == *:* || "$registry" == "localhost" ]]; then
        printf '%s#%s\n' "$registry" "$rest"
    else
        printf '%s\n' "$image"
    fi
}

require_cmd docker
require_cmd git
require_cmd make

if ! docker buildx version >/dev/null 2>&1; then
    echo "docker buildx is required to build TensorRT-LLM release images." >&2
    exit 1
fi

if ! git lfs version >/dev/null 2>&1; then
    echo "git-lfs is required. Install it, then rerun this script." >&2
    exit 1
fi

WORKDIR=""
if [[ -n "${TRTLLM_SRC_DIR:-}" ]]; then
    SRC_DIR="$TRTLLM_SRC_DIR"
else
    WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/trtllm-dsv4-build.XXXXXX")"
    SRC_DIR="$WORKDIR/TensorRT-LLM"
fi

cleanup() {
    if [[ -n "$WORKDIR" && "$KEEP_SRC" != "1" ]]; then
        rm -rf "$WORKDIR"
    elif [[ -n "$WORKDIR" ]]; then
        echo "Keeping TensorRT-LLM checkout at $SRC_DIR"
    fi
}
trap cleanup EXIT

if [[ ! -d "$SRC_DIR/.git" ]]; then
    git clone --recurse-submodules --branch "$TRTLLM_REF" "$TRTLLM_REPO" "$SRC_DIR"
fi

cd "$SRC_DIR"
git fetch origin "$TRTLLM_REF"
git checkout -B "$TRTLLM_REF" "origin/$TRTLLM_REF" 2>/dev/null || git checkout "$TRTLLM_REF"
if [[ -n "$TRTLLM_COMMIT" ]]; then
    git checkout "$TRTLLM_COMMIT"
fi
git submodule update --init --recursive
git lfs install --local
git lfs pull

ACTUAL_COMMIT="$(git rev-parse HEAD)"

echo "Building TensorRT-LLM DeepSeek-V4 image"
echo "  source: $TRTLLM_REPO"
echo "  ref:    $TRTLLM_REF"
echo "  commit: $ACTUAL_COMMIT"
echo "  image:  $IMAGE_WITH_TAG"
echo "  archs:  $CUDA_ARCHS"

make -C docker release_build \
    IMAGE_WITH_TAG="$IMAGE_WITH_TAG" \
    CUDA_ARCHS="$CUDA_ARCHS" \
    GIT_COMMIT="$ACTUAL_COMMIT"

if [[ "$PUSH" == "1" ]]; then
    docker push "$IMAGE_WITH_TAG"
fi

echo
echo "Docker image: $IMAGE_WITH_TAG"
echo "InferenceX/enroot image string: $(to_enroot_image "$IMAGE_WITH_TAG")"
