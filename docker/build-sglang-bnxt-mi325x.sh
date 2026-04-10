#!/bin/bash
# Build SGLang container image for MI325X with Broadcom bnxt_re RDMA support.
#
# Prerequisites:
#   - Docker installed and running on a compute node with GPU access
#   - Broadcom BCM driver archive placed in docker/ directory (see BCM_DRIVER below)
#   - Docker Hub credentials: login as 'clustermax' to semianalysiswork org
#     (PAT in /nfsdata/sa/.j9s/InferenceX/.env.local as DOCKER_HUB_PAT)
#
# This image supports:
#   - AMD Instinct MI325X (gfx942 CDNA3) — also works on MI300X (same arch)
#   - SGLang (latest main) with Qwen3.5 MoE, GLM-5 MoE, DeepSeek-R1 model support
#   - MoRI disaggregated inference with Broadcom Thor 2 IBGDA/RoCEv2
#   - AITER optimized kernels, TileLang NSA backends
#
# Usage:
#   cd /path/to/InferenceX/docker
#   bash build-sglang-bnxt-mi325x.sh
#
# The image is pushed to:
#   docker.io/semianalysiswork/sgl-bnxt-cdna3:latest-bnxt-mori
#
# Override defaults with env vars:
#   SGL_BRANCH=v0.5.10 IMAGE_TAG=semianalysiswork/sgl-bnxt-cdna3:v0.5.10-bnxt-mori bash build-sglang-bnxt-mi325x.sh
#
# Build reference: https://github.com/JordanNanos/sglang/tree/main/docker

set -euo pipefail

# ---------- Configuration ----------
SGL_BRANCH="${SGL_BRANCH:-v0.5.10}"
GPU_ARCH="gfx942"
IMAGE_TAG="${IMAGE_TAG:-semianalysiswork/sgl-bnxt-cdna3:v0.5.10-bnxt}"
DOCKERFILE_REPO="https://github.com/JordanNanos/sglang.git"
DOCKERFILE_REF="main"

# Broadcom BCM driver — must be placed in the build context directory.
# Download from Broadcom support portal (requires account).
BCM_DRIVER="bcm5760x_231.2.63.0a.zip"

# ---------- Clone build repo ----------
WORK_DIR=$(mktemp -d)
echo "[build] Cloning ${DOCKERFILE_REPO} (ref: ${DOCKERFILE_REF}) into ${WORK_DIR}"
git clone --depth 1 --branch "${DOCKERFILE_REF}" "${DOCKERFILE_REPO}" "${WORK_DIR}/sglang"

# ---------- Copy BCM driver into build context ----------
BUILD_CONTEXT="${WORK_DIR}/sglang/docker"
if [[ -f "${BCM_DRIVER}" ]]; then
    cp "${BCM_DRIVER}" "${BUILD_CONTEXT}/"
    echo "[build] BCM driver copied: ${BCM_DRIVER}"
elif [[ -f "/root/cache/${BCM_DRIVER}" ]]; then
    cp "/root/cache/${BCM_DRIVER}" "${BUILD_CONTEXT}/"
    echo "[build] BCM driver copied from /root/cache/"
else
    echo "ERROR: BCM driver not found: ${BCM_DRIVER}"
    echo "Place it in the current directory or /root/cache/"
    exit 1
fi

# ---------- Docker login ----------
if [[ -f /nfsdata/sa/.j9s/InferenceX/.env.local ]]; then
    source /nfsdata/sa/.j9s/InferenceX/.env.local
    echo "${DOCKER_HUB_PAT}" | docker login -u clustermax --password-stdin
fi

# ---------- Build ----------
echo "[build] Building ${IMAGE_TAG}"
echo "[build]   SGL_BRANCH=${SGL_BRANCH}"
echo "[build]   GPU_ARCH=${GPU_ARCH}"
echo "[build]   ENABLE_MORI=1, NIC_BACKEND=ibgda"

docker build \
    --build-arg SGL_BRANCH="${SGL_BRANCH}" \
    --build-arg GPU_ARCH="${GPU_ARCH}" \
    --build-arg ENABLE_MORI=1 \
    --build-arg NIC_BACKEND=ibgda \
    --build-arg BCM_DRIVER="${BCM_DRIVER}" \
    -t "${IMAGE_TAG}" \
    -f "${BUILD_CONTEXT}/rocm.Dockerfile" \
    "${BUILD_CONTEXT}/"

# ---------- Push ----------
echo "[build] Pushing ${IMAGE_TAG}"
docker push "${IMAGE_TAG}"

# ---------- Cleanup ----------
rm -rf "${WORK_DIR}"
echo "[build] Done: ${IMAGE_TAG}"
