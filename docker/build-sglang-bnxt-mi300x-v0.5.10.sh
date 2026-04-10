#!/bin/bash
# Build and publish the MI300X SGLang v0.5.10 + MoRI + Broadcom IBGDA image.
#
# This image extends the official lmsysorg/sglang:v0.5.10-rocm720-mi30x with:
#   - MoRI (Memory over RDMA Infrastructure) disaggregation backend from AMD
#   - Broadcom bnxt_re userspace RDMA libraries for Thor 2 NICs (IBGDA)
#   - Transformers with GLM-5 (glm_moe_dsa) model type support
#
# MoRI and bnxt components are copied from the v0.5.9-bnxt donor image, which
# was built from the jordannanos/sglang fork with MORI + bnxt patches compiled
# for gfx942 (CDNA3). These compiled .so libraries are architecture-compatible
# across SGLang versions.
#
# Usage:
#   bash docker/build-sglang-bnxt-mi300x-v0.5.10.sh
#
#   Or via SLURM:
#   sbatch docker/build-sglang-bnxt-mi300x.sbatch  # after updating script ref
#
# Requires:
#   DOCKER_HUB_PAT  - Docker Hub PAT for the clustermax account
#   Docker daemon accessible (run on compute node or with sudo)
#
# The build must run on a machine that can pull both:
#   - docker.io/semianalysiswork/sgl-bnxt-cdna3:v0.5.9-bnxt (donor)
#   - lmsysorg/sglang:v0.5.10-rocm720-mi30x (base)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCKERFILE="${SCRIPT_DIR}/mi300x-v0.5.10.Dockerfile"
IMAGE_TAG="docker.io/semianalysiswork/sgl-bnxt-cdna3:v0.5.10-bnxt"

ENV_FILE="/nfsdata/sa/j9s/.env.local"
if [ -z "${DOCKER_HUB_PAT:-}" ] && [ -f "${ENV_FILE}" ]; then
  set -a; source "${ENV_FILE}"; set +a
fi
if [ -z "${DOCKER_HUB_PAT:-}" ]; then
  echo "ERROR: DOCKER_HUB_PAT not set"; exit 1
fi

echo "Building: ${IMAGE_TAG}"
echo "Dockerfile: ${DOCKERFILE}"
echo

docker build -f "${DOCKERFILE}" -t "${IMAGE_TAG}" "${SCRIPT_DIR}"

echo
echo "=== Verifying image ==="
docker run --rm "${IMAGE_TAG}" bash -c '
pip show sglang transformers 2>/dev/null | grep -E "Name:|Version:"
echo ---
python3 -c "from mori import MoRIApplication; print(\"MoRI: OK\")" 2>/dev/null || echo "MoRI: FAIL"
python3 -c "from transformers import AutoConfig; c = AutoConfig.from_pretrained(\"zai-org/GLM-5-FP8\", trust_remote_code=True); print(\"GLM-5:\", c.model_type)" 2>/dev/null || echo "GLM-5: FAIL"
'

echo
echo "=== Pushing to Docker Hub ==="
echo "${DOCKER_HUB_PAT}" | docker login -u clustermax --password-stdin
docker push "${IMAGE_TAG}"

echo
echo "Done. Image available at: ${IMAGE_TAG}"
