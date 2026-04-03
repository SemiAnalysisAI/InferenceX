#!/usr/bin/env bash
# Build the MiniMax M2.5 PD Disagg Docker image.
#
# Usage:
#   cd <InferenceX repo root>
#   bash docker/minimax-m25-disagg/build.sh [tag] [base_image]
#
# Examples:
#   bash docker/minimax-m25-disagg/build.sh                          # default tag + base
#   bash docker/minimax-m25-disagg/build.sh my-tag:v1                # custom tag
#   bash docker/minimax-m25-disagg/build.sh latest vllm/vllm-openai-rocm:v0.19.0  # custom base
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TAG="${1:-minimax-m25-disagg:latest}"
BASE_IMAGE="${2:-vllm/vllm-openai-rocm:v0.18.0}"

echo "Building MiniMax M2.5 Disagg image..."
echo "  Tag:        $TAG"
echo "  Base image: $BASE_IMAGE"
echo "  Context:    $REPO_ROOT"

docker build \
    -t "$TAG" \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    -f "$REPO_ROOT/docker/minimax-m25-disagg/Dockerfile" \
    "$REPO_ROOT"

echo ""
echo "Done. Image: $TAG"
echo "To push: docker tag $TAG <registry>/$TAG && docker push <registry>/$TAG"
