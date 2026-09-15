#!/usr/bin/env bash
# Extract the EPP, pd-sidecar, and Envoy binaries into $LLMD_BIN_DIR so a stock
# vllm/vllm-openai image can mount them at runtime (see job.slurm) instead of
# rebuilding a combined image on every vLLM bump. Needs docker (arm64, or x86
# with --platform emulation); re-run only when a source image in binaries.env
# changes. Idempotent.
# Usage: [LLMD_BIN_DIR=dir] [LLMD_BIN_PLATFORM=linux/amd64] ./extract-binaries.sh

set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/binaries.env"

echo "Extracting llm-d binaries -> $LLMD_BIN_DIR (platform $LLMD_BIN_PLATFORM)"
mkdir -p "$LLMD_BIN_DIR"

# (source image, path inside image, output filename)
extract() {
    local image="$1" src="$2" out="$3"
    echo "  $out  <-  $image:$src"
    local cid
    cid="$(docker create --platform "$LLMD_BIN_PLATFORM" "$image")"
    # shellcheck disable=SC2064
    trap "docker rm -f '$cid' >/dev/null 2>&1 || true" RETURN
    docker cp "$cid:$src" "$LLMD_BIN_DIR/$out"
    chmod +x "$LLMD_BIN_DIR/$out"
}

extract "$EPP_FROM_IMAGE"        "$EPP_BIN_PATH"             epp
extract "$ROUTING_SIDECAR_IMAGE" "$ROUTING_SIDECAR_BIN_PATH" pd-sidecar
extract "$ENVOY_FROM_IMAGE"      "$ENVOY_BIN_PATH"          envoy

echo "Done. Contents of $LLMD_BIN_DIR:"
ls -la "$LLMD_BIN_DIR"

# epp/pd-sidecar are static Go binaries; envoy is dynamically linked C++.
echo
echo "NOTE: verify 'ldd $LLMD_BIN_DIR/envoy' resolves cleanly inside the"
echo "      target vLLM image before relying on the mounted-binary path."
