#!/usr/bin/env bash
#
# Extract the EPP, pd-sidecar, and Envoy binaries from their source
# images into $LLMD_BIN_DIR, so a STOCK vllm/vllm-openai image can be run
# with these mounted in at runtime (see job.slurm) instead of rebuilding
# a combined image on every vLLM version bump.
#
# Run this ONCE on a host with docker (arm64, or x86 with --platform
# emulation), and re-run only when a binary's source image changes in
# benchmarks/llm-d/binaries.env. Idempotent: overwrites in place.
#
# Usage:
#   BINARIES_ENV_FILE=binaries.env ./extract-binaries.sh
#   BINARIES_ENV_FILE=binaries.env LLMD_BIN_DIR=/some/dir ./extract-binaries.sh
#   BINARIES_ENV_FILE=binaries.env LLMD_BIN_PLATFORM=linux/amd64 ./extract-binaries.sh
#   BINARIES_ENV_FILE=binaries-b200-v0.10.0.env ./extract-binaries.sh
#     # pull a different set of image pins (e.g. a router-version bump
#     # scoped to one cluster) without touching the shared binaries.env

set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${BINARIES_ENV_FILE:?Set binaries.env or a cluster-specific pins file}"
# Resolve a bare filename against this script's directory, so
# BINARIES_ENV_FILE=foo.env works regardless of the caller's cwd.
[[ "$BINARIES_ENV_FILE" != /* ]] && BINARIES_ENV_FILE="$HERE/$BINARIES_ENV_FILE"
# shellcheck source=/dev/null
source "$BINARIES_ENV_FILE"

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

# Linking sanity reminder: epp/pd-sidecar are Go (static); envoy is a
# dynamically-linked C++ binary. Verify inside the stock vLLM container
# that it resolves:  ldd /usr/local/bin/envoy  (no "not found" lines).
echo
echo "NOTE: verify 'ldd $LLMD_BIN_DIR/envoy' resolves cleanly inside the"
echo "      target vLLM image before relying on the mounted-binary path."
