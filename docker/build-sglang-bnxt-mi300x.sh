#!/bin/bash
# Publish the MI300X/MI325X CDNA3 SGLang+MoRI image to Docker Hub (semianalysiswork).
#
# MI300X (gfx942) and MI325X (gfx942) share the same CDNA3 architecture, so a
# single image covers both platforms.  The source image was built on MI325X hardware
# and is retagged here for the official semianalysiswork registry.
#
# Usage: bash docker/build-sglang-bnxt-mi300x.sh
#
# Requires:
#   GH_PAT          - GitHub PAT with packages:read scope (for GHCR pull)
#   DOCKER_HUB_PAT  - Docker Hub PAT for the clustermax account
#   crane           - installed at /tmp/crane or on PATH
#                     (download: https://github.com/google/go-containerregistry/releases)
#
# Optional env:
#   CRANE_BIN       - override crane binary path (default: first of /tmp/crane, crane)
set -euo pipefail

SRC_IMAGE="ghcr.io/jordannanos/sgl-mi325x-mori:v0.5.9-bnxt-good"
DST_IMAGE="docker.io/semianalysiswork/sgl-bnxt-cdna3:v0.5.9-bnxt"

ENV_FILE="/nfsdata/sa/j9s/.env.local"
if [ -z "${GH_PAT:-}" ] && [ -f "${ENV_FILE}" ]; then
  set -a; source "${ENV_FILE}"; set +a
fi
if [ -z "${GH_PAT:-}" ]; then
  echo "ERROR: GH_PAT not set"; exit 1
fi
if [ -z "${DOCKER_HUB_PAT:-}" ]; then
  echo "ERROR: DOCKER_HUB_PAT not set"; exit 1
fi

# Locate crane binary
CRANE_BIN="${CRANE_BIN:-}"
if [ -z "${CRANE_BIN}" ]; then
  for candidate in /tmp/crane crane; do
    if command -v "$candidate" &>/dev/null 2>&1 || [ -x "$candidate" ]; then
      CRANE_BIN="$candidate"; break
    fi
  done
fi
if [ -z "${CRANE_BIN}" ] || ! ( command -v "${CRANE_BIN}" &>/dev/null || [ -x "${CRANE_BIN}" ] ); then
  echo "ERROR: crane not found. Install from https://github.com/google/go-containerregistry/releases"
  exit 1
fi

echo "crane: ${CRANE_BIN} ($(${CRANE_BIN} version))"
echo "Source: ${SRC_IMAGE}"
echo "Dest:   ${DST_IMAGE}"
echo

# Authenticate
echo "${GH_PAT}" | ${CRANE_BIN} auth login ghcr.io -u jordannanos --password-stdin
echo "${DOCKER_HUB_PAT}" | ${CRANE_BIN} auth login index.docker.io -u clustermax --password-stdin

echo "Copying image (this may take a few minutes)..."
${CRANE_BIN} copy "${SRC_IMAGE}" "${DST_IMAGE}"

echo
echo "Done. Image available at: ${DST_IMAGE}"
