#!/bin/bash
# Retag MI325X image as MI300X using GHCR registry API (no Docker daemon needed).
# Both platforms are gfx942 with Broadcom Thor 2 bnxt_re NICs on the same Vultr/CPE cluster.
#
# Usage: bash docker/build-sglang-bnxt-mi300x.sh
#
# Requires: GH_PAT env var or /nfsdata/sa/j9s/.env.local with GH_PAT set.
set -euo pipefail

SRC_REPO="jordannanos/sgl-mi325x-mori"
SRC_TAG="v0.5.9-bnxt-good"
DST_REPO="jordannanos/sgl-mi300x-mori"
DST_TAG="v0.5.9-bnxt"
ENV_FILE="/nfsdata/sa/j9s/.env.local"

if [ -z "${GH_PAT:-}" ] && [ -f "${ENV_FILE}" ]; then
  set -a; source "${ENV_FILE}"; set +a
fi
if [ -z "${GH_PAT:-}" ]; then
  echo "ERROR: GH_PAT not set"; exit 1
fi

get_token() {
  local scope="$1"
  curl -fsS -u "jordannanos:${GH_PAT}" \
    "https://ghcr.io/token?scope=${scope}&service=ghcr.io" \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['token'])"
}

echo "Getting auth tokens..."
PULL_TOKEN=$(get_token "repository:${SRC_REPO}:pull")
PUSH_TOKEN=$(get_token "repository:${DST_REPO}:push,pull+repository:${SRC_REPO}:pull")

echo "Fetching manifest from ${SRC_REPO}:${SRC_TAG}..."
MANIFEST=$(curl -fsS \
  -H "Authorization: Bearer ${PULL_TOKEN}" \
  -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
  "https://ghcr.io/v2/${SRC_REPO}/manifests/${SRC_TAG}")

DIGESTS=$(echo "${MANIFEST}" | python3 -c "
import json, sys
m = json.load(sys.stdin)
seen = set()
out = []
for d in [m['config']['digest']] + [l['digest'] for l in m['layers']]:
    if d not in seen:
        seen.add(d); out.append(d)
print('\n'.join(out))
")
TOTAL=$(echo "${DIGESTS}" | wc -l)
echo "Cross-mounting ${TOTAL} blobs from ${SRC_REPO} to ${DST_REPO}..."

MOUNTED=0; FAILED=0
while IFS= read -r digest; do
  STATUS=$(curl -fsS -o /dev/null -w "%{http_code}" -X POST \
    -H "Authorization: Bearer ${PUSH_TOKEN}" \
    "https://ghcr.io/v2/${DST_REPO}/blobs/uploads/?from=${SRC_REPO}&mount=${digest}")
  if [[ "${STATUS}" == "201" ]]; then ((MOUNTED++)) || true
  else echo "  WARNING: blob ${digest}: HTTP ${STATUS}"; ((FAILED++)) || true; fi
done <<< "${DIGESTS}"
echo "Blobs: ${MOUNTED} mounted, ${FAILED} failed"

echo "Pushing manifest to ${DST_REPO}:${DST_TAG}..."
HTTP_STATUS=$(curl -fsS -o /dev/null -w "%{http_code}" -X PUT \
  -H "Authorization: Bearer ${PUSH_TOKEN}" \
  -H "Content-Type: application/vnd.docker.distribution.manifest.v2+json" \
  --data-raw "${MANIFEST}" \
  "https://ghcr.io/v2/${DST_REPO}/manifests/${DST_TAG}")

if [[ "${HTTP_STATUS}" == "201" || "${HTTP_STATUS}" == "200" ]]; then
  echo "Done. Image available at: ghcr.io/${DST_REPO}:${DST_TAG}"
else
  echo "ERROR: manifest push returned HTTP ${HTTP_STATUS}"; exit 1
fi
