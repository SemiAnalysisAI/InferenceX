#!/usr/bin/env bash
set -euo pipefail

# Canary-only node-local image staging. The source is served by the MI300X
# runner over the cluster-private network; every allocated node verifies the
# immutable digest before atomically replacing its own destination.

: "${KIMIK3_CANARY_SQUASH_URL:?KIMIK3_CANARY_SQUASH_URL must be set}"
: "${KIMIK3_CANARY_SQUASH_SHA256:?KIMIK3_CANARY_SQUASH_SHA256 must be set}"
: "${KIMIK3_SQUASH_FILE_OVERRIDE:?KIMIK3_SQUASH_FILE_OVERRIDE must be set}"

if [[ ! "$KIMIK3_CANARY_SQUASH_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "ERROR: KIMIK3_CANARY_SQUASH_SHA256 must be a lowercase sha256 digest" >&2
    exit 1
fi
if [[ "$KIMIK3_SQUASH_FILE_OVERRIDE" != /*.sqsh ]]; then
    echo "ERROR: KIMIK3_SQUASH_FILE_OVERRIDE must be an absolute .sqsh path" >&2
    exit 1
fi

mkdir -p "$(dirname "$KIMIK3_SQUASH_FILE_OVERRIDE")"
lock_file="${KIMIK3_SQUASH_FILE_OVERRIDE}.lock"
exec 9>"$lock_file"
if ! flock -w "${KIMIK3_IMAGE_LOCK_TIMEOUT_SECONDS:-3600}" 9; then
    echo "ERROR: timed out waiting for image lock $lock_file" >&2
    exit 1
fi

actual_sha256=""
if [[ -s "$KIMIK3_SQUASH_FILE_OVERRIDE" ]] &&
    unsquashfs -s "$KIMIK3_SQUASH_FILE_OVERRIDE" >/dev/null 2>&1; then
    actual_sha256=$(sha256sum "$KIMIK3_SQUASH_FILE_OVERRIDE" | awk '{print $1}')
fi
if [[ "$actual_sha256" == "$KIMIK3_CANARY_SQUASH_SHA256" ]]; then
    printf 'INFERENCEX_KIMIK3_IMAGE_STAGE hostname=%s action=reuse sha256=%s bytes=%s\n' \
        "$(hostname)" \
        "$actual_sha256" \
        "$(wc -c < "$KIMIK3_SQUASH_FILE_OVERRIDE" | tr -d '[:space:]')"
    exit 0
fi

temporary=$(mktemp "${KIMIK3_SQUASH_FILE_OVERRIDE}.tmp.XXXXXX")
remove_temporary() {
    rm -f "$temporary"
}
trap remove_temporary EXIT
trap 'remove_temporary; exit 130' INT
trap 'remove_temporary; exit 143' TERM
trap 'remove_temporary; exit 129' HUP

curl --fail --location --retry 3 --output "$temporary" \
    "$KIMIK3_CANARY_SQUASH_URL"
actual_sha256=$(sha256sum "$temporary" | awk '{print $1}')
if [[ "$actual_sha256" != "$KIMIK3_CANARY_SQUASH_SHA256" ]]; then
    echo "ERROR: downloaded image sha256 mismatch: expected=$KIMIK3_CANARY_SQUASH_SHA256 actual=$actual_sha256" >&2
    exit 1
fi
if ! unsquashfs -s "$temporary" >/dev/null 2>&1; then
    echo "ERROR: downloaded image is not a valid squashfs" >&2
    exit 1
fi

chmod 0644 "$temporary"
mv -f "$temporary" "$KIMIK3_SQUASH_FILE_OVERRIDE"
temporary=""
trap - EXIT INT TERM HUP

printf 'INFERENCEX_KIMIK3_IMAGE_STAGE hostname=%s action=download sha256=%s bytes=%s\n' \
    "$(hostname)" \
    "$actual_sha256" \
    "$(wc -c < "$KIMIK3_SQUASH_FILE_OVERRIDE" | tr -d '[:space:]')"
