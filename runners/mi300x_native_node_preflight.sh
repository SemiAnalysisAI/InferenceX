#!/usr/bin/env bash
set -euo pipefail

KIMIK3_MODEL_CACHE_ROOT="${KIMIK3_MODEL_CACHE_ROOT:-/raid/hf-hub-cache/models--moonshotai--Kimi-K3}"
KIMIK3_SQUASH_DIR="${KIMIK3_SQUASH_DIR:-/raid/hf-hub-cache/inferencex/squash}"
KIMIK3_IMAGE="${KIMIK3_IMAGE:-${IMAGE:?IMAGE must be set}}"

fail() {
    echo "ERROR: [$(hostname)] $*" >&2
    exit 1
}

TMP_SQUASH=""
remove_tmp_squash() {
    if [[ -n "$TMP_SQUASH" ]]; then
        rm -f "$TMP_SQUASH"
    fi
    return 0
}
trap remove_tmp_squash EXIT
trap 'remove_tmp_squash; exit 130' INT
trap 'remove_tmp_squash; exit 143' TERM

gpu_arch_lines=$(
    rocminfo 2>/dev/null |
        sed -n 's/^[[:space:]]*Name:[[:space:]]*\(gfx[0-9a-z]*\)[[:space:]]*$/\1/p' || true
)
gpu_count=$(printf '%s\n' "$gpu_arch_lines" | grep -c '^gfx' || true)
gfx942_count=$(printf '%s\n' "$gpu_arch_lines" | grep -c '^gfx942$' || true)
if [[ "$gpu_count" != "8" || "$gfx942_count" != "8" ]]; then
    fail "this node must expose exactly 8 gfx942 GPUs; found ${gfx942_count} gfx942 among ${gpu_count} GPU agents"
fi

refs_main="$KIMIK3_MODEL_CACHE_ROOT/refs/main"
if [[ ! -f "$refs_main" ]]; then
    fail "missing model revision pointer $refs_main; stage the snapshot on this node first"
fi
revision=$(tr -d '[:space:]' < "$refs_main")
if ! [[ "$revision" =~ ^[0-9a-f]{40}$ ]]; then
    fail "$refs_main must hold a 40-character revision, found '$revision'"
fi

snapshot_dir="$KIMIK3_MODEL_CACHE_ROOT/snapshots/$revision"
if [[ ! -d "$snapshot_dir" ]]; then
    fail "missing model snapshot directory $snapshot_dir"
fi
if [[ ! -f "$snapshot_dir/config.json" ]]; then
    fail "missing $snapshot_dir/config.json"
fi

weight_index="$snapshot_dir/model.safetensors.index.json"
if [[ ! -f "$weight_index" ]]; then
    fail "missing model.safetensors.index.json in $snapshot_dir"
fi
python3 - "$weight_index" "$snapshot_dir" <<'PY'
import json
import os
import sys

index_path, snapshot_dir = sys.argv[1], sys.argv[2]
with open(index_path) as handle:
    weight_map = (json.load(handle) or {}).get("weight_map") or {}
if not weight_map:
    sys.exit(f"ERROR: {index_path} declares no weight_map entries")

incomplete = sorted(
    shard
    for shard in set(weight_map.values())
    if not os.path.isfile(os.path.join(snapshot_dir, shard))
    or os.path.getsize(os.path.join(snapshot_dir, shard)) == 0
)
if incomplete:
    sys.exit(
        f"ERROR: missing weight shard(s) in {snapshot_dir}: " + ", ".join(incomplete)
    )
PY

mkdir -p "$KIMIK3_SQUASH_DIR"
squash_file="$KIMIK3_SQUASH_DIR/$(printf '%s' "$KIMIK3_IMAGE" | sed 's/[\/:@#]/_/g').sqsh"

export ENROOT_CACHE_PATH="$KIMIK3_SQUASH_DIR/.enroot-cache"
export ENROOT_TEMP_PATH="$KIMIK3_SQUASH_DIR/.enroot-temp"
mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_TEMP_PATH"

lock_file="${squash_file}.lock"
exec 9>"$lock_file"
if ! flock -w "${KIMIK3_IMAGE_LOCK_TIMEOUT_SECONDS:-3600}" 9; then
    fail "timed out waiting for the image lock $lock_file"
fi

if [[ -s "$squash_file" ]] && unsquashfs -s "$squash_file" >/dev/null 2>&1; then
    echo "[$(hostname)] reusing validated image $squash_file"
else
    rm -f "$squash_file"
    TMP_SQUASH=$(mktemp "${squash_file}.XXXXXX")
    rm -f "$TMP_SQUASH"
    echo "[$(hostname)] importing $KIMIK3_IMAGE into $squash_file"
    enroot import -o "$TMP_SQUASH" "docker://$KIMIK3_IMAGE"
    if ! unsquashfs -s "$TMP_SQUASH" >/dev/null 2>&1; then
        fail "imported image $KIMIK3_IMAGE failed unsquashfs validation"
    fi
    mv "$TMP_SQUASH" "$squash_file"
    TMP_SQUASH=""
fi

squash_size_bytes=$(wc -c < "$squash_file" | tr -d '[:space:]')
if [[ "$squash_size_bytes" -le 0 ]]; then
    fail "validated image $squash_file is empty"
fi

printf 'INFERENCEX_KIMIK3_PREFLIGHT hostname=%s revision=%s gpu_count=%s gpu_arch=gfx942 squash_size_bytes=%s\n' \
    "$(hostname)" "$revision" "$gpu_count" "$squash_size_bytes"
