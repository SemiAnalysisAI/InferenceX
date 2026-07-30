#!/usr/bin/env bash
set -euo pipefail

# Per-node readiness gate for the native two-node Kimi-K3 MI300X path.
#
# Runs once on every allocated node, because /raid is node-local: a model cached
# or an image imported on one node is invisible to the other. Verifies the GPU
# topology and the complete pinned weight snapshot, then imports and validates
# this node's own container image. It never downloads the ~1.5 TB checkpoint --
# absent weights are a hard failure, not something to fix inside a timed job.
#
# Emits exactly one machine-readable INFERENCEX_KIMIK3_PREFLIGHT record that the
# launcher cross-checks across nodes. The trace goes to stderr, so it never
# contaminates that record.
#
# Operator runbook: docs/kimik3-mi300x-native-multinode.md

set -x

KIMIK3_MODEL_CACHE_ROOT="${KIMIK3_MODEL_CACHE_ROOT:-/raid/hf-hub-cache/models--moonshotai--Kimi-K3}"
KIMIK3_SQUASH_DIR="${KIMIK3_SQUASH_DIR:-/raid/hf-hub-cache/inferencex/squash}"
KIMIK3_IMAGE="${KIMIK3_IMAGE:-${IMAGE:?IMAGE must be set}}"

fail() {
    echo "ERROR: [$(hostname)] $*" >&2
    exit 1
}

# Only ever the in-progress import; a previously validated final squash is never
# removed, so a failed run cannot destroy a good image other jobs are using.
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
trap 'remove_tmp_squash; exit 129' HUP

# 1. GPU topology. Each GPU agent contributes one bare "Name: gfx<arch>" line;
#    the trailing ISA block spells the arch differently and must not be counted.
gpu_arch_lines=$(
    rocminfo 2>/dev/null |
        sed -n 's/^[[:space:]]*Name:[[:space:]]*\(gfx[0-9a-z]*\)[[:space:]]*$/\1/p' || true
)
gpu_count=$(printf '%s\n' "$gpu_arch_lines" | grep -c '^gfx' || true)
gfx942_count=$(printf '%s\n' "$gpu_arch_lines" | grep -c '^gfx942$' || true)
if [[ "$gpu_count" != "8" || "$gfx942_count" != "8" ]]; then
    fail "this node must expose exactly 8 gfx942 GPUs; found ${gfx942_count} gfx942 among ${gpu_count} GPU agents"
fi

# 2. Pinned revision.
refs_main="$KIMIK3_MODEL_CACHE_ROOT/refs/main"
if [[ ! -f "$refs_main" ]]; then
    fail "missing model revision pointer $refs_main; stage the snapshot on this node first"
fi
revision=$(tr -d '[:space:]' < "$refs_main")
if ! [[ "$revision" =~ ^[0-9a-f]{40}$ ]]; then
    fail "$refs_main must hold a 40-character revision, found '$revision'"
fi

# 3. Snapshot directory.
snapshot_dir="$KIMIK3_MODEL_CACHE_ROOT/snapshots/$revision"
if [[ ! -d "$snapshot_dir" ]]; then
    fail "missing model snapshot directory $snapshot_dir"
fi
if [[ ! -f "$snapshot_dir/config.json" ]]; then
    fail "missing $snapshot_dir/config.json"
fi

# 4. Every shard the weight index names must be present and non-empty. A
#    partially synced snapshot otherwise fails hours later at model load.
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

# 5. Everything this script creates lives below the writable node-local squash
#    tree. /home and /nvme_home are the same nearly-full NFS export, and
#    gharunner cannot create /raid/squash.
mkdir -p "$KIMIK3_SQUASH_DIR"
if [[ -n "${KIMIK3_SQUASH_FILE_OVERRIDE:-}" ]]; then
    if [[ "$KIMIK3_SQUASH_FILE_OVERRIDE" != /*.sqsh ]]; then
        fail "KIMIK3_SQUASH_FILE_OVERRIDE must be an absolute .sqsh path"
    fi
    if [[ ! "${KIMIK3_CANARY_SQUASH_SHA256:-}" =~ ^[0-9a-f]{64}$ ]]; then
        fail "a squash override requires a lowercase KIMIK3_CANARY_SQUASH_SHA256"
    fi
    squash_file="$KIMIK3_SQUASH_FILE_OVERRIDE"
else
    squash_file="$KIMIK3_SQUASH_DIR/$(printf '%s' "$KIMIK3_IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
fi

# 6. Keep Enroot's scratch on the same node-local filesystem, never under $HOME.
export ENROOT_CACHE_PATH="$KIMIK3_SQUASH_DIR/.enroot-cache"
export ENROOT_TEMP_PATH="$KIMIK3_SQUASH_DIR/.enroot-temp"
mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_TEMP_PATH"

# 7. Serialize concurrent importers of the same node-local image.
lock_file="${squash_file}.lock"
exec 9>"$lock_file"
if ! flock -w "${KIMIK3_IMAGE_LOCK_TIMEOUT_SECONDS:-3600}" 9; then
    fail "timed out waiting for the image lock $lock_file"
fi

# 8/9. Reuse a valid image; otherwise import into a sibling temporary file,
#      validate it, and only then swap it in atomically.
if [[ -n "${KIMIK3_SQUASH_FILE_OVERRIDE:-}" ]]; then
    if [[ ! -s "$squash_file" ]] ||
        ! unsquashfs -s "$squash_file" >/dev/null 2>&1; then
        fail "staged squash override is missing or invalid: $squash_file"
    fi
    squash_sha256=$(sha256sum "$squash_file" | awk '{print $1}')
    if [[ "$squash_sha256" != "$KIMIK3_CANARY_SQUASH_SHA256" ]]; then
        fail "staged squash override sha256 mismatch: expected=$KIMIK3_CANARY_SQUASH_SHA256 actual=$squash_sha256"
    fi
    echo "[$(hostname)] reusing verified canary image $squash_file"
elif [[ -s "$squash_file" ]] && unsquashfs -s "$squash_file" >/dev/null 2>&1; then
    echo "[$(hostname)] reusing validated image $squash_file"
    squash_sha256=unverified
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
    squash_sha256=unverified
fi

squash_size_bytes=$(wc -c < "$squash_file" | tr -d '[:space:]')
if [[ "$squash_size_bytes" -le 0 ]]; then
    fail "validated image $squash_file is empty"
fi

# 10. The launcher parses only this line.
printf 'INFERENCEX_KIMIK3_PREFLIGHT hostname=%s revision=%s gpu_count=%s gpu_arch=gfx942 squash_size_bytes=%s squash_sha256=%s\n' \
    "$(hostname)" "$revision" "$gpu_count" "$squash_size_bytes" "$squash_sha256"
