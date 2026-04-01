#!/usr/bin/env bash
# cache_model_locally.sh — Pre-stage model weights from shared storage to local NVMe.
#
# Syncs a model directory from NFS/shared storage to fast local NVMe before
# the inference server starts, using rclone for high-parallelism transfers.
#
# Usage:
#   source utils/cache_model_locally.sh
#   cache_model_locally "/nfs/hub/models--org--repo" "/local-nvme/hub/models--org--repo"
#
# Or as a standalone script:
#   bash utils/cache_model_locally.sh /nfs/hub/models--org--repo /local-nvme/hub/models--org--repo
#
# Features:
#   - Uses rclone sync with 32 parallel transfers for maximum throughput
#   - Preserves HuggingFace cache symlink structure (--links)
#   - Idempotent: rclone skips files already present and identical
#   - Works with both HF hub cache layout and flat model directories
#
# Environment variables:
#   CACHE_TRANSFERS  — number of parallel rclone transfers (default: 32)
#   CACHE_CHECKERS   — number of parallel rclone checkers (default: 32)
#   CACHE_DRY_RUN    — set to 1 to print what would be synced without copying

set -euo pipefail

CACHE_TRANSFERS="${CACHE_TRANSFERS:-32}"
CACHE_CHECKERS="${CACHE_CHECKERS:-32}"
CACHE_DRY_RUN="${CACHE_DRY_RUN:-0}"

cache_model_locally() {
    local src="${1:?Usage: cache_model_locally <source_path> <dest_path>}"
    local dst="${2:?Usage: cache_model_locally <source_path> <dest_path>}"

    if [[ ! -d "$src" ]]; then
        echo "[cache] ERROR: Source path does not exist: $src" >&2
        return 1
    fi

    echo "[cache] Syncing model to local storage..."
    echo "[cache]   Source: $src"
    echo "[cache]   Dest:   $dst"
    echo "[cache]   Transfers: $CACHE_TRANSFERS, Checkers: $CACHE_CHECKERS"

    mkdir -p "$dst"

    local start_time
    start_time=$(date +%s)

    local rclone_opts=(--transfers "$CACHE_TRANSFERS" --checkers "$CACHE_CHECKERS" --links --progress)
    if [[ "$CACHE_DRY_RUN" -eq 1 ]]; then
        rclone_opts+=(--dry-run)
    fi

    rclone sync "$src/" "$dst/" "${rclone_opts[@]}"

    local elapsed=$(( $(date +%s) - start_time ))
    local size
    size=$(du -sh "$dst" 2>/dev/null | cut -f1)

    echo "[cache] Done in ${elapsed}s — $size cached at $dst"
    echo "$dst"
    return 0
}

# If run as a standalone script (not sourced), execute with args
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if [[ $# -lt 2 ]]; then
        echo "Usage: $0 <source_model_path> <dest_model_path>" >&2
        echo "  Env: CACHE_TRANSFERS=$CACHE_TRANSFERS CACHE_CHECKERS=$CACHE_CHECKERS" >&2
        exit 1
    fi
    cache_model_locally "$1" "$2"
fi
