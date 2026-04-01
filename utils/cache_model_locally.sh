#!/usr/bin/env bash
# cache_model_locally.sh — Pre-stage model weights from shared storage to local NVMe.
#
# Syncs a model directory from NFS/shared storage to fast local NVMe before
# the inference server starts, dramatically reducing model load time.
#
# Usage:
#   source utils/cache_model_locally.sh
#   cache_model_locally "/nfs/models/deepseek-r1" "/local-nvme/models/deepseek-r1"
#
# Or as a standalone script:
#   bash utils/cache_model_locally.sh /nfs/models/deepseek-r1 /local-nvme/models/deepseek-r1
#
# Features:
#   - Idempotent: skips files already present on the target
#   - Preserves HuggingFace cache symlink structure
#   - Concurrent execution safe (multiple nodes can cache simultaneously)
#   - Configurable timeout to prevent NFS hangs
#   - Works with both HF hub cache layout and flat model directories
#
# Environment variables:
#   CACHE_PARALLEL_JOBS  — number of parallel rsync jobs for large blobs (default: 4)
#   CACHE_TIMEOUT        — per-file timeout in seconds (default: 600)
#   CACHE_DRY_RUN        — set to 1 to print what would be synced without copying

set -euo pipefail

CACHE_PARALLEL_JOBS="${CACHE_PARALLEL_JOBS:-4}"
CACHE_TIMEOUT="${CACHE_TIMEOUT:-600}"
CACHE_DRY_RUN="${CACHE_DRY_RUN:-0}"

cache_model_locally() {
    local src="${1:?Usage: cache_model_locally <source_path> <dest_path>}"
    local dst="${2:?Usage: cache_model_locally <source_path> <dest_path>}"

    if [[ ! -d "$src" ]]; then
        echo "[cache] ERROR: Source path does not exist: $src" >&2
        return 1
    fi

    # Quick check: if dest has the same number of regular files, skip entirely
    local src_count dst_count
    src_count=$(find "$src" -type f 2>/dev/null | wc -l)
    dst_count=$(find "$dst" -type f 2>/dev/null | wc -l)

    if [[ "$src_count" -eq "$dst_count" ]] && [[ "$dst_count" -gt 0 ]]; then
        echo "[cache] Already cached: $dst ($dst_count files)"
        echo "$dst"
        return 0
    fi

    echo "[cache] Syncing model to local storage..."
    echo "[cache]   Source: $src"
    echo "[cache]   Dest:   $dst"
    echo "[cache]   Parallel jobs: $CACHE_PARALLEL_JOBS"

    mkdir -p "$dst"

    local rsync_opts=(-a --whole-file --ignore-existing --info=name)
    if [[ "$CACHE_DRY_RUN" -eq 1 ]]; then
        rsync_opts+=(--dry-run)
    fi

    local start_time
    start_time=$(date +%s)

    # Check if this is a HuggingFace hub cache directory (has blobs/ subdir)
    if [[ -d "$src/blobs" ]]; then
        echo "[cache] Detected HuggingFace hub cache layout"

        # Step 1: Parallel-sync the large blob files (the actual model weights)
        mkdir -p "$dst/blobs"
        find "$src/blobs" -type f -printf '%f\n' | \
            xargs -P "$CACHE_PARALLEL_JOBS" -I{} \
            timeout "$CACHE_TIMEOUT" rsync "${rsync_opts[@]}" "$src/blobs/{}" "$dst/blobs/{}"

        # Step 2: Sync everything else (symlinks in snapshots/, refs/, etc.) — fast
        rsync "${rsync_opts[@]}" --exclude='blobs/' "$src/" "$dst/"
    else
        # Flat model directory: parallel-sync large files, then the rest
        echo "[cache] Detected flat model directory"

        # Sync large files (>100MB) in parallel
        find "$src" -type f -size +100M -printf '%P\n' | \
            xargs -P "$CACHE_PARALLEL_JOBS" -I{} bash -c \
            'mkdir -p "$(dirname "'"$dst"'/{}")"; timeout '"$CACHE_TIMEOUT"' rsync '"$(printf '%q ' "${rsync_opts[@]}")"' "'"$src"'/{}" "'"$dst"'/{}"'

        # Sync remaining small files and symlinks
        rsync "${rsync_opts[@]}" "$src/" "$dst/"
    fi

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
        echo "  Env: CACHE_PARALLEL_JOBS=$CACHE_PARALLEL_JOBS CACHE_TIMEOUT=$CACHE_TIMEOUT" >&2
        exit 1
    fi
    cache_model_locally "$1" "$2"
fi
