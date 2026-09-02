#!/usr/bin/env bash
set -euo pipefail

# Apply ROCm/ATOM PR #2106 to the pinned image before serving.
# The patch fixes the MiniMax-M3 EAGLE3 draft KV layout and block-table
# freshness issues that can cause prefix-cache warmup OOMs or stale-KV reads.

ATOM_ROOT="${ATOM_ROOT:-/app/ATOM}"
# Pin the PR's current base/head range so a later PR update cannot silently
# change an already reviewed benchmark run.
PATCH_URL="https://github.com/ROCm/ATOM/compare/86476f716e9887a7cb2e423deb88712737f702f5...c63ab67e58d39f1c887d2b6af13ac0b9034a1a1a.diff"
PATCH_FILE="$(mktemp /tmp/atom-pr2106.XXXXXX.patch)"
trap 'rm -f "$PATCH_FILE"' EXIT
PATCH_EXCLUDES=(
    --exclude=atom/model_ops/attentions/backends.py
    --exclude=atom/spec_decode/eagle_proposer.py
)

if [[ ! -d "$ATOM_ROOT" ]]; then
    echo "ERROR: ATOM source tree not found at $ATOM_ROOT" >&2
    exit 1
fi

curl -fsSL "$PATCH_URL" -o "$PATCH_FILE"

if git -C "$ATOM_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if git -C "$ATOM_ROOT" apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
        echo "ATOM PR #2106 already applied"
        exit 0
    fi

    # These files were structurally rewritten after the pinned image's ATOM
    # commit. Their freshness guard is orthogonal to the draft-KV OOM fix and
    # is intentionally omitted until it can be ported against that revision.
    git -C "$ATOM_ROOT" apply --check "${PATCH_EXCLUDES[@]}" "$PATCH_FILE"
    git -C "$ATOM_ROOT" apply "${PATCH_EXCLUDES[@]}" "$PATCH_FILE"
else
    if [[ -f "$ATOM_ROOT/atom/spec_decode/draft_kv_layout.py" ]]; then
        echo "ATOM PR #2106 already applied"
        exit 0
    fi
    if patch --dry-run -p1 -d "$ATOM_ROOT" < "$PATCH_FILE" >/dev/null 2>&1; then
        patch -p1 -d "$ATOM_ROOT" < "$PATCH_FILE"
    else
        echo "ERROR: ATOM PR #2106 does not apply cleanly to $ATOM_ROOT" >&2
        exit 1
    fi
fi
echo "Applied ROCm/ATOM PR #2106 to $ATOM_ROOT"
