#!/usr/bin/env bash
set -euo pipefail

# Apply ROCm/ATOM PR #2147 on top of the pinned MiniMax image checkout.
#
# PR #2147 supersedes PR #2106 for the MiniMax-M3 EAGLE3 draft-KV layout work:
# both fix the draft/target KV pool disagreement that can OOM during prefix-cache
# warmup. The pinned image predates #2147's merge base, so we fast-forward the
# in-image tree to the current PR head rather than applying the PR diff alone.
#
# A small MiniMax EAGLE3 follow-up fix is applied after the forward diff because
# the upstream check compared kernel block sizes (16 vs 128) instead of the
# scheduler block size both sides actually index (128).

ATOM_ROOT="${ATOM_ROOT:-/app/ATOM}"
PINNED_ATOM_SHA="00760297ef69af7ab5d345af9c8fc6da00f5314d"
PR2147_HEAD_SHA="71ad8c69c5377738657907e24c5aff8672c0f004"
PATCH_URL="https://github.com/ROCm/ATOM/compare/${PINNED_ATOM_SHA}...${PR2147_HEAD_SHA}.diff"
PATCH_FILE="$(mktemp /tmp/atom-pr2147.XXXXXX.patch)"
MARKER_FILE="$ATOM_ROOT/.inferencex-pr2147-applied"
DRAFT_KV_FILE="$ATOM_ROOT/atom/spec_decode/draft_kv.py"

if [[ ! -d "$ATOM_ROOT" ]]; then
    echo "ERROR: ATOM source tree not found at $ATOM_ROOT" >&2
    exit 1
fi

trap 'rm -f "$PATCH_FILE"' EXIT

if [[ -f "$MARKER_FILE" ]]; then
    echo "ATOM PR #2147 already applied ($(cat "$MARKER_FILE"))"
    exit 0
fi

curl -fsSL "$PATCH_URL" -o "$PATCH_FILE"
git -C "$ATOM_ROOT" apply --check "$PATCH_FILE"
git -C "$ATOM_ROOT" apply "$PATCH_FILE"

python3 - "$DRAFT_KV_FILE" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
old = """        if pool.block_size != target.block_size:
            raise ValueError(
                f"the draft's blocks are {pool.block_size} tokens and the "
                f"target's {target.block_size}; a draft is indexed by the "
                "target's block tables, so it cannot yet block at anything else"
            )"""
new = """        pool_scheduler_block = pool.block_size * (
            runner.block_size // pool.block_size
        )
        target_scheduler_block = target.block_size * target.block_ratio
        if pool_scheduler_block != target_scheduler_block:
            raise ValueError(
                f"the draft indexes {pool_scheduler_block}-token scheduler "
                f"blocks but the target indexes {target_scheduler_block}-token "
                "scheduler blocks; a draft is indexed by the target's block "
                "tables, so the two must agree at scheduler granularity"
            )"""
if old not in text:
    if "pool_scheduler_block" in text:
        print("MiniMax EAGLE3 scheduler-block fix already present")
    else:
        raise SystemExit(
            "ERROR: expected draft_kv.py block-size guard not found after PR #2147 apply"
        )
else:
    path.write_text(text.replace(old, new, 1))
    print("Applied MiniMax EAGLE3 scheduler-block alignment fix to draft_kv.py")
PY

printf '%s\n' "$PR2147_HEAD_SHA" > "$MARKER_FILE"
echo "Applied ROCm/ATOM PR #2147 (${PINNED_ATOM_SHA} -> ${PR2147_HEAD_SHA}) to $ATOM_ROOT"
