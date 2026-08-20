#!/usr/bin/env bash
# Apply #51052 Kimi-K3 MoRIIO hybrid transfer into the engine container.
#
# This is deliberately a unified diff rather than a full-file overlay. An
# overlay replaces whole modules, so it also pins every unrelated symbol in
# those files to whichever vLLM it was captured from: a cb810-era snapshot
# silently downgraded vllm/v1/core/kv_cache_utils.py and the newer engine died
# with "cannot import name 'update_kv_cache_capacity'". The diff only touches
# what #51052 changes, and fails loudly when it no longer fits the image.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="${K3_MORIIO_PATCH:-$HERE/patches/k3_moriio_51052.patch}"
ROOT="$(python3 -c 'import importlib.util as u, os; print(os.path.dirname(os.path.dirname(u.find_spec("vllm").origin)))')"

if [[ -z "$ROOT" || ! -d "$ROOT/vllm" ]]; then
    echo "[k3-moriio] ERROR: could not resolve vLLM root (ROOT='$ROOT')" >&2
    exit 1
fi

MORIIO_DIR="$ROOT/vllm/distributed/kv_transfer/kv_connector/v1/moriio"
if grep -RqsE '_draft_only_layers|as_attn_mamba' "$MORIIO_DIR" \
   && grep -qs '_wait_all_supported' "$MORIIO_DIR/moriio_engine.py" \
   && python3 -c 'from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import MoRIIOConnector' 2>/dev/null; then
    echo "[k3-moriio] #51052 already applied (markers + import ok)"
    exit 0
fi

if [[ ! -f "$PATCH_FILE" ]]; then
    echo "[k3-moriio] ERROR: patch not found at $PATCH_FILE" >&2
    exit 1
fi

# Patches must be UTF-8 unified diffs over vllm/ only. Full-repo diffs that
# include examples/ or tests/ cannot apply inside the installed wheel layout.
if [[ "$(head -c 2 "$PATCH_FILE" | od -An -tx1 | tr -d ' ')" == "fffe" ]]; then
    echo "[k3-moriio] ERROR: patch is UTF-16; regenerate as UTF-8 LF" >&2
    exit 1
fi
if ! head -1 "$PATCH_FILE" | grep -q '^diff --git '; then
    echo "[k3-moriio] ERROR: patch does not look like a unified diff" >&2
    exit 1
fi
if grep -qE '^diff --git a/(examples|tests)/' "$PATCH_FILE"; then
    echo "[k3-moriio] ERROR: patch contains examples/ or tests/ paths" >&2
    echo "[k3-moriio] Regenerate with paths under vllm/ only (see patches/README.md)" >&2
    exit 1
fi

if grep -q $'\r' "$PATCH_FILE"; then
    tmp=$(mktemp)
    tr -d '\r' < "$PATCH_FILE" > "$tmp"
    PATCH_FILE="$tmp"
    trap 'rm -f "$tmp"' EXIT
fi

if (cd "$ROOT" && git apply -p1 "$PATCH_FILE" 2>/dev/null); then
    echo "[k3-moriio] applied #51052 with git apply"
elif patch -p1 -d "$ROOT" --forward --no-backup-if-mismatch < "$PATCH_FILE"; then
    echo "[k3-moriio] applied #51052 with patch"
else
    echo "[k3-moriio] ERROR: #51052 does not apply to this image." >&2
    echo "[k3-moriio] Regenerate patches/k3_moriio_51052.patch against it; do not" >&2
    echo "[k3-moriio] substitute a full-file overlay, which downgrades unrelated modules." >&2
    find "$ROOT" -name '*.rej' -newermt '-5 minutes' -print >&2 2>/dev/null || true
    exit 1
fi

python3 - <<'PY'
from vllm.distributed.kv_transfer.kv_connector.v1.moriio import moriio_common as c
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnector,
)
from vllm.v1.core import kv_cache_utils

assert hasattr(c, "as_attn_mamba"), "as_attn_mamba missing after patch"
assert MoRIIOConnector is not None
# The engine imports this from kv_cache_utils; a stale replacement of that
# module is exactly the failure mode this script now refuses to create.
assert hasattr(kv_cache_utils, "update_kv_cache_capacity"), (
    "kv_cache_utils lost update_kv_cache_capacity"
)
print("[k3-moriio] #51052 applied; IMPORT_OK")
PY
