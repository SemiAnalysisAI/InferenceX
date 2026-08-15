#!/usr/bin/env bash
# Apply #51052 Kimi-K3 MoRIIO hybrid transfer into the engine container.
# Prefer a full-file overlay (works when `patch` hunks drift on nightly images);
# fall back to the unified diff if no overlay is present.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OVERLAY="${K3_MORIIO_OVERLAY:-$HERE/patches/k3_moriio_51052_overlay.tar}"
PATCH_FILE="${K3_MORIIO_PATCH:-$HERE/patches/k3_moriio_51052.patch}"
ROOT="$(python3 -c 'import importlib.util as u, os; print(os.path.dirname(os.path.dirname(u.find_spec("vllm").origin)))')"

if [[ -z "$ROOT" || ! -d "$ROOT/vllm" ]]; then
    echo "[k3-moriio] ERROR: could not resolve vLLM root (ROOT='$ROOT')" >&2
    exit 1
fi
MORIIO_DIR="$ROOT/vllm/distributed/kv_transfer/kv_connector/v1/moriio"
if grep -RqsE '_draft_only_layers|as_attn_mamba' "$MORIIO_DIR" \
   && python3 -c 'from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import MoRIIOConnector' 2>/dev/null; then
    echo "[k3-moriio] #51052 already applied (markers + import ok)"
    exit 0
fi

if [[ -f "$OVERLAY" ]]; then
    echo "[k3-moriio] installing overlay $OVERLAY -> $ROOT"
    tar xf "$OVERLAY" -C "$ROOT"
    python3 - <<'PY'
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import MoRIIOConnector
from vllm.distributed.kv_transfer.kv_connector.v1.moriio import moriio_common as c
assert hasattr(c, "as_attn_mamba"), "as_attn_mamba missing after overlay"
print("[k3-moriio] applied #51052 via overlay; IMPORT_OK")
PY
    exit 0
fi

if [[ ! -f "$PATCH_FILE" ]]; then
    echo "[k3-moriio] ERROR: neither overlay nor patch present" >&2
    exit 1
fi

if grep -q $'\r' "$PATCH_FILE"; then
    tmp=$(mktemp)
    tr -d '\r' < "$PATCH_FILE" > "$tmp"
    PATCH_FILE="$tmp"
    trap 'rm -f "$tmp"' EXIT
fi

if (cd "$ROOT" && git apply -p1 "$PATCH_FILE"); then
    echo "[k3-moriio] applied #51052 with git apply"
elif patch -p1 -d "$ROOT" --forward --no-backup-if-mismatch < "$PATCH_FILE"; then
    echo "[k3-moriio] applied #51052 with patch"
else
    echo "[k3-moriio] ERROR: failed to apply #51052 patch" >&2
    exit 1
fi
