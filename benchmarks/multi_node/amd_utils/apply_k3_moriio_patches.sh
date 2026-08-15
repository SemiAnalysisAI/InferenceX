#!/usr/bin/env bash
# Apply the #51052 Kimi-K3 MoRIIO connector patch inside the engine container.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="${K3_MORIIO_PATCH:-$HERE/patches/k3_moriio_51052.patch}"
ROOT="$(python3 -c 'import importlib.util as u, os; print(os.path.dirname(os.path.dirname(u.find_spec("vllm").origin)))')"

if [[ -z "$ROOT" || ! -d "$ROOT/vllm" ]]; then
    echo "[k3-moriio] ERROR: could not resolve vLLM root (ROOT='$ROOT')" >&2
    exit 1
fi
if [[ ! -f "$PATCH_FILE" ]]; then
    echo "[k3-moriio] ERROR: patch file missing: $PATCH_FILE" >&2
    exit 1
fi

MORIIO_DIR="$ROOT/vllm/distributed/kv_transfer/kv_connector/v1/moriio"
if grep -RqsE '_draft_only_layers|as_attn_mamba' "$MORIIO_DIR"; then
    echo "[k3-moriio] #51052 already applied"
    exit 0
fi

if (cd "$ROOT" && git apply -p1 "$PATCH_FILE"); then
    echo "[k3-moriio] applied #51052 with git apply"
elif patch -p1 -d "$ROOT" --forward --no-backup-if-mismatch < "$PATCH_FILE"; then
    echo "[k3-moriio] applied #51052 with patch"
else
    echo "[k3-moriio] ERROR: failed to apply #51052 patch" >&2
    exit 1
fi
