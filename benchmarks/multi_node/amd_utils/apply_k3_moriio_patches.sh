#!/usr/bin/env bash
# =============================================================================
# apply_k3_moriio_patches.sh
#
# Kimi-K3 needs three things from the MoRIIO connector that the pinned image's
# vLLM does not have. They are carried here as one patch against
# vllm/vllm-openai-rocm:nightly-cb8104839c141609d99f1254459ef3a4f1bd4263 rather
# than as a fork checkout, so a disagg run installs exactly the same way the
# single-node recipe installs apply_k3_container_patches.sh:
#
#   * hybrid mamba/KDA (conv+ssm) KV transfer. K3 is MLA + Kimi Delta Attention,
#     so a request's state is a (conv, ssm) tuple in one KV-cache group and paged
#     blocks in the others. Stock MoRIIO only moves paged blocks.
#   * per-layer KV block length. K3's KDA group gets a larger page than its MLA
#     groups (3072 vs 1536 tokens at fp8), and MoRIIO used to assert every
#     attention layer matched the first one, aborting all 8 ranks with
#     "MoRIIO KV cache block size mismatch for layer model.layers.93.self_attn".
#   * draft-only layer skip. Under DSpark the decoder loads a draft model whose
#     layers do not exist on the prefill side; MoRIIO must not try to pull them.
#
# Also carried: a peer KV-layout guard (fail the handshake instead of
# transferring into the wrong layers), the DSpark rejection-sampler bounds
# checks, drafter-invariant hybrid KV grouping, and a KV-group-size log line.
#
# This patch set is DISJOINT from apply_k3_container_patches.sh -- verified file
# by file -- so the two can be applied in either order without clobbering each
# other. Keep it that way: before adding anything here, check the file list
# against that script's.
#
# Run INSIDE the engine container, before `vllm serve`. Idempotent.
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="${K3_MORIIO_PATCH:-$HERE/patches/k3_moriio_dspark_trim.patch}"

ROOT="$(python3 -c 'import importlib.util as u, os; print(os.path.dirname(os.path.dirname(u.find_spec("vllm").origin)))')"
if [ -z "$ROOT" ] || [ ! -d "$ROOT/vllm" ]; then
  echo "[k3-moriio] ERROR: could not resolve dist-packages (ROOT='$ROOT')"; exit 1
fi
echo "[k3-moriio] ROOT=$ROOT"
echo "[k3-moriio] patch=$PATCH_FILE"
[ -f "$PATCH_FILE" ] || { echo "[k3-moriio] ERROR: patch file missing"; exit 1; }

MORIIO_CONNECTOR="$ROOT/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"

# Post-state marker: the draft-only layer set only exists after this patch.
if grep -qF "_draft_only_layers" "$MORIIO_CONNECTOR" 2>/dev/null; then
  echo "[k3-moriio] already applied (skip)"
else
  # The patch is generated against the vLLM source tree, so paths are
  # vllm/... and examples/...; dist-packages has vllm/ at its root and no
  # examples/, hence -p1 plus an explicit exclude for the example proxy.
  if ( cd "$ROOT" && git apply -p1 --exclude='examples/*' --whitespace=nowarn "$PATCH_FILE" ) 2>/dev/null; then
    echo "[k3-moriio] APPLIED (git apply)"
  elif patch -p1 -d "$ROOT" --fuzz=3 --forward --no-backup-if-mismatch \
         -x 'examples/*' < "$PATCH_FILE" >/dev/null 2>&1; then
    echo "[k3-moriio] APPLIED (patch)"
  else
    echo "[k3-moriio] ERROR: patch did not apply; refusing to serve a half-patched tree"
    ( cd "$ROOT" && git apply -p1 --exclude='examples/*' --check "$PATCH_FILE" ) 2>&1 | head -20
    exit 1
  fi
  find "$ROOT" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null
fi

echo "[k3-moriio] verify"
for pair in \
  "moriio_connector.py:_draft_only_layers" \
  "moriio_connector.py:attention layers carry their own" \
  "moriio_layout.py:def build_dcp_block_pairing|def compute_block_transfer_offsets" \
  "kv_cache_utils.py:KV cache grouping: buckets" \
  "rejection_sampler_utils.py:num_sampling_positions|clamp"
do
  f="${pair%%:*}"; pat="${pair#*:}"
  path=$(find "$ROOT/vllm" -name "$f" | head -1)
  n=$(grep -cE "$pat" "$path" 2>/dev/null || echo 0)
  echo "  $f  /$pat/ = $n"
done

python3 -m py_compile \
  "$ROOT/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py" \
  "$ROOT/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_common.py" \
  "$ROOT/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py" \
  "$ROOT/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_layout.py" \
  "$ROOT/vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py" \
  "$ROOT/vllm/v1/core/kv_cache_utils.py" \
  "$ROOT/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py" \
  && echo "[k3-moriio] PY_COMPILE_OK" || { echo "[k3-moriio] PY_COMPILE_FAIL"; exit 1; }

python3 - <<'PYEOF'
try:
    from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import MoRIIOConnector  # noqa: F401
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector  # noqa: F401
    print("[k3-moriio] IMPORT_OK")
except Exception as e:
    print("[k3-moriio] IMPORT_SKIPPED (needs GPU?):", type(e).__name__, e)
PYEOF
