#!/usr/bin/env bash
# SGLang fixes the infera-sglang GLM-5.2 points need on the pinned
# lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261018 image, applied in the container
# before the server starts. Waiver: docs/waiver/4629.md. Each fix is marker-gated and
# idempotent, and the run fails if any of them did not land.
#
#   B1 patch_dsa_indexer_hip_dp_padded_rows.py  sglang#33059  aiter indexer rows under DP padding
#   B2 dsa_dp_sync.diff                         sglang#46830  DP-rank D2H sync deadlock
#   B6 patch_kv_metrics_send_noblock.py         sglang#46820  scheduler blocks on the KvMetrics PUSH
#   B7 patch_deepseek_v2_pp_embed_dim.py        sglang#47184  PP middle stage has no embedding
#   B8 patch_mla_absorb_bmm_default_config.py   sglang#46829  absorb-BMM tile config aborts Triton
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_DIR="${SGLANG_DIR:-/sgl-workspace/sglang}"
SRT="$SGLANG_DIR/python/sglang/srt"
git_sgl() { git -c safe.directory='*' -C "$SGLANG_DIR" "$@"; }

echo "[infera-sglang-patches] B1 dsa_indexer DP-padded rows"
python3 "$here/patch_dsa_indexer_hip_dp_padded_rows.py"

echo "[infera-sglang-patches] B2 dsa_dp_sync.diff"
if git_sgl apply --check "$here/dsa_dp_sync.diff" 2>/dev/null; then
    git_sgl apply "$here/dsa_dp_sync.diff"
    echo "  applied"
elif git_sgl apply --check -R "$here/dsa_dp_sync.diff" 2>/dev/null; then
    echo "  already applied"
else
    git_sgl apply --check "$here/dsa_dp_sync.diff"
    exit 1
fi

for patch in patch_kv_metrics_send_noblock.py patch_deepseek_v2_pp_embed_dim.py \
    patch_mla_absorb_bmm_default_config.py; do
    echo "[infera-sglang-patches] $patch"
    python3 "$here/$patch"
done

# source file (under sglang/srt) : text only the patched source contains
markers=(
    "layers/attention/dsa/dsa_indexer.py:_p1v2_trim"
    "layers/attention/dsa_backend.py:must not be None for DRAFT_EXTEND_V2"
    "managers/scheduler_components/kv_events_publisher.py:_infera_kv_metrics_noblock"
    "models/deepseek_v2.py:_infera_pp_embed_dim"
    "models/deepseek_common/attention_forward_methods/forward_mla_rocm.py:_infera_absorb_bmm_default"
)
fail=0
for spec in "${markers[@]}"; do
    file="$SRT/${spec%%:*}" marker="${spec#*:}"
    rm -f "$(dirname "$file")/__pycache__/$(basename "$file" .py)".*.pyc
    if grep -qF -- "$marker" "$file"; then
        echo "  ok      ${spec%%:*}"
    else
        echo "  MISSING ${spec%%:*} :: $marker" >&2
        fail=1
    fi
done
[[ "$fail" == 0 ]] || { echo "[infera-sglang-patches] verification failed" >&2; exit 1; }
echo "[infera-sglang-patches] all five applied"
