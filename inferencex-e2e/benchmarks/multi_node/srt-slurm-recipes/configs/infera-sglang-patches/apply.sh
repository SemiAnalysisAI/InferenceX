#!/usr/bin/env bash
# The SGLang fix the infera-sglang GLM-5.2 points still need on the pinned image, applied in the
# container before the server starts. Waiver: docs/waiver/4629.md. The fix is marker-gated and
# idempotent, and the run fails if it did not land.
#
#   B8 patch_mla_absorb_bmm_default_config.py   sglang#46829  absorb-BMM tile config aborts Triton
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_DIR="${SGLANG_DIR:-/sgl-workspace/sglang}"
SRT="$SGLANG_DIR/python/sglang/srt"

echo "[infera-sglang-patches] B8 patch_mla_absorb_bmm_default_config.py"
python3 "$here/patch_mla_absorb_bmm_default_config.py"

# source file (under sglang/srt) : text only the patched source contains
markers=(
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
echo "[infera-sglang-patches] B8 applied"
