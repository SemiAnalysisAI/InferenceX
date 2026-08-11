#!/usr/bin/env bash
# ROCm/aiter PR #4521 -- "MLA PS mode fp8 -n 16,3 16,4", qh16 subset.
#
# WHY THIS EXISTS
# ---------------
# Run 31148006934 (stock cb810 nightly, k=2, block, MLA_FORCE_PS=1) loaded
#   mla_a16w8_qh16_m16x4_n16x1_coex0_mask1_ps.co
# i.e. a BF16 query. Not because vLLM refused to hand over an fp8 query --
# supports_quant_query_input is already True on this path -- but because base
# aiter has no fp8/fp8 persistent branch at gqa_ratio 16, so the selector fell
# back to the bf16-query kernel. #4521 adds that branch. It is the single
# change under test here.
#
# WHAT IT TAKES
# -------------
# The decisive hunk is C++ (csrc/py_itfs_cu/asm_mla.cu): a gfx950 fp8/fp8
# branch mapping gqa_ratio 16 with max_seqlen_q 3 or 4 onto the qLen=4 HSACO at
# runtime MQA width 3, the persistent guard relaxed >=4 -> >=3, and a
# kPackedQoLenPerWg change in v1_2_device.cuh. That file is JIT-built --
# aiter/jit/optCompilerConfig.json registers module_mla_asm with source
# {AITER_CSRC_DIR}/py_itfs_cu/asm_mla.cu -- so patching it and dropping the
# cached module is enough; no full aiter rebuild.
#
# The PR also ships four qh16/qseqlen4 HSACO binaries which a .diff cannot
# carry. They are vendored next to this script and sha256-verified, taken from
# ca1c24a9dd7e0842acc9a5f4adcbb5e4b412044c.
#
# Fails loudly rather than skipping: a silently unpatched run is a stock
# baseline republished under this experiment's name.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSETS="$HERE/aiter_pr4521_gfx950"

AITER_DIR="$(python3 -c 'import aiter, os; print(os.path.dirname(os.path.dirname(aiter.__file__)))')"
META="$AITER_DIR/aiter_meta"
HSA_DIR="$META/hsa/gfx950/mla"
echo "[4521] AITER_DIR=$AITER_DIR"

for f in "$AITER_DIR/aiter/mla.py" "$AITER_DIR/aiter/ops/attention.py" \
         "$META/csrc/py_itfs_cu/asm_mla.cu" \
         "$META/csrc/kernels/mla/metadata/v1_2_device.cuh" \
         "$HSA_DIR/mla_asm.csv"; do
    [ -f "$f" ] || { echo "[4521] ERROR: missing $f" >&2; exit 1; }
done

# ---- source hunks -----------------------------------------------------------
# Two roots: the python package installs at aiter/, the kernel sources under
# aiter_meta/, so the PR's a/aiter/... and a/csrc/... paths need separate bases.
apply_diff() {
    local root="$1" diff="$2" label="$3"
    if ( cd "$root" && patch -p1 --dry-run --batch --forward < "$diff" >/dev/null 2>&1 ); then
        ( cd "$root" && patch -p1 --batch --forward < "$diff" )
        echo "[4521] applied: $label"
    elif ( cd "$root" && patch -p1 --dry-run --batch --reverse < "$diff" >/dev/null 2>&1 ); then
        echo "[4521] already applied: $label"
    else
        echo "[4521] ERROR: $label does not apply cleanly to this build" >&2
        ( cd "$root" && patch -p1 --batch --forward < "$diff" ) || true
        exit 1
    fi
}
apply_diff "$AITER_DIR" "$HERE/aiter_pr4521_py.diff"   "aiter #4521 python (mla.py, ops/attention.py)"
apply_diff "$META"      "$HERE/aiter_pr4521_csrc.diff" "aiter #4521 csrc (asm_mla.cu, v1_2_device.cuh)"

# ---- kernel registry + HSACO binaries ---------------------------------------
( cd "$ASSETS" && sha256sum -c SHA256SUMS >/dev/null ) \
    || { echo "[4521] ERROR: vendored asset checksums do not match" >&2; exit 1; }
echo "[4521] asset checksums verified"

install -m 0644 "$ASSETS"/mla_a8w8_qh16_qseqlen4_gqaratio16_*.co "$HSA_DIR/"
install -m 0644 "$ASSETS/mla_asm.csv" "$HSA_DIR/mla_asm.csv"
echo "[4521] installed 4 qh16 qseqlen4 HSACOs + mla_asm.csv"

grep -q 'mla_a8w8_qh16_qseqlen4_gqaratio16_cprr_v3_ps.co' "$HSA_DIR/mla_asm.csv" \
    || { echo "[4521] ERROR: CSV missing the qh16 cprr rows" >&2; exit 1; }

# ---- drop the cached JIT module so asm_mla.cu is rebuilt --------------------
# Nothing else invalidates it: the cache keys on module name, not source hash,
# so a stale module_mla_asm.so would silently keep the pre-#4521 dispatch.
removed=0
for d in "$AITER_DIR/aiter/jit/build" "${AITER_JIT_DIR:-}" "$HOME/.aiter" /root/.aiter; do
    [ -n "$d" ] && [ -d "$d" ] || continue
    while IFS= read -r hit; do
        rm -rf "$hit"; removed=$((removed+1)); echo "[4521] dropped JIT cache: $hit"
    done < <(find "$d" -maxdepth 2 -name 'module_mla_asm*' 2>/dev/null)
done
echo "[4521] JIT cache entries dropped: $removed (module_mla_asm rebuilds on first use)"

python3 -c "import ast,sys; [ast.parse(open(p).read()) for p in sys.argv[1:]]" \
    "$AITER_DIR/aiter/mla.py" "$AITER_DIR/aiter/ops/attention.py"
echo "[4521] done"
