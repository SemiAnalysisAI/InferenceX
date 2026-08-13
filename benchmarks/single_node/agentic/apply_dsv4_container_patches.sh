#!/usr/bin/env bash
# =============================================================================
# ADDITIVE patch stack for the 08-09 nightly base
#   vllm/vllm-openai-rocm:nightly-f8d03e77416bf90c49acbe50e233275722f02c4b
#   (vllm 0.26.1rc1.dev528+gf8d03e774)
#
# STRICTLY ADDITIVE. Nothing that ships in the base is overwritten:
#   * only the ONE gluon kernel file the base genuinely lacks is copied in
#     (gfx950/attention/pa_decode_sparse.py). The other three pa_decode_sparse
#     variants SHIP IN THE BASE and are deliberately NOT touched -- an earlier
#     revision of this script copied all four out of aiter@97d0c6e4, silently
#     replacing three base files with a cross-version transplant. Do not
#     reintroduce that.
#   * aiter core (jit/core.py, fused_moe.py) is not modified. The base already
#     exposes _set_current_hip_stream, which the nightly vllm calls for
#     module_rmsnorm_quant; the old wholesale post2-python overlay regressed it
#     and crashed rmsnorm_quant warmup.
#   * MegaMoE/DEP8 is out of scope on this route: its intranode kernel needs
#     mori.ir.flydsl, which is absent from the base (verified: ModuleNotFoundError).
#     FSE is likewise out (aiter #4269 needs aiter/fhmoe.py, also absent).
#
# Carried:
#   vllm  #51473  native MXFP4 TP8 shard allocation (MERGED 2026-08-11 -- i.e.
#                 AFTER this 08-09 base, so unlike the 08-12 base it must be
#                 applied here). One hunk on
#                 vllm/model_executor/layers/fused_moe/oracle/mxfp4.py.
#   vllm  #51714  opt-in AITER gluon kernel for sparse-MLA decode on gfx950
#                 (open). Dormant unless VLLM_ROCM_DSV4_SPARSE_GLUON=1.
#   vllm  #51918  FlyDSL fused mega-MoE backend (open). Dormant unless
#                 --moe-backend flydsl_mega_moe is selected.
#   aiter #4417   large-token FlyDSL MoE launch/output limits (merged
#                 2026-07-30; the base's VENDORED aiter predates it -- the
#                 vendored revision is pinned and does not track aiter main).
#
# Run INSIDE a fresh container of the pinned image:
#   docker exec -i <container> bash /path/to/dsv4_patch_0809.sh
# Idempotent: every step is marker-gated, re-running is a no-op.
# =============================================================================
set -uo pipefail
AITER_SHA="97d0c6e4cb7a0919c12291c7c7d560ad412f15c1"
AITER_REPO="https://github.com/ROCm/aiter"
VLLM_REPO="https://github.com/vllm-project/vllm"
ROOT="$(python -c 'import importlib.util as u, os; print(os.path.dirname(os.path.dirname(u.find_spec("vllm").origin)))')"
[ -d "$ROOT/vllm" ] && [ -d "$ROOT/aiter" ] || { echo "ERROR ROOT=$ROOT"; exit 1; }
echo "[add] ROOT=$ROOT"
echo "[add] vllm  = $(python -c 'import vllm;print(vllm.__version__)' 2>/dev/null)"
WS=/tmp/dsv4_add; mkdir -p "$WS"

# --- 1/4 gluon kernel: ONLY the file the base lacks ---------------------------
# Verified against this base: gfx1250/, _triton_kernels/ and the ops/triton/
# facade all ship. Copying the aiter@97d0c6e4 versions over them is the
# cross-version transplant that has to be avoided, so the loop SKIPS anything
# already present rather than overwriting it.
GLUON_PATHS=(
  aiter/ops/triton/_gluon_kernels/gfx950/attention/pa_decode_sparse.py
  aiter/ops/triton/_gluon_kernels/gfx1250/attention/pa_decode_sparse.py
  aiter/ops/triton/_triton_kernels/attention/pa_decode_sparse.py
  aiter/ops/triton/attention/pa_decode_sparse.py
)
NEED=()
for p in "${GLUON_PATHS[@]}"; do
  if [ -f "$ROOT/$p" ]; then echo "  keep base  $p"; else NEED+=("$p"); fi
done
if [ ${#NEED[@]} -gt 0 ]; then
  SRC="$WS/aiter_src"
  if [ ! -d "$SRC/.git" ]; then git clone --filter=blob:none --no-checkout "$AITER_REPO" "$SRC" 2>&1 | tail -1; fi
  ( cd "$SRC" && git fetch --depth 1 origin "$AITER_SHA" 2>&1 | tail -1 && git checkout -q "$AITER_SHA" -- "${NEED[@]}" )
  for p in "${NEED[@]}"; do
    [ -e "$SRC/$p" ] || { echo "  MISSING in src: $p"; continue; }
    mkdir -p "$(dirname "$ROOT/$p")"; cp -a "$SRC/$p" "$ROOT/$p"; echo "  added      $p"
  done
else
  echo "  (nothing to add)"
fi

# --- 2/4 aiter #4417: large-token FlyDSL MoE launch/output limits --------------
# Two guards: stage2 buffer atomics address the output with 32-bit byte offsets
# (>4 GiB walks off the end), and HIP caps grid.y at 65535. Neither fires at the
# DSv4-Pro TP8 shape measured here -- requires_flydsl_stage2_reduce(65536, 7168,
# 2) is False (~939 MB) -- so this is NOT a fix for the profile-run memfault; it
# is carried because it is a real gap any larger-token sweep row would hit.
# The upstream .diff will NOT apply: the base's aiter predates aiter's typing
# modernization, so every context line still reads Dict[str, Dict] where the
# diff expects dict[str, dict]. The hunks are grafted by anchor instead.
python "$(dirname "$0")/graft_aiter_4417.py" "$ROOT/aiter/ops/flydsl/moe_kernels.py" || \
  { echo "  #4417: GRAFT FAILED"; exit 1; }

# --- 3/4 vllm PRs -------------------------------------------------------------
apply_pr(){ local pr="$1" mf="$ROOT/$2" mk="$3" d="$WS/vllm_$1.diff"
  if grep -qF "$mk" "$mf" 2>/dev/null; then echo "  #$pr: already present"; return 0; fi
  curl -ksSL -o "$d" "$VLLM_REPO/pull/$pr.diff" || { echo "  #$pr: FETCH FAIL"; return 1; }
  if ( cd "$ROOT" && git apply -p1 --3way "$d" ) 2>/dev/null || ( cd "$ROOT" && git apply -p1 "$d" ) 2>/dev/null
  then echo "  #$pr: APPLIED"; else echo "  #$pr: FAILED"; return 1; fi
}
# #51473 carries a tests/ hunk that has no counterpart in an installed wheel, so
# apply only the runtime file rather than the whole PR diff.
apply_51473(){
  local mf="$ROOT/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py"
  local mk="AITER_MXFP4_BF16 and activation == MoEActivation.SILU"
  if grep -qF "$mk" "$mf" 2>/dev/null; then echo "  #51473: already present"; return 0; fi
  curl -ksSL -o "$WS/vllm_51473.diff" "$VLLM_REPO/pull/51473.diff" || { echo "  #51473: FETCH FAIL"; return 1; }
  ( cd "$ROOT" && git apply -p1 --include='vllm/*' --3way "$WS/vllm_51473.diff" ) 2>/dev/null \
    || ( cd "$ROOT" && git apply -p1 --include='vllm/*' "$WS/vllm_51473.diff" ) 2>/dev/null \
    && echo "  #51473: APPLIED" || { echo "  #51473: FAILED"; return 1; }
}
apply_51473 || true
apply_pr 51714 "vllm/v1/attention/ops/rocm_aiter_mla_sparse.py" "_DSV4_SPARSE_GLUON" || true

# #51918 is taken PARTIALLY, on purpose: only vllm/config/kernel.py, which
# registers "flydsl_mega_moe" as an accepted --moe-backend value. The model-side
# hunks are NOT applied.
#
# Two independent reasons, either of which is sufficient:
#   1. The backend cannot run on this base at all. MegaMoE's intranode kernel
#      imports mori.ir.flydsl, and this image has no mori.ir.flydsl (verified:
#      ModuleNotFoundError). Applying the model hunks would buy a backend that
#      raises on first use.
#   2. The model.py hunks do not apply cleanly here. #51918 is written against a
#      tree ~3 days newer than this 08-09 base; the two earlier hunks shift the
#      line numbering enough that the third fails at model.py:300. `git apply
#      --3way` cannot rescue it because site-packages is not a git repo, so
#      there are no blobs to 3-way against. Force-grafting a DEP8 code path that
#      cannot execute anyway is not worth the transplant risk -- that is exactly
#      how the earlier pa_decode_sparse damage happened.
#
# Net effect: the TP8 arm is unaffected (it never selects this backend), and a
# DEP8 row would be rejected at config time with a clear error instead of
# failing deep inside a kernel. DEP8/MegaMoE stays out of scope on this route.
apply_pr_files(){ local pr="$1" inc="$2" mf="$ROOT/$3" mk="$4" d="$WS/vllm_$1.diff"
  if grep -qF "$mk" "$mf" 2>/dev/null; then echo "  #$pr ($inc): already present"; return 0; fi
  curl -ksSL -o "$d" "$VLLM_REPO/pull/$pr.diff" || { echo "  #$pr: FETCH FAIL"; return 1; }
  ( cd "$ROOT" && git apply -p1 --include="$inc" "$d" ) 2>/dev/null \
    && echo "  #$pr ($inc): APPLIED" || { echo "  #$pr ($inc): FAILED"; return 1; }
}
apply_pr_files 51918 "vllm/config/kernel.py" "vllm/config/kernel.py" "flydsl_mega_moe" || true

# --- 4/4 verify ---------------------------------------------------------------
echo "chk gluon gfx950      = $([ -f "$ROOT/aiter/ops/triton/_gluon_kernels/gfx950/attention/pa_decode_sparse.py" ] && echo present || echo MISSING)"
echo "chk aiter #4417       = $(grep -c 'requires_flydsl_stage2_reduce\|resolve_flydsl_grid_y_persist_m' "$ROOT/aiter/ops/flydsl/moe_kernels.py" 2>/dev/null) (expect 5)"
echo "chk vllm #51473       = $(grep -c 'AITER_MXFP4_BF16 and activation == MoEActivation.SILU' "$ROOT/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py" 2>/dev/null) (expect 1)"
echo "chk vllm #51714       = $(grep -c '_DSV4_SPARSE_GLUON' "$ROOT/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py" 2>/dev/null)"
echo "chk vllm #51918       = $(grep -c 'flydsl_mega_moe' "$ROOT/vllm/config/kernel.py" 2>/dev/null)"
python -m py_compile "$ROOT/aiter/ops/triton/attention/pa_decode_sparse.py" \
  "$ROOT/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py" \
  "$ROOT/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py" && echo PY_COMPILE_OK || { echo PY_COMPILE_FAIL; exit 1; }
python - <<'PYEOF'
import importlib
for m in ["aiter.jit.core","aiter.ops.triton.attention.pa_decode_sparse",
          "vllm.model_executor.layers.fused_moe.oracle.mxfp4",
          "vllm.v1.attention.ops.rocm_aiter_mla_sparse","vllm._aiter_ops"]:
    try: importlib.import_module(m); print("IMPORT_OK",m)
    except Exception as e: print("IMPORT_ERR",m,type(e).__name__,(str(e).splitlines() or [''])[-1])
# _set_current_hip_stream is NOT a top-level attribute of aiter.jit.core -- it is
# called there as `module._set_current_hip_stream(...)`, i.e. it lives on the
# compiled .so that core.py loads. hasattr(core, ...) therefore returns False on
# a perfectly healthy tree; an earlier version of this probe read that False as
# a missing symbol. Grep the call site instead: what the old wholesale
# post2-python overlay actually did was regress core.py so the call vanished.
try:
    import aiter.jit.core as c, inspect, re
    n = len(re.findall(r"_set_current_hip_stream", inspect.getsource(c)))
    print(f"core.py _set_current_hip_stream call sites: {n} (expect >=1)")
except Exception as e: print("core probe err",e)
PYEOF
echo "[add] DONE"
