#!/usr/bin/env bash
# ADDITIVE patch for the 08-12 nightly base (3ee2df30). The 08-12 base aiter is
# NEWER than v0.1.19.post2 and already exposes _set_current_hip_stream (which the
# nightly vllm calls for module_rmsnorm_quant) plus the tuned-gemm CSVs. The old
# wholesale post2-python overlay REGRESSED aiter/jit/core.py and dropped that
# symbol -> rmsnorm_quant warmup crash. So here we ONLY ADD the gluon sparse-MLA
# kernels that the base genuinely lacks, and apply the two vllm wiring PRs. We do
# NOT touch aiter core, and we do NOT add MegaMoE (its intranode kernel needs
# mori.ir.flydsl, absent from the base -> DEP8 is out of scope for this route).
set -uo pipefail
AITER_SHA="97d0c6e4cb7a0919c12291c7c7d560ad412f15c1"
AITER_REPO="https://github.com/ROCm/aiter"
VLLM_REPO="https://github.com/vllm-project/vllm"
ROOT="$(python -c 'import importlib.util as u, os; print(os.path.dirname(os.path.dirname(u.find_spec("vllm").origin)))')"
[ -d "$ROOT/vllm" ] && [ -d "$ROOT/aiter" ] || { echo "ERROR ROOT=$ROOT"; exit 1; }
echo "[add] ROOT=$ROOT"
WS=/tmp/dsv4_add; mkdir -p "$WS"

# --- gluon kernels only (additive; base lacks _gluon_kernels + these variants) --
GLUON_PATHS=(
  aiter/ops/triton/_gluon_kernels/gfx950/attention/pa_decode_sparse.py
  aiter/ops/triton/_gluon_kernels/gfx1250/attention/pa_decode_sparse.py
  aiter/ops/triton/_triton_kernels/attention/pa_decode_sparse.py
  aiter/ops/triton/attention/pa_decode_sparse.py
)
SRC="$WS/aiter_src"
if [ ! -d "$SRC/.git" ]; then git clone --filter=blob:none --no-checkout "$AITER_REPO" "$SRC" 2>&1 | tail -1; fi
( cd "$SRC" && git fetch --depth 1 origin "$AITER_SHA" 2>&1 | tail -1 && git checkout -q "$AITER_SHA" -- "${GLUON_PATHS[@]}" )
for p in "${GLUON_PATHS[@]}"; do
  [ -e "$SRC/$p" ] || { echo "  MISSING in src: $p"; continue; }
  mkdir -p "$(dirname "$ROOT/$p")"; cp -a "$SRC/$p" "$ROOT/$p"; echo "  added  $p"
done

# --- aiter #4417: large-token FlyDSL MoE launch/output limits -------------------
# The base's vendored aiter predates #4417 (merged 2026-07-30) even though the
# nightly itself is 08-12 -- the vendored revision is pinned, it does not track
# aiter main. #4417 guards two large-token limits: stage2 buffer atomics address
# the output with 32-bit byte offsets (>4 GiB walks off the end), and HIP caps
# grid.y at 65535. Neither fires at the DSv4-Pro TP8 shape we measure
# (requires_flydsl_stage2_reduce(65536, 7168, 2) is False -- ~939 MB), so this is
# NOT the fix for the inter_dim=384 profile-run memfault; it is carried because
# it is a real gap in the base that any larger-token sweep row would hit.
# The upstream .diff will NOT apply here (its context postdates aiter's typing
# modernization), so graft the hunks by anchor.
python "$(dirname "$0")/graft_aiter_4417.py" "$ROOT/aiter/ops/flydsl/moe_kernels.py" || \
  { echo "  #4417: GRAFT FAILED"; exit 1; }

# --- vllm wiring PRs (dormant unless the env/backend selects them) --------------
apply_pr(){ local pr="$1" mf="$ROOT/$2" mk="$3" d="$WS/vllm_$1.diff"
  if grep -qF "$mk" "$mf" 2>/dev/null; then echo "  #$pr: already present"; return; fi
  curl -ksSL -o "$d" "$VLLM_REPO/pull/$pr.diff" || { echo "  #$pr: FETCH FAIL"; return 1; }
  if ( cd "$ROOT" && git apply -p1 --3way "$d" ) 2>/dev/null || ( cd "$ROOT" && git apply -p1 "$d" ) 2>/dev/null
  then echo "  #$pr: APPLIED"; else echo "  #$pr: FAILED"; return 1; fi
}
apply_pr 51714 "vllm/v1/attention/ops/rocm_aiter_mla_sparse.py" "_DSV4_SPARSE_GLUON" || true
apply_pr 51918 "vllm/config/kernel.py" "flydsl_mega_moe" || true

# --- verify: py_compile + the ABI-sensitive imports (needs GPU) ------------------
echo "chk gluon kernel = $([ -f "$ROOT/aiter/ops/triton/_gluon_kernels/gfx950/attention/pa_decode_sparse.py" ] && echo present || echo MISSING)"
echo "chk aiter #4417 guards = $(grep -c 'requires_flydsl_stage2_reduce\|resolve_flydsl_grid_y_persist_m' "$ROOT/aiter/ops/flydsl/moe_kernels.py" 2>/dev/null) (expect 5)"
echo "chk vllm gluon wiring = $(grep -c '_DSV4_SPARSE_GLUON' "$ROOT/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py" 2>/dev/null)"
python -m py_compile "$ROOT/aiter/ops/triton/attention/pa_decode_sparse.py" \
  "$ROOT/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py" && echo PY_COMPILE_OK || { echo PY_COMPILE_FAIL; exit 1; }
python - <<'PYEOF'
import importlib
for m in ["aiter.jit.core","aiter.ops.triton.attention.pa_decode_sparse",
          "vllm.v1.attention.ops.rocm_aiter_mla_sparse","vllm._aiter_ops"]:
    try: importlib.import_module(m); print("IMPORT_OK",m)
    except Exception as e: print("IMPORT_ERR",m,type(e).__name__,(str(e).splitlines() or [''])[-1])
# the exact symbol the wholesale overlay dropped:
try:
    import aiter.jit.core as c
    print("has _set_current_hip_stream in core:", hasattr(c,"_set_current_hip_stream"))
except Exception as e: print("core probe err",e)
PYEOF
echo "[add] DONE"
