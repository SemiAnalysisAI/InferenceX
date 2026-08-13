#!/usr/bin/env bash
# =============================================================================
# Container patch stack for the 08-09 nightly base
#   vllm/vllm-openai-rocm:nightly-f8d03e77416bf90c49acbe50e233275722f02c4b
#   (vllm 0.26.1rc1.dev528+gf8d03e774)
#
# Everything here comes from an UPSTREAM ref -- either a vllm PR diff or a
# pinned aiter/mori commit. Nothing is sourced from a private measurement
# image. Idempotent: every step is marker-gated, re-running is a no-op.
#
# Run INSIDE a fresh container of the pinned image:
#   docker exec -i <container> bash /path/to/apply_dsv4_container_patches.sh
#
# WHAT THIS ENABLES
#   Three knobs that the base cannot run unpatched, all three verified to
#   import cleanly afterwards (see the checks at the end):
#     * sparse gluon decode  (VLLM_ROCM_DSV4_SPARSE_GLUON=1)
#     * FSE                  (aiter fhmoe)
#     * MegaMoE / DEP8       (--moe-backend flydsl_mega_moe)
#
#   The gluon knob in particular was silently dead before this: the base's
#   aiter facade pa_decode_sparse() predates the extra_cache/extra_indices/
#   extra_indptr keywords that vllm #51714's call site passes, so every worker
#   raised
#       TypeError: pa_decode_sparse() got an unexpected keyword argument
#                  'extra_cache'
#   on its first decode and latched onto the Triton fallback for the rest of
#   the process. Copying only the gfx950 gluon kernel is NOT enough; the facade
#   that dispatches to it has to come along.
#
# WHAT IS NOT DONE
#   * aiter core (jit/core.py) is not modified. The base already exposes
#     _set_current_hip_stream, which the nightly vllm calls for
#     module_rmsnorm_quant; an earlier wholesale post2-python overlay regressed
#     it and crashed rmsnorm_quant warmup.
#   * aiter #4417 is NOT grafted separately. It is already contained in the
#     aiter@$AITER_SHA moe_kernels.py this script installs, so the old
#     anchor-graft helper is gone.
# =============================================================================
set -uo pipefail
AITER_SHA="97d0c6e4cb7a0919c12291c7c7d560ad412f15c1"
AITER_REPO="https://github.com/ROCm/aiter"
MORI_SHA="84a33cc0f15f019c78c995728973b70ea3d10bb7"
MORI_REPO="https://github.com/ROCm/mori"
VLLM_REPO="https://github.com/vllm-project/vllm"
ROOT="$(python -c 'import importlib.util as u, os; print(os.path.dirname(os.path.dirname(u.find_spec("vllm").origin)))')"
[ -d "$ROOT/vllm" ] && [ -d "$ROOT/aiter" ] || { echo "ERROR ROOT=$ROOT"; exit 1; }
echo "[patch] ROOT=$ROOT"
echo "[patch] vllm = $(python -c 'import vllm;print(vllm.__version__)' 2>/dev/null)"
WS=/tmp/dsv4_patch; mkdir -p "$WS"

# --- 1/4 aiter files from the pinned SHA --------------------------------------
# Sourced from aiter@$AITER_SHA, not from any prebuilt image.
#
# A note on why the facade is in this list. An earlier revision of this script
# copied only the ONE file the base lacked (the gfx950 gluon kernel) and
# deliberately skipped the three pa_decode_sparse variants that ship in the
# base, to avoid a cross-version transplant. That was the right instinct but
# the wrong call for the facade specifically: keeping the base's older facade
# is what produced the extra_cache TypeError above. The facade and the kernel
# it dispatches to are one unit and have to move together.
#
# moe_kernels.py is replaced rather than patched. Symbol-diffed base vs
# upstream before doing so: 58 base symbols, 66 upstream, ZERO lost, +8 gained
# (including _flydsl_moe_stage1_impl / _flydsl_moe_stage2_impl, which FSE's
# aiter/ops/flydsl/fhmoe.py imports and the base simply does not define).
AITER_FILES=(
  # sparse gluon decode: the kernel AND the facade that dispatches to it
  aiter/ops/triton/attention/pa_decode_sparse.py
  aiter/ops/triton/_gluon_kernels/gfx950/attention/pa_decode_sparse.py
  # FSE (fused heterogeneous MoE)
  aiter/fhmoe.py
  aiter/ops/flydsl/fhmoe.py
  aiter/ops/flydsl/kernels/fhmoe.py
  aiter/aot/flydsl/fhmoe.py
  # carries FSE's stage1/stage2 impls, and aiter #4417's two guards
  aiter/ops/flydsl/moe_kernels.py
)
ASRC="$WS/aiter_src"
if [ ! -d "$ASRC/.git" ]; then git clone -q --filter=blob:none --no-checkout "$AITER_REPO" "$ASRC" 2>&1 | tail -1; fi
( cd "$ASRC" && git fetch -q --depth 1 origin "$AITER_SHA" 2>&1 | tail -1 && git checkout -q "$AITER_SHA" -- "${AITER_FILES[@]}" ) \
  || { echo "  aiter: CHECKOUT FAILED"; exit 1; }
for p in "${AITER_FILES[@]}"; do
  [ -e "$ASRC/$p" ] || { echo "  MISSING in src: $p"; continue; }
  if [ -f "$ROOT/$p" ] && cmp -s "$ASRC/$p" "$ROOT/$p"; then echo "  same       $p"; continue; fi
  mkdir -p "$(dirname "$ROOT/$p")"; cp -a "$ASRC/$p" "$ROOT/$p"; echo "  installed  $p"
done

# --- 2/4 mori: mori.ir.flydsl + the cov cascade -------------------------------
# MegaMoE's intranode dispatch/combine kernel does `import mori.ir.flydsl`, and
# the base ships mori WITHOUT that subpackage. Three .py files -- no compiled
# artifact, contrary to a first reading of the ModuleNotFoundError.
#
# They cannot be dropped in alone. mori/ir/flydsl/runtime.py calls
# find_bitcode(cov=6) (FlyDSL needs ABI 600), and `cov` is a parameter that
# cascades through three more files the base predates: ir/bitcode.py,
# jit/cache.py, jit/core.py (37 call sites upstream, 0 in the base) plus
# jit/config.py for is_debuginfo_enabled. Installing a subset yields, in order,
#   TypeError: find_bitcode() got an unexpected keyword argument 'cov'
#   ImportError: cannot import name 'is_debuginfo_enabled'
# so the whole cascade goes or none of it does.
#
# libmori_shmem_device.bc is NOT shipped here. It is a 485 KB LLVM IR blob and
# a build artifact; find_bitcode() falls back to mori.jit.core.ensure_bitcode(),
# which compiles it in-container with hipcc on first import (~1 min). That
# keeps this script free of binary payloads.
MORI_FILES=(
  mori/ir/flydsl/__init__.py
  mori/ir/flydsl/ops.py
  mori/ir/flydsl/runtime.py
  mori/ir/bitcode.py
  mori/jit/cache.py
  mori/jit/config.py
  mori/jit/core.py
)
if [ -d "$ROOT/mori" ]; then
  MSRC="$WS/mori_src"
  if [ ! -d "$MSRC/.git" ]; then git clone -q --filter=blob:none --no-checkout "$MORI_REPO" "$MSRC" 2>&1 | tail -1; fi
  ( cd "$MSRC" && git fetch -q --depth 1 origin "$MORI_SHA" 2>&1 | tail -1 && git checkout -q "$MORI_SHA" -- python/mori ) \
    || { echo "  mori: CHECKOUT FAILED"; }
  for p in "${MORI_FILES[@]}"; do
    src="$MSRC/python/$p"
    [ -e "$src" ] || { echo "  MISSING in src: $p"; continue; }
    if [ -f "$ROOT/$p" ] && cmp -s "$src" "$ROOT/$p"; then echo "  same       $p"; continue; fi
    mkdir -p "$(dirname "$ROOT/$p")"; cp -a "$src" "$ROOT/$p"; echo "  installed  $p"
  done
  find "$ROOT/mori" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null
else
  echo "  mori: not installed in this image, skipping MegaMoE deps"
fi
find "$ROOT/aiter" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null

# --- 3/4 vllm PRs -------------------------------------------------------------
fetch_diff(){ local pr="$1"; local d="$WS/vllm_$pr.diff"
  [ -s "$d" ] || curl -ksSL -o "$d" "$VLLM_REPO/pull/$pr.diff" || return 1
  [ -s "$d" ] || return 1; echo "$d"
}

# Split a PR diff down to the hunks for ONE file. `patch -i <whole.diff> <file>`
# does NOT do this -- it applies every hunk in the diff to the named file. Doing
# that once wrote dspark.py's and mtp.py's hunks into model.py and left it with
# a SyntaxError (an import landed mid-docstring). Always split first.
split_diff(){ local d="$1" f="$2" out="$3"
  python - "$d" "$f" "$out" <<'PYEOF'
import re, sys
src, target, out = sys.argv[1], sys.argv[2], sys.argv[3]
parts = re.split(r"(?m)^(?=diff --git )", open(src).read())
keep = [p for p in parts if p.startswith("diff --git a/%s " % target)]
open(out, "w").write("".join(keep))
sys.exit(0 if keep else 1)
PYEOF
}

# Apply one file's hunks. git apply first (exact); fall back to patch --fuzz,
# which tolerates the context drift from #51918 being written against a tree a
# few days newer than this base.
apply_file(){ local pr="$1" f="$2" mk="$3" fuzz="${4:-0}"
  if [ -n "$mk" ] && grep -qF "$mk" "$ROOT/$f" 2>/dev/null; then echo "  #$pr $f: already present"; return 0; fi
  local d; d="$(fetch_diff "$pr")" || { echo "  #$pr: FETCH FAIL"; return 1; }
  local one="$WS/${pr}_$(echo "$f" | tr / _).diff"
  split_diff "$d" "$f" "$one" || { echo "  #$pr $f: no hunks in diff"; return 1; }
  if ( cd "$ROOT" && git apply -p1 "$one" ) 2>/dev/null; then
    echo "  #$pr $f: APPLIED"; return 0
  fi
  if [ "$fuzz" != "0" ]; then
    if ( cd "$ROOT" && patch -p1 --forward --fuzz="$fuzz" --no-backup-if-mismatch -s -i "$one" ) 2>/dev/null; then
      echo "  #$pr $f: APPLIED (fuzz=$fuzz)"
      ( cd "$ROOT" && rm -f "$f.rej" "$f.orig" )
      return 0
    fi
    ( cd "$ROOT" && rm -f "$f.rej" "$f.orig" )
  fi
  echo "  #$pr $f: FAILED"; return 1
}

# #51473 -- native MXFP4 TP8 shard allocation. MERGED 2026-08-11, i.e. AFTER
# this 08-09 base, so unlike the 08-12 line it must be applied here. Its tests/
# hunk has no counterpart in an installed wheel, hence per-file application.
apply_file 51473 "vllm/model_executor/layers/fused_moe/oracle/mxfp4.py" \
  "AITER_MXFP4_BF16 and activation == MoEActivation.SILU" || true

# #51714 -- opt-in gluon sparse-MLA decode for gfx950. Dormant unless
# VLLM_ROCM_DSV4_SPARSE_GLUON=1. Needs the aiter facade from step 1 to work.
apply_file 51714 "vllm/v1/attention/ops/rocm_aiter_mla_sparse.py" "_DSV4_SPARSE_GLUON" || true

# #51918 -- FlyDSL mega-MoE backend, now applied IN FULL.
# A previous revision took only config/kernel.py (the backend name) and skipped
# the model side, on the grounds that mori.ir.flydsl was missing so the backend
# could never run. Step 2 removes that blocker, so the model hunks go in too.
# model.py needs fuzz=3: #51918 targets a tree a few days newer and the earlier
# hunks shift line numbering. All 6 of its hunks land; the result is
# py_compile-clean and imports (checked below).
#
# Each entry carries its own presence marker. Without one the step re-runs on an
# already-patched container, git apply correctly refuses, and the log says
# FAILED -- functionally a no-op, but it reads like a real failure in CI output.
apply_file 51918 "vllm/config/kernel.py"                          "flydsl_mega_moe"          || true
apply_file 51918 "vllm/models/deepseek_v4/amd/dspark.py"          "finalize_mega_moe_layers" || true
apply_file 51918 "vllm/models/deepseek_v4/amd/mega_moe_experts.py" "MegaMoE expert layer"    || true
apply_file 51918 "vllm/models/deepseek_v4/amd/mega_moe_runtime.py" "MegaMoEV2 runtime"       || true
apply_file 51918 "vllm/models/deepseek_v4/amd/mtp.py"             "finalize_mega_moe_layers" || true
apply_file 51918 "vllm/models/deepseek_v4/amd/model.py"           "use_mega_moe" 3           || true

# --- 4/4 verify ---------------------------------------------------------------
echo "chk gluon gfx950 kernel = $([ -f "$ROOT/aiter/ops/triton/_gluon_kernels/gfx950/attention/pa_decode_sparse.py" ] && echo present || echo MISSING)"
echo "chk aiter #4417 guards  = $(grep -c 'requires_flydsl_stage2_reduce\|resolve_flydsl_grid_y_persist_m' "$ROOT/aiter/ops/flydsl/moe_kernels.py" 2>/dev/null) (expect 5)"
echo "chk vllm  #51473        = $(grep -c 'AITER_MXFP4_BF16 and activation == MoEActivation.SILU' "$ROOT/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py" 2>/dev/null) (expect 1)"
echo "chk vllm  #51714        = $(grep -c '_DSV4_SPARSE_GLUON' "$ROOT/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py" 2>/dev/null)"
echo "chk vllm  #51918 name   = $(grep -c 'flydsl_mega_moe' "$ROOT/vllm/config/kernel.py" 2>/dev/null)"
echo "chk vllm  #51918 model  = $(grep -c 'use_mega_moe' "$ROOT/vllm/models/deepseek_v4/amd/model.py" 2>/dev/null) (expect >0)"

python -m py_compile \
  "$ROOT/aiter/ops/triton/attention/pa_decode_sparse.py" \
  "$ROOT/aiter/ops/flydsl/moe_kernels.py" \
  "$ROOT/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py" \
  "$ROOT/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py" \
  "$ROOT/vllm/models/deepseek_v4/amd/model.py" \
  "$ROOT/vllm/models/deepseek_v4/amd/mtp.py" \
  "$ROOT/vllm/models/deepseek_v4/amd/dspark.py" \
  "$ROOT/vllm/models/deepseek_v4/amd/mega_moe_experts.py" \
  "$ROOT/vllm/models/deepseek_v4/amd/mega_moe_runtime.py" \
  "$ROOT/vllm/config/kernel.py" && echo PY_COMPILE_OK || { echo PY_COMPILE_FAIL; exit 1; }

python - <<'PYEOF'
import importlib, inspect, re, sys

fail = False

# The facade signature is the check that actually matters for the gluon knob:
# the kernel file being present says nothing about whether the call site's
# keywords are accepted.
try:
    from aiter.ops.triton.attention.pa_decode_sparse import pa_decode_sparse as f
    p = inspect.signature(f).parameters
    missing = [k for k in ("extra_cache", "extra_indices", "extra_indptr") if k not in p]
    if missing:
        fail = True
        print("FACADE_MISSING", missing, "-- gluon would fall back to Triton at runtime")
    else:
        print("FACADE_OK extra_cache/extra_indices/extra_indptr accepted")
except Exception as e:
    fail = True
    print("FACADE_ERR", type(e).__name__, e)

for m in ["aiter.jit.core",
          "aiter.ops.triton.attention.pa_decode_sparse",
          "aiter.fhmoe",                                    # FSE
          "aiter.ops.flydsl.fhmoe",                         # FSE
          "mori.ir.flydsl",                                 # MegaMoE
          "aiter.ops.flydsl.kernels.flydsl_dispatch_combine_intranode_kernel",
          "vllm.model_executor.layers.fused_moe.oracle.mxfp4",
          "vllm.v1.attention.ops.rocm_aiter_mla_sparse",
          "vllm.models.deepseek_v4.amd.model",
          "vllm._aiter_ops"]:
    try:
        importlib.import_module(m)
        print("IMPORT_OK", m)
    except Exception as e:
        fail = True
        print("IMPORT_ERR", m, type(e).__name__, (str(e).splitlines() or [""])[-1])

# _set_current_hip_stream is NOT a top-level attribute of aiter.jit.core -- it
# is called there as `module._set_current_hip_stream(...)`, i.e. it lives on the
# compiled .so that core.py loads. hasattr(core, ...) therefore returns False on
# a perfectly healthy tree; an earlier version of this probe read that False as
# a missing symbol. Grep the call site instead.
try:
    import aiter.jit.core as c
    n = len(re.findall(r"_set_current_hip_stream", inspect.getsource(c)))
    print(f"core.py _set_current_hip_stream call sites: {n} (expect >=1)")
except Exception as e:
    print("core probe err", e)

print("VERIFY:", "FAILURES_PRESENT" if fail else "ALL_OK")
sys.exit(1 if fail else 0)
PYEOF
rc=$?
echo "[patch] DONE (verify rc=$rc)"
exit $rc
