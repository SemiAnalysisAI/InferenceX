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
#   Two knobs that the base cannot run unpatched, both verified to import
#   cleanly afterwards (see the checks at the end):
#     * sparse gluon decode  (VLLM_ROCM_DSV4_SPARSE_GLUON=1)
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
# WHAT IS NOT DONE, AND WHY
#   * FSE (aiter fhmoe) is NOT enabled. It was attempted and reverted, because
#     it cannot be reached by file-level transplant on this base:
#       - All four fhmoe files are ABSENT from the base's vendored aiter, and
#         they need _flydsl_moe_stage1_impl / _flydsl_moe_stage2_impl, which
#         only exist in the upstream ops/flydsl/moe_kernels.py.
#       - Swapping in that moe_kernels.py looks safe by symbol diff (58 -> 66
#         defined symbols, 0 lost, 8 gained) but the diff only measures what a
#         file DEFINES, not what it CALLS. The upstream file calls
#         compile_mixed_moe_gemm1(v2_output_layout=...) into
#         kernels/mixed_moe_gemm_2stage.py, and the base's copy of that file
#         has no such parameter. Every TP worker then died in the profile run:
#             TypeError: compile_mixed_moe_gemm1() got an unexpected keyword
#                        argument 'v2_output_layout'
#         reached via fused_moe_2stages -> flydsl_moe_stage1, i.e. on the
#         DEFAULT --moe-backend aiter path. It broke the main TP8 arm, not just
#         FSE.
#       - Chasing the signature is not bounded. The intra-aiter import closure
#         of the FSE chain is 30 files that differ from upstream, including
#         kernels/mixed_moe_gemm_2stage.py at 9024 lines in the base vs 146
#         upstream. The base's vendored aiter is a structural fork, not an
#         older revision of the same tree, so "sync a few more files" converges
#         on replacing the vendored tree wholesale.
#     For contrast, the gluon chain's closure is 9 files with 3 differing, none
#     of them on the call path -- which is why that one is safe to transplant
#     and this one is not. FSE needs an image whose vendored aiter already has
#     it.
#   * aiter core (jit/core.py) is not modified. The base already exposes
#     _set_current_hip_stream, which the nightly vllm calls for
#     module_rmsnorm_quant; an earlier wholesale post2-python overlay regressed
#     it and crashed rmsnorm_quant warmup.
#   * aiter #4417 is NOT grafted. Its two guards cannot fire on this recipe:
#     requires_flydsl_stage2_reduce first fires at 299594 tokens against a
#     65536 ceiling (max_num_batched_tokens 16384 x the MTP fan-out of 4), and
#     resolve_flydsl_grid_y_persist_m returns persist_m=1 with 8x of headroom
#     under the grid.y cap. The sweep grid varies concurrency only.
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
# This list is deliberately just the gluon pair. ops/flydsl/moe_kernels.py and
# the four fhmoe files were in it and were removed -- see "WHAT IS NOT DONE"
# above for the TypeError they caused on the default MoE path.
AITER_FILES=(
  # sparse gluon decode: the kernel AND the facade that dispatches to it
  aiter/ops/triton/attention/pa_decode_sparse.py
  aiter/ops/triton/_gluon_kernels/gfx950/attention/pa_decode_sparse.py
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
# moe_kernels.py must be the BASE copy, not the upstream one -- see the FSE note
# in the header. The upstream file calls compile_mixed_moe_gemm1 with a keyword
# the base's mixed_moe_gemm_2stage.py does not accept, which kills the default
# --moe-backend aiter path during the profile run.
echo "chk moe_kernels is base = $(grep -c 'v2_output_layout' "$ROOT/aiter/ops/flydsl/moe_kernels.py" 2>/dev/null) (expect 0)"
echo "chk vllm  #51473        = $(grep -c 'AITER_MXFP4_BF16 and activation == MoEActivation.SILU' "$ROOT/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py" 2>/dev/null) (expect 1)"
echo "chk vllm  #51714        = $(grep -c '_DSV4_SPARSE_GLUON' "$ROOT/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py" 2>/dev/null)"
echo "chk vllm  #51918 name   = $(grep -c 'flydsl_mega_moe' "$ROOT/vllm/config/kernel.py" 2>/dev/null)"
echo "chk vllm  #51918 model  = $(grep -c 'use_mega_moe' "$ROOT/vllm/models/deepseek_v4/amd/model.py" 2>/dev/null) (expect >0)"

python -m py_compile \
  "$ROOT/aiter/ops/triton/attention/pa_decode_sparse.py" \
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

# The default MoE path must still resolve. An upstream moe_kernels.py imports
# cleanly and only explodes later, inside the profile run, when flydsl_moe_stage1
# calls compile_mixed_moe_gemm1 with a keyword the base does not accept -- so an
# import check alone would have passed. Check the signature agreement directly.
#
# Read the keywords off the CALL NODE via AST. Grepping the function body for
# `name=` instead picks up every local assignment in it and reports them as
# rejected kwargs -- a false positive this check produced on a clean tree.
try:
    import ast, textwrap
    from aiter.ops.flydsl.kernels.mixed_moe_gemm_2stage import compile_mixed_moe_gemm1
    import aiter.ops.flydsl.moe_kernels as mk
    sig = inspect.signature(compile_mixed_moe_gemm1)
    accepted = set(sig.parameters)
    var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    tree = ast.parse(textwrap.dedent(inspect.getsource(mk.compile_flydsl_moe_stage1)))
    passed = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            fn = n.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if name == "compile_mixed_moe_gemm1":
                passed |= {kw.arg for kw in n.keywords if kw.arg}
    unknown = sorted(passed - accepted)
    if unknown and not var_kw:
        fail = True
        print("MOE_SIG_MISMATCH compile_mixed_moe_gemm1 rejects:", unknown)
    else:
        print(f"MOE_SIG_OK compile_flydsl_moe_stage1 -> compile_mixed_moe_gemm1 "
              f"({len(passed)} kwargs, all accepted)")
except Exception as e:
    print("MOE_SIG probe err", type(e).__name__, e)

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
