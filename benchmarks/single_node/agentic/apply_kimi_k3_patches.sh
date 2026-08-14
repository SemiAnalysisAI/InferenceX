#!/usr/bin/env bash
# =============================================================================
# Kimi-K3 / MI355X (gfx950) in-container patches — all three in one place.
#
# Everything here patches files inside the running container only
# (site-packages). Nothing outside the container is touched. Each patch is
# idempotent, verifies its own anchor, backs up to <file>.orig, and no-ops if
# the image already ships the fix. A failed anchor aborts that patch cleanly
# rather than corrupting the file, so a future image with different sources
# degrades to "unpatched", never to "broken".
#
#   [1] aiter pybind11 internals mismatch  -> unblocks ROCM_AITER_FA prefill
#   [2] TritonMLA cudagraph support        -> FULL cudagraphs for DSpark (5.52x TPOT)
#   [3] KV block-pool negative-count clamp -> stops the mid-run engine crash
#
# Env:
#   SKIP_KIMI_PATCHES=1   skip everything
#   PYTHON=...            interpreter to use (default python3)
# =============================================================================
set -euo pipefail
PY=${PYTHON:-python3}

if [ "${SKIP_KIMI_PATCHES:-0}" = "1" ]; then
    echo "[kimi-patches] SKIP_KIMI_PATCHES=1, doing nothing."
    exit 0
fi

# Locate an installed module's file, or empty string if unimportable.
_modfile() {
    $PY - "$1" <<'EOF'
import importlib, os, sys
try:
    print(os.path.abspath(importlib.import_module(sys.argv[1]).__file__))
except Exception:
    print("")
EOF
}

# _patch <file> <already-patched-marker> <<'PYEOF' ... old/new python ... PYEOF
# The heredoc body must define OLD and NEW strings.
_patch() {
    local target="$1" marker="$2" label="$3"
    if [ -z "$target" ] || [ ! -f "$target" ]; then
        echo "[$label] target not found; skipping."
        return 0
    fi
    if grep -q "$marker" "$target"; then
        echo "[$label] already patched."
        return 0
    fi
    cp -n "$target" "$target.orig" 2>/dev/null || true
    if $PY - "$target" "$label"; then
        return 0
    else
        echo "[$label] patch failed; left unchanged." >&2
        return 0
    fi
}

# -----------------------------------------------------------------------------
# [1] aiter: JIT modules must use torch's bundled pybind11
# -----------------------------------------------------------------------------
# aiter/jit/utils/cpp_extension.py appends the STANDALONE pybind11 include via
# -I, which outranks the -isystem path carrying torch's bundled copy. The 117
# prebuilt aiter .so are built against torch's (PYBIND11_INTERNALS_VERSION 11);
# the standalone package here is version 12. pybind11 keeps a SEPARATE type
# registry per internals id, so a JIT-built module cannot see aiter_tensor_t
# registered by the prebuilt core and the first call dies during warmup with
#   TypeError: fmha_fwd_bf16_opus_fwd(): incompatible function arguments
# even though arity and types match exactly.
patch_aiter_pybind11() {
    local label="aiter-pybind11"
    local target; target=$(_modfile aiter.jit.utils.cpp_extension)
    if [ -z "$target" ] || [ ! -f "$target" ]; then
        echo "[$label] aiter not present; skipping."; return 0
    fi

    # Only act if the two pybind11s actually disagree.
    local need; need=$($PY - <<'EOF'
import os, re
try:
    import torch, pybind11
except Exception:
    print("no"); raise SystemExit
def ver(p):
    f = os.path.join(p, "pybind11", "detail", "internals.h")
    if not os.path.isfile(f): return None
    m = re.search(r"define\s+PYBIND11_INTERNALS_VERSION\s+(\d+)", open(f).read())
    return int(m.group(1)) if m else None
t = ver(os.path.join(os.path.dirname(torch.__file__), "include"))
s = ver(pybind11.get_include())
print("yes" if (t is not None and s is not None and t != s) else "no")
EOF
)
    if [ "$need" != "yes" ]; then
        echo "[$label] pybind11 internals already agree; nothing to do."; return 0
    fi

    if grep -q "_use_torch_pybind11" "$target"; then
        echo "[$label] already patched."
    else
        cp -n "$target" "$target.orig" 2>/dev/null || true
        $PY - "$target" <<'EOF' || echo "[aiter-pybind11] patch failed; unchanged." >&2
import sys, io
p = sys.argv[1]
src = io.open(p, encoding="utf-8").read()
old = "        extra_include_paths.append(pybind11.get_include())\n"
new = (
    "        # PATCHED: prefer torch's bundled pybind11 so JIT modules land in the\n"
    "        # same pybind11 type registry as the prebuilt .so files.\n"
    "        _use_torch_pybind11 = False\n"
    "        if not torch_exclude:\n"
    "            _use_torch_pybind11 = os.path.isdir(\n"
    "                os.path.join(TORCH_INCLUDE_ROOT, \"pybind11\")\n"
    "            )\n"
    "        if not _use_torch_pybind11:\n"
    "            extra_include_paths.append(pybind11.get_include())\n"
)
if src.count(old) != 1:
    sys.stderr.write("[aiter-pybind11] anchor missing or not unique; aborting.\n")
    sys.exit(1)
io.open(p, "w", encoding="utf-8").write(src.replace(old, new))
print("[aiter-pybind11] patched", p)
EOF
    fi

    # Drop JIT artifacts built against the wrong pybind11 so they rebuild.
    # aiter honours AITER_JIT_DIR and falls back to ~/.aiter when dist-packages
    # is read-only, so ask aiter rather than deriving the path from $target.
    local jitdir
    jitdir=$($PY -c 'from aiter.jit.core import get_user_jit_dir; print(get_user_jit_dir())' 2>/dev/null || true)
    [ -n "$jitdir" ] && [ -d "$jitdir" ] || jitdir=$(dirname "$(dirname "$target")")
    shopt -s nullglob
    for so in "$jitdir"/*.so; do
        if grep -qa "__pybind11_internals_v12" "$so" 2>/dev/null; then
            rm -f "$so"; rm -rf "$jitdir/build/$(basename "${so%.so}")"
            echo "[$label] removed stale v12 module: $(basename "$so")"
        fi
    done
    shopt -u nullglob
}

# -----------------------------------------------------------------------------
# [2] vLLM: let DSpark spec-decode keep FULL cudagraphs
# -----------------------------------------------------------------------------
# TritonMLAMetadataBuilder._cudagraph_support = UNIFORM_SINGLE_TOKEN_DECODE caps
# min_cg_support below UNIFORM_BATCH, so config/compilation.py downgrades
# FULL_AND_PIECEWISE -> PIECEWISE under spec-decode. dflash/speculator.py then
# gives the DSpark drafter CUDAGraphMode.NONE -- fully eager -- and logs nothing.
# TRITON_MLA cannot be swapped out: it is the only ROCm MLA backend with
# supports_non_causal_multi_token_decode=True, which DSpark requires
# (ROCM_AITER_MLA fails with "non-causal attention not supported").
# The builder already calls _init_reorder_batch_threshold(1,
# supports_spec_as_decode=True) "so full-cudagraph capture admits it", so
# UNIFORM_BATCH is the self-consistent value.
# MEASURED 8x MI355X single stream, 600-token gens:
#   before 14.05 tok/s ITL 71.16 ms  ->  after 77.65 tok/s ITL 12.88 ms  (5.52x)
patch_triton_mla_cudagraph() {
    local label="triton-mla-cudagraph"
    local target; target=$(_modfile vllm.v1.attention.backends.mla.triton_mla)
    if [ -z "$target" ] || [ ! -f "$target" ]; then
        echo "[$label] target not found; skipping."; return 0
    fi
    if grep -q "AttentionCGSupport.UNIFORM_BATCH" "$target"; then
        echo "[$label] already patched."; return 0
    fi
    cp -n "$target" "$target.orig" 2>/dev/null || true
    $PY - "$target" <<'EOF' || echo "[triton-mla-cudagraph] patch failed; unchanged." >&2
import sys, io
p = sys.argv[1]
src = io.open(p, encoding="utf-8").read()
old = """    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )"""
new = """    # PATCHED: UNIFORM_SINGLE_TOKEN_DECODE forced a PIECEWISE downgrade under
    # spec-decode, which silently made the DSpark drafter fully eager.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH"""
if src.count(old) != 1:
    sys.stderr.write("[triton-mla-cudagraph] anchor missing or not unique; aborting.\n")
    sys.exit(1)
io.open(p, "w", encoding="utf-8").write(src.replace(old, new))
print("[triton-mla-cudagraph] patched", p)
EOF
}

# -----------------------------------------------------------------------------
# [3] vLLM: clamp the negative block count that corrupts the KV free list
# -----------------------------------------------------------------------------
# single_type_kv_cache_manager.py, in allocate_external_computed_blocks(), is the
# ONLY unguarded get_new_blocks() call site in that file (siblings clamp or
# early-return). When len(req_blocks) exceeds the block count implied by
# num_total_computed_tokens the argument goes NEGATIVE, and a negative count is
# silently destructive rather than rejected:
#   * block_pool.get_new_blocks only rejects  num_blocks > free
#   * popleft_n passes its own  assert num_free_blocks >= n
#   * it runs  num_free_blocks -= n   -> an INCREASE
#   * range(n) iterates zero times, so the linked list is untouched
# num_free_blocks is then inflated relative to the real free list; a later
# legitimate pop walks past the tail and the engine dies mid-run on
#   kv_cache_utils.py  assert curr_block is not None
#   block_pool.py      assert block.ref_cnt == 0
# Load-dependent: c10 died at 3612 s, c12 at 487 s, c16 at 354 s. On the EXTERNAL
# block path, so it needs --kv-transfer-config to appear. NOTE
# --no-async-scheduling was tested and does NOT help (c12 died at 490 s).
patch_kv_blockpool() {
    local label="kv-blockpool"
    local target; target=$(_modfile vllm.v1.core.single_type_kv_cache_manager)
    if [ -z "$target" ] || [ ! -f "$target" ]; then
        echo "[$label] target not found; skipping."; return 0
    fi
    # NB: the marker must be unique to OUR patch. "num_new_blocks = max(" is NOT
    # -- stock already has it at three other call sites (lines ~208/1511/1601),
    # so using it silently skipped the patch on a clean image.
    if grep -q "KIMI-PATCH-KV-BLOCKPOOL" "$target"; then
        echo "[$label] already patched."; return 0
    fi
    cp -n "$target" "$target.orig" 2>/dev/null || true
    $PY - "$target" <<'EOF' || echo "[kv-blockpool] patch failed; unchanged." >&2
import sys, io
p = sys.argv[1]
src = io.open(p, encoding="utf-8").read()
old = """        req_blocks = self.req_to_blocks[request_id]
        allocated_blocks = self.block_pool.get_new_blocks(
            cdiv(num_total_computed_tokens, self.block_size) - len(req_blocks)
        )"""
new = """        req_blocks = self.req_to_blocks[request_id]
        # KIMI-PATCH-KV-BLOCKPOOL: clamp to >= 0; a negative count silently
        # inflates FreeKVCacheBlockQueue.num_free_blocks and corrupts the free list.
        num_new_blocks = max(
            0, cdiv(num_total_computed_tokens, self.block_size) - len(req_blocks)
        )
        allocated_blocks = self.block_pool.get_new_blocks(num_new_blocks)"""
if src.count(old) != 1:
    sys.stderr.write("[kv-blockpool] anchor missing or not unique; aborting.\n")
    sys.exit(1)
io.open(p, "w", encoding="utf-8").write(src.replace(old, new))
print("[kv-blockpool] patched", p)
EOF
}

echo "[kimi-patches] applying in-container patches..."
patch_aiter_pybind11      || true
patch_triton_mla_cudagraph || true
patch_kv_blockpool         || true
echo "[kimi-patches] done."
