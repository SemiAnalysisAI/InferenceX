#!/usr/bin/env bash
# =============================================================================
# Kimi-K3 / MI355X (gfx950) in-container patches.
#
# Everything here patches files inside the running container only
# (site-packages). Nothing outside the container is touched. Each patch is
# idempotent, verifies its own anchor, backs up to <file>.orig, and no-ops if
# the image already ships the fix. A failed anchor aborts that patch cleanly
# rather than corrupting the file, so a future image with different sources
# degrades to "unpatched", never to "broken".
#
#   [1] aiter pybind11 internals mismatch  -> unblocks ROCM_AITER_FA prefill
#   [2] KV block-pool negative-count clamp -> stops the mid-run engine crash
#   [3] live vLLM DCP branch               -> Kimi-K3 DSpark DCP correctness
#   [4] validated AITER Gluon overlay      -> fp8 DCP verify support
#
# Env:
#   SKIP_KIMI_PATCHES=1  skip everything
#   PYTHON=...           interpreter to use (default python3)
#   VLLM_DCP_REPO_URL=...  vLLM fork URL
#   VLLM_DCP_BRANCH=...    branch fetched on every container start
#   VLLM_DCP_BASE=...      image commit used as the patch base
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
# [2] vLLM: clamp the negative block count that corrupts the KV free list
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

# -----------------------------------------------------------------------------
# [3] Fetch and apply the current Kimi-K3 DSpark DCP branch
# -----------------------------------------------------------------------------
# Pull the branch on every fresh container start, then generate a production-only
# patch against the exact vLLM commit in the pinned image. This keeps SA synced
# when the DCP branch is fixed without copying a new multi-thousand-line diff
# into InferenceX. A failed fetch, dry-run, apply, or verification is fatal.
patch_vllm_dcp_branch() (
    local label="vllm-dcp-branch"
    local repo_url="${VLLM_DCP_REPO_URL:-https://github.com/YukioZzz/vllm.git}"
    local branch="${VLLM_DCP_BRANCH:-yichaozhu/k3-dspark-dcp-v3}"
    local base="${VLLM_DCP_BASE:-ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9}"
    if ! command -v git >/dev/null 2>&1 || ! command -v patch >/dev/null 2>&1; then
        echo "[$label] git and patch are required in the serving image." >&2
        return 1
    fi
    local root; root=$($PY -c 'import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))' 2>/dev/null)
    if [ -z "$root" ] || [ ! -d "$root/vllm" ]; then
        echo "[$label] vllm root not found." >&2
        return 1
    fi

    local workdir repo patch_file
    workdir=$(mktemp -d)
    repo="$workdir/vllm"
    patch_file="$workdir/vllm-dcp.patch"
    trap 'rc=$?; if [ "$rc" -ne 0 ]; then echo "[$label] FAILED (exit $rc)." >&2; fi; rm -rf "$workdir"' EXIT

    echo "[$label] fetching $repo_url branch $branch (base $base)..."
    git -C "$workdir" init --quiet vllm
    git -C "$repo" remote add origin "$repo_url"
    if ! git -C "$repo" fetch --quiet --no-tags --depth=1 origin \
        "$base:refs/dcp-patch/base" \
        "refs/heads/$branch:refs/dcp-patch/head"; then
        echo "[$label] FETCH FAILED for $repo_url $branch." >&2
        return 1
    fi

    local base_sha head_sha state_file
    base_sha=$(git -C "$repo" rev-parse refs/dcp-patch/base)
    head_sha=$(git -C "$repo" rev-parse refs/dcp-patch/head)
    state_file="$root/.inferencex-vllm-dcp-head"
    echo "[$label] fetched base=$base_sha head=$head_sha"

    verify_dcp_tree() {
        local path expected actual
        while IFS= read -r -d '' path; do
            if git -C "$repo" cat-file -e "refs/dcp-patch/head:$path" 2>/dev/null; then
                if [ ! -f "$root/$path" ]; then
                    echo "[$label] VERIFY FAILED: missing $path" >&2
                    return 1
                fi
                expected=$(git -C "$repo" rev-parse "refs/dcp-patch/head:$path")
                actual=$(git hash-object "$root/$path")
                if [ "$actual" != "$expected" ]; then
                    echo "[$label] VERIFY FAILED: content mismatch in $path" >&2
                    return 1
                fi
            elif [ -e "$root/$path" ]; then
                echo "[$label] VERIFY FAILED: deleted path still exists: $path" >&2
                return 1
            fi
        done < <(
            git -C "$repo" diff --name-only -z \
                refs/dcp-patch/base refs/dcp-patch/head -- vllm
        )
    }

    if [ -f "$state_file" ] && [ "$(cat "$state_file")" = "$head_sha" ]; then
        verify_dcp_tree
        echo "[$label] already applied and verified head=$head_sha"
        return 0
    fi
    if [ -f "$state_file" ]; then
        echo "[$label] a different DCP head is already applied; use a fresh container." >&2
        echo "[$label] applied=$(cat "$state_file") requested=$head_sha" >&2
        return 1
    fi

    git -C "$repo" diff --binary refs/dcp-patch/base refs/dcp-patch/head \
        -- vllm > "$patch_file"
    if [ ! -s "$patch_file" ]; then
        echo "[$label] generated patch is empty; refusing to continue." >&2
        return 1
    fi
    echo "[$label] production diff:"
    git -C "$repo" diff --stat refs/dcp-patch/base refs/dcp-patch/head -- vllm

    if ! patch -p1 -d "$root" --dry-run --forward < "$patch_file"; then
        echo "[$label] dry-run failed; refusing a partial patch." >&2
        return 1
    fi
    echo "[$label] dry-run succeeded."
    if ! patch -p1 -d "$root" --forward --backup \
        --suffix=.dcp-branch.orig < "$patch_file"; then
        echo "[$label] apply failed after a clean dry-run." >&2
        return 1
    fi
    verify_dcp_tree
    printf '%s\n' "$head_sha" > "$state_file"
    echo "[$label] SUCCESS: applied $base_sha..$head_sha to $root"
)

# -----------------------------------------------------------------------------
# [4] AITER: validated Gluon MLA support for DCP multi-token verify
# -----------------------------------------------------------------------------
AITER_GLUON_SHA="9459fce5dccd81a65e3e49e278f35d38e4b79d4552c6ac6ebdf406a937e063d7"
patch_aiter_gluon_fp8_dcp() {
    local label="aiter-gluon-fp8-dcp"
    local target; target=$(_modfile aiter.ops.triton.gluon.mla_gluon)
    local source="$(dirname "$0")/mla_gluon_k3_dcp.py"
    if [ -z "$target" ] || [ ! -f "$target" ]; then
        echo "[$label] target not found." >&2
        return 1
    fi
    if [ ! -f "$source" ]; then
        echo "[$label] vendored overlay not found: $source" >&2
        return 1
    fi
    local got; got=$(sha256sum "$source" | cut -d" " -f1)
    if [ "$got" != "$AITER_GLUON_SHA" ]; then
        echo "[$label] sha256 mismatch; refusing overlay." >&2
        return 1
    fi
    if [ "$(sha256sum "$target" | cut -d" " -f1)" = "$AITER_GLUON_SHA" ]; then
        echo "[$label] already patched."
        return 0
    fi
    cp -n "$target" "$target.orig" 2>/dev/null || true
    cp "$source" "$target"
    $PY -m py_compile "$target"
    echo "[$label] patched $target"
}

# Per-patch switches, so a local patch can be isolated without disabling the
# others. Note patch [1] is load-bearing: without it ROCM_AITER_FA prefill dies
# at warmup with the fmha_fwd_bf16_opus TypeError, so skipping it does not give
# a clean baseline -- it gives a different crash.
#   SKIP_PATCH_AITER=1      skip [1] aiter pybind11
#   SKIP_PATCH_BLOCKPOOL=1  skip [2] KV block-pool clamp
#
# Order is load-bearing. [3] generates a production diff against the image
# commit and then hash-verifies every touched file against the DCP branch
# HEAD. [2] edits vllm/v1/core/single_type_kv_cache_manager.py, which is also
# in that diff; applying [2] first made CI fail with
#   VERIFY FAILED: content mismatch in .../single_type_kv_cache_manager.py
# (run 32571248409). Apply the branch onto the stock image, then layer [2].
echo "[kimi-patches] applying in-container patches..."
if [ "${SKIP_PATCH_AITER:-0}" = "1" ]; then
    echo "[aiter-pybind11] SKIPPED via SKIP_PATCH_AITER=1"
else
    patch_aiter_pybind11 || true
fi
patch_vllm_dcp_branch
if [ "${SKIP_PATCH_BLOCKPOOL:-0}" = "1" ]; then
    echo "[kv-blockpool] SKIPPED via SKIP_PATCH_BLOCKPOOL=1"
else
    patch_kv_blockpool || true
fi
patch_aiter_gluon_fp8_dcp
echo "[kimi-patches] done."
