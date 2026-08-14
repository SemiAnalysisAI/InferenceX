#!/usr/bin/env bash
# Fix AITER JIT modules building against a different pybind11 than the prebuilt
# .so files shipped in the image.
#
# Symptom (kills the server during warmup, before it binds):
#   TypeError: fmha_fwd_bf16_opus_fwd(): incompatible function arguments
#   RuntimeError: Engine core initialization failed
#
# Cause: aiter/jit/utils/cpp_extension.py adds the standalone pybind11 include
# via -I, which outranks the -isystem path holding torch's bundled pybind11.
# The prebuilt modules (module_aiter_core.so) use torch's pybind11
# (PYBIND11_INTERNALS_VERSION 11); the standalone package is version 12.
# pybind11 keeps a SEPARATE type registry per internals id, so the JIT module
# cannot see aiter_tensor_t registered by the core module.
#
# Effect: lets ROCM_AITER_FA be used for MLA prefill instead of falling back to
# FLASH_ATTN. Measured on 8x MI355X / Kimi-K3 MXFP4 TP8:
#   ~24k ctx  12,953 -> 13,524 tok/s  (+4.4%)
#   ~93k ctx  11,174 -> 13,423 tok/s  (+20.1%)
#
# Idempotent: safe to run repeatedly. No-op if already patched or not needed.
set -euo pipefail

PY=${PYTHON:-python3}

TARGET=$($PY - <<'EOF'
import os
try:
    import aiter.jit.utils.cpp_extension as m
    print(os.path.abspath(m.__file__))
except Exception:
    print("")
EOF
)

if [ -z "$TARGET" ] || [ ! -f "$TARGET" ]; then
    echo "[aiter-fix] aiter cpp_extension not found; nothing to do."
    exit 0
fi

# Is there actually an internals-version mismatch to fix?
NEED=$($PY - <<'EOF'
import os, re
try:
    import torch, pybind11
except Exception:
    print("no"); raise SystemExit
def ver(p):
    f = os.path.join(p, "pybind11", "detail", "internals.h")
    if not os.path.isfile(f):
        return None
    m = re.search(r"define\s+PYBIND11_INTERNALS_VERSION\s+(\d+)", open(f).read())
    return int(m.group(1)) if m else None
t = ver(os.path.join(os.path.dirname(torch.__file__), "include"))
s = ver(pybind11.get_include())
print("yes" if (t is not None and s is not None and t != s) else "no")
EOF
)

if [ "$NEED" != "yes" ]; then
    echo "[aiter-fix] pybind11 internals versions already agree; no patch needed."
    exit 0
fi

if grep -q "_use_torch_pybind11" "$TARGET"; then
    echo "[aiter-fix] already patched: $TARGET"
else
    cp -n "$TARGET" "$TARGET.orig" || true
    $PY - "$TARGET" <<'EOF'
import sys, io
path = sys.argv[1]
src = io.open(path, encoding="utf-8").read()
old = "        extra_include_paths.append(pybind11.get_include())\n"
new = (
    "        # PATCHED: prefer torch's bundled pybind11 so JIT modules land in the\n"
    "        # same pybind11 type registry as the prebuilt .so files. Mismatched\n"
    "        # PYBIND11_INTERNALS_VERSION otherwise yields:\n"
    "        #   TypeError: ...(): incompatible function arguments\n"
    "        _use_torch_pybind11 = False\n"
    "        if not torch_exclude:\n"
    "            _use_torch_pybind11 = os.path.isdir(\n"
    "                os.path.join(TORCH_INCLUDE_ROOT, \"pybind11\")\n"
    "            )\n"
    "        if not _use_torch_pybind11:\n"
    "            extra_include_paths.append(pybind11.get_include())\n"
)
if old not in src:
    sys.stderr.write("[aiter-fix] ERROR: anchor line not found; aborting.\n")
    sys.exit(1)
if src.count(old) != 1:
    sys.stderr.write("[aiter-fix] ERROR: anchor line not unique; aborting.\n")
    sys.exit(1)
io.open(path, "w", encoding="utf-8").write(src.replace(old, new))
print("[aiter-fix] patched", path)
EOF
fi

# Drop JIT artifacts built against the wrong pybind11 so they rebuild.
# Ask aiter for its jit dir: it honours AITER_JIT_DIR and falls back to ~/.aiter
# when dist-packages is not writable, so deriving it from $TARGET is wrong.
JITDIR=$($PY -c 'from aiter.jit.core import get_user_jit_dir; print(get_user_jit_dir())' 2>/dev/null || true)
if [ -z "$JITDIR" ] || [ ! -d "$JITDIR" ]; then
    JITDIR=$(dirname "$(dirname "$TARGET")")
fi
echo "[aiter-fix] jit dir: $JITDIR"
# Sweep every module, not just the one we happened to hit first: any module
# JIT-built before the patch carries the wrong internals id.
shopt -s nullglob
for so in "$JITDIR"/*.so; do
    if grep -qa "__pybind11_internals_v12" "$so" 2>/dev/null; then
        rm -f "$so"
        rm -rf "$JITDIR/build/$(basename "${so%.so}")"
        echo "[aiter-fix] removed stale v12 module: $(basename "$so")"
    fi
done
shopt -u nullglob

echo "[aiter-fix] done."
