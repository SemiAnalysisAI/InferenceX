#!/usr/bin/env bash
# Let DSpark spec-decode keep FULL cudagraphs on ROCm.
#
# vllm/v1/attention/backends/mla/triton_mla.py declares
#   TritonMLAMetadataBuilder._cudagraph_support = UNIFORM_SINGLE_TOKEN_DECODE
# which caps min_cg_support below UNIFORM_BATCH, so config/compilation.py:1443
# downgrades FULL_AND_PIECEWISE -> PIECEWISE whenever spec-decode is enabled:
#   "CUDAGraphMode.FULL_AND_PIECEWISE is not supported with spec-decode for
#    attention backend TritonMLABackend"
# v1/worker/gpu/spec_decode/dflash/speculator.py:110-127 then gives the DSpark
# drafter CUDAGraphMode.NONE -- fully eager, with NO warning logged. Every draft
# layer and the Markov head dispatch kernel-by-kernel from Python each step.
#
# TRITON_MLA cannot simply be swapped out: it is the only ROCm MLA backend with
# supports_non_causal_multi_token_decode = True, which the DSpark draft needs.
# ROCM_AITER_MLA fails with "non-causal attention not supported".
#
# The builder already sets supports_non_causal_multi_token_decode = True and
# calls _init_reorder_batch_threshold(1, supports_spec_as_decode=True) "so
# full-cudagraph capture admits it", so UNIFORM_BATCH is the consistent value.
#
# MEASURED on 8x MI355X, Kimi-K3 MXFP4 TP8, DSpark, single stream, 600-tok gens:
#   before: 14.05 tok/s, ITL 71.16 ms   (PIECEWISE, drafter eager)
#   after : 77.65 tok/s, ITL 12.88 ms   (FULL cudagraphs)   = 5.52x
#   output verified correct in both ("17*23" -> 391, finish_reason stop)
#
# Idempotent. No-op if already patched or if the anchor is absent.
set -euo pipefail
PY=${PYTHON:-python3}

TARGET=$($PY - <<'EOF'
import os
try:
    import vllm.v1.attention.backends.mla.triton_mla as m
    print(os.path.abspath(m.__file__))
except Exception:
    print("")
EOF
)
if [ -z "$TARGET" ] || [ ! -f "$TARGET" ]; then
    echo "[triton-mla-fix] triton_mla.py not found; nothing to do."
    exit 0
fi
if grep -q "AttentionCGSupport.UNIFORM_BATCH" "$TARGET"; then
    echo "[triton-mla-fix] already patched: $TARGET"
    exit 0
fi
cp -n "$TARGET" "$TARGET.orig" || true
$PY - "$TARGET" <<'EOF'
import sys, io
path = sys.argv[1]
src = io.open(path, encoding="utf-8").read()
old = """    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )"""
new = """    # PATCHED: UNIFORM_SINGLE_TOKEN_DECODE forced a PIECEWISE downgrade under
    # spec-decode, which silently made the DSpark drafter fully eager.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH"""
if src.count(old) != 1:
    sys.stderr.write("[triton-mla-fix] anchor missing or not unique; aborting.\n")
    sys.exit(1)
io.open(path, "w", encoding="utf-8").write(src.replace(old, new))
print("[triton-mla-fix] patched", path)
EOF
echo "[triton-mla-fix] done."
