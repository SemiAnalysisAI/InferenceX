#!/usr/bin/env bash
# ============================================================================
# apply_k3_cb8104839c_gluondspark_patches.sh
#
# Reproduces ALL in-container filesystem changes for the Kimi-K3 "gluondspark"
# PIECEWISE-cudagraph stack (bf16 KV, Gluon MLA decode, DSpark MTP) on image:
#   vllm/vllm-openai-rocm:nightly-cb8104839c141609d99f1254459ef3a4f1bd4263
#
# This is the PIECEWISE / bf16-KV stack, NOT the fp8-asm stack. The fp8 stack
# (PR #50619 + bh16bn128 batch-relax) is in apply_k3_cb8104839c_patches.sh.
#
# Changes applied (idempotent):
#   1. KDA fused_recurrent state_indices coercion - the packed-decode KDA
#      kernel rejected a non-contiguous / [B,1] state_indices, which PIECEWISE
#      cudagraph passes (FULL modes hand it a 1-D contiguous tensor). Flatten
#      to contiguous 1-D instead of raising, so PIECEWISE boots.
#   2. aiter PR #4474 - int64 KV stride widening on the >2GB global_load path
#      of _mla_gluon (fixes the concurrency-16 mixed-batch MTP-verify GPU
#      memory-access fault on >2GB KV caches).
#   3. Triton 3.7.0 (AMD ROCm 7.2.0 build) + tabulate (Gluon MLA needs
#      PaddedSharedLayout(cga_layout=...)).
#
# Run INSIDE a fresh container of the image above:
#   docker exec -i <container> bash < apply_k3_cb8104839c_gluondspark_patches.sh
#
# Server-side runtime config for this stack (NOT container changes):
#   --kv-cache-dtype auto (bf16 KV), TP8, --gpu-memory-utilization 0.9,
#   --no-enable-prefix-caching,
#   --speculative-config num_speculative_tokens=2 attention_backend=TRITON_MLA,
#   VLLM_USE_BREAKABLE_CUDAGRAPH=1,
#   --compilation-config '{"max_cudagraph_capture_size":16,
#                          "cudagraph_mode":"PIECEWISE",
#                          "custom_ops":["+fused_rms_norm_gated"]}'.
# ============================================================================
set -euo pipefail

VLLM_DIR="$(python -c 'import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))')"
AITER_DIR="$(python -c 'import aiter, os; print(os.path.dirname(os.path.dirname(aiter.__file__)))')"
echo "[patch] VLLM_DIR=$VLLM_DIR"
echo "[patch] AITER_DIR=$AITER_DIR"

apply_idem() {  # $1=dir  $2=diff  $3=label
  cd "$1"
  if git apply --reverse --check -p1 "$2" 2>/dev/null; then
    echo "[patch] $3: already applied, skipping"
  else
    git apply --check -p1 "$2"
    git apply -p1 "$2"
    echo "[patch] $3: applied"
  fi
}

# ---- 1. KDA state_indices coercion (PIECEWISE cudagraph support) ----
cat > /tmp/kda_state_indices.diff <<'KDA_EOF'
diff --git a/vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py b/vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py
--- a/vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py
+++ b/vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py
@@ -560,8 +560,12 @@ def fused_recurrent_kda_packed_decode(
     if initial_state.stride()[1:] != (V * K, K, 1):
         raise ValueError("`initial_state` must be contiguous within each cache slot.")
-    if state_indices.ndim != 1 or state_indices.stride(0) != 1:
-        raise ValueError("`state_indices` must be contiguous and one-dimensional.")
+    if state_indices.ndim != 1 or state_indices.stride(0) != 1:
+        # PIECEWISE cudagraph can pass a non-contiguous / [B, 1] state_indices
+        # (FULL cudagraph modes prepare a 1-D contiguous one). The packed-decode
+        # kernel indexes state_indices[i_n] once per sequence, so flattening to a
+        # contiguous 1-D tensor is equivalent to the FULL-mode input.
+        state_indices = state_indices.reshape(-1).contiguous()
     if A_log.ndim != 1 or not A_log.is_contiguous():
         raise ValueError("`A_log` must be contiguous and one-dimensional.")
     if not dt_bias.is_contiguous():
KDA_EOF

# ---- 2. aiter PR #4474 (int64 KV stride on >2GB global_load path) ----
cat > /tmp/aiter_pr4474.diff <<'PR4474_EOF'
diff --git a/aiter/ops/triton/gluon/mla_gluon.py b/aiter/ops/triton/gluon/mla_gluon.py
index 64dc4bec418..d36f1fca3b7 100644
--- a/aiter/ops/triton/gluon/mla_gluon.py
+++ b/aiter/ops/triton/gluon/mla_gluon.py
@@ -164,6 +164,11 @@ def _mla_gluon(
     num_iter = gl.cdiv(split_kv_end - split_kv_start, BLOCK_N)
     start_n = split_kv_start

+    # >2GB KV cache (global_load path): widen strides to int64 so kv offsets don't overflow int32.
+    if not WITHIN_2GB:
+        stride_kv_c_bs = stride_kv_c_bs.to(gl.int64)
+        stride_k_pe_bs = stride_k_pe_bs.to(gl.int64)
+
     # early return with empty kv slice to save compute
     if split_kv_start >= split_kv_end:
         return
PR4474_EOF

# ---- apply source patches ----
apply_idem "$VLLM_DIR"  /tmp/kda_state_indices.diff "KDA state_indices PIECEWISE coercion"
apply_idem "$AITER_DIR" /tmp/aiter_pr4474.diff      "aiter PR #4474 (int64 KV stride)"

# ---- 3. Triton 3.7.0 + tabulate ----
TRITON_ROCM_VERSION="${TRITON_ROCM_VERSION:-7.2.0}"
TRITON_VERSION="${TRITON_VERSION:-3.7.0}"
echo "[patch] current triton: $(python -c 'import triton; print(triton.__version__)' 2>/dev/null || echo none)"
python -m pip install --extra-index-url "https://pypi.amd.com/triton/release/rocm-${TRITON_ROCM_VERSION}/simple/" "triton==${TRITON_VERSION}"
python -m pip install tabulate

# ---- verify + import sanity ----
python - <<'PY'
import inspect
import vllm, aiter, triton
import aiter.ops.triton.gluon.mla_gluon
from vllm.models.kimi_k3.amd.ops.third_party.kda import fused_recurrent
from triton.experimental.gluon.language import PaddedSharedLayout
src = inspect.getsource(fused_recurrent.fused_recurrent_kda_packed_decode)
assert "reshape(-1).contiguous()" in src, "KDA state_indices coercion not applied"
assert "cga_layout" in inspect.signature(PaddedSharedLayout.__init__).parameters, \
    f"triton {triton.__version__} lacks PaddedSharedLayout(cga_layout=...)"
print("[patch] IMPORT_OK  vllm", vllm.__version__, " triton", triton.__version__)
PY
echo "[patch] DONE"