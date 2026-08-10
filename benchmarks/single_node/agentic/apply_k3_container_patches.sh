#!/usr/bin/env bash
# =============================================================================
# apply_k3_cb8104839c_fp8_pr_stack.sh
#
# Reproduce the in-container filesystem changes for the Kimi-K3 fp8-KV
# FULL_AND_PIECEWISE cudagraph stack that booted + passed gsm8k on:
#   vllm/vllm-openai-rocm:nightly-cb8104839c141609d99f1254459ef3a4f1bd4263
#
# Run INSIDE a FRESH container of that image:
#   docker exec -i <container> bash < apply_k3_cb8104839c_fp8_pr_stack.sh
#   (or: bash /workspace/apply_k3_cb8104839c_fp8_pr_stack.sh)
#
# This is the PR-stack variant (NOT the 2026-08-08 hybrid/#50619 gist stack in
# apply_k3_fp8_cb810.sh). Image changes applied here, idempotently:
#   aiter #4474  int64 KV stride on the >2GB global_load path of _mla_gluon
#   aiter #4494  a16w16 GEMM: fresh split-K semaphore workspace under cudagraph
#                capture (stale cached ptr/counter faulted on replay)
#   vllm  #51171 Reach FULL cudagraphs for AITER MLA speculative decoding
#   vllm  #50578 Use asm decode for non-divisor small head counts (12h @ TP8)
#   vllm  #51011 Fix fp8 KV cache decode on the AITER MLA backend
#   vllm  #51040 Extend FP8 asm MLA prefill to non-divisor small head counts
#   triton 3.7.0 (AMD ROCm 7.2.0 build) + tabulate
#
# aiter patches apply clean with `git apply`; the four vllm PRs are branches off
# different bases that all edit rocm_aiter_mla.py, so they are applied in order
# with `patch --fuzz` (validated: 0 rejects, in the order below).
#
# RUNTIME NOTE (NOT an image change -- set these in your server script):
#   * Point MODEL_PATH at the model INSIDE the container. If you launch with
#     `-v /data:/model` the model is at /model/Kimi-K3 (NOT /data/Kimi-K3).
#   * fp8 PIECEWISE cudagraph capture memory-faults at capture size 45 on this
#     stack; cap it below that:  "max_cudagraph_capture_size": 44  with
#     "cudagraph_mode": "FULL_AND_PIECEWISE", "custom_ops":["+fused_rms_norm_gated"].
#   * env: VLLM_ROCM_USE_AITER=1, VLLM_ROCM_AITER_MLA_ASM_PADDING=asm,
#     VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1, --kv-cache-dtype fp8,
#     --enable-prefix-caching, DSpark spec-decode attention_backend=TRITON_MLA.
#   Validated 2026-08-10 (this image): boots with cudagraph, 0 faults,
#   gsm8k(LIMIT=20, 5-shot) = 0.90 flexible / 0.85 strict.
# =============================================================================
set -uo pipefail

D="$(python -c 'import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))')"
A="$(python -c 'import aiter, os; print(os.path.dirname(os.path.dirname(aiter.__file__)))')"
WS="${WS:-/tmp/k3_pr_stack}"
mkdir -p "$WS"
say(){ echo; echo "=================== $* ==================="; }
echo "[stack] vllm dir : $D"
echo "[stack] aiter dir: $A"

# ---------------------------------------------------------------------------
say "1/5 triton 3.7.0 + tabulate"
python -m pip install --extra-index-url https://pypi.amd.com/triton/release/rocm-7.2.0/simple/ triton==3.7.0 2>&1 | tail -2
python -m pip install tabulate 2>&1 | tail -1

# ---------------------------------------------------------------------------
say "2/5 fetch PR diffs"
cd "$WS"
fetch(){ # $1=url  $2=out
  curl -sL "$1" -o "$2"
  if ! grep -q '^diff --git ' "$2"; then
    echo "ERROR: $2 is not a diff (fetch failed / auth wall). First line:"; head -1 "$2"; exit 1
  fi
  echo "  fetched $2 ($(grep -c '^diff --git' "$2") files, $(wc -l < "$2") lines)"
}
fetch https://github.com/ROCm/aiter/pull/4474.diff        pr_aiter4474.diff
fetch https://github.com/ROCm/aiter/pull/4494.diff        pr_aiter4494.diff
fetch https://github.com/vllm-project/vllm/pull/51171.diff pr_vllm51171.diff
fetch https://github.com/vllm-project/vllm/pull/50578.diff pr_vllm50578.diff
fetch https://github.com/vllm-project/vllm/pull/51011.diff pr_vllm51011.diff
fetch https://github.com/vllm-project/vllm/pull/51040.diff pr_vllm51040.diff

# ---------------------------------------------------------------------------
say "3/5 filter diffs to installed source (drop tests/ etc.)"
python3 - "$WS" <<'PY'
import re, sys, os
ws = sys.argv[1]
def keep_prefix(src, prefix):
    out = []
    for pt in re.split(r"(?m)^(?=diff --git )", src):
        m = re.match(r"diff --git a/(\S+) b/", pt)
        if m and m.group(1).startswith(prefix):
            out.append(pt)
    return "".join(out)
jobs = [
    ("pr_aiter4474.diff", "aiter/", "pr_aiter4474.src.diff"),
    ("pr_aiter4494.diff", "aiter/", "pr_aiter4494.src.diff"),
    ("pr_vllm51171.diff", "vllm/",  "pr_vllm51171.src.diff"),
    ("pr_vllm50578.diff", "vllm/",  "pr_vllm50578.src.diff"),
    ("pr_vllm51011.diff", "vllm/",  "pr_vllm51011.src.diff"),
    ("pr_vllm51040.diff", "vllm/",  "pr_vllm51040.src.diff"),
]
for inp, pref, out in jobs:
    s = open(os.path.join(ws, inp)).read()
    f = keep_prefix(s, pref)
    open(os.path.join(ws, out), "w").write(f)
    n = f.count("diff --git ")
    print(f"  {out}: {n} file(s) kept")
PY

# ---------------------------------------------------------------------------
say "4/5 apply patches to the live install"
# Each patch is gated on a unique post-state marker so a re-run is a no-op.
# This matters because the vllm PRs are applied with `patch --fuzz`, whose
# "previously applied" detection is unreliable for pure-addition hunks (e.g.
# gpu_worker.py) and would otherwise double-apply on a second run.

apply_aiter(){ # $1=diff $2=marker_file $3=marker $4=label
  if grep -qF "$3" "$2" 2>/dev/null; then echo "  $4: already present (skip)"; return; fi
  cd "$A"
  if git apply -p1 "$1" 2>/dev/null; then echo "  $4: APPLIED (git apply)"
  else patch -p1 -d "$A" --fuzz=3 --forward --no-backup-if-mismatch < "$1" \
         && echo "  $4: APPLIED (patch)" || echo "  $4: FAILED"; fi
}
apply_vllm(){ # $1=diff $2=marker_file $3=marker $4=label
  if grep -qF "$3" "$2" 2>/dev/null; then echo "  $4: already present (skip)"; return; fi
  patch -p1 -d "$D" --fuzz=3 --forward --no-backup-if-mismatch < "$1" \
    && echo "  $4: APPLIED" || echo "  $4: WARN issues (inspect *.rej under $D)"
}

MM="$D/vllm/v1/attention/backends/mla/rocm_aiter_mla.py"
apply_aiter "$WS/pr_aiter4474.src.diff" "$A/aiter/ops/triton/gluon/mla_gluon.py" "to(gl.int64)"                 "aiter#4474"
apply_aiter "$WS/pr_aiter4494.src.diff" "$A/aiter/ops/gemm_op_a16w16.py"         "is_current_stream_capturing"  "aiter#4494"
# vllm: divergent single-file branches -> patch --fuzz in THIS exact order.
apply_vllm  "$WS/pr_vllm51171.src.diff" "$D/vllm/v1/worker/gpu_worker.py" "import get_kv_cache_capacity"                                              "vllm#51171"
apply_vllm  "$WS/pr_vllm50578.src.diff" "$D/vllm/envs.py"                 "VLLM_ROCM_AITER_MLA_ASM_PADDING"                                          "vllm#50578"
apply_vllm  "$WS/pr_vllm51011.src.diff" "$MM"                            "def use_gluon_decode(num_heads: int, max_qo_len: int, kv_cache_dtype: str)" "vllm#51011"
apply_vllm  "$WS/pr_vllm51040.src.diff" "$MM"                            "_pad16 = _real_nhead < 16"                                                "vllm#51040"
echo "  total vllm rejects: $(find "$D/vllm" -name '*.rej' 2>/dev/null | wc -l)"

# ---------------------------------------------------------------------------
say "5/5 verify markers + py_compile + import"
mm="$D/vllm/v1/attention/backends/mla/rocm_aiter_mla.py"
echo "4474_int64      = $(grep -c 'to(gl.int64)' "$A/aiter/ops/triton/gluon/mla_gluon.py")        (expect 2)"
echo "4494_graphsem   = $(grep -c 'is_current_stream_capturing' "$A/aiter/ops/gemm_op_a16w16.py") (expect 1)"
echo "51171_cgsupport = $(grep -c 'get_cudagraph_support' "$D/vllm/v1/attention/backends/mla/triton_mla.py" "$mm" | awk -F: '{s+=$2} END{print s}') (expect >=1)"
echo "51171_gpuworker = $(grep -c 'get_kv_cache_capacity' "$D/vllm/v1/worker/gpu_worker.py")       (expect 3: 1 import + 2 uses)"
echo "50578_env       = $(grep -c 'VLLM_ROCM_AITER_MLA_ASM_PADDING' "$D/vllm/envs.py")             (expect 3)"
echo "51011_gluondec  = $(grep -c 'def use_gluon_decode(num_heads: int, max_qo_len: int, kv_cache_dtype: str)' "$mm") (expect 1)"
echo "51040_pad16     = $(grep -c '_pad16 = _real_nhead < 16' "$mm")                               (expect 1)"
echo "triton          = $(python -c 'import triton; print(triton.__version__)')                    (expect 3.7.0*)"

python -m py_compile \
  "$mm" \
  "$D/vllm/v1/attention/backends/mla/triton_mla.py" \
  "$D/vllm/v1/worker/gpu_worker.py" \
  "$D/vllm/envs.py" \
  "$A/aiter/ops/triton/gluon/mla_gluon.py" \
  "$A/aiter/ops/gemm_op_a16w16.py" \
  && echo "PY_COMPILE_OK" || { echo "PY_COMPILE_FAIL"; exit 1; }

python - <<'PY'
import importlib
for m in ("vllm.envs",
          "vllm.v1.attention.backends.mla.rocm_aiter_mla",
          "vllm.v1.attention.backends.mla.triton_mla",
          "vllm.v1.worker.gpu_worker",
          "aiter.ops.gemm_op_a16w16"):
    importlib.import_module(m)
print("IMPORT_OK")
PY

echo
echo "[stack] DONE. Launch with server_final_CI.sh after setting MODEL_PATH to the"
echo "        in-container model dir and max_cudagraph_capture_size=44 (see header)."