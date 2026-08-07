#!/bin/bash
# =============================================================================
# apply_k3_fp8_cb810.sh
# Reproduce the working Kimi-K3 fp8-KV gluon setup on a FRESH container of:
#   vllm/vllm-openai-rocm:nightly-cb8104839c141609d99f1254459ef3a4f1bd4263
# Run INSIDE the container:   bash /workspace/apply_k3_fp8_cb810.sh
# Then launch with server_gluon_piecewise_nomode3.sh (or any fp8 gluon script)
# with the ENV shown at the end.
# NOTE: model symlink step removed - ensure the model is reachable at the path
#       your server script expects (e.g. /data/Kimi-K3) yourself.
# =============================================================================
set -uo pipefail
D=/usr/local/lib/python3.12/dist-packages
WS=/workspace
say(){ echo; echo "=================== $* ==================="; }

say "1/7 triton 3.7.0 + tabulate"
python -m pip install --extra-index-url https://pypi.amd.com/triton/release/rocm-7.2.0/simple/ triton==3.7.0 2>&1 | tail -2
python -m pip install tabulate 2>&1 | tail -1

say "2/7 fetch PR diffs"
cd "$WS"
curl -sL https://github.com/ROCm/aiter/pull/4474.diff       -o pr4474.diff
curl -sL https://github.com/vllm-project/vllm/pull/50578.diff -o pr50578.diff
curl -sL https://github.com/vllm-project/vllm/pull/51171.diff -o pr51171.diff
curl -sL https://github.com/vllm-project/vllm/pull/50619.diff -o pr50619.diff

say "3/7 build fixed/filtered diffs"
python3 - <<'PY'
import re
def strip_tests(src):
    out=[]
    for pt in re.split(r"(?m)^(?=diff --git )", src):
        m=re.match(r"diff --git a/(\S+) b/",pt)
        if m and m.group(1).startswith("tests/"): continue
        out.append(pt)
    return "".join(out)
# 51171: rewrite gpu_worker hunk#1 context for cb810 base (no set_torch_threads_for_runtime in import)
p=open("/workspace/pr51171.diff").read()
old='''@@ -64,6 +64,7 @@
 from vllm.utils.mem_constants import GiB_bytes
 from vllm.utils.mem_utils import MemorySnapshot, format_gib, memory_profiling
 from vllm.utils.torch_utils import set_random_seed, set_torch_threads_for_runtime
+from vllm.v1.core.kv_cache_utils import get_kv_cache_capacity
 from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
 from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec'''
new='''@@ -66,2 +66,3 @@
 from vllm.utils.torch_utils import set_random_seed
+from vllm.v1.core.kv_cache_utils import get_kv_cache_capacity
 from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput'''
if old in p: p=p.replace(old,new)
open("/workspace/pr51171.fixed.diff","w").write(strip_tests(p))
open("/workspace/pr50578.notest.diff","w").write(strip_tests(open("/workspace/pr50578.diff").read()))
open("/workspace/pr4474.notest.diff","w").write(strip_tests(open("/workspace/pr4474.diff").read()))
# 50619: keep nvidia/mla.py (fallback) + rocm_aiter_mla hunks1&2; drop hunk3(superseded by 51171), wiring, tests
out=[]
for pt in re.split(r"(?m)^(?=diff --git )", open("/workspace/pr50619.diff").read()):
    m=re.match(r"diff --git a/(\S+) b/",pt)
    if not m: continue
    path=m.group(1)
    if path.endswith("kimi_k3/nvidia/mla.py"):
        out.append(pt)
    elif path.endswith("mla/rocm_aiter_mla.py"):
        lines=pt.splitlines(keepends=True); hdr=[]; i=0
        while i<len(lines) and not lines[i].startswith("@@"): hdr.append(lines[i]); i+=1
        hunks=[]; cur=None
        for j in range(i,len(lines)):
            if lines[j].startswith("@@"):
                if cur: hunks.append(cur)
                cur=[lines[j]]
            elif cur: cur.append(lines[j])
        if cur: hunks.append(cur)
        kept=[h for h in hunks if not h[0].startswith("@@ -1073,118")]
        out.append("".join(hdr)+"".join("".join(h) for h in kept))
open("/workspace/pr50619.fp8gluon.diff","w").write("".join(out))
print("diffs built")
PY

say "4/7 apply patches to live dist-packages"
for f in pr4474.notest.diff pr50578.notest.diff pr51171.fixed.diff pr50619.fp8gluon.diff; do
  echo "--- applying $f ---"
  patch -p1 -d "$D" --fuzz=2 --no-backup-if-mismatch < "$WS/$f" || echo "WARN: $f had issues"
done

say "5/7 gist mla_gluon: bh16bn128 batch<=256 + fp8-query dequant (keeps #4474 int64)"
python3 - <<'PY'
import py_compile
F="/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/gluon/mla_gluon.py"
s=open(F).read()
a='''        use_2d_view = False

    assert (
        arch_info.get_arch() == "gfx950"'''
b='''        use_2d_view = False

    if q_nope.dtype == torch.float8_e4m3fn:
        q_nope = q_nope.to(torch.bfloat16)
    if q_pe is not None and q_pe.dtype == torch.float8_e4m3fn:
        q_pe = q_pe.to(torch.bfloat16)

    assert (
        arch_info.get_arch() == "gfx950"'''
if a in s and "q_nope = q_nope.to(torch.bfloat16)" not in s: s=s.replace(a,b)
o='''            assert (
                batch_size == 1
            ), f"mla_gluon[bh16bn128] requires batch_size=1, got {batch_size}"
            NUM_KV_SPLITS = max(1, min(256 // (batch_size * qlen), min_kv_seq_len))'''
n='''            assert (
                1 <= batch_size <= 256
            ), f"mla_gluon[bh16bn128] requires 1 <= batch_size <= 256, got {batch_size}"
            NUM_KV_SPLITS = max(
                1, min(256 // (batch_size * qlen), triton.cdiv(min_kv_seq_len, BLOCK_N))
            )'''
if o in s: s=s.replace(o,n)
open(F,"w").write(s); py_compile.compile(F,doraise=True); print("mla_gluon patched")
PY

say "6/7 KDA state_indices fix (needed for eager warmup)"
python3 - <<'PY'
import py_compile
F="/usr/local/lib/python3.12/dist-packages/vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py"
s=open(F).read()
o='''    if state_indices.ndim != 1 or state_indices.stride(0) != 1:
        raise ValueError("`state_indices` must be contiguous and one-dimensional.")'''
n='''    if state_indices.ndim != 1 or state_indices.stride(0) != 1:
        state_indices = state_indices.reshape(-1).contiguous()'''
if o in s: s=s.replace(o,n); open(F,"w").write(s); py_compile.compile(F,doraise=True); print("KDA patched")
else: print("KDA: anchor not found (already patched or base differs)")
PY

say "7/7 verify"
echo "4474_int64      = $(grep -c 'to(gl.int64)' $D/aiter/ops/triton/gluon/mla_gluon.py)  (expect 2)"
echo "gluon_batch256  = $(grep -c '1 <= batch_size <= 256' $D/aiter/ops/triton/gluon/mla_gluon.py)  (expect 2)"
echo "gluon_dequant   = $(grep -c 'q_nope = q_nope.to(torch.bfloat16)' $D/aiter/ops/triton/gluon/mla_gluon.py)  (expect 1)"
echo "50578_env       = $(grep -c VLLM_ROCM_AITER_MLA_ASM_PADDING $D/vllm/envs.py)  (expect 3)"
echo "51171_flat      = $(grep -c flat_kv_indices $D/vllm/v1/attention/backends/mla/rocm_aiter_mla.py)  (expect ~14)"
echo "51171_gpuworker = $(grep -c get_kv_cache_capacity $D/vllm/v1/worker/gpu_worker.py)  (expect 1)"
echo "50619_qfix      = $(grep -c 'supports_quant_query_input = False' $D/vllm/v1/attention/backends/mla/rocm_aiter_mla.py)  (expect 1)"
echo "50619_nvfb      = $(grep -c 'if not self.impl.supports_quant_query_input' $D/vllm/models/kimi_k3/nvidia/mla.py)  (expect 1)"
echo "kda_fix         = $(grep -c 'state_indices = state_indices.reshape' $D/vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py)  (expect 1)"
echo "triton          = $(python -c 'import triton;print(triton.__version__)')"
python -c "import vllm.envs, vllm.v1.attention.backends.mla.rocm_aiter_mla; print('IMPORT_OK')"

cat <<'ENVMSG'

============================================================
DONE. Launch the fp8 server with these ENV + args:
  export VLLM_ROCM_AITER_MLA_ASM_PADDING=gluon
  export AITER_DISABLE_FMHA_OPUS=1        # fixes fmha int32 overflow -> FULL cudagraph works
  vllm serve <MODEL_PATH> ... \
    --kv-cache-dtype fp8 \
    --max-model-len 1048576 \
    --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","max_cudagraph_capture_size":48, ...}' \
    --speculative-config '{"model":"Inferact/Kimi-K3-DSpark","num_speculative_tokens":2,"method":"dspark","attention_backend":"TRITON_MLA","kv_cache_dtype":"auto"}'
  (server_gluon_piecewise_nomode3.sh already encodes this; add AITER_DISABLE_FMHA_OPUS=1 for FULL cudagraph.)
============================================================
ENVMSG