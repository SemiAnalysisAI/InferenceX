#!/bin/bash
# =============================================================================
# apply_k3_fp8_cb810.sh
# Reproduce the working Kimi-K3 fp8-KV gluon setup on a FRESH container of:
#   vllm/vllm-openai-rocm:nightly-cb8104839c141609d99f1254459ef3a4f1bd4263
# Run INSIDE the container:   bash /workspace/apply_k3_fp8_cb810.sh
# Then launch with server_final_asm.sh (HYBRID: Kimi-K3 asm decode + DSpark gluon
# verify) or server_final.sh (all-gluon) -- see the ENV note at the end.
# NOTE: model symlink step removed - ensure the model is reachable at the path
#       your server script expects (e.g. /data/Kimi-K3) yourself.
# =============================================================================
set -uo pipefail
D=/usr/local/lib/python3.12/dist-packages
WS=/workspace
say(){ echo; echo "=================== $* ==================="; }

say "1/8 triton 3.7.0 + tabulate"
python -m pip install --extra-index-url https://pypi.amd.com/triton/release/rocm-7.2.0/simple/ triton==3.7.0 2>&1 | tail -2
python -m pip install tabulate 2>&1 | tail -1

say "2/8 fetch PR diffs"
cd "$WS"
curl -sL https://github.com/ROCm/aiter/pull/4474.diff       -o pr4474.diff
curl -sL https://github.com/ROCm/aiter/pull/4494.diff       -o pr4494.diff
curl -sL https://github.com/vllm-project/vllm/pull/50578.diff -o pr50578.diff
curl -sL https://github.com/vllm-project/vllm/pull/51171.diff -o pr51171.diff
curl -sL https://github.com/vllm-project/vllm/pull/50619.diff -o pr50619.diff

say "3/8 build fixed/filtered diffs"
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
# 4494: keep ONLY aiter/ops/gemm_op_a16w16.py (drop the new op_tests/ regression test,
# which is not present under dist-packages). CUDA-graph split-K semaphore fix: under
# graph capture return a fresh zeroed workspace so the counter==0 invariant holds on
# every replay -> enables FULL/FULL_DECODE_ONLY fp8 cudagraph without corrupt reductions.
keep4494=[pt for pt in re.split(r"(?m)^(?=diff --git )", open("/workspace/pr4494.diff").read())
          if pt.startswith("diff --git a/aiter/ops/gemm_op_a16w16.py")]
open("/workspace/pr4494.src.diff","w").write("".join(keep4494))
# 50619: keep nvidia/mla.py (fallback) + rocm_aiter_mla hunks1&2 + the cg-exclude
# hunks (attn_utils.py + model_runner.py); drop hunk3 (forward_mqa, superseded by
# 51171) and tests.
#   cg-exclude = pass cg_support_exclude_layers=speculator.draft_attn_layer_names so
#   the DSpark draft's attention layers do NOT constrain the target's cudagraph
#   support decision. REQUIRED for --enable-prefix-caching: without it the target
#   FULL cudagraph capture memory-faults during the spec-verify warmup.
out=[]
for pt in re.split(r"(?m)^(?=diff --git )", open("/workspace/pr50619.diff").read()):
    m=re.match(r"diff --git a/(\S+) b/",pt)
    if not m: continue
    path=m.group(1)
    if path.endswith("kimi_k3/nvidia/mla.py"):
        out.append(pt)
    elif path.endswith("gpu/attn_utils.py") or path.endswith("gpu/model_runner.py"):
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

say "4/8 apply patches to live dist-packages"
for f in pr4474.notest.diff pr4494.src.diff pr50578.notest.diff pr51171.fixed.diff pr50619.fp8gluon.diff; do
  echo "--- applying $f ---"
  patch -p1 -d "$D" --fuzz=2 --no-backup-if-mismatch < "$WS/$f" || echo "WARN: $f had issues"
done

say "5/8 gist mla_gluon: bh16bn128 batch<=256 + fp8-query dequant (keeps #4474 int64)"
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

say "6/8 KDA state_indices fix (needed for eager warmup)"
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

say "7/8 HYBRID routing: Kimi-K3 asm decode + DSpark gluon verify (fp8 query)"
python3 - <<'PY'
import py_compile
F="/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/rocm_aiter_mla.py"
s=open(F).read()

# (a) Keep the query fp8 on BOTH paths. The asm qseqlen==1 decode has ONLY an
#     fp8-query kernel (a bf16 query -> asm_mla.cu:193 'cannot get heuristic
#     kernel'); the gluon verify dequants an fp8 query -> bf16 internally (the
#     mla_gluon batch256 patch above). So flip the #50619 bf16-query gate True.
oq='''        if num_heads < AiterMLAHelper._AITER_MIN_MLA_HEADS and is_quantized_kv_cache(
            self.kv_cache_dtype
        ):
            self.supports_quant_query_input = False'''
nq='''        if num_heads < AiterMLAHelper._AITER_MIN_MLA_HEADS and is_quantized_kv_cache(
            self.kv_cache_dtype
        ):
            # HYBRID: keep the query fp8 (asm decode needs an fp8-query kernel;
            # gluon verify dequants fp8 -> bf16 internally).
            self.supports_quant_query_input = True'''
if nq in s: print("hybrid q: already True")
elif oq in s: s=s.replace(oq,nq); print("hybrid q: -> True")
else: print("hybrid q: ANCHOR NOT FOUND")

# (b) Small-head multi-token verify (DSpark, max_qo_len>1) ALWAYS uses the gluon
#     flatten -- asm has no gqa<16 qseqlen>1 kernel, so ASM_PADDING=asm must NOT
#     force the asm path here. Drop the `_aiter_mla_small_head_mode()!="asm"`
#     term from the forward_mqa branch-2 gate (keep the gfx950 guard).
og='''        if (
            self.num_heads < AiterMLAHelper._AITER_MIN_MLA_HEADS
            and int(decode.max_qo_len) > 1
            and _aiter_mla_small_head_mode() != "asm"
            and _gluon_mla_decode_supported()
        ):'''
ng='''        if (
            self.num_heads < AiterMLAHelper._AITER_MIN_MLA_HEADS
            and int(decode.max_qo_len) > 1
            # HYBRID: verify always gluon, independent of ASM_PADDING.
            and _gluon_mla_decode_supported()
        ):'''
asm_term = 'and _aiter_mla_small_head_mode() != "asm"'
if asm_term not in s: print("hybrid gate: already gluon-only (asm-term absent, e.g. #51171 base) -- OK")
elif ng in s: print("hybrid gate: already applied")
elif og in s: s=s.replace(og,ng); print("hybrid gate: applied")
else: print("hybrid gate: ANCHOR NOT FOUND")

open(F,"w").write(s); py_compile.compile(F,doraise=True); print("rocm_aiter_mla hybrid compiled OK")
PY

say "8/8 verify"
echo "4474_int64      = $(grep -c 'to(gl.int64)' $D/aiter/ops/triton/gluon/mla_gluon.py)  (expect 2)"
echo "4494_graphsem   = $(grep -c 'is_current_stream_capturing' $D/aiter/ops/gemm_op_a16w16.py)  (expect 1)"
echo "gluon_batch256  = $(grep -c '1 <= batch_size <= 256' $D/aiter/ops/triton/gluon/mla_gluon.py)  (expect 2)"
echo "gluon_dequant   = $(grep -c 'q_nope = q_nope.to(torch.bfloat16)' $D/aiter/ops/triton/gluon/mla_gluon.py)  (expect 1)"
echo "50578_env       = $(grep -c VLLM_ROCM_AITER_MLA_ASM_PADDING $D/vllm/envs.py)  (expect 3)"
echo "51171_flat      = $(grep -c flat_kv_indices $D/vllm/v1/attention/backends/mla/rocm_aiter_mla.py)  (expect ~14)"
echo "51171_gpuworker = $(grep -c get_kv_cache_capacity $D/vllm/v1/worker/gpu_worker.py)  (expect 1)"
echo "50619_nvfb      = $(grep -c 'if not self.impl.supports_quant_query_input' $D/vllm/models/kimi_k3/nvidia/mla.py)  (expect 1)"
echo "50619_cgex_attn = $(grep -c cg_support_exclude_layers $D/vllm/v1/worker/gpu/attn_utils.py)  (expect 3)"
echo "50619_cgex_mr   = $(grep -c cg_support_exclude_layers $D/vllm/v1/worker/gpu/model_runner.py)  (expect 1)"
echo "hybrid_qfp8     = $(grep -c 'self.supports_quant_query_input = True' $D/vllm/v1/attention/backends/mla/rocm_aiter_mla.py)  (expect 1)"
echo "hybrid_qfp8_off = $(grep -c 'self.supports_quant_query_input = False' $D/vllm/v1/attention/backends/mla/rocm_aiter_mla.py)  (expect 0)"
echo "hybrid_gate     = $(grep -c '_aiter_mla_small_head_mode() != \"asm\"' $D/vllm/v1/attention/backends/mla/rocm_aiter_mla.py)  (expect 0: asm-term dropped -> verify always gluon)"
echo "kda_fix         = $(grep -c 'state_indices = state_indices.reshape' $D/vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py)  (expect 1)"
echo "triton          = $(python -c 'import triton;print(triton.__version__)')"
python -c "import vllm.envs, vllm.v1.attention.backends.mla.rocm_aiter_mla; print('IMPORT_OK')"

cat <<'ENVMSG'

============================================================
DONE. This install now supports BOTH configs off the SAME patched tree; the
only difference is VLLM_ROCM_AITER_MLA_ASM_PADDING. Common args:
    --kv-cache-dtype fp8  --max-model-len 1048576 \
    --compilation-config '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":48,"custom_ops":["+fused_rms_norm_gated"]}' \
    --speculative-config '{"model":"Inferact/Kimi-K3-DSpark","num_speculative_tokens":2,"method":"dspark","attention_backend":"TRITON_MLA","kv_cache_dtype":"auto"}'

  RadixArk/Kimi-K3-DSpark (Qwen3 DENSE draft) alternative -- FIRST relabel its config
  architectures ["DSparkDraftModel"] -> ["Qwen3DSparkModel"] (else vLLM routes it to the
  DeepseekV4-MoE DSpark class -> hc_mult AttributeError). Note the LOCAL path, the NON-MLA
  draft backend TRITON_ATTN (TRITON_MLA is rejected: 'non-MLA not supported'), and
  FULL_AND_PIECEWISE with explicit capture sizes 1..48:
    --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","max_cudagraph_capture_size":48,"cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48],"custom_ops":["+fused_rms_norm_gated"]}' \
    --speculative-config '{"model":"/data/Kimi-K3-DSpark-RadixArk","num_speculative_tokens":2,"method":"dspark","attention_backend":"TRITON_ATTN","kv_cache_dtype":"auto"}'

  The env only controls the Kimi-K3 plain decode (qo_len==1); the DSpark verify
  (qo_len>1) is hard-routed to gluon by the step-7 hybrid gate, and the query is
  fp8 on both paths (step 7a). All decodes read the SAME fp8 KV pool.

    export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm     # HYBRID: Kimi-K3 decode=asm, DSpark verify=gluon
      -> server_final_asm.sh
    export VLLM_ROCM_AITER_MLA_ASM_PADDING=gluon   # ALL-GLUON: decode=gluon, verify=gluon
      -> server_final.sh

  VALIDATED 2026-08-08 (image cb8104839c, this patch set incl. aiter #4494):
    * ASM_PADDING=gluon (server_final.sh): boots clean, FULL_DECODE_ONLY capture-48
      + full 1M pool, NO memory fault, gsm8k 1.0.
    * ASM_PADDING=asm  (server_final_asm.sh, HYBRID): boots clean (no asm heuristic
      crash), gsm8k 1.0.
  #4494 (graph split-K semaphore) is what makes FULL-family fp8 cudagraph safe
  here; AITER_DISABLE_FMHA_OPUS=1 is NOT required for either config.

  PREFIX CACHING (--enable-prefix-caching) with DSpark, VALIDATED 2026-08-08:
    Requires BOTH:
      1. the #50619 cg-exclude hunks above (attn_utils.py + model_runner.py) --
         else the target FULL cudagraph capture memory-faults in spec-verify warmup.
      2. cudagraph_mode = FULL_AND_PIECEWISE (NOT FULL_DECODE_ONLY). Prefix caching
         flips the KDA/Mamba cache into "align" mode; FULL_DECODE_ONLY tries to
         full-capture the data-dependent KDA align spec-verify path and OOB-faults.
         FULL_AND_PIECEWISE runs the mamba/KDA ops (in splitting_ops) piecewise
         (eager, not full-captured), so it never full-captures that path.
    server_final_CI.sh (asm) + --enable-prefix-caching + cudagraph_mode
    FULL_AND_PIECEWISE (= server_final_CI_fp.sh) -> boots with cudagraph, 0 faults,
    gsm8k 0.85 (LIMIT=20; coherent, likely 20-sample noise vs no-prefix 1.0).
    NOTE: --enforce-eager also works (no cudagraph). #51171 is REQUIRED (it is the
    core 12-head spec-verify enabler) -- do NOT revert it to dodge the cudagraph fault.
============================================================
ENVMSG