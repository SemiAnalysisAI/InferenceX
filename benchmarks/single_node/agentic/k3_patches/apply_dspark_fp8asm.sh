#!/usr/bin/env bash
# =============================================================================
# apply_dspark_fp8asm.sh  (portable — no bind mounts / host paths)
# DSpark + native fp8-asm KV enablement layer for Kimi-K3 on MI355X. Applied by
# apply_k3_fp4_fp8asm_dspark_patches.sh after the aiter rebuild. On vLLM 0.27 the
# former "5 base ASM patches" AND the DSpark rocm_aiter_mla.py surgery (verify
# width, small-head mode, gluon-verify gate, persistent-metadata gate) are all
# UPSTREAMED/native, so steps 0/1/4/4b/4c/5 are native no-ops; only the aiter-side
# get_block key (3/7), the DSpark draft-causal json (2/7) and the KDA stride fix
# (6/7) still apply. See apply_k3_fp4_fp8asm_dspark_patches.sh VERIFY for anchors.
#
# Idempotent: every step guards on its anchor / target text and takes a .bak.
# The sibling patch_*.py scripts are resolved relative to this file's directory.
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D=/usr/local/lib/python3.12/dist-packages
R="$D/vllm/v1/attention/backends/mla/rocm_aiter_mla.py"
# RESEED_SHIPPED_MLA is off by default; only set SHIPPED_MLA_REF if you reseed.
SHIPPED_MLA_REF="${SHIPPED_MLA_REF:-}"
if [ "${RESEED_SHIPPED_MLA:-0}" = "1" ] && [ -n "$SHIPPED_MLA_REF" ] && [ -f "$SHIPPED_MLA_REF" ]; then
  cp -a "$R" "$R.pre_reseed.bak" 2>/dev/null || true
  cp "$SHIPPED_MLA_REF" "$R"
  echo "reseeded rocm_aiter_mla.py from $SHIPPED_MLA_REF"
fi
if [ -f /opt/aiter-local/aiter/mla.py ]; then
  AITER=/opt/aiter-local/aiter/mla.py
elif [ -f /root/aiter/aiter/mla.py ]; then
  AITER=/root/aiter/aiter/mla.py
else
  echo "!! aiter mla.py not found under /opt/aiter-local or /root/aiter — run setup first"
  exit 1
fi
KDA="$D/vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py"
export R AITER KDA SHIPPED_MLA_REF
say(){ echo; echo "=================== $* ==================="; }

say "0/7 InferenceX fp8-asm base patches — NATIVE on vLLM 0.27, nothing to apply"
cd "$SCRIPT_DIR"
# The 5 base ASM patches are upstreamed in vLLM 0.27:
#   fp8-asm decode pad / skip-K3-fp8-PS / fp8-prefill-pad -> native strict gate
#     `self.num_heads % 16 == 0` ($R L375 + L990) + get_mla_padded_q (L873).
#   PS metadata16 -> `self._num_attention_heads = max(16, self.num_heads)` (L314).
#   wvSplitK #50618 -> native `.contiguous()` skinny x_view (utils.py L163/L188).
echo "  native — fp8-asm base patches upstreamed in 0.27"

say "1/7 sanity: 0.27 aiter-MLA backend present (native _mtp_decode_qlen + reorder_batch_threshold)"
# 0.27 sets _mtp_decode_qlen = self.reorder_batch_threshold or 1 natively (L276);
# both markers present confirms this is the expected aiter-MLA backend file.
grep -q _mtp_decode_qlen "$R" && grep -q 'reorder_batch_threshold' "$R" \
  || { echo "!! $R is not the expected 0.27 aiter-MLA backend"; exit 1; }

say "2/7 force DSpark draft causal (dflash_config.causal=true)"
DCACHE=/dev/shm/hf-cache/models--Inferact--Kimi-K3-DSpark/snapshots
CFG="$(ls -d "$DCACHE"/*/ 2>/dev/null | head -1)config.json"
if [ -f "$CFG" ]; then
  cp -n "$CFG" "$CFG.orig.bak"
  python3 - "$CFG" <<'PY'
import json, sys
f = sys.argv[1]
c = json.load(open(f))
d = c.setdefault("dflash_config", {})
if d.get("causal") is True:
    print("  draft already forced causal")
else:
    d["causal"] = True
    json.dump(c, open(f, "w"), indent=2)
    print("  forced causal:", f)
PY
else
  echo "  !! draft config not found at $DCACHE — stage the draft or set it manually"
fi

say "3/7 aiter get_block_n_fp8 — add DSpark verify-width key (16*5=80)"
python3 - <<'PY'
import re, py_compile, shutil, os
F = os.environ.get("AITER", "/root/aiter/aiter/mla.py")
s = open(F).read()
shutil.copy2(F, F + ".pre_dspark.bak")
if "80: 64" not in s:
    s = re.sub(r"(get_block_n_fp8\s*=\s*\{)", r"\g<1>\n    80: 64, 96: 64, 112: 64,", s, count=1)
s = s.replace(
    "min_block_n = get_block_n_fp8[int(nhead * max_seqlen_q)]",
    "min_block_n = get_block_n_fp8.get(int(nhead * max_seqlen_q), 64)",
)
open(F, "w").write(s); py_compile.compile(F, doraise=True)
print("  80-key:", "80: 64" in s, " get():", "get_block_n_fp8.get(" in s)
PY

say "4/7 DSpark verify width — NATIVE on 0.27 (reorder_batch_threshold), nothing to inject"
# method 'dspark' sets speculative_config.parallel_drafting=True
# (config/speculative.py L1046-1047), so _init_reorder_batch_threshold
# (v1/attention/backend.py L734-742) sets
#   reorder_batch_threshold = max(1, 1 + 2*num_speculative_tokens)  # =5 @num_spec=2
# and the builder sets self._mtp_decode_qlen = self.reorder_batch_threshold or 1
# ($R L276) == the old injected 2*num_spec+1. No-op on 0.27.
# (Requires DCP off: backend.py L744-746 forces reorder_batch_threshold=1 when
# decode_context_parallel_size>1 — K3 TP8 runs DCP=1.)
echo "  native — verify width = 1 + 2*num_spec via reorder_batch_threshold"

say "4b/7 _aiter_mla_small_head_mode — NATIVE on 0.27, nothing to do"
# 0.27 ships the full helper reading vllm.envs (rocm_aiter_mla.py L97-121), and
# vllm/envs.py L143 declares VLLM_ROCM_AITER_MLA_ASM_PADDING
# (Literal["auto","gluon","asm"]="auto"), so the env->os.environ fallback and
# the shipped-ref restore are both moot.
echo "  native — _aiter_mla_small_head_mode reads envs.VLLM_ROCM_AITER_MLA_ASM_PADDING"

say "4c/7 skip gluon verify-flatten under ASM_PADDING=asm — NATIVE on 0.27"
# 0.27 folds this into AiterMLAHelper.use_gluon_verify (rocm_aiter_mla.py
# L926-941): returns False when _aiter_mla_small_head_mode()=='asm', and
# forward_mqa gates the verify-flatten branch on use_gluon_verify() (L1230) — so
# with VLLM_ROCM_AITER_MLA_ASM_PADDING=asm the 12-head verify stays on the padded
# asm path, exactly what the old 4c edit forced.
echo "  native — use_gluon_verify()==False under asm; verify stays on asm path"

say "5/7 persistent-metadata gate — NATIVE on 0.27, nothing to broaden"
# 0.27 rebuilt the gate in _build_decode (rocm_aiter_mla.py L697-715):
#   use_persistent_metadata = (
#       not use_gluon_decode(...) and not use_gluon_verify(...)
#       and (num_heads>=16 or max_qo_len<=_ASM_PADDED_MAX_PS_QLEN
#            or is_quantized_kv_cache(kv_dtype))
#       and 1 <= max_qo_len <= self._mtp_decode_qlen )
# For K3 fp8-asm KV the is_quantized_kv_cache clause keeps persistent metadata
# for the 5-wide verify (max_qo_len=5 > _ASM_PADDED_MAX_PS_QLEN=4); asm mode
# forces both gluon predicates False. Subsumes the old uses_asm_decode broadening.
echo "  native — persistent-metadata gate covers asm decode + 5-wide verify"

say "6/7 KDA stride fix (PR #27) — NATIVE on 0.27, nothing to apply"
# vLLM 0.27 carries the KDA per-sequence state-index stride fix natively, under
# refactored names:
#   - fwd (spec-decode) path uses `stride_indices_seq`: indexed loads
#     `state_indices + i_n * stride_indices_seq` ($KDA fwd kernel) + launch
#     `stride_indices_seq=ssm_state_indices.stride(0)`; 2D spec state_indices
#     handled (ndim in (1,2), stride(1)==1).
#   - packed_decode path RENAMED it to `stride_state_indices`: indexed load
#     `state_indices + i_n * stride_state_indices` + launch
#     `stride_state_indices=state_indices.stride(0)`.
#   - the old unit-stride/contiguity requirement is already dropped
#     (`state_indices must be one-dimensional`).
# The old bug pattern `tl.load(state_indices + i_n)` (no stride) is GONE. VERIFY
# checks both names.
if [ -f "$KDA" ]; then python3 -c "import py_compile,os;py_compile.compile(os.environ['KDA'],doraise=True)"; fi
echo "  native — KDA seq-stride fix present (fwd: stride_indices_seq, packed_decode: stride_state_indices)"

say "7/7 verify"
echo "aiter 80-key   = $(grep -c '80: 64' "$AITER")                (expect >=1)"
echo "aiter get()    = $(grep -c 'get_block_n_fp8.get(' "$AITER")   (expect 1)"
echo "dspark qlen    = $(grep -c '_mtp_decode_qlen = self.reorder_batch_threshold' "$R")  (expect >=1, native)"
echo "asm verify gate= $(grep -c 'use_gluon_verify' "$R")           (expect >=2, native)"
echo "kda fwd stride = $(grep -c 'stride_indices_seq' "$KDA")        (expect >=4, native)"
echo "kda pdec stride= $(grep -c 'stride_state_indices' "$KDA")      (expect >=3, native)"
python -c "import vllm.v1.attention.backends.mla.rocm_aiter_mla; print('IMPORT_OK')"
echo
echo "DONE. Serve with:"
echo "  export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm"
echo "  NUM_SPEC=2 PORT=8890 GPU_MEM=0.88 MAX_NUM_SEQS=16 bash _serve_k3_bench_spec.sh"
