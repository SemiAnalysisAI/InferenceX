#!/usr/bin/env bash
# =============================================================================
# apply_dspark_fp8asm.sh  (portable — no bind mounts / host paths)
# DSpark + native fp8-asm KV enablement layer for Kimi-K3 on MI355X. Applied by
# apply_k3_fp4_fp8asm_dspark_patches.sh AFTER the 5 base vLLM ASM patches and the aiter
# rebuild. Sits on the SHIPPED recipe rocm_aiter_mla.py (main + #51011 + #51040
# + #51606), which the base ASM patches install.
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

say "0/7 InferenceX fp8-asm base patches (decode, prefill, PS skip, wvSplitK)"
cd "$SCRIPT_DIR"
need_asm=0
grep -q 'PATCH(fp8-asm)' "$R" || need_asm=1
grep -q 'PATCH(skip-k3-fp8-ps)' "$R" || need_asm=1
grep -q 'PATCH(vLLM #50618)' "$D/vllm/model_executor/layers/utils.py" || need_asm=1
if [ "$need_asm" = 1 ]; then
  for p in patch_fp8asm.py patch_fp8_prefill.py patch_ps_metadata16.py patch_skip_k3_fp8_ps.py patch_wvsplitk.py; do
    echo "  applying $p ..."
    python3 "$SCRIPT_DIR/$p"
  done
else
  echo "  all 5 InferenceX ASM patches already present"
fi

say "1/7 sanity: shipped recipe backend present (_mtp_decode_qlen)"
grep -q _mtp_decode_qlen "$R" || { echo "!! recipe rocm_aiter_mla.py not installed — apply the base recipe first"; exit 1; }

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

say "4/7 recipe backend — DSpark verify width (_mtp_decode_qlen = 2*num_spec+1)"
python3 - <<'PY'
import py_compile, shutil, os
F = os.environ["R"]
s = open(F).read()
shutil.copy2(F, F + ".pre_dsparkqlen.bak")
if 'speculative_config.method == "dspark"' in s:
    print("  dspark qlen branch already present")
else:
    anchor = "        else:\n            self._mtp_decode_qlen = 1"
    branch = (
        "        elif (\n"
        "            speculative_config is not None\n"
        '            and speculative_config.method == "dspark"\n'
        "            and speculative_config.num_speculative_tokens is not None\n"
        "        ):\n"
        "            # DSpark drafts a semi-autoregressive block; the target verifies\n"
        "            # 1 anchor + 2*num_spec positions.\n"
        "            self._mtp_decode_qlen = 2 * int(speculative_config.num_speculative_tokens) + 1\n"
    )
    if s.count(anchor) == 1:
        s = s.replace(anchor, branch + anchor, 1)
        open(F, "w").write(s); py_compile.compile(F, doraise=True)
        print("  dspark qlen branch added")
    else:
        print(f"  !! anchor count {s.count(anchor)} — add the dspark branch manually")
PY

say "4b/7 recipe backend — _aiter_mla_small_head_mode (shipped helper + os.environ fallback)"
python3 - <<'PY'
import os, py_compile, re, shutil

F = os.environ["R"]
# Optional shipped-recipe reference for restoring the Gluon helpers; empty by
# default (the base ASM patches already install a recipe that carries them).
REF = os.environ.get("SHIPPED_MLA_REF", "") or ""
s = open(F).read()

# Some patch orderings drop Gluon support helpers; restore from the shipped file.
if "_gluon_mla_decode_supported" not in s and os.path.isfile(REF):
    ref = open(REF).read()
    m = re.search(
        r"@functools\.lru_cache\(maxsize=1\)\ndef _gluon_mla_decode_supported\(\)[\s\S]*?return on_gfx950\(\)\n",
        ref,
    )
    if m:
        block = m.group(0)
        if "def _aiter_mla_small_head_mode" in s:
            s = s.replace("\ndef _aiter_mla_small_head_mode", "\n" + block + "\n\ndef _aiter_mla_small_head_mode", 1)
        else:
            s = s.replace("\nclass AiterMLABackend", "\n" + block + "\n\ndef _aiter_mla_small_head_mode() -> str:\n    import os\n\n    return (os.environ.get(\"VLLM_ROCM_AITER_MLA_ASM_PADDING\") or \"auto\").lower()\n\n\nclass AiterMLABackend", 1)
        open(F, "w").write(s)
        print("  restored _gluon_mla_decode_supported from shipped ref")
        s = open(F).read()

# Nightly cb8104839 may lack envs.VLLM_ROCM_AITER_MLA_ASM_PADDING — read the env var directly.
if "envs.VLLM_ROCM_AITER_MLA_ASM_PADDING" in s:
    s = s.replace(
        "    import vllm.envs as envs\n\n    mode = (envs.VLLM_ROCM_AITER_MLA_ASM_PADDING or \"auto\").lower()",
        "    import os\n\n    mode = (os.environ.get(\"VLLM_ROCM_AITER_MLA_ASM_PADDING\") or \"auto\").lower()",
    )
    open(F, "w").write(s)
    py_compile.compile(F, doraise=True)
    print("  full small_head_mode helper: envs -> os.environ")
elif 'os.environ.get("VLLM_ROCM_AITER_MLA_ASM_PADDING")' in s:
    if "Small-head (<16) MLA decode" in s:
        print("  full small_head_mode helper with os.environ already present")
    elif os.path.isfile(REF):
        ref = open(REF).read()
        m = re.search(
            r"def _aiter_mla_small_head_mode\(\)[\s\S]*?return mode\n",
            ref,
        )
        if m:
            block = m.group(0).replace(
                "    import vllm.envs as envs\n\n    mode = (envs.VLLM_ROCM_AITER_MLA_ASM_PADDING or \"auto\").lower()",
                "    import os\n\n    mode = (os.environ.get(\"VLLM_ROCM_AITER_MLA_ASM_PADDING\") or \"auto\").lower()",
            )
            s = re.sub(
                r"\ndef _aiter_mla_small_head_mode\(\)[\s\S]*?(?=\n\nclass AiterMLABackend)",
                "\n" + block,
                s,
                count=1,
            )
            open(F, "w").write(s)
            py_compile.compile(F, doraise=True)
            print("  upgraded stub -> full shipped small_head_mode helper")
        else:
            print("  stub small_head_mode helper present (nightly gap)")
    else:
        print("  stub small_head_mode helper present (nightly gap)")
elif "def _aiter_mla_small_head_mode" not in s:
    helper = (
        "\n\n"
        "def _aiter_mla_small_head_mode() -> str:\n"
        '    """Return VLLM_ROCM_AITER_MLA_ASM_PADDING mode (auto/gluon/asm)."""\n'
        "    import os\n\n"
        '    return (os.environ.get("VLLM_ROCM_AITER_MLA_ASM_PADDING") or "auto").lower()\n\n'
    )
    anchor = "class AiterMLABackend(MLACommonBackend):"
    if anchor not in s:
        print("  !! AiterMLABackend anchor not found — add small_head_mode manually")
    else:
        s = s.replace(anchor, helper + anchor, 1)
        open(F, "w").write(s)
        py_compile.compile(F, doraise=True)
        print("  stub small_head_mode helper added")
else:
    print("  !! _aiter_mla_small_head_mode present but unreadable — inspect manually")
PY

say "4c/7 recipe backend — skip gluon verify flatten when ASM_PADDING=asm"
python3 - <<'PY'
import py_compile, os
F = os.environ["R"]
s = open(F).read()
old = (
    "        if (\n"
    "            self.num_heads < AiterMLAHelper._AITER_MIN_MLA_HEADS\n"
    "            and int(decode.max_qo_len) > 1\n"
    "        ):\n"
)
new = (
    "        if (\n"
    "            self.num_heads < AiterMLAHelper._AITER_MIN_MLA_HEADS\n"
    "            and int(decode.max_qo_len) > 1\n"
    '            and _aiter_mla_small_head_mode() != "asm"\n'
    "        ):\n"
)
if 'and _aiter_mla_small_head_mode() != "asm"' in s and "multi-token verify (DSpark)" in s:
    print("  gluon verify flatten already gated")
elif old in s:
    s = s.replace(old, new, 1)
    open(F, "w").write(s)
    py_compile.compile(F, doraise=True)
    print("  gluon verify flatten gated on asm mode")
else:
    print("  !! gluon verify flatten anchor not found — inspect decode path manually")
PY

say "5/7 recipe backend — broaden persistent-metadata gate"
python3 - <<'PY'
import py_compile, shutil, os
F = os.environ["R"]
s = open(F).read()
shutil.copy2(F, F + ".pre_gate.bak")
old = (
    "        use_persistent_metadata = (\n"
    "            self.num_heads >= AiterMLAHelper._AITER_MIN_MLA_HEADS\n"
    "            and max_qo_len >= 1\n"
    "            and max_qo_len <= self._mtp_decode_qlen\n"
    "        )"
)
new = (
    "        uses_asm_decode = (\n"
    "            self.num_heads >= AiterMLAHelper._AITER_MIN_MLA_HEADS\n"
    '            or _aiter_mla_small_head_mode() == "asm"\n'
    "            or max_qo_len > 1\n"
    "        )\n"
    "        use_persistent_metadata = (\n"
    "            uses_asm_decode\n"
    "            and max_qo_len >= 1\n"
    "            and max_qo_len <= self._mtp_decode_qlen\n"
    "        )"
)
if "uses_asm_decode" in s:
    print("  gate already broadened")
elif old in s:
    s = s.replace(old, new, 1); open(F, "w").write(s); py_compile.compile(F, doraise=True)
    print("  gate broadened")
else:
    print("  !! gate anchor not found — inspect use_persistent_metadata manually")
PY

say "6/7 KDA stride fix (Fangzhou-Ai/vllm PR #27; correctness)"
python3 - <<'PY'
import py_compile, re, shutil, os
F = os.environ["KDA"]
if not os.path.exists(F):
    print("  !! KDA file not found — skip"); raise SystemExit(0)
s = open(F).read()
shutil.copy2(F, F + ".pre_pr27.bak")
changed = False

def sub_once(pattern, repl, text, label):
    global changed
    text2, n = re.subn(pattern, repl, text, count=1, flags=re.DOTALL)
    if n:
        changed = True
        print(f"  + {label}")
    return text2

# fused_recurrent_kda_fwd_kernel — sibling already takes stride_indices_seq upstream.
s = sub_once(
    r"(def fused_recurrent_kda_fwd_kernel\([\s\S]*?    stride_state_token: tl\.constexpr,\n)"
    r"(?!    stride_indices_seq: tl\.constexpr,\n)"
    r"(    IS_SPEC_DECODING: tl\.constexpr,)",
    r"\1    stride_indices_seq: tl.constexpr,\n\2",
    s,
    "fwd kernel stride_indices_seq param",
)

# fused_recurrent_kda_packed_decode_kernel — the cudagraph/eager path that failed at 10/15.
s = sub_once(
    r"(def fused_recurrent_kda_packed_decode_kernel\([\s\S]*?    stride_state_token: tl\.constexpr,\n)"
    r"(?!    stride_indices_seq: tl\.constexpr,\n)"
    r"(    H: tl\.constexpr,)",
    r"\1    stride_indices_seq: tl.constexpr,\n\2",
    s,
    "packed_decode kernel stride_indices_seq param",
)

s2 = s.replace(
    "state_idx = tl.load(state_indices + i_n).to(tl.int64)",
    "state_idx = tl.load(state_indices + i_n * stride_indices_seq).to(tl.int64)",
)
if s2 != s:
    changed = True
    print("  + packed_decode state_idx stride load")
s = s2

s2 = s.replace(
    "    if state_indices.ndim != 1 or state_indices.stride(0) != 1:\n"
    "        state_indices = state_indices.reshape(-1).contiguous()",
    "    if state_indices.ndim != 1:\n"
    '        raise ValueError("`state_indices` must be one-dimensional.")',
)
if s2 != s:
    changed = True
    print("  + drop unit-stride requirement on state_indices")
s = s2

s2 = s.replace(
    "    if state_indices.ndim != 1 or state_indices.stride(0) != 1:\n"
    '        raise ValueError("`state_indices` must be contiguous and one-dimensional.")',
    "    if state_indices.ndim != 1:\n"
    '        raise ValueError("`state_indices` must be one-dimensional.")',
)
if s2 != s:
    changed = True
    print("  + drop contiguous check on state_indices")
s = s2

if not re.search(
    r"def fused_recurrent_kda_fwd\([\s\S]*?stride_indices_seq=ssm_state_indices\.stride\(0\)",
    s,
):
    s2 = s.replace(
        "        stride_state_token=initial_state.stride(0),\n"
        "        IS_SPEC_DECODING=",
        "        stride_state_token=initial_state.stride(0),\n"
        "        stride_indices_seq=ssm_state_indices.stride(0),\n"
        "        IS_SPEC_DECODING=",
        1,
    )
    if s2 != s:
        changed = True
        print("  + fwd kernel launch stride_indices_seq")
    s = s2

if not re.search(
    r"fused_recurrent_kda_packed_decode_kernel\[grid\][\s\S]*?"
    r"stride_indices_seq=state_indices\.stride\(0\)",
    s,
):
    s = sub_once(
        r"(fused_recurrent_kda_packed_decode_kernel\[grid\]\([\s\S]*?"
        r"        stride_state_token=initial_state\.stride\(0\),\n)"
        r"(        H=H,)",
        r"\1        stride_indices_seq=state_indices.stride(0),\n\2",
        s,
        "packed_decode kernel launch stride_indices_seq",
    )

if changed:
    open(F, "w").write(s)
py_compile.compile(F, doraise=True)
print("  KDA PR#27 applied" if changed else "  KDA PR#27 already present")
PY

say "7/7 verify"
echo "aiter 80-key   = $(grep -c '80: 64' "$AITER")                (expect >=1)"
echo "aiter get()    = $(grep -c 'get_block_n_fp8.get(' "$AITER")   (expect 1)"
echo "dspark qlen    = $(grep -c 'method == \"dspark\"' "$R")        (expect >=1)"
echo "asm gate       = $(grep -c 'uses_asm_decode' "$R")            (expect >=2)"
echo "kda stride     = $(grep -c 'stride_indices_seq' "$KDA")        (expect >=5)"
python -c "import vllm.v1.attention.backends.mla.rocm_aiter_mla; print('IMPORT_OK')"
echo
echo "DONE. Serve with:"
echo "  export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm"
echo "  NUM_SPEC=2 PORT=8890 GPU_MEM=0.88 MAX_NUM_SEQS=16 bash _serve_k3_bench_spec.sh"
