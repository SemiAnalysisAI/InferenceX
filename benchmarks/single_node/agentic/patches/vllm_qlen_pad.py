#!/usr/bin/env python3
"""Pad the DSpark verify block to a supported asm qSeqLen, then trim.

The gfx950 asm registry (hsa/gfx950/mla/mla_asm.csv) only has kernels at
qSeqLen 1, 2, 4 and 8 -- there is no 3.  For K3's required bf16-query/fp8-KV
combination there is exactly ONE row, qSeqLen=4:

    bf16,fp8,Gqa16,ps=1,qSeqLen=4 -> mla_a16w8_qh16_m16x4_n16x1_coex0_mask1_ps.co

so k=2 (verify qlen = k+1 = 3) has nothing to route to.  This pads the query
block 3 -> 4, runs the qSeqLen=4 kernel, and drops the pad rows from the output.

WHY PREPEND, NOT APPEND
-----------------------
The kernel derives each query row's causal window from the request's seq_len and
max_qo_len -- row t covers roughly `seq_len - max_qo_len + 1 + t` KV entries.
seq_len counts the tokens actually scheduled this step and does NOT change when
we pad the query tensor.  So raising max_qo_len 3 -> 4 shifts every window down
by one:

    real qlen 3:  row t -> context + 1 + t          (correct)
    padded to 4,
    appended:     row t -> context + t              (WRONG, off by one)
    prepended:    row 0 -> context      (pad, discarded)
                  row 1 -> context + 1  } the real rows keep
                  row 2 -> context + 2  } their correct windows
                  row 3 -> context + 3  }

Prepending puts the real tokens at t = 1..3, which restores the original
offsets.  Appending would silently corrupt attention for every real row.

This reasoning is derived from the window arithmetic, NOT from reading the asm
kernel source.  It MUST be validated with an accuracy eval -- a wrong causal
window produces fluent, plausible output with subtly wrong attention, which no
throughput number will catch.

Only the query tensor is padded.  No KV is written here (forward_mqa only reads
kv_c_and_k_pe_cache; the slot writes happen earlier in the model runner), and
paged_kv_* metadata is per-request rather than per-token, so none of it changes.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

REL = "vllm/v1/attention/backends/mla/rocm_aiter_mla.py"

PRELUDE = '''
_QLEN_PAD_SUPPORTED = (1, 2, 4, 8)


def _qlen_pad_enabled() -> bool:
    return os.environ.get("VLLM_ROCM_MLA_QLEN_PAD", "0") == "1"

'''

INSERT_BEFORE_PAD = """        _qp_n = 0
        _qp_indptr = decode.qo_indptr
        _qp_max = decode.max_qo_len
        _qp_real = int(decode.max_qo_len)
        if (
            _qlen_pad_enabled()
            and _qp_real > 1
            and _qp_real not in _QLEN_PAD_SUPPORTED
        ):
            _qp_tgt = min(v for v in _QLEN_PAD_SUPPORTED if v > _qp_real)
            _qp_n = _qp_tgt - _qp_real
            _qp_nreq = q.shape[0] // _qp_real
            _qp_tail = q.shape[1:]
            _qp_q = q.view(_qp_nreq, _qp_real, *_qp_tail)
            # Prepend: real rows must land at t = _qp_n.._qp_tgt-1 so their
            # causal windows are unchanged. See module docstring.
            _qp_head = _qp_q[:, :1].expand(_qp_nreq, _qp_n, *_qp_tail)
            q = torch.cat([_qp_head, _qp_q], dim=1).reshape(-1, *_qp_tail)
            q = q.contiguous()
            B = q.shape[0]
            _qp_indptr = torch.arange(
                _qp_nreq + 1, device=q.device, dtype=decode.qo_indptr.dtype
            ) * _qp_tgt
            _qp_max = _qp_tgt
            logger.info_once(
                "VLLM_ROCM_MLA_QLEN_PAD: verify qlen %d -> %d (prepended %d)",
                _qp_real,
                _qp_tgt,
                _qp_n,
            )
"""

TRIM = """        if _qp_n:
            o = o.view(-1, _qp_max, *o.shape[1:])[:, _qp_n:, ...]
            o = o.reshape(-1, *o.shape[2:]).contiguous()
"""


def die(msg: str) -> None:
    sys.exit(f"vllm_qlen_pad.py: {msg}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target")
    ap.add_argument("--diff-out")
    a = ap.parse_args()

    if a.target:
        path = a.target
    else:
        import vllm

        path = os.path.join(os.path.dirname(os.path.dirname(vllm.__file__)), REL)
    if not os.path.exists(path):
        die(f"{path} not found")
    src = open(path).read()
    orig = src

    if "_QLEN_PAD_SUPPORTED" in src:
        print("vllm_qlen_pad.py: already applied, skipping")
        return

    # 1. module-level helper, after the last top-level import
    m = list(re.finditer(r"(?m)^from vllm\.[^\n]*\n", src))
    if not m:
        die("no vllm imports found to anchor the prelude")
    at = m[-1].end()
    src = src[:at] + PRELUDE + src[at:]

    # 2. pad, immediately before the head-padding call in forward_mqa
    anchor = "        mla_padded_q = AiterMLAHelper.get_mla_padded_q(self.num_heads, q)\n"
    if src.count(anchor) != 1:
        die(f"pad anchor matched {src.count(anchor)} times, expected 1")
    src = src.replace(anchor, INSERT_BEFORE_PAD + anchor)

    # 3. route the padded metadata into the kernel call
    call = "            decode.qo_indptr,\n            decode.max_qo_len,\n"
    if src.count(call) != 1:
        die(f"call anchor matched {src.count(call)} times, expected 1")
    src = src.replace(call, "            _qp_indptr,\n            _qp_max,\n")

    # 4. trim, before the head-unpadding return
    ret = (
        "        return AiterMLAHelper.get_mla_unpadded_o(self.num_heads, o), None\n"
    )
    if src.count(ret) != 1:
        die(f"return anchor matched {src.count(ret)} times, expected 1")
    src = src.replace(ret, TRIM + ret)

    open(path, "w").write(src)
    if a.diff_out:
        import difflib

        with open(a.diff_out, "w") as fh:
            fh.writelines(
                difflib.unified_diff(
                    orig.splitlines(True), src.splitlines(True), REL, REL
                )
            )
    print(f"vllm_qlen_pad.py: patched {path}")


if __name__ == "__main__":
    main()
