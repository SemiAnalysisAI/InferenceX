#!/usr/bin/env python3
"""Let DSpark verify actually reach the AITER PS ASM kernel.

Applied on top of vllm-project/vllm#51088, which adds VLLM_ROCM_MLA_FORCE_PS
but stops short of routing multi-token verify to asm. Two edits, both anchored
by exact string match rather than a context diff -- the installed
rocm_aiter_mla.py drifts from the GitHub source for the same image tag (#51088's
own hunks land at offset +1, and a context-diff version of edit 2 applied with
`fuzz 2 (offset -28 lines)` in run 30978612788).

Edit 1 -- `_mtp_decode_qlen` recognizes "dspark"
------------------------------------------------
    if speculative_config.method in ("mtp", "deepseek_mtp"):
        self._mtp_decode_qlen = num_speculative_tokens + 1
    else:
        self._mtp_decode_qlen = 1

Kimi-K3 DSpark reports `method == "dspark"`, so this falls to the else and
pins the value at 1. That feeds:

    use_persistent_metadata = (...) and max_qo_len <= self._mtp_decode_qlen

and verify runs at max_qo_len = k+1, so the comparison is `4 <= 1` and
persistent metadata is never built. aiter then gets ps=0 and fails selection:

    get_heuristic_kernel_mla: cannot get heuristic kernel!
      q_type:bf16 kv_type:fp8 gqa:16 ps:0 prefill:0 causal:0 qseqlen:4

The registry has exactly that shape at ps=1 --
`mla_a16w8_qh16_m16x4_n16x1_coex0_mask1_ps.co` -- and nothing at ps=0, which is
why run 30978612788 died at engine init with every worker down.

The same value also sizes the PS metadata buffers via
`get_mla_metadata_info_v1(max_num_reqs, self._mtp_decode_qlen, ...)`, so 1
under-allocates them for a k+1 verify block. Fixing the qlen fixes both.

Edit 2 -- multi-token verify falls through to asm
-------------------------------------------------
#51088 adds `and not VLLM_ROCM_MLA_FORCE_PS` to `use_gluon_decode`, but
`forward_mqa` has a second, unconditional branch that returns for
`num_heads < 16 and max_qo_len > 1`. Under DSpark every target step is a verify
step, so that branch always wins and the asm route below is unreachable. Run
30971662995 is the evidence: it loaded a qseqlen1 PS kernel, then served every
step on Gluon, with acceptance identical to the Gluon baseline (1.21 mean /
10.6% against 1.21 / 10.8%).

Both edits are no-ops unless VLLM_ROCM_MLA_FORCE_PS=1, except edit 1, which is
a strict correctness improvement for any DSpark run.
"""

from __future__ import annotations

import argparse
import difflib
import os
import sys

REL = "vllm/v1/attention/backends/mla/rocm_aiter_mla.py"


def die(msg: str) -> None:
    sys.exit(f"vllm_dspark_ps_enable.py: {msg}")


def find_target(override: str | None = None) -> str:
    if override:
        if not os.path.exists(override):
            die(f"{override} not found")
        return override
    import vllm

    path = os.path.join(os.path.dirname(os.path.dirname(vllm.__file__)), REL)
    if not os.path.exists(path):
        die(f"{path} not found")
    return path


# --------------------------------------------------------------------------
# Edit 1: teach _mtp_decode_qlen about DSpark.
# --------------------------------------------------------------------------
QLEN_OLD = '''and speculative_config.method in ("mtp", "deepseek_mtp")'''
QLEN_NEW = '''and speculative_config.method in ("mtp", "deepseek_mtp", "dspark")'''

# --------------------------------------------------------------------------
# Edit 2: stop the small-head multi-token branch from swallowing verify.
# --------------------------------------------------------------------------
MQA_OLD = """        if (
            self.num_heads < AiterMLAHelper._AITER_MIN_MLA_HEADS
            and int(decode.max_qo_len) > 1
        ):"""
MQA_NEW = """        if (
            self.num_heads < AiterMLAHelper._AITER_MIN_MLA_HEADS
            and int(decode.max_qo_len) > 1
            and not VLLM_ROCM_MLA_FORCE_PS
        ):"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff-out")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--target")
    args = ap.parse_args()

    path = find_target(args.target)
    original = open(path, encoding="utf-8").read()
    text = original

    if "deepseek_mtp\", \"dspark\"" in text and "not VLLM_ROCM_MLA_FORCE_PS\n        ):" in text:
        print(f"{path} already patched; nothing to do")
        return

    if "VLLM_ROCM_MLA_FORCE_PS" not in text:
        die(
            "VLLM_ROCM_MLA_FORCE_PS not present -- vllm#51088 must be applied "
            "before this patcher."
        )

    for name, old in (("_mtp_decode_qlen method tuple", QLEN_OLD),
                      ("forward_mqa small-head branch", MQA_OLD)):
        if text.count(old) != 1:
            die(
                f"anchor for {name} matched {text.count(old)} times, expected 1. "
                "The image drifted; re-derive the anchors."
            )

    text = text.replace(QLEN_OLD, QLEN_NEW)
    text = text.replace(MQA_OLD, MQA_NEW)

    compile(text, path, "exec")

    diff = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            text.splitlines(keepends=True),
            fromfile=f"a/{REL}",
            tofile=f"b/{REL}",
        )
    )
    if args.diff_out:
        with open(args.diff_out, "w", encoding="utf-8") as fh:
            fh.write(diff)
    if args.check:
        print(diff)
        return

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(diff)
    print(f"patched {path}: dspark -> _mtp_decode_qlen, verify -> asm PS route")


if __name__ == "__main__":
    main()
