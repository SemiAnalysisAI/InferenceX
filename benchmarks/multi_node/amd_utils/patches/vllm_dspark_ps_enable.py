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
import re
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
QLEN_RE = re.compile(
    r'(speculative_config\.method\s+in\s+\(\s*"mtp"\s*,\s*"deepseek_mtp"\s*)(\))'
)


def qlen_sub(text: str) -> tuple[str, int]:
    return QLEN_RE.subn(r'\1, "dspark"\2', text)


# --------------------------------------------------------------------------
# Edit 2: stop the small-head multi-token branch from swallowing verify.
#
# Whitespace-tolerant on purpose. An exact-string version of this anchor
# reported 0 matches in run 30980537097 while matching by hand in the very same
# container, so the literal is not something to depend on; only the token
# sequence is. On failure we dump the region rather than guessing again.
# --------------------------------------------------------------------------
MQA_RE = re.compile(
    r"(\n(?P<ind>[ \t]*)if\s*\(\s*\n"
    r"[ \t]*self\.num_heads\s*<\s*AiterMLAHelper\._AITER_MIN_MLA_HEADS\s*\n"
    r"[ \t]*and\s+int\(decode\.max_qo_len\)\s*>\s*1\s*\n)"
    r"(?P<close>[ \t]*\):)"
)


def mqa_sub(text: str) -> tuple[str, int]:
    def repl(m: "re.Match[str]") -> str:
        ind = m.group("ind")
        return f"{m.group(1)}{ind}    and not VLLM_ROCM_MLA_FORCE_PS\n{m.group('close')}"

    return MQA_RE.subn(repl, text)


def dump_region(text: str, needle: str, ctx: int = 12) -> str:
    lines = text.splitlines()
    out = []
    for i, line in enumerate(lines):
        if needle in line:
            lo, hi = max(0, i - ctx), min(len(lines), i + ctx)
            out.append(f"--- around line {i + 1} ---")
            out += [f"{n + 1:5d}| {lines[n]}" for n in range(lo, hi)]
    return "\n".join(out) or f"(no line containing {needle!r})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff-out")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--target")
    args = ap.parse_args()

    path = find_target(args.target)
    original = open(path, encoding="utf-8").read()
    text = original

    if "VLLM_ROCM_MLA_FORCE_PS" not in text:
        die(
            "VLLM_ROCM_MLA_FORCE_PS not present -- vllm#51088 must be applied "
            "before this patcher."
        )

    text, n_qlen = qlen_sub(text)
    if n_qlen != 1:
        if '"dspark"' in text:
            print("  _mtp_decode_qlen already lists dspark")
        else:
            print(dump_region(original, "deepseek_mtp"), file=sys.stderr)
            die(f"_mtp_decode_qlen anchor matched {n_qlen} times, expected 1")

    text, n_mqa = mqa_sub(text)
    if n_mqa != 1:
        if "and not VLLM_ROCM_MLA_FORCE_PS\n" in text:
            print("  forward_mqa branch already guarded")
        else:
            print(dump_region(original, "_AITER_MIN_MLA_HEADS"), file=sys.stderr)
            die(f"forward_mqa anchor matched {n_mqa} times, expected 1")

    if text == original:
        print(f"{path} already patched; nothing to do")
        return

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
