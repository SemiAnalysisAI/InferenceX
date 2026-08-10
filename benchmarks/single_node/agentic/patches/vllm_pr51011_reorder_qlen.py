#!/usr/bin/env python3
"""Port of vllm-project/vllm#51011 onto nightly-cb8104839c.

#51011 ("[ROCm][MLA] [K3] Fix fp8 KV cache decode on the AITER MLA backend")
makes three changes. Only ONE of them can do anything on this branch; the other
two are already covered by patches we apply first, and re-implementing them
would mean rewriting lines #51088 and vllm_dspark_ps_enable.py already rewrote.
What is taken and what is not:

TAKEN -- _mtp_decode_qlen sized from reorder_batch_threshold
-------------------------------------------------------------
Upstream replaces a method-name whitelist

    if speculative_config.method in ("mtp", "deepseek_mtp"):
        self._mtp_decode_qlen = num_speculative_tokens + 1

with

    self._mtp_decode_qlen = self.reorder_batch_threshold or 1

vllm_dspark_ps_enable.py already fixes the *symptom* by adding "dspark" to the
whitelist, which yields num_spec + 1 = 4 at k=3. reorder_batch_threshold is not
4. This image computes it at backend.py:675 as

    1 + (2 if speculative_config.parallel_drafting else 1) * num_spec

and speculative.py sets parallel_drafting = True for method in ("dflash",
"dspark"), so the real threshold at k=3 is 1 + 2*3 = 7.

That gap is the whole point of this patch. MLACommonMetadataBuilder asserts
`m.max_query_len <= self.reorder_batch_threshold` (mla_attention.py:1960), so
the router may legitimately hand _build_decode a step with qlen up to 7, while
our buffers are sized for 4 and the gate reads `max_qo_len <= self
._mtp_decode_qlen`. Any step in 5..7 closes the gate, aiter falls back to
kernel-internal metadata, and indexes get_block_n_fp8[num_heads * qlen] -- a
table holding only {8, 16, 24, 32, 48, 64, 128, 256, 384, 512}. At 16 heads
(12 padded) qlen 5, 6 and 7 are all KeyError, raised mid-run rather than at
startup.

WHETHER K3 ACTUALLY EMITS A qlen > 4 STEP IS THE OPEN QUESTION. DSpark verify
is k+1 = 4 tokens, so the common step is 4 and this patch would then only
enlarge the metadata buffers and change nothing measurable. Sizing for 7 is
correct either way -- it is the value the router is allowed to produce -- but
do not assume a throughput delta. A null result here is a real answer.

NOT TAKEN -- the fp8 guard on use_gluon_decode / use_gluon_verify
------------------------------------------------------------------
Upstream adds `is_quantized_kv_cache(kv_cache_dtype) -> never Gluon` and hoists
the forward_mqa verify branch into a use_gluon_verify helper. Every cell on this
branch runs MLA_FORCE_PS=1, and #51088 already ANDs `not
VLLM_ROCM_MLA_FORCE_PS` into the use_gluon_decode callsite while
vllm_dspark_ps_enable.py adds the same condition to the forward_mqa branch. Both
predicates are therefore already hard False before any dtype test runs. Adding
the guard would change the signature of a staticmethod with two callsites for
zero behavioral difference, so it is deliberately skipped.

NOT TAKEN -- the routing-based persistent-metadata gate
--------------------------------------------------------
Upstream replaces `num_heads >= 16` with "not routed to either Gluon path".
#51088 already replaced the same expression with

    (num_heads >= _AITER_MIN_MLA_HEADS or (VLLM_ROCM_MLA_FORCE_PS
     and not use_gluon_decode))

which opens the gate for our 12-head rank for the same reason. Re-applying the
upstream form would collide with that hunk.

So this patcher is one edit. That is not a shortcut -- it is the entire
behavioral surface of #51011 under MLA_FORCE_PS=1.
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import sys

REL = "vllm/v1/attention/backends/mla/rocm_aiter_mla.py"


def die(msg: str) -> None:
    sys.exit(f"vllm_pr51011_reorder_qlen.py: {msg}")


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
# Match the whole whitelist block, from the comment through the else-branch.
#
# Tolerant of vllm_dspark_ps_enable.py having already inserted "dspark" into
# the tuple, because the recipe runs that patcher first and this one is a
# strict superset of its Edit 1. Anchored on the two literal assignments rather
# than on the tuple contents, so neither ordering changes the match.
# --------------------------------------------------------------------------
BLOCK_RE = re.compile(
    r"[ \t]*#[^\n]*verification runs decode[^\n]*\n"
    r"[ \t]*#[^\n]*\n"
    r"(?P<ind>[ \t]*)speculative_config = vllm_config\.speculative_config\n"
    r"[ \t]*if \(\n"
    r"(?:[^\n]*\n)*?"
    r"[ \t]*self\._mtp_decode_qlen = int\(speculative_config\.num_speculative_tokens\) \+ 1\n"
    r"[ \t]*else:\n"
    r"[ \t]*self\._mtp_decode_qlen = 1\n"
)

REPLACEMENT = '''{ind}# vllm#51011: size the metadata from reorder_batch_threshold, the largest
{ind}# query length decode can be handed -- MLACommonMetadataBuilder asserts
{ind}# max_query_len <= reorder_batch_threshold, and the threshold already
{ind}# accounts for the drafting scheme (1 + 2*num_spec when parallel_drafting,
{ind}# so 7 for DSpark at k=3, not the 4 a method-name whitelist yields).
{ind}# Undersizing closes the persistent gate for any step in 5..7, and aiter
{ind}# then indexes get_block_n_fp8[num_heads * qlen], which has no entry for
{ind}# 16*5, 16*6 or 16*7 -- a mid-run KeyError.
{ind}self._mtp_decode_qlen = self.reorder_batch_threshold or 1
{ind}logger.warning_once(
{ind}    "LOCAL PATCH vllm#51011: _mtp_decode_qlen=%s from "
{ind}    "reorder_batch_threshold",
{ind}    self._mtp_decode_qlen,
{ind})
'''


def block_sub(text: str) -> tuple[str, int]:
    def repl(m: "re.Match[str]") -> str:
        return REPLACEMENT.format(ind=m.group("ind"))

    return BLOCK_RE.subn(repl, text)


def dump_region(text: str, needle: str, ctx: int = 14) -> str:
    lines = text.splitlines()
    out: list[str] = []
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

    text, n = block_sub(original)
    if n != 1:
        if "self._mtp_decode_qlen = self.reorder_batch_threshold" in original:
            print("  _mtp_decode_qlen already sized from reorder_batch_threshold")
            return
        print(dump_region(original, "_mtp_decode_qlen"), file=sys.stderr)
        die(f"_mtp_decode_qlen block matched {n} times, expected 1")

    # The replacement reads self.reorder_batch_threshold, which the base
    # MLACommonMetadataBuilder sets in its own __init__ via
    # _init_reorder_batch_threshold. super().__init__ runs at the top of this
    # __init__, so the attribute exists by the time we read it -- but fail loudly
    # here rather than at engine init if a future image reorders that.
    if "_init_reorder_batch_threshold" not in original and (
        "reorder_batch_threshold" not in original
    ):
        print(
            "  NOTE: reorder_batch_threshold not referenced in this file; it is "
            "inherited from MLACommonMetadataBuilder. Verify at runtime via the "
            "LOCAL PATCH vllm#51011 log line.",
            file=sys.stderr,
        )

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
    print(f"patched {path}: _mtp_decode_qlen <- reorder_batch_threshold (vllm#51011)")


if __name__ == "__main__":
    main()
