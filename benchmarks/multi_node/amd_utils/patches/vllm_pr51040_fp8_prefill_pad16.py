#!/usr/bin/env python3
"""Port of vllm-project/vllm#51040 onto nightly-cb8104839c.

#51040 ("[ROCm][K3] Extend FP8 asm MLA prefill to non-divisor small head
counts") lets the fp8 PS asm prefill run at K3's 12 heads/rank. Today the gate
is `num_heads % 16 == 0`, which is False for 12, so K3 has never taken this
path -- every prefill falls through to the bf16 flash/triton route regardless of
KV dtype.

Six edits, all present verbatim in this image (verified against the file
GitHub serves for cb8104839c):

  1. builder gate   `_fp8_prefill_enabled`   line 333
  2. impl gate      `_fp8_prefill_enabled`   line 878
  3. `num_head_k = self.num_heads`           lines 390 and 484 (both)
  4. `nhead = self.num_heads`                line 922 -> replicate-pad q/k/v
  5. `out_3d = out.view(...)`                line 949 -> scratch when padded
  6. copy-back after the mla_reduce_v1 call

WHY THE PADDING IS EXACT
------------------------
MLA computes each query head independently against the shared KV, so heads
added to q cannot influence the real ones. q, k and v are padded in the SAME
head order by the same helper, so padded head i pairs with padded k/v head i and
real heads 0..11 keep their own k/v. The output is sliced back to 12. This is
the same argument the decode path's pad-to-16 already rests on.

Note this image's get_mla_padded_q is #51088's ZERO-pad variant for non-divisor
counts, not upstream's tile-and-slice. Both are exact here for the same reason
(the pad heads are discarded), and the zero-pad version is contiguous by
construction, which is what asm_mla.cu:805 requires.

TWO REASONS THIS MAY MEASURE NOTHING -- READ BEFORE QUOTING A DELTA
--------------------------------------------------------------------
a) forward_mha bails to super().forward_mha whenever
   `prefill_metadata.chunked_context is not None`. The agentic trace runs ISL
   p50 ~60K at ~90% prefix-cache hit, so most prefills ARE context-chunked and
   never reach this kernel. The fp8 prefill path may simply not be entered.

b) `_fp8_mla_prefill_supported()` must return True on the deployed aiter build.
   The patch logs the resolved value of _fp8_prefill_enabled on both the builder
   and the impl so a silent False is visible in server.log rather than being
   mistaken for "the kernel ran and did not help".

Grep server.log for "LOCAL PATCH vllm#51040" to confirm which of these happened.

UPSTREAM STATUS: #51040 is OPEN, has ONE review comment, and its CI has never
run -- the pre-run-check gate fails on author reputation (1 merged PR, no
verified/ready label), so no buildkite job has ever executed against it. We are
its first real execution. Treat a crash here as a finding about the PR, not
about our stack.
"""

from __future__ import annotations

import argparse
import difflib
import os
import sys

REL = "vllm/v1/attention/backends/mla/rocm_aiter_mla.py"


def die(msg: str) -> None:
    sys.exit(f"vllm_pr51040_fp8_prefill_pad16.py: {msg}")


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


GATE_OLD = """        self._fp8_prefill_enabled = (
            _fp8_mla_prefill_supported() and self.num_heads % 16 == 0
        )
"""

# Builder: kv_cache_dtype_str is normalized to "fp8" at line 277 of this image,
# well before the gate, so the string compare is equivalent to a quant check for
# every fp8 spelling. Deliberately NOT is_quantized_kv_cache: upstream uses two
# different predicates on the two sides of this PR (kv_cache_interface's on the
# builder, torch_utils' on the impl), and torch_utils' returns True for
# int8/int4 per-token-head caches, which this kernel cannot serve.
GATE_BUILDER_NEW = """        # vllm#51040: allow non-divisor small head counts (K3 = 12/rank at TP8)
        # via pad-to-16, and only when the KV cache is actually FP8 -- on a bf16
        # cache the PS workspace must not be reserved.
        self._fp8_prefill_enabled = _fp8_mla_prefill_supported() and (
            kv_cache_dtype_str == "fp8"
            and (self.num_heads % 16 == 0 or 0 < self.num_heads < 16)
        )
        logger.warning_once(
            "LOCAL PATCH vllm#51040 (builder): fp8_prefill_enabled=%s "
            "num_heads=%s kv_cache_dtype=%s supported=%s",
            self._fp8_prefill_enabled,
            self.num_heads,
            kv_cache_dtype_str,
            _fp8_mla_prefill_supported(),
        )
"""

GATE_IMPL_NEW = """        # vllm#51040: see the builder gate. Same predicate, spelled against the
        # impl's raw kv_cache_dtype argument.
        self._fp8_prefill_enabled = _fp8_mla_prefill_supported() and (
            str(kv_cache_dtype).startswith("fp8")
            and (self.num_heads % 16 == 0 or 0 < self.num_heads < 16)
        )
        logger.warning_once(
            "LOCAL PATCH vllm#51040 (impl): fp8_prefill_enabled=%s "
            "num_heads=%s kv_cache_dtype=%s supported=%s",
            self._fp8_prefill_enabled,
            self.num_heads,
            kv_cache_dtype,
            _fp8_mla_prefill_supported(),
        )
"""

NUM_HEAD_K_OLD = "        num_head_k = self.num_heads\n"
NUM_HEAD_K_NEW = """        # vllm#51040: non-divisor head counts are padded to 16 in
        # _mla_fp8_prefill_attn, so build the PS metadata for the padded count or
        # the work/reduce maps will not match. Also lowers the partial-tile
        # count: gcd(16, cu_num=256)=16 (~960 tiles) vs gcd(12,256)=4 (~4032).
        num_head_k = max(16, self.num_heads)
"""

NHEAD_OLD = "        nhead = self.num_heads\n"
NHEAD_NEW = """        # vllm#51040: the PS asm prefill and mla_reduce_v1 both require
        # 16-aligned heads, and the metadata above was built for
        # max(16, num_heads). Pad q/k/v to 16 in the same head order -- MLA
        # attention is independent per query head over the shared KV, so the pad
        # heads cannot affect the real ones -- then slice the output back.
        _real_nhead = self.num_heads
        _pad16 = _real_nhead < 16
        if _pad16:
            q = AiterMLAHelper.get_mla_padded_q(_real_nhead, q)
            k = AiterMLAHelper.get_mla_padded_q(_real_nhead, k)
            v = AiterMLAHelper.get_mla_padded_q(_real_nhead, v)
        nhead = 16 if _pad16 else self.num_heads
"""

OUT3D_OLD = "        out_3d = out.view(total_q, nhead, v_head_dim)\n"
OUT3D_NEW = """        if _pad16:
            # vllm#51040: padded heads cannot alias the real-head storage of
            # `out`, so the kernels need their own buffer.
            out_3d = torch.empty(
                total_q, nhead, v_head_dim, dtype=out.dtype, device=out.device
            )
        else:
            out_3d = out.view(total_q, nhead, v_head_dim)
"""

# Tail of _mla_fp8_prefill_attn: the mla_reduce_v1 call ends with final_lse.
# forward_mha is the next def, which makes this a unique anchor.
TAIL_OLD = """            out_3d,
            final_lse,
        )

    def forward_mha(
"""
TAIL_NEW = """            out_3d,
            final_lse,
        )

        if _pad16:
            # vllm#51040: drop the pad heads back out into the caller's buffer.
            out.view(total_q, _real_nhead, v_head_dim).copy_(
                out_3d[:, :_real_nhead, :]
            )

    def forward_mha(
"""


def replace_exactly(text: str, old: str, new: str, count: int, label: str) -> str:
    n = text.count(old)
    if n != count:
        die(f"{label}: anchor found {n} times, expected {count}")
    return text.replace(old, new, count)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff-out")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--target")
    args = ap.parse_args()

    path = find_target(args.target)
    original = open(path, encoding="utf-8").read()

    if "vllm#51040" in original:
        print(f"{path} already carries vllm#51040; nothing to do")
        return

    text = original

    # The two _fp8_prefill_enabled assignments are byte-identical, so they
    # cannot be told apart by the assignment alone. Split the file at the impl
    # class and patch each half once. AiterMLAImpl is defined after the builder.
    marker = "\nclass AiterMLAImpl("
    if text.count(marker) != 1:
        die(f"expected exactly one '{marker.strip()}' definition")
    head, tail = text.split(marker, 1)
    tail = marker + tail

    head = replace_exactly(head, GATE_OLD, GATE_BUILDER_NEW, 1, "builder fp8 gate")
    tail = replace_exactly(tail, GATE_OLD, GATE_IMPL_NEW, 1, "impl fp8 gate")
    text = head + tail

    text = replace_exactly(text, NUM_HEAD_K_OLD, NUM_HEAD_K_NEW, 2, "num_head_k")
    text = replace_exactly(text, NHEAD_OLD, NHEAD_NEW, 1, "nhead pad")
    text = replace_exactly(text, OUT3D_OLD, OUT3D_NEW, 1, "out_3d scratch")
    text = replace_exactly(text, TAIL_OLD, TAIL_NEW, 1, "copy-back")

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
    print(f"patched {path}: fp8 asm prefill pad-to-16 (vllm#51040)")


if __name__ == "__main__":
    main()
