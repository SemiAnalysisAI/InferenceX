#!/usr/bin/env python3
"""Apply ROCm/aiter#4474 -- int64 KV offsets in _mla_gluon's >2GB path.

Upstream PR
-----------
ROCm/aiter#4474, "[Bugfix][Triton] Fix int32 KV-offset overflow in _mla_gluon
>2GB path". Five lines, inserted just before the empty-split early return:

    # >2GB KV cache (global_load path): widen strides to int64 so kv offsets
    # don't overflow int32.
    if not WITHIN_2GB:
        stride_kv_c_bs = stride_kv_c_bs.to(gl.int64)
        stride_k_pe_bs = stride_k_pe_bs.to(gl.int64)

Why this is very likely OUR crash
---------------------------------
Kimi-K3 DSpark has been dying with hipErrorIllegalAddress / HSA 0x1016 inside
mla_gluon, reached from forward_mqa. The kernel computes KV addresses as
`kv_loc * stride_kv_c_bs + ...` and `WITHIN_2GB` selects only the *load
instruction* (buffer_load vs global_load) -- the offset arithmetic itself stayed
int32. Our pools are over the 2 GB line in both dtypes:

    bf16 KV: 2,156,093 rows x 576 x 2B = 2.48 GB
    fp8  KV: 4,240,725 rows x 576 x 1B = 2.44 GB

so the byte offset wraps int32 (2,147,483,647) and the kernel reads a bogus
address.

This also explains the one signature nothing else accounted for: the crash lands
at a fixed number of completions PER CONCURRENCY SLOT -- 12 at c1, 53 at c4, 108
at c8, i.e. ~13 turns per slot every time. The offset only overflows once
allocated pages reach high indices in the pool, which depends on how much
context has accumulated, not on how many requests are in flight. A fixed
per-slot turn count is exactly what a fill-the-pool threshold produces, and it
is why time-to-crash scaled with concurrency while turns-per-slot did not.

This supersedes the earlier batch>1 hypothesis, which explained the presence of
a crash but not its timing.

Caveats, stated because they are not resolved:
  * The PR describes a host-side SIGSEGV; ours is a device-side fault. Both are
    consistent with a wrapped address, but they are not the same symptom.
  * Whether the generated code overflows in bytes or elements has not been
    confirmed by reading the lowered IR. The element count for bf16
    (1.24e9) fits int32; only the byte count (2.48e9) does not.

Ordering
--------
MUST run AFTER the gist mla_gluon patch, which replaces the whole file. The
launcher sequences it that way. Anchored on exact text and idempotent, so a
re-run or a drifted gist revision fails loudly instead of silently no-oping.
"""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import os
import sys

REL = "aiter/ops/triton/gluon/mla_gluon.py"

ANCHOR = """    num_iter = gl.cdiv(split_kv_end - split_kv_start, BLOCK_N)
    start_n = split_kv_start

    # early return with empty kv slice to save compute
"""

REPLACEMENT = """    num_iter = gl.cdiv(split_kv_end - split_kv_start, BLOCK_N)
    start_n = split_kv_start

    # ROCm/aiter#4474 -- >2GB KV cache (global_load path): widen strides to
    # int64 so kv offsets don't overflow int32. WITHIN_2GB selects the load
    # instruction but historically left this arithmetic in int32, so a K3-sized
    # pool (2.48 GB bf16 / 2.44 GB fp8) wraps and faults.
    if not WITHIN_2GB:
        stride_kv_c_bs = stride_kv_c_bs.to(gl.int64)
        stride_k_pe_bs = stride_k_pe_bs.to(gl.int64)

    # early return with empty kv slice to save compute
"""


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def find_target() -> str:
    spec = importlib.util.find_spec("aiter")
    if spec is None or not spec.submodule_search_locations:
        die("aiter is not importable in this interpreter")
    path = os.path.join(
        list(spec.submodule_search_locations)[0], "ops", "triton", "gluon", "mla_gluon.py"
    )
    if not os.path.isfile(path):
        die(f"{path} not found")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff-out")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    path = find_target()
    original = open(path, encoding="utf-8").read()

    if "stride_kv_c_bs.to(gl.int64)" in original:
        print(f"{path} already has the int64 promotion; nothing to do")
        return

    if original.count(ANCHOR) != 1:
        die(
            f"aiter#4474 anchor matched {original.count(ANCHOR)} times, expected 1. "
            "The gist revision or the image's aiter drifted -- re-derive before "
            "running. Refusing to leave the kernel half-patched."
        )

    text = original.replace(ANCHOR, REPLACEMENT)
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
    print(diff)

    if args.check:
        print("--check: anchor OK, result compiles; not written")
        return

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"Patched {path}: aiter#4474 int64 KV offsets (>2GB path)")


if __name__ == "__main__":
    main()
