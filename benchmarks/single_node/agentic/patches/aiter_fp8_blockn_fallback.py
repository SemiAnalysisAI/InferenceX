#!/usr/bin/env python3
"""Make aiter's fp8 MLA block-N lookup tolerate unlisted (nhead x qlen) products.

The failure
-----------
Kimi-K3 DSpark c8 with fp8 KV + padded asm verify (run 30784128673) dies during
engine init, in CUDA-graph capture:

    File "aiter/mla.py", line 203, in get_meta_param
        min_block_n = get_block_n_fp8[int(nhead * max_seqlen_q)]
    KeyError: 240

`get_block_n_fp8` is a sparse tuning table keyed on `nhead * max_seqlen_q`:

    {8: 64, 16: 128, 24: 128, 32: 128, 48: 64, 64: 64,
     128: 32, 256: 32, 384: 32, 512: 32}

With vllm#50578 padding K3's 12 heads/rank up to 16, a verify step at qlen 8
gives 16*8 = 128, which IS in the table. The engine-init capture asks for 240,
i.e. qlen 15 at nhead 16 -- an intermediate query length vLLM exercises during
graph capture that upstream never tabulated. (Why 15 specifically is not
established; this patch deliberately handles ANY unlisted product rather than
special-casing one value.)

Note this is reachable only under fp8: the lookup sits behind
`if dtype == dtypes.fp8 and not ignore_total_kv`. That is why the bf16 asm-verify
arm got past init and this one did not.

Why a fallback is safe
----------------------
`min_block_n` is used for exactly one thing -- clamping `num_kv_splits`:

    num_kv_splits = min(num_kv_splits,
                        (total_kv + bs*min_block_n - 1) // (bs*min_block_n))

It does not select a kernel, size a buffer, or affect addressing. A
conservative value costs split parallelism, not correctness.

The fallback takes the value of the largest tabulated key <= the requested one
(and the smallest key's value if below the table). For 240 that is key 128 ->
32, and the next key up (256) is also 32, so the interpolation is unambiguous
here. The table is monotone non-increasing from 128 onward, so this rule can
only ever pick a block_n that is >= the "true" neighbouring value, i.e. it
errs toward fewer splits.

RESIDUAL RISK: the table may be sparse because the asm fp8 kernel genuinely has
no implementation at those products, in which case this moves the failure later
rather than fixing it. That is still a cheap, fast, informative outcome -- it
fails at init either way.

Enabled only when DSPARK_FP8_BLOCKN_FALLBACK=1 (the launcher sets it for fp8
spec-decode arms). Idempotent; fails loudly.
"""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import os
import sys

REL = "aiter/mla.py"

ANCHOR = """    if dtype == dtypes.fp8 and not ignore_total_kv:
        min_block_n = get_block_n_fp8[int(nhead * max_seqlen_q)]
"""

REPLACEMENT = '''    if dtype == dtypes.fp8 and not ignore_total_kv:
        # LOCAL PATCH -- get_block_n_fp8 is a sparse tuning table and vLLM's
        # CUDA-graph capture asks for products it does not list (KeyError: 240,
        # i.e. nhead 16 x qlen 15, on Kimi-K3 DSpark verify with vllm#50578
        # head padding). min_block_n only clamps num_kv_splits below -- it does
        # not pick a kernel or size a buffer -- so falling back to the nearest
        # tabulated key at or below the request costs split parallelism, never
        # correctness.
        _blockn_key = int(nhead * max_seqlen_q)
        if _blockn_key in get_block_n_fp8:
            min_block_n = get_block_n_fp8[_blockn_key]
        else:
            _lower = [k for k in get_block_n_fp8 if k <= _blockn_key]
            _pick = max(_lower) if _lower else min(get_block_n_fp8)
            min_block_n = get_block_n_fp8[_pick]
'''


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def find_target() -> str:
    spec = importlib.util.find_spec("aiter")
    if spec is None or not spec.submodule_search_locations:
        die("aiter is not importable in this interpreter")
    path = os.path.join(list(spec.submodule_search_locations)[0], "mla.py")
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

    if "LOCAL PATCH -- get_block_n_fp8" in original:
        print(f"{path} already patched; nothing to do")
        return

    if original.count(ANCHOR) != 1:
        die(
            f"get_block_n_fp8 anchor matched {original.count(ANCHOR)} times, "
            "expected 1 -- aiter drifted; re-derive before running."
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
    print(f"Patched {path}: get_block_n_fp8 nearest-key fallback")


if __name__ == "__main__":
    main()
