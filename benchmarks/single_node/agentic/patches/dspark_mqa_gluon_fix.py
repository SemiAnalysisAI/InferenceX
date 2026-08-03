#!/usr/bin/env python3
"""Fix the Kimi-K3 DSpark multi-token verify path in ROCM_AITER_MLA.

Target: vllm/v1/attention/backends/mla/rocm_aiter_mla.py inside the
`vllm/vllm-openai-rocm:kimi-k3` image (vLLM 0.1.dev19253+g5f76ae224, a private
AMD build -- that sha does not resolve in vllm-project/vllm, so the installed
file cannot be assumed byte-equal to anything upstream; every edit below is
anchored on an exact snippet and the script exits non-zero if an anchor is
missing).

The bug
-------
DSpark drafts 7 tokens, so the target-model verify step runs at
`max_qo_len = 8`. K3 has 96 attention heads => 12 heads/rank at TP8, below
AiterMLAHelper._AITER_MIN_MLA_HEADS (16), and the asm path has no
gqa<16 / qseqlen>1 kernel. `forward_mqa` therefore takes a "flatten each verify
token to its own qseqlen=1 gluon decode" branch. That branch faults:

    rocm_aiter_mla.py:1044 forward_mqa
      aiter/ops/triton/gluon/mla_gluon.py:937 mla_gluon
        RuntimeError: Triton Error [HIP]: Code 700, illegal memory access

on all 16 worker tracebacks, in runs 30499842314 / 30501609272 / 30503579730
(enforce_eager, so the traceback survives). Runs 30519967494 and 30524588650
show the same fault asynchronously under CUDA graphs, as
`HSA_STATUS_ERROR_EXCEPTION code: 0x1016`. One bug, three signatures.

Root cause
----------
`AiterMLADecodeMetadata.min_kv_seq_len` is declared with a default of 1 and is
**never assigned** by the decode-metadata builder. So the ordinary decode call
site passes `min_kv_seq_len=1`, and mla_gluon computes

    NUM_KV_SPLITS = max(1, min(256 // batch_size, cdiv(min_kv_seq_len, BLOCK_N)))
                  = 1

i.e. every non-spec decode on this fleet takes the single-split fast path where
stage-1 writes `o` directly and `_mla_softmax_reducev_kernel` never runs. The
DSpark flatten branch is the only caller that passes a real minimum (~300K at
our trace depth), so it is the only thing in the backend that has ever turned on
Gluon split-K + the stage-2 reduce in the `bh16bn64` regime.

Fixes applied
-------------
FIX 1 -- qo_indptr (correctness, blocks conc > 1)

  The builder's non-cudagraph branch sets `qo_indptr = arange(0, num_reqs + 1)`,
  which encodes "exactly one query token per request". That is true only when
  max_qo_len == 1. This script makes it mirror `query_start_loc_device`
  unconditionally; for max_qo_len == 1 the two are numerically identical, so the
  asm decode path is unaffected.

FIX 2 -- row->request mapping (correctness, blocks conc > 1)

  The stock flatten branch builds `new_indptr` with `num_reqs * max_qo_len` rows
  but the kernel is launched with `batch_size = q_nope.shape[0]`, which is
  sum(qo_len_i). Those differ as soon as one request has a shorter query than
  the max -- a request in its first decode step after prefill, or one finishing.
  Rows then attend the WRONG request's KV: in-bounds, so no crash, just silently
  wrong output. Invisible at conc 1, because num_reqs == 1 makes B == qlen
  identically -- which is exactly why c1 (run 30516895411) is the only cell that
  ever passed. Now derived from qo_indptr, with a hard assert against B.

FIX 3 -- no split-K (the crash)

  Pass `min_kv_seq_len=1` so this branch takes the same single-split path every
  other decode on this fleet already uses. Set DSPARK_MQA_SPLITK=1 to restore
  the stock behaviour, which is the A/B that confirms split-K as the fault.

FIX 4 -- hoist the expansion out of the per-layer path (the OOM)

  The stock branch materializes an int64 `arange(total)` plus a gather of size
  max_qo_len * sum(seq_len) on EVERY MLA layer of EVERY step, and forces a
  device sync with `.item()`. At --max-num-seqs 128 that is the 13.99 GiB
  `torch.OutOfMemoryError` in run 30512072425. The expansion depends only on the
  step's metadata, not on the layer, so it is computed once and cached on the
  decode-metadata object -- 24 MLA layers, so ~24x less allocation and one sync
  per step instead of 24.

Fails loudly: a silently-unpatched DSpark run reproduces a known crash on a
whole MI355X node for nothing.
"""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import os
import sys

REL = "vllm/v1/attention/backends/mla/rocm_aiter_mla.py"


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def find_target() -> str:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        die("vllm is not importable in this interpreter")
    root = os.path.dirname(list(spec.submodule_search_locations)[0])
    path = os.path.join(root, REL)
    if not os.path.isfile(path):
        die(f"{path} not found")
    return path


# --------------------------------------------------------------------------
# FIX 1: qo_indptr must carry real query offsets, not a one-token-per-request
# placeholder, because FIX 2 derives the row->request map from it.
# --------------------------------------------------------------------------
QO_OLD = """        else:
            qo_indptr = torch.arange(
                0, num_reqs + 1, step=1, dtype=torch.int32, device=device
            )
"""

QO_NEW = """        else:
            # LOCAL FIX (DSpark): the stock placeholder encodes "one query token
            # per request", true only when max_qo_len == 1. forward_mqa's
            # multi-token verify path needs real per-request query offsets to map
            # verify rows back to requests. For max_qo_len == 1 this is
            # numerically identical to the arange it replaces, so the asm decode
            # path is unaffected.
            qo_indptr = query_start_loc_device[: num_reqs + 1].to(torch.int32)
"""

# --------------------------------------------------------------------------
# FIX 2/3/4: replace the flatten branch wholesale.
# --------------------------------------------------------------------------
MQA_OLD = """            qlen = int(decode.max_qo_len)
            if type(q) is tuple:
                q_nope, q_pe = q
            else:
                q_nope, q_pe = torch.split(
                    q, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
                )
            B, num_q_heads, _ = q_nope.shape
            o = torch.empty(
                B,
                num_q_heads,
                self.kv_lora_rank,
                dtype=decode.attn_out_dtype,
                device=q_nope.device,
            )
            kv_buffer = kv_c_and_k_pe_cache.reshape(-1, kv_c_and_k_pe_cache.shape[-1])
            # Expand per-request paged-KV to per-verify-token: each request's KV
            # block is repeated qlen times (row r*qlen+t attends request r's
            # committed prefix). Fully vectorized (no host loop).
            old_indptr = decode.paged_kv_indptr
            per_req_len = old_indptr[1:] - old_indptr[:-1]
            dev = q_nope.device
            row_req = torch.arange(
                per_req_len.shape[0], device=dev
            ).repeat_interleave(qlen)
            row_len = per_req_len[row_req]
            new_indptr = torch.cat(
                [old_indptr.new_zeros(1), row_len.cumsum(0)]
            ).to(torch.int32)
            total = int(new_indptr[-1].item())
            within = torch.arange(
                total, device=dev, dtype=torch.int64
            ) - new_indptr[:-1].to(torch.int64).repeat_interleave(row_len)
            src = old_indptr[row_req].to(torch.int64).repeat_interleave(row_len) + within
            new_indices = decode.paged_kv_indices[src]
            mla_gluon = _get_mla_gluon()
            mla_gluon(
                q_nope=q_nope,
                q_pe=q_pe,
                kv_c=kv_buffer,
                o=o,
                page_table=new_indices,
                seq_info=new_indptr,
                sm_scale=self.scale,
                k_pe=None,
                kv_pe_offset=self.kv_lora_rank,
                use_2d_view=False,
                kv_scale=1.0,
                min_kv_seq_len=int(per_req_len.min()),
            )
            return o, None
"""

MQA_NEW = '''            if type(q) is tuple:
                q_nope, q_pe = q
            else:
                q_nope, q_pe = torch.split(
                    q, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
                )
            B, num_q_heads, _ = q_nope.shape
            o = torch.empty(
                B,
                num_q_heads,
                self.kv_lora_rank,
                dtype=decode.attn_out_dtype,
                device=q_nope.device,
            )
            kv_buffer = kv_c_and_k_pe_cache.reshape(-1, kv_c_and_k_pe_cache.shape[-1])
            new_indices, new_indptr, min_len = _dspark_mqa_expand(decode, B)
            mla_gluon = _get_mla_gluon()
            mla_gluon(
                q_nope=q_nope,
                q_pe=q_pe,
                kv_c=kv_buffer,
                o=o,
                page_table=new_indices,
                seq_info=new_indptr,
                sm_scale=self.scale,
                k_pe=None,
                kv_pe_offset=self.kv_lora_rank,
                use_2d_view=False,
                kv_scale=1.0,
                # LOCAL FIX 3: min_kv_seq_len drives
                #   NUM_KV_SPLITS = max(1, min(256 // B, cdiv(min_kv, BLOCK_N)))
                # in mla_gluon. Because the metadata builder never assigns
                # AiterMLADecodeMetadata.min_kv_seq_len (it keeps its default of
                # 1), every other caller in this backend gets NUM_KV_SPLITS == 1
                # and takes the single-split fast path. Passing the real minimum
                # here made this branch the ONLY user of Gluon split-K + the
                # stage-2 reduce in the bh16bn64 regime -- untested code, and the
                # site of the HIP 700 / HSA 0x1016 fault. Take the same path
                # everything else takes. DSPARK_MQA_SPLITK=1 restores the stock
                # behaviour for the confirming A/B.
                min_kv_seq_len=min_len,
            )
            return o, None
'''

# --------------------------------------------------------------------------
# Helper injected at module scope, just above `class AiterMLAImpl`.
# --------------------------------------------------------------------------
HELPER_ANCHOR = "class AiterMLAImpl(MLACommonImpl[AiterMLAMetadata]):"

HELPER = '''import os as _dspark_os

_DSPARK_MQA_SPLITK = _dspark_os.environ.get("DSPARK_MQA_SPLITK", "0") == "1"


def _dspark_mqa_expand(decode, num_rows: int):
    """Expand per-request paged KV metadata to per-verify-token rows.

    Row r of the flattened verify batch attends the full committed prefix of
    whichever request produced it, so each request's KV index range is repeated
    once per query token it contributed.

    Returns (page_table, seq_info, min_kv_seq_len) ready for mla_gluon.

    Cached on the decode-metadata object: the expansion depends only on the
    step's metadata, and every MLA layer in the step asks for the same thing.
    Recomputing it per layer is what produced the 13.99 GiB OOM in run
    30512072425 (an int64 arange plus a gather of max_qo_len * sum(seq_len), 24
    times per step, each preceded by a .item() device sync).
    """
    cached = getattr(decode, "_dspark_mqa_cache", None)
    if cached is not None and cached[0] == num_rows:
        return cached[1], cached[2], cached[3]

    old_indptr = decode.paged_kv_indptr
    per_req_len = old_indptr[1:] - old_indptr[:-1]
    num_reqs = per_req_len.shape[0]
    dev = old_indptr.device

    # LOCAL FIX 2: derive the row -> request map from the real query offsets.
    # The stock code assumed every request contributes exactly max_qo_len rows,
    # which is false whenever one request has a shorter query than the max (a
    # request in its first decode step after prefill, or one finishing). It is
    # true identically at num_reqs == 1, which is why only conc 1 ever passed.
    qo_indptr = decode.qo_indptr
    qo_len = (qo_indptr[1 : num_reqs + 1] - qo_indptr[:num_reqs]).to(torch.int64)
    row_req = torch.repeat_interleave(torch.arange(num_reqs, device=dev), qo_len)
    if row_req.shape[0] != num_rows:
        raise AssertionError(
            "DSpark MQA row map disagrees with the query batch: qo_indptr implies "
            f"{row_req.shape[0]} rows, q has {num_rows}. qo_indptr is stale or is "
            "still the one-token-per-request placeholder (FIX 1 not applied?)."
        )

    row_len = per_req_len[row_req]
    new_indptr = torch.cat([old_indptr.new_zeros(1), row_len.cumsum(0)]).to(
        torch.int32
    )
    total = int(new_indptr[-1].item())
    within = torch.arange(total, device=dev, dtype=torch.int64) - new_indptr[
        :-1
    ].to(torch.int64).repeat_interleave(row_len)
    src = old_indptr[row_req].to(torch.int64).repeat_interleave(row_len) + within
    new_indices = decode.paged_kv_indices[src]

    min_len = int(per_req_len.min()) if _DSPARK_MQA_SPLITK else 1

    try:
        decode._dspark_mqa_cache = (num_rows, new_indices, new_indptr, min_len)
    except (AttributeError, TypeError):
        # frozen/slotted dataclass -- correctness is unaffected, we just pay the
        # per-layer cost the way the stock code did.
        pass
    return new_indices, new_indptr, min_len


'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff-out")
    ap.add_argument(
        "--check",
        action="store_true",
        help="verify anchors and print the patched result without writing",
    )
    args = ap.parse_args()

    path = find_target()
    original = open(path, encoding="utf-8").read()
    text = original

    if "_dspark_mqa_expand" in text:
        print(f"{path} already patched; nothing to do")
        return

    for name, old in (("qo_indptr", QO_OLD), ("forward_mqa flatten", MQA_OLD)):
        if text.count(old) != 1:
            die(
                f"anchor for {name} matched {text.count(old)} times, expected 1. "
                "The image drifted; re-derive the anchors before running."
            )
    if text.count(HELPER_ANCHOR) != 1:
        die("helper anchor (class AiterMLAImpl) not found exactly once")

    text = text.replace(QO_OLD, QO_NEW)
    text = text.replace(MQA_OLD, MQA_NEW)
    text = text.replace(HELPER_ANCHOR, HELPER + HELPER_ANCHOR)

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
        print("--check: anchors OK, result compiles; not written")
        return

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"Patched {path}")
    print(
        "DSPARK_MQA_SPLITK="
        + ("1 (stock split-K restored)" if os.environ.get("DSPARK_MQA_SPLITK") == "1" else "0 (single-split)")
    )


if __name__ == "__main__":
    main()
