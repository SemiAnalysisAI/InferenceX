#!/usr/bin/env python3
"""Route DSpark verify through mla_gluon's NATIVE MTP path instead of flattening.

The root cause this removes
---------------------------
`forward_mqa` flattens the k+1 verify tokens into fake batch rows: it calls
mla_gluon with a 3-D q of shape [num_reqs * qlen, nhead, dim] and rebuilds an
expanded page table so each row re-attends its request's prefix.

But mla_gluon already accepts multi-token queries natively:

    q_nope,  # [batch, nhead, kv_lora_rank] or MTP [batch, qlen, nhead, kv_lora_rank]
    o,       # [batch, nhead, kv_lora_rank] or MTP [batch, qlen, nhead, kv_lora_rank]

    # Accept plain decode (3-D, qlen=1) or MTP (4-D, [batch, qlen, nhead, dim]).
    batch_size, qlen, nhead, head_dim_ckv = q_nope.shape

Present in the gist kernel this recipe installs AND in aiter main. vLLM simply
does not use it. That one choice produces, directly:

  * batch = num_reqs * qlen instead of num_reqs. At mns 16 / k 7 that is 144 and
    at capture time 225, against mla_gluon's 128 ceiling -- runs 30786158942
    ("got 225") and 30786908516 ("got 144") died on exactly this.
  * a per-layer int64 arange + gather of size qlen * sum(seq_len), which is the
    13.99 GiB OOM in run 30512072425.
  * a uniform-qlen assumption that silently mapped rows to the wrong request
    (fixed separately, but only because the flattening created the problem).

Passing the 4-D layout makes batch = num_reqs (8 at c8), needs no page-table
expansion at all -- the original paged_kv_indices / paged_kv_indptr are used
as-is -- and deletes all three failure modes together.

Semantics: this is a REAL change, not a refactor
------------------------------------------------
The MTP kernel is causal along qlen: "Each query position q_pos attends KV
[0, seq_len-qlen+q_pos] (causal tail)." The flatten branch is explicitly
non-causal -- every verify token attends the same full range.

Causal-tail is the standard spec-decode verify semantics: at verify time the
draft tokens are already in the KV cache, so verify position i must see the
prefix plus drafts 0..i-1 and NOT drafts after it. The flatten branch lets every
position see all drafts, which is the wrong direction (a position can attend its
own future). If the flatten branch's numbers were right, it was despite this.

ACCEPTANCE LENGTH IS THE TEST. Gluon verify on the flatten path measured
6.41-8.00 with monotonic per-position decay and 94.6% average. If this patch is
semantically correct, acceptance should hold or improve. If it collapses toward
1, the causality assumption is wrong and this must be reverted -- do not judge
this arm on throughput.

Enabled by DSPARK_MTP_NATIVE=1. Requires the legacy flatten branch to be present
(the kimi-k3 image); newer vLLM builds that already pad to a uniform MTP qo_len
have a different layout and need their own anchor. Fails loudly either way.
"""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import os
import sys

REL = "vllm/v1/attention/backends/mla/rocm_aiter_mla.py"

# The whole flatten body, from qlen= through the return. Must match the stock
# kimi-k3 image; dspark_mqa_gluon_fix.py must NOT have run first.
OLD = """            qlen = int(decode.max_qo_len)
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

NEW = '''            qlen = int(decode.max_qo_len)
            if type(q) is tuple:
                q_nope, q_pe = q
            else:
                q_nope, q_pe = torch.split(
                    q, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
                )
            B, num_q_heads, _ = q_nope.shape
            kv_buffer = kv_c_and_k_pe_cache.reshape(-1, kv_c_and_k_pe_cache.shape[-1])
            old_indptr = decode.paged_kv_indptr
            per_req_len = old_indptr[1:] - old_indptr[:-1]
            num_reqs = per_req_len.shape[0]
            mla_gluon = _get_mla_gluon()

            # LOCAL PATCH -- native MTP path (DSPARK_MTP_NATIVE=1).
            #
            # mla_gluon accepts a 4-D query [batch, qlen, nhead, dim] directly.
            # Using it keeps batch == num_reqs instead of num_reqs * qlen, so the
            # 128 batch ceiling is never approached, and it needs NO page-table
            # expansion: the original paged_kv_indices / paged_kv_indptr are
            # passed straight through, removing the per-layer int64 arange +
            # gather that caused the 13.99 GiB OOM.
            #
            # Only valid when the rows really are num_reqs x qlen. vLLM can
            # schedule a shorter query for a request that just finished prefill,
            # and then the reshape would silently mis-associate rows -- so this
            # falls back to the flatten path rather than guess.
            if _DSPARK_MTP_NATIVE and B == num_reqs * qlen:
                o4 = torch.empty(
                    num_reqs,
                    qlen,
                    num_q_heads,
                    self.kv_lora_rank,
                    dtype=decode.attn_out_dtype,
                    device=q_nope.device,
                )
                mla_gluon(
                    q_nope=q_nope.view(num_reqs, qlen, num_q_heads, -1),
                    q_pe=q_pe.view(num_reqs, qlen, num_q_heads, -1),
                    kv_c=kv_buffer,
                    o=o4,
                    page_table=decode.paged_kv_indices,
                    seq_info=old_indptr,
                    sm_scale=self.scale,
                    k_pe=None,
                    kv_pe_offset=self.kv_lora_rank,
                    use_2d_view=False,
                    kv_scale=1.0,
                    # Single-split, matching every other decode on this fleet.
                    # The builder never assigns min_kv_seq_len, so passing a real
                    # value here would make this the only caller using split-K.
                    min_kv_seq_len=1,
                )
                return o4.view(B, num_q_heads, self.kv_lora_rank), None

            o = torch.empty(
                B,
                num_q_heads,
                self.kv_lora_rank,
                dtype=decode.attn_out_dtype,
                device=q_nope.device,
            )
            dev = q_nope.device
            row_req = torch.arange(num_reqs, device=dev).repeat_interleave(qlen)
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
                min_kv_seq_len=1,
            )
            return o, None
'''

HELPER_ANCHOR = "class AiterMLAImpl(MLACommonImpl[AiterMLAMetadata]):"

FLAG = '''import os as _dspark_mtp_os

_DSPARK_MTP_NATIVE = _dspark_mtp_os.environ.get("DSPARK_MTP_NATIVE", "0") == "1"


'''


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff-out")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    path = find_target()
    original = open(path, encoding="utf-8").read()

    if "_DSPARK_MTP_NATIVE" in original:
        print(f"{path} already patched; nothing to do")
        return

    if original.count(OLD) != 1:
        die(
            f"flatten-branch anchor matched {original.count(OLD)} times, expected 1. "
            "Either this is not the legacy layout, or dspark_mqa_gluon_fix.py "
            "already rewrote the branch -- run this one INSTEAD of that, not after."
        )
    if original.count(HELPER_ANCHOR) != 1:
        die("class AiterMLAImpl anchor not found exactly once")

    text = original.replace(OLD, NEW).replace(HELPER_ANCHOR, FLAG + HELPER_ANCHOR)
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
    print(f"Patched {path}: native MTP verify path")
    print(
        "DSPARK_MTP_NATIVE="
        + ("1 (4-D MTP)" if os.environ.get("DSPARK_MTP_NATIVE") == "1" else "0 (flatten)")
    )


if __name__ == "__main__":
    main()
