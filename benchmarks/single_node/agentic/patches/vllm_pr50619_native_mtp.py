#!/usr/bin/env python3
"""Cherry-pick vllm-project/vllm#50619 onto the kimi-k3 image's AITER MLA backend.

PR 50619 ("[ROCm][Perf] Fix Kimi-K3 DSpark FP8 MLA verification") replaces the
expanded-row DSpark verify path with mla_gluon's native 4-D MTP layout
``[batch, qlen, nhead, dim]``, which the kernel masks with a causal tail: query
position ``t`` attends KV ``[0, seq_len - qlen + t]``.

Why this matters more than the 8x it buys
-----------------------------------------
The image's flatten branch is **non-causal**, and says so in its own comment.
It builds ``row_len = per_req_len[row_req]`` -- every verify row gets the
request's *entire* KV range. ``paged_kv_indptr`` is ``cumsum(seq_lens)`` in
tokens (``_expand_page_indices_kernel`` flattens block indices to token
indices, and the builder notes "one page per token in the flat view"), and
``seq_lens`` already counts the tokens scheduled this step. So draft token t
attends draft tokens t+1..qlen-1: the target sees the draft's own future
tokens and therefore agrees with them.

That is why every measurement on this path reported acceptance 6.41-8.00 of 8,
and 8.00 flat at conc 1. Those numbers are leakage, not quality. Upstream's
pre-50619 code carries the opposite convention with an explicit comment
("position t must not see t+1") and PR 50619 keeps it, scoring GSM8K 1271/1319
against a 1269/1319 no-speculation baseline. Acceptance alone cannot separate
"correct" from "agreeing with itself"; the quality gate can.

Deltas against the PR as written
--------------------------------
1. **Uniform-qlen fallback.** The PR asserts ``num_tokens % qlen == 0`` and
   raises otherwise, because its base (upstream 38a466e7b) carries
   ``_uniform_padded_mtp_qo_len`` / ``pad_uniform_mtp`` in the metadata builder
   -- 1241 lines against this image's 1109. This image has none of that. Rather
   than drag in unrelated upstream drift, a mixed-query-length step falls back
   to the flatten path *with the causal row length* (row t of request r spans
   ``seq_len_r - qo_len_r + 1 + t``), which generalizes the PR's formula to a
   per-request query length. Semantics are identical either way; only speed
   differs, and the steady-state verify step is uniform.

2. **Real per-request query lengths.** The image's builder only fills
   ``qo_indptr`` from ``query_start_loc`` under full cudagraphs; otherwise it is
   ``arange(num_reqs + 1)``, a one-token-per-request placeholder. The fallback
   in (1) needs the true lengths, and deciding uniformity must not cost a
   device sync per MLA layer. Both are solved in the builder, where
   ``query_start_loc_cpu`` is already differenced: stash ``mtp_uniform`` (a
   host-side bool) and ``mtp_qo_len`` (one small H2D per step).

3. **fp8 query capability only.** The PR's companion edit to
   ``kimi_k3/nvidia/mla.py`` calls ``self.impl.do_kv_cache_update``, which does
   not exist on this image's MLA impl (it is defined only on the non-MLA
   backends). The capability flag here is inert while KV is bf16
   -- ``is_quantized_kv_cache`` is False -- so it ships now and the fp8 arm
   lands with the rope-outside cache-write port.

Not ported: the ``attn_utils`` / ``model_runner`` draft-cudagraph exclusion.
Both sites exist and match, but changing the target's graph mode would confound
the A/B against run 30875699986. Separate change.

Requires the vendored Gluon kernel (MLA_GLUON_PATCH=1); the image's stock
mla_gluon has almost no qlen support and cannot take a 4-D query.
"""

from __future__ import annotations

import argparse
import difflib
import os
import sys

REL = "vllm/v1/attention/backends/mla/rocm_aiter_mla.py"


def die(msg: str) -> None:
    sys.exit(f"vllm_pr50619_native_mtp.py: {msg}")


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
# 1. Metadata fields: uniformity verdict + real per-request query lengths.
# --------------------------------------------------------------------------
FIELDS_OLD = """    # The max query output length: int
    max_qo_len: int | None = None
"""

FIELDS_NEW = """    # The max query output length: int
    max_qo_len: int | None = None
    # PR 50619: whether every request in this step contributes exactly
    # max_qo_len query tokens. Decided host-side in the builder from
    # query_start_loc_cpu, because forward_mqa runs once per MLA layer and
    # must not pay a device sync to find out.
    mtp_uniform: bool = False
    # PR 50619: true per-request query lengths, int64 [num_reqs]. qo_indptr is
    # only real under full cudagraphs (otherwise arange), and the non-uniform
    # causal fallback needs the actual lengths.
    mtp_qo_len: torch.Tensor | None = None
"""

# --------------------------------------------------------------------------
# 2. Builder: compute both, host-side, where qo_len already exists.
# --------------------------------------------------------------------------
BUILD_OLD = """        qo_len = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        max_qo_len = qo_len.max().item()
"""

BUILD_NEW = """        qo_len = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        max_qo_len = qo_len.max().item()
        # PR 50619: this is the only place per-request query lengths are
        # available without a device sync. num_reqs counts padded request
        # slots and query_start_loc_cpu does not, so reconcile the two first:
        # a padded slot has seq_len 0, and giving it the uniform length lets
        # its rows clamp to empty instead of desyncing the row map.
        mtp_qo_len_cpu = qo_len
        if mtp_qo_len_cpu.numel() < num_reqs:
            mtp_qo_len_cpu = torch.cat(
                [
                    mtp_qo_len_cpu,
                    mtp_qo_len_cpu.new_full(
                        (num_reqs - mtp_qo_len_cpu.numel(),), max_qo_len
                    ),
                ]
            )
        elif mtp_qo_len_cpu.numel() > num_reqs:
            mtp_qo_len_cpu = mtp_qo_len_cpu[:num_reqs]
        mtp_uniform = bool((mtp_qo_len_cpu == max_qo_len).all().item())
        mtp_qo_len = mtp_qo_len_cpu.to(
            device=device, dtype=torch.int64, non_blocking=True
        )
"""

CTOR_OLD = """            max_qo_len=max_qo_len,
            use_gluon_decode=use_gluon_decode,
"""

CTOR_NEW = """            max_qo_len=max_qo_len,
            mtp_uniform=mtp_uniform,
            mtp_qo_len=mtp_qo_len,
            use_gluon_decode=use_gluon_decode,
"""

# --------------------------------------------------------------------------
# 3. fp8 query capability (inert under bf16 KV).
# --------------------------------------------------------------------------
INIT_OLD = """        AiterMLAHelper.check_num_heads_validity(num_heads)

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
"""

INIT_NEW = """        AiterMLAHelper.check_num_heads_validity(num_heads)

        # PR 50619: decode with fewer than 16 heads is served by Gluon, whose
        # fp8 KV regime (bh16bn128) takes a bf16 query and folds the KV dequant
        # scale into the QK temperature. A pre-quantized query is rejected with
        # "q_nope/q_pe must be bf16", so tell the common layer to leave the
        # query alone -- as TritonMLAImpl already does for its own fp8 KV path.
        # No-op while KV is bf16.
        if num_heads < AiterMLAHelper._AITER_MIN_MLA_HEADS and is_quantized_kv_cache(
            self.kv_cache_dtype
        ):
            self.supports_quant_query_input = False

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
"""

IMPORT_OLD = """from vllm.triton_utils import tl, triton
"""

IMPORT_NEW = """from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import is_quantized_kv_cache
"""

# --------------------------------------------------------------------------
# 4. forward_mqa: one merged Gluon branch, causal in both regimes.
# --------------------------------------------------------------------------
MQA_OLD_HEAD = """        decode = attn_metadata.decode
        if decode.use_gluon_decode:"""

MQA_NEW = '''        decode = attn_metadata.decode
        if decode.use_gluon_decode or (
            self.num_heads < AiterMLAHelper._AITER_MIN_MLA_HEADS
            and int(decode.max_qo_len) > 1
        ):
            if type(q) is tuple:
                q_nope, q_pe = q
            else:
                q_nope, q_pe = torch.split(
                    q, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
                )
            num_tokens, num_q_heads, _ = q_nope.shape
            qlen = int(decode.max_qo_len)
            num_reqs = decode.paged_kv_indptr.shape[0] - 1
            kv_buffer = kv_c_and_k_pe_cache.reshape(-1, kv_c_and_k_pe_cache.shape[-1])

            if qlen == 1:
                # Plain single-token decode, unchanged.
                page_table = decode.paged_kv_indices
                seq_info = decode.paged_kv_indptr
                min_len = decode.min_kv_seq_len
            elif getattr(decode, "mtp_uniform", False) and num_tokens == num_reqs * qlen:
                # PR 50619 native MTP. Gluon reads the 4-D query and applies the
                # causal tail itself, so the original paged-KV metadata goes
                # straight through: batch stays num_reqs instead of becoming
                # num_reqs*qlen, and the per-layer index expansion disappears.
                q_nope = q_nope.view(num_reqs, qlen, num_q_heads, self.kv_lora_rank)
                q_pe = q_pe.view(num_reqs, qlen, num_q_heads, self.qk_rope_head_dim)
                page_table = decode.paged_kv_indices
                seq_info = decode.paged_kv_indptr
                min_len = decode.min_kv_seq_len
            else:
                # Mixed query lengths: no 4-D layout to take, so flatten -- but
                # causally, which the image's branch does not do.
                page_table, seq_info, min_len = _pr50619_causal_flatten(
                    decode, num_tokens
                )

            o = torch.empty(
                *q_nope.shape[:-1],
                self.kv_lora_rank,
                dtype=decode.attn_out_dtype,
                device=q_nope.device,
            )
            mla_gluon = _get_mla_gluon()
            mla_gluon(
                q_nope=q_nope,
                q_pe=q_pe,
                kv_c=kv_buffer,
                o=o,
                page_table=page_table,
                seq_info=seq_info,
                sm_scale=self.scale,
                k_pe=None,
                kv_pe_offset=self.kv_lora_rank,
                use_2d_view=False,
                kv_scale=1.0,
                min_kv_seq_len=min_len,
            )
            return o.view(num_tokens, num_q_heads, self.kv_lora_rank), None

        if False:'''

# --------------------------------------------------------------------------
# Helper injected at module scope, just above `class AiterMLAImpl`.
# --------------------------------------------------------------------------
HELPER_ANCHOR = "class AiterMLAImpl(MLACommonImpl[AiterMLAMetadata]):"

HELPER = '''def _pr50619_causal_flatten(decode, num_rows: int):
    """Causal per-verify-token expansion for mixed query lengths.

    Row t of request r attends ``[0, context_r + t]`` where
    ``context_r = seq_len_r - qo_len_r``, i.e. ``context_r + 1 + t`` entries.
    That is PR 50619's row length generalized from the uniform ``qlen`` to the
    request's own query length. paged_kv_indices lists a request's tokens in
    ascending position order, so each row's causal window is a prefix of that
    request's slice and only the row length changes. Padding requests have
    seq_len 0 and clamp to an empty row.

    Cached on the decode metadata: every MLA layer in the step asks for the
    same expansion, and recomputing it per layer is what produced the 13.99 GiB
    OOM in run 30512072425.
    """
    cached = getattr(decode, "_pr50619_flat_cache", None)
    if cached is not None and cached[0] == num_rows:
        return cached[1], cached[2], cached[3]

    old_indptr = decode.paged_kv_indptr
    dev = old_indptr.device
    per_req_len = (old_indptr[1:] - old_indptr[:-1]).to(torch.int64)
    num_reqs = per_req_len.shape[0]

    qo_len = decode.mtp_qo_len
    if qo_len is None:
        raise AssertionError(
            "PR 50619 causal flatten needs mtp_qo_len, which the metadata "
            "builder should have set. The builder patch did not apply."
        )
    qo_len = qo_len.to(torch.int64)
    if qo_len.shape[0] != num_reqs:
        raise AssertionError(
            f"mtp_qo_len has {qo_len.shape[0]} entries against {num_reqs} "
            "paged-KV requests; metadata is inconsistent."
        )

    row_req = torch.repeat_interleave(torch.arange(num_reqs, device=dev), qo_len)
    if row_req.shape[0] != num_rows:
        raise AssertionError(
            f"query lengths imply {row_req.shape[0]} rows, q has {num_rows}."
        )
    # Position of each row within its own request's verify block.
    starts = torch.cumsum(qo_len, 0) - qo_len
    within_req = torch.arange(num_rows, device=dev) - starts.repeat_interleave(qo_len)
    row_len = (
        per_req_len[row_req] - qo_len[row_req] + 1 + within_req
    ).clamp_(min=0)

    new_indptr = torch.cat([old_indptr.new_zeros(1), row_len.cumsum(0)]).to(
        torch.int32
    )
    total = int(new_indptr[-1].item())
    within = torch.arange(total, device=dev, dtype=torch.int64) - new_indptr[
        :-1
    ].to(torch.int64).repeat_interleave(row_len)
    src = old_indptr[row_req].to(torch.int64).repeat_interleave(row_len) + within
    new_indices = decode.paged_kv_indices[src]

    # Take the single-split path, as every other caller in this backend does.
    min_len = 1

    try:
        decode._pr50619_flat_cache = (num_rows, new_indices, new_indptr, min_len)
    except (AttributeError, TypeError):
        pass
    return new_indices, new_indptr, min_len


'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff-out")
    ap.add_argument("--check", action="store_true")
    ap.add_argument(
        "--target",
        help="patch this file instead of the installed vllm (offline anchor check)",
    )
    args = ap.parse_args()

    path = find_target(args.target)
    original = open(path, encoding="utf-8").read()
    text = original

    if "_pr50619_causal_flatten" in text:
        print(f"{path} already patched; nothing to do")
        return

    if "_dspark_mqa_expand" in text:
        die(
            "dspark_mqa_gluon_fix.py is already applied to this file. The two "
            "patchers own the same branch and are mutually exclusive; set "
            "DSPARK_MQA_FIX=0 when DSPARK_PR50619=1."
        )

    anchors = [
        ("decode metadata fields", FIELDS_OLD),
        ("builder qo_len", BUILD_OLD),
        ("decode metadata ctor", CTOR_OLD),
        ("AiterMLAImpl.__init__", INIT_OLD),
        ("triton_utils import", IMPORT_OLD),
        ("forward_mqa gluon branch", MQA_OLD_HEAD),
        ("helper anchor", HELPER_ANCHOR),
    ]
    for name, old in anchors:
        if text.count(old) != 1:
            die(
                f"anchor for {name} matched {text.count(old)} times, expected 1. "
                "The image drifted; re-derive the anchors before running."
            )

    text = text.replace(FIELDS_OLD, FIELDS_NEW)
    text = text.replace(BUILD_OLD, BUILD_NEW)
    text = text.replace(CTOR_OLD, CTOR_NEW)
    text = text.replace(INIT_OLD, INIT_NEW)
    text = text.replace(IMPORT_OLD, IMPORT_NEW)
    # The old gluon-decode branch and the old flatten branch both become dead
    # code under `if False:`; they are left in place rather than excised so the
    # diff stays reviewable against the image.
    text = text.replace(MQA_OLD_HEAD, MQA_NEW)
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
    if args.check:
        print(diff)
        return

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"patched {path} with vllm-project/vllm#50619 (causal native MTP verify)")


if __name__ == "__main__":
    main()
