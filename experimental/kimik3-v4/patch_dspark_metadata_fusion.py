#!/usr/bin/env python3
"""Apply staged DSpark decode-metadata launch reductions to an installed vLLM.

This is intentionally an exact, marker-gated patch for the pinned Kimi-K3 image.
Set ``K3_DSPARK_FUSION_STAGE`` to:

1. Share ``num_computed_tokens`` across KV-cache groups.
2. Reuse pure-spec GDN indices and replace redundant masked indexing with views.
3. Fuse Mamba block-table index generation and gather into one Triton kernel.
4. Share GDN query lengths and use views for pure-spec accepted-token rows.
5. Skip the Gluon-only flattened KV view when fp8 DSpark uses ASM verify.
6. Fuse uniform MLA indptr construction and block-table expansion.
7. Share identical AITER persistent MLA schedules across KV-cache groups.
8. Fuse shared query-length and computed-token arithmetic.
9. Fuse accepted-token initialization, gather, and graph-padding fill.
10. Fuse rejection-sampler token and position gathers.
11. Fuse context RoPE-key transpose with position replication.
12. Elide zero-offset rejection chunk normalization.
13. Fuse context latent-KV transpose with RMS normalization.
14. Reuse freshly proposed draft tokens instead of gathering them back.
15. Fold MoE padding-mask initialization into token combination.
16. Fuse persistent draft-token scatter into one Triton kernel.
"""

from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path


STAGE = int(os.environ.get("K3_DSPARK_FUSION_STAGE", "0"))
if STAGE == 0:
    print("K3 DSpark metadata fusion: disabled")
    raise SystemExit(0)
if STAGE not in range(1, 17):
    raise SystemExit("K3_DSPARK_FUSION_STAGE must be between 1 and 16")

spec = importlib.util.find_spec("vllm")
if spec is None or spec.origin is None:
    raise SystemExit("Could not locate the installed vLLM package")
ROOT = Path(spec.origin).parent


def replace_once(path: Path, old: str, new: str, marker: str) -> None:
    source = path.read_text()
    if marker in source:
        print(f"{path.relative_to(ROOT)}: {marker} already present")
        return
    count = source.count(old)
    if count != 1:
        if os.environ.get("K3_DSPARK_FUSION_SKIP_MISSING") == "1":
            print(f"WARN: skip {marker!r} ({count} contexts in {path.relative_to(ROOT)})")
            return
        raise RuntimeError(
            f"{path}: expected one patch context for {marker!r}, found {count}"
        )
    updated = source.replace(old, new, 1)
    ast.parse(updated, filename=str(path))
    path.write_text(updated)
    print(f"{path.relative_to(ROOT)}: applied {marker}")


def ensure_triton_import(path: Path) -> None:
    """Insert triton_utils when a kernel hunk landed but the import hunk did not."""
    source = path.read_text()
    if "@triton.jit" not in source:
        return
    if "from vllm.triton_utils import tl, triton" in source:
        return
    needle = "import torch\n"
    if needle not in source:
        if os.environ.get("K3_DSPARK_FUSION_SKIP_MISSING") == "1":
            print(f"WARN: skip triton import repair in {path.relative_to(ROOT)}")
            return
        raise RuntimeError(f"{path}: cannot insert triton import")
    updated = source.replace(
        needle,
        needle + "\nfrom vllm.triton_utils import tl, triton\n",
        1,
    )
    ast.parse(updated, filename=str(path))
    path.write_text(updated)
    print(f"{path.relative_to(ROOT)}: ensured triton import")


if STAGE >= 1:
    path = ROOT / "v1/worker/gpu/attn_utils.py"
    old = """\
    attn_metadata: dict[str, Any] = {}
    num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
    for i in range(num_kv_cache_groups):
"""
    new = """\
    attn_metadata: dict[str, Any] = {}
    num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)

    # K3 DSpark metadata fusion stage 1: every KV-cache group receives the same
    # query starts and sequence lengths. Compute this device tensor once instead
    # of rebuilding it in every CommonAttentionMetadata instance.
    shared_num_computed_tokens = seq_lens - (
        query_start_loc_gpu[1:] - query_start_loc_gpu[:-1]
    )

    for i in range(num_kv_cache_groups):
"""
    replace_once(path, old, new, "K3 DSpark metadata fusion stage 1")

    old = """\
            rswa_prefix_lens=rswa_prefix_lens,
            **common_attn_metadata_extra_kwargs,
"""
    new = """\
            rswa_prefix_lens=rswa_prefix_lens,
            _num_computed_tokens_cache=shared_num_computed_tokens,
            **common_attn_metadata_extra_kwargs,
"""
    replace_once(path, old, new, "_num_computed_tokens_cache=shared_num_computed_tokens")


if STAGE >= 2:
    path = ROOT / "v1/attention/backends/gdn_attn.py"
    old = """\
        self.spec_token_indx: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
"""
    new = """\
        # K3 DSpark metadata fusion stage 2: pure speculative decode uses a
        # contiguous token range every step. Materialize it once at startup.
        self.spec_token_indx: torch.Tensor = torch.arange(
            self.decode_cudagraph_max_bs * (self.num_spec + 1),
            dtype=torch.int32,
            device=device,
        )
"""
    replace_once(path, old, new, "K3 DSpark metadata fusion stage 2")

    old = """\
                spec_token_indx = torch.arange(
                    spec_token_size,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
"""
    new = """\
                # K3 DSpark stage 2 reuse: this range is invariant in pure-spec decode.
                spec_token_indx = self.spec_token_indx[:spec_token_size]
"""
    replace_once(path, old, new, "K3 DSpark stage 2 reuse")

    old = """\
                non_spec_token_indx = torch.empty(
                    0, dtype=torch.int32, device=query_start_loc.device
                )
                # Filter by spec_sequence_masks to exclude padded sequences
                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks_cpu, : self.num_spec + 1
                ]
"""
    new = """\
                non_spec_token_indx = torch.empty(
                    0, dtype=torch.int32, device=query_start_loc.device
                )
                # Real speculative rows are contiguous and padded rows are at
                # the back in this pure-spec branch, so masking is redundant.
                spec_state_indices_tensor = block_table_tensor[
                    :num_spec_decodes, : self.num_spec + 1
                ]
"""
    replace_once(
        path,
        old,
        new,
        "Real speculative rows are contiguous and padded rows are at",
    )


if STAGE >= 3:
    path = ROOT / "v1/attention/backends/utils.py"
    old = """\
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import PIN_MEMORY, async_tensor_h2d, np_to_pinned_tensor
"""
    new = """\
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import PIN_MEMORY, async_tensor_h2d, np_to_pinned_tensor
"""
    replace_once(path, old, new, "from vllm.triton_utils import tl, triton")

    old = """\
def mamba_get_block_table_tensor(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    kv_cache_spec: KVCacheSpec,
    mamba_cache_mode: str,
) -> torch.Tensor:
"""
    new = """\
@triton.jit
def _k3_mamba_block_table_kernel(
    output,
    block_table,
    seq_lens,
    block_table_stride,
    output_stride,
    mamba_block_size: tl.constexpr,
    num_output_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # K3 DSpark metadata fusion stage 3: one program handles one request and
    # fuses start-index arithmetic, clamping, dtype widening, and gather.
    req_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    seq_len = tl.load(seq_lens + req_idx).to(tl.int32)
    start = (seq_len - 1) // mamba_block_size
    start = tl.maximum(start, 0)
    mask = offsets < num_output_blocks
    values = tl.load(
        block_table + req_idx * block_table_stride + start + offsets,
        mask=mask,
        other=0,
    )
    tl.store(output + req_idx * output_stride + offsets, values, mask=mask)


def mamba_get_block_table_tensor(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    kv_cache_spec: KVCacheSpec,
    mamba_cache_mode: str,
) -> torch.Tensor:
"""
    replace_once(path, old, new, "K3 DSpark metadata fusion stage 3")

    old = """\
        # NOTE: For 0-length requests in CUDA graph, use a start_index of 0
        # to handle the invalid block table.
        start_indices = (seq_lens - 1) // kv_cache_spec.block_size
        start_indices.clamp_(min=0)
        # Use int32 for arithmetic to avoid dtype promotion overhead,
        # then convert to int64 for gather (which requires Long indices)
        offsets = torch.arange(
            1 + kv_cache_spec.num_speculative_blocks,
            device=block_table.device,
            dtype=torch.int32,
        )
        indices_to_gather = (start_indices.unsqueeze(1) + offsets).to(torch.int64)
        return torch.gather(block_table, 1, indices_to_gather)
"""
    new = """\
        num_output_blocks = 1 + kv_cache_spec.num_speculative_blocks
        output = torch.empty(
            (seq_lens.shape[0], num_output_blocks),
            dtype=block_table.dtype,
            device=block_table.device,
        )
        _k3_mamba_block_table_kernel[(seq_lens.shape[0],)](
            output,
            block_table,
            seq_lens,
            block_table.stride(0),
            output.stride(0),
            mamba_block_size=kv_cache_spec.block_size,
            num_output_blocks=num_output_blocks,
            BLOCK_N=triton.next_power_of_2(num_output_blocks),
        )
        return output
"""
    replace_once(
        path,
        old,
        new,
        "_k3_mamba_block_table_kernel[(seq_lens.shape[0],)]",
    )


if STAGE >= 4:
    path = ROOT / "v1/attention/backend.py"
    old = """\
    _num_computed_tokens_cache: torch.Tensor | None = None
    _token_to_req_indices_cache: torch.Tensor | None = None
"""
    new = """\
    # K3 DSpark metadata fusion stage 4: builders for different KV-cache groups
    # share the same query starts, so cache query lengths alongside computed tokens.
    _query_lens_cache: torch.Tensor | None = None
    _num_computed_tokens_cache: torch.Tensor | None = None
    _token_to_req_indices_cache: torch.Tensor | None = None
"""
    replace_once(path, old, new, "K3 DSpark metadata fusion stage 4")

    old = """\
    def naive_query_lens(self) -> torch.Tensor:
        \"\"\"Naive because it assumes that query ends where the next query starts.\"\"\"
        return self.query_start_loc[1:] - self.query_start_loc[:-1]
"""
    new = """\
    def naive_query_lens(self) -> torch.Tensor:
        \"\"\"Naive because it assumes that query ends where the next query starts.\"\"\"
        if self._query_lens_cache is None:
            self._query_lens_cache = (
                self.query_start_loc[1:] - self.query_start_loc[:-1]
            )
        return self._query_lens_cache
"""
    replace_once(path, old, new, "if self._query_lens_cache is None")

    path = ROOT / "v1/worker/gpu/attn_utils.py"
    old = """\
    shared_num_computed_tokens = seq_lens - (
        query_start_loc_gpu[1:] - query_start_loc_gpu[:-1]
    )
"""
    new = """\
    shared_query_lens = query_start_loc_gpu[1:] - query_start_loc_gpu[:-1]
    shared_num_computed_tokens = seq_lens - shared_query_lens
"""
    replace_once(path, old, new, "shared_query_lens = query_start_loc_gpu")

    old = """\
            rswa_prefix_lens=rswa_prefix_lens,
            _num_computed_tokens_cache=shared_num_computed_tokens,
"""
    new = """\
            rswa_prefix_lens=rswa_prefix_lens,
            _query_lens_cache=shared_query_lens,
            _num_computed_tokens_cache=shared_num_computed_tokens,
"""
    replace_once(path, old, new, "_query_lens_cache=shared_query_lens")

    path = ROOT / "v1/attention/backends/gdn_attn.py"
    old = """\
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
"""
    new = """\
            query_lens = m.naive_query_lens()
"""
    replace_once(path, old, new, "query_lens = m.naive_query_lens()")

    old = """\
            assert num_accepted_tokens is not None
            num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]
"""
    new = """\
            assert num_accepted_tokens is not None
            # Pure-spec real rows are contiguous with graph-padding rows at the
            # back; mixed batches retain the general boolean-index path.
            num_accepted_tokens = (
                num_accepted_tokens[:num_spec_decodes]
                if num_prefills == 0 and num_decodes == 0
                else num_accepted_tokens[spec_sequence_masks_cpu]
            )
"""
    replace_once(path, old, new, "Pure-spec real rows are contiguous with graph-padding")


if STAGE >= 5:
    path = ROOT / "v1/attention/backends/mla/rocm_aiter_mla.py"
    old = """\
        if self._flat_kv_enabled and max_qo_len > 1:
"""
    new = """\
        # K3 DSpark metadata fusion stage 5: this flattened causal KV view is
        # consumed only by the Gluon verify branch. FP8 DSpark is routed to the
        # ASM persistent verify, so building the view there is pure overhead.
        if (
            self._flat_kv_enabled
            and max_qo_len > 1
            and AiterMLAHelper.use_gluon_verify(
                self.num_heads, max_qo_len, self._kv_cache_dtype_str
            )
        ):
"""
    replace_once(path, old, new, "K3 DSpark metadata fusion stage 5")


if STAGE >= 6:
    path = ROOT / "v1/attention/backends/mla/rocm_aiter_mla.py"
    old = """\
        # indptr: cumsum of seq_lens (one page per token in the flat view)
        paged_kv_indptr = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=device),
                seq_lens_for_kernel.cumsum(dim=0, dtype=torch.int32),
            ]
        )
"""
    new = """\
        # K3 DSpark metadata fusion stage 6: a uniform, unpadded multi-token
        # decode can build persistent indptrs and expand its block table in one
        # kernel. Other decode shapes retain the general eager implementation.
        use_k3_fused_decode_metadata = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
            and not pad_uniform_mtp
            and max_qo_len > 1
        )
        if use_k3_fused_decode_metadata:
            paged_kv_indptr = self.paged_kv_indptr[: 1 + num_kernel_reqs]
        else:
            # indptr: cumsum of seq_lens (one page per token in the flat view)
            paged_kv_indptr = torch.cat(
                [
                    torch.zeros(1, dtype=torch.int32, device=device),
                    seq_lens_for_kernel.cumsum(dim=0, dtype=torch.int32),
                ]
            )
"""
    replace_once(path, old, new, "K3 DSpark metadata fusion stage 6")

    old = """\
        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.paged_kv_indices.fill_(-1)

        # Expand block_table entries into per-token flat indices.
        # When kernel_block_size=1, this degrades to a direct copy (identical
        # to the original _copy_page_indices_kernel).
        # When kernel_block_size=K>1, block_table entry b covering K tokens
        # gets expanded to flat indices b*K, b*K+1, ..., b*K+(K-1).
        _expand_page_indices_kernel[(num_reqs,)](
            self.paged_kv_indices,
            block_table_tensor,
            block_table_tensor.stride(0),
            paged_kv_indptr,
            KERNEL_BLOCK_SIZE=self.kernel_block_size,
            BLOCK_SIZE=1024,
            QLEN=1,
        )
        paged_kv_indices = self.paged_kv_indices

        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.paged_kv_indptr[: 1 + num_kernel_reqs].copy_(
                paged_kv_indptr, non_blocking=True
            )
            self.paged_kv_indptr[1 + num_kernel_reqs :].fill_(paged_kv_indptr[-1])
            paged_kv_indptr = self.paged_kv_indptr[: 1 + num_kernel_reqs]

            # paged_kv_last_page_len already uses the pre-initialized buffer slice
            # (set above), so no copy needed - buffer is always 1s.

            if pad_uniform_mtp:
                qo_indptr_src = torch.arange(
                    0,
                    (num_kernel_reqs + 1) * max_qo_len,
                    step=max_qo_len,
                    dtype=torch.int32,
                    device=device,
                )
            else:
                qo_indptr_src = query_start_loc_device[: 1 + num_kernel_reqs]
            self.qo_indptr[: 1 + num_kernel_reqs].copy_(
                qo_indptr_src, non_blocking=True
            )
            self.qo_indptr[1 + num_kernel_reqs :] = qo_indptr_src[-1]
            qo_indptr = self.qo_indptr[: 1 + num_kernel_reqs]

        else:
            if max_qo_len == 1:
                qo_indptr = torch.arange(
                    0,
                    num_kernel_reqs + 1,
                    step=1,
                    dtype=torch.int32,
                    device=device,
                )
            else:
                if pad_uniform_mtp:
                    qo_indptr = torch.arange(
                        0,
                        (num_kernel_reqs + 1) * max_qo_len,
                        step=max_qo_len,
                        dtype=torch.int32,
                        device=device,
                    )
                else:
                    qo_indptr = query_start_loc_device[: 1 + num_kernel_reqs]
"""
    new = """\
        if use_k3_fused_decode_metadata:
            qo_indptr = self.qo_indptr[: 1 + num_kernel_reqs]
            _k3_build_mla_decode_metadata_kernel[(self.paged_kv_indptr.numel(),)](
                self.paged_kv_indices,
                block_table_tensor,
                seq_lens_for_kernel,
                self.paged_kv_indptr,
                self.qo_indptr,
                block_table_tensor.stride(0),
                num_kernel_reqs,
                MAX_NUM_REQS=self.paged_kv_indptr.numel() - 1,
                QLEN=max_qo_len,
                KERNEL_BLOCK_SIZE=self.kernel_block_size,
                TOKEN_BLOCK_SIZE=1024,
                REQ_BLOCK_SIZE=triton.next_power_of_2(num_kernel_reqs),
            )
            paged_kv_indices = self.paged_kv_indices
        else:
            if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
                self.paged_kv_indices.fill_(-1)

            # Expand block_table entries into per-token flat indices.
            _expand_page_indices_kernel[(num_reqs,)](
                self.paged_kv_indices,
                block_table_tensor,
                block_table_tensor.stride(0),
                paged_kv_indptr,
                KERNEL_BLOCK_SIZE=self.kernel_block_size,
                BLOCK_SIZE=1024,
                QLEN=1,
            )
            paged_kv_indices = self.paged_kv_indices

            if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
                self.paged_kv_indptr[: 1 + num_kernel_reqs].copy_(
                    paged_kv_indptr, non_blocking=True
                )
                self.paged_kv_indptr[1 + num_kernel_reqs :].fill_(
                    paged_kv_indptr[-1]
                )
                paged_kv_indptr = self.paged_kv_indptr[: 1 + num_kernel_reqs]

                if pad_uniform_mtp:
                    qo_indptr_src = torch.arange(
                        0,
                        (num_kernel_reqs + 1) * max_qo_len,
                        step=max_qo_len,
                        dtype=torch.int32,
                        device=device,
                    )
                else:
                    qo_indptr_src = query_start_loc_device[: 1 + num_kernel_reqs]
                self.qo_indptr[: 1 + num_kernel_reqs].copy_(
                    qo_indptr_src, non_blocking=True
                )
                self.qo_indptr[1 + num_kernel_reqs :] = qo_indptr_src[-1]
                qo_indptr = self.qo_indptr[: 1 + num_kernel_reqs]

            else:
                if max_qo_len == 1:
                    qo_indptr = torch.arange(
                        0,
                        num_kernel_reqs + 1,
                        step=1,
                        dtype=torch.int32,
                        device=device,
                    )
                else:
                    if pad_uniform_mtp:
                        qo_indptr = torch.arange(
                            0,
                            (num_kernel_reqs + 1) * max_qo_len,
                            step=max_qo_len,
                            dtype=torch.int32,
                            device=device,
                        )
                    else:
                        qo_indptr = query_start_loc_device[: 1 + num_kernel_reqs]
"""
    replace_once(
        path,
        old,
        new,
        "_k3_build_mla_decode_metadata_kernel[(self.paged_kv_indptr.numel(),)]",
    )

    old = """\
@triton.jit
def _expand_page_indices_kernel(
"""
    new = """\
@triton.jit
def _k3_build_mla_decode_metadata_kernel(
    page_indices,
    block_table,
    seq_lens,
    paged_kv_indptr,
    qo_indptr,
    block_table_stride,
    num_reqs,
    MAX_NUM_REQS: tl.constexpr,
    QLEN: tl.constexpr,
    KERNEL_BLOCK_SIZE: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
    REQ_BLOCK_SIZE: tl.constexpr,
):
    \"\"\"Build uniform DSpark MLA indptrs and flat page indices together.\"\"\"
    req_idx = tl.program_id(0)
    req_offsets = tl.arange(0, REQ_BLOCK_SIZE)
    req_mask = req_offsets < num_reqs
    all_seq_lens = tl.load(seq_lens + req_offsets, mask=req_mask, other=0).to(
        tl.int32
    )
    total_tokens = tl.sum(all_seq_lens, axis=0)
    start_idx = tl.sum(
        tl.where(req_offsets < req_idx, all_seq_lens, 0), axis=0
    )

    # Every row through MAX_NUM_REQS is initialized. Inactive graph-padding
    # rows repeat the final offset and therefore expose an empty KV range.
    row_active = req_idx < num_reqs
    tl.store(
        paged_kv_indptr + req_idx,
        tl.where(row_active, start_idx, total_tokens),
        mask=req_idx <= MAX_NUM_REQS,
    )
    tl.store(
        qo_indptr + req_idx,
        tl.where(row_active, req_idx * QLEN, num_reqs * QLEN),
        mask=req_idx <= MAX_NUM_REQS,
    )

    num_tokens = tl.load(seq_lens + req_idx, mask=row_active, other=0).to(tl.int32)
    row_ptr = block_table + req_idx * block_table_stride
    offset = tl.arange(0, TOKEN_BLOCK_SIZE)
    for i in tl.range(0, num_tokens, TOKEN_BLOCK_SIZE):
        token_offsets = i + offset
        mask = row_active & (token_offsets < num_tokens)
        block_idx = token_offsets // KERNEL_BLOCK_SIZE
        offset_in_block = token_offsets % KERNEL_BLOCK_SIZE
        block_ids = tl.load(row_ptr + block_idx, mask=mask, other=0)
        flat_indices = block_ids * KERNEL_BLOCK_SIZE + offset_in_block
        tl.store(
            page_indices + start_idx + token_offsets,
            flat_indices,
            mask=mask,
        )


@triton.jit
def _expand_page_indices_kernel(
"""
    replace_once(path, old, new, "Build uniform DSpark MLA indptrs")


if STAGE >= 7:
    path = ROOT / "v1/attention/backend.py"
    old = """\
    _query_lens_cache: torch.Tensor | None = None
    _num_computed_tokens_cache: torch.Tensor | None = None
"""
    new = """\
    _query_lens_cache: torch.Tensor | None = None
    # K3 DSpark metadata fusion stage 7: per-step backend schedules that depend
    # only on shared sequence geometry, not on a group's block-table contents.
    _backend_metadata_cache: dict[object, object] | None = None
    _num_computed_tokens_cache: torch.Tensor | None = None
"""
    replace_once(path, old, new, "K3 DSpark metadata fusion stage 7")

    path = ROOT / "v1/worker/gpu/attn_utils.py"
    old = """\
    shared_query_lens = query_start_loc_gpu[1:] - query_start_loc_gpu[:-1]
    shared_num_computed_tokens = seq_lens - shared_query_lens

    for i in range(num_kv_cache_groups):
"""
    new = """\
    shared_query_lens = query_start_loc_gpu[1:] - query_start_loc_gpu[:-1]
    shared_num_computed_tokens = seq_lens - shared_query_lens
    shared_backend_metadata_cache: dict[object, object] = {}

    for i in range(num_kv_cache_groups):
"""
    replace_once(
        path,
        old,
        new,
        "shared_backend_metadata_cache: dict[object, object]",
    )

    old = """\
            _query_lens_cache=shared_query_lens,
            _num_computed_tokens_cache=shared_num_computed_tokens,
"""
    new = """\
            _query_lens_cache=shared_query_lens,
            _backend_metadata_cache=shared_backend_metadata_cache,
            _num_computed_tokens_cache=shared_num_computed_tokens,
"""
    replace_once(
        path,
        old,
        new,
        "_backend_metadata_cache=shared_backend_metadata_cache",
    )

    path = ROOT / "v1/attention/backends/mla/rocm_aiter_mla.py"
    old = """\
        if use_persistent_metadata:
            from aiter import get_mla_metadata_v1

            uni_qo_len = (
                max_qo_len if pad_uniform_mtp or torch.all(qo_len == max_qo_len) else -1
            )
            get_mla_metadata_v1(
                qo_indptr,
                paged_kv_indptr,
                paged_kv_last_page_len,
                self._num_attention_heads,
                1,
                True,
                self._mla_work_meta_data,
                self._mla_work_info_set,
                self._mla_work_indptr,
                self._mla_reduce_indptr,
                self._mla_reduce_final_map,
                self._mla_reduce_partial_map,
                page_size=1,
                kv_granularity=16,
                max_seqlen_qo=max_qo_len,
                uni_seqlen_qo=uni_qo_len,
                fast_mode=True,
                dtype_q=self._mla_q_dtype,
                dtype_kv=self._mla_kv_dtype,
            )
            has_persistent_metadata = True
"""
    new = """\
        if use_persistent_metadata:
            from aiter import get_mla_metadata_v1

            uni_qo_len = (
                max_qo_len if pad_uniform_mtp or torch.all(qo_len == max_qo_len) else -1
            )
            cache = getattr(self, "_k3_backend_metadata_cache", None)
            cache_key = (
                "k3_aiter_mla_persistent_schedule",
                num_kernel_reqs,
                max_qo_len,
                uni_qo_len,
                self._num_attention_heads,
                self._mla_q_dtype,
                self._mla_kv_dtype,
            )
            cached_schedule = cache.get(cache_key) if cache is not None else None
            if cached_schedule is None:
                get_mla_metadata_v1(
                    qo_indptr,
                    paged_kv_indptr,
                    paged_kv_last_page_len,
                    self._num_attention_heads,
                    1,
                    True,
                    self._mla_work_meta_data,
                    self._mla_work_info_set,
                    self._mla_work_indptr,
                    self._mla_reduce_indptr,
                    self._mla_reduce_final_map,
                    self._mla_reduce_partial_map,
                    page_size=1,
                    kv_granularity=16,
                    max_seqlen_qo=max_qo_len,
                    uni_seqlen_qo=uni_qo_len,
                    fast_mode=True,
                    dtype_q=self._mla_q_dtype,
                    dtype_kv=self._mla_kv_dtype,
                )
                if cache is not None:
                    cache[cache_key] = (
                        self._mla_work_meta_data,
                        self._mla_work_info_set,
                        self._mla_work_indptr,
                        self._mla_reduce_indptr,
                        self._mla_reduce_final_map,
                        self._mla_reduce_partial_map,
                    )
            else:
                (
                    self._mla_work_meta_data,
                    self._mla_work_info_set,
                    self._mla_work_indptr,
                    self._mla_reduce_indptr,
                    self._mla_reduce_final_map,
                    self._mla_reduce_partial_map,
                ) = cached_schedule
            has_persistent_metadata = True
"""
    replace_once(path, old, new, "k3_aiter_mla_persistent_schedule")

    old = """\
    ) -> AiterMLAMetadata:
        attn_metadata = super().build(
"""
    new = """\
    ) -> AiterMLAMetadata:
        self._k3_backend_metadata_cache = (
            common_attn_metadata._backend_metadata_cache
        )
        attn_metadata = super().build(
"""
    replace_once(
        path,
        old,
        new,
        "self._k3_backend_metadata_cache =",
    )


if STAGE >= 8:
    path = ROOT / "v1/worker/gpu/attn_utils.py"
    old = """\
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.utils.torch_utils import get_dtype_size
"""
    new = """\
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import get_dtype_size
"""
    replace_once(path, old, new, "from vllm.triton_utils import tl, triton")

    old = """\
def build_attn_metadata(
"""
    new = """\
@triton.jit
def _k3_query_geometry_kernel(
    query_start_loc,
    seq_lens,
    query_lens,
    num_computed_tokens,
    num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    # K3 DSpark metadata fusion stage 8: one load/subtract chain produces both
    # tensors shared by every target and draft KV-cache group.
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_reqs
    starts = tl.load(query_start_loc + offsets, mask=mask, other=0)
    ends = tl.load(query_start_loc + offsets + 1, mask=mask, other=0)
    lens = ends - starts
    seq = tl.load(seq_lens + offsets, mask=mask, other=0)
    tl.store(query_lens + offsets, lens, mask=mask)
    tl.store(num_computed_tokens + offsets, seq - lens, mask=mask)


def build_attn_metadata(
"""
    replace_once(path, old, new, "K3 DSpark metadata fusion stage 8")

    old = """\
    shared_query_lens = query_start_loc_gpu[1:] - query_start_loc_gpu[:-1]
    shared_num_computed_tokens = seq_lens - shared_query_lens
    shared_backend_metadata_cache: dict[object, object] = {}
"""
    new = """\
    shared_query_lens = torch.empty_like(seq_lens)
    shared_num_computed_tokens = torch.empty_like(seq_lens)
    geometry_block_size = 128
    _k3_query_geometry_kernel[
        (triton.cdiv(num_reqs, geometry_block_size),)
    ](
        query_start_loc_gpu,
        seq_lens,
        shared_query_lens,
        shared_num_computed_tokens,
        num_reqs,
        BLOCK_SIZE=geometry_block_size,
    )
    shared_backend_metadata_cache: dict[object, object] = {}
"""
    replace_once(
        path,
        old,
        new,
        "_k3_query_geometry_kernel[",
    )


if STAGE >= 9:
    path = ROOT / "v1/worker/gpu/model_states/mamba_hybrid.py"
    old = """\
@dataclass
class MambaHybridAttnMetadata(ModelSpecificAttnMetadata):
"""
    new = """\
@triton.jit
def _k3_gather_accepted_tokens_kernel(
    output,
    accepted_tokens,
    idx_mapping,
    num_active_reqs,
    num_padded_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    # K3 DSpark metadata fusion stage 9: initialize graph-padding rows to one
    # while gathering active accepted-token counts in the same kernel.
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_mask = offsets < num_padded_reqs
    active_mask = offsets < num_active_reqs
    mapped = tl.load(idx_mapping + offsets, mask=active_mask, other=0)
    accepted = tl.load(accepted_tokens + mapped, mask=active_mask, other=1)
    tl.store(output + offsets, tl.where(active_mask, accepted, 1), mask=output_mask)


@dataclass
class MambaHybridAttnMetadata(ModelSpecificAttnMetadata):
"""
    replace_once(path, old, new, "K3 DSpark metadata fusion stage 9")

    old = """\
            num_accepted_tokens = self.num_accepted_tokens_gpu.new_ones(num_reqs)
            num_accepted_tokens[: input_batch.num_reqs] = self.num_accepted_tokens_gpu[
                input_batch.idx_mapping
            ]
"""
    new = """\
            num_accepted_tokens = torch.empty(
                num_reqs,
                dtype=self.num_accepted_tokens_gpu.dtype,
                device=self.num_accepted_tokens_gpu.device,
            )
            accepted_block_size = 128
            _k3_gather_accepted_tokens_kernel[
                (triton.cdiv(num_reqs, accepted_block_size),)
            ](
                num_accepted_tokens,
                self.num_accepted_tokens_gpu,
                input_batch.idx_mapping,
                input_batch.num_reqs,
                num_reqs,
                BLOCK_SIZE=accepted_block_size,
            )
"""
    replace_once(
        path,
        old,
        new,
        "_k3_gather_accepted_tokens_kernel[",
    )


if STAGE >= 10:
    path = ROOT / "v1/worker/gpu/spec_decode/rejection_sampler.py"
    old = """\
class RejectionSampler:
"""
    new = """\
@triton.jit
def _k3_gather_rejection_inputs_kernel(
    input_ids,
    positions,
    logits_indices,
    draft_sampled,
    sampled_positions,
    num_logits,
    BLOCK_SIZE: tl.constexpr,
):
    # K3 DSpark metadata fusion stage 10: input IDs and positions use the same
    # logits index, so gather both outputs in one pass.
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_logits
    indices = tl.load(logits_indices + offsets, mask=mask, other=0)
    tl.store(
        draft_sampled + offsets,
        tl.load(input_ids + indices, mask=mask, other=0),
        mask=mask,
    )
    tl.store(
        sampled_positions + offsets,
        tl.load(positions + indices, mask=mask, other=0),
        mask=mask,
    )


class RejectionSampler:
"""
    replace_once(path, old, new, "K3 DSpark metadata fusion stage 10")

    old = """\
        draft_sampled = input_batch.input_ids[input_batch.logits_indices]
        pos = input_batch.positions[input_batch.logits_indices]
"""
    new = """\
        num_logits = input_batch.logits_indices.numel()
        draft_sampled = torch.empty(
            num_logits,
            dtype=input_batch.input_ids.dtype,
            device=input_batch.input_ids.device,
        )
        pos = torch.empty(
            num_logits,
            dtype=input_batch.positions.dtype,
            device=input_batch.positions.device,
        )
        rejection_input_block_size = 128
        _k3_gather_rejection_inputs_kernel[
            (triton.cdiv(num_logits, rejection_input_block_size),)
        ](
            input_batch.input_ids,
            input_batch.positions,
            input_batch.logits_indices,
            draft_sampled,
            pos,
            num_logits,
            BLOCK_SIZE=rejection_input_block_size,
        )
"""
    replace_once(
        path,
        old,
        new,
        "_k3_gather_rejection_inputs_kernel[",
    )


if STAGE >= 11:
    path = ROOT / "models/kimi_k3/nvidia/dspark_mla.py"
    old = """\
from vllm.models.kimi_k3.nvidia.model import KimiMLP
from vllm.utils.torch_utils import is_quantized_kv_cache
"""
    new = """\
from vllm.models.kimi_k3.nvidia.model import KimiMLP
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import is_quantized_kv_cache
"""
    replace_once(path, old, new, "from vllm.triton_utils import tl, triton")

    old = """\
class K3DSparkDecoderLayer(nn.Module):
"""
    new = """\
@triton.jit
def _k3_transpose_context_kpe_and_positions_kernel(
    source,
    output,
    positions,
    repeated_positions,
    num_context_tokens,
    source_token_stride,
    source_layer_stride,
    output_layer_stride,
    rope_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # K3 DSpark metadata fusion stage 11: transpose one RoPE-key layer and
    # replicate its positions in the same pass.
    layer = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    token = offsets // rope_dim
    dim = offsets % rope_dim
    mask = token < num_context_tokens
    values = tl.load(
        source
        + token * source_token_stride
        + layer * source_layer_stride
        + dim,
        mask=mask,
        other=0,
    )
    tl.store(
        output + layer * output_layer_stride + offsets,
        values,
        mask=mask,
    )
    position_mask = mask & (dim == 0)
    tl.store(
        repeated_positions + layer * num_context_tokens + token,
        tl.load(positions + token, mask=position_mask, other=0),
        mask=position_mask,
    )


class K3DSparkDecoderLayer(nn.Module):
"""
    replace_once(path, old, new, "K3 DSpark metadata fusion stage 11")

    old = """\
        all_k_pe = all_k_pe.permute(1, 0, 2).contiguous()
        all_k_pe_flat = all_k_pe.view(num_layers * num_ctx, 1, self._context_rope_dim)
        repeated_positions = self._context_positions_repeated[: num_layers * num_ctx]
        repeated_positions.view(num_layers, num_ctx).copy_(context_positions)
"""
    new = """\
        transposed_k_pe = torch.empty(
            (num_layers, num_ctx, self._context_rope_dim),
            dtype=all_k_pe.dtype,
            device=all_k_pe.device,
        )
        repeated_positions = self._context_positions_repeated[: num_layers * num_ctx]
        transpose_block_size = 256
        _k3_transpose_context_kpe_and_positions_kernel[
            (
                num_layers,
                triton.cdiv(
                    num_ctx * self._context_rope_dim,
                    transpose_block_size,
                ),
            )
        ](
            all_k_pe,
            transposed_k_pe,
            context_positions,
            repeated_positions,
            num_ctx,
            all_k_pe.stride(0),
            all_k_pe.stride(1),
            transposed_k_pe.stride(0),
            rope_dim=self._context_rope_dim,
            BLOCK_SIZE=transpose_block_size,
        )
        all_k_pe_flat = transposed_k_pe.view(
            num_layers * num_ctx, 1, self._context_rope_dim
        )
"""
    replace_once(
        path,
        old,
        new,
        "_k3_transpose_context_kpe_and_positions_kernel[",
    )


if STAGE >= 12:
    path = ROOT / "v1/worker/gpu/spec_decode/rejection_sampler.py"
    old = """\
            chunk_cu_num_logits_np = cu_num_logits_np[start : end + 1] - lo
            chunk_cu_num_logits = input_batch.cu_num_logits[start : end + 1] - lo
"""
    new = """\
            chunk_cu_num_logits_np = cu_num_logits_np[start : end + 1] - lo
            # K3 DSpark metadata fusion stage 12: the steady-state batch fits
            # in the first rejection chunk, whose offset is zero. Preserve the
            # general normalization path only for later chunks.
            chunk_cu_num_logits = input_batch.cu_num_logits[start : end + 1]
            if lo:
                chunk_cu_num_logits = chunk_cu_num_logits - lo
"""
    replace_once(path, old, new, "K3 DSpark metadata fusion stage 12")


# Stage 13 is an intentionally isolated experiment: it reduced one launch but
# regressed target-to-draft latency materially, so later retained stages skip it.
if STAGE == 13:
    path = ROOT / "models/kimi_k3/nvidia/dspark_mla.py"
    old = """\
@triton.jit
def _k3_transpose_context_kpe_and_positions_kernel(
"""
    new = """\
@triton.jit
def _k3_transpose_context_kvc_rms_kernel(
    source,
    output,
    weights,
    num_context_tokens,
    source_token_stride,
    source_layer_stride,
    output_layer_stride,
    output_token_stride,
    weight_layer_stride,
    epsilon,
    kv_lora_rank: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # K3 DSpark metadata fusion stage 13: RMS-normalize one latent-KV row
    # while writing it directly into the layer-major layout.
    layer = tl.program_id(0)
    token = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < kv_lora_rank
    values = tl.load(
        source
        + token * source_token_stride
        + layer * source_layer_stride
        + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    variance = tl.sum(values * values, axis=0) / kv_lora_rank
    normalized = values * tl.rsqrt(variance + epsilon)
    layer_weights = tl.load(
        weights + layer * weight_layer_stride + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output
        + layer * output_layer_stride
        + token * output_token_stride
        + offsets,
        normalized * layer_weights,
        mask=mask & (token < num_context_tokens),
    )


@triton.jit
def _k3_transpose_context_kpe_and_positions_kernel(
"""
    replace_once(path, old, new, "K3 DSpark metadata fusion stage 13")

    old = """\
        all_kv_c = all_kv_c.permute(1, 0, 2).contiguous()
        all_kv_c_normed = torch.empty_like(all_kv_c)
        ops.rms_norm(
            all_kv_c_normed,
            all_kv_c,
            self._context_kv_norm_weights,
            self._context_rms_norm_eps,
        )
"""
    new = """\
        all_kv_c_normed = torch.empty(
            (num_layers, num_ctx, self._context_kv_lora_rank),
            dtype=all_kv_c.dtype,
            device=all_kv_c.device,
        )
        context_rms_block_size = triton.next_power_of_2(
            self._context_kv_lora_rank
        )
        _k3_transpose_context_kvc_rms_kernel[(num_layers, num_ctx)](
            all_kv_c,
            all_kv_c_normed,
            self._context_kv_norm_weights,
            num_ctx,
            all_kv_c.stride(0),
            all_kv_c.stride(1),
            all_kv_c_normed.stride(0),
            all_kv_c_normed.stride(1),
            self._context_kv_norm_weights.stride(0),
            self._context_rms_norm_eps,
            kv_lora_rank=self._context_kv_lora_rank,
            BLOCK_SIZE=context_rms_block_size,
        )
"""
    replace_once(
        path,
        old,
        new,
        "_k3_transpose_context_kvc_rms_kernel[(num_layers, num_ctx)]",
    )


if STAGE >= 14:
    path = ROOT / "v1/worker/gpu/model_runner.py"
    old = """\
        if self.speculator is not None:
            assert self.sampler is not None
            # Let the target override the hidden state fed to the drafter
"""
    new = """\
        # K3 DSpark metadata fusion stage 14: retain the active proposal view so
        # the output handler does not gather the same rows back immediately
        # after scattering them into persistent request state.
        draft_tokens_for_output: torch.Tensor | None = None
        if self.speculator is not None:
            assert self.sampler is not None
            # Let the target override the hidden state fed to the drafter
"""
    replace_once(path, old, new, "K3 DSpark metadata fusion stage 14")

    old = """\
            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens

        if self.num_speculative_steps > 0:
            # Spec-decode and diffusion LLMs both use draft tokens but the latter does
            # not have a speculator (i.e. self.speculator is None)
            self.draft_tokens_handler.set_draft_tokens(
                input_batch,
                self.req_states.draft_tokens[input_batch.idx_mapping],
            )
"""
    new = """\
            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens
            draft_tokens_for_output = draft_tokens

        if self.num_speculative_steps > 0:
            # Diffusion LLMs do not have a speculator, so retain the general
            # request-state gather only for that path.
            if draft_tokens_for_output is None:
                draft_tokens_for_output = self.req_states.draft_tokens[
                    input_batch.idx_mapping
                ]
            self.draft_tokens_handler.set_draft_tokens(
                input_batch,
                draft_tokens_for_output,
            )
"""
    replace_once(
        path,
        old,
        new,
        "draft_tokens_for_output = draft_tokens",
    )


if STAGE >= 15:
    path = ROOT / "v1/worker/gpu/input_batch.py"
    old = """\
    cu_num_logits_ptr,
    logits_indices_ptr,
    BLOCK_SIZE: tl.constexpr,
    NUM_NEW_SAMPLED_TOKENS: tl.constexpr = 1,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
"""
    new = """\
    cu_num_logits_ptr,
    logits_indices_ptr,
    is_padding_ptr,
    num_reqs,
    num_tokens,
    num_tokens_after_padding,
    BLOCK_SIZE: tl.constexpr,
    PADDING_BLOCK_SIZE: tl.constexpr,
    WRITE_PADDING: tl.constexpr,
    NUM_NEW_SAMPLED_TOKENS: tl.constexpr = 1,
):
    batch_idx = tl.program_id(0)
    if WRITE_PADDING:
        # K3 DSpark metadata fusion stage 15: initialize the MoE padding mask
        # in the already-required token-combination launch.
        padding_offsets = batch_idx * PADDING_BLOCK_SIZE + tl.arange(
            0, PADDING_BLOCK_SIZE
        )
        padding_mask = padding_offsets < num_tokens_after_padding
        tl.store(
            is_padding_ptr + padding_offsets,
            padding_offsets >= num_tokens,
            mask=padding_mask,
        )

    if batch_idx >= num_reqs:
        return
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
"""
    replace_once(path, old, new, "K3 DSpark metadata fusion stage 15")

    old = """\
    num_logits: int,
    num_new_sampled_tokens: int = 1,  # excl accepted draft tokens, a.k.a bonus tokens
) -> torch.Tensor:
"""
    new = """\
    num_logits: int,
    num_new_sampled_tokens: int = 1,  # excl accepted draft tokens, a.k.a bonus tokens
    is_padding: torch.Tensor | None = None,
    num_tokens: int | None = None,
    num_tokens_after_padding: int | None = None,
) -> torch.Tensor:
"""
    replace_once(path, old, new, "is_padding: torch.Tensor | None = None")

    old = """\
    _combine_sampled_and_draft_tokens_kernel[(num_reqs,)](
        input_ids,
        idx_mapping,
        last_sampled_tokens,
        query_start_loc,
        seq_lens,
        prefill_len,
        draft_tokens,
        draft_tokens.stride(0),
        cu_num_logits,
        logits_indices,
        NUM_NEW_SAMPLED_TOKENS=num_new_sampled_tokens,
        # NOTE(woosuk): Add num_new_sampled_tokens to ensure the block covers the
        # last sampled token in addition to all draft tokens.
        BLOCK_SIZE=triton.next_power_of_2(
            num_speculative_steps + num_new_sampled_tokens
        ),
    )
"""
    new = """\
    write_padding = is_padding is not None
    if write_padding:
        assert num_tokens is not None
        assert num_tokens_after_padding is not None
        padding_block_size = 128
        num_programs = max(
            num_reqs,
            triton.cdiv(num_tokens_after_padding, padding_block_size),
        )
        is_padding_ptr = is_padding
    else:
        num_tokens = 0
        num_tokens_after_padding = 0
        padding_block_size = 1
        num_programs = num_reqs
        is_padding_ptr = input_ids
    _combine_sampled_and_draft_tokens_kernel[(num_programs,)](
        input_ids,
        idx_mapping,
        last_sampled_tokens,
        query_start_loc,
        seq_lens,
        prefill_len,
        draft_tokens,
        draft_tokens.stride(0),
        cu_num_logits,
        logits_indices,
        is_padding_ptr,
        num_reqs,
        num_tokens,
        num_tokens_after_padding,
        NUM_NEW_SAMPLED_TOKENS=num_new_sampled_tokens,
        # NOTE(woosuk): Add num_new_sampled_tokens to ensure the block covers the
        # last sampled token in addition to all draft tokens.
        BLOCK_SIZE=triton.next_power_of_2(
            num_speculative_steps + num_new_sampled_tokens
        ),
        PADDING_BLOCK_SIZE=padding_block_size,
        WRITE_PADDING=write_padding,
    )
"""
    replace_once(
        path,
        old,
        new,
        "_combine_sampled_and_draft_tokens_kernel[(num_programs,)]",
    )

    path = ROOT / "v1/worker/gpu/model_runner.py"
    old = """\
        if envs.VLLM_MOE_SKIP_PADDING:
            # Mark trailing cudagraph-padding rows so kernels can skip work for
            # them when supported.
            self.input_buffers.is_padding[:num_tokens].fill_(False)
            self.input_buffers.is_padding[num_tokens:num_tokens_after_padding].fill_(
                True
            )
"""
    new = """\
        # K3 DSpark stage 15 initializes the MoE padding mask in
        # combine_sampled_and_draft_tokens below.
"""
    replace_once(
        path,
        old,
        new,
        "K3 DSpark stage 15 initializes the MoE padding mask",
    )

    old = """\
            total_num_logits,
            self.model_state.num_new_sampled_tokens_per_step,
        )
"""
    new = """\
            total_num_logits,
            self.model_state.num_new_sampled_tokens_per_step,
            is_padding=(
                self.input_buffers.is_padding
                if envs.VLLM_MOE_SKIP_PADDING
                else None
            ),
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens_after_padding,
        )
"""
    replace_once(
        path,
        old,
        new,
        "num_tokens_after_padding=num_tokens_after_padding",
    )


if STAGE >= 16:
    path = ROOT / "v1/worker/gpu/model_runner.py"
    old = """\
from vllm.sequence import IntermediateTensors
"""
    new = """\
from vllm.sequence import IntermediateTensors
from vllm.triton_utils import tl, triton
"""
    replace_once(path, old, new, "from vllm.triton_utils import tl, triton")

    old = """\
class GPUModelRunner(LoRAModelRunnerMixin):
"""
    new = """\
@triton.jit
def _k3_scatter_draft_tokens_kernel(
    persistent_draft_tokens,
    active_draft_tokens,
    idx_mapping,
    persistent_stride,
    active_stride,
    num_reqs,
    num_draft_tokens,
    BLOCK_SIZE: tl.constexpr,
):
    # K3 DSpark metadata fusion stage 16: scatter one active proposal row to
    # persistent request state without ATen's materialization helper.
    req_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (req_idx < num_reqs) & (offsets < num_draft_tokens)
    state_idx = tl.load(idx_mapping + req_idx, mask=req_idx < num_reqs, other=0)
    values = tl.load(
        active_draft_tokens + req_idx * active_stride + offsets,
        mask=mask,
        other=0,
    )
    tl.store(
        persistent_draft_tokens + state_idx * persistent_stride + offsets,
        values,
        mask=mask,
    )


class GPUModelRunner(LoRAModelRunnerMixin):
"""
    replace_once(path, old, new, "K3 DSpark metadata fusion stage 16")

    old = """\
            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens
            draft_tokens_for_output = draft_tokens
"""
    new = """\
            num_active_reqs = input_batch.idx_mapping.numel()
            num_draft_tokens = draft_tokens.shape[1]
            _k3_scatter_draft_tokens_kernel[(num_active_reqs,)](
                self.req_states.draft_tokens,
                draft_tokens,
                input_batch.idx_mapping,
                self.req_states.draft_tokens.stride(0),
                draft_tokens.stride(0),
                num_active_reqs,
                num_draft_tokens,
                BLOCK_SIZE=triton.next_power_of_2(num_draft_tokens),
            )
            draft_tokens_for_output = draft_tokens
"""
    replace_once(
        path,
        old,
        new,
        "_k3_scatter_draft_tokens_kernel[(num_active_reqs,)]",
    )


for py in ROOT.rglob("*.py"):
    ensure_triton_import(py)

print(f"K3 DSpark metadata fusion stage {STAGE}: PATCH_OK")
