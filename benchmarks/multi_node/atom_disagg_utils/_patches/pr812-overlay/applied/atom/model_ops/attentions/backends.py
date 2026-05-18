# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Generic, Optional, Type, TypeVar

if TYPE_CHECKING:
    from atom.kv_transfer.disaggregation.types import KVTransferTensors

import torch
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_ops.attention_mla import MLAModules
from atom.utils import CpuGpuBuffer
from atom.utils.block_convert import block_table_convert_triton
from atom.utils.tbo.ubatch_splitting import UBatchSlice, split_attn_metadata
from atom.utils.forward_context import AttentionMetaData
from torch import nn

logger = logging.getLogger("atom")
T = TypeVar("T", bound="BroadcastableModelInput")


class BroadcastableModelInput(ABC):

    @abstractmethod
    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        """
        Extract broadcastable fields. Override for fields that require some
        custom deserialization.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_broadcasted_tensor_dict(
        cls: Type[T],
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> T:
        """
        Pop fields from the given tensor_dict and populate a new instance of
        BroadcastableModelInput.
        """
        raise NotImplementedError


class AttentionBackend(ABC):
    """Abstract class for attention backends."""

    # For some attention backends, we allocate an output tensor before
    # calling the custom op. When piecewise cudagraph is enabled, this
    # makes sure the output tensor is allocated inside the cudagraph.
    accept_output_buffer: bool = False

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_builder_cls() -> Type["AttentionMetadataBuilder"]:
        raise NotImplementedError

    @staticmethod
    def get_impl_cls() -> Type["AttentionImpl"]:
        return AttentionImpl


class AttentionMetadataBuilder(ABC, Generic[T]):
    """Abstract class for attention metadata builders."""

    @abstractmethod
    def __init__(self, block_size: int) -> None:
        """Create the builder, remember some configuration and parameters."""
        raise NotImplementedError

    @abstractmethod
    def prepare_decode(self, batch: ScheduledBatch, bs: int):
        raise NotImplementedError

    @abstractmethod
    def prepare_prefill(self, batch: ScheduledBatch):
        raise NotImplementedError

    @abstractmethod
    def build(self, batch: ScheduledBatch, bs: int):
        raise NotImplementedError

    @abstractmethod
    def build_for_cudagraph_capture(self, bs: int) -> AttentionMetaData:
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Per-request cache (model-managed state outside the paged KV pool). #
    # ------------------------------------------------------------------ #
    # Used by attention types that maintain per-request stateful buffers
    # which do not fit the paged KV cache model — e.g. GDN recurrent state,
    # DeepseekV4 ring buffer + compressor state. ModelRunner queries these
    # methods at startup to size the per-request slot pool, deduct its
    # bytes from the KV pool budget, and allocate the underlying tensors.
    #
    # Stateless attentions (standard MHA / MLA) leave the defaults:
    # `compute_per_req_cache_bytes()` returns 0, `allocate_per_req_cache()`
    # returns an empty dict, so no per-req pool is allocated.

    def compute_per_req_cache_bytes(self) -> int:
        """Total bytes (across all attention layers) for ONE request's
        per-request cache.

        ModelRunner multiplies this by `max_num_seqs * slots_per_req()` to
        size the per-req cache tensors and deduct that memory from the KV
        pool budget.
        """
        return 0

    def slots_per_req(self) -> int:
        """Number of contiguous slot indices one request occupies.

        Default = 1 (single committed state, no speculative lookahead).
        GDN-style attentions override with `1 + model_runner.num_spec_tokens`
        because their state-update kernel reserves one extra slot per
        speculated token for rollback on rejection. Override only if the
        attention has a different lookahead layout.
        """
        return 1

    def allocate_per_req_cache(self, num_slots: int) -> dict[str, "torch.Tensor"]:
        """Allocate per-request cache tensors.

        Called by ModelRunner.allocate_kv_cache() once `num_slots` is known.
        Builder returns a dict mapping attribute name → tensor; ModelRunner
        does `setattr(self, name, tensor)` so model layers can access them
        as `model_runner.<name>` (preserving existing names like
        `mamba_k_cache` / `mamba_v_cache`).
        """
        return {}

    def get_kv_transfer_tensors(self) -> "KVTransferTensors | None":
        """Return RDMA transfer regions for PD disaggregation.

        Each attention backend overrides this to describe its block-indexed
        and slot-indexed tensor regions.  The KV connector uses the result
        to register RDMA memory and compute transfer offsets without knowing
        the backend's internal layout.

        Returns ``None`` when KV transfer is not configured or tensors have
        not been allocated yet.
        """
        return None

    def compute_block_bytes(self) -> int:
        """Per-block bytes contributed by this attention type's primary KV
        tensors (kv_cache + kv_scale + any side caches like the V3.2
        indexer cache).

        Mirror of `allocate_kv_cache_tensors`: used by ModelRunner
        get_num_blocks() to size the unified pool BEFORE any tensor is
        allocated, so the budget math sees the same per-block cost the
        builder will actually allocate. Per-request cache bytes are NOT
        included here — they're accounted for via
        `compute_per_req_cache_bytes()`.

        Default returns 0 (no primary KV pool).
        """
        return 0

    def allocate_kv_cache_tensors(
        self, num_kv_heads: int, num_draft_layers: int
    ) -> dict[str, Any]:
        """Allocate the model's primary paged KV cache tensors.

        Called by ModelRunner.allocate_kv_cache() after num_physical_kvcache_blocks
        is known. Builders own the per-attention-type tensor layout (single
        576-dim MLA tensor vs split-K/V MHA tensor; full-rank vs hybrid-only-
        full-attn-rows for Qwen3-Next; per-module deferred for MiMo-V2). The
        runner only setattr's the returned dict onto itself, so model layers
        can access tensors as `model_runner.<name>` (preserving existing
        names: kv_cache, kv_scale, index_cache, etc.).

        Values may be Tensors, None (deferred allocation), or scalar metadata
        (e.g. aligned_index_dim) needed downstream by build_kv_cache_tensor.
        Returns empty dict for builders that do not own the main KV pool.
        """
        return {}

    def build_kv_cache_tensor(self, layer_id: int, module):
        """Build the vLLM-style `KVCacheTensor` registration entry for one
        attention module, OR return None if this builder does not recognize
        the module type.

        Called from ModelRunner.allocate_kv_cache()'s binding loop for every
        module of the model. The builder owns:
          - module-type detection (e.g. `hasattr(module, "use_mla")`)
          - per-attention-type slot index math (attn_idx, gdn_idx, ...)
          - per-module tensor slicing from runner-owned tensors
            (self.model_runner.kv_cache, .mamba_k_cache, ...)
          - any `setattr(module, "k_cache", ...)` side effects per the
            existing module convention
          - returning a `KVCacheTensor` ModelRunner appends to its registry

        Builders override this for the module types they handle; subclasses
        chain via `super().build_kv_cache_tensor(...)` to inherit shared
        paths (e.g. `GDNAttentionMetadataBuilder` handles
        `base_linear_attention` and delegates `base_attention` MHA modules
        to its `AiterAttentionMetadataBuilder` parent).

        Default returns None for unknown module types.
        """
        return None


class CommonAttentionBuilder(AttentionMetadataBuilder[T], Generic[T]):
    def __init__(self, model_runner):
        self.model_runner = model_runner
        assert model_runner.block_size % self.block_size == 0
        self.block_ratio = model_runner.block_size // self.block_size
        self.device = model_runner.device
        config = model_runner.config
        hf_config = config.hf_config
        self.max_num_batched_tokens = model_runner.max_num_batched_tokens
        self.max_bs = model_runner.max_bs
        self.max_num_blocks_per_seq = (
            config.max_model_len + self.block_size - 1
        ) // self.block_size

        i64_kwargs = {"dtype": torch.int64, "device": self.device}
        i32_kwargs = {"dtype": torch.int32, "device": self.device}

        attn_metadata = {
            "slot_mapping": CpuGpuBuffer(self.max_num_batched_tokens, **i64_kwargs),
            "context_lens": CpuGpuBuffer(self.max_bs, **i32_kwargs),
            "block_tables": CpuGpuBuffer(
                self.max_bs,
                self.max_num_blocks_per_seq // self.block_ratio,
                **i32_kwargs,
            ),
            "cu_seqlens_q": CpuGpuBuffer(self.max_bs + 1, **i32_kwargs),
            "cu_seqlens_k": CpuGpuBuffer(self.max_bs + 1, **i32_kwargs),
            # seq_starts for cp_mha_gather_cache: always zeros (prefix at position 0)
            "seq_starts": CpuGpuBuffer(self.max_bs, **i32_kwargs),
        }
        if self.block_ratio > 1:
            attn_metadata["block_tables_converted"] = CpuGpuBuffer(
                self.max_bs,
                self.max_num_blocks_per_seq,
                **i32_kwargs,
            )

        attn_metadata["cu_seqlens_q"].cpu.copy_(
            torch.arange(0, self.max_bs + 1, step=1, dtype=torch.int32)
        )
        attn_metadata["cu_seqlens_q"].copy_to_gpu()
        attn_metadata["seq_starts"].cpu.zero_()
        attn_metadata["seq_starts"].copy_to_gpu()
        self.model_runner.forward_vars.update(attn_metadata)
        self.has_sliding_window = hasattr(hf_config, "sliding_window")

    def prepare_block_tables(self, batch: ScheduledBatch):
        var = self.model_runner.forward_vars
        block_tables = var["block_tables"].np
        for i, block_table in enumerate(batch.block_tables):
            block_tables[i] = 0
            block_tables[i, : len(block_table)] = block_table

    def prepare_prefill(self, batch: ScheduledBatch):
        bs = batch.total_seqs_num_prefill
        sum_scheduled_tokens = batch.total_tokens_num_prefill
        var = self.model_runner.forward_vars
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        has_cached = False
        # seqs = list(batch.seqs.values())
        # seqs = seqs[:bs]
        for i in range(bs):
            seqlen = batch.context_lens[i]
            cached_seqlen = batch.num_cached_tokens[i]
            if cached_seqlen > 0:
                has_cached = True
            positions.extend(list(range(cached_seqlen, seqlen)))
            seqlen_q = seqlen - cached_seqlen
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not batch.block_tables:
                continue
            num_blocks = (
                seqlen + self.model_runner.block_size - 1
            ) // self.model_runner.block_size
            num_cached_blocks = (
                cached_seqlen + self.model_runner.block_size - 1
            ) // self.model_runner.block_size
            last_block_tokens = batch.last_block_num_tokens[i]
            block_table = batch.block_tables[i]
            for blk_idx in range(num_cached_blocks, num_blocks):
                start = block_table[blk_idx] * self.model_runner.block_size
                if blk_idx != num_blocks - 1:
                    end = start + self.model_runner.block_size
                else:
                    end = start + last_block_tokens
                slot_mapping.extend(list(range(start, end)))
        if has_cached:
            self.prepare_block_tables(batch)
        # Validate metadata consistency
        assert (
            len(positions) == sum_scheduled_tokens
        ), f"positions length {len(positions)} != sum_scheduled_tokens {sum_scheduled_tokens}"
        if batch.block_tables:
            assert (
                len(slot_mapping) == sum_scheduled_tokens
            ), f"slot_mapping length {len(slot_mapping)} != sum_scheduled_tokens {sum_scheduled_tokens}"
        assert (
            cu_seqlens_q[-1] == sum_scheduled_tokens
        ), f"cu_seqlens_q[-1]={cu_seqlens_q[-1]} != sum_scheduled_tokens={sum_scheduled_tokens}"
        var["positions"].np[:sum_scheduled_tokens] = positions
        var["slot_mapping"].np[:sum_scheduled_tokens] = -1
        var["slot_mapping"].np[: len(slot_mapping)] = slot_mapping
        var["cu_seqlens_q"].np[: bs + 1] = cu_seqlens_q
        var["cu_seqlens_k"].np[: bs + 1] = cu_seqlens_k
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True)
        var["context_lens"].np[:bs] = batch.context_lens[:bs]
        min_seqlen_q = 0
        dropout_p = 0.0
        vars_used = [
            ("cu_seqlens_q", bs + 1),
            ("slot_mapping", sum_scheduled_tokens),
            ("context_lens", bs),
        ]
        if has_cached:
            vars_used.append(("block_tables", bs))
            vars_used.append(("seq_starts", bs))

        ctx = {el: var[el].copy_to_gpu(num) for el, num in vars_used}
        if self.block_ratio > 1 and "block_tables" in ctx:
            block_table_convert_triton(
                var["block_tables"].gpu[:bs],
                var["block_tables_converted"].gpu[:bs],
                var["context_lens"].gpu[:bs],
                self.block_ratio,
            )
            ctx["block_tables_converted"] = var["block_tables_converted"].gpu[:bs]
        num_cached_tokens = None
        if has_cached:
            num_cached_tokens = torch.tensor(
                batch.num_cached_tokens[:bs], dtype=torch.int32, pin_memory=True
            ).cuda(non_blocking=True)
            total_tokens = sum(batch.context_lens[:bs])
        total_kv = total_tokens if has_cached else sum_scheduled_tokens
        attn_metadata = AttentionMetaData(
            cu_seqlens_k=cu_seqlens_k.cuda(non_blocking=True),
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            min_seqlen_q=min_seqlen_q,
            dropout_p=dropout_p,
            has_cached=has_cached,
            total_kv=total_kv,
            num_cached_tokens=num_cached_tokens,
            **ctx,
        )
        positions = var["positions"].copy_to_gpu(sum_scheduled_tokens)

        return attn_metadata, positions
        # return var["positions"].copy_to_gpu(sum_scheduled_tokens)

    def build_ubatch_prefill_metadata(
        self,
        attn_metadata: AttentionMetaData,
        ub_slice: UBatchSlice,
        padded_bs: int,
    ) -> AttentionMetaData:
        return split_attn_metadata(attn_metadata, ub_slice, padded_bs)

    def build(self, batch: ScheduledBatch, bs: int):
        is_prefill = batch.total_tokens_num_prefill > 0
        if is_prefill:
            return self.prepare_prefill(batch)
        else:
            return self.prepare_decode(batch, bs)


class AttentionImpl(nn.Module):
    @abstractmethod
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        kv_cache_dtype: str = "auto",
        layer_num: int = 0,
        mla_modules: MLAModules = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        position: torch.Tensor = None,
    ) -> torch.Tensor:
        raise NotImplementedError
