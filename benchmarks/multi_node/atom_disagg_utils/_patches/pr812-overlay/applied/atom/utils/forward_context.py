# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict, Optional, Set, Union

import numpy as np
import torch
from atom.config import Config, KVCacheTensor, ParallelConfig

if TYPE_CHECKING:
    from atom.plugin.attention import MetadataForPluginMode


def _compute_chunked_local_num_tokens(
    num_tokens_across_dp_cpu: list[int], max_num_tokens: int, chunk_idx: int
) -> list[int]:
    dp_size = len(num_tokens_across_dp_cpu)

    local_size = [-1] * dp_size
    for i in range(dp_size):
        dp_tokens = num_tokens_across_dp_cpu[i]
        local_size[i] = min(max_num_tokens, dp_tokens - (max_num_tokens * chunk_idx))
        if local_size[i] <= 0:
            local_size[i] = 1  # ensure lockstep even if done
    return local_size


@dataclass
class DPMetadata:
    max_tokens_across_dp_cpu: torch.Tensor
    cu_tokens_across_dp_cpu: torch.Tensor
    max_tokens_across_dp: int  # Pre-computed int value for cudagraph compatibility
    local_sizes: Optional[list[int]] = None

    @staticmethod
    def num_tokens_across_dp(
        num_tokens: int, dp_size: int, dp_rank: int
    ) -> torch.Tensor:
        """
        Gather the num_tokens across all DP ranks and return results in a
        CPU tensor of size dp_size.
        """
        num_tokens_across_dp = [0] * dp_size
        num_tokens_across_dp[dp_rank] = num_tokens
        num_tokens_tensor = torch.tensor(
            num_tokens_across_dp, device="cpu", dtype=torch.int32
        )
        import torch.distributed as dist
        from aiter.dist.parallel_state import get_dp_group

        dist.all_reduce(num_tokens_tensor, group=get_dp_group().cpu_group)
        return num_tokens_tensor

    @staticmethod
    def make(
        parallel_config: ParallelConfig,
        # attn_metadata: Any,
        num_tokens: int,
        num_tokens_across_dp: Optional[torch.Tensor] = None,
    ) -> "DPMetadata":

        assert parallel_config.data_parallel_size > 1
        dp_size = parallel_config.data_parallel_size
        dp_rank = parallel_config.data_parallel_rank
        batchsize = num_tokens

        # If num_tokens_across_dp is None, it will be computed by all_reduce
        # Otherwise, num_tokens_across_dp[dp_rank] should be equal to batchsize
        assert (
            num_tokens_across_dp is None or num_tokens_across_dp[dp_rank] == batchsize
        )
        if num_tokens_across_dp is None:
            num_tokens_across_dp = DPMetadata.num_tokens_across_dp(
                batchsize, dp_size, dp_rank
            )
        max_tokens_across_dp_cpu = torch.max(num_tokens_across_dp)
        cu_tokens_across_dp_cpu = torch.cumsum(num_tokens_across_dp, dim=0)
        max_tokens_across_dp = (
            max_tokens_across_dp_cpu.item()
        )  # Pre-compute int for cudagraph
        return DPMetadata(
            max_tokens_across_dp_cpu, cu_tokens_across_dp_cpu, max_tokens_across_dp
        )

    @contextmanager
    def chunked_sizes(self, max_chunk_size_per_rank: int, chunk_idx: int):
        """
        Context manager to compute and temporarily set the per-rank local token
        sizes for a specific chunk during chunked forward execution.
        This is necessary to ensure each DP (data parallel) rank processes its
        designated portion of tokens in lockstep with others, even when the
        token counts are uneven or some ranks have completed their input early.
        For chunked execution, we break up the total tokens on each rank into
        multiple chunks (of at most `max_chunk_size_per_rank`), and for a given
        `chunk_idx`, this context manager sets `self.local_sizes` to the number
        of tokens to process in that chunk on each rank.
        It uses cumulative sizes (`cu_tokens_across_dp_cpu`) to derive the
        number of tokens per rank, and calls `_compute_chunked_local_num_tokens`
        to determine the chunk-wise split.
        `self.local_sizes` is only valid inside the context.
        Args:
            max_chunk_size_per_rank: The max number of tokens each rank is
                                     allowed to process in this chunk.
            chunk_idx: The index of the chunk to compute sizes for.
        """
        cu_sizes = self.cu_tokens_across_dp_cpu
        num_tokens_across_dp_cpu = [
            (cu_sizes[i] - cu_sizes[i - 1]).item() if i > 0 else cu_sizes[0].item()
            for i in range(len(cu_sizes))
        ]
        self.local_sizes = _compute_chunked_local_num_tokens(
            num_tokens_across_dp_cpu, max_chunk_size_per_rank, chunk_idx
        )
        try:
            yield self.local_sizes
        finally:
            self.local_sizes = None

    def get_chunk_sizes_across_dp_rank(self) -> Optional[list[int]]:
        return self.local_sizes


@dataclass
class SpecDecodeMetadata:
    draft_token_ids: torch.Tensor
    num_spec_steps: int
    num_draft_tokens_np: np.ndarray
    cu_num_draft_tokens: torch.Tensor
    target_logits_indices: torch.Tensor
    bonus_logits_indices: torch.Tensor


@dataclass
class Context:
    # This context is used to store the basic context of the forward.
    positions: torch.Tensor
    is_prefill: bool = False
    is_dummy_run: bool = False
    batch_size: int = 0
    graph_bs: int = 0
    is_draft: bool = False
    # Optional flat token ids for the current forward. Read by callbacks
    # invoked inside Dynamo-opaque custom ops (e.g. V4 MoE hash routing)
    # that need the token ids but cannot receive them as a function arg
    # (the op signature is fixed by the consumer's plugin contract).
    input_ids: Optional[torch.Tensor] = None

    def __init__(
        self,
        positions: torch.Tensor,
        is_prefill: bool = False,
        is_dummy_run: bool = False,
        batch_size: int = 0,
        graph_bs: int = 0,
        is_draft: bool = False,
        input_ids: Optional[torch.Tensor] = None,
    ):
        self.positions = positions
        self.is_prefill = is_prefill
        self.is_dummy_run = is_dummy_run
        self.batch_size = batch_size
        self.graph_bs = graph_bs
        self.is_draft = is_draft
        self.input_ids = input_ids


@dataclass
class AttentionMetaData:
    """Attention metadata for prefill and decode batched together."""

    cu_seqlens_q: Optional[torch.Tensor] = None
    cu_seqlens_k: Optional[torch.Tensor] = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    min_seqlen_q: int = 0
    slot_mapping: Optional[torch.Tensor] = None
    context_lens: Optional[torch.Tensor] = None
    block_tables: Optional[torch.Tensor] = None
    dropout_p: float = 0.0

    kv_indptr: Optional[torch.Tensor] = None
    kv_indices: Optional[torch.Tensor] = None
    kv_last_page_lens: Optional[torch.Tensor] = None
    cu_seqlen_ks: Optional[torch.Tensor] = None
    cu_seqlen_ke: Optional[torch.Tensor] = None
    sparse_kv_indptr: Optional[torch.Tensor] = None

    work_meta_data: Optional[torch.Tensor] = None
    work_indptr: Optional[torch.Tensor] = None
    work_info_set: Optional[torch.Tensor] = None
    reduce_indptr: Optional[torch.Tensor] = None
    reduce_final_map: Optional[torch.Tensor] = None
    reduce_partial_map: Optional[torch.Tensor] = None

    block_tables_converted: Optional[torch.Tensor] = None

    # for prefix cache
    has_cached: bool = False
    total_kv: Optional[int] = None
    num_cached_tokens: Optional[torch.Tensor] = None
    seq_starts: Optional[torch.Tensor] = None

    # only used for plugin mode to store the metadata for attn
    plugin_metadata: Optional["MetadataForPluginMode"] = None

    def __init__(
        self,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: int = 0,
        max_seqlen_k: int = 0,
        min_seqlen_q: int = 0,
        slot_mapping: Optional[torch.Tensor] = None,
        context_lens: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        kv_indptr: Optional[torch.Tensor] = None,
        kv_indices: Optional[torch.Tensor] = None,
        kv_last_page_lens: Optional[torch.Tensor] = None,
        cu_seqlen_ks: Optional[torch.Tensor] = None,
        cu_seqlen_ke: Optional[torch.Tensor] = None,
        sparse_kv_indptr: Optional[torch.Tensor] = None,
        work_meta_data: Optional[torch.Tensor] = None,
        work_indptr: Optional[torch.Tensor] = None,
        work_info_set: Optional[torch.Tensor] = None,
        reduce_indptr: Optional[torch.Tensor] = None,
        reduce_final_map: Optional[torch.Tensor] = None,
        reduce_partial_map: Optional[torch.Tensor] = None,
        block_tables_converted: Optional[torch.Tensor] = None,
        sparse_cu_seqlens_q: Optional[torch.Tensor] = None,
        token_to_seq_idxs: Optional[torch.Tensor] = None,
        plugin_metadata: Optional["MetadataForPluginMode"] = None,
        has_cached: bool = False,
        total_kv: Optional[int] = None,
        num_cached_tokens: Optional[torch.Tensor] = None,
        seq_starts: Optional[torch.Tensor] = None,
    ):
        self.has_cached = has_cached
        self.total_kv = total_kv
        self.num_cached_tokens = num_cached_tokens
        self.seq_starts = seq_starts
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_k = max_seqlen_k
        self.min_seqlen_q = min_seqlen_q
        self.slot_mapping = slot_mapping
        self.context_lens = context_lens
        self.block_tables = block_tables
        self.dropout_p = dropout_p
        self.kv_indptr = kv_indptr
        self.kv_indices = kv_indices
        self.kv_last_page_lens = kv_last_page_lens
        self.cu_seqlen_ks = cu_seqlen_ks
        self.cu_seqlen_ke = cu_seqlen_ke
        self.sparse_kv_indptr = sparse_kv_indptr
        self.work_meta_data = work_meta_data
        self.work_indptr = work_indptr
        self.work_info_set = work_info_set
        self.reduce_indptr = reduce_indptr
        self.reduce_final_map = reduce_final_map
        self.reduce_partial_map = reduce_partial_map
        if block_tables_converted is not None:
            self.block_tables = block_tables_converted
        self.sparse_cu_seqlens_q = sparse_cu_seqlens_q
        self.token_to_seq_idxs = token_to_seq_idxs
        if plugin_metadata is not None:
            self.plugin_metadata = plugin_metadata

    def asdict_zerocopy(self, skip_fields: Optional[Set[str]] = None) -> Dict[str, Any]:
        """Similar to dataclasses.asdict, but avoids deepcopying."""
        if skip_fields is None:
            skip_fields = set()
        # Note that if we add dataclasses as fields, they will need
        # similar handling.
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name not in skip_fields
        }


@dataclass
class GDNAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_spec_decodes: int
    num_spec_decode_tokens: int
    num_actual_tokens: int

    has_initial_state: torch.Tensor | None = None

    spec_query_start_loc: torch.Tensor | None = None  # shape: [num_spec_decodes + 1,]
    non_spec_query_start_loc: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes + 1,]
    )

    spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
    non_spec_state_indices_tensor: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes,]
    )
    spec_sequence_masks: torch.Tensor | None = None  # shape: [batch,]
    spec_token_indx: torch.Tensor | None = None
    non_spec_token_indx: torch.Tensor | None = None

    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None


@dataclass
class ForwardContext:
    # copy from vllm_config.compilation_config.static_forward_context
    no_compile_layers: dict[int, Any] = field(default_factory=dict)

    attn_metadata: Optional[
        Union["AttentionMetaData", dict[str, "AttentionMetaData"]]
    ] = None

    kv_cache_data: dict[str, KVCacheTensor] = None

    context: Optional[Context] = None

    dp_metadata: Optional[DPMetadata] = None

    spec_decode_metadata: Optional[SpecDecodeMetadata] = None

    ubatch_slices: Optional[list[Any]] = None

    def __post_init__(self):
        if not hasattr(self, "no_compile_layers") or self.no_compile_layers is None:
            self.no_compile_layers = {}
        if self.attn_metadata is None:
            self.attn_metadata = {}


_forward_context: Optional[ForwardContext] = ForwardContext()
_forward_kv_cache_context: Optional[ForwardContext] = ForwardContext()

# Thread-local storage for TBO dual-thread execution

_forward_context_local = threading.local()


def get_forward_context() -> ForwardContext:
    """Get the current forward context."""
    # Check thread-local first (used by TBO threads)
    ctx = getattr(_forward_context_local, "ctx", None)
    if ctx is not None:
        return ctx

    assert _forward_context is not None, (
        "Forward context is not set. "
        "Please use `set_forward_context` to set the forward context."
    )
    return _forward_context


def set_forward_context(
    attn_metadata: AttentionMetaData,
    atom_config: Config,
    context: Context,
    num_tokens: Optional[int] = None,
    num_tokens_across_dp: Optional[torch.Tensor] = None,
    spec_decode_metadata: Optional[SpecDecodeMetadata] = None,
    ubatch_slices: Optional[list[Any]] = None,
) -> None:
    global _forward_context
    dp_metadata: Optional[DPMetadata] = None
    if atom_config.parallel_config.data_parallel_size > 1 and num_tokens is not None:
        dp_metadata = DPMetadata.make(
            atom_config.parallel_config,
            # attn_metadata,
            num_tokens or 0,
            num_tokens_across_dp,
        )

    _forward_context = ForwardContext(
        attn_metadata=attn_metadata,
        no_compile_layers=atom_config.compilation_config.static_forward_context,
        kv_cache_data=_forward_kv_cache_context.kv_cache_data,
        context=context,
        dp_metadata=dp_metadata,
        spec_decode_metadata=spec_decode_metadata,
        ubatch_slices=ubatch_slices,
    )  # _forward_context.attn_metadata = attn_metadata
    # _forward_context.no_compile_layers = atom_config.compilation_config.static_forward_context
    # _forward_context = ForwardContext(no_compile_layers=atom_config.compilation_config.static_forward_context, attn_metadata=attn_metadata)


def reset_forward_context() -> None:
    global _forward_context
    _forward_context = ForwardContext()


# ---------------------------------------------------------------------------
# KV Connector global instances (lazy initialization)
# ---------------------------------------------------------------------------

_logger = logging.getLogger("atom")

_global_kvconnector: Optional[Any] = None
_global_kvconnector_scheduler: Optional[Any] = None


def get_kvconnector(role: str = "worker", config: Optional[Config] = None) -> Any:
    """Get or lazily initialize the global KV connector instance.

    The connector is role-dependent:
      - ``"worker"``: Returns a :class:`KVConnectorBase` (worker-side, per TP rank).
      - ``"scheduler"``: Returns a :class:`KVConnectorSchedulerBase` (scheduler-side).

    The concrete backend is selected by :class:`KVConnectorFactory` based on
    ``config.kv_transfer_config["kv_connector"]`` (default: ``"moriio"``).

    Args:
        role: Either ``"worker"`` or ``"scheduler"``.
        config: Engine config; required on first call to trigger initialization.

    Returns:
        The KV connector instance, or ``None`` if KV transfer is not configured.
    """
    global _global_kvconnector, _global_kvconnector_scheduler

    if not (hasattr(config, "kv_transfer_config") and config.kv_transfer_config):
        return _global_kvconnector

    if role == "worker":
        from aiter.dist.parallel_state import get_tp_group

        try:
            tp_rank = get_tp_group().rank_in_group
        except Exception:
            _logger.warning(
                "get_tp_group() failed (dist not initialized?), returning None"
            )
            return None

        if _global_kvconnector is None:
            from atom.kv_transfer.disaggregation import KVConnectorFactory

            _global_kvconnector = KVConnectorFactory.create_connector(
                config, role="worker"
            )
            _logger.debug("Initialized global KVConnector at tp_rank %d", tp_rank)

    elif role == "scheduler":
        from atom.kv_transfer.disaggregation import KVConnectorFactory

        _global_kvconnector_scheduler = KVConnectorFactory.create_connector(
            config, role="scheduler"
        )
        _logger.debug("Initialized global KVConnectorScheduler")
        return _global_kvconnector_scheduler

    else:
        raise ValueError(f"Unknown KV connector role: {role!r}")

    return _global_kvconnector


def set_kv_cache_data(
    kv_cache_data: dict[int, KVCacheTensor],
    config: Optional[Config] = None,
    transfer_tensors: Any = None,
) -> None:
    """Register KV cache data globally and with the KV connector if enabled."""
    global _forward_kv_cache_context

    if hasattr(config, "kv_transfer_config") and config.kv_transfer_config:
        connector = get_kvconnector(config=config)
        if connector is not None:
            connector.register_kv_caches(kv_cache_data, transfer_tensors)

    _forward_kv_cache_context.kv_cache_data = kv_cache_data
