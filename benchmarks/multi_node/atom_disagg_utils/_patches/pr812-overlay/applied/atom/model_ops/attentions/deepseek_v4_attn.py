# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek V4 hybrid-attention backend.

Per paper §3.6.1, V4 splits cache into two parts:

  1. State cache (per-request, fixed-size pool, dynamically assigned)
     - SWA segment: most recent n_win tokens KV per layer (every layer)
     - Compressor tail buffers: uncompressed pending tokens + scores
       (CSA Main / CSA Indexer / HCA Main, fp32 for softmax-pool stability)

  2. Classical KV cache (PagedAttention-style, multi-block per request,
     block_size = lcm(m, m'))
     - CSA Main compressed KV
     - CSA Indexer compressed KV
     - HCA Main compressed KV

PR3-pre2a  (done): Compressor state buffers (kv_state + score_state ×3 owners)
                   migrated to per_req_cache pool.
PR3-pre2c-A (done): SWA buffer migration to per_req_cache pool.
PR3-pre2c-B (this revision): classical KV cache (compressed entries) moved
                   under the block_table per paper §3.6.1. Three pools allocated
                   (csa_main_kv / csa_idx_kv / hca_main_kv), shape
                   `[num_blocks, n_layers_of_type, k, head_dim]`. block_size =
                   lcm(m, m') = 128 original tokens. Compressor + Indexer
                   .kv_cache attributes bound to per-layer pool slices.
PR3-main:   multi-sequence dispatch (slot=0 -> per-seq slot).

Per-slot cost (V4-Pro, BF16 SWA + fp32 tail buffers, 30 CSA + 31 HCA + 1 dense):
  SWA:         62 layers * 128 * 512 * 2B  =  8.0 MB
  CSA Main:    30 * 2 * (8 * 1024)  * 4B   =  1.875 MB
  CSA Indexer: 30 * 2 * (8 * 256)   * 4B   =  0.469 MB
  HCA Main:    31 * 2 * (128 * 512) * 4B   = 16.0 MB
  Total                                      = ~26.5 MB / slot
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type, cast

import numpy as np
import torch
from aiter import dtypes
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_ops.attentions.backends import (
    AttentionBackend,
    AttentionMetadataBuilder,
    CommonAttentionBuilder,
)
from atom.utils import CpuGpuBuffer
from atom.utils.forward_context import AttentionMetaData, Context

# ---------------------------------------------------------------------------
# Typed metadata surface for V4. The base AttentionMetaData class is shared
# across all backends; carrying V4-specific dynamic attributes there would
# pollute it. Subclassing here gives pyright/pylance a typed surface so
# `attn_metadata.v4_kv_indices_csa` etc. don't trigger
# reportAttributeAccessIssue, while runtime behaviour stays identical
# (V4 builder constructs / promotes instances to this subclass).
# ---------------------------------------------------------------------------


@dataclass
class AttentionMetaData_DSV4(AttentionMetaData):
    """DeepSeek-V4 attention metadata.

    Extends the shared `AttentionMetaData` with V4-specific per-fwd
    metadata that `DeepseekV4AttentionMetadataBuilder` populates. The
    base class is shared across backends; carrying V4 fields there would
    pollute it. Subclassing gives pyright/pylance a typed surface so
    `attn_metadata.kv_indices_csa` etc. don't trip
    `reportAttributeAccessIssue`.

    Lifecycle: built per fwd by `prepare_decode` / `prepare_prefill` /
    `build_for_cudagraph_capture`. `is_pure_decode`-gated fields are only
    populated when the builder confirms a uniform-tokens-per-seq +
    non-fresh-prefill batch (doc §7.4); other paths leave them at
    defaults.

    Shape symbols used below:
      bs         = scheduled_bs            actual decode/prefill seqs
      padded_bs  = capture-time graph_bs   (= bs in eager / prefill)
      T          = total_tokens this fwd   (= sum of token_num_per_seq)
      padded_T   = padded_bs * max_q_len   (>= T; captured kernels iterate this)
      win        = self.window_size        (128 for V4-Pro)
      index_topk = self.index_topk         (1024 for V4-Pro)
    """

    # ----- CPU mirrors (avoid GPU→CPU `.item()` / `.tolist()` syncs) -----
    state_slot_mapping_cpu: Optional[Any] = None
    """[bs] np.int32 — per-seq state cache slot id (host copy)."""
    start_pos_per_seq_cpu: Optional[Any] = None
    """[bs] np.int64 — absolute start position per seq."""

    # ----- Per-seq GPU scalars (single-source-of-truth, shared by kernels) -----
    state_slot_mapping: Optional[torch.Tensor] = None
    """[bs] int32 GPU — per-seq state cache slot. Shared by swa_write +
    Compressor + paged-decode kernels (looked up via batch_id_per_token)."""
    n_committed_csa_per_seq: Optional[torch.Tensor] = None
    """[bs] int32 GPU — `ctx_len // 4`. Consumed by csa_translate_pack
    (kernel masks `(k < n_committed) & (k < index_topk)` — clamp lives in
    kernel, not builder) AND by the indexer (cast to long inline)."""

    # ----- Per-fwd hoisted (built in `_attach_v4_per_fwd_meta`) -----
    batch_id_per_token: Optional[torch.Tensor] = None
    """[padded_T] int64 GPU — the SINGLE per-token mapping
    (token_idx → seq_idx). int64 dtype is required by PyTorch fancy-index
    (used in the indexer); triton kernels (swa_write, csa_translate_pack)
    read int64 fine. Padded tail [T:padded_T] = -1 sentinel; consumer
    kernels skip on `bid < 0`. All other per-token quantities resolved as
    `per_seq_data[batch_id_per_token[t]]` — no [T] aliases of seq data."""
    swa_write_indices: Optional[torch.Tensor] = None
    """[padded_T] int64 GPU — src row id into per-fwd KV for swa_write.
    `[0:num_write]` = real (last `win` tokens per seq);
    `[num_write:padded_T]` = -1 sentinel. `None` for warmup / empty fwd."""
    compress_plans: Optional[Dict[int, Any]] = None
    """dict[ratio:int -> CompressPlan] — packed plan tensors per
    compress_ratio (4=CSA, 128=HCA)."""

    # ----- Phase B paged-decode metadata (set when is_pure_decode == True) -----
    is_pure_decode: bool = False
    """uniform tokens-per-seq AND no fresh prefill (doc §7.4) — gates the
    paged-decode dispatch in V4Attention.forward."""
    kv_indices_swa: Optional[torch.Tensor] = None
    """[T*win] int32 GPU — flat paged offsets into `unified_kv` for SWA path."""
    kv_indices_csa: Optional[torch.Tensor] = None
    """[csa_indptr[T]] int32 GPU — packed paged offsets for CSA layers
    (window prefix + per-token compress entries via csa_translate_pack)."""
    kv_indices_hca: Optional[torch.Tensor] = None
    """[hca_indptr[T]] int32 GPU — packed paged offsets for HCA layers
    (window prefix + n_committed_hca compress entries; layer-invariant)."""
    kv_indptr_swa: Optional[torch.Tensor] = None
    """[padded_T+1] int32 GPU — uniform stride `win` cumsum;
    `[T+1:padded_T+1]` = T*win (last value repeated → kv_len=0 sentinel)."""
    kv_indptr_csa: Optional[torch.Tensor] = None
    """[padded_T+1] int32 GPU — packed cumsum of per-token CSA kv_len
    (= `win + min(n_committed_csa[bid], index_topk)`).
    Padded tail = last value."""
    kv_indptr_hca: Optional[torch.Tensor] = None
    """[padded_T+1] int32 GPU — packed cumsum of per-token HCA kv_len
    (= `win + n_committed_hca[bid]`). Padded tail = last value."""
    swa_pages: int = 0
    """Boundary in `unified_kv`: index < swa_pages → SWA region; index >=
    swa_pages → compress region. Equal to `max_per_req_cache_slots * win`."""

    # ----- Indexer / sparse-layout side metadata -----
    indexer_meta: Optional[Dict[str, Any]] = None
    """dict — `Indexer.forward_batched` per-fwd GPU tensors. Notable keys:
      cu_committed_gpu              [bs+1] int32  per-seq committed cumsum
      seq_base_per_token_gpu        [T] int32  prefill subtract base (also
                                                aliased as cu_starts_gpu for
                                                fp8_mqa_logits)
      cu_ends_gpu                   [T] int32  per-token end offset for
                                                fp8_mqa_logits (causal cap)
      total_committed               int  sum of n_committed_csa_per_seq

    Note: decode logits / topk-indices scratch are allocated per-fwd inside
    `Indexer._score_topk_decode` (write-once, no CPU mirror, CG-stable via
    the captured graph's private memory pool).

    The indexer's downstream contract: `_score_topk_*` returns RAW seq-local
    `[T, index_topk] int32` with kernel-native -1 in tail cols (cells past
    the per-token visibility cap). `csa_translate_pack` consumes this layout
    directly — no separate width-mask / offset / future-threshold staging
    needed.
    """
    skip_prefix_len_csa: Optional[torch.Tensor] = None
    """[padded_T] int32 GPU — per-token write offset for csa_translate_pack
    within the per-token prefix region. Decode path: filled with
    `window_size` (full SWA prefix occupies the head of each region).
    Prefill path: equals `prefix_swa_count_per_token[t]` — 0 for pure
    prefill (no prior chunk), or the `< chunk_start` portion of the SWA
    window for chunked prefill. CG-padded tail slots: 0 (kernel bails on
    `bid<0` so the value is irrelevant)."""

    # ----- Prefill-only paged-prefill index buffers (set in `_build_paged_prefill_meta`) -----
    # Two-source paged_prefill kernel reads:
    #   prefix region from `unified_kv` (SWA history + CSA/HCA compress)
    #   extend region from per-fwd `kv` tensor (in-chunk SWA tail)
    # Per-ratio prefix buffers (SWA-only stride for Dense, SWA + compress
    # for CSA/HCA). Extend buffer is layer-invariant, shared by all 3.
    kv_indices_prefix_swa: Optional[torch.Tensor] = None
    """[sum(prefix_swa_count)] int32 GPU — flat paged offsets into
    `unified_kv` for Dense (ratio==0) layers' prefix region (SWA history
    only)."""
    kv_indptr_prefix_swa: Optional[torch.Tensor] = None
    """[total_tokens + 1] int32 GPU — packed cumsum of `prefix_swa_count`."""
    kv_indices_prefix_csa: Optional[torch.Tensor] = None
    """[sum(prefix_swa_count + min(n_csa, index_topk))] int32 GPU — SWA
    history (head) + CSA topk (tail) per token. SWA section is filled by
    builder; CSA section is filled per-layer by `csa_translate_pack`."""
    kv_indptr_prefix_csa: Optional[torch.Tensor] = None
    """[total_tokens + 1] int32 GPU — packed cumsum of
    `prefix_swa_count + min(n_committed_csa, index_topk)`."""
    kv_indices_prefix_hca: Optional[torch.Tensor] = None
    """[sum(prefix_swa_count + n_committed_hca)] int32 GPU — SWA history
    (head) + HCA all-committed compress entries (tail). Layer-invariant,
    fully filled by builder."""
    kv_indptr_prefix_hca: Optional[torch.Tensor] = None
    """[total_tokens + 1] int32 GPU — packed cumsum of
    `prefix_swa_count + n_committed_hca`."""
    kv_indices_extend: Optional[torch.Tensor] = None
    """[sum(extend_count)] int32 GPU — flat row offsets into the per-fwd
    `kv` tensor (in-chunk SWA tail) for the extend region. Layer-invariant
    (same `kv` shared by all 3 ratios; one builder pass)."""
    kv_indptr_extend: Optional[torch.Tensor] = None
    """[total_tokens + 1] int32 GPU — packed cumsum of `extend_count`."""


# ---------------------------------------------------------------------------
# Builder-local helpers (private). Used by `_build_paged_prefill_meta` and
# `_attach_v4_per_fwd_meta` for ragged-segment index math + per-token
# sliding-window topk index generation. Live here (not in the model file)
# because their only callers are inside this builder.
# ---------------------------------------------------------------------------


def _segment_indices(
    seq_ids: np.ndarray, lens: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """For ragged segments (one per `seq_ids[i]` of length `lens[i]`), return
    flat (per-row seq id, per-row local position) arrays of total length
    `sum(lens)`.
    """
    total = int(lens.sum())
    if total == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
    token_seq_ids = np.repeat(seq_ids.astype(np.int64), lens)
    cum = np.concatenate(([0], np.cumsum(lens.astype(np.int64))[:-1]))
    local_pos = np.arange(total, dtype=np.int64) - np.repeat(cum, lens)
    return token_seq_ids, local_pos


def _build_window_topk_batched(
    positions: torch.Tensor,  # [total_tokens] int (abs token positions)
    start_pos_per_token: torch.Tensor,  # [total_tokens] int (each token's seq start_pos)
    window_size: int,
    out: Optional[
        torch.Tensor
    ] = None,  # [total_tokens, window_size] int32 — CG-stable buffer
) -> torch.Tensor:  # [total_tokens, window_size] int32
    """Per-token sliding-window topk indices for the whole batch.

    Three-branch semantics:
      - sp == 0 (fresh prefill): matrix entries = abs positions in the window
        [pos-win+1, pos] clamped to [0, pos]; mask future via -1.
      - 0 < sp < win-1 (prefix mode): all tokens in the seq share a single
        matrix [0..sp, -1, ..., -1] (matches original semantics, including
        MTP-N where the same start_pos is reused).
      - sp >= win-1 (cyclic mode): cyclic ring offsets starting at sp+1 mod win.

    `out`: when supplied, the final int32 result is written into `out`
    (sliced to `[total_tokens, window_size]`) so its storage address is
    stable across captures — required for CUDAGraph replay correctness.
    Intermediates remain transient (not consumed by the captured graph).
    """
    device = positions.device
    total = positions.size(0)
    arange_w = torch.arange(window_size, device=device, dtype=positions.dtype).view(
        1, window_size
    )
    pos_col = positions.view(total, 1)
    sp_col = start_pos_per_token.view(total, 1)
    neg1 = positions.new_full((), -1)

    # Case A: sp == 0 (fresh prefill) — abs positions [pos-win+1, pos] clamped.
    case_a = (pos_col - window_size + 1).clamp(min=0) + arange_w
    case_a = torch.where(case_a > pos_col, neg1, case_a)

    # Case B: 0 < sp < win-1 (prefix mode) — shared per-seq matrix.
    case_b = arange_w.expand(total, window_size).clone()
    case_b = torch.where(arange_w > sp_col, neg1, case_b)

    # Case C: sp >= win-1 (cyclic mode) — ring offsets.
    sp_mod = sp_col % window_size
    case_c = (sp_mod + 1 + arange_w) % window_size

    sp_eq_0 = sp_col == 0
    sp_in_prefix = (sp_col > 0) & (sp_col < window_size - 1)

    res = case_c
    res = torch.where(sp_in_prefix.expand_as(res), case_b, res)
    res = torch.where(sp_eq_0.expand_as(res), case_a, res)
    if out is None:
        return res.to(torch.int32)
    out[:total].copy_(res, non_blocking=True)
    return out[:total]


def _build_window_topk_np(
    positions: np.ndarray,
    start_pos_per_token: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """CPU numpy equivalent of ``_build_window_topk_batched``.

    For small decode batches the ~25 GPU kernel launches dominate; this
    numpy version runs in microseconds on CPU and feeds a single H2D copy.
    """
    total = positions.shape[0]
    arange_w = np.arange(window_size, dtype=np.int64).reshape(1, window_size)
    pos_col = positions.reshape(total, 1)
    sp_col = start_pos_per_token.reshape(total, 1)

    case_a = np.maximum(pos_col - window_size + 1, 0) + arange_w
    case_a = np.where(case_a > pos_col, -1, case_a)

    case_b = np.broadcast_to(arange_w, (total, window_size)).copy()
    case_b = np.where(arange_w > sp_col, -1, case_b)

    sp_mod = sp_col % window_size
    case_c = (sp_mod + 1 + arange_w) % window_size

    sp_eq_0 = sp_col == 0
    sp_in_prefix = (sp_col > 0) & (sp_col < window_size - 1)

    res = case_c
    res = np.where(sp_in_prefix, case_b, res)
    res = np.where(sp_eq_0, case_a, res)
    return res.astype(np.int32)


class DeepseekV4Backend(AttentionBackend):
    """Backend selector entry for V4 hybrid attention.

    V4 forward is custom (does not go through ATOM's standard AttentionImpl);
    this backend exists primarily so the metadata builder is reachable from
    `ModelRunner.attn_metadata_builder` and the per-request cache abstraction
    can size + own V4's state caches.
    """

    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_V4"

    @staticmethod
    def get_builder_cls() -> Type["AttentionMetadataBuilder"]:
        return DeepseekV4AttentionMetadataBuilder


class DeepseekV4AttentionMetadataBuilder(CommonAttentionBuilder):
    """Per-request cache owner for V4's state-cache buffers.

    Inherits CommonAttentionBuilder for the standard prefill/decode prep
    (slot_mapping, block_tables, cu_seqlens). PR3-pre2c-B sets `block_size`
    to lcm(m, m') = 128 (V4-Pro: m=4 CSA, m'=128 HCA), matching paper §3.6.1's
    requirement that each classical KV cache block hold an integral number of
    compressed entries per layer (k1=lcm/m=32 CSA, k2=lcm/m'=1 HCA).
    """

    block_size = 128

    def __init__(self, model_runner):
        super().__init__(model_runner)
        hf = model_runner.config.hf_config
        ratios = list(getattr(hf, "compress_ratios", ()))
        assert ratios, "deepseek_v4 hf_config must define compress_ratios"
        self.compress_ratios = ratios
        self.num_layers = len(ratios)
        # Per-buffer-type layer indexing.
        # Buffers are layer-major: shape [num_layers_of_type, num_slots, *state_shape].
        self.csa_layers = [i for i, r in enumerate(ratios) if r == 4]
        self.hca_layers = [i for i, r in enumerate(ratios) if r == 128]
        self.dense_layers = [i for i, r in enumerate(ratios) if r == 0]
        self.layer_id_to_csa_pos = {lid: p for p, lid in enumerate(self.csa_layers)}
        self.layer_id_to_hca_pos = {lid: p for p, lid in enumerate(self.hca_layers)}
        # Unique (ratio, is_overlap) pairs needed for compress-plan generation.
        # CSA ratio=4 has overlap=True; HCA ratio=128 has overlap=False.
        unique = []
        if self.csa_layers:
            unique.append((4, True))
        if self.hca_layers:
            unique.append((128, False))
        self._unique_compress_ratios_overlap = unique

        # Geometry from HF config.
        self.head_dim = getattr(hf, "kv_head_dim", 512)
        self.index_head_dim = getattr(hf, "index_head_dim", 128)
        self.window_size = getattr(hf, "sliding_window", 128)
        self.index_topk = getattr(hf, "index_topk", 1024)
        # `deepgemm_fp8_paged_mqa_logits` decode-path output column count
        # = max compressed K positions per seq. CSA ratio=4 is the
        # max-density ratio (1 indexer slot per 4 source tokens).
        self.max_model_len_idx = model_runner.config.max_model_len // 4

        # Compressor state shape: [coff * ratio, coff * head_dim], fp32.
        # CSA: ratio=4, overlap=True -> coff=2 -> [8, 2*head_dim]
        # HCA: ratio=128, overlap=False -> coff=1 -> [128, head_dim]
        self.csa_main_state_shape = (2 * 4, 2 * self.head_dim)
        self.csa_idx_state_shape = (2 * 4, 2 * self.index_head_dim)
        self.hca_main_state_shape = (128, self.head_dim)

        # Classical KV pool geometry. block_size=128 original tokens means
        # each V4 block holds k1=128/4=32 CSA entries and k2=128/128=1 HCA
        # entry per layer (paper §3.6.1).
        self.k1_csa = self.block_size // 4  # = 32
        self.k2_hca = self.block_size // 128  # = 1

        self._state_dtype = torch.float32  # fp32 required for softmax-pool
        self._swa_dtype = torch.bfloat16  # SWA window matches KV dtype
        self._classical_dtype = torch.bfloat16  # CSA Main / HCA Main KV is BF16
        # CSA Indexer cache is FP8 + 4-byte fp32 scale per row, aligned to 16
        # bytes (matches V3.2 sparse MLA pattern; avoids torch inductor
        # unaligned-access slowdowns). Written by `indexer_k_quant_and_cache`,
        # read by `cp_gather_indexer_k_quant_cache`.
        self._aligned_index_dim = ((self.index_head_dim + 4 + 15) // 16) * 16

        # MTP token-per-fwd factor for paged-decode buffer sizing. V4-Pro
        # `num_nextn_predict_layers = 1` → mtp_k = 1 → max_q_len = 2 per req.
        # `model_runner.drafter` is created BEFORE `attn_metadata_builder`
        # (model_runner.__init__ ordering), so this hasattr is reliable.
        self.max_spec_steps = (
            int(model_runner.drafter.mtp_k) if hasattr(model_runner, "drafter") else 0
        )
        self.max_decode_tokens = self.max_bs * (1 + self.max_spec_steps)
        # Worst-case HCA per-token committed compress count
        # (= max_model_len // 128 for V4-Pro = 8192 at 1M context).
        self.max_committed_hca = model_runner.config.max_model_len // 128

        # Sparse-attn + per-fwd metadata buffers (CG-A: pre-allocate for fixed
        # GPU pointers, prerequisite for CUDAGraph capture). All H2D copies in
        # the V4 metadata builder go through these buffers via the
        # `np[:n] = arr; copy_to_gpu(n)` pattern instead of per-call
        # `torch.as_tensor(arr)` allocations.
        self._alloc_v4_metadata_buffers()

    @property
    def prep_stream(self):
        return self.model_runner.async_execute_stream

    # ------------------------------------------------------------------ #
    # AttentionMetadataBuilder hooks (per-request cache abstraction).    #
    # ------------------------------------------------------------------ #

    def compute_per_req_cache_bytes(self) -> int:
        """Bytes for ONE request's state cache across all layers.

        State cache contents (paper §3.6.1):
          - SWA segment: [n_win, head_dim] BF16, every layer.
          - Compressor tail buffers: [kv_state, score_state] fp32 pairs
            for every Compressor instance (CSA Main / CSA Indexer / HCA Main).
        """
        elem_state = self._state_dtype.itemsize  # fp32 = 4
        elem_swa = self._swa_dtype.itemsize  # bf16 = 2
        # Tail buffers (kv_state + score_state pair per Compressor instance).
        csa_main = self._numel(self.csa_main_state_shape) * 2 * elem_state
        csa_idx = self._numel(self.csa_idx_state_shape) * 2 * elem_state
        hca_main = self._numel(self.hca_main_state_shape) * 2 * elem_state
        # SWA window per layer.
        swa_per_layer = self.window_size * self.head_dim * elem_swa
        return (
            len(self.csa_layers) * (csa_main + csa_idx)
            + len(self.hca_layers) * hca_main
            + self.num_layers * swa_per_layer
        )

    def slots_per_req(self) -> int:
        # No spec decoding lookahead in V4 (pre-PR3-main).
        return 1

    def compute_block_bytes(self) -> int:
        """Per-V4-block bytes for the three classical KV pools.

        Each V4 block (block_size=128 original tokens) stores per layer:
          - CSA Main:   k1=32 entries × head_dim BF16
          - CSA Indexer: k1=32 entries × aligned_index_dim bytes FP8
                        (= ((index_head_dim + 4 + 15) // 16) * 16 — 16-byte
                        alignment matches V3.2 sparse MLA index cache and
                        avoids unaligned-access slowdowns in torch inductor.
                        FP8 quantized data + 4-byte fp32 scale interleaved
                        per row; written by `indexer_k_quant_and_cache`,
                        read by `cp_gather_indexer_k_quant_cache`).
          - HCA Main:   k2=1 entry × head_dim BF16
        """
        elem_bf16 = self._classical_dtype.itemsize
        csa_main_per_block = self.k1_csa * self.head_dim * elem_bf16
        csa_idx_per_block = self.k1_csa * self._aligned_index_dim  # fp8 = 1B
        hca_main_per_block = self.k2_hca * self.head_dim * elem_bf16
        return (
            len(self.csa_layers) * (csa_main_per_block + csa_idx_per_block)
            + len(self.hca_layers) * hca_main_per_block
        )

    def allocate_kv_cache_tensors(
        self, num_kv_heads: int, num_draft_layers: int
    ) -> dict[str, torch.Tensor]:
        """Allocate KV pools that depend only on `num_blocks`.

        After Phase A (CG-friendly indexer), the SWA window AND the per-layer
        compressed pool are physically merged into a single `unified_kv`
        tensor per layer (allocated in `allocate_per_req_cache`, which is
        called later when both `num_blocks` and `num_slots` are known).

        Only the CSA Indexer FP8 cache stays as a standalone batched tensor
        — it lives in its own dtype (FP8 + fp32 scale) and is consumed by
        `cp_gather_indexer_k_quant_cache`, not the sparse-attn kernel.
        Layer-major axis order `[n_csa, NB, k1, aligned_dim]` so each
        per-CSA slice `pool[pos]` is contiguous in storage; the kernel
        infers `block_size` from `kv_cache.shape[1]`.
        """
        runner = self.model_runner
        device = runner.device
        num_blocks = runner.num_physical_kvcache_blocks
        n_csa = len(self.csa_layers)
        return {
            "v4_csa_idx_kv": torch.zeros(
                (n_csa, num_blocks, self.k1_csa, self._aligned_index_dim),
                dtype=dtypes.fp8,
                device=device,
            ),
        }

    def allocate_per_req_cache(self, num_slots: int) -> dict[str, object]:
        """Allocate per-layer `unified_kv` + Compressor state caches.

        Per-layer `unified_kv` layout (decode-time paged_decode kernel reads
        a single base ptr; offsets `[0, swa_pages)` are SWA, `[swa_pages, ..)`
        are compress):
            Dense layer: [num_slots*win,            head_dim] BF16
            CSA   layer: [num_slots*win + NB*k1,    head_dim] BF16
            HCA   layer: [num_slots*win + NB*k2,    head_dim] BF16

        `build_kv_cache_tensor` slices per-layer views to bind into
        `attn.swa_kv` (SWA portion, reshape to [num_slots, win, head_dim])
        and `compressor.kv_cache` (compress portion, reshape to
        [num_blocks, k_per_block, head_dim]). The full unified pool is also
        bound as `attn.unified_kv` for the paged_decode dispatch.

        Tensors are setattr'd onto ModelRunner; `v4_unified_kv` is a list of
        per-layer tensors (length `num_layers`). State caches stay
        layer-major batched (compressor scatter binds per-layer slices).

        Total bytes match the pre-Phase-A layout (SWA + CSA Main + HCA Main
        bytes redistributed across per-layer tensors; no extra overhead).
        """
        assert self._swa_dtype == self._classical_dtype, (
            "unified_kv requires SWA dtype == classical KV dtype "
            f"(got SWA={self._swa_dtype}, classical={self._classical_dtype})"
        )
        device = self.model_runner.device
        num_blocks = self.model_runner.num_physical_kvcache_blocks
        n_csa = len(self.csa_layers)
        n_hca = len(self.hca_layers)
        swa_pages = num_slots * self.window_size
        head_dim = self.head_dim
        dtype = self._swa_dtype

        # Per-layer unified_kv: SWA prefix + (CSA/HCA) compress tail.
        unified_kv: list[torch.Tensor] = []
        ratios = self.compress_ratios
        for layer_id in range(self.num_layers):
            ratio = ratios[layer_id]
            if ratio == 4:
                compress_pages = num_blocks * self.k1_csa
            elif ratio == 128:
                compress_pages = num_blocks * self.k2_hca
            else:
                compress_pages = 0  # Dense
            unified_kv.append(
                torch.zeros(
                    (swa_pages + compress_pages, head_dim),
                    dtype=dtype,
                    device=device,
                )
            )

        # ---- Compressor state tensors (compute-contiguous) ------------------
        csa_main_kv = self._zero_state(
            (n_csa, num_slots, *self.csa_main_state_shape), device
        )
        csa_main_score = self._neg_inf_state(
            (n_csa, num_slots, *self.csa_main_state_shape), device
        )
        csa_idx_kv = self._zero_state(
            (n_csa, num_slots, *self.csa_idx_state_shape), device
        )
        csa_idx_score = self._neg_inf_state(
            (n_csa, num_slots, *self.csa_idx_state_shape), device
        )
        hca_main_kv = self._zero_state(
            (n_hca, num_slots, *self.hca_main_state_shape), device
        )
        hca_main_score = self._neg_inf_state(
            (n_hca, num_slots, *self.hca_main_state_shape), device
        )

        # ---- RDMA staging pool, for RDMA performance ---------------------
        state_tensors = [
            csa_main_kv,
            csa_main_score,
            csa_idx_kv,
            csa_idx_score,
            hca_main_kv,
            hca_main_score,
        ]
        state_slot_stride = sum(t[0, 0].numel() * t.shape[0] for t in state_tensors)
        pool_size = int(os.environ.get("ATOM_PD_STAGING_POOL", "8"))
        state_pool = torch.zeros(
            pool_size * state_slot_stride,
            dtype=self._state_dtype,
            device=device,
        )

        return {
            "v4_unified_kv": unified_kv,
            "v4_csa_main_kv_state": csa_main_kv,
            "v4_csa_main_score_state": csa_main_score,
            "v4_csa_idx_kv_state": csa_idx_kv,
            "v4_csa_idx_score_state": csa_idx_score,
            "v4_hca_main_kv_state": hca_main_kv,
            "v4_hca_main_score_state": hca_main_score,
            "v4_state_pool": state_pool,
            "v4_state_pool_size": pool_size,
            "v4_state_slot_stride": state_slot_stride,
        }

    def build_kv_cache_tensor(self, layer_id: int, module):
        """Bind V4 modules' state-cache + classical-cache views.

        Called by ModelRunner.allocate_kv_cache() for every nn.Module:
          - V4 Attention: bind swa_kv (per_req_cache pool).
          - V4 Compressor: bind kv_state, score_state (per_req_cache pool)
            AND kv_cache (classical pool slice — per CSA/HCA layer).
          - V4 Indexer:    bind kv_cache (csa_idx_kv slice — per CSA layer).

        Returns None always — V4 forward consumes module attributes directly,
        not the global `forward_context.kv_cache_data` registry that ATOM's
        standard MHA path uses.
        """
        # Local imports to avoid circular dependency at module load time.
        from atom.models.deepseek_v4 import (
            Compressor as _V4Compressor,
            DeepseekV4Attention as _V4Attention,
            Indexer as _V4Indexer,
        )

        runner = self.model_runner
        num_slots = self.model_runner.max_per_req_cache_slots
        swa_pages = num_slots * self.window_size

        if isinstance(module, _V4Attention):
            # Bind both:
            #   - `attn.unified_kv`: the full per-layer pool (paged_decode reads).
            #   - `attn.swa_kv`: a [num_slots, win, head_dim] view onto the
            #     SWA prefix (compatibility with `swa_write` and the existing
            #     prefill/non-CG path that index_select's by state slot).
            unified = runner.v4_unified_kv[module.layer_id]
            module.unified_kv = unified
            module.swa_kv = unified[:swa_pages].view(
                num_slots, self.window_size, self.head_dim
            )
            return None

        if isinstance(module, _V4Indexer):
            # Indexer.kv_cache — CSA Indexer compressed pool, per CSA layer.
            # prefix: "layers.<L>.attn.indexer"
            #
            # Shape MUST stay [NB, k1_csa, aligned_dim] (3D, block_size dim
            # explicit) because `cp_gather_indexer_k_quant_cache` infers
            # block_size from `kv_cache.shape[1]` to compute
            # `physical_block * block_size + slot_in_block`. Flattening to
            # [NB*k1, 1, aligned_dim] makes the kernel see block_size=1 and
            # OOB-index block_table. Matches V3.2's [num_blocks, block_size,
            # head_dim] layout (deepseek_v2.py:1049).
            layer_id_from_prefix = int(module.prefix.split(".")[1])
            pos = self.layer_id_to_csa_pos[layer_id_from_prefix]
            module.kv_cache = runner.v4_csa_idx_kv[pos]
            return None

        if isinstance(module, _V4Compressor):
            # Compressor.prefix is set by the parent constructor:
            #   "layers.<L>.attn.compressor"          -> CSA Main / HCA Main
            #   "layers.<L>.attn.indexer.compressor"  -> CSA Indexer's inner
            parts = module.prefix.split(".")
            layer_id_from_prefix = int(parts[1])
            is_indexer_inner = "indexer" in parts
            ratio = module.compress_ratio

            if is_indexer_inner:
                assert ratio == 4, "Indexer-inner Compressor only on CSA layers"
                pos = self.layer_id_to_csa_pos[layer_id_from_prefix]
                module.kv_state = runner.v4_csa_idx_kv_state[pos]
                module.score_state = runner.v4_csa_idx_score_state[pos]
                # Inner compressor writes target the SAME storage as the
                # outer Indexer.kv_cache (csa_idx_kv). Same [NB, k1_csa,
                # aligned_dim] FP8 shape — `Compressor.forward` resolves
                # slot via block_table+ci internally (no flat slot_mapping
                # needed; matches CSA Main's path).
                idx_kv = runner.v4_csa_idx_kv[pos]
                module.kv_cache = idx_kv
                # FP8 quant path: bind a strided fp32 view of the per-block
                # scale region. Layout per block: [k1*head_dim FP8 region]
                # then [k1 fp32 scale region] then padding (cache_kernels.cu
                # :1209-1239). Strides expressed in fp32 elements.
                nb, k1, aligned_dim = idx_kv.shape
                head_dim = self.index_head_dim
                assert (
                    k1 * aligned_dim
                ) % 4 == 0, f"per-block bytes ({k1 * aligned_dim}) must be 4-aligned"
                block_fp32_stride = (k1 * aligned_dim) // 4
                scale_fp32_offset = (k1 * head_dim) // 4
                module.cache_scale = (
                    idx_kv.view(torch.float32)
                    .view(-1)
                    .as_strided(
                        size=(nb, k1),
                        stride=(block_fp32_stride, 1),
                        storage_offset=scale_fp32_offset,
                    )
                )
            elif ratio == 4:
                pos = self.layer_id_to_csa_pos[layer_id_from_prefix]
                module.kv_state = runner.v4_csa_main_kv_state[pos]
                module.score_state = runner.v4_csa_main_score_state[pos]
                # CSA Main compressed pool now lives in the tail of the
                # owning layer's `unified_kv`. Compressor.forward writes via
                # `kv_cache[block_id, slot_in_block, :] = entry`, so we hand
                # it the standard [num_blocks, k1, head_dim] view.
                num_blocks = runner.num_physical_kvcache_blocks
                unified = runner.v4_unified_kv[layer_id_from_prefix]
                module.kv_cache = unified[swa_pages:].view(
                    num_blocks, self.k1_csa, self.head_dim
                )
            elif ratio == 128:
                pos = self.layer_id_to_hca_pos[layer_id_from_prefix]
                module.kv_state = runner.v4_hca_main_kv_state[pos]
                module.score_state = runner.v4_hca_main_score_state[pos]
                num_blocks = runner.num_physical_kvcache_blocks
                unified = runner.v4_unified_kv[layer_id_from_prefix]
                module.kv_cache = unified[swa_pages:].view(
                    num_blocks, self.k2_hca, self.head_dim
                )
            else:
                raise ValueError(
                    f"Unknown V4 compress_ratio={ratio} on Compressor at "
                    f"prefix={module.prefix!r}"
                )
            return None

        return super().build_kv_cache_tensor(layer_id, module)

    def get_kv_transfer_tensors(self):
        from atom.kv_transfer.disaggregation.types import (
            KVTransferRegion,
            KVTransferTensors,
        )

        runner = self.model_runner
        if not hasattr(runner, "v4_unified_kv"):
            return None

        num_slots = runner.max_per_req_cache_slots
        swa_pages = num_slots * self.window_size
        elem_bf16 = 2
        elem_fp32 = 4

        block_regions: list[KVTransferRegion] = []
        slot_regions: list[KVTransferRegion] = []

        # Block regions: compress tail per layer
        for layer_id in range(self.num_layers):
            uv = runner.v4_unified_kv[layer_id]
            compress_base = uv.data_ptr() + swa_pages * self.head_dim * elem_bf16
            compress_total = (
                uv.numel() * elem_bf16 - swa_pages * self.head_dim * elem_bf16
            )
            if compress_total <= 0:
                continue
            ratio = self.compress_ratios[layer_id]
            if ratio == 4:
                bpb = self.k1_csa * self.head_dim * elem_bf16
            elif ratio == 128:
                bpb = self.k2_hca * self.head_dim * elem_bf16
            else:
                continue
            block_regions.append(KVTransferRegion(compress_base, compress_total, bpb))

        # Block regions: CSA Indexer KV (FP8)
        for pos in range(len(self.csa_layers)):
            t = runner.v4_csa_idx_kv[pos]
            bpb = self.k1_csa * self._aligned_index_dim
            block_regions.append(
                KVTransferRegion(t.data_ptr(), t.numel() * t.element_size(), bpb)
            )

        # Slot regions: SWA per layer
        swa_slot_bytes = self.window_size * self.head_dim * elem_bf16
        for layer_id in range(self.num_layers):
            uv = runner.v4_unified_kv[layer_id]
            slot_regions.append(
                KVTransferRegion(
                    uv.data_ptr(),
                    swa_pages * self.head_dim * elem_bf16,
                    swa_slot_bytes,
                )
            )

        # Staging pool for compressor states (not in slot_regions — managed
        # separately by the connector with pool acquire/release).
        staging_region = None
        gather_slot = None
        scatter_slot = None
        if hasattr(runner, "v4_state_pool"):
            pool = runner.v4_state_pool
            stride = runner.v4_state_slot_stride
            pool_size = runner.v4_state_pool_size
            staging_region = KVTransferRegion(
                pool.data_ptr(),
                pool.numel() * elem_fp32,
                stride * elem_fp32,
            )
            state_tensors = [
                runner.v4_csa_main_kv_state,
                runner.v4_csa_main_score_state,
                runner.v4_csa_idx_kv_state,
                runner.v4_csa_idx_score_state,
                runner.v4_hca_main_kv_state,
                runner.v4_hca_main_score_state,
            ]
            gather_slot = self._make_gather_slot(pool, stride, state_tensors)
            scatter_slot = self._make_scatter_slot(pool, stride, state_tensors)

        return KVTransferTensors(
            block_regions=block_regions,
            slot_regions=slot_regions,
            num_blocks=runner.num_physical_kvcache_blocks,
            num_slots=num_slots,
            staging_region=staging_region,
            staging_pool_size=pool_size if staging_region else 0,
            gather_slot=gather_slot,
            scatter_slot=scatter_slot,
        )

    # ------------------------------------------------------------------ #
    # CommonAttentionBuilder abstract methods (V4 forward consumes only  #
    # `positions`; other metadata is populated for forward parity with   #
    # the rest of ATOM and to support PR3-main multi-sequence wiring).   #
    # ------------------------------------------------------------------ #

    def _attach_v4_indexer_meta(
        self,
        attn_metadata: AttentionMetaData_DSV4,
        cu_seqlens_q_np,
        start_pos_per_seq,
        scheduled_bs: int,
        total_tokens: int,
        positions_gpu=None,
    ) -> None:
        """Build and attach the CSA Indexer per-fwd GPU metadata.

        Hoists per-CSA-layer H2D calls (batch_id_per_token / cu_committed /
        n_committed / seq_base_per_token / cu_ends) into a single per-fwd
        build. None for warmup or empty fwd; `_build_v4_indexer_meta`
        handles both.
        """
        import numpy as np

        attn_metadata.indexer_meta = self._build_v4_indexer_meta(
            attn_metadata=attn_metadata,
            cu_seqlens_q_np=cu_seqlens_q_np,
            start_pos_per_seq_np=np.asarray(start_pos_per_seq, dtype=np.int64),
            positions_gpu=positions_gpu,
            scheduled_bs=scheduled_bs,
            total_tokens=total_tokens,
            device=self.device,
        )

    def _build_v4_indexer_meta(
        self,
        *,
        attn_metadata: AttentionMetaData_DSV4,
        cu_seqlens_q_np,
        start_pos_per_seq_np,
        positions_gpu,
        scheduled_bs: int,
        total_tokens: int,
        device,
    ):
        """Build per-fwd GPU index tensors consumed by `Indexer.forward_batched`.

        Returns None for warmup batches (the indexer falls back to its
        inline H2D path) or when CSA / Indexer is not on the model. CSA
        ratio is fixed at 4; we always build under that assumption.

        Reuses two shared GPU tensors set by `_attach_v4_per_fwd_meta`
        (which MUST be called before this helper):
          - `attn_metadata.batch_id_per_token`        [padded_T] int32
          - `attn_metadata.n_committed_csa_per_seq`   [bs] int32
        These were previously re-staged here as `v4_indexer_batch_id_per_token`
        (int64) and `v4_indexer_n_committed_per_seq` (int64) — one extra H2D
        each per fwd. Now we read int32 → cast to int64 on GPU.

        The FP8 indexer K-cache write happens inside `fused_compress_attn`
        (the unified Indexer-inner Compressor path) via the same block_tables
        that CSA Main uses; no separate slot_mapping is built here.
        """

        # Caller contract: scheduled_bs >= 1, total_tokens >= 1 (same
        # invariants as `_attach_v4_per_fwd_meta` — guaranteed by every
        # prepare_*/CG-capture path).
        bs = scheduled_bs
        ratio = 4  # CSA
        cu_seqlens_q_arr = cu_seqlens_q_np[: bs + 1].astype(np.int64)
        token_num_per_seq = (cu_seqlens_q_arr[1:] - cu_seqlens_q_arr[:bs]).astype(
            np.int64
        )
        start_pos_per_seq = start_pos_per_seq_np[:bs].astype(np.int64)
        end_pos_per_seq = start_pos_per_seq + token_num_per_seq
        n_committed_per_seq = end_pos_per_seq // ratio  # host copy for cumsum
        cu_committed_cpu = np.concatenate(
            [
                np.zeros(1, dtype=np.int64),
                np.cumsum(n_committed_per_seq, dtype=np.int64),
            ]
        )
        # Empty-batch guard: when no seq has committed K yet
        # (`cu_committed_cpu[-1] == 0`, e.g. fresh prefill with prompt
        # shorter than the CSA `ratio`), `cp_gather_indexer_k_quant_cache`
        # would launch with grid.x = 0 and fail with HIP "invalid
        # configuration argument". Bump the last cumsum by one so the
        # kernel sees a single dummy row to gather (charged to the last
        # seq's first cache block). Downstream readers
        # (`fp8_mqa_logits` + `top_k_per_row_prefill`) honor per-token
        # `cu_starts`/`cu_ends` derived from `cu_committed_gpu[:-1]` and
        # `n_committed_per_seq`, both of which remain 0 — so the dummy
        # row is never read and the output is `-1` sentinels everywhere,
        # matching the all-empty semantics. Pure host-side scalar
        # arithmetic on a value already host-synced two lines up; no new
        # CG/torch.compile graph branch is introduced.
        cu_committed_cpu[-1] = max(int(cu_committed_cpu[-1]), 1)
        total_committed = int(cu_committed_cpu[-1])

        # batch_id_per_token + n_committed_csa: reuse the shared GPU
        # tensors set in `_attach_v4_per_fwd_meta` (which MUST run before
        # this helper — see prepare_decode/prefill ordering). int64
        # batch_id is mandated by PyTorch fancy indexing; int32 n_committed
        # is the gather SOURCE so any dtype works (and both downstream
        # kernels — `deepgemm_fp8_paged_mqa_logits`, `top_k_per_row_decode`
        # — want int32 anyway).
        batch_id_per_token_gpu = attn_metadata.batch_id_per_token[:total_tokens]
        n_committed_per_seq_gpu = attn_metadata.n_committed_csa_per_seq
        # cu_committed_gpu is consumed both as `cu_starts/cu_ends` for the
        # fp8_mqa_logits per-token range AND as `cu_seq_lens` for the
        # cp_gather_indexer_k_quant_cache call (per-seq cumulative committed K).
        cu_committed_gpu = self._stage(
            "v4_indexer_cu_committed", cu_committed_cpu.astype(np.int32)
        )

        # Layer-invariant GPU derivations (each was previously rebuilt ~30x
        # per fwd inside the per-CSA-layer body).
        seq_base_per_token_gpu = cu_committed_gpu[batch_id_per_token_gpu].to(
            torch.int32
        )  # [total_tokens] int32 — per-token offset into concat'd seqs'
        # compressed K. Used as `cu_starts` for fp8_mqa_logits AND as the
        # subtraction base for prefill `top_k_per_row_prefill`'s GLOBAL output
        # → seq-local conversion (the indexer kernel writes
        # `seq_base + col_in_seq`; we recover col_in_seq by subtracting).
        visible_end_gpu = torch.minimum(
            (positions_gpu[:total_tokens] + 1) // ratio,
            n_committed_per_seq_gpu[batch_id_per_token_gpu],
        ).to(
            torch.int32
        )  # [total_tokens] int32 — per-token causal upper bound
        cu_ends_gpu = (
            seq_base_per_token_gpu + visible_end_gpu
        )  # [total_tokens] int32 — fp8_mqa_logits per-token end offset

        return {
            "total_committed": total_committed,
            "cu_committed_gpu": cu_committed_gpu,
            "n_committed_per_seq_gpu": n_committed_per_seq_gpu,  # int32, [bs]
            "batch_id_per_token_gpu": batch_id_per_token_gpu,  # int64, [total_tokens]
            # Prefill-only fields below — decode never consults them. NOT
            # in pre-allocated buffers (per-fwd derived); CG capture path
            # would see stale pointers, but the decode path doesn't touch
            # them, so it's fine.
            "seq_base_per_token_gpu": seq_base_per_token_gpu,
            "cu_starts_gpu": seq_base_per_token_gpu,  # alias for fp8_mqa_logits
            "cu_ends_gpu": cu_ends_gpu,
        }

    def prepare_decode(self, batch: ScheduledBatch, bs: int):
        """V4-style decode prep: populates positions, cu_seqlens_q,
        block_tables, and state_slot_mapping.

        Uses stream overlap (like AiterMLAMetadataBuilder) to hide H2D
        latency behind CPU numpy work: basic H2D copies fire on
        ``prep_stream`` while ``_build_compress_plans`` runs on the CPU.
        """
        import numpy as np

        var = self.model_runner.forward_vars
        scheduled_bs = batch.total_seqs_num_decode
        context_lens_np = np.asarray(batch.context_lens, dtype=np.int32)
        max_seqlen_q = batch.num_spec_step + 1
        positions_np = np.tile(
            np.arange(max_seqlen_q, dtype=np.int32), scheduled_bs
        ) + np.repeat(context_lens_np - max_seqlen_q, max_seqlen_q)
        sum_scheduled_tokens = batch.total_tokens_num_decode

        var["positions"].np[:sum_scheduled_tokens] = positions_np

        cu_seqlens_q_np = np.arange(
            0, (scheduled_bs + 1) * max_seqlen_q, max_seqlen_q, dtype=np.int32
        )
        var["cu_seqlens_q"].np[: scheduled_bs + 1] = cu_seqlens_q_np

        var["context_lens"].np[:scheduled_bs] = context_lens_np

        # Inline block_tables CPU fill (H2D deferred to prep_stream).
        block_tables_np = var["block_tables"].np
        for i, block_table in enumerate(batch.block_tables[:scheduled_bs]):
            block_tables_np[i] = 0
            block_tables_np[i, : len(block_table)] = block_table

        state_slot_np = np.asarray(
            batch.per_req_cache_groups[:scheduled_bs], dtype=np.int32
        )
        if len(state_slot_np) < scheduled_bs:
            state_slot_np = np.zeros(scheduled_bs, dtype=np.int32)
        ss_buf = var["v4_meta_state_slot_groups"]
        ss_buf.np[:scheduled_bs] = state_slot_np

        # ---- fire H2D on prep_stream ----
        prep_stream = self.prep_stream
        current_stream = torch.cuda.current_stream()
        prep_stream.wait_stream(current_stream)
        with torch.cuda.stream(prep_stream):
            positions = var["positions"].copy_to_gpu(sum_scheduled_tokens)
            cu_seqlens_q_gpu = var["cu_seqlens_q"].copy_to_gpu(scheduled_bs + 1)
            context_lens_gpu = var["context_lens"].copy_to_gpu(scheduled_bs)
            block_tables_gpu = var["block_tables"].copy_to_gpu(scheduled_bs)
            state_slot_gpu = ss_buf.copy_to_gpu(scheduled_bs)

        # ---- CPU numpy work, overlapped with prep_stream H2D ----
        start_pos_per_seq_cpu = positions_np[cu_seqlens_q_np[:scheduled_bs]]

        extend_lens_np = np.full(scheduled_bs, max_seqlen_q, dtype=np.int32)
        compress_plans = self._build_compress_plans(
            extend_lens_np,
            context_lens_np,
            torch.device(self.device),
            for_decode_cg=True,
        )

        # ---- sync, build attn_metadata, per-fwd meta ----
        current_stream.wait_stream(prep_stream)

        attn_metadata = AttentionMetaData_DSV4(
            cu_seqlens_q=cu_seqlens_q_gpu,
            cu_seqlens_k=None,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=int(context_lens_np.max()) if len(context_lens_np) else 1,
            min_seqlen_q=0,
            dropout_p=0.0,
            has_cached=False,
            total_kv=int(context_lens_np.sum()),
            num_cached_tokens=None,
            block_tables=block_tables_gpu,
            context_lens=context_lens_gpu,
        )
        attn_metadata.state_slot_mapping = state_slot_gpu
        attn_metadata.state_slot_mapping_cpu = state_slot_np
        attn_metadata.start_pos_per_seq_cpu = start_pos_per_seq_cpu
        attn_metadata.compress_plans = compress_plans

        padded_bs = int(bs)
        self._attach_v4_per_fwd_meta(
            attn_metadata,
            positions_np,
            cu_seqlens_q_np,
            start_pos_per_seq_cpu,
            state_slot_np,
            scheduled_bs,
            sum_scheduled_tokens,
            padded_bs=padded_bs,
            max_q_len=max_seqlen_q,
        )
        self._attach_v4_indexer_meta(
            attn_metadata,
            cu_seqlens_q_np,
            start_pos_per_seq_cpu,
            scheduled_bs,
            sum_scheduled_tokens,
            positions_gpu=positions,
        )
        return attn_metadata, positions

    def prepare_prefill(self, batch: ScheduledBatch):
        """V4 prefill prep: extends parent to always populate block_tables
        and state_slot_mapping.

        The parent only emits block_tables when has_cached (prefix cache hit);
        V4 always needs block_tables because Compressor scatters compressed
        entries into the classical KV pool from token 0 onwards.

        Also publishes CPU mirrors (`v4_*_cpu`) consumed by the V4 forward
        path to avoid `.item()` / `.tolist()` syncs (PR-A Phase 2).
        """
        import numpy as np

        base_md, positions = super().prepare_prefill(batch)
        # Promote to V4 typed metadata so V4-specific attribute assignments
        # below are well-typed. Safe because AttentionMetaData_DSV4 only adds
        # fields with defaults; the parent dataclass is non-slotted.
        base_md.__class__ = AttentionMetaData_DSV4
        attn_metadata = cast(AttentionMetaData_DSV4, base_md)
        scheduled_bs = batch.total_seqs_num_prefill
        if attn_metadata.block_tables is None:
            attn_metadata.block_tables = self._populate_block_tables(
                batch, scheduled_bs
            )
        state_slot_gpu, state_slot_np = self._populate_state_slot_mapping(
            batch, scheduled_bs, return_cpu=True
        )
        attn_metadata.state_slot_mapping = state_slot_gpu
        # PR-A Phase 2 CPU mirrors (generic, not V4-specific). The parent
        # populated forward_vars CPU buffers; read them back as numpy slices.
        var = self.model_runner.forward_vars
        sum_scheduled_tokens = batch.total_tokens_num_prefill
        positions_np = np.asarray(var["positions"].np[:sum_scheduled_tokens])
        cu_seqlens_q_np = np.asarray(var["cu_seqlens_q"].np[: scheduled_bs + 1])
        attn_metadata.state_slot_mapping_cpu = state_slot_np
        attn_metadata.start_pos_per_seq_cpu = positions_np[
            cu_seqlens_q_np[:scheduled_bs]
        ]
        # Compress plans (per ratio) for batched fused_compress + update_states.
        # Prefill batch: extend_lens read from cu_seqlens_q_np.
        # Must run BEFORE `_attach_v4_indexer_meta` (the indexer consumes
        # plan.compress_plan_cpu to derive its FP8 write-side slot_mapping).
        extend_lens_np = (
            cu_seqlens_q_np[1 : scheduled_bs + 1] - cu_seqlens_q_np[:scheduled_bs]
        ).astype(np.int32)
        # context_lens for prefill = positions[seq_start_in] + extend_lens
        # (= absolute seq_len incl. this fwd's tokens).
        context_lens_np = (attn_metadata.start_pos_per_seq_cpu + extend_lens_np).astype(
            np.int32
        )
        attn_metadata.compress_plans = self._build_compress_plans(
            extend_lens_np, context_lens_np, positions.device, for_decode_cg=False
        )
        # Prefill goes through eager (no CG): defaults make padded_total_tokens
        # collapse to total_tokens — no padding logic kicks in. Must still run
        # BEFORE `_attach_v4_indexer_meta` so the indexer-side meta builder can
        # reuse the shared GPU tensors (batch_id_per_token, n_committed_csa).
        self._attach_v4_per_fwd_meta(
            attn_metadata,
            positions_np,
            cu_seqlens_q_np,
            attn_metadata.start_pos_per_seq_cpu,
            attn_metadata.state_slot_mapping_cpu,
            scheduled_bs,
            sum_scheduled_tokens,
        )
        self._attach_v4_indexer_meta(
            attn_metadata,
            cu_seqlens_q_np,
            attn_metadata.start_pos_per_seq_cpu,
            scheduled_bs,
            sum_scheduled_tokens,
            positions_gpu=positions,
        )
        # Two-source paged_prefill index buffers (extend + per-ratio prefix).
        # Eager-only — direct H2D, no forward_vars staging required. Sets
        # attn_metadata.{kv_indices,kv_indptr}_{extend,prefix_swa,prefix_csa,prefix_hca}
        # plus skip_prefix_len_csa and swa_pages.
        self._build_paged_prefill_meta(
            attn_metadata,
            positions_np,
            cu_seqlens_q_np,
            attn_metadata.start_pos_per_seq_cpu,
            attn_metadata.state_slot_mapping_cpu,
            scheduled_bs,
            sum_scheduled_tokens,
        )
        return attn_metadata, positions

    def build_ubatch_prefill_metadata(
        self,
        attn_metadata: AttentionMetaData,
        ub_slice,
        padded_bs: int,
    ) -> AttentionMetaData_DSV4:
        """Split prefill AttentionMetaData for V4 TBO micro-batches."""
        from atom.utils.tbo.ubatch_splitting import split_attn_metadata

        ub_attn = split_attn_metadata(attn_metadata, ub_slice, padded_bs)
        ub_attn.__class__ = AttentionMetaData_DSV4

        src = cast(AttentionMetaData_DSV4, attn_metadata)
        rs = ub_slice.request_slice
        ts = ub_slice.token_slice
        ub_num_reqs = rs.stop - rs.start
        ub_num_tokens = ts.stop - ts.start

        if src.state_slot_mapping is not None:
            ub_attn.state_slot_mapping = src.state_slot_mapping[rs]
        if src.state_slot_mapping_cpu is not None:
            ub_attn.state_slot_mapping_cpu = src.state_slot_mapping_cpu[rs]
        if src.start_pos_per_seq_cpu is not None:
            ub_attn.start_pos_per_seq_cpu = src.start_pos_per_seq_cpu[rs]

        var = self.model_runner.forward_vars
        positions_np = np.asarray(var["positions"].np[ts.start : ts.stop])
        full_cu = var["cu_seqlens_q"].np
        ub_cu = np.asarray(full_cu[rs.start : rs.stop + 1], dtype=np.int32).copy()
        ub_cu -= ub_cu[0]

        extend_lens_np = (ub_cu[1:] - ub_cu[:ub_num_reqs]).astype(np.int32)
        context_lens_np = (ub_attn.start_pos_per_seq_cpu + extend_lens_np).astype(
            np.int32
        )
        device = src.state_slot_mapping.device

        from atom.model_ops.v4_kernels import make_compress_plans

        if self._unique_compress_ratios_overlap:
            ub_attn.compress_plans = make_compress_plans(
                np.ascontiguousarray(extend_lens_np, dtype=np.int32),
                np.ascontiguousarray(context_lens_np, dtype=np.int32),
                self._unique_compress_ratios_overlap,
                device,
                plan_buffers=None,
                decode_capacity_per_ratio=None,
            )
        else:
            ub_attn.compress_plans = {}

        # Multiple helpers read context_lens and block_tables from
        # forward_vars by position [0:scheduled_bs]. For ubatch 1 the
        # relevant rows live at [rs.start:rs.stop], not [0:ub_num_reqs].
        # Temporarily place the ubatch's slices at the front so helpers
        # see the right values.
        bt = var["block_tables"].np
        saved_ctx = var["context_lens"].np[:ub_num_reqs].copy()
        saved_bt = bt[:ub_num_reqs].copy()
        try:
            var["context_lens"].np[:ub_num_reqs] = context_lens_np
            bt[:ub_num_reqs] = bt[rs.start : rs.stop].copy()

            self._attach_v4_per_fwd_meta(
                ub_attn,
                positions_np,
                ub_cu,
                ub_attn.start_pos_per_seq_cpu,
                ub_attn.state_slot_mapping_cpu,
                ub_num_reqs,
                ub_num_tokens,
            )

            positions_gpu = var["positions"].gpu[ts.start : ts.stop]
            self._attach_v4_indexer_meta(
                ub_attn,
                ub_cu,
                ub_attn.start_pos_per_seq_cpu,
                ub_num_reqs,
                ub_num_tokens,
                positions_gpu=positions_gpu,
            )

            self._build_paged_prefill_meta(
                ub_attn,
                positions_np,
                ub_cu,
                ub_attn.start_pos_per_seq_cpu,
                ub_attn.state_slot_mapping_cpu,
                ub_num_reqs,
                ub_num_tokens,
            )
        finally:
            bt[:ub_num_reqs] = saved_bt
            var["context_lens"].np[:ub_num_reqs] = saved_ctx

        # Clone all GPU tensors that are views into shared CpuGpuBuffers.
        # Without this, building the next ubatch overwrites this ubatch's
        # data via the same underlying buffer.
        if ub_attn.batch_id_per_token is not None:
            ub_attn.batch_id_per_token = ub_attn.batch_id_per_token.clone()
        if ub_attn.n_committed_csa_per_seq is not None:
            ub_attn.n_committed_csa_per_seq = ub_attn.n_committed_csa_per_seq.clone()
        if ub_attn.swa_write_indices is not None:
            ub_attn.swa_write_indices = ub_attn.swa_write_indices.clone()
        if ub_attn.indexer_meta is not None:
            im = ub_attn.indexer_meta
            if im.get("cu_committed_gpu") is not None:
                im["cu_committed_gpu"] = im["cu_committed_gpu"].clone()
            if im.get("batch_id_per_token_gpu") is not None:
                im["batch_id_per_token_gpu"] = im["batch_id_per_token_gpu"].clone()
            if im.get("n_committed_per_seq_gpu") is not None:
                im["n_committed_per_seq_gpu"] = im["n_committed_per_seq_gpu"].clone()

        return ub_attn

    def _attach_v4_per_fwd_meta(
        self,
        attn_metadata: AttentionMetaData_DSV4,
        positions_np: np.ndarray,
        cu_seqlens_q_np,
        start_pos_per_seq_cpu,
        state_slot_mapping_cpu,
        scheduled_bs: int,
        total_tokens: int,
        *,
        padded_bs: Optional[int] = None,
        max_q_len: Optional[int] = None,
    ) -> None:
        """Hoist per-fwd, layer-invariant metadata used by every V4 layer.

        These tensors only depend on `positions`, `cu_seqlens_q`, `state_slot_mapping`
        and `window_size` — none of which change across layers — so building
        them once per fwd saves ~64 redundant constructions for V4-Pro.

        Sets:
          - `attn_metadata.swa_write_indices`: [padded_T] int64 row ids
            into per-token KV. Real entries are the last `win` tokens per seq;
            tail (entries beyond the per-fwd `num_write`) is sentinel-filled
            with -1 so a fixed grid `padded_T` is always safe (CUDAGraph).
          - `attn_metadata.batch_id_per_token`: [padded_T] int32 batch id
            per token (single per-token mapping; consumed by `swa_write`,
            Phase B/C/E paged-decode kernels, and the indexer).
          - `attn_metadata.n_committed_csa_per_seq`: [bs] int32 per-seq
            `ctx_len // 4` (shared by csa_translate_pack + indexer; kernels
            do their own `min(., index_topk)` clamp via mask).
          - `attn_metadata.state_slot_mapping`: [bs] int32 GPU view of
            per-seq state cache slot (already set by prepare_*; passed
            through unchanged here).

        Caller contract: `scheduled_bs >= 1` and `total_tokens >= 1`.
        warmup_model + dummy_run paths both enforce these via min-1 fallbacks
        (model_runner.warmup_model:1003-1011, _populate_state_slot_mapping
        zeros-fill); CG capture uses graph_bs >= 1 too.
        """
        win = self.window_size

        cu_seqlens_q_arr = np.asarray(
            cu_seqlens_q_np[: scheduled_bs + 1], dtype=np.int64
        )
        token_num_per_seq = cu_seqlens_q_arr[1:] - cu_seqlens_q_arr[:scheduled_bs]
        start_pos_per_seq_np = np.asarray(
            start_pos_per_seq_cpu[:scheduled_bs], dtype=np.int64
        )
        if padded_bs is None or max_q_len is None:
            padded_total_tokens = total_tokens
        else:
            padded_total_tokens = max(int(padded_bs) * int(max_q_len), total_tokens)

        is_decode_only = scheduled_bs == total_tokens and bool(
            (token_num_per_seq == 1).all()
        )

        var = self.model_runner.forward_vars

        # ---- CPU numpy work (all on main thread) ----
        batch_id_per_token_np = np.full(padded_total_tokens, -1, dtype=np.int64)
        batch_id_per_token_np[:total_tokens] = np.repeat(
            np.arange(scheduled_bs, dtype=np.int64), token_num_per_seq
        )

        ctx_per_seq_np = np.asarray(
            var["context_lens"].np[:scheduled_bs],
            dtype=np.int64,
        )
        n_committed_csa_per_seq_np = (ctx_per_seq_np // 4).astype(np.int32)

        write_starts = cu_seqlens_q_arr[:scheduled_bs] + np.maximum(
            0, token_num_per_seq - win
        )
        write_ends = cu_seqlens_q_arr[1:]
        if is_decode_only:
            write_indices_np = np.arange(total_tokens, dtype=np.int64)
        else:
            write_indices_np = np.concatenate(
                [
                    np.arange(s, e, dtype=np.int64)
                    for s, e in zip(write_starts, write_ends)
                ]
            )
        num_write = int(write_indices_np.shape[0])
        wi_buf = var["v4_meta_swa_write_indices"]
        cap = wi_buf.np.shape[0]
        assert (
            padded_total_tokens <= cap
        ), f"v4_meta_swa_write_indices too small: need {padded_total_tokens}, have {cap}"
        if num_write > 0:
            wi_buf.np[:num_write] = write_indices_np
        if num_write < padded_total_tokens:
            wi_buf.np[num_write:padded_total_tokens].fill(-1)

        # ---- Build window_topk on CPU (avoids ~25 small GPU kernels) ----
        if is_decode_only:
            sp_per_token = start_pos_per_seq_np
        else:
            sp_per_token = np.repeat(start_pos_per_seq_np, token_num_per_seq)
        positions_long = positions_np[:total_tokens].astype(np.int64)
        window_topk_np = _build_window_topk_np(positions_long, sp_per_token, win)
        wt_buf = var["v4_meta_window_topk"]
        wt_buf.np[:total_tokens, :win] = window_topk_np

        # ---- Stage all buffers to GPU ----
        window_topk_gpu = wt_buf.copy_to_gpu(total_tokens)

        attn_metadata.batch_id_per_token = self._stage(
            "v4_batch_id_per_token", batch_id_per_token_np
        )
        attn_metadata.n_committed_csa_per_seq = self._stage(
            "v4_n_committed_csa_per_seq", n_committed_csa_per_seq_np
        )
        attn_metadata.swa_write_indices = wi_buf.copy_to_gpu(padded_total_tokens)

        self._attach_v4_paged_decode_meta(
            attn_metadata=attn_metadata,
            token_num_per_seq=token_num_per_seq,
            start_pos_per_seq_np=start_pos_per_seq_np,
            state_slot_mapping_cpu=state_slot_mapping_cpu,
            window_topk_gpu=window_topk_gpu,
            scheduled_bs=scheduled_bs,
            total_tokens=total_tokens,
            padded_total_tokens=padded_total_tokens,
        )

    def _clear_v4_paged_decode_meta(self, attn_metadata) -> None:
        """Set all Phase-B paged-decode fields to default (non-decode batches).

        Note: `batch_id_per_token` and `n_committed_csa_per_seq` are
        intentionally NOT cleared here — both are also consumed by
        `swa_write` / the indexer on prefill / mixed paths and are built
        unconditionally in `_attach_v4_per_fwd_meta`.
        """
        attn_metadata.is_pure_decode = False
        attn_metadata.swa_pages = 0
        for k in (
            "kv_indices_swa",
            "kv_indices_csa",
            "kv_indices_hca",
            "kv_indptr_swa",
            "kv_indptr_csa",
            "kv_indptr_hca",
        ):
            setattr(attn_metadata, k, None)

    def _attach_v4_paged_decode_meta(
        self,
        attn_metadata,
        token_num_per_seq,
        start_pos_per_seq_np,
        state_slot_mapping_cpu,
        window_topk_gpu,
        scheduled_bs: int,
        total_tokens: int,
        padded_total_tokens: Optional[int] = None,
    ) -> None:
        """Phase B: build per-fwd paged-decode index buffers (layer-invariant).

        Writes into stable forward_vars buffers (attn_metadata fields are
        the V4-namespaced counterparts on `AttentionMetaData_DSV4`):
          - kv_indices_swa : SWA paged offsets, uniform stride win
          - kv_indices_csa : SWA prefix written here; CSA compress section
                             left UNINITIALIZED — V4Attention.forward fills
                             it per-layer via csa_translate_pack (Phase C)
          - kv_indices_hca : SWA prefix + HCA compress section, both fully
                             written (HCA is layer-invariant)
          - kv_indptr_{swa,csa,hca} : 3 indptr cumsums (SWA uniform, CSA /
                             HCA packed; per doc §6.3). Padded tail repeats
                             last value → kv_len=0 sentinel for CG-padded slots.

        Reuses (built earlier in `_attach_v4_per_fwd_meta`):
          - batch_id_per_token : single per-token mapping (with -1 sentinel)
          - n_committed_csa_per_seq : per-seq `ctx_len // 4`

        Skipped (defaults via `_clear_v4_paged_decode_meta`) when not
        is_pure_decode (prefill, mixed, fresh-prefill).
        """
        if scheduled_bs == 0 or total_tokens == 0:
            self._clear_v4_paged_decode_meta(attn_metadata)
            return

        # is_pure_decode = uniform tokens-per-seq AND no fresh prefill
        # (per doc §7.4). Catches both regular decode (1 tok/seq) and MTP-1
        # (2 tok/seq); rejects mixed batches and fresh-prefill seqs.
        tok_first = int(token_num_per_seq[0])
        is_uniform_tok = bool((token_num_per_seq == tok_first).all())
        no_fresh_prefill = bool((start_pos_per_seq_np > 0).all())
        is_pure_decode = is_uniform_tok and no_fresh_prefill

        if not is_pure_decode or len(state_slot_mapping_cpu) < scheduled_bs:
            self._clear_v4_paged_decode_meta(attn_metadata)
            return

        var = self.model_runner.forward_vars
        win = self.window_size
        # swa_pages = num_slots * win, layer-invariant; matches the boundary
        # between SWA and compress regions in unified_kv (Phase A).
        swa_pages = self.model_runner.max_per_req_cache_slots * win

        T = total_tokens

        # ----- Per-seq scalars (CPU numpy) -----
        # context_lens / block_tables CPU mirrors are populated by the caller
        # (prepare_decode / build_for_cudagraph_capture) BEFORE this runs.
        ctx_per_seq = np.asarray(var["context_lens"].np[:scheduled_bs], dtype=np.int64)
        # The single per-token mapping. Built once in `_attach_v4_per_fwd_meta`
        # (so swa_write / indexer can also consume it). Pull the GPU view from
        # attn_metadata; recompute the np copy here only for cumsum math below.
        batch_id_per_token_np = np.repeat(
            np.arange(scheduled_bs, dtype=np.int32), token_num_per_seq
        )  # [T] int32 — host copy for indptr cumsums
        batch_id_per_token_gpu = attn_metadata.batch_id_per_token

        n_committed_csa_per_seq = ctx_per_seq // 4  # host copy for cumsum below
        n_committed_hca_per_seq = (ctx_per_seq // 128).astype(np.int64)

        # ----- 3 indptr cumsums (CPU numpy, per doc §6.3) -----
        # Per-token kv_len = f(per-seq quantity)[batch_id_per_token]. CSA
        # length is clamped to index_topk because csa_translate_pack only
        # writes that many rows per seq (kernel mask `(k < n_committed) &
        # (k < index_topk)`); the host-side clamp here just keeps the indices
        # buffer correctly sized — the staged GPU n_committed_csa stays raw.
        index_topk = self.index_topk
        n_committed_csa_clamped_per_token = np.minimum(
            n_committed_csa_per_seq[batch_id_per_token_np], index_topk
        )
        n_committed_hca_per_token = n_committed_hca_per_seq[batch_id_per_token_np]

        # CG-padding-aware T_for_indptr: indptr buffer must size to the
        # captured kernel grid (= padded_total_tokens) so padded slots see
        # `kv_len = indptr[t+1] - indptr[t] = 0` and the inner loop bails.
        T_pad = (
            total_tokens if padded_total_tokens is None else int(padded_total_tokens)
        )
        if T_pad < T:
            T_pad = T

        # SWA: uniform stride for real [0:T+1]. Padded entries [T+1:T_pad+1]
        # repeat the last value (T*win) → kv_len = 0 for padded tokens, kernel
        # bails out of the inner loop without reading kv_indices.
        swa_indptr_np = np.empty(T_pad + 1, dtype=np.int32)
        swa_indptr_np[: T + 1] = np.arange(T + 1, dtype=np.int64) * win
        if T_pad > T:
            swa_indptr_np[T + 1 :].fill(int(T * win))
        # CSA: packed, per-token len = win + min(n_committed_csa, index_topk)
        csa_per_tok = win + n_committed_csa_clamped_per_token.astype(np.int64)
        csa_indptr_np = np.zeros(T_pad + 1, dtype=np.int32)
        csa_indptr_np[1 : T + 1] = np.cumsum(csa_per_tok).astype(np.int32)
        if T_pad > T:
            csa_indptr_np[T + 1 :].fill(int(csa_indptr_np[T]))
        # HCA: packed, per-token len = win + n_committed_hca
        hca_per_tok = win + n_committed_hca_per_token
        hca_indptr_np = np.zeros(T_pad + 1, dtype=np.int32)
        hca_indptr_np[1 : T + 1] = np.cumsum(hca_per_tok).astype(np.int32)
        if T_pad > T:
            hca_indptr_np[T + 1 :].fill(int(hca_indptr_np[T]))

        swa_indptr_gpu = self._stage("v4_kv_indptr_swa", swa_indptr_np)
        csa_indptr_gpu = self._stage("v4_kv_indptr_csa", csa_indptr_np)
        hca_indptr_gpu = self._stage("v4_kv_indptr_hca", hca_indptr_np)
        # batch_id_per_token + n_committed_csa_per_seq already staged in
        # `_attach_v4_per_fwd_meta`.

        # ----- HCA compress paged offsets (CPU numpy, vectorized) -----
        block_tables_np_full = var["block_tables"].np[:scheduled_bs]
        hca_total_indices = int(hca_indptr_np[T])
        hca_indices_np = np.full(hca_total_indices, -1, dtype=np.int32)
        n_h_per_token = n_committed_hca_per_seq[batch_id_per_token_np[:T]].astype(
            np.int64
        )
        total_hca_entries = int(n_h_per_token.sum())
        if total_hca_entries > 0:
            token_indices = np.repeat(np.arange(T, dtype=np.int32), n_h_per_token)
            cu_n_h = np.zeros(T + 1, dtype=np.int64)
            np.cumsum(n_h_per_token, out=cu_n_h[1:])
            entry_offsets = np.arange(total_hca_entries, dtype=np.int64) - np.repeat(
                cu_n_h[:T], n_h_per_token
            )
            write_pos = (
                hca_indptr_np[token_indices].astype(np.int64) + win + entry_offsets
            )
            bid_expanded = batch_id_per_token_np[token_indices]
            hca_indices_np[write_pos] = (
                swa_pages
                + block_tables_np_full[bid_expanded, entry_offsets.astype(np.intp)]
            ).astype(np.int32)
        # Stage to GPU (HCA compress tail; window prefix scattered below).
        hca_indices_gpu = self._stage("v4_kv_indices_hca", hca_indices_np)

        # ----- SWA paged offsets (GPU-side from window_topk_gpu) -----
        # `state_slot_per_token = state_slot_per_seq[batch_id_per_token]` —
        # done as a transient GPU gather (no persistent per-token buffer).
        # Reuse `attn_metadata.state_slot_mapping` (= forward_vars[
        # "v4_meta_state_slot_groups"] gpu view, set by `_populate_state_slot_mapping`)
        # — same data as the legacy `v4_meta_state_slot_i32` re-staging, no
        # second H2D needed.
        device = self.device
        state_slot_per_seq_gpu = attn_metadata.state_slot_mapping
        # Use only the real-data prefix of batch_id_per_token here — the
        # padded tail carries -1 sentinels, and `swa_paged_2d` is computed
        # against `window_topk_gpu` of shape [T, win] only.
        state_slot_per_token_gpu = state_slot_per_seq_gpu[
            batch_id_per_token_gpu[:T].long()
        ]  # [T] int32 transient
        # `window_topk_gpu` [T, win] int32 — passed in from caller
        # (`_attach_v4_per_fwd_meta` builds it once per fwd). Builder-internal
        # intermediate; not exposed on attn_metadata since no kernel reads it.
        swa_paged_2d = torch.where(
            window_topk_gpu >= 0,
            state_slot_per_token_gpu[:, None] * win + window_topk_gpu,
            torch.full_like(window_topk_gpu, -1),
        )  # [T, win] int32
        swa_paged_flat = swa_paged_2d.reshape(-1)  # [T*win] int32

        # ----- Write SWA paged offsets to all 3 buffer heads (GPU) -----
        # SWA buffer: uniform stride win → simple flat copy.
        swa_indices_gpu = var["v4_kv_indices_swa"].gpu
        swa_indices_gpu[: T * win].copy_(swa_paged_flat)

        # CSA / HCA buffers: window prefix at packed positions
        # [indptr[t], indptr[t]+win) — scatter via index_copy_.
        win_arange = torch.arange(win, device=device, dtype=torch.int64)
        csa_win_pos = (
            csa_indptr_gpu[:T].to(torch.int64).unsqueeze(1) + win_arange.unsqueeze(0)
        ).reshape(-1)
        hca_win_pos = (
            hca_indptr_gpu[:T].to(torch.int64).unsqueeze(1) + win_arange.unsqueeze(0)
        ).reshape(-1)
        csa_indices_gpu = var["v4_kv_indices_csa"].gpu
        csa_indices_gpu.index_copy_(0, csa_win_pos, swa_paged_flat)
        hca_indices_gpu.index_copy_(0, hca_win_pos, swa_paged_flat)

        # ----- skip_prefix_len_csa: decode = full SWA window per token -----
        # csa_translate_pack consumes this to know where the CSA section
        # starts within each per-token region. Decode path fills [0:T] with
        # `win`; padded tail [T:padded_T] = 0 (kernel bails on bid<0).
        # Pre-allocated buffer gives a stable address for CG capture.
        skip_csa_gpu = var["v4_skip_prefix_len_csa"].gpu
        skip_csa_gpu[:T].fill_(win)
        skip_csa_gpu[T:].zero_()

        # ----- Stash on attn_metadata for V4Attention.forward consumption -----
        # batch_id_per_token + n_committed_csa_per_seq already set in
        # `_attach_v4_per_fwd_meta` (single source of truth, also consumed by
        # swa_write / indexer outside the is_pure_decode branch).
        attn_metadata.is_pure_decode = True
        attn_metadata.kv_indices_swa = swa_indices_gpu[: T * win]
        attn_metadata.kv_indices_csa = csa_indices_gpu[: int(csa_indptr_np[T])]
        attn_metadata.kv_indices_hca = hca_indices_gpu  # already exact len
        attn_metadata.kv_indptr_swa = swa_indptr_gpu
        attn_metadata.kv_indptr_csa = csa_indptr_gpu
        attn_metadata.kv_indptr_hca = hca_indptr_gpu
        attn_metadata.skip_prefix_len_csa = skip_csa_gpu
        attn_metadata.swa_pages = swa_pages

    def _build_paged_prefill_meta(
        self,
        attn_metadata: AttentionMetaData_DSV4,
        positions_np: np.ndarray,
        cu_seqlens_q_np: np.ndarray,
        start_pos_per_seq_np: np.ndarray,
        state_slot_mapping_cpu: np.ndarray,
        scheduled_bs: int,
        total_tokens: int,
    ) -> None:
        """Build per-fwd index buffers consumed by sparse_attn_v4_paged_prefill.

        Two-source layout:
          - prefix region (per-ratio): SWA history from prior chunks +
            CSA topk OR HCA all-committed from `unified_kv`. Three buffers
            (Dense / CSA / HCA) per fwd.
          - extend region (shared): in-chunk SWA tail from per-fwd `kv`
            tensor. One buffer.

        Per-token length formulas:
          extend_count[t]      = min(token_pos_in_chunk[t] + 1, win)
          prefix_swa_count[t]  = max(0, chunk_start[bid] - max(0, p_global - win + 1))
          prefix_swa_count[t] + extend_count[t] = min(p_global + 1, win)

        Per-ratio prefix kv_len:
          Dense:  prefix_swa_count[t]
          CSA:    prefix_swa_count[t] + min(n_committed_csa[bid], index_topk)
          HCA:    prefix_swa_count[t] + n_committed_hca[bid]

        Eager-only (chunked prefill is dynamic-shaped; no CG capture). Per-fwd
        `torch.from_numpy(...).to(device)` is fine — no forward_vars staging.

        Builder fills: extend buffer, prefix_swa buffer (Dense), HCA section
        of prefix_hca buffer, SWA prefix sections of all 3 prefix buffers.
        Per-layer csa_translate_pack later fills the CSA section of
        prefix_csa buffer.

        Sets attn_metadata fields (per `AttentionMetaData_DSV4` docstrings):
          - kv_indices_extend / kv_indptr_extend (shared)
          - kv_indices_prefix_swa / kv_indptr_prefix_swa  (Dense)
          - kv_indices_prefix_csa / kv_indptr_prefix_csa  (CSA, CSA section UNINIT)
          - kv_indices_prefix_hca / kv_indptr_prefix_hca  (HCA, fully filled)
          - skip_prefix_len_csa = prefix_swa_count_per_token (per-token)
          - swa_pages
        """
        if scheduled_bs == 0 or total_tokens == 0:
            return

        device = self.device
        win = self.window_size
        index_topk = self.index_topk
        T = total_tokens
        # warmup_model runs BEFORE allocate_kv_cache binds the paged pool
        # (max_per_req_cache_slots not set yet, unified_kv is a 1-page
        # placeholder). V4Attention.forward detects `is_dummy_run` and
        # short-circuits the sparse_attn dispatch entirely, so we don't need
        # valid prefix/extend indices during warmup. Skip the CPU work too.
        num_slots = getattr(self.model_runner, "max_per_req_cache_slots", 0)
        if num_slots == 0:
            return
        swa_pages = num_slots * win

        # ----- Per-token quantities (CPU numpy) -----
        cu_seqlens_q_arr = np.asarray(
            cu_seqlens_q_np[: scheduled_bs + 1], dtype=np.int64
        )
        token_num_per_seq = (
            cu_seqlens_q_arr[1:] - cu_seqlens_q_arr[:scheduled_bs]
        ).astype(np.int64)
        chunk_start_per_seq = np.asarray(
            start_pos_per_seq_np[:scheduled_bs], dtype=np.int64
        )
        # batch_id_per_token_np mirrors what _attach_v4_per_fwd_meta computed
        # but we need a CPU copy for the cumsum / segment math below.
        batch_id_per_token_np = np.repeat(
            np.arange(scheduled_bs, dtype=np.int64), token_num_per_seq
        )  # [T] int64
        positions_arr = np.asarray(positions_np[:T], dtype=np.int64)
        token_pos_in_chunk = (
            positions_arr - chunk_start_per_seq[batch_id_per_token_np]
        )  # [T] int64
        # SWA window low bound (clamped at 0); used to derive prefix_swa_count
        # AND the absolute global positions inside the prefix SWA section.
        swa_window_low_global = np.maximum(0, positions_arr - win + 1)  # [T] int64

        extend_count_np = np.minimum(token_pos_in_chunk + 1, win).astype(np.int64)
        prefix_swa_count_np = np.maximum(
            0, chunk_start_per_seq[batch_id_per_token_np] - swa_window_low_global
        ).astype(np.int64)

        # n_committed_{csa,hca}[bid] from per-seq context_lens (already populated
        # by parent). HCA: ctx_len // 128, CSA already exposed as int32 GPU tensor;
        # rebuild host copy for cumsum below.
        var = self.model_runner.forward_vars
        ctx_per_seq_np = np.asarray(
            var["context_lens"].np[:scheduled_bs], dtype=np.int64
        )
        n_committed_csa_per_seq_np = ctx_per_seq_np // 4
        n_committed_hca_per_seq_np = ctx_per_seq_np // 128
        n_csa_per_token = np.minimum(
            n_committed_csa_per_seq_np[batch_id_per_token_np], index_topk
        )  # [T] int64 — clamped because csa_translate_pack writes at most
        #   index_topk per token (kernel mask `(k < n) & (k < index_topk)`)
        n_hca_per_token = n_committed_hca_per_seq_np[batch_id_per_token_np]  # [T] int64

        # ----- indptr cumsums (CPU) -----
        prefix_swa_indptr_np = np.zeros(T + 1, dtype=np.int64)
        prefix_swa_indptr_np[1:] = np.cumsum(prefix_swa_count_np)
        prefix_csa_indptr_np = np.zeros(T + 1, dtype=np.int64)
        prefix_csa_indptr_np[1:] = np.cumsum(prefix_swa_count_np + n_csa_per_token)
        prefix_hca_indptr_np = np.zeros(T + 1, dtype=np.int64)
        prefix_hca_indptr_np[1:] = np.cumsum(prefix_swa_count_np + n_hca_per_token)
        extend_indptr_np = np.zeros(T + 1, dtype=np.int64)
        extend_indptr_np[1:] = np.cumsum(extend_count_np)

        # ----- Extend kv_indices (in `kv` tensor): per-token rows -----
        # extend window for token t: kv rows
        # `[cu_seqlens_q[bid] + (token_pos_in_chunk[t] - extend_count[t] + 1
        #   ... cu_seqlens_q[bid] + token_pos_in_chunk[t]]`. Build via segment
        # expansion.
        ext_seg_tok, ext_seg_k = _segment_indices(
            np.arange(T, dtype=np.int64), extend_count_np
        )
        # Pre-gather per-token vars for the segment positions (vectorised).
        ext_cu_q = cu_seqlens_q_arr[:scheduled_bs][batch_id_per_token_np]  # [T] int64
        ext_indices_np = (
            ext_cu_q[ext_seg_tok]
            + (
                token_pos_in_chunk[ext_seg_tok]
                - extend_count_np[ext_seg_tok]
                + 1
                + ext_seg_k
            )
        ).astype(np.int32)

        # ----- Prefix SWA paged offsets -----
        # For each token's prefix SWA segment: positions
        # `[swa_window_low_global[t] + k for k in range(prefix_swa_count[t])]`,
        # paged into `unified_kv` SWA region:
        #   paged = state_slot[bid] * win + (global_pos % win)
        swa_seg_tok, swa_seg_k = _segment_indices(
            np.arange(T, dtype=np.int64), prefix_swa_count_np
        )
        state_slot_arr = np.asarray(
            state_slot_mapping_cpu[:scheduled_bs], dtype=np.int64
        )
        prefix_swa_global_pos = (
            swa_window_low_global[swa_seg_tok] + swa_seg_k
        )  # [sum prefix_swa_count] int64
        prefix_swa_paged_np = (
            state_slot_arr[batch_id_per_token_np[swa_seg_tok]] * win
            + (prefix_swa_global_pos % win)
        ).astype(np.int32)

        # ----- HCA compress paged offsets (layer-invariant, fully built here) -----
        # Per token, HCA section is `[block_tables[bid, k] for k in range(n_hca[bid])]`
        # mapped to `swa_pages + phys` (HCA block_capacity = 1).
        block_tables_np = var["block_tables"].np[:scheduled_bs]
        hca_seg_tok, hca_seg_k = _segment_indices(
            np.arange(T, dtype=np.int64), n_hca_per_token
        )
        hca_phys = block_tables_np[
            batch_id_per_token_np[hca_seg_tok], hca_seg_k
        ]  # [sum n_hca] int32
        hca_compress_paged_np = (swa_pages + hca_phys).astype(np.int32)

        # ----- Assemble flat prefix buffers (vectorised, no Python per-token loop) -----
        # Each per-token segment is laid out as [SWA_prefix..., compress...].
        # Dense (SWA only): buffer == `prefix_swa_paged_np` directly (no compress).
        # CSA / HCA: scatter SWA segments into [indptr[t], indptr[t]+swa_n[t])
        # and compress segments into [indptr[t]+swa_n[t], indptr[t]+swa_n[t]+comp_n[t]).
        # Both scatter dst positions derived from `_segment_indices` output:
        #   for SWA segment i (token = swa_seg_tok[i], col = swa_seg_k[i]):
        #     dst = prefix_*_indptr[token] + col
        #   for compress segment j (token = comp_seg_tok[j], col = comp_seg_k[j]):
        #     dst = prefix_*_indptr[token] + prefix_swa_count[token] + col
        prefix_csa_total = int(prefix_csa_indptr_np[T])
        prefix_hca_total = int(prefix_hca_indptr_np[T])

        # Dense prefix buffer = SWA only, already in per-token cumsum order
        # (since _segment_indices walks tokens in order and emits cols 0..n-1).
        kv_indices_prefix_swa_np = prefix_swa_paged_np  # alias

        # CSA prefix buffer: SWA section scattered, CSA section pre-filled -1
        # (csa_translate_pack writes valid entries per-layer; -1 sentinel keeps
        # unfilled tail slots safe — paged_prefill kernel skips slot < 0).
        kv_indices_prefix_csa_np = np.full(prefix_csa_total, -1, dtype=np.int32)
        if prefix_swa_paged_np.size > 0:
            csa_swa_dst = prefix_csa_indptr_np[swa_seg_tok] + swa_seg_k
            kv_indices_prefix_csa_np[csa_swa_dst] = prefix_swa_paged_np

        # HCA prefix buffer: SWA section + HCA all-committed section, both
        # scattered via segment indices (no Python per-token loop).
        kv_indices_prefix_hca_np = np.empty(prefix_hca_total, dtype=np.int32)
        if prefix_swa_paged_np.size > 0:
            hca_swa_dst = prefix_hca_indptr_np[swa_seg_tok] + swa_seg_k
            kv_indices_prefix_hca_np[hca_swa_dst] = prefix_swa_paged_np
        if hca_compress_paged_np.size > 0:
            hca_comp_dst = (
                prefix_hca_indptr_np[hca_seg_tok]
                + prefix_swa_count_np[hca_seg_tok]
                + hca_seg_k
            )
            kv_indices_prefix_hca_np[hca_comp_dst] = hca_compress_paged_np

        # ----- One-shot H2D for everything -----
        attn_metadata.kv_indices_extend = torch.from_numpy(ext_indices_np).to(device)
        attn_metadata.kv_indptr_extend = torch.from_numpy(
            extend_indptr_np.astype(np.int32)
        ).to(device)
        attn_metadata.kv_indices_prefix_swa = torch.from_numpy(
            kv_indices_prefix_swa_np
        ).to(device)
        attn_metadata.kv_indptr_prefix_swa = torch.from_numpy(
            prefix_swa_indptr_np.astype(np.int32)
        ).to(device)
        attn_metadata.kv_indices_prefix_csa = torch.from_numpy(
            kv_indices_prefix_csa_np
        ).to(device)
        attn_metadata.kv_indptr_prefix_csa = torch.from_numpy(
            prefix_csa_indptr_np.astype(np.int32)
        ).to(device)
        attn_metadata.kv_indices_prefix_hca = torch.from_numpy(
            kv_indices_prefix_hca_np
        ).to(device)
        attn_metadata.kv_indptr_prefix_hca = torch.from_numpy(
            prefix_hca_indptr_np.astype(np.int32)
        ).to(device)
        attn_metadata.skip_prefix_len_csa = torch.from_numpy(
            prefix_swa_count_np.astype(np.int32)
        ).to(device)
        attn_metadata.swa_pages = swa_pages

    def _build_compress_plans(
        self, extend_lens_np, seq_lens_np, device, *, for_decode_cg: bool
    ):
        """Build per-ratio CompressPlan dict consumed by batched compressor.

        Reuse this from prepare_decode / prepare_prefill / prepare_capture —
        caller supplies extend_lens / seq_lens (np int32) and target device.
        Plan tensors are written into the pre-allocated
        `v4_compress_plan_{ratio}` / `v4_write_plan_{ratio}` CpuGpuBuffers
        (fixed pointers for CUDAGraph capture); the kernels skip
        sentinel-marked tail rows.

        `for_decode_cg`: True for decode runtime AND decode CG capture —
        the returned plan_gpu is sliced to a fixed `_decode_compress_cap`
        per ratio so capture/replay shapes match. False for eager prefill —
        the plan_gpu is sliced to the actual `n_compress` (smallest grid).
        """
        from atom.model_ops.v4_kernels import make_compress_plans

        if not self._unique_compress_ratios_overlap:
            return {}
        # Ensure inputs are np int32 (callers may pass torch tensors / lists).
        if isinstance(extend_lens_np, torch.Tensor):
            extend_lens_np = extend_lens_np.cpu().numpy().astype(np.int32)
        if isinstance(seq_lens_np, torch.Tensor):
            seq_lens_np = seq_lens_np.cpu().numpy().astype(np.int32)
        var = self.model_runner.forward_vars
        plan_buffers = {
            ratio: {
                "compress": var[f"v4_compress_plan_{ratio}"],
                "write": var[f"v4_write_plan_{ratio}"],
            }
            for ratio, _ in self._unique_compress_ratios_overlap
        }
        return make_compress_plans(
            np.ascontiguousarray(extend_lens_np, dtype=np.int32),
            np.ascontiguousarray(seq_lens_np, dtype=np.int32),
            self._unique_compress_ratios_overlap,
            device,
            plan_buffers=plan_buffers,
            decode_capacity_per_ratio=(
                self._decode_compress_cap if for_decode_cg else None
            ),
        )

    def _populate_block_tables(
        self, batch: ScheduledBatch, scheduled_bs: int
    ) -> torch.Tensor:
        """Populate `forward_vars["block_tables"]` from the batch and return
        the GPU view sliced to `scheduled_bs` rows.

        Mirrors `CommonAttentionBuilder.prepare_block_tables` but is invoked
        unconditionally (parent only calls it when has_cached).
        """
        var = self.model_runner.forward_vars
        block_tables_np = var["block_tables"].np
        for i, block_table in enumerate(batch.block_tables[:scheduled_bs]):
            block_tables_np[i] = 0
            block_tables_np[i, : len(block_table)] = block_table
        return var["block_tables"].copy_to_gpu(scheduled_bs)

    def _populate_state_slot_mapping(
        self, batch: ScheduledBatch, scheduled_bs: int, return_cpu: bool = False
    ):
        """Build `[scheduled_bs]` int32 tensor of per-request state-cache slots.

        With slots_per_req() == 1, slot index == per_req_cache_group. This
        is what V4 forward uses to index `swa_kv` and `Compressor.kv_state`
        (the per-request state pool, distinct from the per-token paged-KV
        `slot_mapping`).

        When `return_cpu=True`, returns `(gpu_tensor, cpu_numpy)`. The CPU
        copy is consumed by the V4 forward path to avoid `.tolist()` syncs
        (PR-A Phase 2).
        """
        import numpy as np

        groups_np = np.asarray(
            batch.per_req_cache_groups[:scheduled_bs], dtype=np.int32
        )
        # Warmup / dummy_run batches don't allocate per_req_cache slots
        # (per_req_cache_groups is empty). Fall back to slot 0 for all seqs
        # so V4 forward can take the normal path uniformly — slot 0's state
        # cache is reset on the first real prefill (start_pos==0 path masks
        # state reads, fresh writes overwrite warmup pollution).
        if len(groups_np) < scheduled_bs:
            groups_np = np.zeros(scheduled_bs, dtype=np.int32)
        gpu = self._stage("v4_meta_state_slot_groups", groups_np)
        if return_cpu:
            return gpu, groups_np
        return gpu

    def build_for_cudagraph_capture(
        self, bs: int
    ) -> tuple[AttentionMetaData_DSV4, Context]:
        """Build attn_metadata for CUDAGraph capture using a synthetic decode batch.

        Synthesizes bs sequences each at start_pos=window_size (so SWA window
        is full + 1 CSA committed entry — exercises the production decode
        codepath: state-cache reads, sparse_attn gather, indexer fp8 logits).

        Per-fwd metadata is populated through the SAME helpers prepare_decode
        uses (`_attach_v4_indexer_meta`, `_attach_v4_per_fwd_meta`,
        `_build_compress_plans`), so all GPU views point to the pre-allocated
        buffers in `forward_vars`. Replay-time prepare_decode writes into the
        SAME buffers — captured graph reads stable addresses.

        NOTE on dynamic-shape kernels (`update_compressor_states` / `swa_write`):
        these currently use variable kernel grids (`grid=(num_compress,)`),
        which CUDAGraph capture rejects. A follow-up PR converts them to fixed
        grid + sentinel masking. Until then, capture itself can succeed (the
        helpers run on CPU + small H2D), but model.forward inside torch.cuda.graph
        will likely fail at the first such kernel launch — the user can detect
        this via capture log output. (`fused_compress_attn` is already
        CG-safe: launches at the decode-tight slice
        (`_decode_compress_cap[ratio]`, baked at capture) and
        sentinel-skips inactive rows internally for both BF16 Main and FP8
        Indexer paths.)
        """
        device = self.model_runner.device
        var = self.model_runner.forward_vars
        # Honor MTP at capture time: V4-Pro `mtp_k=1` → 2 tokens/req. The
        # outer `model_runner.capture_cudagraph` populates cu_seqlens_q with
        # the same layout, so capture and replay see identical shapes.
        max_q_len = 1 + self.max_spec_steps
        total_tokens = bs * max_q_len
        win = self.window_size

        # Synthetic state: each seq has already produced `win` tokens; this
        # fwd is `max_q_len` decode/draft steps at positions
        # [win, win+max_q_len). Hits is_pure_decode (start_pos > 0, uniform
        # tok-per-seq), exercising Phase B/C/E paths during capture.
        start_pos = win
        positions_np = (np.arange(total_tokens, dtype=np.int64) % max_q_len) + start_pos
        cu_seqlens_q_np = np.arange(0, bs + 1, dtype=np.int32) * max_q_len
        context_lens_np = np.full(bs, start_pos + max_q_len, dtype=np.int32)
        # Slot mapping: use real per-req cache slots [0..bs-1].
        state_slot_np = np.arange(bs, dtype=np.int32)
        # Block tables: block 0 for every seq (placeholder; capture warmup
        # fills it via real reads but the data is throwaway).
        block_tables_np = np.zeros(
            (bs, var["block_tables"].np.shape[1]), dtype=np.int32
        )

        # Stage CPU mirrors → forward_vars + capture-time GPU views.
        var["positions"].np[:total_tokens] = positions_np
        positions = var["positions"].copy_to_gpu(total_tokens)
        var["cu_seqlens_q"].np[: bs + 1] = cu_seqlens_q_np
        cu_seqlens_q_gpu = var["cu_seqlens_q"].copy_to_gpu(bs + 1)
        var["context_lens"].np[:bs] = context_lens_np
        context_lens_gpu = var["context_lens"].copy_to_gpu(bs)
        var["block_tables"].np[:bs] = block_tables_np
        block_tables_gpu = var["block_tables"].copy_to_gpu(bs)
        state_slot_gpu = self._stage("v4_meta_state_slot_groups", state_slot_np)

        attn_metadata = AttentionMetaData_DSV4(
            cu_seqlens_q=cu_seqlens_q_gpu,
            cu_seqlens_k=None,
            max_seqlen_q=max_q_len,
            max_seqlen_k=int(context_lens_np.max()) if bs else 1,
            min_seqlen_q=0,
            dropout_p=0.0,
            has_cached=False,
            total_kv=int(context_lens_np.sum()),
            num_cached_tokens=None,
            block_tables=block_tables_gpu,
            context_lens=context_lens_gpu,
        )
        attn_metadata.state_slot_mapping = state_slot_gpu
        attn_metadata.state_slot_mapping_cpu = state_slot_np
        attn_metadata.start_pos_per_seq_cpu = positions_np[cu_seqlens_q_np[:bs]]

        # Build compress_plans + per-fwd meta + indexer meta via the same
        # helpers used at runtime — guarantees addresses match.
        extend_lens_np = np.full(bs, max_q_len, dtype=np.int32)
        attn_metadata.compress_plans = self._build_compress_plans(
            extend_lens_np, context_lens_np, device, for_decode_cg=True
        )
        # Capture: padded_bs == scheduled_bs == bs (synthetic batch is full).
        # Must run BEFORE `_attach_v4_indexer_meta` so the indexer-side meta
        # builder can reuse the shared per-fwd GPU tensors.
        self._attach_v4_per_fwd_meta(
            attn_metadata,
            positions_np,
            cu_seqlens_q_np,
            attn_metadata.start_pos_per_seq_cpu,
            attn_metadata.state_slot_mapping_cpu,
            bs,
            total_tokens,
            padded_bs=bs,
            max_q_len=max_q_len,
        )
        self._attach_v4_indexer_meta(
            attn_metadata,
            cu_seqlens_q_np,
            attn_metadata.start_pos_per_seq_cpu,
            bs,
            total_tokens,
            positions_gpu=positions,
        )

        context = Context(
            positions=positions,
            is_prefill=False,
            batch_size=bs,
            graph_bs=bs,
        )
        return attn_metadata, context

    # ------------------------------------------------------------------ #
    # Helpers.                                                           #
    # ------------------------------------------------------------------ #

    def _alloc_v4_metadata_buffers(self) -> None:
        """Pre-allocate every CpuGpuBuffer the V4 metadata builder writes into.

        Bounds:
          - per-seq:        max_bs
          - per-token:      max_num_batched_tokens
          - window_topk:    max_num_batched_tokens * window_size
          - csa compress:   max_num_batched_tokens * index_topk
          - hca compress:   max_num_batched_tokens * max_num_blocks_per_seq
          - csa gather:     max_bs * max_num_blocks_per_seq * (block_size // 4)
          - decode swa dst: max_bs * window_size

        Memory footprint at typical config (max_bs=16, mnbt=8192, win=128,
        index_topk=1024, max_num_blocks_per_seq=64): ~80 MB total. Allocated
        once at builder init; pointers stay fixed for CUDAGraph capture.
        """
        i32 = {"dtype": torch.int32, "device": self.device}
        i64 = {"dtype": torch.int64, "device": self.device}
        mnbt = self.max_num_batched_tokens
        bs = self.max_bs
        win = self.window_size

        bufs: dict = {}

        # `kv_indptr` is touched unconditionally by the global capture loop
        # (model_runner.capture_cudagraph: `forward_vars["kv_indptr"].zero_()`).
        # MLA backends own this buffer; V4 doesn't use it for its own kernels
        # but allocates a min-size stub so the capture loop runs. Sized for
        # potential future reuse if a V4-side MLA kernel needs paged KV indices.
        bufs["kv_indptr"] = CpuGpuBuffer(bs + 1, **i32)

        # _attach_v4_per_fwd_meta + _populate_state_slot_mapping.
        # state_slot is staged ONCE into v4_meta_state_slot_groups (set by
        # `_populate_state_slot_mapping`); attn_metadata.state_slot_mapping
        # exposes that GPU view to all downstream consumers (no second
        # H2D-staged copy).
        bufs["v4_meta_start_pos_per_seq"] = CpuGpuBuffer(bs, **i64)
        bufs["v4_meta_token_num_per_seq"] = CpuGpuBuffer(bs, **i64)
        bufs["v4_meta_state_slot_groups"] = CpuGpuBuffer(bs, **i32)
        bufs["v4_meta_swa_write_indices"] = CpuGpuBuffer(mnbt, **i64)
        # Window-topk batched output buffer (per-token sliding-window indices).
        # Sized [mnbt, win] int32 — `_build_window_topk_batched(out=)` writes
        # into this; the [total_tokens, win] view feeds `_attach_v4_paged_decode_meta`'s
        # SWA paged-offsets derivation. Builder-internal intermediate, but the
        # buffer must be CG-capture-stable since it backs a tensor read by the
        # captured graph (the SWA paged offsets derived from it land in
        # `kv_indices_swa/csa/hca` which the captured kernels do read).
        bufs["v4_meta_window_topk"] = CpuGpuBuffer(mnbt, win, **i32)

        # Phase B: paged-decode index buffers (consumed by Phase C/E).
        # Sized to worst-case decode shape `T = max_bs * (1 + max_spec_steps)`
        # — these buffers are decode-only; prefill goes through
        # `_build_paged_prefill_meta` (per-fwd alloc) and never touches them.
        # Per-buffer footprint at V4-Pro (T=32, win=128, index_topk=1024,
        # max_committed_hca=8192): swa 16KB / csa 144KB / hca 1.04MB; the rest
        # negligible.
        # Per-seq state (valid_count_csa) + the single per-token batch_id
        # mapping (`v4_batch_id_per_token`) replace per-token aliases of seq-
        # level data — downstream kernels do
        # `data[batch_id_per_token[t]]` instead of carrying a [T]-sized copy.
        T_dec = self.max_decode_tokens
        bufs["v4_kv_indices_swa"] = CpuGpuBuffer(T_dec * win, **i32)
        bufs["v4_kv_indices_csa"] = CpuGpuBuffer(T_dec * (win + self.index_topk), **i32)
        bufs["v4_kv_indices_hca"] = CpuGpuBuffer(
            T_dec * (win + self.max_committed_hca), **i32
        )
        bufs["v4_kv_indptr_swa"] = CpuGpuBuffer(T_dec + 1, **i32)
        bufs["v4_kv_indptr_csa"] = CpuGpuBuffer(T_dec + 1, **i32)
        bufs["v4_kv_indptr_hca"] = CpuGpuBuffer(T_dec + 1, **i32)
        # Per-seq `ctx_len // 4` (raw, no clamp). Consumed by csa_translate_pack
        # (kernel masks `(k < n_committed) & (k < index_topk)`) AND by the
        # indexer (cast to int64 inline). Built unconditionally in
        # `_attach_v4_per_fwd_meta`.
        bufs["v4_n_committed_csa_per_seq"] = CpuGpuBuffer(bs, **i32)
        # Single per-token mapping shared across ALL V4 consumers:
        #   - swa_write / csa_translate_pack (triton kernels, read int64 fine)
        #   - _build_v4_indexer_meta (PyTorch fancy index, REQUIRES int64)
        # int64 dtype satisfies the PyTorch constraint with one buffer rather
        # than maintaining an int32 + int64 mirror. Sized to `mnbt`
        # (worst-case prefill total tokens) since swa_write fires on prefill
        # paths too. Phase B decode only uses [:T_dec] of this buffer.
        bufs["v4_batch_id_per_token"] = CpuGpuBuffer(mnbt, **i64)

        # _build_v4_indexer_meta (CSA only — but allocate unconditionally;
        # never accessed when CSA layers are absent).
        # int32 — `cp_gather_indexer_k_quant_cache` kernel signature is `int32_t*`
        # for cu_seq_lens. Also reused as cu_starts/cu_ends for fp8_mqa_logits
        # (which accepts both int32 and int64).
        bufs["v4_indexer_cu_committed"] = CpuGpuBuffer(bs + 1, **i32)
        # NOTE: decode-path `logits` ([T, max_model_len_idx] fp32) and
        # `topk_indices` ([T, index_topk] int32) are NOT pre-allocated —
        # they are write-once GPU scratch with no CPU mirror, allocated
        # per-fwd inside `Indexer._score_topk_decode` via `torch.empty`.
        # Under CUDAGraph capture they land in the graph's private pool
        # and replay reuses the same address; eager keeps the standard
        # caching-allocator fast path.
        # Per-token write offset consumed by `csa_translate_pack` (decode path
        # fills with `window_size`; prefill path will fill with per-token
        # prior_swa_count once the new dual-source kernel lands). Allocated
        # `mnbt` (worst-case prefill) so prefill writes don't overflow.
        bufs["v4_skip_prefix_len_csa"] = CpuGpuBuffer(mnbt, **i32)

        # Compress plan buffers (per-ratio) — pre-allocated for CUDAGraph
        # plan-tensor address stability. `make_compress_plans(..., plan_buffers=)`
        # writes into these and sentinel-fills the trailing rows. Worst-case
        # sizes: num_compress ≤ ⌈mnbt/ratio⌉ + bs (one boundary per seq plus
        # alignment slack); num_write ≤ bs * STATE_SIZE (per-seq ring window
        # carries STATE_SIZE rows per fwd at most).
        #
        # The decode CG path uses a much tighter capacity than the prefill
        # worst case — the kernel grid is dictated by the slice of this
        # buffer that we hand to the kernel, and decode only ever needs
        # `max_decode_tokens // ratio + max_bs` rows (vs `mnbt // ratio + bs`
        # for prefill, which is ~13× larger at typical config). We still
        # allocate the full prefill capacity (eager prefill needs it), but
        # both decode capture and replay slice down to `_decode_compress_cap`
        # so the captured grid is the decode-tight bound. capture and
        # replay MUST use the same value (CG kernel call args are baked).
        self._decode_compress_cap: dict[int, int] = {}
        for ratio, is_overlap in self._unique_compress_ratios_overlap:
            state_size = (2 if is_overlap else 1) * ratio
            max_compress = mnbt // ratio + bs
            max_write = min(mnbt, bs * state_size)
            bufs[f"v4_compress_plan_{ratio}"] = CpuGpuBuffer(max_compress, 4, **i32)
            bufs[f"v4_write_plan_{ratio}"] = CpuGpuBuffer(max_write, 4, **i32)
            # Pre-fill with sentinel so capture-time buffer state is valid
            # even before the first non-empty fwd.
            bufs[f"v4_compress_plan_{ratio}"].cpu.fill_(-1)
            bufs[f"v4_compress_plan_{ratio}"].copy_to_gpu()
            bufs[f"v4_write_plan_{ratio}"].cpu.fill_(-1)
            bufs[f"v4_write_plan_{ratio}"].copy_to_gpu()
            # Decode-tight bound. Worst case = total_tokens_decode boundaries
            # all firing simultaneously, each in its own ratio-aligned slot.
            # `total_tokens_decode = max_decode_tokens` (= max_bs * (1+spec)).
            # The `+ max_bs` covers per-seq alignment slack (each seq can hit
            # at most one extra boundary when extend_len isn't ratio-aligned).
            self._decode_compress_cap[ratio] = self.max_decode_tokens // ratio + bs

        self.model_runner.forward_vars.update(bufs)

    def _stage(self, name: str, arr) -> torch.Tensor:
        """Write numpy `arr` into `forward_vars[name]` (CpuGpuBuffer) and
        return its GPU view sliced to len(arr). Auto-casts dtype to match
        the buffer (e.g. int64 → int32). Asserts the buffer is large enough.
        """
        buf = self.model_runner.forward_vars[name]
        n = arr.shape[0] if arr.ndim > 0 else 1
        if n == 0:
            return buf.gpu[:0]
        cap = buf.np.shape[0]
        assert n <= cap, (
            f"V4 buffer {name!r} too small: need {n}, have {cap}. "
            f"Increase the corresponding bound in _alloc_v4_metadata_buffers."
        )
        if arr.dtype != buf.np.dtype:
            arr = arr.astype(buf.np.dtype, copy=False)
        buf.np[:n] = arr
        return buf.copy_to_gpu(n)

    @staticmethod
    def _numel(shape: tuple) -> int:
        n = 1
        for s in shape:
            n *= s
        return n

    @staticmethod
    def _contiguous_strides(shape: tuple) -> tuple:
        strides = []
        acc = 1
        for s in reversed(shape):
            strides.append(acc)
            acc *= s
        return tuple(reversed(strides))

    @staticmethod
    def _make_gather_slot(
        buf: torch.Tensor,
        stride: int,
        state_tensors: list[torch.Tensor],
    ):
        """Return a callable that copies compute tensors → staging buffer for one slot."""
        offsets_and_sizes = []
        off = 0
        for t in state_tensors:
            n_layers = t.shape[0]
            per_layer = t[0, 0].numel()
            total = n_layers * per_layer
            offsets_and_sizes.append((off, n_layers, per_layer))
            off += total
        assert off == stride

        def gather_slot(compute_slot: int, pool_idx: int) -> None:
            dst_start = pool_idx * stride
            for t, (off, n_layers, per_layer) in zip(state_tensors, offsets_and_sizes):
                buf[dst_start + off : dst_start + off + n_layers * per_layer] = t[
                    :, compute_slot
                ].reshape(-1)

        return gather_slot

    @staticmethod
    def _make_scatter_slot(
        buf: torch.Tensor,
        stride: int,
        state_tensors: list[torch.Tensor],
    ):
        """Return a callable that copies staging buffer → compute tensors for one slot."""
        offsets_and_sizes = []
        off = 0
        for t in state_tensors:
            n_layers = t.shape[0]
            per_layer = t[0, 0].numel()
            total = n_layers * per_layer
            offsets_and_sizes.append((off, n_layers, per_layer))
            off += total
        assert off == stride

        def scatter_slot(compute_slot: int, pool_idx: int) -> None:
            src_start = pool_idx * stride
            for t, (off, n_layers, per_layer) in zip(state_tensors, offsets_and_sizes):
                chunk = buf[src_start + off : src_start + off + n_layers * per_layer]
                t[:, compute_slot] = chunk.view(t[:, compute_slot].shape)

        return scatter_slot

    def _zero_state(self, shape: tuple, device) -> torch.Tensor:
        return torch.zeros(shape, dtype=self._state_dtype, device=device)

    def _neg_inf_state(self, shape: tuple, device) -> torch.Tensor:
        return torch.full(shape, float("-inf"), dtype=self._state_dtype, device=device)
