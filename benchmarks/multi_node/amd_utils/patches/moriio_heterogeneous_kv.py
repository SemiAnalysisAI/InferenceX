#!/usr/bin/env python3
"""Patch vLLM's MoRIIOConnector to transfer heterogeneous KV caches per-layer.

Why
---
MiniMax-M3 (MiniMaxM3SparseForCausalLM) is a hybrid sparse-attention model:

  * main attention layers register a ``FullAttentionSpec`` KV cache:
      rank-5 ``[2, num_blocks, block_size, num_kv_heads, head_dim]``, **fp8**, K+V
  * the lightning indexer (sparse layers) registers a separate
    ``MLAAttentionSpec`` index cache (``MiniMaxM3IndexerCache``):
      rank-3 ``[num_blocks, block_size, head_dim]``, **bf16**, key-only

The upstream MoRIIOConnector assumes a *single uniform* KV layout: it derives
``self.kv_cache_shape`` / ``block_len`` / ``element_size`` from the **first**
cache, and ``_read_blocks`` computes the transfer offsets **once** from
``first_layer`` and reuses them for **every** layer (see the in-code TODO
"block_len needs to be per-layer for ... hybrid attn"). For M3 this transfers
the bf16 key-only rank-3 index cache using the fp8 K+V rank-5 main-cache sizing,
corrupting the indexer state on the decode worker. The sparse layers then select
the wrong KV blocks and the model emits incoherent tokens (gsm8k ~= 0).

This is the vLLM analogue of the already-shipped SGLang MoRI DSA fix in
``patches/mori_conn.py`` (see patches/README.md).

Fix
---
Compute transfer geometry **per layer** from each layer's own tensor
(``shape`` / ``stride`` / ``element_size`` / rank), instead of from the first
cache. For homogeneous models every layer's geometry equals the first cache's,
so behaviour is unchanged; only hybrid models (M3) are affected.

Two minimal, targeted edits (READ path, which the M3 recipe uses with
``read_mode: true``):

  1. ``_compute_block_transfer_offsets`` -> use ``self.kv_caches[layer_name]``'s
     own shape (rank/dims) instead of the global ``self.kv_cache_shape``.
  2. ``_read_blocks`` -> call ``_compute_block_transfer_offsets`` inside the
     per-layer loop instead of once for ``first_layer``.

Idempotent: re-running detects the ``PATCHED heterogeneous-kv`` marker and exits.
"""
import os
import sys


def _default_target() -> str:
    try:
        import vllm
    except Exception:
        return ""
    return os.path.join(
        os.path.dirname(vllm.__file__),
        "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py",
    )


OLD1 = '''        assert self.kv_cache_shape is not None, "KV caches shape not initialized"
        is_mla = len(self.kv_cache_shape) == 3
        stride = self.kv_caches[layer_name].stride()
        sz = self.kv_caches[layer_name].element_size()
        if is_mla:
            blknum, blksize, hs = self.kv_cache_shape
            hn = 1
            block_stride = stride[0]
        else:
            _, blknum, blksize, hn, hs = self.kv_cache_shape'''

NEW1 = '''        # [PATCHED heterogeneous-kv] Use this layer's own shape so caches with a
        # different rank/dtype (MiniMax-M3: bf16 key-only rank-3 index cache vs
        # fp8 K+V rank-5 main cache) are sized per-layer, not from the first cache.
        layer_shape = tuple(self.kv_caches[layer_name].shape)
        assert layer_shape, "KV caches shape not initialized"
        is_mla = len(layer_shape) == 3
        stride = self.kv_caches[layer_name].stride()
        sz = self.kv_caches[layer_name].element_size()
        if is_mla:
            blknum, blksize, hs = layer_shape
            hn = 1
            block_stride = stride[0]
        else:
            _, blknum, blksize, hn, hs = layer_shape'''

OLD2 = '''        first_layer = list(self.layer_name_to_local_kv_cache_metadata.keys())[0]
        offs = self._compute_block_transfer_offsets(
            first_layer, local_block_ids, remote_block_ids, remote_moriio_meta
        )

        for layer_name in self.layer_name_to_local_kv_cache_metadata:
            sess_idx = list(self.layer_name_to_local_kv_cache_metadata.keys()).index(
                layer_name
            )
            # TODO : apply multi-session batch-read when moriio support it
            transfer_status = self.moriio_wrapper.read_remote_data(
                offs[2], offs[0], offs[1], sessions[sess_idx]
            )'''

NEW2 = '''        for layer_name in self.layer_name_to_local_kv_cache_metadata:
            sess_idx = list(self.layer_name_to_local_kv_cache_metadata.keys()).index(
                layer_name
            )
            # [PATCHED heterogeneous-kv] Per-layer offsets so the bf16 key-only
            # MiniMax-M3 index cache is transferred with its own geometry instead
            # of the first (main fp8 K+V) layer's.
            offs = self._compute_block_transfer_offsets(
                layer_name, local_block_ids, remote_block_ids, remote_moriio_meta
            )
            # TODO : apply multi-session batch-read when moriio support it
            transfer_status = self.moriio_wrapper.read_remote_data(
                offs[2], offs[0], offs[1], sessions[sess_idx]
            )'''


def main() -> int:
    target = sys.argv[1] if len(sys.argv) > 1 else _default_target()
    if not target or not os.path.isfile(target):
        print(f"[PATCH] moriio_connector.py not found ({target!r}); skipping")
        return 0
    src = open(target).read()
    if "PATCHED heterogeneous-kv" in src:
        print("[PATCH] moriio heterogeneous-kv already applied")
        return 0
    if OLD1 not in src:
        print("[PATCH] WARN: _compute_block_transfer_offsets pattern not found; "
              "connector version changed — skipping (no-op)")
        return 0
    if OLD2 not in src:
        print("[PATCH] WARN: _read_blocks pattern not found; "
              "connector version changed — skipping (no-op)")
        return 0
    src = src.replace(OLD1, NEW1, 1).replace(OLD2, NEW2, 1)
    # Validate it still compiles before writing.
    try:
        compile(src, target, "exec")
    except SyntaxError as e:
        print(f"[PATCH] ERROR: patched source fails to compile: {e}")
        return 1
    open(target, "w").write(src)
    print("[PATCH] Applied: moriio heterogeneous-kv per-layer transfer "
          "(MiniMax-M3 sparse index cache)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
