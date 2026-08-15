#!/usr/bin/env python3
"""Keep aiter custom-all-reduce IPC-meta sequence numbers in lockstep across ranks.

`IPCBufferPool.flush_graph_buffers()` returns early when a rank recorded zero new
graph buffers, skipping `_gather_ipc_meta()` — but that call is what increments the
per-rank `_ipc_seq` used to build TCP-store keys. A rank that skips it is
permanently one sequence behind, so on a later capture round it blocks in
`store.get()` on a key prefix no other rank will ever write, and the whole TP group
deadlocks at the end of CUDA graph capture (seen on Kimi-K3 TP8xPP2: PP1 ranks with
a single "Registering N cuda graph addresses" line hang while the rest spin in NCCL).

Fix: always participate in the IPC-meta exchange, with empty payloads when there is
nothing to register, so every rank advances `_ipc_seq` identically.

Same import discipline as patch_gemm_n6288_chunk: no torch/aiter at module scope.
"""
from __future__ import annotations

import os
import sys


def install() -> None:
    if os.environ.get("_AITER_CA_FLUSH_SYNC_INSTALLING") == "1":
        return
    os.environ["_AITER_CA_FLUSH_SYNC_INSTALLING"] = "1"
    try:
        from aiter.dist.device_communicators import custom_all_reduce as car

        pool_cls = car.IPCBufferPool
        if getattr(pool_cls.flush_graph_buffers, "_kimik3_flush_sync", False):
            print("[patch_ca_graph_flush_sync] already installed", file=sys.stderr, flush=True)
            return

        _orig = pool_cls.flush_graph_buffers

        def flush_graph_buffers(self, ar_ptr):
            if self._graph_count_fn(ar_ptr) == 0:
                import torch

                self._gather_ipc_meta(
                    (
                        torch.empty(0, dtype=torch.uint8),
                        torch.empty(0, dtype=torch.int64),
                    )
                )
                return
            return _orig(self, ar_ptr)

        flush_graph_buffers._kimik3_flush_sync = True  # type: ignore[attr-defined]
        pool_cls.flush_graph_buffers = flush_graph_buffers
        print(
            "[patch_ca_graph_flush_sync] installed IPCBufferPool.flush_graph_buffers "
            "(empty-payload gather keeps _ipc_seq aligned)",
            file=sys.stderr,
            flush=True,
        )
    finally:
        os.environ.pop("_AITER_CA_FLUSH_SYNC_INSTALLING", None)


def main() -> int:
    try:
        install()
    except Exception as exc:
        print(f"[patch_ca_graph_flush_sync] FAILED: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
