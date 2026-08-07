#!/usr/bin/env python3
"""Mooncake TransferEngine adapter for the KV-transfer suite.

P2PHANDSHAKE metadata (no etcd); the peer session id is ip:rpc_port. Sync
transfer calls block to completion, so a timed transfer is one call (bulk) or
one capped batch loop (paged). The wheel links libcudart.so.12, which cu13
images do not carry; the adapter dlopens it from the nvidia-cuda-runtime-cu12
package before the import so no launcher-side LD_LIBRARY_PATH seam is needed.
The wheel also links libcuda.so.1 itself, which is why this backend is
NVIDIA-only (import fails on ROCm nodes, measured on mi355x).
"""

from __future__ import annotations

import ctypes
import time

import kv_workload
from kv_backend import KVBackend

BATCH_CAP = 8192


def _preload_cudart() -> None:
    try:
        ctypes.CDLL("libcudart.so.12", mode=ctypes.RTLD_GLOBAL)
        return
    except OSError:
        pass
    import importlib.metadata as md
    import pathlib

    for entry in md.files("nvidia-cuda-runtime-cu12") or []:
        if entry.name == "libcudart.so.12":
            ctypes.CDLL(str(pathlib.Path(entry.locate()).resolve()), mode=ctypes.RTLD_GLOBAL)
            return
    raise RuntimeError("libcudart.so.12 unavailable; install nvidia-cuda-runtime-cu12")


class MooncakeBackend(KVBackend):
    name = "mooncake"
    maturity = "production"

    def __init__(self, args, role, device):
        super().__init__(args, role, device)
        _preload_cudart()
        from mooncake.engine import TransferEngine

        try:
            import importlib.metadata as md

            self.library_version = md.version("mooncake-transfer-engine")
        except Exception:
            self.library_version = None
        self._engine = TransferEngine()
        self._ip = kv_workload.iface_ipv4(args.socket_ifname)
        local = f"{self._ip}:{args.kv_mc_port + (0 if role == 'target' else 1)}"
        rc = self._engine.initialize(local, "P2PHANDSHAKE", "rdma", "")
        if rc != 0:
            raise RuntimeError(f"mooncake initialize failed rc={rc}")
        self._pool = None
        self._bulk = None
        self._peer = None

    def register(self, pool, bulk) -> None:
        self._pool, self._bulk = pool, bulk
        if self._engine.register_memory(pool.ptr, pool.nbytes) != 0 \
                or self._engine.register_memory(bulk.ptr, bulk.nbytes) != 0:
            raise RuntimeError("mooncake memory registration failed")

    def publish(self) -> dict:
        return {"session": f"{self._ip}:{self._engine.get_rpc_port()}",
                "pool_base": self._pool.ptr, "bulk_base": self._bulk.ptr}

    def connect(self, peer: dict) -> None:
        self._peer = peer

    def make_paged(self, cfg, op, local_table, remote_table):
        start = time.perf_counter()
        local = (self._pool.ptr + kv_workload.page_offsets(cfg, local_table)).tolist()
        remote = (self._peer["pool_base"] + kv_workload.page_offsets(cfg, remote_table)).tolist()
        sizes = [cfg["page_bytes"]] * len(local)
        chunks = [(i, min(i + BATCH_CAP, len(local))) for i in range(0, len(local), BATCH_CAP)]
        engine, session = self._engine, self._peer["session"]
        func = engine.batch_transfer_sync_read if op == "pull" \
            else engine.batch_transfer_sync_write
        prep_s = time.perf_counter() - start

        def transfer_once():
            for i, j in chunks:
                if func(session, local[i:j], remote[i:j], sizes[i:j]) != 0:
                    raise RuntimeError("mooncake batch transfer failed")

        return transfer_once, prep_s

    def make_bulk(self, nbytes, op):
        engine, session = self._engine, self._peer["session"]
        func = engine.transfer_sync_read if op == "pull" else engine.transfer_sync_write
        local, remote = self._bulk.ptr, self._peer["bulk_base"]

        def transfer_once():
            if func(session, local, remote, nbytes) != 0:
                raise RuntimeError("mooncake bulk transfer failed")

        return transfer_once, 0.0
