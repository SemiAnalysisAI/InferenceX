#!/usr/bin/env python3
"""Mooncake TransferEngine adapter for the KV-transfer suite.

P2PHANDSHAKE metadata (no etcd); the peer session id is ip:rpc_port. The
binding is sync-only in the shape production uses it: SGLang's mooncake
connector posts blocking calls from a transfer thread pool (the binding
releases the GIL), so post() here hands the sync call to a worker thread and
wait() joins it. The wheel links libcudart.so.12, which cu13
images do not carry; the adapter dlopens it from the nvidia-cuda-runtime-cu12
package before the import so no launcher-side LD_LIBRARY_PATH seam is needed.
The wheel also links libcuda.so.1 itself, which is why this backend is
NVIDIA-only (import fails on ROCm nodes, measured on mi355x).
"""

from __future__ import annotations

import ctypes
import os
import time
from concurrent.futures import ThreadPoolExecutor

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
        # Same-fabric GB pairs: the NVLink-IPC transport claims cross-node
        # segments inside one NVLink domain and then fails the address import
        # (nvlink_transport "Requested address not found", first kv CI run on
        # gb200). This row measures the rdma lane, so pin the transport off;
        # the ROCm twin (MC_USE_HIP_IPC) misclaims the same way on mi355x.
        os.environ.setdefault("MC_USE_NVLINK_IPC", "0")
        self._engine = TransferEngine()
        self._ip = kv_workload.iface_ipv4(args.socket_ifname)
        local = f"{self._ip}:{args.kv_mc_port + (0 if role == 'target' else 1)}"
        rc = self._engine.initialize(local, "P2PHANDSHAKE", "rdma", "")
        if rc != 0:
            raise RuntimeError(f"mooncake initialize failed rc={rc}")
        self._pool = None
        self._bulk = None
        self._peer = None
        workers = max(int(v) for v in str(getattr(args, "batch_sizes", "1")).split())
        self._exec = ThreadPoolExecutor(max_workers=workers)

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

    def _split(self, run, prep_s):
        """(post, wait, prep_s) around a blocking call via the worker pool."""
        pending: list = []

        def post():
            pending.append(self._exec.submit(run))

        def wait():
            pending.pop(0).result()

        return post, wait, prep_s

    def make_paged(self, cfg, op, local_tables, remote_tables):
        start = time.perf_counter()
        local = (self._pool.ptr + kv_workload.page_offsets(cfg, local_tables)).tolist()
        remote = (self._peer["pool_base"] + kv_workload.page_offsets(cfg, remote_tables)).tolist()
        sizes = kv_workload.desc_sizes(cfg).tolist()
        chunks = [(i, min(i + BATCH_CAP, len(local))) for i in range(0, len(local), BATCH_CAP)]
        session = self._peer["session"]
        func = self._engine.batch_transfer_sync_read if op == "pull" \
            else self._engine.batch_transfer_sync_write

        def run():
            for i, j in chunks:
                rc = func(session, local[i:j], remote[i:j], sizes[i:j])
                if rc != 0:
                    raise RuntimeError(f"mooncake batch transfer failed rc={rc}")

        return self._split(run, time.perf_counter() - start)

    def make_bulk(self, nbytes, op):
        session = self._peer["session"]
        func = self._engine.transfer_sync_read if op == "pull" \
            else self._engine.transfer_sync_write
        local, remote = self._bulk.ptr, self._peer["bulk_base"]

        def run():
            rc = func(session, local, remote, nbytes)
            if rc != 0:
                raise RuntimeError(f"mooncake bulk transfer failed rc={rc}")

        return self._split(run, 0.0)

    def teardown(self) -> None:
        self._exec.shutdown(wait=False)
