#!/usr/bin/env python3
"""MoRI-IO adapter (AMD's native P2P engine). Transfers address (region,
offset, size), so the paged list becomes offset lists over one registration.
Engine/Memory descriptors are packed blobs exchanged through the harness; the
engine's own control plane binds host/port from the SKU's socket interface.
Posts are capped at ``BATCH_CAP`` offsets per batch call to bound SQ/WR usage
and awaited together, the shape the SGLang MoRI-IO connector posts.
"""

from __future__ import annotations

import time

import kv_workload
from kv_backend import KVBackend

BATCH_CAP = 16384


class MoRIIOBackend(KVBackend):
    name = "mori-io"
    maturity = "production"

    def __init__(self, args, role, device):
        super().__init__(args, role, device)
        from mori.io import (BackendType, IOEngine, IOEngineConfig,
                             MemoryLocationType, PollCqMode, RdmaBackendConfig)

        self._gpu_location = MemoryLocationType.GPU

        try:
            import mori

            self.library_version = getattr(mori, "__version__", None)
        except Exception:
            self.library_version = None
        self._mori_io = __import__("mori.io", fromlist=["EngineDesc", "MemoryDesc"])
        host = kv_workload.iface_ipv4(args.socket_ifname) if args.socket_ifname else ""
        port = int(args.kv_mori_port) + (0 if role == "target" else 1)
        self._engine = IOEngine(key=role, config=IOEngineConfig(host=host, port=port))
        self._engine.create_backend(BackendType.RDMA, RdmaBackendConfig(
            qp_per_transfer=int(args.kv_mori_qp),
            post_batch_size=-1,
            num_worker_threads=1,
            poll_cq_mode=PollCqMode.POLLING,
            enable_notification=False,
            enable_transfer_chunking=bool(args.kv_mori_chunking),
            chunk_bytes=65536,
            max_chunks_per_transfer=64,
        ))
        self._pool_mem = None
        self._bulk_mem = None
        self._sessions = None

    def register(self, pool, bulk, reg_layout=None) -> None:
        self._pool_mem = self._engine.register_memory(
            pool.ptr, pool.nbytes, pool.device, self._gpu_location)
        self._bulk_mem = self._engine.register_memory(
            bulk.ptr, bulk.nbytes, bulk.device, self._gpu_location)

    def publish(self) -> dict:
        return {
            "engine": bytes(self._engine.get_engine_desc().pack()),
            "pool": bytes(self._pool_mem.pack()),
            "bulk": bytes(self._bulk_mem.pack()),
        }

    def connect(self, peer: dict) -> None:
        self._engine.register_remote_engine(self._mori_io.EngineDesc.unpack(peer["engine"]))
        remote_pool = self._mori_io.MemoryDesc.unpack(peer["pool"])
        remote_bulk = self._mori_io.MemoryDesc.unpack(peer["bulk"])
        self._sessions = {
            "pool": self._engine.create_session(self._pool_mem, remote_pool),
            "bulk": self._engine.create_session(self._bulk_mem, remote_bulk),
        }

    @staticmethod
    def _wait(statuses):
        for status in statuses:
            status.Wait()
            if not status.Succeeded():
                raise RuntimeError(f"mori-io transfer failed: {status.Message()}")

    def make_paged(self, cfg, op, local_tables, remote_tables):
        start = time.perf_counter()
        local = kv_workload.page_offsets(cfg, local_tables).tolist()
        remote = kv_workload.page_offsets(cfg, remote_tables).tolist()
        sizes = kv_workload.desc_sizes(cfg).tolist()
        chunks = [(i, min(i + BATCH_CAP, len(local))) for i in range(0, len(local), BATCH_CAP)]
        session = self._sessions["pool"]
        func = session.batch_read if op == "pull" else session.batch_write
        engine = self._engine
        prep_s = time.perf_counter() - start
        statuses: list = []

        def post():
            statuses.clear()
            statuses.extend(
                func(local[i:j], remote[i:j], sizes[i:j], engine.allocate_transfer_uid())
                for i, j in chunks
            )

        def wait():
            self._wait(statuses)

        return post, wait, prep_s

    # Verbs providers cap a single WR's message size (1 GiB on the Pollara path:
    # a 2.3 GB single-WR bulk read dies ibv_post_send EINVAL). Split client-side;
    # MoRI's own enable_transfer_chunking covers the same ground server-config-side,
    # but a library-default row must not depend on a tuned engine config.
    BULK_WR_CAP = 1 << 30

    def make_bulk(self, nbytes, op):
        session = self._sessions["bulk"]
        func = session.read if op == "pull" else session.write
        engine = self._engine
        spans = [(offset, min(offset + self.BULK_WR_CAP, nbytes))
                 for offset in range(0, nbytes, self.BULK_WR_CAP)]

        statuses: list = []

        def post():
            statuses.clear()
            statuses.extend(
                func(start, start, end - start, engine.allocate_transfer_uid())
                for start, end in spans
            )

        def wait():
            self._wait(statuses)

        return post, wait, 0.0
