#!/usr/bin/env python3
"""Mooncake TransferEngine adapter, driven the way vLLM 0.32.0's MooncakeConnector drives it.

maturity "candidate": vLLM 0.32.0 cannot start this connector for DSV4 (docs/kv-transfer.md).
"""

from __future__ import annotations

import fcntl
import os
import socket
import struct
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import kv_workload

WORKERS = 10  # vLLM MooncakeConnector's executor default


def iface_ipv4(iface: str) -> str:
    """IPv4 of a named interface (SIOCGIFADDR); the P2P handshake address."""
    packed = struct.pack("256s", iface.encode()[:15])
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        return socket.inet_ntoa(fcntl.ioctl(sock.fileno(), 0x8915, packed)[20:24])


class MooncakeBackend:
    maturity = "candidate"

    def __init__(self, args, role, device):
        # Inside one NVLink domain the NVLink-IPC transport claims cross-node segments
        # and then fails the address import; this row measures the rdma lane. vLLM sets
        # no MC_* variable; this one only keeps the row runnable.
        os.environ.setdefault("MC_USE_NVLINK_IPC", "0")
        from mooncake.engine import TransferEngine
        import mooncake

        # The engine build actually imported, not the pin prepare_backend.sh
        # attempted: image-provided builds register under varying dist names.
        self.library_version = kv_workload.library_version(
            ("mooncake-transfer-engine-efa-cuda13", "mooncake-transfer-engine-cuda13",
             "mooncake-transfer-engine", "mooncake"), mooncake)
        protocol = "efa" if os.environ.get("COLLX_RDMA_FABRIC") == "efa" else "rdma"
        self.transport = "libfabric" if protocol == "efa" else "verbs"
        if not args.socket_ifname:
            raise RuntimeError("mooncake needs --socket-ifname for its P2P handshake address")
        self._engine = TransferEngine()
        self._ip = iface_ipv4(args.socket_ifname)
        local = f"{self._ip}:{args.kv_mc_port + (0 if role == 'target' else 1)}"
        # The validated HCAs the row's network label describes (and NIXL's UCX runs on).
        # EFA keeps the engine's own choice, the configuration that ran on b300.
        devices = "" if protocol == "efa" else ",".join(
            dev.split(":")[0] for dev in os.environ.get("COLLX_RDMA_DEVICES", "").split(",") if dev)
        rc = self._engine.initialize(local, "P2PHANDSHAKE", protocol, devices)
        if rc != 0:
            raise RuntimeError(f"mooncake initialize failed rc={rc} protocol={protocol}")
        self.library_config = {"engine_contract": "vllm", "connector": "MooncakeConnector",
                               "protocol": protocol, "workers": WORKERS,
                               "api": "batch_transfer_sync_write", "devices": devices or None}
        self._pool = self._bulk = self._peer = None
        self._exec = ThreadPoolExecutor(max_workers=WORKERS)

    def register(self, pool, bulk, row_bytes: int) -> None:
        self._pool, self._bulk = pool, bulk
        if self._engine.register_memory(pool.ptr, pool.nbytes) != 0 \
                or self._engine.register_memory(bulk.ptr, bulk.nbytes) != 0:
            raise RuntimeError("mooncake memory registration failed")

    def publish(self) -> dict:
        return {"session": f"{self._ip}:{self._engine.get_rpc_port()}",
                "pool_base": self._pool.ptr, "bulk_base": self._bulk.ptr}

    def connect(self, peer: dict) -> None:
        self._peer = peer

    def request_entries(self, cfg, local_rows, remote_rows):
        return kv_workload.mooncake_entries(cfg, local_rows, remote_rows)

    def _submit(self, run):
        pending: list = []

        def post():
            pending.append(self._exec.submit(run))

        def wait():
            pending.pop(0).result()

        return post, wait

    def make_burst(self, cfg, op, requests):
        """One sync write batch for the whole ready burst, as vLLM sends a
        ready group of requests to one peer."""
        local, remote, sizes = [], [], []
        for local_rows, remote_rows, _ in requests:
            lo, ro, sz = self.request_entries(cfg, local_rows, remote_rows)
            local.append(lo)
            remote.append(ro)
            sizes.append(sz)
        src = (np.uint64(self._pool.ptr) + np.concatenate(local)).tolist()
        dst = (np.uint64(self._peer["pool_base"]) + np.concatenate(remote)).tolist()
        lens = np.concatenate(sizes).tolist()
        session, engine = self._peer["session"], self._engine

        def run():
            rc = engine.batch_transfer_sync_write(session, src, dst, lens)
            if rc != 0:
                raise RuntimeError(f"mooncake batch write failed rc={rc}")

        return [self._submit(run)]

    def make_bulk(self, nbytes, op):
        session, engine = self._peer["session"], self._engine
        local, remote = self._bulk.ptr, self._peer["bulk_base"]

        def run():
            rc = engine.transfer_sync_write(session, local, remote, nbytes)
            if rc != 0:
                raise RuntimeError(f"mooncake bulk write failed rc={rc}")

        return self._submit(run)

    def release(self) -> None:
        pass

    def teardown(self) -> None:
        self._exec.shutdown(wait=False)
