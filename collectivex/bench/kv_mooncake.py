#!/usr/bin/env python3
"""Mooncake TransferEngine adapter, driven the way vLLM's MooncakeConnector
drives it (32ad1400d7).

vLLM's connector is push-only: the prefill side sends each ready group of
requests to its decode peer as ONE batch_transfer_sync_write, run on a
10-worker executor, over P2PHANDSHAKE metadata (no etcd). It registers each
layer's cache view as its own region, and on DSV4's padded block-row layout a
layer's stride (the row) exceeds its page, so no two blocks coalesce: every
(layer, block) is its own entry, and pages move without the row's padding
(kv_workload.layer_entries). NIC choice is the engine's own auto-discovery
unless the registry pins a device (Pollara GPU-paired NICs, `{gpu}` expands to
the physical GPU index). On EFA pools the upstream EFA build and protocol
"efa" carry it (prepare_backend installs that build); elsewhere verbs RC.
NVIDIA pools run the pinned cuda13 wheel vLLM's image swaps in; ROCm runs the
image-provided build (AMD's atom-dev tree), since upstream wheels link CUDA.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import kv_workload
from kv_backend import KVBackend, library_version

WORKERS = 10  # vLLM MooncakeConnector's executor default


def _physical_gpu_index() -> int:
    """The physical GPU index behind this rank's visible device 0: GPU-paired
    NIC selection (rdma{gpu}) needs the host-level index, which the Slurm
    visibility mask carries."""
    for var in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        first = os.environ.get(var, "").split(",")[0].strip()
        if first.isdigit():
            return int(first)
    return 0


class MooncakeBackend(KVBackend):
    name = "mooncake"
    maturity = "production"

    def __init__(self, args, role, device):
        # Same-fabric GB pairs: the NVLink-IPC transport claims cross-node
        # segments inside one NVLink domain and then fails the address import
        # (nvlink_transport "Requested address not found", first kv CI run on
        # gb200). This row measures the rdma lane, so pin the transport off;
        # the ROCm twin (MC_USE_HIP_IPC) misclaims the same way on mi355x.
        # vLLM sets no MC_* variable; this one only keeps the row runnable.
        os.environ.setdefault("MC_USE_NVLINK_IPC", "0")
        from mooncake.engine import TransferEngine
        import mooncake

        # The engine build actually imported, not the pin prepare_backend.sh
        # attempted: image-provided builds register under varying dist names.
        self.library_version = library_version(
            ("mooncake-transfer-engine-efa-cuda13", "mooncake-transfer-engine-cuda13",
             "mooncake-transfer-engine", "mooncake"), mooncake)
        protocol = "efa" if os.environ.get("COLLX_RDMA_FABRIC") == "efa" else "rdma"
        self.transport = "libfabric" if protocol == "efa" else "verbs"
        if not args.socket_ifname:
            raise RuntimeError("mooncake needs --socket-ifname for its P2P handshake address")
        self._engine = TransferEngine()
        self._ip = kv_workload.iface_ipv4(args.socket_ifname)
        local = f"{self._ip}:{args.kv_mc_port + (0 if role == 'target' else 1)}"
        nic_filter = args.kv_device.replace("{gpu}", str(_physical_gpu_index()))
        self.nic_filter = nic_filter or None
        rc = self._engine.initialize(local, "P2PHANDSHAKE", protocol, nic_filter)
        if rc != 0:
            raise RuntimeError(f"mooncake initialize failed rc={rc} "
                               f"protocol={protocol} nic_filter={nic_filter!r}")
        self.engine_config = {"protocol": protocol, "workers": WORKERS,
                              "api": "batch_transfer_sync_write"}
        self._pool = self._bulk = self._peer = None
        self._exec = ThreadPoolExecutor(max_workers=WORKERS)

    def register(self, pool, bulk, row_bytes: int, reg_layout=None) -> None:
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
        local, sizes = kv_workload.layer_entries(cfg, local_rows)
        remote, _ = kv_workload.layer_entries(cfg, remote_rows)
        return local, remote, sizes

    def _submit(self, run):
        pending: list = []

        def post():
            pending.append(self._exec.submit(run))

        def wait():
            pending.pop(0).result()

        return post, wait

    @staticmethod
    def _push_only(op: str) -> None:
        if op != "push":
            raise RuntimeError("vLLM's MooncakeConnector only pushes; mooncake has no pull row")

    def make_paged(self, cfg, op, local_rows, remote_rows, request_id: int = 0):
        return self.make_burst(cfg, op, [(local_rows, remote_rows, request_id)])[0]

    def make_burst(self, cfg, op, requests):
        """One sync write batch for the whole ready burst, as vLLM sends a
        ready group of requests to one peer."""
        self._push_only(op)
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
        self._push_only(op)
        session, engine = self._peer["session"], self._engine
        local, remote = self._bulk.ptr, self._peer["bulk_base"]

        def run():
            rc = engine.transfer_sync_write(session, local, remote, nbytes)
            if rc != 0:
                raise RuntimeError(f"mooncake bulk write failed rc={rc}")

        return self._submit(run)

    def teardown(self) -> None:
        self._exec.shutdown(wait=False)
