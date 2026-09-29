#!/usr/bin/env python3
"""NIXL adapter, driven the way vLLM's NixlConnector drives it (32ad1400d7).

UCX carries it on verbs fabrics (IB, RoCE); on AWS EFA, which is not a verbs
HCA and has no UCX transport, the wheel's LIBFABRIC plugin carries it over the
host libfabric the cluster's enroot hook mounts (vLLM's documented EFA config,
"backends": ["LIBFABRIC"]). Agent config and env follow vLLM: num_threads 4 on
UCX (unset on LIBFABRIC), telemetry capture on, UCX_RCACHE_MAX_UNRELEASED=1024
set before nixl loads. As in vLLM, each side preps ONE descriptor list over
every block row of the registered pool (local at registration, remote at
connect), and every request is a fresh make_prepped_xfer over its row indices
with a notification message, released once it completes; handles are never
reposted. Agent metadata rides the harness exchange (`add_remote_agent`),
not NIXL's TCP listener.
"""

from __future__ import annotations

import os

import numpy as np

from kv_backend import KVBackend, library_version

# b300's former CX NICs refused cuda registrations somewhere between 7083 and
# 8847 MiB (an ~8 GiB MR wall); UCX surfaced no error and the initiator later
# segfaulted in ucp_worker_add_rkey_config resolving the region's rkey.
# Registering the pool in pieces below the wall sidesteps it everywhere; the
# pieces are cut on the row grid so no descriptor straddles two of them.
REG_CHUNK_BYTES = 4 << 30


def reg_spans(nbytes: int, layout,
              cap: int = REG_CHUNK_BYTES) -> list[tuple[int, int]]:
    """(offset, length) registration pieces covering ``nbytes`` exactly.

    ``layout`` is the pool's (base, row_bytes, nbytes) triples, contiguous
    from zero (run_kv._harmonize). Each is cut into pieces of the largest
    multiple of its row_bytes at most ``cap``; without a layout the pool is
    registered whole."""
    if not layout:
        return [(0, nbytes)]
    spans = []
    for base, packed, region_nbytes in layout:
        chunk = max(cap // packed, 1) * packed
        spans.extend((base + off, min(chunk, region_nbytes - off))
                     for off in range(0, region_nbytes, chunk))
    covered = sum(length for _, length in spans)
    if covered < nbytes:  # tail the layout does not describe
        spans.append((covered, nbytes - covered))
    return spans


class NIXLBackend(KVBackend):
    name = "nixl"
    maturity = "production"

    def __init__(self, args, role, device):
        # vLLM sets this before nixl (and so UCX) loads (nixl_utils.py).
        os.environ.setdefault("UCX_RCACHE_MAX_UNRELEASED", "1024")
        from nixl._api import nixl_agent, nixl_agent_config

        self.library_version = library_version(("nixl-cu13", "nixl", "nixl-cu12"))
        # The registry pin run_kv hands to UCX_NET_DEVICES for this case;
        # None means UCX chose among the operator inventory itself.
        self.nic_filter = args.kv_device or None
        # The network profile marks EFA pools; there FI_PROVIDER=efa is already set.
        plugin = "LIBFABRIC" if os.environ.get("COLLX_RDMA_FABRIC") == "efa" else "UCX"
        self.transport = plugin.lower()
        if plugin == "UCX":
            config = nixl_agent_config(num_threads=4, capture_telemetry=True)
            self.engine_config = {"backends": [plugin], "num_threads": 4}
        else:
            config = nixl_agent_config(backends=[plugin], capture_telemetry=True)
            self.engine_config = {"backends": [plugin], "num_threads": 0}
        self._agent = nixl_agent(role, config)
        self._handles = []
        self._pool = self._bulk = self._peer = None
        self._local_rows = self._remote_rows = None
        self._row_bytes = 0

    def register(self, pool, bulk, row_bytes: int, reg_layout=None) -> None:
        self._pool, self._bulk, self._row_bytes = pool, bulk, row_bytes
        entries = [(pool.ptr + off, length, pool.device, f"pool{i}")
                   for i, (off, length) in
                   enumerate(reg_spans(pool.nbytes, reg_layout))]
        # bulk rides one whole-request descriptor, so it can never be split;
        # BULK_CAP bounds it.
        entries.append((bulk.ptr, bulk.nbytes, bulk.device, "bulk"))
        reg = self._agent.get_reg_descs(entries, mem_type="cuda")
        if self._agent.register_memory(reg) is None:
            raise RuntimeError("nixl memory registration failed")
        self._local_rows = self._agent.prep_xfer_dlist(
            "NIXL_INIT_AGENT", self._row_descs(pool.ptr, pool.nbytes, pool.device),
            mem_type="cuda")

    def _row_descs(self, base: int, nbytes: int, dev: int) -> np.ndarray:
        rows = nbytes // self._row_bytes
        out = np.empty((rows, 3), dtype=np.uint64)
        out[:, 0] = np.uint64(base) + np.arange(rows, dtype=np.uint64) * np.uint64(self._row_bytes)
        out[:, 1] = self._row_bytes
        out[:, 2] = dev
        return out

    def publish(self) -> dict:
        return {
            "agent": bytes(self._agent.get_agent_metadata()),
            "pool_base": self._pool.ptr,
            "pool_nbytes": self._pool.nbytes,
            "bulk_base": self._bulk.ptr,
            "dev": self._pool.device,
        }

    def connect(self, peer: dict) -> None:
        self._peer = peer
        remote = self._agent.add_remote_agent(peer["agent"])
        self._remote_name = remote.decode() if isinstance(remote, (bytes, bytearray)) else str(remote)
        self._remote_rows = self._agent.prep_xfer_dlist(
            self._remote_name,
            self._row_descs(peer["pool_base"], peer["pool_nbytes"], peer["dev"]),
            mem_type="cuda")

    def _post_wait(self, make):
        agent = self._agent
        handle = []

        def post():
            h = make()
            handle.append(h)
            self._handles.append(h)
            if agent.transfer(h) == "ERR":
                raise RuntimeError("nixl post failed")

        def wait():
            h = handle[-1]
            while True:
                state = agent.check_xfer_state(h)
                if state == "DONE":
                    return
                if state == "ERR":
                    raise RuntimeError("nixl transfer errored")

        return post, wait

    def make_paged(self, cfg, op, local_rows, remote_rows, request_id: int = 0):
        local = np.asarray(local_rows, dtype=np.int32)
        remote = np.asarray(remote_rows, dtype=np.int32)
        notif = f"{request_id}:2".encode()  # vLLM: f"{remote_req_id}:{world_size}"
        agent, xfer = self._agent, "READ" if op == "pull" else "WRITE"
        return self._post_wait(lambda: agent.make_prepped_xfer(
            xfer, self._local_rows, local, self._remote_rows, remote, notif_msg=notif))

    def make_bulk(self, nbytes, op):
        # No vLLM analogue: one descriptor, built per post like a request.
        local_np = np.array([[self._bulk.ptr, nbytes, self._bulk.device]], dtype=np.uint64)
        remote_np = np.array([[self._peer["bulk_base"], nbytes, self._peer["dev"]]],
                             dtype=np.uint64)
        agent, xfer = self._agent, "READ" if op == "pull" else "WRITE"
        return self._post_wait(lambda: agent.initialize_xfer(
            xfer, agent.get_xfer_descs(local_np, mem_type="cuda"),
            agent.get_xfer_descs(remote_np, mem_type="cuda"), self._remote_name))

    def release(self) -> None:
        for handle in self._handles:
            try:
                self._agent.release_xfer_handle(handle)
            except Exception:
                pass
        self._handles.clear()

    def teardown(self) -> None:
        self.release()
        if self._peer is not None:
            try:
                self._agent.remove_remote_agent(self._remote_name)
            except Exception:
                pass
