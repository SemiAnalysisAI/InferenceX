#!/usr/bin/env python3
"""NIXL adapter: the library Dynamo, vLLM NixlConnector, and SGLang disagg
ship. UCX carries it on verbs fabrics (IB, RoCE); on AWS EFA, which is not a
verbs HCA and has no UCX transport, the wheel's LIBFABRIC plugin carries it
over the host libfabric the cluster's enroot hook mounts. Agent metadata rides
the harness exchange (`add_remote_agent`), not NIXL's TCP listener, so the
adapter needs no port and no listener race.
Remote descriptors are built locally from the peer's published pool base; both
block tables are seed-keyed, the same information a decode worker gets from the
prefill side's block table message. Transfers use NIXL's two-step API: each
side's pool descriptor list is prepped once, and every request is a
`make_prepped_xfer` over its block tables as row indices into those lists.
"""

from __future__ import annotations

import os
import time

import numpy as np

from kv_backend import KVBackend, library_version

# b300's CX NICs refuse cuda registrations somewhere between 7083 and 8847 MiB
# (an ~8 GiB MR wall); UCX surfaces no error and the initiator later segfaults
# in ucp_worker_add_rkey_config resolving the region's rkey. Registering the
# pool in pieces below the wall sidesteps it everywhere; each region is cut on
# its own packed-block grid so no transfer descriptor straddles two pieces.
REG_CHUNK_BYTES = 4 << 30

LOCAL_SIDE = "NIXL_INIT_AGENT"

def reg_spans(nbytes: int, layout,
              cap: int = REG_CHUNK_BYTES) -> list[tuple[int, int]]:
    """(offset, length) registration pieces covering ``nbytes`` exactly.

    ``layout`` is the pool's shared region layout — (base, packed_bytes,
    region_nbytes) triples, contiguous from zero and valid for every planned
    config (run_kv._harmonize). Each region is cut into pieces of the largest
    multiple of its packed_bytes at most ``cap``; without a layout the pool
    is registered whole."""
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


def pool_desc_array(base: int, layout, dev: int) -> np.ndarray:
    """(addr, len, devId) uint64 rows, one per packed block of ``layout``, in layout order."""
    rows = []
    for region_base, packed, region_nbytes in layout:
        blocks = region_nbytes // packed
        region = np.empty((blocks, 3), dtype=np.uint64)
        region[:, 0] = (np.uint64(base + region_base)
                        + np.arange(blocks, dtype=np.uint64) * np.uint64(packed))
        region[:, 1] = packed
        region[:, 2] = dev
        rows.append(region)
    return np.concatenate(rows)


def desc_indices(layout, cfg: dict, tables: dict) -> np.ndarray:
    """Rows of one request's blocks in the pool list (uint32, C-contiguous)."""
    first_row = {}
    row = 0
    for region_base, packed, region_nbytes in layout:
        first_row[region_base] = row
        row += region_nbytes // packed
    return np.ascontiguousarray(np.concatenate([
        np.asarray(tables[region["name"]], dtype=np.int64) + first_row[region["base"]]
        for region in cfg["regions"]]), dtype=np.uint32)


class NIXLBackend(KVBackend):
    name = "nixl"
    maturity = "production"

    def __init__(self, args, role, device):
        from nixl._api import nixl_agent, nixl_agent_config

        self.library_version = library_version(("nixl", "nixl-cu13", "nixl-cu12"))
        # The registry pin run_kv hands to UCX_NET_DEVICES for this case;
        # None means UCX chose among the operator inventory itself.
        self.nic_filter = args.kv_device or None
        # The network profile marks EFA pools; there FI_PROVIDER=efa is already set.
        self.transport = "LIBFABRIC" if os.environ.get("COLLX_RDMA_FABRIC") == "efa" else "UCX"
        # prog thread on, listener off: metadata goes through the harness exchange.
        self._agent = nixl_agent(role, nixl_agent_config(True, False, 0,
                                                         backends=[self.transport]))
        self._handles = []
        self._dlists = []
        self._pool = None
        self._bulk = None
        self._peer = None
        self._layout = None
        self._pool_lists = None
        self._bulk_lists = {}

    def register(self, pool, bulk, reg_layout=None) -> None:
        self._pool, self._bulk, self._layout = pool, bulk, reg_layout
        entries = [(pool.ptr + off, length, pool.device, f"pool{i}")
                   for i, (off, length) in
                   enumerate(reg_spans(pool.nbytes, reg_layout))]
        # bulk rides one whole-request descriptor, so it can never be split;
        # BULK_CAP bounds it.
        entries.append((bulk.ptr, bulk.nbytes, bulk.device, "bulk"))
        reg = self._agent.get_reg_descs(entries, mem_type="cuda")
        if self._agent.register_memory(reg) is None:
            raise RuntimeError("nixl memory registration failed")

    def publish(self) -> dict:
        return {
            "agent": bytes(self._agent.get_agent_metadata()),
            "pool_base": self._pool.ptr,
            "bulk_base": self._bulk.ptr,
            "dev": self._pool.device,
        }

    def connect(self, peer: dict) -> None:
        self._peer = peer
        remote = self._agent.add_remote_agent(peer["agent"])
        self._remote_name = remote.decode() if isinstance(remote, (bytes, bytearray)) else str(remote)

    def _prep(self, side: str, descs: np.ndarray):
        handle = self._agent.prep_xfer_dlist(side, descs, mem_type="cuda")
        self._dlists.append(handle)
        return handle

    def _pool_sides(self):
        if self._pool_lists is None:
            local = pool_desc_array(self._pool.ptr, self._layout, self._pool.device)
            remote = pool_desc_array(self._peer["pool_base"], self._layout, self._peer["dev"])
            self._pool_lists = (self._prep(LOCAL_SIDE, local),
                                self._prep(self._remote_name, remote))
        return self._pool_lists

    def _bulk_sides(self, nbytes: int):
        if nbytes not in self._bulk_lists:
            local = np.array([[self._bulk.ptr, nbytes, self._bulk.device]], dtype=np.uint64)
            remote = np.array([[self._peer["bulk_base"], nbytes, self._peer["dev"]]],
                              dtype=np.uint64)
            self._bulk_lists[nbytes] = (self._prep(LOCAL_SIDE, local),
                                        self._prep(self._remote_name, remote))
        return self._bulk_lists[nbytes]

    def _make(self, sides, local_idx: np.ndarray, remote_idx: np.ndarray, op: str,
              started: float):
        local_side, remote_side = sides
        handle = self._agent.make_prepped_xfer(
            "READ" if op == "pull" else "WRITE",
            local_side, local_idx, remote_side, remote_idx,
        )
        prep_s = time.perf_counter() - started
        self._handles.append(handle)
        agent = self._agent

        def post():
            if agent.transfer(handle) == "ERR":
                raise RuntimeError("nixl post failed")

        def wait():
            while True:
                state = agent.check_xfer_state(handle)
                if state == "DONE":
                    return
                if state == "ERR":
                    raise RuntimeError("nixl transfer errored")

        return post, wait, prep_s

    def make_paged(self, cfg, op, local_tables, remote_tables):
        sides = self._pool_sides()
        start = time.perf_counter()
        return self._make(sides, desc_indices(self._layout, cfg, local_tables),
                          desc_indices(self._layout, cfg, remote_tables), op, start)

    def make_bulk(self, nbytes, op):
        sides = self._bulk_sides(nbytes)
        start = time.perf_counter()
        first = np.zeros(1, dtype=np.uint32)
        return self._make(sides, first, first, op, start)

    def release(self) -> None:
        for handle in self._handles:
            try:
                self._agent.release_xfer_handle(handle)
            except Exception:
                pass
        self._handles.clear()

    def teardown(self) -> None:
        self.release()
        for handle in self._dlists:
            try:
                self._agent.release_dlist_handle(handle)
            except Exception:
                pass
        if self._peer is not None:
            try:
                self._agent.remove_remote_agent(self._remote_name)
            except Exception:
                pass
