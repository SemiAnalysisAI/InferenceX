#!/usr/bin/env python3
"""NIXL (UCX) adapter: the library Dynamo, vLLM NixlConnector, and SGLang
disagg ship. Agent metadata rides the harness exchange (`add_remote_agent`),
not NIXL's TCP listener, so the adapter needs no port and no listener race.
Remote descriptors are built locally from the peer's published pool base; both
block tables are seed-keyed, the same information a decode worker gets from the
prefill side's block table message.
"""

from __future__ import annotations

import time

import numpy as np

import kv_workload
from kv_backend import KVBackend


class NIXLBackend(KVBackend):
    name = "nixl"
    maturity = "production"

    def __init__(self, args, role, device):
        super().__init__(args, role, device)
        from nixl._api import nixl_agent, nixl_agent_config

        try:
            import importlib.metadata as md

            for dist_name in ("nixl", "nixl-cu13", "nixl-cu12"):
                try:
                    self.library_version = md.version(dist_name)
                    break
                except md.PackageNotFoundError:
                    continue
        except Exception:
            self.library_version = None
        # prog thread on, listener off: metadata goes through the harness exchange.
        self._agent = nixl_agent(role, nixl_agent_config(True, False, 0, backends=["UCX"]))
        self._handles = []
        self._pool = None
        self._bulk = None
        self._peer = None

    def register(self, pool, bulk) -> None:
        self._pool, self._bulk = pool, bulk
        reg = self._agent.get_reg_descs(
            [(pool.ptr, pool.nbytes, pool.device, "pool"),
             (bulk.ptr, bulk.nbytes, bulk.device, "bulk")], mem_type="cuda")
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

    def _make(self, local_np: np.ndarray, remote_np: np.ndarray, op: str):
        start = time.perf_counter()
        local_descs = self._agent.get_xfer_descs(local_np, mem_type="cuda")
        remote_descs = self._agent.get_xfer_descs(remote_np, mem_type="cuda")
        handle = self._agent.initialize_xfer(
            "READ" if op == "pull" else "WRITE",
            local_descs, remote_descs, self._remote_name,
        )
        prep_s = time.perf_counter() - start
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
        local_np = kv_workload.desc_array(self._pool.ptr, cfg, local_tables, self._pool.device)
        remote_np = kv_workload.desc_array(self._peer["pool_base"], cfg, remote_tables,
                                           self._peer["dev"])
        return self._make(local_np, remote_np, op)

    def make_bulk(self, nbytes, op):
        local_np = np.array([[self._bulk.ptr, nbytes, self._bulk.device]], dtype=np.uint64)
        remote_np = np.array([[self._peer["bulk_base"], nbytes, self._peer["dev"]]],
                             dtype=np.uint64)
        return self._make(local_np, remote_np, op)

    def teardown(self) -> None:
        for handle in self._handles:
            try:
                self._agent.release_xfer_handle(handle)
            except Exception:
                pass
        if self._peer is not None:
            try:
                self._agent.remove_remote_agent(self._remote_name)
            except Exception:
                pass
