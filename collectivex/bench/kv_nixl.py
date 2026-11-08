#!/usr/bin/env python3
"""NIXL adapter, driven the way vLLM's NixlConnector drives it."""

from __future__ import annotations

import os
import time

import numpy as np

import kv_workload


class NIXLBackend:
    maturity = "production"

    def __init__(self, args, role, device, rank: int = 0):
        # vLLM sets this before nixl (and so UCX) loads (nixl_utils.py).
        os.environ.setdefault("UCX_RCACHE_MAX_UNRELEASED", "1024")
        from nixl._api import nixl_agent, nixl_agent_config

        self.library_version = kv_workload.library_version(("nixl-cu13", "nixl", "nixl-cu12"))
        # The network profile marks EFA pools; there FI_PROVIDER=efa is already set.
        plugin = "LIBFABRIC" if os.environ.get("COLLX_RDMA_FABRIC") == "efa" else "UCX"
        self.transport = plugin.lower()
        self.library_config = {"engine_contract": "vllm", "connector": "NixlConnector",
                               "backends": [plugin]}
        if plugin == "UCX":
            config = nixl_agent_config(num_threads=4, capture_telemetry=True)
            self.library_config["num_threads"] = 4
        else:
            config = nixl_agent_config(backends=[plugin], capture_telemetry=True)
        self._agent = nixl_agent(f"{role}-{rank}", config)
        # vLLM checks a transfer once per engine step; 0 polls continuously (the library's own
        # completion time). COLLX_KV_POLL_MS sets a step-like interval for comparison runs.
        self._poll_s = float(os.environ.get("COLLX_KV_POLL_MS") or 0) / 1e3
        self.library_config["poll_ms"] = self._poll_s * 1e3
        self._telemetry: list[tuple[float, float]] = []
        self._handles = []
        self._pool = self._bulk = self._peer = None
        self._local_rows = self._remote_rows = None
        self._row_bytes = self._rows = 0
        preset = args.workload_name.removeprefix("kv-")
        self._kinds = kv_workload.desc_kinds({"preset": preset, "row_bytes": 0})

    def register(self, pool, bulk, row_bytes: int) -> None:
        self._pool, self._bulk, self._row_bytes = pool, bulk, row_bytes
        self._rows = pool.nbytes // row_bytes
        reg = self._agent.get_reg_descs([(pool.ptr, pool.nbytes, pool.device, "pool"),
                                         (bulk.ptr, bulk.nbytes, bulk.device, "bulk")],
                                        mem_type="cuda")
        if self._agent.register_memory(reg) is None:
            raise RuntimeError("nixl memory registration failed")
        self._local_rows = self._agent.prep_xfer_dlist(
            "NIXL_INIT_AGENT", self._row_descs(pool.ptr, pool.nbytes, pool.device),
            mem_type="cuda")

    def _row_descs(self, base: int, nbytes: int, dev: int) -> np.ndarray:
        """The prepped list vLLM builds once per side: every descriptor kind of every row,
        kind-major (id = kind x rows + row). A whole-row preset has the one kind."""
        rows = nbytes // self._row_bytes
        starts = np.uint64(base) + np.arange(rows, dtype=np.uint64) * np.uint64(self._row_bytes)
        out = np.empty((len(self._kinds), rows, 3), dtype=np.uint64)
        for index, (offset, size) in enumerate(self._kinds):
            out[index, :, 0] = starts + np.uint64(offset)
            out[index, :, 1] = size or self._row_bytes
        out[:, :, 2] = dev
        return out.reshape(-1, 3)

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
                    # NIXL's own timing of this transfer, the figures vLLM exports as
                    # vllm:nixl_xfer_time_seconds and vllm:nixl_post_time_seconds.
                    stats = agent.get_xfer_telemetry(h)
                    self._telemetry.append((stats.xferDuration / 1e3, stats.postDuration / 1e3))
                    return
                if state == "ERR":
                    raise RuntimeError("nixl transfer errored")
                if self._poll_s:
                    time.sleep(self._poll_s)

        return post, wait

    def request_entries(self, cfg, local_rows, remote_rows):
        return kv_workload.nixl_entries(cfg, local_rows, remote_rows)

    def make_burst(self, cfg, op, requests):
        """One prepped transfer per request, as vLLM posts them."""
        return [self.make_paged(cfg, op, local, remote, request_id)
                for local, remote, request_id in requests]

    def make_paged(self, cfg, op, local_rows, remote_rows, request_id: int = 0):
        kinds, positions = kv_workload.request_descs(cfg)
        local = (kinds * self._rows + np.asarray(local_rows)[positions]).astype(np.int32)
        remote = (kinds * self._rows + np.asarray(remote_rows)[positions]).astype(np.int32)
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

    def release(self) -> list[tuple[float, float]]:
        """Release the burst's handles; returns its transfers' (xfer ms, post ms) as NIXL timed them."""
        telemetry, self._telemetry = self._telemetry, []
        for handle in self._handles:
            try:
                self._agent.release_xfer_handle(handle)
            except Exception:
                pass
        self._handles.clear()
        return telemetry

    def teardown(self) -> None:
        self.release()
        if self._peer is not None:
            try:
                self._agent.remove_remote_agent(self._remote_name)
            except Exception:
                pass
