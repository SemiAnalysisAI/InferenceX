# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Shared type definitions for the KV cache disaggregation subsystem.

This module is the single source of truth for data structures exchanged
between the scheduler, engine core, worker-side connectors, and the
KV output aggregator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

EngineId = str
ReqId = str
TransferId = int

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class KVTransferRegion:
    """One RDMA-registerable tensor region."""

    base_addr: int
    total_bytes: int
    unit_bytes: int  # bytes per block (block-indexed) or per slot (slot-indexed)


@dataclass
class KVTransferTensors:
    block_regions: list[KVTransferRegion]
    slot_regions: list[KVTransferRegion]
    num_blocks: int
    num_slots: int = 0
    staging_region: KVTransferRegion | None = None
    staging_pool_size: int = 0
    gather_slot: Callable[[int, int], None] | None = None
    scatter_slot: Callable[[int, int], None] | None = None


@dataclass
class KVConnectorOutput:
    """Per-worker snapshot of finished KV cache transfers.

    Each TP worker produces one of these per scheduler step.  The
    :class:`KVOutputAggregator` combines them to determine which
    request IDs have finished on *all* workers.

    Attributes:
        finished_sending: Request IDs whose KV send completed on this worker.
        finished_recving: Request IDs whose KV receive completed on this worker.
        expected_finished_count: How many finished notifications should be
            expected per request (used by the aggregator).
    """

    finished_sending: set[str] = field(default_factory=set)
    finished_recving: set[str] = field(default_factory=set)
    expected_finished_count: int = 0

    def is_empty(self) -> bool:
        """Return True if no transfers finished on this worker."""
        return not self.finished_sending and not self.finished_recving

    def __repr__(self) -> str:
        return (
            f"KVConnectorOutput(sending={self.finished_sending}, "
            f"recving={self.finished_recving})"
        )


@dataclass
class ReqMeta:
    """Per-request metadata needed for KV cache transfer.

    Captures both local and remote block locations together with
    networking information to reach the remote engine.
    """

    local_block_ids: list[int]
    remote_block_ids: list[int]
    remote_host: str
    remote_port: int
    remote_handshake_port: int
    remote_engine_id: str
    tp_size: int
    remote_dp_size: int
    remote_dp_rank: int = 0
    transfer_id: int = 0
    local_slot_index: int = -1


@dataclass
class RemoteAllocInfo:
    """Allocation information received from the remote (decode) side."""

    block_ids: list[int] = field(default_factory=list)
    writes_done: int = 0
    decode_dp_rank: int = 0
    transfer_offset: tuple[list[int], list[int], list[int]] | None = None


@dataclass
class RemoteMeta:
    """Minimal metadata describing a remote block allocation."""

    block_ids: list[int]
    host: str
    port: int
    engine_id: str
    request_id: str


class ConnectorMetadata:
    """Snapshot of pending KV transfer requests, passed from scheduler to workers.

    The scheduler populates this each step with new receive / save requests,
    and the worker-side connector consumes it in ``start_load_kv``.
    """

    def __init__(self) -> None:
        self.reqs_to_recv: dict[ReqId, ReqMeta] = {}
        self.reqs_to_save: dict[ReqId, ReqMeta] = {}
        self.reqs_to_send: dict[ReqId, float] = {}
        self.reqs_in_batch: set[ReqId] = set()
        self.reqs_not_processed: set[ReqId] = set()
        self.request_id_to_transfer_id: dict[ReqId, int] = {}

    @staticmethod
    def _build_req_meta(
        req_id: ReqId,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
    ) -> ReqMeta:
        """Construct a :class:`ReqMeta` from raw transfer parameters."""
        return ReqMeta(
            local_block_ids=local_block_ids,
            remote_block_ids=kv_transfer_params["remote_block_ids"],
            remote_engine_id=kv_transfer_params["remote_engine_id"],
            remote_host=kv_transfer_params["remote_host"],
            remote_port=kv_transfer_params["remote_port"],
            remote_handshake_port=kv_transfer_params["remote_handshake_port"],
            remote_dp_size=kv_transfer_params.get("remote_dp_size", 1),
            remote_dp_rank=kv_transfer_params.get("remote_dp_rank", 0),
            tp_size=(
                kv_transfer_params.get("tp_size")
                if "tp_size" in kv_transfer_params
                else kv_transfer_params.get("remote_tp_size", 1)
            ),
            transfer_id=kv_transfer_params.get("transfer_id", 0),
            local_slot_index=kv_transfer_params.get("local_slot_index", -1),
        )

    def add_new_req_to_save(
        self,
        request_id: ReqId,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
    ) -> None:
        self.reqs_to_save[request_id] = self._build_req_meta(
            request_id, local_block_ids, kv_transfer_params
        )

    def add_new_req_to_recv(
        self,
        request_id: ReqId,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
    ) -> None:
        self.reqs_to_recv[request_id] = self._build_req_meta(
            request_id, local_block_ids, kv_transfer_params
        )
