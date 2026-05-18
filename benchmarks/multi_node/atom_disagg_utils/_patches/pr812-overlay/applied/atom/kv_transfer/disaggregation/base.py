# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Abstract base classes for KV cache connectors.

These interfaces decouple the engine from any specific transfer backend
(e.g. MoRIIO/RDMA, NCCL P2P).  Concrete implementations live in
separate modules and are registered via :class:`KVConnectorFactory`.

Two roles are defined:

- **Worker-side** (:class:`KVConnectorBase`): runs inside each TP rank,
  handles RDMA / network I/O, and reports transfer completion.
- **Scheduler-side** (:class:`KVConnectorSchedulerBase`): runs in the
  scheduler process, manages transfer lifecycle and metadata.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from atom.kv_transfer.disaggregation.types import ConnectorMetadata


class KVConnectorBase(ABC):
    """Worker-side KV connector interface (one instance per TP rank)."""

    is_producer: bool

    @abstractmethod
    def register_kv_caches(
        self, kv_caches: dict[str, Any], transfer_tensors: Any = None
    ) -> None:
        """Register local KV cache tensors for remote access.

        Called once after model loading and KV cache allocation.
        """
        ...

    @abstractmethod
    def start_load_kv(self, metadata: ConnectorMetadata) -> None:
        """Initiate async KV loads for pending receive requests.

        Called by the worker each engine step.
        """
        ...

    @abstractmethod
    def get_finished(self) -> tuple[set, set]:
        """Return ``(done_sending, done_recving)`` request ID sets.

        Called by the worker each engine step to report transfer status.
        """
        ...

    def get_finished_recv_blocks(self) -> list[int]:
        """Return block IDs from recently completed receives for GPU memory fence.

        RDMA writes to HBM may not be immediately visible to GPU compute
        kernels. Connectors using RDMA should override this to return
        blocks that need a GPU-side read-write cycle to ensure coherence.
        """
        return []


class KVConnectorSchedulerBase(ABC):
    """Scheduler-side KV connector interface."""

    is_producer: bool

    @abstractmethod
    def get_num_new_matched_tokens(self, seq: Any) -> tuple[int, bool]:
        """Check if *seq* needs remote KV prefill.

        Returns:
            ``(num_tokens, needs_async_load)``
        """
        ...

    @abstractmethod
    def build_connector_meta(self) -> ConnectorMetadata:
        """Build a metadata snapshot of pending transfer requests."""
        ...

    @abstractmethod
    def update_state_after_alloc(self, seq: Any) -> None:
        """Update internal state after the scheduler allocates blocks."""
        ...

    @abstractmethod
    def request_finished(self, seq: Any) -> None:
        """Populate KV transfer output metadata when a request completes."""
        ...
