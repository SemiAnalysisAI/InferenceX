#!/usr/bin/env python3
"""Backend contract for the KV-cache transfer suite.

A KV backend is a point-to-point transfer library adapter. The harness owns the
data (pool/bulk tensors, pattern fill, verification) and the protocol (which
rank is target, lockstep barriers); the adapter owns registration, connection,
and posting. All transfers are one-sided from the initiator, so completion is
host-visible on the initiator and timing is wall clock around post→complete —
no CUDA events, because no local kernel participates in an RDMA READ/WRITE.

Roles: rank 0 = target (owns the remote pool), rank 1 = initiator (posts every
transfer). `pull` (READ) models a decode worker fetching KV from prefill —
vLLM's NixlConnector; `push` (WRITE) models prefill writing into decode —
SGLang's disagg sender. The measured quantity is the same one-sided completion
either way; only the payload's direction differs.

Exchange of connection payloads goes through the harness-provided `exchange`
callable (a torch.distributed object broadcast), so adapters never open their
own side channels beyond what their library requires.
"""

from __future__ import annotations

import time


class KVBackend:
    """One transfer library on one rank. Subclasses implement the five hooks."""

    name = "abstract"
    #: maturity mirrors EPBackend.maturity ("production" | "candidate").
    maturity = "candidate"
    library_version: str | None = None

    def __init__(self, args, role: str, device) -> None:
        self.args = args
        self.role = role
        self.device = device

    # -- lifecycle ------------------------------------------------------------
    def register(self, pool, bulk) -> None:
        """Register the pool + bulk tensors with the library."""
        raise NotImplementedError

    def publish(self) -> dict:
        """Payload the peer needs to reach this rank (addresses, packed descs)."""
        raise NotImplementedError

    def connect(self, peer: dict) -> None:
        """Consume the peer's payload; after this, transfers may be prepared."""
        raise NotImplementedError

    def teardown(self) -> None:  # pragma: no cover - adapter-specific
        pass

    # -- transfers (initiator only) --------------------------------------------
    def make_paged(self, cfg: dict, op: str, local_table, remote_table):
        """Return (transfer_once, prep_seconds) for one request's paged KV.

        ``transfer_once()`` posts the whole descriptor list and blocks until the
        transfer completes — exactly one request handoff. Preparation cost
        (descriptor build + handle creation) is amortized by engines through
        prepped-handle reuse, so it is reported separately, never inside the
        timed transfer.
        """
        raise NotImplementedError

    def make_bulk(self, nbytes: int, op: str):
        """Return (transfer_once, prep_seconds) for one contiguous transfer of
        ``nbytes`` — the single-descriptor wire-speed ceiling row."""
        raise NotImplementedError


def time_transfers(transfer_once, warmup: int, reps: int) -> list[float]:
    """Wall-clock ms per completed transfer, warmups dropped."""
    samples = []
    for rep in range(warmup + reps):
        start = time.perf_counter()
        transfer_once()
        if rep >= warmup:
            samples.append((time.perf_counter() - start) * 1e3)
    return samples
