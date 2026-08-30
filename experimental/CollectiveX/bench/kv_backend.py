#!/usr/bin/env python3
"""Backend contract for the KV-cache transfer suite.

The harness owns the data (kv_pool pools, pattern fill, verification) and the
protocol (rank 0 = target, rank 1 = initiator, lockstep barriers); an adapter
owns registration, connection, and posting. Transfers are one-sided from the
initiator, so completion is host-visible and timing is wall clock around
post-to-complete; no CUDA events, because no local kernel participates.
`pull` (READ) is the vLLM NixlConnector shape, `push` (WRITE) the SGLang disagg
shape; the measured quantity is the same completion either way. Connection
payloads ride the harness object exchange, never adapter side channels.
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
    def make_paged(self, cfg: dict, op: str, local_tables, remote_tables):
        """Return (post, wait, prep_seconds) for one request's paged KV.

        ``post()`` submits the whole descriptor list asynchronously; ``wait()``
        blocks until it completes — split so a batch of requests overlaps like
        a decode step admitting several requests at once. Preparation cost
        (descriptor build + handle creation) is amortized by engines through
        prepped-handle reuse, so it is reported separately, never inside the
        timed transfer.
        """
        raise NotImplementedError

    def make_bulk(self, nbytes: int, op: str):
        """Return (post, wait, prep_seconds) for one contiguous transfer of
        ``nbytes`` — the single-descriptor contiguous baseline row (logical
        payload over host-observed completion; not a proven physical wire
        rate — backends may split large operations internally)."""
        raise NotImplementedError


def time_bursts(transfers, warmup: int, reps: int) -> tuple[list[float], list[float]]:
    """(burst_ms, request_ms), warmups dropped. ``transfers`` is a list of
    (post, wait) pairs — one per request. A burst posts every request, then
    drains the waits in posting order; burst_ms is post-of-first to
    completion-of-last, and request_ms records each individual request's
    host-observed completion offset from the burst start. Because the waits
    drain in posting order, a request's mark upper-bounds its true completion
    (a later request that finished early is observed at its wait's turn)."""
    burst_ms: list[float] = []
    request_ms: list[float] = []
    for rep in range(warmup + reps):
        start = time.perf_counter()
        for post, _ in transfers:
            post()
        marks = []
        for _, wait in transfers:
            wait()
            marks.append((time.perf_counter() - start) * 1e3)
        if rep >= warmup:
            burst_ms.append(marks[-1])
            request_ms.extend(marks)
    return burst_ms, request_ms
