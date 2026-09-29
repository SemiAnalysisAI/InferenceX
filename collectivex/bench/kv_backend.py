#!/usr/bin/env python3
"""Backend contract for the KV-cache transfer suite.

The harness owns the data (kv_pool pools, pattern fill, verification) and the
protocol (rank 0 = target, rank 1 = initiator, lockstep barriers); an adapter
owns registration, connection, and posting, in the shape vLLM's connector for
that library posts. Transfers are one-sided from the initiator, so completion
is host-visible and timing is wall clock around post-to-complete; no CUDA
events, because no local kernel participates. `pull` is READ (vLLM's default
NixlConnector path), `push` is WRITE (vLLM's NIXL push path and its Mooncake
connector); the measured quantity is the same completion either way.
Connection payloads ride the harness object exchange, never side channels.
"""

from __future__ import annotations

import time

import kv_workload


class KVBackend:
    """One transfer library on one rank, constructed as ``Backend(args, role,
    device)``. Subclasses implement register/publish/connect/make_paged/
    make_bulk; request_entries and make_burst have vLLM-NIXL defaults."""

    name = "abstract"
    #: maturity mirrors EPBackend.maturity ("production" | "candidate").
    maturity = "candidate"
    library_version: str | None = None
    #: the engine NIC filter this case ran under; None = library/UCX choice.
    nic_filter: str | None = None
    #: the wire transport under the library, in one vocabulary across
    #: backends: "ucx", "libfabric", or "verbs".
    transport: str | None = None
    #: the engine knobs the row ran under (threads, workers, QPs, ...).
    engine_config: dict = {}

    # -- lifecycle ------------------------------------------------------------
    def register(self, pool, bulk, row_bytes: int, reg_layout=None) -> None:
        """Register the pool + bulk buffers with the library. ``row_bytes`` is
        the pool's block-row size; ``reg_layout`` is its (base, row_bytes,
        nbytes) layout (run_kv._harmonize), which an adapter may use to split
        an oversized registration on the row grid; ignoring it is valid."""
        raise NotImplementedError

    def publish(self) -> dict:
        """Payload the peer needs to reach this rank (addresses, packed descs)."""
        raise NotImplementedError

    def connect(self, peer: dict) -> None:
        """Consume the peer's payload; after this, transfers may be prepared."""
        raise NotImplementedError

    def release(self) -> None:
        """Drop the transfer handles of the burst just completed (initiator)."""

    def teardown(self) -> None:  # pragma: no cover - adapter-specific
        pass

    # -- transfers (initiator only) --------------------------------------------
    def request_entries(self, cfg: dict, local_rows, remote_rows):
        """(local offsets, remote offsets, sizes), bytes from each pool base,
        of what one request moves. Default: vLLM NIXL's whole-row descriptors."""
        sizes = kv_workload.desc_sizes(cfg)
        return (kv_workload.page_offsets(cfg, local_rows),
                kv_workload.page_offsets(cfg, remote_rows), sizes)

    def make_paged(self, cfg: dict, op: str, local_rows, remote_rows, request_id: int = 0):
        """Return (post, wait) for one request. ``post()`` builds the
        request's transfer (handle creation is per request in vLLM, so it is
        timed) and submits it; ``wait()`` blocks until it completes."""
        raise NotImplementedError

    def make_burst(self, cfg: dict, op: str, requests) -> list:
        """(post, wait) pairs for a burst of ``requests`` = [(local_rows,
        remote_rows, request_id)]. Default: one transfer per request."""
        return [self.make_paged(cfg, op, local, remote, request_id)
                for local, remote, request_id in requests]

    def make_bulk(self, nbytes: int, op: str):
        """Return (post, wait) for one contiguous transfer of ``nbytes`` — the
        single-descriptor contiguous baseline row (logical payload over
        host-observed completion; not a proven physical wire rate)."""
        raise NotImplementedError


def library_version(dists, module=None) -> str | None:
    """The first installed distribution's version, else ``module.__version__``."""
    import importlib.metadata as md

    for name in dists:
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            pass
    return getattr(module, "__version__", None)


def spans(n: int, cap: int) -> list[tuple[int, int]]:
    """[start, end) pieces of at most ``cap`` covering ``range(n)``."""
    return [(i, min(i + cap, n)) for i in range(0, n, cap)]


def time_bursts(build, warmup: int, reps: int, settle=None,
                rep0: int = 0) -> tuple[list[float], list[float]]:
    """(burst_ms, request_ms), warmups dropped. ``build(rep)`` returns the
    burst's (post, wait) pairs and runs untimed; the burst posts every
    request, then drains the waits in posting order. burst_ms is post-of-first
    to completion-of-last; request_ms records each request's host-observed
    completion offset from the burst start (an upper bound, since waits drain
    in order). ``settle()`` runs untimed after each burst (handle release)."""
    burst_ms: list[float] = []
    request_ms: list[float] = []
    for rep in range(warmup + reps):
        transfers = build(rep0 + rep)
        start = time.perf_counter()
        for post, _ in transfers:
            post()
        marks = []
        for _, wait in transfers:
            wait()
            marks.append((time.perf_counter() - start) * 1e3)
        if settle is not None:
            settle()
        if rep >= warmup:
            burst_ms.append(marks[-1])
            request_ms.extend(marks)
    return burst_ms, request_ms
