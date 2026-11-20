# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sticky credit router with fair load balancing.

Routes credits to workers: sticky routing for multi-turn sessions,
least-loaded selection for first turns. Lock-free via asyncio serialization.

Terminology:
    session: A unique execution of a conversation template, identified by
        x_correlation_id (UUID). All turns in a session route to the same worker.
    conversation_id: Template ID from the dataset (can be reused across sessions).

Includes:
- WorkerLoad: Worker load tracking for fair load balancing
- StickyCreditRouter: Main router class
"""

import asyncio
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from aiperf.common.enums import CommAddress, ConversationBranchMode
from aiperf.common.mixins import CommunicationMixin
from aiperf.common.protocols import (
    StreamingPullClientProtocol,
    StreamingRouterClientProtocol,
)
from aiperf.config.comm import BaseZMQCommunicationConfig, ZMQDualBindConfig
from aiperf.credit.messages import (
    CancelCredits,
    CreditReturn,
    FirstToken,
    StreamingContent,
    WorkerReady,
    WorkerShutdown,
    WorkerToRouterMessage,
)
from aiperf.credit.structs import Credit

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

# =============================================================================
# Data Models
# =============================================================================


@dataclass(slots=True)
class _StickyEntry:
    """Sticky-routing state for a root or SPAWN-child correlation id.

    Tracks which worker owns the session and a refcount so FORK children that
    pin themselves to the parent's worker can keep the entry alive past the
    parent's own final turn. ``parent_final_seen`` records whether the owning
    session has finished its final turn; the entry is popped only once both
    ``ref_count`` hits zero and that flag is set.
    """

    worker_id: str
    ref_count: int = 1
    parent_final_seen: bool = False


@dataclass(slots=True)
class WorkerLoad:
    """Worker load tracking for fair load balancing.

    Note on virtual_sent_credits vs total_sent_credits:
        - total_sent_credits: Actual count of credits sent (for metrics/debugging)
        - virtual_sent_credits: Used for fairness tie-breaking, initialized to
          average when worker joins mid-benchmark to prevent "thundering herd"
          where a new worker with 0 credits gets all requests.

    Note on active_sessions:
        - active_sessions and active_session_ids only represent the number of sticky sessions assigned
          to the worker. An entry is held from the session's first non-final turn (or a final turn that
          forks) until it is released: when SENDING the final turn, or, for a forking final turn, when its
          FORK children release it.
        - Roots and SPAWN children each count as a session; FORK children share their parent's.
    """

    worker_id: str
    total_sent_credits: int = 0
    virtual_sent_credits: int = (
        0  # For fairness comparison (initialized to avg on join)
    )
    total_completed_credits: int = 0
    total_cancelled_credits: int = 0
    total_errors_reported: int = 0
    in_flight_credits: int = 0
    active_credit_ids: set[int] = field(default_factory=set)
    active_sessions: int = 0  # Sticky sessions assigned to this worker
    active_session_ids: set[str] = field(default_factory=set)
    last_sent_at_ns: int = (
        0  # For tie-breaking (guaranteed unique in single-threaded asyncio)
    )


# ==============================================================================
# Credit Router Protocol
# ==============================================================================


@runtime_checkable
class CreditRouterProtocol(Protocol):
    """Protocol for routing credits to workers.

    Decouples credit issuing strategies from routing implementation.
    Enables mocking for tests and alternative routing strategies.
    """

    async def send_credit(self, credit: Credit) -> None:
        """Send credit to worker via routing strategy.

        Args:
            credit: Credit to send to worker
        """
        ...

    async def wait_for_workers(self, timeout: float) -> None:
        """Wait for the router to observe at least one registered worker.

        Best-effort startup gate, not an absolute guarantee: a worker can
        unregister again after this returns. Raises on timeout.

        Args:
            timeout: Seconds to wait for the first worker before giving up.
        """
        ...

    async def cancel_all_credits(self) -> None:
        """Cancel all in-flight credits.

        Used during phase timeout or system shutdown.
        """
        ...

    def mark_credits_complete(self) -> None:
        """Mark that all credits have been issued and returned.

        Called by orchestrator when benchmark completes normally.
        Suppresses warnings about orphaned sessions during shutdown.
        """
        ...

    def set_return_callback(
        self,
        callback: Callable[[str, CreditReturn], Awaitable[None]],
    ) -> None:
        """Register callback for credit returns.

        Args:
            callback: Async function called when credit returns.
                     Signature: (worker_id: str, message: CreditReturn) -> None
        """
        ...

    def set_first_token_callback(
        self,
        callback: Callable[[FirstToken], Awaitable[None]],
    ) -> None:
        """Register callback for first token events (prefill concurrency release).

        Args:
            callback: Async function called when first token is received.
                     Signature: (message: FirstToken) -> None
        """
        ...

    def set_streaming_content_callback(
        self, callback: Callable[[StreamingContent], Awaitable[None]]
    ) -> None:
        """Register the phase-coverage observer for in-flight content."""
        ...


# =============================================================================
# Sticky Credit Router
# =============================================================================


class StickyCreditRouter(CommunicationMixin):
    """Routes credits to workers with sticky sessions and fair load balancing.

    All messages between the Worker and TimingManager service flow through the CreditRouter.

    IMPORTANT:
        - This class has been highly optimized for performance, as it is a hot path.
        - Please be careful when making changes to ensure performance is not degraded.
        - All operations are atomic because there are no await calls between reads and writes.
        - Methods are intentionally large/inlined to avoid function call overhead in the hot path.
        - The class is designed for single-threaded asyncio use only.

    Credit Routing:
        - First turn → least-loaded worker (creates sticky session).
        - Subsequent turns → same worker via sticky session lookup.
        - Final turn → cleanup sticky session.
        - FORK children → parent's worker via the parent's refcounted entry.
        - SPAWN children → parent's worker if its entry is live (else least-loaded),
          then pinned under their own id until their final turn.

    Load Balancing:
        - Least-loaded worker selection for new sessions using fair load balancing
            - Determined by the worker(s) with the fewest in-flight credits.
        - Tie-breaking for multiple workers in this order:
            - `active_sessions`: Prefer workers with fewer committed multi-turn sessions
            - `virtual_sent_credits`: Prefer workers with fewer historical credits (virtual to handle
                late-joining workers fairly - they start at average, not zero)
            - `last_sent_at_ns`: Prefer workers with oldest send time (LRU-like fairness)

    Credit Returns:
        - All CreditReturns and FirstTokens flow through the CreditRouter and
          are forwarded via callbacks that are directly awaited for responsiveness.

    Lock-free:
        - Ensure there are no await calls in critical paths.

    Hot path complexity:
        - sticky session lookup is O(1)
        - min load tracking/lookup is O(1)
        - load balancing for new sessions is O(k) where k = workers tied at min load
        - credit sent/returned tracking is O(1)

    Cold path complexity:
        - worker register/unregister is O(n) where n = number of workers
        - credit cancellation is O(n × k) where n = number of workers, k = average in-flight credits per worker
    """

    def _init_credit_channels(
        self, comm_config: BaseZMQCommunicationConfig | None
    ) -> None:
        """Bind the credit dispatch ROUTER and the dedicated credit-return PULL.

        Dispatch (Credit/CancelCredits) goes router->worker over CREDIT_ROUTER;
        CreditReturn/FirstToken fan in worker->router over a separate PUSH/PULL
        channel (CREDIT_RETURN), so neither socket is bidirectional. In dual-bind
        (k8s controller) mode each also binds its TCP address so remote worker
        pods can connect; controller-side services otherwise use IPC.
        """
        dual_bind = (
            isinstance(comm_config, ZMQDualBindConfig)
            and not comm_config.controller_host
        )

        dispatch_bind = (
            comm_config.credit_router_tcp_bind_address if dual_bind else None
        )
        if dispatch_bind:
            self.info(
                f"Dual-bind mode: credit router will also bind to {dispatch_bind}"
            )
        self._router_client: StreamingRouterClientProtocol = (
            self.comms.create_streaming_router_client(
                address=CommAddress.CREDIT_ROUTER,
                bind=True,
                additional_bind_address=dispatch_bind,
            )
        )
        self._router_client.register_receiver(self._handle_router_message)

        return_bind = (
            comm_config.credit_return_push_pull_tcp_bind_address if dual_bind else None
        )
        if return_bind:
            self.info(
                f"Dual-bind mode: credit return PULL will also bind to {return_bind}"
            )
        self._return_pull_client: StreamingPullClientProtocol = (
            self.comms.create_streaming_pull_client(
                CommAddress.CREDIT_RETURN,
                bind=True,
                additional_bind_address=return_bind,
            )
        )
        self._return_pull_client.register_receiver(self._handle_return_pull_message)

    def __init__(
        self,
        run: "BenchmarkRun",
        service_id: str,
        **kwargs,
    ) -> None:
        super().__init__(run=run, service_id=service_id, **kwargs)

        self._init_credit_channels(run.cfg.comm_config)

        self._on_return_callback: (
            Callable[[str, CreditReturn], Awaitable[None]] | None
        ) = None
        self._on_first_token_callback: (
            Callable[[FirstToken], Awaitable[None]] | None
        ) = None
        self._on_streaming_content_callback: (
            Callable[[StreamingContent], Awaitable[None]] | None
        ) = None

        # Sticky sessions: owner correlation id -> _StickyEntry
        # Routes all turns of a session (and FORK children pinned to it) to the
        # same worker, because workers cache UserSession state by
        # x_correlation_id. Entries are keyed by the owning root's or SPAWN
        # child's own id; FORK children route via their parent's key.
        self._sticky_sessions: dict[str, _StickyEntry] = {}

        self._cancellation_pending: bool = False
        self._credits_complete: bool = False

        # Snapshot list for iteration - avoids dict.values() overhead in hot path.
        # Rebuilt on worker add/remove (rare) to keep routing fast (common).
        self._workers_cache: list[WorkerLoad] = []
        self._workers: dict[str, WorkerLoad] = {}

        # Map load level -> set of worker_ids at that load (O(1) add/remove)
        self._workers_by_load: dict[int, set[str]] = defaultdict(set)
        # Keep track of the minimum load to avoid recalculating it on every credit sent O(1) vs O(n)
        self._min_load: int = 0

        # Set while >=1 worker is registered; lets wait_for_workers() gate a
        # phase on worker readiness (see that method for the race it closes).
        self._worker_available_event: asyncio.Event = asyncio.Event()

    # =============================================================================
    # Public Methods
    # =============================================================================

    def set_return_callback(
        self, callback: Callable[[str, CreditReturn], Awaitable[None]]
    ) -> None:
        """Set callback for credit returns (enables concurrency control)."""
        self._on_return_callback = callback

    def set_first_token_callback(
        self, callback: Callable[[FirstToken], Awaitable[None]]
    ) -> None:
        """Set callback for first token events (enables prefill concurrency release)."""
        self._on_first_token_callback = callback

    def set_streaming_content_callback(
        self, callback: Callable[[StreamingContent], Awaitable[None]]
    ) -> None:
        """Set the coverage observer without changing request or prefill counts."""
        self._on_streaming_content_callback = callback

    async def wait_for_workers(self, timeout: float) -> None:
        """Close the startup race where a phase issues its first credit before
        any worker has sent ``WorkerReady`` (which makes ``send_credit`` raise
        on empty workers). Called once per phase before the first credit.

        Best-effort startup gate, not an absolute postcondition: the last worker
        can unregister between this returning and the first ``send_credit``, so
        callers must not treat a non-empty pool as guaranteed afterwards.

        Args:
            timeout: Seconds to wait for the first worker before giving up.

        Raises:
            RuntimeError: If no worker registers within ``timeout`` seconds.
        """
        if self._workers:
            return
        try:
            await asyncio.wait_for(self._worker_available_event.wait(), timeout)
        except TimeoutError as exc:
            raise RuntimeError(
                f"No workers registered with the credit router within {timeout}s "
                "(tunable via AIPERF_SERVICE_START_TIMEOUT); cannot start credit issuance"
            ) from exc

    async def send_credit(self, credit: Credit) -> None:
        """Determine the worker based on sticky sessions or least-loaded and send the credit to the worker.

        This method:
        - Determines the worker based on sticky sessions or least-loaded
        - Updates the worker load and sticky sessions
        - Sends the credit to the worker
        """
        if not self._workers:
            raise RuntimeError("No workers available for routing")

        if not credit.x_correlation_id:
            raise RuntimeError("x_correlation_id must be set in Credit")

        # A SPAWN child pins under its own id from its first turn, so it keeps
        # one worker even when the parent has no entry (parent finished, or
        # gated or rootless at replay start). FORK children route via the
        # parent's entry; roots pin to self.
        is_final = credit.is_final_turn
        is_dag_child = credit.parent_correlation_id is not None
        # Identity check: CaseInsensitiveStrEnum.__eq__ is costly on this hot
        # path, and every Credit is built with enum members.
        is_spawn_child = (
            is_dag_child and credit.branch_mode is ConversationBranchMode.SPAWN
        )
        own_key = credit.x_correlation_id
        own_entry = self._sticky_sessions.get(own_key) if is_spawn_child else None
        if own_entry is not None:
            routing_key = own_key
            sticky_entry = own_entry
        else:
            routing_key = credit.parent_correlation_id or own_key
            sticky_entry = self._sticky_sessions.get(routing_key)
        sticky_worker_id = sticky_entry.worker_id if sticky_entry is not None else None

        # Use existing sticky session if worker still valid
        if sticky_worker_id and sticky_worker_id in self._workers:
            worker_id = sticky_worker_id
        else:
            # Least-loaded selection with O(k) tie-breaking where k = workers at min load.
            # Min load lookup is O(1) due to caching.
            least_loaded_workers = self._workers_by_load[self._min_load]
            if len(least_loaded_workers) == 1:
                # Pop the single worker directly. _track_credit_sent will add it to the new load level.
                worker_id = least_loaded_workers.pop()
            else:
                # Multiple workers at min load - find best via single-pass scan.
                # O(k) where k = workers at min load.
                #
                # Tie-breaking priority (lower wins):
                #   1. active_sessions: Fewer committed multi-turn sessions
                #   2. virtual_sent_credits: Fewer historical credits
                #   3. last_sent_at_ns: Oldest send time (LRU-like fairness)
                #
                # Both virtual_sent_credits and last_sent_at_ns are initialized to
                # non-zero values on worker registration to prevent thundering herd.
                # Manual loop is benchmarked faster than min() with lambdas.
                best_worker_id = None
                best_load_key = None
                for _worker_id in least_loaded_workers:
                    load = self._workers[_worker_id]
                    load_key = (
                        load.active_sessions,
                        load.virtual_sent_credits,
                        load.last_sent_at_ns,
                    )
                    if best_load_key is None or load_key < best_load_key:
                        best_load_key = load_key
                        best_worker_id = _worker_id

                worker_id = best_worker_id

            # Create or rebind the sticky entry for non-final turns; also create
            # it when the final turn declares FORK children so the orchestrator's
            # register_child_routing can find it.
            #
            # DAG children never mint an entry keyed by the parent's id: nothing
            # would release it (final-turn eviction is keyed by the owner's own
            # id; release_child_routing only decrements an existing entry), so
            # it would leak active_sessions and bias load balancing. SPAWN
            # children get their own entry below instead.
            if not is_final or credit.has_forks:
                if sticky_entry is None and not is_dag_child:
                    sticky_entry = _StickyEntry(worker_id=worker_id)
                    self._sticky_sessions[routing_key] = sticky_entry
                    load = self._workers[worker_id]
                    load.active_sessions += 1
                    load.active_session_ids.add(routing_key)
                elif (
                    sticky_entry is not None
                    and sticky_entry.worker_id not in self._workers
                ):
                    sticky_entry.worker_id = worker_id
                    load = self._workers[worker_id]
                    load.active_sessions += 1
                    load.active_session_ids.add(routing_key)

        if is_spawn_child and own_entry is None and (not is_final or credit.has_forks):
            own_entry = self._sticky_sessions[own_key] = _StickyEntry(
                worker_id=worker_id
            )
            load = self._workers[worker_id]
            load.active_sessions += 1
            load.active_session_ids.add(own_key)

        # Owning session's final turn (a root or SPAWN child, keyed by its own
        # id): mark parent_final_seen and decrement the reservation. FORK
        # children never touch the parent entry (managed via
        # release_child_routing). If this turn forks, leave the entry
        # in place so register_child_routing lands on the same _StickyEntry.
        if is_final and (not is_dag_child or is_spawn_child):
            entry = own_entry if is_spawn_child else sticky_entry
            if entry is not None:
                entry.parent_final_seen = True
                entry.ref_count -= 1
                if entry.ref_count <= 0 and not credit.has_forks:
                    self._sticky_sessions.pop(own_key, None)
                    load = self._workers[worker_id]
                    load.active_sessions -= 1
                    load.active_session_ids.discard(own_key)

        self._track_credit_sent(worker_id, credit.id)

        await self._router_client.send_to(worker_id, credit)

    async def cancel_all_credits(self) -> None:
        """Send cancellation requests to all workers with in-flight credits."""
        # Mark cancellation first, so we suppress warnings for workers that unregister with in-flight credits.
        self._cancellation_pending = True

        # Build up the map of worker_id to credit_ids snapshot to cancel in an atomic way
        # This works because there are no await calls in this loop, they are all done afterwards.
        to_cancel: dict[str, set[int]] = {}
        for worker_load in self._workers_cache:
            if worker_load.in_flight_credits > 0:
                if self.is_debug_enabled:
                    self.debug(
                        f"Worker {worker_load.worker_id} has {worker_load.in_flight_credits} in-flight credits to cancel: {worker_load.active_credit_ids}"
                    )
                # Make sure to use copy of the set to avoid race conditions.
                to_cancel[worker_load.worker_id] = worker_load.active_credit_ids.copy()

        total_cancelled_credits = 0
        for worker_id, credit_ids in to_cancel.items():
            if self.is_debug_enabled:
                self.debug(
                    f"Sending CancelCredits to worker {worker_id} for {len(credit_ids)} credits"
                )

            await self._router_client.send_to(
                worker_id,
                CancelCredits(credit_ids=credit_ids),
            )
            total_cancelled_credits += len(credit_ids)

        if total_cancelled_credits > 0:
            self.info(
                f"Sent cancellation requests for {total_cancelled_credits} in-flight credits across {len(to_cancel)} workers"
            )
        else:
            self.debug("No in-flight credits to cancel")

    def mark_credits_complete(self) -> None:
        """Mark credits complete - suppresses orphan warnings during shutdown."""
        self._credits_complete = True

    def register_child_routing(self, parent_correlation_id: str) -> None:
        """Increment the sticky-routing refcount for a parent's entry.

        Called by ``BranchOrchestrator`` before dispatching each FORK child so
        the parent's sticky entry survives past its own final turn until every
        FORK child has terminated. If the parent has no active
        sticky entry we log a warning and continue without raising - the
        child will route via least-loaded selection rather than co-locating
        with the parent's worker, losing prefix-cache locality but not
        breaking correctness.
        """
        entry = self._sticky_sessions.get(parent_correlation_id)
        if entry is not None:
            entry.ref_count += 1
        else:
            self.warning(
                lambda: f"register_child_routing: parent "
                f"{parent_correlation_id!r} has no sticky entry; "
                f"child will not co-locate with parent's worker"
            )

    def release_child_routing(self, parent_correlation_id: str) -> None:
        """Decrement the sticky-routing refcount when a FORK child terminates.

        Called by ``BranchOrchestrator`` when a child session reaches a leaf
        or errors out. If the refcount reaches zero and the parent's own final
        turn has already been observed, the sticky entry is evicted.
        """
        entry = self._sticky_sessions.get(parent_correlation_id)
        if entry is None:
            return
        entry.ref_count -= 1
        if entry.ref_count <= 0 and entry.parent_final_seen:
            worker_id = entry.worker_id
            self._sticky_sessions.pop(parent_correlation_id, None)
            load = self._workers.get(worker_id)
            if load is not None:
                load.active_sessions -= 1
                load.active_session_ids.discard(parent_correlation_id)

    def evict_unclaimed_sticky(self, parent_correlation_id: str) -> None:
        """Force-pop a sticky entry retained for FORK children that never registered.

        Parent final turns with ``has_forks=True`` keep the sticky entry alive
        (``parent_final_seen=True``, ``ref_count`` already decremented) so
        ``register_child_routing`` can find it. When every child fails before
        that register (e.g. ``start_branch_child`` raises), nothing else
        decrements ``active_sessions``. Called from
        ``BranchOrchestrator._finalize_failed_dispatches``; safe no-op when
        children hold refs or the parent final has not been seen.
        """
        entry = self._sticky_sessions.get(parent_correlation_id)
        if entry is None or not entry.parent_final_seen or entry.ref_count > 0:
            return
        worker_id = entry.worker_id
        self._sticky_sessions.pop(parent_correlation_id, None)
        load = self._workers.get(worker_id)
        if load is not None:
            load.active_sessions -= 1
            load.active_session_ids.discard(parent_correlation_id)

    # =============================================================================
    # Private Methods
    # =============================================================================

    async def _handle_return_pull_message(self, message: WorkerToRouterMessage) -> None:
        """Adapt the identity-less PULL fan-in to the shared handler.

        The PUSH/PULL return channel has no ZMQ envelope identity, so the worker
        id rides inside CreditReturn (FirstToken does not need it). Unpack it and
        delegate to the common handler.

        Ordering note: CreditReturn/FirstToken/StreamingContent arrive on this PULL
        channel while WorkerReady/WorkerShutdown stay on the DEALER, so returns and
        lifecycle messages are no longer mutually ordered (on the single bidirectional
        DEALER they were). That is safe because a worker only emits WorkerShutdown
        after all its returns have been sent, and the timing manager's phase /
        cancellation barrier drains outstanding returns before workers are torn down;
        a return therefore cannot legitimately land after its worker's unregister
        outside the teardown window, where ``_cancellation_pending`` /
        ``_credits_complete`` already suppress the ``_warn_missing_worker`` path.
        """
        worker_id = getattr(message, "worker_id", None) or ""
        await self._handle_router_message(worker_id, message)

    async def _handle_router_message(
        self, worker_id: str, message: WorkerToRouterMessage
    ) -> None:
        """Dispatch worker return, streaming activity, and lifecycle messages."""
        match message:
            case CreditReturn():
                self._track_credit_returned(
                    worker_id,
                    message.credit.id,
                    message.cancelled,
                    message.error is not None,
                )
                if self._on_return_callback:
                    # Await directly instead of execute_async - credit returns release
                    # concurrency slots, so delays here directly impact throughput.
                    await self._on_return_callback(worker_id, message)
            case FirstToken():
                if self._on_first_token_callback:
                    # Forward TTFT to orchestrator so it can release the prefill slot.
                    await self._on_first_token_callback(message)
            case StreamingContent():
                if self._on_streaming_content_callback:
                    await self._on_streaming_content_callback(message)
            case WorkerReady():
                self._register_worker(worker_id)
            case WorkerShutdown():
                self._unregister_worker(worker_id)
            case _:
                self.warning(f"Unknown message type: {type(message).__name__}")

    def _register_worker(self, worker_id: str) -> None:
        """Register worker for routing, create WorkerLoad entry.

        Late-joining workers initialize:
        - virtual_sent_credits to average (prevents thundering herd on credits)
        - last_sent_at_ns to current time (prevents winning all timestamp tie-breaks)
        """
        if worker_id not in self._workers:
            # Initialize to averages to prevent thundering herd
            avg_virtual = 0
            if self._workers_cache:
                avg_virtual = sum(
                    w.virtual_sent_credits for w in self._workers_cache
                ) // len(self._workers_cache)

            self._workers[worker_id] = WorkerLoad(
                worker_id=worker_id,
                virtual_sent_credits=avg_virtual,
                last_sent_at_ns=time.perf_counter_ns(),
            )
            if self.is_trace_enabled:
                self.trace(
                    f"Worker registered: {worker_id} (total={len(self._workers)}, "
                    f"virtual_credits={avg_virtual})"
                )
            self._workers_cache = list(self._workers.values())
            # We know that new workers are load 0, and load 0 is the absolute minimum load,
            # so we can cheat and just set minimum load to 0 without recalculating.
            self._min_load = 0
            self._workers_by_load[0].add(worker_id)
            self._worker_available_event.set()

    def _unregister_worker(self, worker_id: str) -> None:
        """Unregister worker. Sticky sessions are cleared and reassigned on next access."""
        if worker_load := self._workers.pop(worker_id, None):
            if worker_load.in_flight_credits > 0 and not self._cancellation_pending:
                self.warning(
                    f"Worker {worker_id} unregistered with {worker_load.in_flight_credits} in-flight credits"
                )
            if self.is_trace_enabled:
                self.trace(
                    f"Worker unregistered: {worker_id} (remaining={len(self._workers)})"
                )
            self._workers_by_load[worker_load.in_flight_credits].discard(worker_id)
            if not self._workers:
                self._worker_available_event.clear()

            # Remove all orphaned sticky sessions now and warn once up front
            orphaned_session_ids = worker_load.active_session_ids
            if orphaned_session_ids and not (
                self._cancellation_pending or self._credits_complete
            ):
                self.warning(
                    f"Worker {worker_id} unregistered with {len(orphaned_session_ids)} active sessions, will reassign"
                )
            for x_correlation_id in orphaned_session_ids:
                self._sticky_sessions.pop(x_correlation_id, None)

        if not worker_load:
            # Warn but continue - may happen if shutdown message arrives before ready message.
            self.warning(
                f"Worker {worker_id} not found when unregistering. This should not happen."
            )

        self._workers_cache = list(self._workers.values())

        if not worker_load or (
            worker_load.in_flight_credits == self._min_load
            and len(self._workers_by_load[self._min_load]) == 0
        ):
            # Recalculate min_load if the removed worker was the last at the current minimum.
            if len(self._workers_cache) > 0:
                self._min_load = min(w.in_flight_credits for w in self._workers_cache)
            else:
                self._min_load = 0

    def _track_credit_sent(self, worker_id: str, credit_id: int) -> None:
        """Update worker load: increment in_flight_credits. Lock-free."""
        if worker_load := self._workers.get(worker_id):
            old_load = worker_load.in_flight_credits

            worker_load.total_sent_credits += 1
            worker_load.virtual_sent_credits += 1
            worker_load.in_flight_credits += 1
            worker_load.active_credit_ids.add(credit_id)
            worker_load.last_sent_at_ns = time.perf_counter_ns()

            new_load = worker_load.in_flight_credits
            # Keep the workers by load updated for faster load balancing.
            self._workers_by_load[old_load].discard(worker_id)
            self._workers_by_load[new_load].add(worker_id)

            if old_load == self._min_load and len(self._workers_by_load[old_load]) == 0:
                # We only send credits one at a time, so if this worker was the last at the minimum load,
                # it is safe to assume that the new minimum load is this worker's new load. Saving a recalculation.
                self._min_load = new_load

        else:
            self._warn_missing_worker(worker_id, "sent")

    def _track_credit_returned(
        self, worker_id: str, credit_id: int, cancelled: bool, error_reported: bool
    ) -> None:
        """Update worker load: decrement in_flight_credits. Lock-free."""
        if worker_load := self._workers.get(worker_id):
            worker_load.active_credit_ids.discard(credit_id)

            if cancelled:
                worker_load.total_cancelled_credits += 1
            else:
                worker_load.total_completed_credits += 1
            if error_reported:
                worker_load.total_errors_reported += 1

            old_load = worker_load.in_flight_credits
            if worker_load.in_flight_credits > 0:
                worker_load.in_flight_credits -= 1
                new_load = worker_load.in_flight_credits

                self._workers_by_load[old_load].discard(worker_id)
                self._workers_by_load[new_load].add(worker_id)
                if new_load < self._min_load:
                    self._min_load = new_load
            else:
                self.error(
                    f"Worker {worker_id} in_flight_credits already 0 when tracking returned credit {credit_id}"
                )
        else:
            self._warn_missing_worker(worker_id, "returned")

    def _warn_missing_worker(self, worker_id: str, credit_action: str) -> None:
        """Warn if worker is missing when tracking credit sent or returned."""
        if self._cancellation_pending:
            # Even during cancellation, the workers should still be registered, but if they are not it won't cause any issues.
            self.warning(
                f"Worker {worker_id} not found when tracking credit {credit_action} during cancellation."
            )
        else:
            self.error(
                f"Worker {worker_id} not found when tracking credit {credit_action}. This should not happen."
            )
