# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase progress tracker for credit counting with event coordination.

Wraps CreditCounter and adds:
- Event management (all_credits_sent, all_credits_returned)
- Count freezing coordination
- Stats creation (combines counter + lifecycle)
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.models import CreditPhaseStats
from aiperf.timing.phase.credit_counter import CreditCounter

if TYPE_CHECKING:
    from aiperf.credit.structs import TurnToSend
    from aiperf.timing.config import CreditPhaseConfig
    from aiperf.timing.phase.lifecycle import PhaseLifecycle

_logger = AIPerfLogger(__name__)


class PhaseProgressTracker:
    """Tracks credit progress with event coordination.

    Wraps CreditCounter and adds:
    - Event management (all_credits_sent_event, all_credits_returned_event)
    - Count freezing (sent counts frozen when sending completes)
    - Stats creation (combines counter + lifecycle)

    Used by:
    - CreditIssuer: increment_sent, freeze_sent_counts, set all_credits_sent_event
    - CreditCallbackHandler: increment_returned, increment_prefill_released, set all_credits_returned_event
    - PhaseRunner: create_stats, wait on events

    CRITICAL: increment_sent and increment_returned are atomic (no await between
    read and write). This is enforced by CreditCounter.
    """

    def __init__(self, config: CreditPhaseConfig) -> None:
        """Initialize progress tracker.

        Args:
            config: Phase configuration with stop thresholds.
        """
        self._config = config
        self._counter = CreditCounter(config)
        self._last_streaming_content_ns: int | None = None

        # Events for synchronization
        self.all_credits_sent_event: asyncio.Event = asyncio.Event()
        self.all_credits_returned_event: asyncio.Event = asyncio.Event()

    # =========================================================================
    # Counter Properties (delegated to CreditCounter via protocol)
    # =========================================================================

    @property
    def counter(self) -> CreditCounter:
        """Credit counter."""
        return self._counter

    @property
    def in_flight(self) -> int:
        """In-flight credits (sent but not returned)."""
        return self._counter.in_flight

    @property
    def in_flight_sessions(self) -> int:
        """In-flight sessions (started but final turn not returned)."""
        return self._counter.in_flight_sessions

    @property
    def in_flight_prefills(self) -> int:
        """Requests in prefill phase (sent but TTFT not yet received)."""
        return self._counter.in_flight_prefills

    # =========================================================================
    # Increment Methods (wrapped with event coordination)
    # =========================================================================

    def increment_sent(self, turn: TurnToSend) -> tuple[int, bool]:
        """Atomically increment sent count.

        Args:
            turn: The turn being sent.

        Returns:
            (credit_index, is_final_credit)
            - credit_index: Sequential ID for this credit
            - is_final_credit: True if this was the final credit to send

        CRITICAL: No async calls in this method - preserves atomicity.

        If is_final_credit=True, caller should:
        1. Call freeze_sent_counts()
        2. Set all_credits_sent_event
        """
        return self._counter.increment_sent(turn)

    def increment_returned(
        self,
        is_final_turn: bool,
        cancelled: bool,
        errored: bool = False,
        *,
        is_child: bool = False,
    ) -> bool:
        """Atomically increment returned count.

        Args:
            is_final_turn: Whether this turn is the final turn of a session.
            cancelled: Whether the credit was cancelled.
            errored: Whether the request returned with a non-None error. Bumps
                ``request_errors`` (request-level; ticks for children too).
            is_child: True when the returned credit is a DAG descendant
                (``credit.agent_depth > 0``). Child returns bump the
                request-level counters (``requests_completed`` /
                ``requests_cancelled``) for observability — they're
                real HTTP requests — but skip session-level bookkeeping
                (``completed_sessions`` / ``cancelled_sessions``)
                because children inherit the parent's session slot.

        Returns:
            True if ALL credits returned (this was the final return).

        CRITICAL: No async calls in this method - preserves atomicity.

        If returns True, caller should set all_credits_returned_event.
        The ``CreditCallbackHandler`` defers the event fire via
        ``BranchOrchestrator.has_pending_branch_work()`` when the
        DAG still has in-flight descendants.

        Note: Late arrivals (after phase complete) are handled by caller
        checking lifecycle.is_complete before calling this method.
        """
        return self._counter.increment_returned(
            is_final_turn, cancelled, errored=errored, is_child=is_child
        )

    def increment_prefill_released(self) -> None:
        """Increment prefill released count.

        Called when:
        1. TTFT received (first token callback)
        2. Credit returns without TTFT (prefill never completed)
        """
        self._counter.increment_prefill_released()

    def account_lane_session(self, session_turns: int) -> None:
        """Count a deferred-lane session that acquires its slot directly.

        Delegates to :meth:`CreditCounter.account_lane_session`. Called by
        ``CreditIssuer.acquire_lane_credit`` when a gated parent
        (``root_pending=True``) holds a lane slot directly: the parent's join
        turn later bumps ``completed_sessions``, so its session is counted in
        ``sent_sessions`` here to keep ``in_flight_sessions`` non-negative.
        Mirrors the turn-0 arm of :meth:`increment_sent`.

        CRITICAL: No async calls in this method - preserves atomicity.
        """
        self._counter.account_lane_session(session_turns)

    # =========================================================================
    # Freezing Methods
    # =========================================================================

    def freeze_sent_counts(self) -> None:
        """Freeze sent counts when sending completes.

        After freezing, final_requests_sent becomes the authoritative count
        for checking if all credits have returned.

        Called by CreditIssuer when is_final_credit=True.
        """
        self._counter.freeze_sent_counts()

    def freeze_completed_counts(self) -> None:
        """Freeze completed counts when phase completes.

        Called by PhaseRunner when phase transitions to COMPLETE.
        """
        self._counter.freeze_completed_counts()

    # =========================================================================
    # Query Methods
    # =========================================================================

    def check_all_returned_or_cancelled(self) -> bool:
        """True if all sent credits have been returned or cancelled.

        Used by PhaseRunner to check if phase can complete without
        waiting for the event.
        """
        return self._counter.check_all_returned_or_cancelled()

    # =========================================================================
    # Stats Creation
    # =========================================================================

    def observe_streaming_content(self, timestamp_ns: int | None) -> None:
        """Retain the latest content timestamp independently of completion counts."""
        if timestamp_ns is not None:
            self._last_streaming_content_ns = max(
                self._last_streaming_content_ns or 0, timestamp_ns
            )

    def create_stats(self, lifecycle: PhaseLifecycle) -> CreditPhaseStats:
        """Create immutable stats snapshot.

        Combines counter progress with lifecycle timestamps.

        Args:
            lifecycle: Phase lifecycle for timestamp data.

        Returns:
            Immutable CreditPhaseStats snapshot.
        """
        return CreditPhaseStats(
            phase=self._config.phase,
            phase_index=self._config.phase_index,
            profiling_index=self._config.profiling_index,
            phase_name=self._config.phase_name,
            phase_kind=self._config.phase_kind,
            # Timestamps from lifecycle
            start_ns=lifecycle.started_at_ns,
            last_streaming_content_ns=self._last_streaming_content_ns,
            sent_end_ns=lifecycle.sending_complete_at_ns,
            requests_end_ns=lifecycle.complete_at_ns,
            # Configuration (stop conditions)
            total_expected_requests=self._config.total_expected_requests,
            expected_duration_sec=self._config.expected_duration_sec,
            expected_num_sessions=self._config.expected_num_sessions,
            expected_grace_period_sec=self._config.grace_period_sec,
            # Progress from counter
            requests_sent=self._counter.requests_sent,
            requests_completed=self._counter.requests_completed,
            requests_cancelled=self._counter.requests_cancelled,
            request_errors=self._counter.request_errors,
            sent_sessions=self._counter.sent_sessions,
            completed_sessions=self._counter.completed_sessions,
            cancelled_sessions=self._counter.cancelled_sessions,
            total_session_turns=self._counter.total_session_turns,
            # Final counts (frozen values)
            final_requests_sent=self._counter.final_requests_sent,
            final_requests_completed=self._counter.final_requests_completed,
            final_requests_cancelled=self._counter.final_requests_cancelled,
            final_request_errors=self._counter.final_request_errors,
            final_sent_sessions=self._counter.final_sent_sessions,
            final_completed_sessions=self._counter.final_completed_sessions,
            final_cancelled_sessions=self._counter.final_cancelled_sessions,
            # Metadata from lifecycle
            timeout_triggered=lifecycle.timeout_triggered,
            grace_period_timeout_triggered=lifecycle.grace_period_triggered,
            was_cancelled=lifecycle.was_cancelled,
        )

    def create_stats_with_baseline_window(
        self,
        lifecycle: PhaseLifecycle,
        *,
        baseline_start_ns: int | None,
        baseline_end_ns: int | None,
    ) -> CreditPhaseStats:
        """Create stats annotated with metric baseline gate timestamps."""
        return self.create_stats(lifecycle).model_copy(
            update={
                "baseline_start_ns": baseline_start_ns,
                "baseline_end_ns": baseline_end_ns,
            }
        )
