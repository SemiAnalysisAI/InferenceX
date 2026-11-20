# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from collections import defaultdict

from aiperf.common.enums import CreditPhase
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import (
    CreditPhaseStats,
    ErrorDetails,
    MetricRecordMetadata,
    PhaseRecordsStats,
    WorkerProcessingStats,
)


class CreditPhaseRecordsTracker(AIPerfLoggerMixin):
    """Credit Phase Records Tracker. This is used to track the progress of a credit phase, as well
    as provide atomic operations for incrementing the processed and error counts.

    Thread Safety:
        The increment_* methods guarantee no partial updates.
        Safe for asyncio concurrent access without locks because:
            1. Updates use atomic incrementation (no in-place mutation)
            2. No await points between read and write in atomic methods
        3. Asyncio event loop serializes all operations

    Key Methods:
        - increment_processed(): Atomically increment the processed count
        - increment_errors(): Atomically increment the errors count
        - create_stats(): Create a new immutable RecordsPhaseStats object for the phase (for use in messages).
        - mark_started(): Mark the phase as started (set the start_ns).
        - mark_processing_complete(): Mark the phase as processing complete (set the processing_end_ns).
    """

    def __init__(self, phase: CreditPhase, **kwargs) -> None:
        super().__init__(**kwargs)
        # Must be set by the caller
        self._phase: CreditPhase = phase
        self._phase_index: int | None = None
        self._profiling_index: int | None = None
        self._phase_name: str | None = None
        self._phase_kind: str | None = None
        self._total_expected_requests: int | None = None

        # Timestamp fields
        self._start_ns: int | None = None
        self._sent_end_ns: int | None = None
        self._requests_end_ns: int | None = None
        self._last_streaming_content_ns: int | None = None
        self._baseline_start_ns: int | None = None
        self._baseline_end_ns: int | None = None
        # Records processing timestamp fields
        self._records_end_ns: int | None = None

        # Progress fields
        self._success_records: int = 0
        self._error_records: int = 0

        # Final count fields
        self._final_requests_completed: int | None = None
        self._final_requests_cancelled: int | None = None
        self._final_request_errors: int | None = None

        # Timeout/cancel fields
        self._timeout_triggered: bool = False
        self._grace_period_timeout_triggered: bool = False
        self._was_cancelled: bool = False

        # Completion fields
        self._sent_all_records_received: bool = False

        # Worker fields
        self._worker_stats: dict[str, WorkerProcessingStats] = defaultdict(
            WorkerProcessingStats
        )

    @property
    def phase(self) -> CreditPhase:
        """The phase of the credit phase tracker."""
        return self._phase

    @property
    def total_records(self) -> int:
        """The total number of records processed, errored, or filtered out."""
        return self._success_records + self._error_records

    @property
    def is_active(self) -> bool:
        """Check if the phase is active."""
        return self._start_ns is not None and self._records_end_ns is None

    def create_stats(self) -> PhaseRecordsStats:
        """Create a new immutable RecordsPhaseStats object for the phase (for use in messages)."""
        return PhaseRecordsStats(
            phase=self._phase,
            phase_index=self._phase_index,
            profiling_index=self._profiling_index,
            phase_name=self._phase_name,
            phase_kind=self._phase_kind,
            start_ns=self._start_ns,
            sent_end_ns=self._sent_end_ns,
            requests_end_ns=self._requests_end_ns,
            last_streaming_content_ns=self._last_streaming_content_ns,
            baseline_start_ns=self._baseline_start_ns,
            baseline_end_ns=self._baseline_end_ns,
            records_end_ns=self._records_end_ns,
            total_expected_requests=self._total_expected_requests,
            success_records=self._success_records,
            error_records=self._error_records,
            final_requests_completed=self._final_requests_completed,
            final_requests_cancelled=self._final_requests_cancelled,
            final_request_errors=self._final_request_errors,
            timeout_triggered=self._timeout_triggered,
            grace_period_timeout_triggered=self._grace_period_timeout_triggered,
            was_cancelled=self._was_cancelled,
        )

    def update_from_credit_phase_stats(self, credit_stats: CreditPhaseStats) -> None:
        """Update the phase info."""
        self._phase_index = credit_stats.phase_index
        self._profiling_index = credit_stats.profiling_index
        self._phase_name = credit_stats.phase_name
        self._phase_kind = credit_stats.phase_kind
        self._start_ns = credit_stats.start_ns
        self._sent_end_ns = credit_stats.sent_end_ns
        self._requests_end_ns = credit_stats.requests_end_ns
        self._last_streaming_content_ns = credit_stats.last_streaming_content_ns
        self._baseline_start_ns = credit_stats.baseline_start_ns
        self._baseline_end_ns = credit_stats.baseline_end_ns
        self._total_expected_requests = credit_stats.total_expected_requests
        self._final_requests_completed = credit_stats.final_requests_completed
        self._final_requests_cancelled = credit_stats.final_requests_cancelled
        self._final_request_errors = credit_stats.final_request_errors
        self._timeout_triggered = credit_stats.timeout_triggered
        self._grace_period_timeout_triggered = (
            credit_stats.grace_period_timeout_triggered
        )
        self._was_cancelled = credit_stats.was_cancelled

    def increment_success_records(self) -> None:
        """Increment the success records count."""
        self._success_records += 1

    def increment_error_records(self) -> None:
        """Increment the error records count."""
        self._error_records += 1

    def increment_worker_success_records(self, worker_id: str) -> None:
        """Increment the success records count for a worker."""
        self._worker_stats[worker_id].success_records += 1

    def increment_worker_error_records(self, worker_id: str) -> None:
        """Increment the error records count for a worker."""
        self._worker_stats[worker_id].error_records += 1

    def _mark_all_records_received(self) -> bool:
        if self._sent_all_records_received:
            return False
        self._records_end_ns = time.time_ns()
        self._sent_all_records_received = True
        return True

    def check_and_set_all_records_received(self) -> bool:
        """Check if all records have been received and set the flag if so.
        Returns:
            True if all records have been received and the flag was not already set, False otherwise.
        """
        if self._sent_all_records_received:
            return False

        all_records_received = self._final_requests_completed is not None and (
            self._success_records + self._error_records
            >= self._final_requests_completed
        )
        if all_records_received:
            self._mark_all_records_received()

        return all_records_received


class RecordsTracker:
    """Records Tracker. This is used to track the progress of the records phases.

    Fields:
        phase: The type of credit phase
        total_expected_requests: The total number of expected requests to process. If None, the phase is not request count based.
    """

    def __init__(self) -> None:
        self._phase_trackers: dict[
            tuple[CreditPhase, int | None], CreditPhaseRecordsTracker
        ] = {}
        self._latest_phase_index: dict[CreditPhase, int | None] = {}

    def _get_phase_tracker(
        self, phase: CreditPhase, phase_index: int | None = None
    ) -> CreditPhaseRecordsTracker:
        """Get the phase tracker for one concrete phase instance."""
        key = (phase, phase_index)
        if key not in self._phase_trackers:
            self._phase_trackers[key] = CreditPhaseRecordsTracker(phase)
        return self._phase_trackers[key]

    def _record_latest_phase_index(
        self, phase: CreditPhase, phase_index: int | None
    ) -> None:
        if phase_index is None:
            self._latest_phase_index.setdefault(phase, None)
            return
        current = self._latest_phase_index.get(phase)
        if current is None or phase_index >= current:
            self._latest_phase_index[phase] = phase_index

    def _latest_tracker_for_phase(
        self, phase: CreditPhase
    ) -> CreditPhaseRecordsTracker | None:
        if phase not in self._latest_phase_index:
            return None
        return self._get_phase_tracker(phase, self._latest_phase_index[phase])

    def create_overall_worker_stats(self) -> dict[str, WorkerProcessingStats]:
        """Create a new dictionary of WorkerProcessingStats objects for ALL phases."""
        all_worker_stats = defaultdict(WorkerProcessingStats)
        for phase_tracker in self._phase_trackers.values():
            for worker_id, worker_stats in phase_tracker._worker_stats.items():
                all_worker_stats[
                    worker_id
                ].success_records += worker_stats.success_records
                all_worker_stats[worker_id].error_records += worker_stats.error_records
        return dict(all_worker_stats)

    def create_stats_for_phase(
        self, phase: CreditPhase, phase_index: int | None = None
    ) -> PhaseRecordsStats:
        """Create stats for one concrete phase instance.

        When ``phase_index`` is omitted, this preserves the legacy behavior by
        returning the latest tracker for that phase kind.
        """
        if phase_index is None:
            phase_tracker = self._latest_tracker_for_phase(phase)
            if phase_tracker is None:
                return CreditPhaseRecordsTracker(phase).create_stats()
            return phase_tracker.create_stats()
        return self._get_phase_tracker(phase, phase_index).create_stats()

    def create_aggregate_stats_for_phase(self, phase: CreditPhase) -> PhaseRecordsStats:
        """Create stats spanning all concrete instances of ``phase``."""
        concrete_stats = [
            tracker.create_stats()
            for (tracker_phase, phase_index), tracker in self._phase_trackers.items()
            if tracker_phase == phase and phase_index is not None
        ]
        orphan_stats = [
            tracker.create_stats()
            for (tracker_phase, phase_index), tracker in self._phase_trackers.items()
            if tracker_phase == phase and phase_index is None
        ]
        stats = (concrete_stats + orphan_stats) or [
            tracker.create_stats()
            for (tracker_phase, _), tracker in self._phase_trackers.items()
            if tracker_phase == phase
        ]
        if not stats:
            return self.create_stats_for_phase(phase)

        starts = [s.start_ns for s in stats if s.start_ns is not None]
        sent_ends = [s.sent_end_ns for s in stats if s.sent_end_ns is not None]
        request_ends = [
            s.requests_end_ns for s in stats if s.requests_end_ns is not None
        ]
        baseline_starts = [
            s.baseline_start_ns for s in stats if s.baseline_start_ns is not None
        ]
        baseline_ends = [
            s.baseline_end_ns for s in stats if s.baseline_end_ns is not None
        ]
        record_ends = [s.records_end_ns for s in stats if s.records_end_ns is not None]

        def optional_sum(values: list[int | None]) -> int | None:
            concrete = [v for v in values if v is not None]
            return sum(concrete) if concrete else None

        return PhaseRecordsStats(
            phase=phase,
            start_ns=min(starts) if starts else None,
            sent_end_ns=max(sent_ends) if sent_ends else None,
            requests_end_ns=max(request_ends) if request_ends else None,
            last_streaming_content_ns=max(
                (
                    s.last_streaming_content_ns
                    for s in stats
                    if s.last_streaming_content_ns is not None
                ),
                default=None,
            ),
            baseline_start_ns=min(baseline_starts) if baseline_starts else None,
            baseline_end_ns=max(baseline_ends) if baseline_ends else None,
            records_end_ns=max(record_ends) if record_ends else None,
            total_expected_requests=optional_sum(
                [s.total_expected_requests for s in stats]
            ),
            success_records=sum(s.success_records for s in stats),
            error_records=sum(s.error_records for s in stats),
            final_requests_completed=optional_sum(
                [s.final_requests_completed for s in stats]
            ),
            final_requests_cancelled=optional_sum(
                [s.final_requests_cancelled for s in stats]
            ),
            final_request_errors=optional_sum([s.final_request_errors for s in stats]),
            timeout_triggered=any(s.timeout_triggered for s in stats),
            grace_period_timeout_triggered=any(
                s.grace_period_timeout_triggered for s in stats
            ),
            was_cancelled=any(s.was_cancelled for s in stats),
        )

    def total_records_for_phase(
        self, phase: CreditPhase, phase_index: int | None = None
    ) -> int:
        """Return the running total for one concrete phase without building a model.

        Lightweight int accessor for hot per-record paths (e.g. the
        failed-request abort check) that only need the counter, avoiding a full
        validated ``PhaseRecordsStats`` construction per record.

        When ``phase_index`` is omitted, use the latest concrete tracker rather
        than creating an orphan ``(phase, None)`` tracker.
        """
        tracker = (
            self._get_phase_tracker(phase, phase_index)
            if phase_index is not None
            else self._latest_tracker_for_phase(phase)
        )
        return tracker.total_records if tracker is not None else 0

    def error_records_for_phase(
        self, phase: CreditPhase, phase_index: int | None = None
    ) -> int:
        """Return the running error count for one concrete phase."""
        tracker = (
            self._get_phase_tracker(phase, phase_index)
            if phase_index is not None
            else self._latest_tracker_for_phase(phase)
        )
        return tracker._error_records if tracker is not None else 0

    def update_phase_info(self, credit_phase_stats: CreditPhaseStats) -> None:
        """Update the phase tracker."""
        phase_tracker = self._get_phase_tracker(
            credit_phase_stats.phase, credit_phase_stats.phase_index
        )
        self._record_latest_phase_index(
            credit_phase_stats.phase, credit_phase_stats.phase_index
        )
        phase_tracker.update_from_credit_phase_stats(credit_phase_stats)

    def update_from_request(
        self, metadata: MetricRecordMetadata, error: ErrorDetails | None
    ) -> None:
        """Update the phase tracker for one completed request.

        Drives the per-request lockstep off the request envelope: a request is
        counted as a success when ``error is None``, otherwise as an error.
        """
        phase_index = metadata.phase_index
        if phase_index is None:
            latest_index = self._latest_phase_index.get(metadata.benchmark_phase)
            if latest_index is not None:
                phase_index = latest_index
        phase_tracker = self._get_phase_tracker(metadata.benchmark_phase, phase_index)
        self._record_latest_phase_index(metadata.benchmark_phase, phase_index)
        if error is None:
            phase_tracker.increment_success_records()
            phase_tracker.increment_worker_success_records(metadata.worker_id)
        else:
            phase_tracker.increment_error_records()
            phase_tracker.increment_worker_error_records(metadata.worker_id)

    def was_phase_cancelled(self, phase: CreditPhase) -> bool:
        """Check if the phase was cancelled."""
        return any(
            tracker._was_cancelled
            for (tracker_phase, _), tracker in self._phase_trackers.items()
            if tracker_phase == phase
        )

    def mark_phase_cancelled(self, phase: CreditPhase) -> None:
        """Mark a phase as cancelled (e.g., from ProfileCancelCommand).

        This should be called when the cancel command is received to ensure
        the cancelled state is tracked even before the CreditPhaseCompleteMessage
        arrives with the updated stats.
        """
        for (tracker_phase, _), tracker in self._phase_trackers.items():
            if tracker_phase == phase:
                tracker._was_cancelled = True

    def check_and_set_all_records_received_for_phase(
        self, phase: CreditPhase, phase_index: int | None = None
    ) -> bool:
        """Check if all records have been received and set the flag if so."""
        if phase_index is not None:
            phase_tracker = self._get_phase_tracker(phase, phase_index)
            return phase_tracker.check_and_set_all_records_received()

        concrete_phase_items = [
            (phase_index, tracker)
            for (tracker_phase, phase_index), tracker in self._phase_trackers.items()
            if tracker_phase == phase and phase_index is not None
        ]
        orphan_phase_items = [
            (phase_index, tracker)
            for (tracker_phase, phase_index), tracker in self._phase_trackers.items()
            if tracker_phase == phase and phase_index is None
        ]
        phase_items = (
            concrete_phase_items + orphan_phase_items
            if concrete_phase_items
            else orphan_phase_items
        ) or [
            (tracked_phase_index, tracker)
            for (
                tracker_phase,
                tracked_phase_index,
            ), tracker in self._phase_trackers.items()
            if tracker_phase == phase
        ]
        if not phase_items:
            return False

        gated_trackers: list[CreditPhaseRecordsTracker] = []
        for tracked_phase_index, tracker in phase_items:
            stats = tracker.create_stats()
            if stats.final_requests_completed is None:
                if tracked_phase_index is None:
                    continue
                return False
            gated_trackers.append(tracker)
            if stats.total_records < stats.final_requests_completed:
                return False

        if not gated_trackers:
            return False

        newly_completed = False
        for tracker in gated_trackers:
            newly_completed = (
                tracker.check_and_set_all_records_received() or newly_completed
            )
        return newly_completed

    def check_and_set_all_records_received_for_stats(
        self, stats: CreditPhaseStats | PhaseRecordsStats
    ) -> bool:
        """Check completion for the concrete phase represented by stats."""
        return self.check_and_set_all_records_received_for_phase(
            stats.phase, stats.phase_index
        )

    def create_active_phase_stats_list(self) -> list[PhaseRecordsStats]:
        """Get the active phase stats."""
        return [
            pt.create_stats() for pt in self._phase_trackers.values() if pt.is_active
        ]
