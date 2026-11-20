# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import time
from typing import ClassVar

from pydantic import ConfigDict, Field

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import CreditPhase
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.types import PhaseKind


class BasePhaseStats(AIPerfBaseModel):
    """Base model for phase stats. This is used to track the progress of the credit phases."""

    model_config = ConfigDict(frozen=True)

    phase: CreditPhase = Field(
        ..., description="The type of credit phase, such as warmup or profiling."
    )
    phase_index: int | None = Field(
        default=None, ge=0, description="Absolute index in the ordered phases list."
    )
    profiling_index: int | None = Field(
        default=None,
        ge=0,
        description="Index among profiling-kind phases; None for warmup.",
    )
    phase_name: str | None = Field(
        default=None, description="User-provided unique phase name."
    )
    phase_kind: PhaseKind | None = Field(
        default=None, description="Phase semantic kind: warmup or profiling."
    )

    # Timestamp fields
    last_streaming_content_ns: int | None = Field(
        default=None,
        ge=0,
        description="Latest parsed streaming content timestamp (wall-clock nanoseconds) "
        "reported by streaming activity or a returned credit in this phase, including cancelled requests.",
    )
    start_ns: int | None = Field(
        default=None,
        ge=0,
        description="The start time of the credit phase in nanoseconds.",
    )
    sent_end_ns: int | None = Field(
        default=None,
        ge=0,
        description="The time of the last sent credit in nanoseconds. If None, the phase has not sent all credits.",
    )
    requests_end_ns: int | None = Field(
        default=None,
        ge=0,
        description="The time in which the last credit was returned from the workers in nanoseconds. If None, the phase has not completed.",
    )
    baseline_start_ns: int | None = Field(
        default=None,
        ge=0,
        description="The timestamp after the phase START baseline request was published, when available.",
    )
    baseline_end_ns: int | None = Field(
        default=None,
        ge=0,
        description="The timestamp after the phase END baseline request was published, when available.",
    )

    # Expectation / stop condition fields
    total_expected_requests: int | None = Field(
        default=None,
        gt=0,
        description="The total number of expected requests to send to the workers. If None, the phase is not request count based.",
    )
    expected_duration_sec: float | None = Field(
        default=None,
        gt=0,
        description="The expected duration of the credit phase in seconds. If None, the phase is not time based.",
    )
    expected_num_sessions: int | None = Field(
        default=None,
        gt=0,
        description="The expected number of user sessions to send to the workers. If None, the phase is not session count based.",
    )
    expected_grace_period_sec: float | None = Field(
        default=None,
        ge=0,
        description="The expected grace period duration in seconds. If None, there is no grace period limit.",
    )

    # Final count fields
    final_requests_sent: int | None = Field(
        default=None,
        ge=0,
        description="The final number of requests sent to the workers. If None, the phase has not completed.",
    )
    final_requests_completed: int | None = Field(
        default=None,
        ge=0,
        description="The final number of requests completed by the workers (success OR error, but NOT cancelled). If None, the phase has not completed.",
    )
    final_requests_cancelled: int | None = Field(
        default=None,
        ge=0,
        description="The final number of requests cancelled by the workers. If None, the phase has not completed.",
    )
    final_request_errors: int | None = Field(
        default=None,
        ge=0,
        description="The final number of requests that returned errors from the workers. If None, the phase has not completed.",
    )
    final_sent_sessions: int | None = Field(
        default=None,
        ge=0,
        description="The final number of unique user sessions that have been sent so far. If None, the phase has not completed.",
    )
    final_completed_sessions: int | None = Field(
        default=None,
        ge=0,
        description="The final number of unique user sessions that have been completed so far. If None, the phase has not completed.",
    )
    final_cancelled_sessions: int | None = Field(
        default=None,
        ge=0,
        description="The final number of unique user sessions that were cancelled (final turn cancelled). If None, the phase has not completed.",
    )

    # Timeout/cancellation fields
    timeout_triggered: bool = Field(
        default=False,
        description="Whether the phase timed out (only valid if the phase is time based and has finished sending credits).",
    )
    grace_period_timeout_triggered: bool = Field(
        default=False,
        description="Whether the phase timed out within the grace period (only valid if the phase is time based and has finished sending credits).",
    )
    was_cancelled: bool = Field(
        default=False, description="Whether the credit phase was cancelled."
    )

    @property
    def is_started(self) -> bool:
        return self.start_ns is not None

    @property
    def is_sending_complete(self) -> bool:
        return self.sent_end_ns is not None

    @property
    def is_requests_complete(self) -> bool:
        return self.requests_end_ns is not None


class CreditPhaseStats(BasePhaseStats):
    """Immutable model for phase credit stats. This is used to track the progress of the credit phases."""

    record_type: ClassVar[str] = "credit_phase_stats"

    model_config = ConfigDict(frozen=True)

    # Credit progress fields
    requests_sent: int = Field(
        default=0,
        ge=0,
        description="The number of requests sent to the workers so far.",
    )
    requests_completed: int = Field(
        default=0,
        ge=0,
        description="The number of requests completed by the workers so far.",
    )
    requests_cancelled: int = Field(
        default=0,
        ge=0,
        description="The number of requests cancelled by the workers so far.",
    )
    request_errors: int = Field(
        default=0,
        ge=0,
        description="The number of requests that returned errors from the workers so far.",
    )
    sent_sessions: int = Field(
        default=0,
        ge=0,
        description="The number of unique user sessions that have been sent so far.",
    )
    completed_sessions: int = Field(
        default=0,
        ge=0,
        description="The number of unique user sessions that have been completed so far.",
    )
    cancelled_sessions: int = Field(
        default=0,
        ge=0,
        description="The number of unique user sessions that were cancelled (final turn cancelled).",
    )
    total_session_turns: int = Field(
        default=0,
        ge=0,
        description="The total number of turns in all user sessions so far (not all have been sent or returned yet).",
    )

    @property
    def in_flight_sessions(self) -> int:
        """Sessions started but not yet finished (no final turn returned)."""
        return self.sent_sessions - self.completed_sessions - self.cancelled_sessions

    @property
    def in_flight_requests(self) -> int:
        """Calculate the number of in-flight requests (sent but not completed).

        NOTE: This can also be seen as the current actual "concurrency" value for the phase
        """
        return self.requests_sent - self.requests_completed - self.requests_cancelled

    @property
    def requests_elapsed_time(self) -> float:
        """Get the elapsed time."""
        if self.start_ns is None:
            return 0.0
        if self.requests_end_ns is not None:
            return (self.requests_end_ns - self.start_ns) / NANOS_PER_SECOND
        return (time.time_ns() - self.start_ns) / NANOS_PER_SECOND

    @property
    def requests_error_percent(self) -> float:
        """The error percentage of the requests completed."""
        if self.final_requests_completed is not None:
            if self.final_requests_completed == 0:
                return 0.0
            return (self.final_request_errors / self.final_requests_completed) * 100

        if self.requests_completed == 0:
            return 0.0
        return (self.request_errors / self.requests_completed) * 100

    @property
    def requests_progress_percent(self) -> float | None:
        """The progress percentage of the requests completed."""

        if self.start_ns is None:
            return None

        if self.is_requests_complete:
            return 100

        percentages = []
        pct_complete, pct_time_elapsed = 0, 0
        if self.total_expected_requests:
            pct_complete = (
                self.requests_completed / self.total_expected_requests
            ) * 100
            percentages.append(pct_complete)
        if self.expected_duration_sec:
            elapsed_ns = time.time_ns() - self.start_ns
            expected_duration_ns = self.expected_duration_sec * NANOS_PER_SECOND
            pct_time_elapsed = (elapsed_ns / expected_duration_ns) * 100
            percentages.append(pct_time_elapsed)
        if self.expected_num_sessions:
            pct_sessions_complete = (
                self.completed_sessions / self.expected_num_sessions
            ) * 100
            percentages.append(pct_sessions_complete)

        if not percentages:
            return None

        # Return the highest percentage, because the first condition met
        # will win when multiple conditions exist. Cap at 100%.
        return min(max(percentages), 100)


class PhaseRecordsStats(BasePhaseStats):
    """Immutable model for phase records stats. This is used to track the progress of the records phases."""

    model_config = ConfigDict(frozen=True)

    # Timestamp fields
    records_end_ns: int | None = Field(
        default=None,
        ge=0,
        description="The time at which the phase completed processing all records (time.time_ns()).",
    )

    # Progress fields
    success_records: int = Field(
        default=0, ge=0, description="The number of records processed successfully."
    )
    error_records: int = Field(
        default=0, ge=0, description="The number of records processed with errors."
    )

    @property
    def total_records(self) -> int:
        """The total number of records processed (success + errors)."""
        return self.success_records + self.error_records

    @property
    def records_elapsed_time(self) -> float:
        """Get the elapsed time."""
        if self.start_ns is None:
            return 0.0
        if self.records_end_ns is not None:
            return (self.records_end_ns - self.start_ns) / NANOS_PER_SECOND
        return (time.time_ns() - self.start_ns) / NANOS_PER_SECOND

    @property
    def records_error_percent(self) -> float:
        """The error percentage of the records processed."""
        if self.total_records == 0:
            return 0.0
        return (self.error_records / self.total_records) * 100

    @property
    def records_progress_percent(self) -> float | None:
        """The progress percent of the records processed."""
        if self.final_requests_completed:
            return (self.total_records / self.final_requests_completed) * 100

        if self.total_expected_requests:
            return (self.total_records / self.total_expected_requests) * 100

        return None

    @property
    def is_records_complete(self) -> bool:
        return self.records_end_ns is not None


class ProcessingStats(AIPerfBaseModel):
    """Model for phase processing stats. How many requests were processed and
    how many errors were encountered."""

    processed: int = Field(
        default=0, description="The number of records processed successfully"
    )
    errors: int = Field(
        default=0, description="The number of record errors encountered"
    )

    @property
    def total_records(self) -> int:
        """The total number of records (processed + errors)."""
        return self.processed + self.errors
