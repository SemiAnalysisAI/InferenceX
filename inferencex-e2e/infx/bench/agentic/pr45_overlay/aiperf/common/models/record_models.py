# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import Annotated, Any, AnyStr, ClassVar, Protocol, runtime_checkable

import orjson
from pydantic import (
    ConfigDict,
    Field,
    NonNegativeInt,
    PlainSerializer,
    PrivateAttr,
    RootModel,
    SerializationInfo,
    SerializeAsAny,
    field_validator,
    model_serializer,
)
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic.functional_validators import AfterValidator

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.constants import STAT_KEYS
from aiperf.common.enums import (
    CacheBustTarget,
    CreditPhase,
    MetricConsoleGroup,
    MetricValueTypeT,
)
from aiperf.common.exceptions import InvalidInferenceResultError
from aiperf.common.finite import FiniteFloat
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.models.branch_stats import BranchStats
from aiperf.common.models.dataset_models import Turn
from aiperf.common.models.error_models import ErrorDetails, ErrorDetailsCount
from aiperf.common.models.export_models import JsonMetricResult, TelemetryExportData
from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.common.models.server_metrics_models import ServerMetricsResults
from aiperf.common.models.spec_decode_models import SpecDecodeAcceptanceRecord
from aiperf.common.models.trace_models import BaseTraceData, TraceDataExport
from aiperf.common.models.usage_models import Usage
from aiperf.common.types import JsonObject, MetricTagT, PhaseKind
from aiperf.common.utils import load_json_str

_logger = AIPerfLogger(__name__)
_SSE_COMMENT_FIELD_NAME = "comment"
_SSE_DATA_FIELD_NAME = "data"
_SSE_DATA_PREFIX = "data:"
# ASCII padding only: str.strip() would also eat U+0085/U+2028/U+2029 and
# \x0b-\x0c/\x1c-\x1f, which are data, not SSE whitespace.
_SSE_PADDING = " \t"


class MetricResult(JsonMetricResult):
    """The result values of a single metric."""

    tag: MetricTagT = Field(description="The unique identifier of the metric")
    # NOTE: We do not use a MetricUnitT here, as that is harder to de-serialize from JSON strings with pydantic.
    #       If we need an instance of a MetricUnitT, lookup the unit based on the tag in the MetricRegistry.
    header: str = Field(
        description="The user friendly name of the metric (e.g. 'Inter Token Latency')"
    )
    count: int | None = Field(
        default=None,
        description="The total number of records used to calculate the metric",
    )
    current: float | None = Field(
        default=None,
        description="The most recent value of the metric (used for realtime dashboard display only)",
    )
    sum: int | float | None = Field(
        default=None,
        description="The sum of all the metric values across all records",
    )
    console_group: MetricConsoleGroup | None = Field(
        default=None,
        description="Optional console-grouping override for analyzer-injected results "
        "whose tags are not in MetricRegistry. The registered metric class's "
        "`console_group` ClassVar is the source of truth for everything else; this "
        "field is only consulted by the console exporter when a tag isn't registered. "
        "Dropped from every public dump (CSV / JSON exports / REST API); only IPC "
        "passes `context={'include_internal': True}` to keep it across process boundaries.",
    )

    @model_serializer(mode="wrap")
    def _drop_internal_fields(self, handler, info: SerializationInfo) -> dict[str, Any]:
        """Strip internal-only fields (`console_group`) from every dump unless
        the caller opts in with ``context={'include_internal': True}`` -- i.e.
        cross-process IPC. User-facing CSV/JSON/REST exports never set the
        flag, so they always see the public shape."""
        data = handler(self)
        if isinstance(data, dict) and not (
            info.context and info.context.get("include_internal")
        ):
            data.pop("console_group", None)
        return data

    def to_display_unit(self) -> MetricResult:
        """Convert the metric result to its display unit."""
        from aiperf.metrics.display_units import to_display_unit
        from aiperf.metrics.metric_registry import MetricRegistry

        return to_display_unit(self, MetricRegistry)

    def to_json_result(self) -> JsonMetricResult:
        """Convert the metric result to a JsonMetricResult.

        `count` is omitted for non-RECORD metrics (derived/aggregate scalars),
        where it would trivially be 1 and risks being misread as the request
        count. Tags from other registries (e.g. GPU telemetry) are not in
        MetricRegistry; those keep `count` as-is. Future MetricType members
        also keep `count` by default — opt them in here explicitly.
        """
        from aiperf.common.enums import MetricType
        from aiperf.metrics.metric_registry import MetricRegistry

        metric_class = MetricRegistry.get_class_or_none(self.tag)
        is_scalar = metric_class is not None and metric_class.type in {
            MetricType.AGGREGATE,
            MetricType.DERIVED,
        }

        result = JsonMetricResult(
            unit=self.unit,
            count=None if is_scalar else self.count,
        )
        for stat in STAT_KEYS:
            setattr(result, stat, getattr(self, stat, None))
        return result


class RecordData(AIPerfBaseModel):
    """Base for typed records that travel on the generic ``RecordsMessage`` envelope.

    Subclasses declare a SERIALIZED ``record_type`` discriminator (a ``Literal``
    field, not a ClassVar) so AutoRoutedModel can reconstruct the concrete type
    on the receiving side of the ZMQ boundary. A ClassVar discriminator would not
    serialize, so the receiver would see bare dicts and fail to route them. This
    mirrors the ``BaseTraceData`` / ``trace_type`` discriminated-union pattern.

    ``RecordData.from_json(dict)`` routes to the registered subclass by its
    ``record_type`` value; ``getattr(instance, "record_type")`` returns the field
    value, so the records-manager routing table keys off instances unchanged.
    """

    discriminator_field: ClassVar[str] = "record_type"
    # The base RecordData has no standalone shape (typed fields live on subclasses),
    # so an unregistered record_type must raise rather than silently degrade.
    strict_routing: ClassVar[bool] = True

    record_type: str = Field(
        description="Discriminator: the record_type channel this record routes on.",
    )


class MetricValue(AIPerfBaseModel):
    """The value of a metric converted to display units for export."""

    value: MetricValueTypeT
    unit: str


class MetricRecordMetadata(AIPerfBaseModel):
    """The metadata of a metric record for export."""

    session_num: int = Field(
        ...,
        description="The sequential number of the session in the benchmark. For single-turn datasets, this will be the"
        " request index. For multi-turn datasets, this will be the session index.",
    )
    x_request_id: str | None = Field(
        default=None,
        description="The X-Request-ID header of the request. This is a unique ID for the request.",
    )
    x_correlation_id: str | None = Field(
        default=None,
        description="The X-Correlation-ID header of the request. This is a shared ID for each user session/conversation in multi-turn.",
    )
    conversation_id: str | None = Field(
        default=None,
        description="The ID of the conversation (if applicable). This can be used to lookup the original request data from the inputs.json file.",
    )
    turn_index: int | None = Field(
        default=None,
        description="The index of the turn in the conversation (if applicable). This can be used to lookup the original request data from the inputs.json file.",
    )
    source_trace_id: str | None = Field(
        default=None,
        description=(
            "Original trace/conversation id that produced this reconstructed "
            "request, when provided by the dataset loader."
        ),
    )
    source_outer_idx: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Zero-based index of the original top-level source request within "
            "source_trace_id, when provided by the dataset loader."
        ),
    )
    source_inner_idx: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Zero-based index within the nested source request list identified "
            "by source_outer_idx, when provided by the dataset loader."
        ),
    )
    source_kind: str | None = Field(
        default=None,
        description=(
            "Loader-specific label for the request's origin, when provided by "
            "the dataset loader: the Weka reconstruction kind (weka_main, "
            "weka_flat) or the SPEED-Bench row's benchmark category (coding, "
            "math, low_entropy). Lets a consumer group records by their source "
            "without re-reading the input dataset."
        ),
    )
    credit_issued_ns: int | None = Field(
        default=None,
        description="Wall clock timestamp (time.time_ns) when the credit was issued by the rate limiter. "
        "This is the control point for accurate rate measurement, before ZeroMQ transit to workers.",
    )
    request_start_ns: int = Field(
        ...,
        description="The wall clock timestamp of the request start time measured as time.time_ns().",
    )
    request_ack_ns: int | None = Field(
        default=None,
        description="The wall clock timestamp of the request acknowledgement from the server, measured as time.time_ns(), if applicable. "
        "This is only applicable to streaming requests, and servers that send 200 OK back immediately after the request is received.",
    )
    request_end_ns: int = Field(
        ...,
        description="The wall clock timestamp of the request end time measured as time.time_ns(). If the request failed, "
        "this will be the time of the error.",
    )
    worker_id: str = Field(
        ..., description="The ID of the AIPerf worker that processed the request."
    )
    record_processor_id: str = Field(
        ...,
        description="The ID of the AIPerf record processor that processed the record.",
    )
    benchmark_phase: CreditPhase = Field(
        ...,
        description="The benchmark phase of the record, either warmup or profiling.",
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
    was_cancelled: bool = Field(
        default=False,
        description="Whether the request was cancelled during execution.",
    )
    cancellation_time_ns: int | None = Field(
        default=None,
        description="The wall clock timestamp of the request cancellation time measured as time.time_ns(), if applicable. "
        "This is only applicable to requests that were cancelled.",
    )
    context_overflow_skip: bool = Field(
        default=False,
        description="True iff the record was classified as a context-overflow event "
        "AND the active scenario uses AGENTIC_REPLAY timing. Set on the worker side "
        "by ``RecordProcessor`` and consumed by ``RecordsManager``: the record still "
        "increments ``total_records`` (so the records-side counter stays in lockstep "
        "with the credit-side ``final_requests_completed`` and the completion barrier "
        "converges), but it is skipped from the error tracker, the per-record "
        "accumulators (latency/throughput/etc.), and the stream exporters. Net effect: "
        "the overflow event doesn't show up in any user-facing metric, while the run "
        "still terminates cleanly.",
    )
    agent_depth: int = Field(
        default=0,
        description="The DAG agent depth of the session that produced this record. 0 for root sessions, "
        "incremented by 1 for each nested subagent fork. Use to filter records by DAG layer.",
    )
    parent_correlation_id: str | None = Field(
        default=None,
        description="The x_correlation_id of the parent session that spawned this record's session via a "
        "DAG subagent fork. None for root sessions. Use to group sibling branches of the same DAG.",
    )
    root_correlation_id: str | None = Field(
        default=None,
        description="The x_correlation_id of the depth-0 root of this record's session TREE. Stable "
        "across the whole tree (root + every descendant subagent at any depth); equals x_correlation_id "
        "for a root session. Groups every record of one agentic session (root + subagents) under a single "
        "lane and lets analysis reconstruct exactly-N session-tree concurrency.",
    )


class TimesliceResult(AIPerfBaseModel):
    """Per-timeslice results: window bounds + metric results.

    Combines ``start_ns`` / ``end_ns`` / ``is_complete`` with the metric
    results computed for that slice. Stored in chronological order in
    :attr:`ProfileResults.timeslices`; position in the parent list is the
    slice's chronological index, matching the ``BaseTimeslice`` wire shape.

    ``is_complete`` is ``None`` for fully-closed windows (space-efficient
    default matching ``BaseTimeslice``) and ``False`` for the trailing
    partial window when the benchmark stopped before the next slice
    boundary. Partial slices should be excluded from aggregate statistics
    to avoid skewing rate calculations.

    Metric results are keyed by metric tag for direct lookup. The
    JSON/CSV exporters flatten them to per-tag fields in the wire format.
    """

    start_ns: int = Field(
        ge=0,
        description="Timeslice start timestamp in nanoseconds",
    )
    end_ns: int = Field(
        ge=0,
        description="Timeslice end timestamp in nanoseconds",
    )
    is_complete: bool | None = Field(
        default=None,
        description="False for partial timeslices (typically the final slice). "
        "None for complete timeslices covering the full configured duration.",
    )
    metric_results: dict[MetricTagT, MetricResult] = Field(
        default_factory=dict,
        description="Metric results computed for this timeslice's window, "
        "keyed by metric tag.",
    )

    @field_validator("metric_results", mode="before")
    @classmethod
    def _coerce_metric_results(cls, value: Any) -> Any:
        """Accept ``list[MetricResult]`` for ergonomic construction and rekey
        by ``tag``. Existing dict input passes through unchanged."""
        if isinstance(value, list):
            return {r.tag: r for r in value}
        return value


class PhaseProfileResults(AIPerfBaseModel):
    """Metric summary for one concrete named phase."""

    phase_index: int | None = Field(
        default=None, ge=0, description="Absolute index in the ordered phases list."
    )
    profiling_index: int | None = Field(
        default=None,
        ge=0,
        description="Index among profiling-kind phases; None for warmup.",
    )
    phase_name: str = Field(description="User-provided unique phase name.")
    phase_kind: PhaseKind = Field(
        description="Phase semantic kind: warmup or profiling."
    )
    records: list[MetricResult] = Field(
        default_factory=list, description="Metric results scoped to this phase."
    )
    start_ns: int | None = Field(
        default=None,
        ge=0,
        description="Phase start time in nanoseconds, when available.",
    )
    end_ns: int | None = Field(
        default=None,
        ge=0,
        description="Phase request completion time in nanoseconds, when available.",
    )
    baseline_start_ns: int | None = Field(
        default=None,
        ge=0,
        description="Phase START baseline request publish time in nanoseconds, when available.",
    )
    baseline_end_ns: int | None = Field(
        default=None,
        ge=0,
        description="Phase END baseline request publish time in nanoseconds, when available.",
    )
    was_cancelled: bool = Field(
        default=False, description="Whether this phase was cancelled early."
    )
    successful_request_count: int = Field(
        default=0, ge=0, description="Successful records for this phase."
    )
    error_request_count: int = Field(
        default=0, ge=0, description="Errored records for this phase."
    )
    error_summary: list[ErrorDetailsCount] = Field(
        default_factory=list,
        description="A list of the unique phase error details and their counts",
    )
    telemetry_results: TelemetryExportData | None = Field(
        default=None,
        description="GPU telemetry summary scoped to this concrete phase.",
    )
    server_metrics_results: ServerMetricsResults | None = Field(
        default=None,
        description="Server metrics summary scoped to this concrete phase.",
    )
    telemetry_warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal telemetry warnings for phase artifact export.",
    )
    server_metrics_warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal server metrics warnings for phase artifact export.",
    )


class ProfileMetricDurationCoverage(AIPerfBaseModel):
    """Observed profiling-metric coverage relative to a configured phase duration."""

    phase_name: str = Field(description="Name of the profiling phase checked.")
    expected_duration_seconds: float = Field(
        gt=0.0, description="Configured profiling duration in seconds."
    )
    required_ratio: float = Field(
        gt=0.0, le=1.0, description="Minimum required metric coverage ratio."
    )
    ttft_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of the phase covered by TTFT observations.",
    )
    inter_token_latency_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of the phase covered by inter-token-latency observations.",
    )
    streaming_content_ratio: FiniteFloat = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of the phase reached by parsed streaming content, "
        "including requests cancelled before producing a metric record.",
    )

    @property
    def passed(self) -> bool:
        """Return whether latency or streaming content proves late phase activity."""
        return (
            self.ttft_ratio >= self.required_ratio
            or self.inter_token_latency_ratio >= self.required_ratio
            or self.streaming_content_ratio >= self.required_ratio
        )


class ProfileResults(AIPerfBaseModel):
    """The results of a profile run."""

    records: list[MetricResult] | None = Field(
        default=None, description="The records of the profile results"
    )
    warmup_records: list[MetricResult] | None = Field(
        default=None,
        description="Metric results computed only from warmup-phase records. "
        "Top-level records remain profiling-only.",
    )
    timeslices: list[TimesliceResult] | None = Field(
        default=None,
        description="Per-timeslice results in chronological order. Each entry "
        "bundles the slice's window bounds (start_ns, end_ns, is_complete) "
        "with its metric results. Position in the list is the slice's "
        "chronological index. Produced by the MetricsAccumulator engine.",
    )
    total_expected: int | None = Field(
        default=None,
        description="The total number of inference requests expected to be made (if known)",
    )
    completed: int = Field(
        ..., description="The number of inference requests completed"
    )
    start_ns: int = Field(
        ..., description="The start time of the profile run in nanoseconds"
    )
    end_ns: int = Field(
        ..., description="The end time of the profile run in nanoseconds"
    )
    was_cancelled: bool = Field(
        default=False,
        description="Whether the profile run was cancelled early",
    )
    successful_request_count: int = Field(
        default=0,
        ge=0,
        description="The number of inference requests that returned successful responses",
    )
    error_request_count: int = Field(
        default=0,
        ge=0,
        description="The number of inference requests that returned errors",
    )
    error_summary: list[ErrorDetailsCount] = Field(
        default_factory=list,
        description="A list of the unique error details and their counts",
    )
    branch_stats: BranchStats | None = Field(
        default=None,
        description="DAG branch orchestration counters for the run. "
        "None for non-DAG runs; a populated snapshot for DAG-shaped "
        "runs. Forwarded to profile_export_aiperf.json under the "
        "``branch_stats`` key when present.",
    )
    context_overflow_count: int = Field(
        default=0,
        ge=0,
        description="Count of AGENTIC_REPLAY context-overflow records skipped from "
        "normal metric accumulation and stream export, retained only for "
        "aggregate runtime submission validation.",
    )
    metric_duration_coverage: list[ProfileMetricDurationCoverage] = Field(
        default_factory=list,
        description="Post-run latency-signal duration-coverage checks, when enabled.",
    )
    runtime_submission_invalid_reasons: list[str] = Field(
        default_factory=list,
        description="Runtime reason tags that invalidate a scenario submission.",
    )
    phase_records: list[PhaseProfileResults] | None = Field(
        default=None,
        description="Internal per-phase metric summaries used for phase artifacts.",
    )
    pooled_spec_decode_acceptance_histogram: (
        dict[NonNegativeInt, NonNegativeInt] | None
    ) = Field(
        default=None,
        description="Run-level pooled speculative-decoding acceptance histogram "
        "for this phase: accepted-draft count j mapped to the total number of "
        "verify steps that accepted exactly j draft tokens, summed elementwise "
        "across every request. Its counts sum to ``total_spec_decode_steps``. "
        "None when spec decode is off; forwarded to the aggregate JSON export and "
        "the dedicated console histogram line.",
    )

    def get(self, tag: MetricTagT) -> MetricResult | None:
        """Get a metric result by tag, if it exists."""
        for record in self.records or []:
            if record.tag == tag:
                return record
        return None


class ProcessRecordsResult(AIPerfBaseModel):
    """Result of the process records command."""

    results: ProfileResults = Field(..., description="The profile results")
    errors: list[ErrorDetails] = Field(
        default_factory=list,
        description="Any error that occurred while processing the profile results",
    )
    fatal_errors: list[ErrorDetails] = Field(
        default_factory=list,
        description="Post-processing validation errors that require a non-zero exit.",
    )

    def get(self, tag: MetricTagT) -> MetricResult | None:
        """Get a metric result by tag, if it exists."""
        return self.results.get(tag)


################################################################################
# Inference Client Response Models
################################################################################


@runtime_checkable
class InferenceServerResponse(Protocol):
    """Protocol for inference server response objects.

    Defines the interface for response objects that can parse themselves
    into different formats. Any object implementing these methods can be
    used as a response in the inference pipeline.

    This protocol-based approach allows for:
    - Duck typing (structural subtyping)
    - Easier testing with mocks
    - Flexibility in implementation
    - No concrete inheritance required
    """

    perf_ns: int
    """Timestamp of the response in nanoseconds (perf_counter_ns)."""

    def get_raw(self) -> Any | None:
        """Get the raw representation of the response.

        Returns:
            Raw response data or None
        """
        ...

    def get_text(self) -> str | None:
        """Get the text representation of the response.

        Returns:
            Text content or None
        """
        ...

    def get_json(self) -> JsonObject | None:
        """Get the JSON representation of the response.

        Automatically parses text content as JSON if applicable.

        Returns:
            Parsed JSON dict or None if parsing fails
        """
        ...


@dataclass(slots=True)
class SSEField:
    """Lightweight field in an SSE message.

    Using dataclass(slots=True) instead of Pydantic for memory efficiency during
    high-throughput streaming. Each SSE message can have multiple fields, and with
    thousands of concurrent requests each generating hundreds of chunks, Pydantic overhead
    was the #1 memory allocator.
    """

    name: str
    """The name of the field. e.g. 'data', 'event', 'id', 'retry', 'comment'."""

    value: str | None = None
    """The value of the field."""


@dataclass(slots=True)
class TextResponse:
    """Raw text response from an inference client including an optional content type."""

    # Reject extra fields so Pydantic's union discrimination (e.g. in
    # RequestRecord.responses) doesn't match the wrong dataclass type.
    __pydantic_config__ = ConfigDict(extra="forbid")

    perf_ns: int
    """The performance timestamp of the response in nanoseconds (perf_counter_ns)."""

    text: str
    """The raw text body of the response."""

    content_type: str | None = None
    """The content type of the response. e.g. 'text/plain', 'application/json'."""

    def get_raw(self) -> Any | None:
        """Get the raw representation of the response."""
        return self.text

    def get_text(self) -> str | None:
        """Get the text representation of the response."""
        return self.text

    def get_json(self) -> JsonObject | None:
        """Get the JSON representation of the response."""
        try:
            if not self.text:
                return None
            return load_json_str(self.text)
        except orjson.JSONDecodeError:
            return None


# Nested stdlib dataclasses ignore byte encoding settings; use Pydantic here.
# Reject extra fields so response unions cannot select the wrong dataclass.
@pydantic_dataclass(
    slots=True,
    config=ConfigDict(extra="forbid", ser_json_bytes="base64", val_json_bytes="base64"),
)
class BinaryResponse:
    """Raw binary response from an inference client for non-text content types."""

    perf_ns: int
    """The performance timestamp of the response in nanoseconds (perf_counter_ns)."""

    raw_bytes: bytes
    """The original response bytes; JSON encodes them as URL-safe base64."""

    content_type: str | None = None
    """The content type of the response. e.g. 'video/mp4', 'application/octet-stream'."""

    def get_raw(self) -> Any | None:
        """Get the raw representation of the response."""
        return self.raw_bytes

    def get_text(self) -> str | None:
        """Get the text representation of the response."""
        return None

    def get_json(self) -> JsonObject | None:
        """Get the JSON representation of the response."""
        return None


@dataclass(slots=True)
class SSEMessage:
    """Individual SSE message from an SSE stream. Delimited by \\n\\n.

    Uses dataclass(slots=True) instead of Pydantic for ~6x faster construction
    and ~10x smaller memory footprint per instance. Pydantic handles serialization
    and deserialization automatically when this appears inside Pydantic model fields.
    """

    # Reject extra fields so Pydantic's union discrimination (e.g. in
    # RequestRecord.responses) doesn't match the wrong dataclass type.
    __pydantic_config__ = ConfigDict(extra="forbid")

    perf_ns: int
    """The performance timestamp of the message in nanoseconds (perf_counter_ns)."""

    packets: list[SSEField] = field(default_factory=list)
    """The parsed SSE fields (data, event, id, retry, comment) in this message."""

    @classmethod
    def parse(cls, raw_message: AnyStr, perf_ns: int) -> SSEMessage:
        """Parse a raw SSE message into an SSEMessage object.

        Parsing logic based on the official HTML SSE Living Standard:
        https://html.spec.whatwg.org/multipage/server-sent-events.html#parsing-an-event-stream

        Args:
            raw_message: The raw SSE message to parse. Can be a string or a bytes object.
            perf_ns: The performance timestamp of the response.

        Returns:
            The parsed SSEMessage.
        """
        if isinstance(raw_message, bytes):
            raw_message = raw_message.decode("utf-8")

        if (
            "\n" not in raw_message
            and "\r" not in raw_message
            and raw_message.startswith(_SSE_DATA_PREFIX)
        ):
            return cls(
                perf_ns=perf_ns,
                packets=[
                    SSEField(
                        name=_SSE_DATA_FIELD_NAME,
                        value=raw_message[len(_SSE_DATA_PREFIX) :].strip(_SSE_PADDING),
                    )
                ],
            )

        message = cls(perf_ns=perf_ns)
        # SSE fields end at CR/LF; Unicode line separators belong to the data.
        if "\r" in raw_message:
            raw_message = raw_message.replace("\r\n", "\n").replace("\r", "\n")
        for line in raw_message.split("\n"):
            if not (line := line.strip(_SSE_PADDING)):
                continue

            prev_value = message.packets[-1].value if message.packets else None
            # Detect continuation: if the previous packet's value is an incomplete
            # JSON object (starts with '{' but doesn't end with '}') and this line
            # isn't a new data field, the server embedded a literal newline in the
            # JSON value. Append this line as a continuation. This can happen when
            # ignore_eos=True and the model emits weird tokens.
            if (
                prev_value
                and prev_value.startswith("{")
                and not prev_value.endswith("}")
                and not line.startswith("data:")
            ):
                # Use \\n (JSON escape) not \n (raw newline) — the original raw 0x0A
                # byte is illegal in JSON strings; \n is the valid encoding.
                message.packets[-1].value = f"{prev_value}\\n{line}"
                continue

            parts = line.split(":", 1)
            if len(parts) < 2:
                # Fields without a colon have no value, so the whole line is the field name
                message.packets.append(
                    SSEField(name=parts[0].strip(_SSE_PADDING), value=None)
                )
                continue

            field_name, value = parts

            if field_name == "":
                # Field name is empty, so this is a comment
                field_name = _SSE_COMMENT_FIELD_NAME

            # Spec says strip only one leading space; we strip all ASCII padding
            # to normalize inconsistent servers for downstream exact comparisons
            # (e.g. "[DONE]", "error").
            message.packets.append(
                SSEField(
                    name=field_name.strip(_SSE_PADDING),
                    value=value.strip(_SSE_PADDING),
                )
            )

        return message

    def extract_data_content(self) -> str:
        """Extract and combine the data contents from the SSE message.

        Per the SSE spec, multiple data fields are combined and delimited by a single newline.

        Returns:
            str: The combined data contents of the SSE message, joined by newlines.
        """
        if len(self.packets) == 1 and self.packets[0].name == _SSE_DATA_FIELD_NAME:
            return self.packets[0].value or ""

        return "\n".join(
            packet.value
            for packet in self.packets
            if packet.name == _SSE_DATA_FIELD_NAME and packet.value
        )

    def get_raw(self) -> Any | None:
        """Get the raw representation of the SSE message."""
        return self.packets

    def get_text(self) -> str | None:
        """Get the text representation of the SSE message."""
        if data_content := self.extract_data_content():
            return data_content
        return None

    def get_json(self) -> JsonObject | None:
        """Get the JSON representation of the response."""
        data_content = None
        try:
            if len(self.packets) == 1 and self.packets[0].name == _SSE_DATA_FIELD_NAME:
                data_content = self.packets[0].value
            else:
                data_content = self.get_text()
            if data_content in ("", None, "[DONE]"):
                return None
            return load_json_str(data_content)
        except orjson.JSONDecodeError:
            return None


class RecordContext(AIPerfBaseModel):
    """Slim per-record context attached to ``RequestRecord``.

    Carries *only* the fields the record-processor pipeline reads
    post-transport. The full ``RequestInfo`` (model endpoint, transport
    headers, URL params, pre-send-only timing fields) stays on the worker
    and never crosses ZMQ — eliminating ~500-900 bytes of dead weight per
    record at high request rates.

    ``RequestInfo`` inherits from this class so production-side callers
    that build a full request info can still assign it to
    ``RequestRecord.request_info`` (it IS a ``RecordContext``); the worker's
    ``inference_client._enrich_request_record`` explicitly down-casts to a
    pure ``RecordContext`` before the ZMQ hop so the subclass extras are
    dropped.

    Disambiguation note: aiperf has four "Context" types that are easy to
    confuse but live in distinct subsystems:

    - ``RecordContext`` (this class): per-record fields the record-processor
      reads post-transport; rides on every ``RequestRecord``.
    - ``CreditContext`` (``aiperf.credit.structs``): timing-side struct the
      credit issuer attaches to a credit before the worker picks it up.
    - ``PhaseCallbackContext`` (``aiperf.credit.callback_handler``): inputs
      passed to credit-phase begin/end callbacks (phase + stats snapshot).
    - ``MetricContext`` (``aiperf.metrics.prometheus_formatter``):
      NamedTuple of label values used to format a single Prometheus sample.

    They do not interconvert; pick the one named for the subsystem you are in.
    """

    # --- Identity / routing (read by MetricRecordMetadata builder) -----------

    credit_num: int = Field(
        ...,
        ge=0,
        description="The sequential number of the credit in the credit phase. This is used to track the progress of the credit phase,"
        " as well as the order that requests are sent in.",
    )
    credit_phase: CreditPhase = Field(
        ...,
        description="The type of credit phase (either warmup or profiling)",
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
    conversation_id: str = Field(
        ...,
        description="The ID of the conversation (if applicable).",
    )
    turn_index: int = Field(
        ...,
        description="The index of the turn in the conversation (if applicable).",
    )
    source_trace_id: str | None = Field(
        default=None,
        description=(
            "Original trace/conversation id that produced this reconstructed "
            "request, when provided by the dataset loader."
        ),
    )
    source_outer_idx: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Zero-based index of the original top-level source request within "
            "source_trace_id, when provided by the dataset loader."
        ),
    )
    source_inner_idx: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Zero-based index within the nested source request list identified "
            "by source_outer_idx, when provided by the dataset loader."
        ),
    )
    source_kind: str | None = Field(
        default=None,
        description=(
            "Loader-specific label for the request's origin, when provided by "
            "the dataset loader: the Weka reconstruction kind (weka_main, "
            "weka_flat) or the SPEED-Bench row's benchmark category (coding, "
            "math, low_entropy). Lets a consumer group records by their source "
            "without re-reading the input dataset."
        ),
    )
    x_request_id: str = Field(
        ...,
        description="The X-Request-ID header of the request. This is a unique ID for the request.",
    )
    x_correlation_id: str = Field(
        ...,
        description="The X-Correlation-ID header of the request. This is the ID of the credit drop.",
    )
    credit_issued_ns: int | None = Field(
        default=None,
        ge=0,
        description="Wall clock timestamp (time.time_ns) when the credit was issued by the rate limiter. "
        "This is the control point for accurate rate measurement, before ZeroMQ transit to workers.",
    )

    # --- DAG ------------------------------------------------------------------

    agent_depth: int = Field(
        default=0,
        description="The DAG agent depth of the session that produced this request. 0 for root sessions, "
        "incremented by 1 for each nested subagent fork. Sourced from the originating Credit.",
    )
    parent_correlation_id: str | None = Field(
        default=None,
        description="The x_correlation_id of the parent session that spawned this session via a DAG "
        "subagent fork. None for root sessions. Sourced from the originating Credit.",
    )
    root_correlation_id: str | None = Field(
        default=None,
        description="The x_correlation_id of the depth-0 root of this record's session TREE. "
        "Stable across the whole tree (root + every descendant subagent at any depth); equals "
        "x_correlation_id for a root session. Sourced from the originating Credit. Use to group "
        "every record of one agentic session (root + subagents) and to reconstruct exactly-N "
        "session-tree concurrency in analysis.",
    )

    # --- Hoisted metric inputs (avoid shipping full Turn structs) -------------

    payload_bytes: bytes | None = Field(
        default=None,
        description="Canonical pre-encoded JSON bytes of the request body sent to the server. "
        "Populated by ``inference_client`` before transport dispatch. Used by the raw-record "
        "exporter to replay the exact wire payload, and tokenised by the record processor.",
    )
    max_tokens: int | None = Field(
        default=None,
        description="``max_tokens`` from the originating turn. Populated at record-enrichment "
        "time so the record processor reads it directly off the record without the full ``turns`` "
        "list on the wire.",
    )
    audio_duration_seconds: float | None = Field(
        default=None,
        description="``audio_duration_seconds`` from the originating turn. Populated at "
        "record-enrichment time so the record processor reads it directly off the record without "
        "the full ``turns`` list on the wire. None for non-ASR requests.",
    )
    scheduled_send_ms: FiniteFloat | None = Field(
        default=None,
        description="Absolute schedule timestamp (ms, schedule-relative) from the "
        "dispatched turn's ``Turn.timestamp``. Populated at record-enrichment time "
        "so fixed-schedule replay lag metrics can read it off the slim record "
        "without the full ``turns`` list on the wire. None for delay-scheduled "
        "continuation turns and non-fixed-schedule datasets.",
    )

    # --- Cache-bust marker (sourced from Credit, exported in raw JSONL) -------

    cache_bust_marker: str | None = Field(
        default=None,
        description="Pre-rendered cache-bust marker text for this request, "
        "sourced from ``Credit.cache_bust_marker``. Already includes whitespace "
        "boundaries. None when the cache-bust feature is disabled or no marker "
        "applied to this request. Exported in the raw JSONL so a replay tool "
        "can correlate the inserted bytes with the originating session.",
    )
    cache_bust_target: CacheBustTarget | None = Field(
        default=None,
        description="Where the marker was injected for this request, sourced "
        "from ``Credit.cache_bust_target``. None when cache-bust is disabled. "
        "Pairs with ``cache_bust_marker`` for raw-JSONL provenance.",
    )


class RequestInfo(RecordContext):
    """Full request info used Worker-side for transport dispatch.

    Extends ``RecordContext`` with pre-send-only fields that never need to
    cross the ZMQ hop to the record processor: ``ModelEndpointInfo``
    (URLs / headers / extras), transport timing (``drop_perf_ns``,
    ``cancel_after_ns``), round-robin URL index, and the
    connection-lease-release marker. ``inference_client`` builds these
    on-the-fly during transport dispatch; ``_enrich_request_record``
    down-casts to a pure ``RecordContext`` before attaching to the record.
    """

    model_endpoint: ModelEndpointInfo = Field(
        ...,
        description="The model endpoint that the request was sent to.",
    )
    turns: list[Turn] = Field(
        default_factory=list,
        description="The actual turns of the request, consumed by "
        "``format_payload`` to build the wire body. Lives on ``RequestInfo`` "
        "(not ``RecordContext``) so the full Turn list never crosses the "
        "ZMQ hop to the record processor — only the canonical "
        "``payload_bytes`` travel.",
    )
    system_message: str | None = Field(
        default=None,
        description="Optional shared system message extracted from "
        "``Conversation.system_message`` at request time. Consumed by the "
        "endpoint's ``format_payload`` (or top-level ``instructions`` on the "
        "Responses API) and inlined into ``payload_bytes`` before transport; "
        "lives on ``RequestInfo`` because the record processor reads only "
        "``payload_bytes`` downstream.",
    )
    user_context_message: str | None = Field(
        default=None,
        description="Optional per-conversation user context message extracted "
        "from ``Conversation.user_context_message`` at request time. Same "
        "inlining contract as ``system_message``.",
    )
    endpoint_headers: dict[str, str] = Field(
        default_factory=dict,
        description="Endpoint-specific headers (auth, API keys, custom headers).",
    )
    endpoint_params: dict[str, str] = Field(
        default_factory=dict,
        description="Endpoint-specific URL query parameters.",
    )
    cancel_after_ns: int | None = Field(
        default=None,
        ge=0,
        description="The delay in nanoseconds after which the request should be cancelled, or None if the request should not be cancelled.",
    )
    drop_perf_ns: int | None = Field(
        default=None,
        ge=0,
        description="The time in nanoseconds (perf_counter_ns) when the credit was dropped by the timing manager. "
        "This is used to calculate the credit drop latency.",
    )
    is_final_turn: bool = Field(
        default=True,
        description="Whether this is the final turn in the conversation. "
        "Used by per-conversation connection strategy to release the connection lease.",
    )
    url_index: int | None = Field(
        default=None,
        ge=0,
        description="Index of the URL to use when multiple --url values are configured. "
        "None means use the default (first) URL. Used for round-robin load balancing.",
    )
    previous_response_id: str | None = Field(
        default=None,
        description="Response ID from the previous turn (e.g. 'resp_<hash>') "
        "used for stateful chaining in the Responses API.",
    )
    system_prompt_in_stored_history: bool = Field(
        default=False,
        description="Whether the stored server-side history behind "
        "previous_response_id already contains the system prompt as an input "
        "item. Only meaningful when previous_response_id is set.",
    )


class RequestRecord(AIPerfBaseModel):
    """Record of a request with its associated responses."""

    _parsed_responses_cache: list[ParsedResponse] | None = PrivateAttr(default=None)
    """Memoized endpoint-final parsed responses, local to this process and never
    serialized. Populated by endpoint response processing and treated as read-only.
    Records are single-pass; responses must be complete before memoization."""

    request_info: RecordContext | None = Field(
        default=None,
        description="Slim per-record context (see ``RecordContext``). Built "
        "by ``inference_client._enrich_request_record`` from the full "
        "``RequestInfo`` that drove the request — stripping the transport-"
        "only extras so only the fields the record processor actually "
        "reads cross ZMQ.",
    )
    request_headers: dict[str, str] | None = Field(
        default=None,
        description="The headers of the request.",
    )
    model_name: str | None = Field(
        default=None,
        description="The name of the model targeted by the request.",
    )
    timestamp_ns: int = Field(
        default_factory=time.time_ns,
        description="The wall clock timestamp of the request in nanoseconds. DO NOT USE FOR LATENCY CALCULATIONS. (time.time_ns).",
    )
    start_perf_ns: int = Field(
        default_factory=time.perf_counter_ns,
        description="The start reference time of the request in nanoseconds used for latency calculations (perf_counter_ns).",
    )
    end_perf_ns: int | None = Field(
        default=None,
        description="The end time of the request in nanoseconds (perf_counter_ns).",
    )
    recv_start_perf_ns: int | None = Field(
        default=None,
        description="The start time of the streaming response in nanoseconds (perf_counter_ns).",
    )
    status: int | None = Field(
        default=None,
        description="The HTTP status code of the response.",
    )
    # TODO: Maybe we could improve this with subclassing the responses to allow for more specific types.
    #       This would allow us to remove the SerializeAsAny and use a more specific type. Look at how we handle
    #       the CommandMessage and CommandResponse classes for an example.
    # NOTE: We need to use SerializeAsAny to allow for generic subclass support
    # NOTE: The order of the types is important, as that is the order they are type checked.
    #       Start with the most specific types and work towards the most general types.
    responses: SerializeAsAny[list[SSEMessage | TextResponse | BinaryResponse]] = Field(
        default_factory=list,
        description="The raw responses received from the request.",
    )
    error: ErrorDetails | None = Field(
        default=None,
        description="The error details if the request failed.",
    )
    context_overflow: bool = Field(
        default=False,
        description="True iff this request's error response was classified "
        "as a server-side context-overflow event by "
        "``aiperf.common.scenario.is_context_overflow_response`` "
        "(InferenceX AgentX scenario, RFC section 7). Set on the worker side at "
        "response-parsing time; consumed by the ``ContextOverflowCountMetric`` "
        "aggregate counter.",
    )
    credit_drop_latency: int | None = Field(
        default=None,
        description="The latency of the credit drop in nanoseconds from when it was first received by a Worker to when the inference request was actually sent. "
        "This can be used to trace internal latency in order to identify bottlenecks or other issues.",
        ge=0,
    )
    cancellation_perf_ns: int | None = Field(
        default=None,
        ge=0,
        description="The time in nanoseconds (perf_counter_ns) when the request was actually cancelled, if applicable.",
    )
    trace_data: SerializeAsAny[BaseTraceData] | None = Field(
        default=None,
        description="Comprehensive trace data captured via a trace config. "
        "Includes detailed timing for connection establishment, DNS resolution, request/response events, etc. "
        "The type of the trace data is determined by the transport and library used.",
    )

    @field_validator("trace_data", mode="before")
    @classmethod
    def route_trace_data(cls, v: Any) -> BaseTraceData | None:
        """Route nested trace_data to correct subclass based on trace_type discriminator."""
        if isinstance(v, dict):
            return BaseTraceData.from_json(v)
        return v

    @property
    def was_cancelled(self) -> bool:
        """Check if the request was cancelled."""
        return self.cancellation_perf_ns is not None

    # TODO: Most of these properties will be removed once we have proper record handling and metrics.

    @property
    def has_error(self) -> bool:
        """Check if the request record has an error."""
        return self.error is not None

    @property
    def valid(self) -> bool:
        """Check if the request record is valid by ensuring that the start time
        and response timestamps are within valid ranges.

        Returns:
            bool: True if the record is valid, False otherwise.
        """
        return not self.has_error and (
            0 <= self.start_perf_ns < sys.maxsize
            and len(self.responses) > 0
            and all(0 < response.perf_ns < sys.maxsize for response in self.responses)
        )

    def create_error_from_invalid(self) -> None:
        """Convert any invalid request records to error records for combined processing."""
        if not self.valid and not self.has_error:
            _logger.debug(
                lambda: f"Converting invalid request record to error record: {self}"
            )
            err = InvalidInferenceResultError("Invalid inference result")
            if len(self.responses) == 0:
                err.add_note("No responses were received")
            if self.start_perf_ns <= 0 or self.start_perf_ns >= sys.maxsize:
                err.add_note(
                    f"Start perf ns timestamp is invalid: {self.start_perf_ns}"
                )
            for i, response in enumerate(self.responses):
                if response.perf_ns <= 0 or response.perf_ns >= sys.maxsize:
                    err.add_note(
                        f"Response {i} perf ns timestamp is invalid: {response.perf_ns}"
                    )
            self.error = ErrorDetails.from_exception(err)


@dataclass(slots=True)
class BaseResponseData:
    """Base class for all response data."""

    # Reject extra fields so Pydantic's union discrimination (e.g. in
    # ParsedResponse.data) doesn't match the wrong dataclass type.
    __pydantic_config__ = ConfigDict(extra="forbid")

    def get_text(self) -> str:
        """Get the text of the response."""
        return ""


@dataclass(slots=True)
class TextResponseData(BaseResponseData):
    """Parsed text response data."""

    text: str
    """The parsed text of the response."""

    def get_text(self) -> str:
        """Get the text of the response."""
        return self.text


@dataclass(slots=True)
class ReasoningResponseData(BaseResponseData):
    """Parsed reasoning response data."""

    content: str | None = None
    """The parsed content of the response."""

    reasoning: str | None = None
    """The parsed reasoning of the response."""

    def get_text(self) -> str:
        """Get the text of the response."""
        return "".join([self.reasoning or "", self.content or ""])


@dataclass(slots=True)
class ToolCallResponseData(BaseResponseData):
    """Parsed tool-call response data (streaming delta or complete message).

    Mirrors the ``ReasoningResponseData`` shape - two fields, one for the
    type's primary content and one for any prose that arrived alongside
    it. Both contribute to client-side OSL (Output Sequence Length) via
    :meth:`get_text`; the distinct fields let downstream metrics that
    want to categorise output (e.g. "what fraction of OSL was tool-call
    dispatch?") read each portion separately.
    """

    tool_call_text: str
    """Combined model-generated text from tool calls - every call's
    ``function.name`` and ``function.arguments`` concatenated in
    ``output[]`` order."""

    content: str | None = None
    """Optional prose ``content`` emitted alongside the tool calls in the
    same chunk/message. Carries the prose portion when the model talks
    while dispatching a tool (~18% of turns in agentic traffic) so
    client-side OSL counts both portions and matches the server's
    ``usage.completion_tokens``. ``None`` when the response is pure
    tool-call (no prose accompanying the dispatch)."""

    def get_text(self) -> str:
        """Return ``content`` followed by ``tool_call_text`` - the
        combined string the tokeniser sees for this response."""
        return (self.content or "") + self.tool_call_text


class RAGSources(RootModel[dict[str, Any] | list[Any]]):
    """RAG sources can be either a dictionary or list format."""


@dataclass(slots=True)
class EmbeddingResponseData(BaseResponseData):
    """Parsed embedding response data."""

    embeddings: list[list[float]]
    """The embedding vectors from the response."""


@dataclass(slots=True)
class RankingsResponseData(BaseResponseData):
    """Parsed rankings response data."""

    rankings: list[dict[str, Any]]
    """The rankings results from the response."""


@dataclass(slots=True)
class ImageRetrievalResponseData(BaseResponseData):
    """Parsed image retrieval response data."""

    data: list[dict[str, Any]]
    """The image retrieval data from the response."""

    def get_text(self) -> str:
        """Get the text of the response (empty for image retrieval)."""
        return ""


@dataclass(slots=True)
class ImageDataItem:
    """Parsed image item response data."""

    url: str | None = None
    """The URL of the generated image."""

    b64_json: str | None = None
    """The base64 encoded image."""

    revised_prompt: str | None = None
    """The revised prompt that was used for image generation."""

    partial_image_index: int | None = None
    """The index of the partial image in the response."""


@dataclass(slots=True)
class ImageResponseData(BaseResponseData):
    """Parsed image response data."""

    images: list[ImageDataItem] = field(default_factory=list)
    """The generated images from the response."""

    size: str | None = None
    """The size of the generated images."""

    quality: str | None = None
    """The quality of the generated images."""

    output_format: str | None = None
    """The output format of the generated images."""

    background: str | None = None
    """The background of the generated images."""


@dataclass(slots=True)
class VideoResponseData(BaseResponseData):
    """Parsed video generation response data.

    Matches SGLang/OpenAI VideoResponse schema for async job-based video generation.
    """

    video_id: str | None = None
    """Unique identifier for the video job."""

    object: str | None = None
    """Object type, always 'video'."""

    status: str | None = None
    """Job status: queued, in_progress, completed, failed."""

    progress: int | None = None
    """Completion percentage (0-100)."""

    url: str | None = None
    """URL to download completed video (only when status=completed)."""

    size: str | None = None
    """Video resolution (e.g., '1280x720')."""

    seconds: str | None = None
    """Video duration in seconds."""

    quality: str | None = None
    """Quality setting for the generated video."""

    model: str | None = None
    """Model used for generation."""

    created_at: int | None = None
    """Unix timestamp of job creation."""

    completed_at: int | None = None
    """Unix timestamp of job completion."""

    expires_at: int | None = None
    """Unix timestamp when video assets expire."""

    inference_time_s: float | None = None
    """Generation time in seconds (SGLang metric)."""

    peak_memory_mb: float | None = None
    """Peak memory usage in MB (SGLang metric)."""

    error: dict[str, Any] | None = None
    """Error details if job failed."""


def find_last_non_empty_usage(responses: list[ParsedResponse]) -> Usage | None:
    """Return the last response chunk's usage that has any data, walking
    the list backwards.

    Streaming chunks fall into two real-world patterns: (a) `usage = None`
    until a single final chunk carries the full usage, or (b) cumulative
    running totals where the last chunk holds the final values. Both
    collapse to "find the last non-empty Usage." A vendor never changes
    shape mid-stream and never explicitly nulls a field it had previously
    set, so a per-field walkback into earlier chunks would only matter
    for synthetic adversarial cases that don't occur in practice.

    Returns None if no chunk had any usage data. An empty Usage (`{}`) is
    falsy and treated the same as no usage.

    Used by:
    - `ParsedResponseRecord.final_usage` (cached at the record level so
      every metric reading the merged usage walks at most once per record)
    - `InferenceResultParser._compute_server_token_counts` (called before
      the record is constructed; reads input/reasoning/completion token
      counts off the same Usage to keep them mutually consistent)
    """
    for response in reversed(responses):
        if response.usage:
            return response.usage
    return None


def first_content_chunk_completion_tokens(
    responses: list[ParsedResponse],
) -> int | None:
    """Return the cumulative ``completion_tokens`` reported on the first content
    chunk (the chunk whose arrival defines TTFT).

    Requires per-chunk (``continuous_usage_stats``) usage: it walks forward to the
    first response carrying content (``data`` is not None) and reads its usage's
    ``completion_tokens``. This is deliberately the raw completion count
    (reasoning included), matching ``OutputSequenceLengthMetric`` -- OSL is
    ``output + reasoning`` = ``completion_tokens`` -- so inter-token latency
    subtracts operands in the same unit. Because per-chunk usage is cumulative,
    that value is the number of tokens delivered through the first content chunk.
    Returns ``None`` when no content chunk carries usage (e.g. the server only
    reports the final total), so callers fall back to assuming one token in the
    first chunk.
    """
    for response in responses:
        if response.data and response.usage:
            return response.usage.completion_tokens
    return None


@dataclass(slots=True)
class ParsedResponse:
    """Parsed response from a inference client."""

    perf_ns: int
    """The performance timestamp of the response in nanoseconds (perf_counter_ns)."""

    # NOTE: SerializeAsAny is used to allow for generic subclass support at runtime,
    #       allowing for user-defined response data classes.
    data: SerializeAsAny[
        ReasoningResponseData
        | TextResponseData
        | ToolCallResponseData
        | EmbeddingResponseData
        | RankingsResponseData
        | ImageRetrievalResponseData
        | ImageResponseData
        | VideoResponseData
        | BaseResponseData
        | None
    ] = None
    """The parsed response data. Can be any of the response data classes,
    or a user-defined class inheriting from BaseResponseData.
    May be None for usage-only responses in streaming mode."""

    usage: (
        Annotated[dict[str, Any], AfterValidator(Usage), PlainSerializer(dict)] | None
    ) = None
    """API-reported usage information. Structure varies by provider.
    Access token counts via properties like usage.prompt_tokens, usage.completion_tokens,
    or by accessing the usage dictionary directly."""

    sources: RAGSources | None = None
    """The sources used in the RAG query of the response. Can be a dictionary of source
    documents, a list of sources, or None. Only applicable to RAG responses."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata from the response useful for analysis (rate limits, content filters, etc.)."""

    spec_decode_stats: dict[str, Any] | None = None
    """Raw speculative-decoding payload captured from the response root (e.g.
    vLLM's ``metrics.speculative_decoding``), or None when absent. Left
    uninterpreted here; a ``SpecDecodeAdapterProtocol`` converts it into the
    engine-neutral ``SpecDecodeAcceptanceRecord`` at record-assembly time."""

    def __post_init__(self) -> None:
        # Coerce raw dicts to Usage, since dataclass __init__ doesn't run
        # Pydantic validation like BaseModel did.
        if self.usage is not None and not isinstance(self.usage, Usage):
            self.usage = Usage(self.usage)


@dataclass(slots=True)
class TokenCounts:
    """Token counts for a record."""

    input: int | None = None
    """The number of input tokens. None if token count could not be calculated."""

    output: int | None = None
    """The number of output tokens across all responses. None if token count could not be calculated."""

    reasoning: int | None = None
    """The number of reasoning tokens. None if token count could not be calculated or the model does not support reasoning."""

    first_content_chunk_tokens: int | None = None
    """The number of tokens delivered in the first content chunk -- the chunk whose
    arrival defines TTFT -- taken from the server's per-chunk usage (cumulative
    ``completion_tokens`` at that chunk). Inter-token latency subtracts this from the
    decode-token count rather than assuming exactly one token arrived first, which
    corrects the TPS/user inflation a server produces when it bundles multiple tokens
    into the first streamed chunk (e.g. TRT-LLM ``stream-interval``). ``None`` unless
    ``--per-chunk-usage`` is set and the server reports per-chunk usage; inter-token
    latency then falls back to assuming one token in the first chunk."""


@dataclass(slots=True)
class MediaCounts:
    """Multimodal content-part counts for a record.

    Computed once by ``InferenceResultParser`` at parse time via the endpoint's
    ``extract_payload_inputs`` hook (which walks the wire payload's message-array
    shape) and stashed on ``ParsedResponseRecord`` so record-metric classes
    (``NumImagesMetric`` et al.) never re-parse ``payload_bytes`` per metric per
    record. Zero-valued when the payload carries no recognised media parts.
    """

    images: int = 0
    """Count of image content parts in the wire payload."""

    audios: int = 0
    """Count of audio content parts in the wire payload."""

    videos: int = 0
    """Count of video content parts in the wire payload."""


@dataclass
class ParsedResponseRecord:
    """Record of a request and its associated responses, already parsed and ready for metrics.

    Uses @dataclass without slots to allow @cached_property (requires __dict__).
    """

    request: RequestRecord
    """The original request record."""

    responses: list[ParsedResponse]
    """The parsed responses."""

    token_counts: TokenCounts | None = None
    """The token counts for the response. None if the token counts could not be calculated."""

    media_counts: MediaCounts = field(default_factory=MediaCounts)
    """Multimodal content-part counts derived once from the wire payload (images/audios/videos)."""

    spec_decode_acceptance: SpecDecodeAcceptanceRecord | None = None
    """Engine-neutral per-request speculative-decoding acceptance record, filled
    by a ``SpecDecodeAdapterProtocol`` when the response carried spec-decode
    stats. ``None`` when: spec decode is off or the request had no verify steps
    (no payload, or a zero-step payload the parser drops); the request produced
    multiple sequences (``n > 1``, which is suppressed); no registered adapter
    recognized the payload; or the payload was malformed and the adapter
    rejected it."""

    spec_decode_zero_step: bool = False
    """True when the engine reported spec-decode stats for this request but it
    ran no verify step (``num_spec_steps == 0``). Its record is dropped, so
    ``spec_decode_acceptance`` is None; this flag lets the run count such
    requests separately from runs where spec decode is off."""

    @cached_property
    def final_usage(self) -> Usage | None:
        """API-reported usage from the last streaming response chunk that had any.

        Thin wrapper around `find_last_non_empty_usage`. Cached, so the walk
        happens at most once per record regardless of how many metrics consult
        it. See the helper's docstring for the rationale behind "last
        non-empty chunk wins" instead of a per-key merge.
        """
        return find_last_non_empty_usage(self.responses)

    @cached_property
    def start_perf_ns(self) -> int:
        """Get the start time of the request in nanoseconds (perf_counter_ns)."""
        return self.request.start_perf_ns

    @cached_property
    def timestamp_ns(self) -> int:
        """Get the wall clock timestamp of the request in nanoseconds. DO NOT USE FOR LATENCY CALCULATIONS. (time.time_ns)."""
        return self.request.timestamp_ns

    # TODO: How do we differentiate the end of the request vs the time of the last response?
    #       Which one should we use for the latency metrics?
    @cached_property
    def end_perf_ns(self) -> int:
        """Get the end time of the request in nanoseconds (perf_counter_ns).
        If request.end_perf_ns is not set, use the time of the last response.
        If there are no responses, use sys.maxsize.
        """
        return (
            self.request.end_perf_ns
            if self.request.end_perf_ns
            else self.responses[-1].perf_ns
            if self.responses
            else sys.maxsize
        )

    @cached_property
    def content_responses(self) -> list[ParsedResponse]:
        """Get only responses with actual content (data is not None or empty).

        This excludes usage-only or [DONE] responses that may appear at the end of streaming responses.
        Useful for timing metrics that should measure content delivery.
        """
        return [response for response in self.responses if response.data]

    @property
    def has_error(self) -> bool:
        """Check if the response record has an error."""
        return self.request.has_error

    @cached_property
    def valid(self) -> bool:
        """Check if the response record is valid.

        Checks:
        - Request has no errors
        - Has at least one content response
        - Start time is before the end time
        - Response timestamps are within valid ranges

        Returns:
            bool: True if the record is valid, False otherwise.
        """
        return (
            not self.has_error
            and len(self.content_responses) > 0
            and 0 <= self.start_perf_ns < self.end_perf_ns < sys.maxsize
            and all(0 < response.perf_ns < sys.maxsize for response in self.responses)
        )

    def create_error_from_invalid(self) -> None:
        """Convert any invalid request records to error records for combined processing."""
        if not self.valid and not self.has_error:
            _logger.debug(
                lambda: f"Converting invalid request record to error record: {self}"
            )
            err = InvalidInferenceResultError("Invalid inference result")
            if len(self.responses) == 0 or len(self.content_responses) == 0:
                err.add_note(
                    "No responses with actual content were received from the server (only usage/metadata, null/empty data, or [DONE] markers)"
                )
            if self.start_perf_ns <= 0 or self.start_perf_ns >= sys.maxsize:
                err.add_note(
                    f"Start perf ns timestamp is invalid: {self.start_perf_ns}"
                )
            for i, response in enumerate(self.responses):
                if response.perf_ns <= 0 or response.perf_ns >= sys.maxsize:
                    err.add_note(
                        f"Response {i} perf ns timestamp is invalid: {response.perf_ns}"
                    )
            self.request.error = ErrorDetails.from_exception(err)


class MetricRecordInfo(AIPerfBaseModel):
    """The full info of a metric record including the metadata, metrics, and error for export."""

    metadata: MetricRecordMetadata = Field(
        ...,
        description="The metadata of the record. Should match the metadata in the MetricRecordsMessage.",
    )
    metrics: dict[str, MetricValue] = Field(
        ...,
        description="A dictionary containing all metric values along with their units.",
    )
    trace_data: SerializeAsAny[TraceDataExport] | None = Field(
        default=None,
        description="Comprehensive trace data captured via a trace config with wall-clock timestamps. "
        "Includes detailed timing for connection establishment, DNS resolution, request/response events, etc. "
        "The type of the trace data is determined by the transport and library used.",
    )
    spec_decode_acceptance: SpecDecodeAcceptanceRecord | None = Field(
        default=None,
        description="Engine-neutral per-request speculative-decoding acceptance "
        "record (histogram, per-step counts, and aggregate tallies) carried into "
        "the records trace at --export-level records. None when spec decode is off "
        "or the request had no verify steps.",
    )
    error: ErrorDetails | None = Field(
        default=None,
        description="The error details if the request failed.",
    )


class RawRecordInfo(AIPerfBaseModel):
    """The full info of a raw record including the request record for export."""

    metadata: MetricRecordMetadata = Field(
        ...,
        description="The metadata of the record. Should match the metadata in the MetricRecordsMessage.",
    )
    start_perf_ns: int = Field(
        default_factory=time.perf_counter_ns,
        description="The start reference time of the request in nanoseconds used for latency calculations (perf_counter_ns).",
    )
    payload: dict[str, Any] | None = Field(
        default=None,
        description="The raw request payload sent to the server. Exactly one "
        "of ``payload`` or ``payload_bytes`` is populated per record — "
        "``payload_bytes`` is preferred by the JSONL writer (bytes are "
        "spliced as a JSON fragment without a loads+dumps round-trip).",
    )
    payload_bytes: bytes | None = Field(
        default=None,
        exclude=True,
        description="Canonical pre-encoded JSON bytes of the request body, "
        "inherited from ``RequestInfo.payload_bytes``. Spliced directly into "
        "the JSONL line via ``orjson.Fragment`` to avoid the pointless "
        "decode-then-encode round-trip ``payload: dict`` would require.",
    )
    request_headers: dict[str, str] | None = Field(
        default=None,
        description="The headers of the request.",
    )
    status: int | None = Field(
        default=None,
        description="The status code of the response.",
    )
    response_headers: dict[str, str] | None = Field(
        default=None,
        description="The headers of the response.",
    )
    responses: SerializeAsAny[list[SSEMessage | TextResponse | BinaryResponse]] = Field(
        ...,
        description="The raw responses received from the request.",
    )
    error: ErrorDetails | None = Field(
        default=None,
        description="The error details if the request failed.",
    )
    cache_bust_marker: str | None = Field(
        default=None,
        description="Cache-bust marker text injected into the wire payload for "
        "this request, copied from the originating ``Credit``. None when the "
        "cache-bust feature is disabled. Surfaced here so raw-JSONL consumers "
        "can correlate inserted bytes with the originating session without "
        "re-parsing ``payload``.",
    )
    cache_bust_target: CacheBustTarget | None = Field(
        default=None,
        description="Where the marker was injected (``system_prefix``, "
        "``system_suffix``, ``first_turn_prefix``, or ``first_turn_suffix``). "
        "None when cache-bust is disabled.",
    )
