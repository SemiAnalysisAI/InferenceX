# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from aiperf.accuracy.models import AccuracySummary, ProcessAccuracyResult
from aiperf.common.accumulator_protocols import (
    AccumulatorProtocol,
    ExportContext,
    StreamExporterProtocol,
    SummaryContext,
)
from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import (
    CommAddress,
    CommandType,
    CreditPhase,
    MessageType,
    ProfileCancelReason,
    make_result_producer_capability,
)
from aiperf.common.environment import Environment
from aiperf.common.hooks import background_task, on_command, on_message, on_pull_message
from aiperf.common.messages import (
    AllRecordsReceivedMessage,
    DatasetConfiguredNotification,
    NetworkLatencyRecordMessage,
    ProcessAccuracyResultMessage,
    ProcessAllResultsMessage,
    ProcessRecordsCommand,
    ProcessRecordsResultMessage,
    ProcessServerMetricsResultMessage,
    ProcessTelemetryResultMessage,
    ProfileCancelCommand,
    ProfileCompleteCommand,
    RealtimeMetricsCommand,
    RealtimeMetricsMessage,
    RecordsMessage,
    RecordsProcessingStatsMessage,
    ServerMetricsRecordMessage,
    StartRealtimeTelemetryCommand,
    TelemetryRecordsMessage,
)
from aiperf.common.messages.inference_messages import MetricRecordsData
from aiperf.common.mixins import PullClientMixin
from aiperf.common.models import (
    BasePhaseStats,
    BranchStats,
    CreditPhaseStats,
    ErrorDetails,
    ErrorDetailsCount,
    MetricResult,
    PhaseProfileResults,
    PhaseRecordsStats,
    ProcessRecordsResult,
    ProcessServerMetricsResult,
    ProcessTelemetryResult,
    ProfileMetricDurationCoverage,
    ProfileResults,
    TimesliceResult,
    WorkerProcessingStats,
)
from aiperf.common.scenario.registry import get_scenario
from aiperf.common.types import MetricTagT
from aiperf.common.utils import yield_to_event_loop
from aiperf.config.comm import ZMQDualBindConfig
from aiperf.credit.messages import (
    CreditPhaseCompleteMessage,
    CreditPhaseProgressMessage,
    CreditPhaseSendingCompleteMessage,
    CreditPhaseStartMessage,
    CreditsCompleteMessage,
)
from aiperf.gpu_telemetry.protocols import GPUTelemetryAccumulatorProtocol
from aiperf.metrics.accumulator_models import AccumulatorMetricsSummary
from aiperf.metrics.cache_reporting_hint import (
    CACHE_REPORTING_HINT,
    usage_without_cache_in_record,
)
from aiperf.network_latency.accumulator import NetworkLatencyAccumulator
from aiperf.plugin import plugins
from aiperf.plugin.enums import (
    AccumulatorType,
    PluginType,
    StreamExporterType,
    UIType,
)
from aiperf.records import records_manager_processing
from aiperf.records.dataset_gate import DatasetConfigCatchUp, await_dataset_configured
from aiperf.records.error_tracker import ErrorTracker
from aiperf.records.records_manager_processing import (
    LoadedAnalyzer,
    generate_realtime_metrics,
    load_accumulators,
    load_analyzers,
    load_stream_exporters,
)
from aiperf.records.records_tracker import RecordsTracker
from aiperf.server_metrics.protocols import ServerMetricsAccumulatorProtocol

if TYPE_CHECKING:
    from aiperf.config.config import BenchmarkConfig
    from aiperf.config.resolution.plan import BenchmarkRun


_LATENCY_LINE_LABELS: tuple[tuple[str, str], ...] = (
    ("ttft", "time_to_first_token"),
    # Use the scalar per-record metric (avg gap across the response), not the
    # list-valued ``inter_chunk_latency``. List metrics don't aggregate into
    # displayable percentiles in the realtime path, so the row used to show
    # only dashes mid-run even when the per-record JSONL had real values.
    ("itl", "inter_token_latency"),
    ("e2e", "request_latency"),
)
_INTERACTIVITY_LABEL: tuple[str, str] = (
    "intvty",
    "output_token_throughput_per_user",
)
_SEQ_LENGTH_LABELS: tuple[tuple[str, str], ...] = (
    ("isl", "input_sequence_length"),
    ("osl", "output_sequence_length"),
)
# Each block line is its own log record (carries its own log prefix), so the
# continuation rows sit at a small fixed indent under the header line rather
# than aligning under the old inline "[realtime MM:SS profiling] " text.
_REALTIME_ROW_INDENT = 2
# Percentile names per row group. Latency/interactivity rows report p95 in the
# third column; sequence-length rows report p90 there (the agentic long-tail is
# more interesting at p90 for token counts). Each row keeps its own ``pNN=``
# labels, so the column can hold p95 on one row and p90 on the next.
_LATENCY_PERCENTILES: tuple[str, ...] = ("p50", "p75", "p95", "p99")
_TOKEN_PERCENTILES: tuple[str, ...] = ("p50", "p75", "p90", "p99")
_SERVER_SNAPSHOT_METRIC_DISPLAY: dict[str, tuple[str, str]] = {
    "prefix_cache_hit_rate": ("Prefix Cache Hit Rate", "%"),
    "external_prefix_cache_hit_rate": ("External Prefix Cache Hit Rate", "%"),
    "kv_cache_usage_pct": ("KV Cache Usage", "%"),
    "cpu_kv_cache_usage_pct": ("CPU KV Cache Usage", "%"),
    "num_running": ("Server Running Requests", "req"),
    "num_waiting": ("Server Waiting Requests", "req"),
    "num_preemptions": ("Server Preemptions", "req"),
    "input_token_throughput_srv": ("Server Input Throughput", "tokens/s"),
    "output_token_throughput_srv": ("Server Output Throughput", "tokens/s"),
    "unique_input_tokens_srv": ("Unique Input Tokens", "tokens"),
}


def _format_elapsed(seconds: float) -> str:
    total = int(seconds)
    if total < 3600:
        return f"{total // 60:02d}:{total % 60:02d}"
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def _format_ms(value: float | None) -> str:
    if value is None:
        return "-"
    if value < 1.0:
        return "<1ms"
    return f"{int(round(value)):,}ms"


def _format_int(value: float | None) -> str:
    """Compact int formatter for token-rate percentiles. Returns ``-`` for None."""
    if value is None:
        return "-"
    return f"{int(round(value)):,}"


def _server_snapshot_to_metric_results(
    server_snapshot: dict[str, float],
) -> list[MetricResult]:
    """Convert live server snapshot scalars into realtime MetricResult rows."""
    metrics: list[MetricResult] = []
    for tag, value in server_snapshot.items():
        header, unit = _SERVER_SNAPSHOT_METRIC_DISPLAY.get(
            tag,
            (tag.replace("_", " ").title(), ""),
        )
        metrics.append(
            MetricResult(
                tag=tag,
                header=header,
                unit=unit,
                avg=value,
                current=value,
            )
        )
    return metrics


def _render_realtime_block(
    metric_results: list[MetricResult],
    phase_stats: PhaseRecordsStats,
    prev_snapshot: tuple[int, float] | None,
    server_snapshot: dict[str, float] | None = None,
) -> str:
    """Render a compact realtime stats block for the aiperf logger.

    Format (``[realtime MM:SS profiling]`` header, a summary counter row, then
    one labeled percentile row per metric)::

        [realtime 00:49 profiling]
          rps=14.2 (avg 13.1)  tput_in=1,097,271/s  tput_out=10,441/s  done=641 ok=641 err=0
          ttft    p50=   30ms  p75=   48ms  p95=   106ms  p99=   155ms
          itl     p50=    5ms  p75=    5ms  p95=     5ms  p99=     5ms
          e2e     p50=2,241ms  p75=4,853ms  p95=13,526ms  p99=22,003ms
          intvty  p50=    200  p75=    201  p95=     211  p99=     254  (1/tpot tok/s)
          isl     p50= 67,234  p75= 97,141  p90= 179,564  p99= 384,325  (tokens)
          osl     p50=    443  p75=    967  p90=   2,034  p99=   4,396  (tokens)
          tot     in=53,555,186  out=509,605
          trace   theoretical_prefix_cache_hit=97.5%

    The header sits on its own line and the summary counters drop to the
    first indented row so the line no longer wraps in narrow terminals; each
    line is emitted as a separate log record (see ``_report_realtime_metrics``).
    Every row keeps its own ``pNN=`` labels (so it stays readable even when log
    lines from other services interleave), while the values are right-aligned in
    per-column widths so the digits and ``ms`` suffixes line up into a grid.

    Latency MetricResult percentile values are already in display units
    (milliseconds for time-based metrics, see ``to_display_unit`` and the
    accumulator's ``summarize`` path), so ``_format_ms`` consumes them as-is.
    Returns an empty string when no requests have completed yet so callers
    can suppress the block entirely on the first tick.

    Records-side stats only — ``in_flight_requests`` is a credit-side concept
    that this function doesn't have access to and is therefore omitted from
    the output.
    """
    if phase_stats.total_records == 0:
        return ""

    by_tag: dict[str, MetricResult] = {m.tag: m for m in metric_results}
    elapsed = phase_stats.records_elapsed_time

    rps_avg_mr = by_tag.get("request_throughput")
    rps_avg = getattr(rps_avg_mr, "avg", None)
    rps_avg_str = f"{rps_avg:.1f}" if rps_avg is not None else "-"

    if prev_snapshot is not None:
        prev_completed, prev_elapsed = prev_snapshot
        dt = elapsed - prev_elapsed
        rps_delta = (phase_stats.total_records - prev_completed) / dt if dt > 0 else 0.0
        rps_delta_str = f"{rps_delta:.1f}"
    else:
        rps_delta_str = rps_avg_str

    tput_out_mr = by_tag.get("output_token_throughput")
    tput_out_avg = getattr(tput_out_mr, "avg", None)
    tput_out_str = f"{int(round(tput_out_avg)):,}" if tput_out_avg is not None else "-"

    tput_in_mr = by_tag.get("input_token_throughput")
    tput_in_avg = getattr(tput_in_mr, "avg", None)
    tput_in_str = f"{int(round(tput_in_avg)):,}" if tput_in_avg is not None else "-"

    header = f"[realtime {_format_elapsed(elapsed)} profiling]"

    indent = " " * _REALTIME_ROW_INDENT

    # Build the percentile rows as (label, percentile_names, value_strings,
    # suffix) tuples first, so column widths can be derived from the actual
    # rendered values before any line is formatted. Latency/interactivity rows
    # use ms-formatted values; sequence-length rows use comma-grouped ints.
    #
    # Interactivity = 1 / inter-token-latency per request, percentiled across
    # requests. Characterizes the user-perceived decode speed; tail (low
    # percentile) is the slowest-decoding user, head (high percentile) is the
    # snappiest. Aggregate tput_in/tput_out on line 1 are bandwidth.
    StatRow = tuple[str, tuple[str, ...], list[str], str]
    stat_rows: list[StatRow] = []
    for label, tag in _LATENCY_LINE_LABELS:
        mr = by_tag.get(tag)
        values = [_format_ms(getattr(mr, p, None)) for p in _LATENCY_PERCENTILES]
        stat_rows.append((label, _LATENCY_PERCENTILES, values, ""))
    intvty_label, intvty_tag = _INTERACTIVITY_LABEL
    mr = by_tag.get(intvty_tag)
    stat_rows.append(
        (
            intvty_label,
            _LATENCY_PERCENTILES,
            [_format_int(getattr(mr, p, None)) for p in _LATENCY_PERCENTILES],
            "(1/tpot tok/s)",
        )
    )

    # Sequence-length distribution rows — useful for spotting long-tail
    # agentic prompts mid-run. Reads the same MetricResults the aggregator
    # already publishes; no extra plumbing. A row is omitted entirely when its
    # metric has no data, rather than rendering a row of dashes.
    for label, tag in _SEQ_LENGTH_LABELS:
        mr = by_tag.get(tag)
        values = [_format_int(getattr(mr, p, None)) for p in _TOKEN_PERCENTILES]
        if all(v == "-" for v in values):
            continue
        stat_rows.append((label, _TOKEN_PERCENTILES, values, "(tokens)"))

    label_w = max(len(label) for label, *_ in stat_rows)
    col_w = [max(len(values[i]) for _, _, values, _ in stat_rows) for i in range(4)]

    rows: list[str] = [
        f"{indent}rps={rps_delta_str} (avg {rps_avg_str})  "
        f"tput_in={tput_in_str}/s  "
        f"tput_out={tput_out_str}/s  "
        f"done={phase_stats.total_records:,} "
        f"ok={phase_stats.success_records:,} "
        f"err={phase_stats.error_records:,}"
    ]
    for label, percentiles, values, suffix in stat_rows:
        cells = "  ".join(
            f"{name}={value.rjust(col_w[i])}"
            for i, (name, value) in enumerate(zip(percentiles, values, strict=True))
        )
        line = f"{indent}{label:<{label_w}}  {cells}"
        rows.append(f"{line}  {suffix}" if suffix else line)

    # Cumulative token totals — running counters, useful for spotting
    # whether the ratio of output:input tokens is matching the workload's
    # expected agentic pattern.
    total_isl_mr = by_tag.get("total_isl")
    total_osl_mr = by_tag.get("total_osl")
    total_isl = getattr(total_isl_mr, "avg", None)
    total_osl = getattr(total_osl_mr, "avg", None)
    if total_isl is not None or total_osl is not None:
        in_str = f"{int(round(total_isl)):,}" if total_isl is not None else "-"
        out_str = f"{int(round(total_osl)):,}" if total_osl is not None else "-"
        rows.append(f"{indent}{'tot':<{label_w}}  in={in_str}  out={out_str}")

    theoretical_prefix_mr = by_tag.get("theoretical_prefix_cache_hit")
    theoretical_prefix_hit = getattr(theoretical_prefix_mr, "current", None)
    if theoretical_prefix_hit is None:
        theoretical_prefix_hit = getattr(theoretical_prefix_mr, "avg", None)
    if theoretical_prefix_hit is not None:
        rows.append(
            f"{indent}{'trace':<{label_w}} theoretical_prefix_cache_hit={theoretical_prefix_hit:.1f}%"
        )

    # Server-side row — cumulative cache hit rate, KV usage, and scheduler
    # queue depth from the live ServerMetricsAccumulator snapshot. Sourced
    # from the /metrics scrape, so populates only when server-metrics
    # collection is enabled and the inference server actually serves
    # Prometheus. Each part is rendered only when its backing metric is
    # present, so e.g. cpu_kv / ext_cache_hit show up only on offload=cpu
    # runs.
    if server_snapshot:
        srv_parts: list[str] = []
        if "prefix_cache_hit_rate" in server_snapshot:
            srv_parts.append(
                f"prefix_cache_hit={server_snapshot['prefix_cache_hit_rate']:.1f}%"
            )
        if "unique_input_tokens_srv" in server_snapshot:
            srv_parts.append(
                f"unique_in_srv={int(round(server_snapshot['unique_input_tokens_srv'])):,}"
            )
        if "external_prefix_cache_hit_rate" in server_snapshot:
            srv_parts.append(
                f"ext_cache_hit={server_snapshot['external_prefix_cache_hit_rate']:.1f}%"
            )
        if "kv_cache_usage_pct" in server_snapshot:
            srv_parts.append(f"kv_usage={server_snapshot['kv_cache_usage_pct']:.1f}%")
        if "cpu_kv_cache_usage_pct" in server_snapshot:
            srv_parts.append(
                f"cpu_kv_usage={server_snapshot['cpu_kv_cache_usage_pct']:.1f}%"
            )
        if "num_running" in server_snapshot or "num_waiting" in server_snapshot:
            running = int(server_snapshot.get("num_running", 0))
            waiting = int(server_snapshot.get("num_waiting", 0))
            srv_parts.append(f"queue={running}r/{waiting}w")
        if "input_token_throughput_srv" in server_snapshot:
            srv_parts.append(
                f"tput_in_srv={int(round(server_snapshot['input_token_throughput_srv'])):,}/s"
            )
        if "output_token_throughput_srv" in server_snapshot:
            srv_parts.append(
                f"tput_out_srv={int(round(server_snapshot['output_token_throughput_srv'])):,}/s"
            )
        if srv_parts:
            rows.append(f"{indent}{'srv':<{label_w}} {' '.join(srv_parts)}")

    return "\n".join([header, *rows])


@dataclass
class ErrorTrackingState:
    """State container for tracking errors with counts and thread-safe access.

    Provides common error tracking functionality for all metrics subsystems
    (telemetry, server metrics, regular metrics).
    """

    error_counts: dict[ErrorDetails, int] = field(
        default_factory=lambda: defaultdict(int)
    )


_logger = AIPerfLogger(__name__)


def _pooled_spec_decode_histogram(
    summary_ctx: SummaryContext,
) -> dict[int, int] | None:
    """Pull the pooled acceptance histogram off the metric_records summary.

    The dict rides on ``AccumulatorMetricsSummary`` (not in ``records``) because
    it is a dict aggregate outside the scalar/list metric machinery. Only the
    ``metric_results`` accumulator pools one, so exactly one populated histogram
    is expected; if more than one accumulator populates one the single-source
    assumption has broken, so warn and use the first rather than silently
    picking a dict-ordering winner. None when spec decode was off for the
    exported phase. A module-level function (not a method) so mocked
    RecordsManager instances in unit tests cannot shadow it with an auto-mock.
    """
    populated = [
        summary.pooled_spec_decode_acceptance_histogram
        for summary in summary_ctx.accumulator_outputs.values()
        if isinstance(summary, AccumulatorMetricsSummary)
        and summary.pooled_spec_decode_acceptance_histogram
    ]
    if len(populated) > 1:
        _logger.warning(
            f"Expected one accumulator to pool a spec-decode acceptance "
            f"histogram, found {len(populated)}; using the first. Reconciliation "
            "with total_spec_decode_steps may be unreliable."
        )
    return populated[0] if populated else None


class RecordsManager(PullClientMixin, BaseComponentService):
    """Collects and processes benchmark results from workers.

    The RecordsManager receives metric records from workers and accumulates them
    for final processing. The timing manager is the ground truth for what requests
    completed within the benchmark window - when it signals phase completion with
    a final_completed_count, the RecordsManager waits until it has processed that
    many records before finalizing results.
    """

    def _has_multiple_phase_instances(self, phase: CreditPhase) -> bool:
        """Return whether this run has more than one concrete phase of a kind."""
        try:
            phases = self.run.cfg.phases
        except AttributeError:
            return False
        phase_kind = "warmup" if phase == CreditPhase.WARMUP else "profiling"
        return sum(1 for cfg_phase in phases if cfg_phase.kind == phase_kind) > 1

    def _has_multiple_profiling_phases(self) -> bool:
        """Return whether this run has more than one profiling-kind phase."""
        return self._has_multiple_phase_instances(CreditPhase.PROFILING)

    def _check_all_records_received(self, phase: CreditPhase) -> bool:
        """Check record completion for a phase kind."""
        return self._records_tracker.check_and_set_all_records_received_for_phase(phase)

    def __init__(
        self,
        run: BenchmarkRun,
        service_id: str | None = None,
        **kwargs,
    ) -> None:
        # For dual-bind mode (Kubernetes), also bind to TCP for remote record processors.
        # Controller binds to IPC + TCP; workers connect via TCP.
        additional_bind_address: str | None = None
        comm_config = run.cfg.comm_config
        if (
            isinstance(comm_config, ZMQDualBindConfig)
            and not comm_config.controller_host
        ):
            additional_bind_address = comm_config.records_push_pull_tcp_bind_address

        super().__init__(
            run=run,
            service_id=service_id,
            pull_client_address=CommAddress.RECORDS,
            pull_client_bind=True,
            pull_client_max_concurrency=Environment.ZMQ.PULL_MAX_CONCURRENCY,
            pull_client_additional_bind_address=additional_bind_address,
            **kwargs,
        )

        # Advertise the result domains this manager produces so the controller's
        # ResultJoinCoordinator waits for them on shutdown. Accuracy is a separate
        # domain, joined only when accuracy mode is enabled for this run.
        self.extra_capabilities = (make_result_producer_capability("profile"),)
        if run.cfg.accuracy is not None and run.cfg.accuracy.enabled:
            self.extra_capabilities += (make_result_producer_capability("accuracy"),)

        self._records_tracker = RecordsTracker()
        self._error_tracker = ErrorTracker()

        # DatasetConfiguredNotification (SUB) and metric records (PULL) arrive on
        # independent channels with no ordering guarantee. Gate record processing on
        # this event so results processors are configured (e.g. accuracy task names)
        # before any record is accumulated.
        self._dataset_configured_event: asyncio.Event = asyncio.Event()
        self._dataset_config_catch_up = DatasetConfigCatchUp(
            request_client=self.comms.create_request_client(
                address=CommAddress.DATASET_MANAGER_PROXY_FRONTEND,
                bind=False,
            ),
            on_configured=self._on_dataset_configured,
            service_id=self.service_id,
        )

        self._previous_realtime_records: int | None = None
        # Server-metric snapshot from the prior realtime tick. The realtime block
        # must re-render when live server metrics (cache hit rate, KV usage,
        # queue depth) move even while the record count is momentarily static --
        # gating on record count alone froze the server row during lulls.
        self._previous_realtime_server_snapshot: dict[str, float] | None = None
        # (completed_records, elapsed_seconds) from the prior realtime tick, used
        # to render the instantaneous (delta) RPS in the realtime stats block.
        self._prev_realtime_snapshot: tuple[int, float] | None = None
        self._prev_realtime_phase_index: int | None = None

        # Latest BranchStats snapshot received via CreditPhaseCompleteMessage
        # for the PROFILING phase. None for non-DAG runs (TimingManager
        # publishes None when no BranchOrchestrator is wired). Spliced
        # into ProfileResults when the records pipeline finalizes.
        self._latest_branch_stats: BranchStats | None = None

        # Per-phase BranchStats snapshots. Populated on every
        # CreditPhaseCompleteMessage that carries a non-None
        # ``branch_stats``; ``_snapshot_branch_stats`` reads back the
        # value for a specific phase. Used by analyzer paths that need
        # warmup vs profiling separation.
        self._phase_branch_stats: dict[CreditPhase, BranchStats] = {}
        self._complete_credit_phases: set[CreditPhase] = set()
        self._credits_complete_received = False
        self._all_records_received_phases: set[CreditPhase] = set()

        self._telemetry_state = ErrorTrackingState()
        self._server_metrics_state = ErrorTrackingState()
        self._skipped_context_overflow_counts_by_phase: dict[CreditPhase, int] = {
            CreditPhase.WARMUP: 0,
            CreditPhase.PROFILING: 0,
        }

        self._gpu_telemetry_accumulator: GPUTelemetryAccumulatorProtocol | None = None
        self._server_metrics_accumulator: ServerMetricsAccumulatorProtocol | None = None

        # In-process accumulator for RTT probe samples. Computes the run-level
        # mean RTT delivered to MetricsAccumulator before summarize(). None unless
        # network latency probing is active.
        self._network_latency_accumulator: NetworkLatencyAccumulator | None = (
            NetworkLatencyAccumulator(benchmark_id=self.run.benchmark_id)
            if self.run.cfg.network_latency.should_probe
            else None
        )
        self._network_latency_state = ErrorTrackingState()

        self._accumulators: dict[AccumulatorType, AccumulatorProtocol] = (
            load_accumulators(self)
        )
        self._stream_exporters: dict[StreamExporterType, StreamExporterProtocol] = (
            load_stream_exporters(self)
        )
        # Summarize-time cross-accumulator analyzers (e.g. energy efficiency),
        # each carrying its live-instance and summary dependencies.
        self._analyzers: list[LoadedAnalyzer] = load_analyzers(self)
        self._routing_table = self._build_routing_table()
        self._warned_unrouted_record_types: set[str] = set()
        self._warned_missing_cache_reporting: bool = False
        self._log_routing_table()

        # Single-flight guard for _process_results: the background finalize task,
        # the PROCESS_RECORDS command, and PROFILE_CANCEL can all reach it and
        # would otherwise double-publish and double-finalize stream exporters.
        self._process_results_lock = asyncio.Lock()
        self._processed_results: dict[CreditPhase, ProcessRecordsResult] = {}

        self._metric_record_accumulators = [
            accumulator
            for accumulator in self._accumulators.values()
            if accumulator in self._routing_table.get("metric_records", [])
        ]
        self._gpu_telemetry_accumulator = self._accumulators.get(
            AccumulatorType.GPU_TELEMETRY
        )
        self._server_metrics_accumulator = self._accumulators.get(
            AccumulatorType.SERVER_METRICS
        )
        self._accuracy_accumulator = self._accumulators.get(AccumulatorType.ACCURACY)

        # Failed-request abort threshold (AGENTIC_REPLAY, A5 #7): abort the run
        # once the profiling failure ratio exceeds the configured threshold.
        profiling_phases = self.run.cfg.get_profiling_phases()
        profiling_phase = profiling_phases[0] if profiling_phases else None
        self._failed_request_threshold: float | None = (
            profiling_phase.failed_request_threshold if profiling_phase else None
        )
        conc_val = profiling_phase.concurrency if profiling_phase else None
        self._failed_request_grace_floor = max(
            int(conc_val) if isinstance(conc_val, (int, float)) else 1, 10
        )
        self._failed_request_abort_triggered = False
        self._profiling_started = False
        # Keyed by phase_index: under `seamless` a later profiling phase starts
        # while the previous one drains, and both keep publishing.
        self._progress_stall_last_total: dict[int | None, int] = {}
        self._progress_stall_since: dict[int | None, float] = {}
        # Latest profiling CreditPhaseStats per phase. The credit side is the
        # only source of sent/completed/cancelled counts; the records tracker's
        # PhaseRecordsStats carries record counts and has no notion of what is
        # still in flight.
        self._latest_profiling_credit_stats: dict[int | None, CreditPhaseStats] = {}
        # Latch so a stalled run requests cancellation once, not every tick.
        self._progress_stall_triggered: bool = False

    def _build_routing_table(self) -> dict[str, list[Any]]:
        """Build record_type string -> handler mapping from plugin metadata."""
        table: dict[str, list[Any]] = {}
        for entry in plugins.iter_entries(PluginType.ACCUMULATOR):
            handler = self._accumulators.get(AccumulatorType(entry.name))
            if handler is None:
                continue
            record_types = (
                entry.metadata.get("record_types", []) if entry.metadata else []
            )
            for record_type in record_types:
                table.setdefault(record_type, []).append(handler)

        for entry in plugins.iter_entries(PluginType.STREAM_EXPORTER):
            handler = self._stream_exporters.get(StreamExporterType(entry.name))
            if handler is None:
                continue
            record_types = (
                entry.metadata.get("record_types", []) if entry.metadata else []
            )
            for record_type in record_types:
                table.setdefault(record_type, []).append(handler)
        return table

    async def _dispatch_record(
        self, record: Any, *, warn_if_unrouted: bool = True
    ) -> list[BaseException]:
        """Dispatch one typed record to all handlers registered for its record_type.

        ``warn_if_unrouted`` gates the "no handlers" warning: leave it True for
        data-plane records (which would be silently lost), but set it False for
        control-plane records that are already consumed elsewhere and only
        OPTIONALLY streamed (e.g. ``credit_phase_stats``, whose stats are applied
        via ``update_phase_info`` before dispatch and only routed to the
        default-absent OTel streamer) -- otherwise every non-OTel run warns falsely.
        """
        record_type = getattr(record, "record_type", None)
        if record_type is None:
            error = TypeError(f"Record {type(record).__name__} has no record_type")
            self.error(str(error))
            return [error]

        handlers = self._routing_table.get(record_type, [])
        if not handlers:
            # Warn once per unrouted type: records silently vanish here while the
            # request still counts as a success, so this must not stay debug-only.
            if (
                warn_if_unrouted
                and record_type not in self._warned_unrouted_record_types
            ):
                self._warned_unrouted_record_types.add(record_type)
                self.warning(
                    f"No handlers registered for record type {record_type!r}; "
                    "records of this type are being dropped. Check that a producer's "
                    "record_type matches an accumulator/stream_exporter record_types "
                    "entry in plugins.yaml."
                )
            return []

        results = await asyncio.gather(
            *[handler.process_record(record) for handler in handlers],
            return_exceptions=True,
        )
        errors: list[BaseException] = []
        for handler, result in zip(handlers, results, strict=True):
            # A handler-level CancelledError (captured by return_exceptions) means
            # one handler's coroutine was cancelled, NOT this task -- genuine task
            # cancellation makes the gather itself raise and never reaches here. We
            # must count it like any other handler failure rather than re-raising,
            # or the caller skips the tracker update + (timeout-less) completion
            # barrier and the phase never converges.
            if isinstance(result, BaseException):
                self.error(
                    f"Handler {handler.__class__.__name__} failed for "
                    f"{record_type}: {result!r}"
                )
                # Best-effort handlers (streaming telemetry like OTel/MLflow) must
                # never pollute the benchmark's phase error count -- a downed
                # collector is not an inference failure. Log and drop, honoring the
                # handler's ``is_best_effort`` marker.
                if getattr(handler, "is_best_effort", False):
                    continue
                errors.append(result)
        return errors

    def _log_routing_table(self) -> None:
        """Log the metadata-derived record routing table."""
        self.debug(
            lambda: (
                f"Routing table: {len(self._accumulators)} accumulators, "
                f"{len(self._stream_exporters)} stream exporters, "
                f"{len(self._routing_table)} record types"
            )
        )
        for record_type, handlers in self._routing_table.items():
            handler_names = [handler.__class__.__name__ for handler in handlers]
            self.debug(lambda rt=record_type, hn=handler_names: f"  {rt} -> {hn}")

    async def _maybe_trigger_failed_request_abort(
        self, phase: CreditPhase, *, phase_index: int | None = None
    ) -> None:
        """Abort the run when the PROFILING failure rate exceeds the threshold.

        No-op when ``--failed-request-threshold`` is unset, when this method
        already fired once for this run, or when the total record count has
        not yet crossed the grace floor (``max(concurrency, 10)``). Otherwise
        checks the concrete profiling phase identified by ``phase_index`` and
        broadcasts ProfileCancelCommand on the message bus so the other
        services stop their work, then locally finalizes the records manager.
        A PUB socket does not receive its own message, so relying on the
        broadcast for local finalization would leave the run hanging.
        """
        if self._failed_request_threshold is None:
            return
        if self._failed_request_abort_triggered:
            return
        if phase != CreditPhase.PROFILING:
            return

        total = self._records_tracker.total_records_for_phase(phase, phase_index)
        if total < self._failed_request_grace_floor:
            return

        error_records = self._records_tracker.error_records_for_phase(
            phase, phase_index
        )
        rate = error_records / total if total > 0 else 0.0
        if rate <= self._failed_request_threshold:
            return

        self._failed_request_abort_triggered = True
        self.warning(
            f"--failed-request-threshold exceeded: "
            f"{error_records}/{total} = {rate:.3f} > "
            f"{self._failed_request_threshold:.3f} "
            f"(grace floor {self._failed_request_grace_floor}). "
            "Broadcasting ProfileCancelCommand to terminate the run."
        )
        if not await self._request_profile_cancel(
            ProfileCancelReason.FAILED_REQUEST_THRESHOLD,
            f"{error_records}/{total} profiling requests failed "
            f"({rate:.1%}), exceeding the "
            f"--failed-request-threshold limit of "
            f"{self._failed_request_threshold:.1%}. "
            "Check inference server logs.",
        ):
            self._failed_request_abort_triggered = False

    async def _request_profile_cancel(
        self, reason: ProfileCancelReason, reason_detail: str
    ) -> bool:
        """End the run through the cancel path. True if the request went out.

        A PUB socket does not receive its own message, so the local cancel
        handler runs directly after the broadcast; relying on the broadcast for
        local finalization would leave the run hanging.
        """
        try:
            command = ProfileCancelCommand(
                service_id=self.service_id,
                reason=reason,
                reason_detail=reason_detail,
            )
            await self.publish(command)
        except Exception as exc:
            self.warning(
                f"Failed to publish ProfileCancelCommand for {reason}: {exc!r}"
            )
            return False
        await self._on_profile_cancel_command(command)
        return True

    def _remember_profiling_credit_stats(self, stats: BasePhaseStats) -> None:
        """Keep the newest profiling credit-phase snapshot for the watchdog.

        ``RecordsTracker.create_aggregate_stats_for_phase`` returns a
        ``PhaseRecordsStats``, which counts *records* and has no
        ``in_flight_requests``/``requests_completed``. Only the credit side
        tracks what was sent versus what came back, so the watchdog reads this
        snapshot instead.

        Keyed by ``phase_index`` rather than held in a single slot: under
        ``seamless`` a later profiling phase starts while the previous one
        drains, and both keep publishing every interval. Sharing one slot let
        the two interleave, and because each carries its own
        ``requests_completed`` the count moved on nearly every tick -- which
        reads as progress and resets the stall clock, so a stuck request in
        either phase was never caught.
        """
        if isinstance(stats, CreditPhaseStats) and stats.phase == CreditPhase.PROFILING:
            self._latest_profiling_credit_stats[stats.phase_index] = stats

    def _longest_stalled_phase(
        self, now: float
    ) -> tuple[int | None, CreditPhaseStats, float] | None:
        """The profiling phase stuck longest, or ``None`` if all are moving.

        Each phase is judged on its own counters, and advances its own clock as
        a side effect. A single shared clock cannot work here: under
        ``seamless`` two profiling phases publish concurrently, and either one
        alone failing to complete is enough to hang the run.
        """
        worst: tuple[int | None, CreditPhaseStats, float] | None = None
        for phase_index, stats in self._latest_profiling_credit_stats.items():
            # A missing entry reads as "never seen", so the first sight of a
            # phase starts its clock rather than judging it.
            progressed = (
                stats.requests_completed
                != self._progress_stall_last_total.get(phase_index)
            )
            if stats.in_flight_requests <= 0 or progressed:
                # Nothing outstanding, or progress since the last tick. Reset
                # this phase's clock so a quiet stretch does not accumulate
                # toward a later, unrelated stall.
                self._progress_stall_last_total[phase_index] = stats.requests_completed
                self._progress_stall_since[phase_index] = now
                continue
            stalled_for = now - self._progress_stall_since.get(phase_index, now)
            if worst is None or stalled_for > worst[2]:
                worst = (phase_index, stats, stalled_for)
        return worst

    @background_task(
        interval=lambda self: Environment.RECORD.PROGRESS_STALL_CHECK_INTERVAL,
        immediate=False,
    )
    async def _watch_for_progress_stall(self) -> None:
        """Fail a run whose in-flight requests stop completing.

        Distinct from the record-stall handling, which covers aggregation
        falling behind *after* every credit has come back and finalizes with
        partial results. Here no credit ever comes back: a request that is
        dispatched and never completes leaves ``final_requests_completed``
        unset, so the completion barrier is not merely unmet but unevaluable,
        and nothing downstream can re-trigger it.
        ``--request-timeout-seconds`` is not a backstop -- it defaults to six
        hours and never applies to a request that was never dispatched.

        The condition is "requests are in flight and none are completing", not
        "no records arrived". Zero records is normal and expected whenever
        nothing is pending: a low request rate, a fixed-schedule replay sitting
        in a recorded idle gap, a slow dataset build. Only an outstanding
        request that never lands is a stall, and that is what makes the run
        unable to finish on its own.

        That also removes the need to gate on phase completion. A completed
        phase has nothing in flight, and ``_complete_credit_phases`` holds the
        phase *kind*, so gating on it would have disarmed the watchdog for
        every profiling phase after the first in a multi-phase run.
        """
        timeout = Environment.RECORD.PROGRESS_STALL_TIMEOUT
        if timeout <= 0:
            raise asyncio.CancelledError("progress stall watchdog disabled")
        if not self._profiling_started:
            return

        if not self._latest_profiling_credit_stats:
            # No profiling progress has been reported yet; there is nothing to
            # judge, and no clock may start before the first snapshot.
            return

        worst = self._longest_stalled_phase(time.monotonic())
        if worst is None:
            return
        stalled_index, stats, stalled_for = worst
        in_flight = stats.in_flight_requests
        total = stats.requests_completed
        phase_label = "" if stalled_index is None else f" in phase {stalled_index}"

        if stalled_for < timeout:
            # Stay quiet until a full check interval has genuinely elapsed:
            # the tick right after the clock resets would otherwise report
            # "0s with no completion", which reads as alarming and is not.
            if stalled_for >= Environment.RECORD.PROGRESS_STALL_CHECK_INTERVAL:
                self.warning(
                    f"{in_flight:,} request(s) in flight{phase_label} with no "
                    f"completion for {stalled_for:.0f}s "
                    f"({total:,} completed so far). "
                    f"Failing at {timeout:.0f}s "
                    "(AIPERF_RECORD_PROGRESS_STALL_TIMEOUT=0 to wait "
                    "indefinitely)."
                )
            return

        # Raising here would only be logged: @background_task defaults to
        # stop_on_error=False, so the task would retry forever and the run
        # would still hang -- the exact outcome this watchdog exists to end.
        # Cancel rather than publish a terminal result directly: a terminal
        # result fills only the profile domain, and the controller would still
        # wait on server_metrics, which cannot arrive until the profiling phase
        # completes -- which the stuck request is what prevents.
        if self._progress_stall_triggered:
            return
        self._progress_stall_triggered = True
        await self._request_profile_cancel(
            ProfileCancelReason.PROGRESS_STALL,
            f"Benchmark stalled: {in_flight:,} request(s) in flight"
            f"{phase_label} with no completion for {stalled_for:.0f}s "
            f"({total:,} completed). A "
            f"request was dispatched but never returned, so the phase can "
            f"never report complete. Set "
            f"AIPERF_RECORD_PROGRESS_STALL_TIMEOUT=0 to disable this check.",
        )

    def _maybe_hint_missing_cache_reporting(
        self, record_data: MetricRecordsData
    ) -> None:
        """Warn once, mid-run, when the server reports token usage but no prompt-cache
        reads — the signature of a cache-capable server that hasn't been told to
        report ``cached_tokens``. Fires on the first qualifying record so a long run
        can be aborted and re-launched with the flag set; the end-of-run console
        exporter emits the same hint for anyone who only reads the final summary.
        """
        if self._warned_missing_cache_reporting:
            return
        if usage_without_cache_in_record(record_data.metrics):
            self._warned_missing_cache_reporting = True
            self.warning(CACHE_REPORTING_HINT)

    @on_pull_message(MessageType.RECORDS)
    async def _on_records(self, message: RecordsMessage) -> None:
        """Handle a per-request records envelope generically.

        One ``RecordsMessage`` == one inference request. Each contained record
        self-identifies via its serialized ``record_type`` field and is dispatched
        to its registered handlers; the per-request lockstep keys off the message
        envelope (``message.metadata`` / ``message.error``), never off sniffing a
        record type.
        """
        if not await await_dataset_configured(
            self,
            self._dataset_configured_event,
            getattr(self, "_dataset_config_catch_up", None),
        ):
            return
        if self.is_trace_enabled:
            self.trace(f"Received records: {message}")

        phase = message.metadata.benchmark_phase

        # Context-overflow records in AGENTIC_REPLAY scenarios bypass normal
        # user-facing per-record processing but still advance the records-side
        # success counter so the completion barrier converges. Keep only a
        # narrow aggregate side-channel count for runtime submission validation.
        if getattr(message.metadata, "context_overflow_skip", False):
            await self._handle_context_overflow_skip(message, phase)
            return

        dispatch_errors: list[BaseException] = []
        for record in message.records:
            if isinstance(record, MetricRecordsData):
                self._maybe_hint_missing_cache_reporting(record)
            dispatch_errors.extend(await self._dispatch_record(record))

        self._records_tracker.update_from_request(message.metadata, message.error)
        if message.error:
            self._error_tracker.increment_error_count_for_phase(
                phase, message.error, phase_index=message.metadata.phase_index
            )
        # A metric accumulator/exporter that failed to ingest this record yields
        # incomplete metrics; surface it in the phase error summary rather than
        # marking the record cleanly processed and silently dropping the failure.
        for error in dispatch_errors:
            self._error_tracker.increment_error_count_for_phase(
                phase,
                ErrorDetails.from_exception(error),
                phase_index=message.metadata.phase_index,
            )

        await self._maybe_trigger_failed_request_abort(
            phase, phase_index=message.metadata.phase_index
        )

        if (
            phase in self._complete_credit_phases
            and (
                phase != CreditPhase.PROFILING
                or not self._has_multiple_profiling_phases()
                or self._credits_complete_received
            )
            and self._check_all_records_received(phase)
        ):
            await self._handle_all_records_received_once(phase)

    async def _handle_context_overflow_skip(
        self, message: RecordsMessage, phase: CreditPhase
    ) -> None:
        """Advance the records-side success counter for a skipped-overflow record."""
        self._skipped_context_overflow_counts_by_phase[phase] = (
            self._skipped_context_overflow_counts_by_phase.get(phase, 0) + 1
        )
        # Intentional skip: count as success so --failed-request-threshold and
        # console error counts stay honest. message.error (if any) describes
        # the overflow classification, not a failed request.
        self._records_tracker.update_from_request(message.metadata, None)
        if (
            phase in self._complete_credit_phases
            and (
                phase != CreditPhase.PROFILING
                or not self._has_multiple_profiling_phases()
                or self._credits_complete_received
            )
            and self._check_all_records_received(phase)
        ):
            await self._handle_all_records_received_once(phase)

    @on_pull_message(MessageType.TELEMETRY_RECORDS)
    async def _on_telemetry_records(self, message: TelemetryRecordsMessage) -> None:
        """Handle telemetry records message from Telemetry Manager."""
        if message.valid:
            for record in message.records:
                for error in await self._dispatch_record(record):
                    self._telemetry_state.error_counts[
                        ErrorDetails.from_exception(error)
                    ] += 1
        elif message.error:
            self._telemetry_state.error_counts[message.error] += 1

    @on_pull_message(MessageType.SERVER_METRICS_RECORD)
    async def _on_server_metrics_records(
        self, message: ServerMetricsRecordMessage
    ) -> None:
        """Handle server metrics record message from Server Metrics Manager."""
        if message.valid:
            for error in await self._dispatch_record(message.record):
                self._server_metrics_state.error_counts[
                    ErrorDetails.from_exception(error)
                ] += 1
        elif message.error:
            self._server_metrics_state.error_counts[message.error] += 1

    @on_pull_message(MessageType.NETWORK_LATENCY_RECORD)
    async def _on_network_latency_records(
        self, message: NetworkLatencyRecordMessage
    ) -> None:
        """Handle a network latency RTT probe sample from the NetworkLatencyManager."""
        if message.valid:
            if self._network_latency_accumulator is not None:
                self._network_latency_accumulator.add_sample(message.sample)
            for error in await self._dispatch_record(message.sample):
                self._network_latency_state.error_counts[
                    ErrorDetails.from_exception(error)
                ] += 1
        elif message.error:
            self._network_latency_state.error_counts[message.error] += 1

    async def _handle_all_records_received_once(self, phase: CreditPhase) -> None:
        """Idempotently trigger all-records finalization for one phase kind."""
        handled = getattr(self, "_all_records_received_phases", set())
        if phase in handled:
            return
        handled.add(phase)
        self._all_records_received_phases = handled
        await self._handle_all_records_received(phase)

    async def _handle_all_records_received(self, phase: CreditPhase) -> None:
        """Handle the case where all records have been received."""
        if phase != CreditPhase.PROFILING:
            self.debug(lambda: f"Skipping non-profiling phase: {phase}")
            return

        phase_stats = (
            self._records_tracker.create_aggregate_stats_for_phase(phase)
            if phase == CreditPhase.PROFILING
            else self._records_tracker.create_stats_for_phase(phase)
        )
        self.info(
            lambda: (
                f"Processed {phase_stats.success_records} valid requests and {phase_stats.error_records} errors ({phase_stats.total_records} total)."
            )
        )

        self.info("Received all records, processing now...")
        self.execute_async(
            self._finalize_and_process_results(
                phase=phase,
                cancelled=self._records_tracker.was_phase_cancelled(phase),
            )
        )
        await yield_to_event_loop()

    async def _finalize_and_process_results(
        self, phase: CreditPhase, cancelled: bool
    ) -> None:
        """Finalize server metrics collection and process results.

        This runs as a background task to avoid blocking the message pump.
        """
        phase_stats = (
            self._records_tracker.create_aggregate_stats_for_phase(phase)
            if phase == CreditPhase.PROFILING
            else self._records_tracker.create_stats_for_phase(phase)
        )

        # Send a message to the event bus to signal that we received all the records
        await self.publish(
            AllRecordsReceivedMessage(
                service_id=self.service_id,
                request_ns=time.time_ns(),
                final_processing_stats=phase_stats,
            )
        )

        # Trigger final server metrics scrape and wait for completion
        # This ensures final metrics are pushed before we export results
        response = await self.send_command_and_wait_for_response(
            ProfileCompleteCommand(service_id=self.service_id), timeout=10.0
        )

        if isinstance(response, ErrorDetails):
            self.warning(f"Server metrics final scrape timed out or failed: {response}")
        else:
            self.debug("Server metrics final scrape completed")

        self.debug("Waiting for server metrics flush period...")
        # Wait for server metrics flush period to allow final metrics to be collected
        # This ensures metrics that are still being processed by the server are captured
        flush_period = Environment.SERVER_METRICS.COLLECTION_FLUSH_PERIOD
        phase_stats = self._records_tracker.create_aggregate_stats_for_phase(
            CreditPhase.PROFILING
        )
        flush_end_ns = (phase_stats.requests_end_ns or time.time_ns()) + (
            (flush_period or 0) * NANOS_PER_SECOND
        )
        sleep_dur_sec = (flush_end_ns - time.time_ns()) / NANOS_PER_SECOND
        if sleep_dur_sec > 0:
            self.info(
                f"Waiting {sleep_dur_sec:.1f}s for server metrics flush period..."
            )
            await asyncio.sleep(sleep_dur_sec)

        self.debug("Server metrics flush period complete, processing now...")
        await self._process_results(phase=phase, cancelled=cancelled)
        self.info("_finalize_and_process_results completed")

    @on_message(MessageType.DATASET_CONFIGURED_NOTIFICATION)
    async def _on_dataset_configured(
        self, message: DatasetConfiguredNotification
    ) -> None:
        for handler in (*self._accumulators.values(), *self._stream_exporters.values()):
            if hasattr(handler, "on_dataset_configured"):
                handler.on_dataset_configured(message.metadata)
        self._dataset_configured_event.set()

    @on_message(MessageType.CREDIT_PHASE_START)
    async def _on_credit_phase_start(
        self, phase_start_msg: CreditPhaseStartMessage
    ) -> None:
        """Handle a credit phase start message in order to track the total number of expected requests."""
        self._records_tracker.update_phase_info(phase_start_msg.stats)
        await self._dispatch_record(phase_start_msg.stats, warn_if_unrouted=False)
        if phase_start_msg.config.phase == CreditPhase.PROFILING:
            self._profiling_started = True
        self.info(f"Credit phase start: {phase_start_msg.config.phase}")

    @on_message(MessageType.CREDIT_PHASE_PROGRESS)
    async def _on_credit_phase_progress(
        self, message: CreditPhaseProgressMessage
    ) -> None:
        """Handle a credit phase progress message to track and stream live timing snapshots."""
        self._remember_profiling_credit_stats(message.stats)
        self._records_tracker.update_phase_info(message.stats)
        await self._dispatch_record(message.stats, warn_if_unrouted=False)

    @on_message(MessageType.CREDIT_PHASE_SENDING_COMPLETE)
    async def _on_credit_phase_sending_complete(
        self, message: CreditPhaseSendingCompleteMessage
    ) -> None:
        """Handle a credit phase sending complete message in order to track the final request count."""
        if message.stats.phase == CreditPhase.PROFILING:
            self.info(
                f"Sent {message.stats.final_requests_sent:,} requests. Waiting for all to complete..."
            )
        self._remember_profiling_credit_stats(message.stats)
        self._records_tracker.update_phase_info(message.stats)
        await self._dispatch_record(message.stats, warn_if_unrouted=False)

    @on_message(MessageType.CREDIT_PHASE_COMPLETE)
    async def _on_credit_phase_complete(
        self, message: CreditPhaseCompleteMessage
    ) -> None:
        """Handle a credit phase complete message in order to track the end time, and check if all records have been received."""
        self._remember_profiling_credit_stats(message.stats)
        self._records_tracker.update_phase_info(message.stats)
        await self._dispatch_record(message.stats, warn_if_unrouted=False)
        self._complete_credit_phases.add(message.stats.phase)
        # Capture per-phase BranchStats for any phase that publishes them.
        if message.branch_stats is not None:
            self._phase_branch_stats[message.stats.phase] = message.branch_stats
        if message.stats.phase == CreditPhase.PROFILING:
            # Capture the BranchStats snapshot so it flows into
            # ProfileResults when the records pipeline finalizes.
            # Non-DAG runs publish None and leave this unset.
            if message.branch_stats is not None:
                self._latest_branch_stats = message.branch_stats
            phase_stats = self._records_tracker.create_stats_for_phase(
                message.stats.phase
            )
            self.info(
                lambda: (
                    f"Received CREDIT_PHASE_COMPLETE message, Phase complete: {phase_stats!r}"
                )
            )
            if phase_stats.final_requests_completed is None:
                self.notice(
                    "Phase completion observed before final request count was available; "
                    f"waiting for final phase stats (currently {phase_stats.total_records:,} records processed)..."
                )
            else:
                self.notice(
                    "All requests have completed, please wait for the results to be processed "
                    f"(currently {phase_stats.total_records:,} of {phase_stats.final_requests_completed:,} records processed)..."
                )

        # This check is to prevent a race condition where the records manager processes
        # all records before the timing manager has sent the final completed count.
        if (
            message.stats.phase != CreditPhase.PROFILING
            or not self._has_multiple_profiling_phases()
            or self._credits_complete_received
        ) and self._check_all_records_received(message.stats.phase):
            await self._handle_all_records_received_once(message.stats.phase)

    def _snapshot_branch_stats(self, phase: CreditPhase) -> BranchStats | None:
        """Return the orchestrator-published BranchStats for ``phase``.

        Returns ``None`` for non-DAG runs or for phases where the
        TimingManager never published sub-agent counters on
        ``CreditPhaseCompleteMessage``.
        """
        return self._phase_branch_stats.get(phase)

    @on_message(MessageType.CREDITS_COMPLETE)
    async def _on_credits_complete(self, message: CreditsCompleteMessage) -> None:
        """Handle a credits complete message in order to track the end time, and check if all records have been received."""
        self.info(
            "All credits complete, please wait for the results to be processed..."
        )
        self._credits_complete_received = True
        if (
            CreditPhase.PROFILING in self._complete_credit_phases
            and self._records_tracker.check_and_set_all_records_received_for_phase(
                CreditPhase.PROFILING
            )
        ):
            await self._handle_all_records_received_once(CreditPhase.PROFILING)

    @background_task(
        interval=Environment.RECORD.PROGRESS_REPORT_INTERVAL, immediate=False
    )
    async def _report_records_task(self) -> None:
        """Report the records processing stats."""
        active_phase_stats = self._records_tracker.create_stats_for_phase(
            CreditPhase.PROFILING
        )
        if active_phase_stats.total_records == 0:
            return  # TODO: What about worker stats?
        overall_worker_stats = self._records_tracker.create_overall_worker_stats()
        await self._publish_processing_stats(active_phase_stats, overall_worker_stats)

    async def _publish_processing_stats(
        self,
        phase_stats: PhaseRecordsStats,
        worker_stats: dict[str, WorkerProcessingStats],
    ) -> None:
        """Publish the profile processing stats."""
        message = RecordsProcessingStatsMessage(
            service_id=self.service_id,
            request_ns=time.time_ns(),
            processing_stats=phase_stats,
            worker_stats=worker_stats,
        )
        await self.publish(message)

    @on_command(CommandType.PROCESS_RECORDS)
    async def _on_process_records_command(
        self, message: ProcessRecordsCommand
    ) -> ProcessRecordsResult:
        """Handle the process records command by forwarding it to all of the results processors, and returning the results."""
        self.debug(lambda: f"Received process records command: {message}")
        return await self._process_results(
            phase=CreditPhase.PROFILING, cancelled=message.cancelled
        )

    @on_command(CommandType.PROFILE_CANCEL)
    async def _on_profile_cancel_command(
        self, message: ProfileCancelCommand
    ) -> ProcessRecordsResult:
        """Handle the profile cancel command by processing current results.

        This marks the phase as cancelled in the records tracker and processes
        all currently received records. Called when user presses Ctrl+C.
        """
        self.warning(f"Received profile cancel command: {message}")

        # Mark the phase as cancelled in the tracker
        self._records_tracker.mark_phase_cancelled(CreditPhase.PROFILING)

        return await self._process_results(phase=CreditPhase.PROFILING, cancelled=True)

    @property
    def service_config(self) -> BenchmarkConfig:
        """The resolved benchmark config for this run.

        Compatibility accessor for the realtime-stats path: the renderer gate
        reads ``service_config.ui_type`` so headless runs emit the per-tick log
        block while ``--ui dashboard`` suppresses it (the dashboard renders the
        same metrics itself).
        """
        return self.run.cfg

    @background_task(interval=None, immediate=True)
    async def _report_realtime_inference_metrics_task(self) -> None:
        """Report inference metrics at regular intervals.

        The dashboard/realtime gate is checked inside the loop so the framework's
        ``interval=None`` semantics (run body once and break) don't permanently
        kill the task when the gate is currently False — see
        ``task_manager_mixin.py`` rule for ``interval=None``.

        ``--stats-interval 0`` disables only the per-tick log block. The
        ``RealtimeMetricsMessage`` keeps publishing for dashboards / k8s job-WS
        subscribers at the per-UI default cadence (the value
        ``realtime_metrics_interval`` returns when unset), so a
        ``--ui dashboard --stats-interval 0`` run still drives the live panel.
        """
        configured_interval = self.run.cfg.runtime.realtime_metrics_interval(
            self.run.cfg.ui_type
        )
        log_block_enabled = configured_interval != 0
        # When the log block is disabled (interval 0), still tick the publish
        # loop at a sane per-UI default cadence instead of busy-spinning.
        interval = (
            configured_interval
            if log_block_enabled
            else self._default_realtime_interval()
        )
        while not self.stop_requested:
            await asyncio.sleep(interval)

            if (
                self.run.cfg.ui_type != UIType.DASHBOARD
                and not Environment.UI.REALTIME_METRICS_ENABLED
            ):
                continue

            phase_stats = self._records_tracker.create_stats_for_phase(
                CreditPhase.PROFILING
            )
            server_snapshot = self._collect_realtime_server_snapshot(
                start_ns=phase_stats.start_ns
            )
            if not self._has_realtime_update(
                phase_stats.total_records, server_snapshot
            ):
                continue
            self._previous_realtime_records = phase_stats.total_records
            self._previous_realtime_server_snapshot = dict(server_snapshot)
            await self._report_realtime_metrics(
                server_snapshot=server_snapshot,
                emit_log_block=log_block_enabled,
            )

    def _default_realtime_interval(self) -> float:
        """Resolve the per-UI default realtime cadence (interval-0 publish fallback).

        Mirrors ``realtime_metrics_interval`` when ``REALTIME_METRICS_INTERVAL``
        is unset: 5.0s under ``--ui dashboard``, 30.0s otherwise. Used so the
        dashboard keeps polling even when the log block is disabled with
        ``--stats-interval 0``.
        """
        return 5.0 if self.run.cfg.ui_type == UIType.DASHBOARD else 30.0

    def _has_realtime_update(
        self, total_records: int, server_snapshot: dict[str, float]
    ) -> bool:
        """Whether the realtime block needs rebuilding this tick.

        True when EITHER the record count OR the live server-metrics snapshot
        (cache hit rate, KV usage, queue depth) changed since the last emit.
        Gating on record count alone froze the server-metrics row whenever the
        count was momentarily static during a lull.
        """
        return (
            total_records != self._previous_realtime_records
            or server_snapshot != self._previous_realtime_server_snapshot
        )

    @on_command(CommandType.START_REALTIME_TELEMETRY)
    async def _on_start_realtime_telemetry_command(
        self, message: StartRealtimeTelemetryCommand
    ) -> None:
        """Handle command to start the realtime telemetry background task.

        This is called when the user dynamically enables the telemetry dashboard
        by pressing the telemetry option in the UI without having passed the 'dashboard' parameter
        at startup.
        """
        if self._gpu_telemetry_accumulator:
            self._gpu_telemetry_accumulator.start_realtime_telemetry()
        else:
            self.error(
                "GPU telemetry accumulator not found, cannot start realtime telemetry"
            )

    @on_command(CommandType.REALTIME_METRICS)
    async def _on_realtime_metrics_command(
        self, message: RealtimeMetricsCommand
    ) -> None:
        """Handle a real-time metrics command."""
        await self._report_realtime_metrics()

    def _collect_realtime_server_snapshot(
        self, start_ns: int | None = None
    ) -> dict[str, float]:
        """Return the current live server metrics snapshot, if available."""
        server_snapshot: dict[str, float] = {}
        if self._server_metrics_accumulator is None:
            return server_snapshot
        try:
            snapshot_fn = getattr(
                self._server_metrics_accumulator,
                "realtime_snapshot",
                None,
            )
            if callable(snapshot_fn):
                server_snapshot = snapshot_fn(start_ns=start_ns) or {}
        except Exception as exc:  # noqa: BLE001
            self.debug(lambda exc=exc: f"server_snapshot failed: {exc!r}")
        return server_snapshot

    async def _report_realtime_metrics(
        self,
        server_snapshot: dict[str, float] | None = None,
        emit_log_block: bool = True,
    ) -> None:
        """Report inference metrics (used by command handler).

        Publishes a ``RealtimeMetricsMessage`` for the dashboard / k8s job-WS
        subscribers, then — for non-dashboard UIs — renders and emits the
        per-tick realtime stats log block (one log record per line so the rows
        don't interleave with other services' writes on the shared console
        stream). The dashboard renders the same metrics itself, so the log
        block is suppressed under ``--ui dashboard``. ``emit_log_block=False``
        (set when ``--stats-interval 0`` disables the log block) suppresses the
        log line while still publishing the message for dashboards.
        """
        phase_stats = self._records_tracker.create_stats_for_phase(
            CreditPhase.PROFILING
        )
        # Realtime metrics only need the metric_records accumulators —
        # GPU telemetry / server metrics live on separate fan-outs.
        raw_metrics = await generate_realtime_metrics(
            self._metric_record_accumulators,
            phase_index=phase_stats.phase_index,
        )
        if server_snapshot is None:
            server_snapshot = self._collect_realtime_server_snapshot(
                start_ns=phase_stats.start_ns
            )

        publish_metrics = [
            *raw_metrics,
            *_server_snapshot_to_metric_results(server_snapshot),
        ]
        display_metrics = records_manager_processing.filter_display_metrics(
            publish_metrics
        )
        if not display_metrics:
            return
        await self.publish(
            RealtimeMetricsMessage(
                service_id=self.service_id,
                metrics=display_metrics,
            )
        )

        # Realtime block uses the *raw* (unfiltered) metric set so per-user
        # throughput rows can show ``prefill_throughput_per_user`` etc. —
        # those have ``console_group=NONE`` (hidden from the dashboard table)
        # and ``filter_display_metrics`` strips them, leaving the row blank.
        prev_realtime_phase_index = getattr(self, "_prev_realtime_phase_index", None)
        prev_realtime_snapshot = (
            self._prev_realtime_snapshot
            if prev_realtime_phase_index == phase_stats.phase_index
            else None
        )
        rendered = _render_realtime_block(
            raw_metrics,
            phase_stats,
            prev_realtime_snapshot,
            server_snapshot=server_snapshot,
        )
        if rendered:
            self._prev_realtime_snapshot = (
                phase_stats.total_records,
                phase_stats.records_elapsed_time,
            )
            self._prev_realtime_phase_index = phase_stats.phase_index
            if emit_log_block and self.run.cfg.ui_type != UIType.DASHBOARD:
                # One record per line: multi-line records interleave with
                # other services' writes on the shared console stream.
                for line in rendered.splitlines():
                    self.info(line)

    async def _run_analyzers(self, ctx: SummaryContext) -> list[MetricResult]:
        """Run summarize-time analyzer plugins that join across accumulators.

        An analyzer is skipped unless every accumulator it needs a LIVE instance
        of (``required_accumulators``) is loaded AND every accumulator whose
        SUMMARY it reads (``required_summaries``) was produced — e.g. the
        energy-efficiency analyzer queries the live GPU accumulator and reads the
        metrics summary. One analyzer's failure is logged and does not abort the
        rest. Returns the flattened MetricResults to merge into the summary.
        """
        if not self._analyzers:
            return []
        loaded = {str(acc_type) for acc_type in self._accumulators}
        summarized = {str(acc_type) for acc_type in ctx.accumulator_outputs}
        results: list[MetricResult] = []
        for loaded_analyzer in self._analyzers:
            analyzer = loaded_analyzer.analyzer
            name = analyzer.__class__.__name__
            missing_acc = [
                r for r in loaded_analyzer.required_accumulators if r not in loaded
            ]
            missing_sum = [
                r for r in loaded_analyzer.required_summaries if r not in summarized
            ]
            if missing_acc or missing_sum:
                self.debug(
                    lambda n=name, a=missing_acc, s=missing_sum: (
                        f"Skipping analyzer {n}: missing accumulators {a}, summaries {s}"
                    )
                )
                continue
            try:
                results.extend(await analyzer.analyze(ctx))
            except Exception as e:  # noqa: BLE001 - one analyzer must not abort the summary
                self.error(f"Analyzer {name} failed: {e!r}")
        return results

    async def _summarize_one_accumulator(
        self,
        acc_type: AccumulatorType,
        accumulator: AccumulatorProtocol,
        ctx: ExportContext,
    ) -> tuple[AccumulatorType, object]:
        """Run summarize/export_results on a single accumulator with timeout.

        Returns the result (or exception object) so a single bad accumulator
        cannot abort the rest. Accumulators that support phase/window-scoped
        export (marked with ``supports_phase_scoped_export`` — MetricsAccumulator)
        get ``export_results(ctx)`` so warmup
        records are excluded from profiling summaries; otherwise prefers
        ``summarize()`` and falls back to ``export_results(ctx)``.
        """
        name = accumulator.__class__.__name__
        self.debug(f"Starting summarize for accumulator {acc_type}: {name}")
        try:
            # ``is True`` (not truthiness) so a MagicMock's auto-created attribute
            # does not spuriously route mock accumulators through export_results.
            if getattr(accumulator, "supports_phase_scoped_export", False) is True and (
                hasattr(accumulator, "export_results")
            ):
                res = await asyncio.wait_for(
                    accumulator.export_results(ctx),
                    timeout=Environment.RECORD.PROCESS_RECORDS_TIMEOUT,
                )
            elif hasattr(accumulator, "summarize"):
                res = await asyncio.wait_for(
                    accumulator.summarize(),
                    timeout=Environment.RECORD.PROCESS_RECORDS_TIMEOUT,
                )
            else:
                res = await asyncio.wait_for(
                    accumulator.export_results(ctx),
                    timeout=Environment.RECORD.PROCESS_RECORDS_TIMEOUT,
                )
            self.debug(f"Completed summarize for accumulator {acc_type}: {name}")
            return acc_type, res
        except Exception as e:  # noqa: BLE001 - one bad accumulator must not abort the rest
            self.error(f"Error in summarize for accumulator {acc_type} ({name}): {e!r}")
            return acc_type, e

    def _bucket_accumulator_summary(
        self,
        acc_type: AccumulatorType,
        summary: object,
        records_results: list[MetricResult],
        error_results: list[ErrorDetails],
    ) -> list[TimesliceResult]:
        """Route a single accumulator summary into the right ProfileResults bucket."""
        timeslices: list[TimesliceResult] = []
        if isinstance(summary, BaseException):
            error_results.append(ErrorDetails.from_exception(summary))
        elif isinstance(summary, AccumulatorMetricsSummary):
            records_results.extend(summary.results.values())
            if summary.timeslices is not None:
                timeslices = summary.timeslices
        elif isinstance(summary, list):
            records_results.extend(r for r in summary if isinstance(r, MetricResult))
        elif isinstance(summary, ErrorDetails):
            error_results.append(summary)
        else:
            self.debug(
                lambda s=summary, a=acc_type: (
                    f"Accumulator {a} returned unrecognized shape: {type(s).__name__}"
                )
            )
        return timeslices

    async def _summarize_metric_record_accumulators(
        self, phase: CreditPhase, cancelled: bool
    ) -> tuple[
        list[MetricResult], list[TimesliceResult], list[ErrorDetails], SummaryContext
    ]:
        """Summarize the metric_records accumulators (the byte-exact engine).

        Telemetry / server-metrics accumulators are summarized separately via
        the dedicated ``_publish_telemetry_results`` / ``_publish_server_metrics_results``
        side-channels so they are not double-processed here.

        Also returns a populated ``SummaryContext`` — every loaded accumulator
        instance plus the metric_records summaries keyed by ``AccumulatorType`` —
        so summarize-time ``analyzer`` plugins can join across accumulators
        (e.g. energy efficiency joins GPU telemetry to inference tokens).
        """
        records_results: list[MetricResult] = []
        timeslices: list[TimesliceResult] = []
        error_results: list[ErrorDetails] = []

        phase_stats = RecordsManager._create_result_stats_for_phase(self, phase)
        summary_ctx = SummaryContext(
            accumulators=dict(self._accumulators),
            start_ns=phase_stats.start_ns or 0,
            end_ns=phase_stats.requests_end_ns or 0,
            phase=phase,
            cancelled=cancelled,
        )

        # Only the metric_records-typed accumulators feed the summary records.
        acc_items = [
            (acc_type, acc)
            for acc_type, acc in self._accumulators.items()
            if acc in self._metric_record_accumulators
        ]
        if not acc_items:
            return records_results, timeslices, error_results, summary_ctx

        ctx = ExportContext(
            start_ns=phase_stats.start_ns,
            end_ns=phase_stats.requests_end_ns,
            phase=phase,
            error_summary=self._error_tracker.get_error_summary_for_phase(phase),
            cancelled=cancelled,
        )
        summaries = await asyncio.gather(
            *[
                self._summarize_one_accumulator(acc_type, acc, ctx)
                for acc_type, acc in acc_items
            ],
            return_exceptions=False,
        )
        for acc_type, summary in summaries:
            # Expose each accumulator's summary for cross-accumulator analyzers.
            summary_ctx.accumulator_outputs[acc_type] = summary
            ts = self._bucket_accumulator_summary(
                acc_type, summary, records_results, error_results
            )
            if ts:
                timeslices = ts
        self._adjust_multi_phase_aggregate_rates(phase, records_results)
        return records_results, timeslices, error_results, summary_ctx

    def _adjust_multi_phase_aggregate_rates(
        self, phase: CreditPhase, records_results: list[MetricResult]
    ) -> None:
        if not self._has_multiple_phase_instances(phase):
            return
        duration_ns = self._phase_active_duration_ns(phase)
        if not duration_ns:
            return
        by_tag = {result.tag: result for result in records_results}
        duration_sec = duration_ns / NANOS_PER_SECOND
        self._set_metric_avg(by_tag, "benchmark_duration", duration_sec)
        self._set_rate_metric(
            by_tag, "request_throughput", "request_count", duration_sec
        )
        self._set_rate_metric(
            by_tag, "input_token_throughput", "total_isl", duration_sec
        )
        self._set_rate_metric(
            by_tag, "output_token_throughput", "total_osl", duration_sec
        )
        self._set_rate_metric(by_tag, "goodput", "good_request_count", duration_sec)

        total_isl = self._metric_avg(by_tag, "total_isl")
        total_osl = self._metric_avg(by_tag, "total_osl")
        if total_isl is not None and total_osl is not None:
            self._set_metric_avg(
                by_tag, "total_token_throughput", (total_isl + total_osl) / duration_sec
            )

    def _phase_active_duration_ns(self, phase: CreditPhase) -> int | None:
        total = 0
        for stats in self._iter_concrete_phase_stats(phase):
            if stats.start_ns is None or stats.requests_end_ns is None:
                continue
            duration = stats.requests_end_ns - stats.start_ns
            if duration > 0:
                total += duration
        return total or None

    @staticmethod
    def _metric_avg(
        by_tag: dict[MetricTagT, MetricResult], tag: MetricTagT
    ) -> float | None:
        result = by_tag.get(tag)
        value = getattr(result, "avg", None)
        return float(value) if value is not None else None

    @classmethod
    def _set_rate_metric(
        cls,
        by_tag: dict[MetricTagT, MetricResult],
        rate_tag: MetricTagT,
        numerator_tag: MetricTagT,
        duration_sec: float,
    ) -> None:
        numerator = cls._metric_avg(by_tag, numerator_tag)
        if numerator is None:
            return
        cls._set_metric_avg(by_tag, rate_tag, numerator / duration_sec)

    @staticmethod
    def _set_metric_avg(
        by_tag: dict[MetricTagT, MetricResult], tag: MetricTagT, value: float
    ) -> None:
        result = by_tag.get(tag)
        if result is not None:
            result.avg = value

    def _create_result_stats_for_phase(self, phase: CreditPhase) -> PhaseRecordsStats:
        if self._has_multiple_phase_instances(phase):
            return self._records_tracker.create_aggregate_stats_for_phase(phase)
        return self._records_tracker.create_stats_for_phase(phase)

    def _iter_concrete_phase_stats(
        self, phase: CreditPhase | None = None
    ) -> list[PhaseRecordsStats]:
        phase_trackers = getattr(self._records_tracker, "_phase_trackers", {})
        if not isinstance(phase_trackers, dict):
            return []
        stats = [
            tracker.create_stats()
            for (tracker_phase, phase_index), tracker in phase_trackers.items()
            if (phase is None or tracker_phase == phase) and phase_index is not None
        ]
        return sorted(
            stats,
            key=lambda item: (
                item.phase_index if item.phase_index is not None else 10**9
            ),
        )

    async def _build_phase_profile_results(
        self, phase: CreditPhase, cancelled: bool
    ) -> list[PhaseProfileResults] | None:
        concrete_phase_stats = self._iter_concrete_phase_stats()
        if len(concrete_phase_stats) <= 1:
            return None

        acc_items = [
            (acc_type, acc)
            for acc_type, acc in self._accumulators.items()
            if acc in self._metric_record_accumulators
        ]
        cfg = getattr(getattr(self, "run", None), "cfg", None)
        telemetry_errors = self._phase_error_counts(self._telemetry_state.error_counts)
        server_metrics_errors = self._phase_error_counts(
            self._server_metrics_state.error_counts
        )

        phase_results: list[PhaseProfileResults] = []
        for stats in concrete_phase_stats:
            if stats.phase_index is None:
                continue
            ctx = self._phase_metric_export_context(stats, cancelled)
            records_results, error_results = await self._phase_metric_results(
                acc_items, ctx
            )
            telemetry_results, telemetry_warnings = await self._phase_telemetry_results(
                stats,
                telemetry_errors,
                bool(getattr(cfg, "gpu_telemetry_disabled", False)),
            )
            (
                server_metrics_results,
                server_metrics_warnings,
            ) = await self._phase_server_metrics_results(
                stats,
                server_metrics_errors,
                bool(getattr(cfg, "server_metrics_disabled", False)),
            )
            phase_results.append(
                self._create_phase_profile_result(
                    stats=stats,
                    cancelled=cancelled,
                    records_results=records_results,
                    error_results=error_results,
                    telemetry_results=telemetry_results,
                    server_metrics_results=server_metrics_results,
                    telemetry_warnings=telemetry_warnings,
                    server_metrics_warnings=server_metrics_warnings,
                )
            )
        return phase_results or None

    @staticmethod
    def _phase_error_counts(
        error_counts: dict[ErrorDetails, int],
    ) -> list[ErrorDetailsCount]:
        return [
            ErrorDetailsCount(error_details=error_details, count=count)
            for error_details, count in error_counts.items()
        ]

    def _phase_metric_export_context(
        self, stats: PhaseRecordsStats, cancelled: bool
    ) -> ExportContext:
        return ExportContext(
            start_ns=stats.start_ns,
            end_ns=stats.requests_end_ns,
            phase=stats.phase,
            phase_index=stats.phase_index,
            phase_name=stats.phase_name,
            phase_kind=stats.phase_kind or stats.phase.value,
            is_phase_scoped=True,
            error_summary=self._error_tracker.get_error_summary_for_phase(
                stats.phase, phase_index=stats.phase_index
            ),
            cancelled=cancelled,
        )

    @staticmethod
    def _phase_baseline_export_context(
        stats: PhaseRecordsStats, error_summary: list[ErrorDetailsCount]
    ) -> ExportContext:
        return ExportContext(
            start_ns=stats.baseline_start_ns or stats.start_ns,
            end_ns=stats.baseline_end_ns or stats.requests_end_ns,
            phase=stats.phase,
            phase_index=stats.phase_index,
            phase_name=stats.phase_name,
            phase_kind=stats.phase_kind or stats.phase.value,
            is_phase_scoped=True,
            error_summary=error_summary,
        )

    async def _phase_metric_results(
        self,
        acc_items: list[tuple[AccumulatorType, AccumulatorProtocol]],
        ctx: ExportContext,
    ) -> tuple[list[MetricResult], list[ErrorDetails]]:
        records_results: list[MetricResult] = []
        error_results: list[ErrorDetails] = []
        if not acc_items:
            return records_results, error_results

        summaries = await asyncio.gather(
            *[
                self._summarize_one_accumulator(acc_type, acc, ctx)
                for acc_type, acc in acc_items
            ],
            return_exceptions=False,
        )
        for acc_type, summary in summaries:
            self._bucket_accumulator_summary(
                acc_type, summary, records_results, error_results
            )
        return records_results, error_results

    def _validate_profile_metric_duration_coverage(
        self,
        phase: CreditPhase,
        cancelled: bool,
    ) -> tuple[list[ProfileMetricDurationCoverage], list[ErrorDetails]]:
        """Apply a scenario's post-run latency-signal coverage requirement."""
        if phase != CreditPhase.PROFILING or cancelled:
            return [], []

        scenario_name = self.run.cfg.scenario
        if scenario_name is None:
            return [], []
        # Scenario resolution validates the name before services start.
        scenario_spec = get_scenario(scenario_name)
        required_ratio = scenario_spec.minimum_profile_metric_coverage_ratio
        if required_ratio is None:
            return [], []

        accumulator = self._accumulators.get(AccumulatorType.METRIC_RESULTS)
        calculate = getattr(accumulator, "profile_metric_duration_coverage", None)
        if not callable(calculate):
            error = ErrorDetails(
                type="ProfileMetricCoverageError",
                message=(
                    "Profiling metric coverage could not be validated because the "
                    "metrics accumulator does not expose coverage timestamps."
                ),
            )
            return [], [error]

        phase_configs = self.run.cfg.get_profiling_phases()
        concrete_stats = self._iter_concrete_phase_stats(CreditPhase.PROFILING)
        aggregate_stats = RecordsManager._create_result_stats_for_phase(
            self, CreditPhase.PROFILING
        )
        coverage_results: list[ProfileMetricDurationCoverage] = []
        fatal_errors: list[ErrorDetails] = []

        for profiling_index, phase_config in enumerate(phase_configs):
            if phase_config.duration is None:
                continue
            if phase_config.duration < scenario_spec.min_benchmark_duration_seconds:
                self.info(
                    "Skipping profiling metric coverage validation for "
                    f"{phase_config.name!r}: configured duration "
                    f"{float(phase_config.duration):.1f}s is below scenario minimum "
                    f"{scenario_spec.min_benchmark_duration_seconds}s."
                )
                continue
            stats = next(
                (
                    item
                    for item in concrete_stats
                    if item.phase_name == phase_config.name
                    or item.profiling_index == profiling_index
                ),
                aggregate_stats if len(phase_configs) == 1 else None,
            )
            if stats is None or stats.start_ns is None:
                fatal_errors.append(
                    ErrorDetails(
                        type="ProfileMetricCoverageError",
                        message=(
                            "Profiling metric coverage could not be validated for "
                            f"phase {phase_config.name!r} because its start time is "
                            "unavailable."
                        ),
                    )
                )
                continue

            ctx = ExportContext(
                start_ns=stats.start_ns,
                end_ns=stats.requests_end_ns,
                phase=CreditPhase.PROFILING,
                phase_index=stats.phase_index,
                phase_name=phase_config.name,
                phase_kind="profiling",
                is_phase_scoped=True,
                cancelled=False,
            )
            coverage = calculate(
                ctx,
                phase_name=phase_config.name,
                expected_duration_seconds=float(phase_config.duration),
                required_ratio=required_ratio,
            )
            last_content_ns = stats.last_streaming_content_ns or stats.start_ns
            coverage.streaming_content_ratio = min(
                max(
                    (last_content_ns - stats.start_ns)
                    / (float(phase_config.duration) * NANOS_PER_SECOND),
                    0.0,
                ),
                1.0,
            )
            coverage_results.append(coverage)
            if coverage.passed:
                self.info(
                    "Profiling metric coverage passed for "
                    f"{phase_config.name!r}: TTFT={coverage.ttft_ratio:.1%}, "
                    "inter-token latency="
                    f"{coverage.inter_token_latency_ratio:.1%}, "
                    f"streaming content={coverage.streaming_content_ratio:.1%} "
                    f"(required={required_ratio:.1%})."
                )
                continue

            allowed_tail_seconds = float(phase_config.duration) * (1.0 - required_ratio)
            message = (
                f"Profiling metric coverage below the required {required_ratio:.1%} "
                f"for phase {phase_config.name!r}: TTFT={coverage.ttft_ratio:.1%}, "
                "inter-token latency="
                f"{coverage.inter_token_latency_ratio:.1%}, "
                f"streaming content={coverage.streaming_content_ratio:.1%} over the configured "
                f"{float(phase_config.duration):.1f}s duration. No latency or streaming "
                f"content signal extended into the final {allowed_tail_seconds:.1f}s before "
                "the nominal profiling end; check inference server logs for a stalled "
                "or unavailable server."
            )
            self.error(message)
            fatal_errors.append(
                ErrorDetails(
                    type="ProfileMetricCoverageError",
                    message=message,
                    details=coverage.model_dump(mode="json"),
                )
            )

        return coverage_results, fatal_errors

    async def _phase_telemetry_results(
        self,
        stats: PhaseRecordsStats,
        telemetry_errors: list[ErrorDetailsCount],
        disabled: bool,
    ) -> tuple[Any | None, list[str]]:
        if self._gpu_telemetry_accumulator is None or disabled:
            return None, []
        try:
            results = await self._gpu_telemetry_accumulator.export_results(
                self._phase_baseline_export_context(stats, telemetry_errors)
            )
        except Exception as e:  # noqa: BLE001 - phase artifact remains best-effort
            return None, [f"GPU telemetry phase export failed: {type(e).__name__}: {e}"]
        if results is None or not getattr(results, "endpoints", None):
            return None, []
        return results, []

    async def _phase_server_metrics_results(
        self,
        stats: PhaseRecordsStats,
        server_metrics_errors: list[ErrorDetailsCount],
        disabled: bool,
    ) -> tuple[Any | None, list[str]]:
        if self._server_metrics_accumulator is None or disabled:
            return None, []
        try:
            results = await self._server_metrics_accumulator.export_results(
                self._phase_baseline_export_context(stats, server_metrics_errors)
            )
        except Exception as e:  # noqa: BLE001 - phase artifact remains best-effort
            return None, [
                f"Server metrics phase export failed: {type(e).__name__}: {e}"
            ]
        if results is None or not getattr(results, "endpoint_summaries", None):
            return None, []
        return results, []

    def _create_phase_profile_result(
        self,
        *,
        stats: PhaseRecordsStats,
        cancelled: bool,
        records_results: list[MetricResult],
        error_results: list[ErrorDetails],
        telemetry_results: Any | None,
        server_metrics_results: Any | None,
        telemetry_warnings: list[str],
        server_metrics_warnings: list[str],
    ) -> PhaseProfileResults:
        return PhaseProfileResults(
            phase_index=stats.phase_index,
            profiling_index=stats.profiling_index,
            phase_name=stats.phase_name or f"{stats.phase.value}_{stats.phase_index}",
            phase_kind=stats.phase_kind or stats.phase.value,
            records=records_results,
            start_ns=stats.start_ns,
            end_ns=stats.requests_end_ns,
            baseline_start_ns=stats.baseline_start_ns,
            baseline_end_ns=stats.baseline_end_ns,
            was_cancelled=stats.was_cancelled or cancelled,
            successful_request_count=stats.success_records,
            error_request_count=stats.error_records,
            error_summary=self._error_tracker.get_error_summary_for_phase(
                stats.phase, phase_index=stats.phase_index
            ),
            telemetry_results=telemetry_results,
            server_metrics_results=server_metrics_results,
            telemetry_warnings=telemetry_warnings,
            server_metrics_warnings=server_metrics_warnings,
        )

    def _has_records_for_phase(self, phase: CreditPhase) -> bool:
        phase_trackers = getattr(self._records_tracker, "_phase_trackers", {})
        if not isinstance(phase_trackers, dict):
            return False
        return any(
            tracker.total_records > 0
            for (tracker_phase, _), tracker in phase_trackers.items()
            if tracker_phase == phase
        )

    async def _summarize_warmup_metric_records(self) -> list[MetricResult] | None:
        """Return warmup-only inference metrics, or None when no warmup records exist."""
        if not self._has_records_for_phase(CreditPhase.WARMUP):
            return None

        (
            records_results,
            _,
            error_results,
            _summary_ctx,
        ) = await self._summarize_metric_record_accumulators(
            CreditPhase.WARMUP,
            self._records_tracker.was_phase_cancelled(CreditPhase.WARMUP),
        )
        if error_results:
            for error in error_results:
                self.error(f"Warmup metric summary error: {error}")

        warmup_context_overflow_count = (
            self._skipped_context_overflow_counts_by_phase.get(CreditPhase.WARMUP, 0)
        )
        if warmup_context_overflow_count:
            records_results.append(
                MetricResult(
                    tag="context_overflow_count",
                    header="Context Overflow Count",
                    unit="requests",
                    avg=float(warmup_context_overflow_count),
                    count=1,
                )
            )

        return records_results or None

    async def _finalize_stream_exporters(self) -> None:
        """Flush all stream exporters concurrently; log per-exporter errors.

        Without this flush the publish below races partial files — the
        controller could write the readiness marker while the JSONL/CSV files
        were still mid-flush.
        """
        if not self._stream_exporters:
            return
        results = await asyncio.gather(
            *[exporter.finalize() for exporter in self._stream_exporters.values()],
            return_exceptions=True,
        )
        for (exp_type, _), result in zip(
            self._stream_exporters.items(), results, strict=True
        ):
            if isinstance(result, BaseException):
                self.error(f"Stream exporter {exp_type} finalize failed: {result!r}")

    async def _publish_all_results(
        self,
        result: ProcessRecordsResult,
    ) -> None:
        """Publish ProcessAllResultsMessage for the SystemController fan-in."""
        try:
            await self.publish(
                ProcessAllResultsMessage(
                    service_id=self.service_id,
                    results=result,
                )
            )
        except Exception as e:  # noqa: BLE001 - publish failure must not abort the per-record result path
            self.error(f"Failed to publish ProcessAllResultsMessage: {e!r}")

    def _deliver_network_rtt_to_accumulators(self) -> None:
        """Set the run-level mean network RTT (ns) on each metric-record accumulator.

        Two cases, resolved here just before MetricsAccumulator.summarize():

        1. Manual mean (``--network-latency-mean``): if ``network_latency.mean_ms``
           is set, the NetworkLatencyManager service is never spawned; convert the
           mean ms to ns and deliver it directly.
        2. Automatic (``--network-latency-automatic``): the accumulator computed a
           mean over successful probe samples. If zero successful samples were
           collected, log a warning and apply no adjustment.

        A resolved RTT of 0 (or no RTT) is a no-op: the adjustment would emit
        network_adjusted_* metrics identical to the raw ones, so it is skipped.
        Also a no-op when network latency calibration is disabled entirely.
        """
        network_cfg = self.run.cfg.network_latency
        if not network_cfg.enabled:
            return

        if network_cfg.mean_ms is not None:
            rtt_ns: float | None = network_cfg.mean_ms * 1e6
        else:
            rtt_ns = (
                self._network_latency_accumulator.mean_rtt_ns
                if self._network_latency_accumulator is not None
                else None
            )
            if rtt_ns is None:
                self.warning(
                    "Network latency calibration enabled but no successful RTT "
                    "probes were collected; skipping network_adjusted_* metrics."
                )

        # A resolved RTT of 0/None is a no-op (adjusted == raw): skip injection so we
        # don't emit duplicate network_adjusted_* metrics. The None case already warned.
        if not rtt_ns:
            return

        if network_cfg.mean_ms is not None:
            self.notice(
                f"Network latency calibration: subtracting a fixed mean RTT of "
                f"{rtt_ns / 1e6:.3f} ms from latency metrics (network_adjusted_* metrics)."
            )
        else:
            sample_count = self._network_latency_accumulator.successful_sample_count
            self.notice(
                f"Network latency calibration: subtracting measured mean RTT of "
                f"{rtt_ns / 1e6:.3f} ms (over {sample_count} TCP-handshake probes) "
                "from latency metrics (network_adjusted_* metrics)."
            )

        # Deliver to the primary MetricsAccumulator engine, which injects
        # network_adjusted_* in its own summarize() from the columnar latency arrays.
        for target in self._metric_record_accumulators:
            set_rtt = getattr(target, "set_network_rtt_ns", None)
            if callable(set_rtt):
                set_rtt(rtt_ns)

    async def _process_results(
        self, phase: CreditPhase, cancelled: bool
    ) -> ProcessRecordsResult:
        """Process the accumulated records into final benchmark results.

        Single-flight: the natural finalize task and the PROCESS_RECORDS /
        PROFILE_CANCEL commands can race. The lock serializes them and the
        per-phase cache makes every call after the first return the same result
        instead of re-publishing and re-finalizing the stream exporters.
        """
        async with self._process_results_lock:
            cached = self._processed_results.get(phase)
            if cached is not None:
                self.debug(
                    lambda: (
                        f"Results for phase {phase} already processed; "
                        "returning cached result"
                    )
                )
                return cached
            result = await self._process_results_impl(phase, cancelled)
            self._processed_results[phase] = result
            return result

    async def _process_results_impl(
        self, phase: CreditPhase, cancelled: bool
    ) -> ProcessRecordsResult:
        """Process the accumulated records into final benchmark results."""
        self.debug(lambda: f"Processing records (cancelled: {cancelled})")
        self.info("Processing records results...")

        # Deliver the run-level mean network RTT before summarize() so
        # network_adjusted_* metrics can be injected.
        self._deliver_network_rtt_to_accumulators()

        (
            records_results,
            timeslices,
            error_results,
            summary_ctx,
        ) = await self._summarize_metric_record_accumulators(phase, cancelled)

        warmup_records_results = await self._summarize_warmup_metric_records()

        await self._finalize_stream_exporters()

        phase_stats = RecordsManager._create_result_stats_for_phase(self, phase)
        phase_records = await RecordsManager._build_phase_profile_results(
            self, phase, cancelled
        )
        metric_duration_coverage, fatal_errors = (
            self._validate_profile_metric_duration_coverage(phase, cancelled)
        )
        # Cross-accumulator analyzer plugins (e.g. energy efficiency) run after
        # all accumulators have summarized, reading peers via the SummaryContext.
        records_results.extend(await self._run_analyzers(summary_ctx))

        result = ProcessRecordsResult(
            results=ProfileResults(
                records=records_results,
                warmup_records=warmup_records_results,
                timeslices=timeslices or None,
                completed=phase_stats.total_records,
                start_ns=phase_stats.start_ns or time.time_ns(),
                end_ns=phase_stats.requests_end_ns or time.time_ns(),
                error_summary=self._error_tracker.get_error_summary_for_phase(phase),
                was_cancelled=cancelled,
                successful_request_count=phase_stats.success_records,
                error_request_count=phase_stats.error_records,
                branch_stats=self._latest_branch_stats
                if phase == CreditPhase.PROFILING
                else None,
                context_overflow_count=self._skipped_context_overflow_counts_by_phase.get(
                    phase, 0
                ),
                metric_duration_coverage=metric_duration_coverage,
                runtime_submission_invalid_reasons=(
                    ["insufficient_profile_metric_coverage"] if fatal_errors else []
                ),
                phase_records=phase_records,
                pooled_spec_decode_acceptance_histogram=_pooled_spec_decode_histogram(
                    summary_ctx
                ),
            ),
            errors=error_results,
            fatal_errors=fatal_errors,
        )
        self.debug(lambda: f"Process records result: {result}")
        self.debug("Publishing ProcessRecordsResultMessage...")
        await self.publish(
            ProcessRecordsResultMessage(
                service_id=self.service_id,
                results=result,
            )
        )
        self.debug("ProcessRecordsResultMessage published")

        if self.run.cfg.gpu_telemetry_disabled:
            self.debug("GPU telemetry collection is disabled, skipping publish")
        else:
            try:
                self.debug("Starting _publish_telemetry_results...")
                await self._publish_telemetry_results(phase)
                self.debug("_publish_telemetry_results completed")
            except Exception as e:
                self.exception(f"Failed to publish telemetry results: {e!r}")

        if self.run.cfg.server_metrics_disabled:
            self.debug("Server metrics collection is disabled, skipping publish")
        else:
            try:
                self.debug("Starting _publish_server_metrics_results...")
                await self._publish_server_metrics_results()
                self.debug("_publish_server_metrics_results completed")
            except Exception as e:
                self.exception(f"Failed to publish server metrics results: {e!r}")

        accuracy_enabled = (
            self.run.cfg.accuracy is not None and self.run.cfg.accuracy.enabled
        )
        if accuracy_enabled and phase == CreditPhase.PROFILING:
            try:
                await self._publish_accuracy_results(phase)
            except Exception as e:
                self.exception(f"Failed to publish accuracy results: {e!r}")
        else:
            self.debug("Accuracy publish skipped (disabled or non-profiling phase)")

        # Publish the unified ProcessAllResultsMessage over the populated
        # accumulators. The per-stream result messages above remain the
        # shutdown trigger; this is supplementary.
        await self._publish_all_results(result)

        self.debug("_process_results completed, returning result")
        return result

    async def _process_telemetry_results(self) -> ProcessTelemetryResult:
        """Process telemetry results by exporting the accumulated telemetry data.

        Returns:
            ProcessTelemetryResult: Contains TelemetryExportData with pre-computed GPU telemetry stats and any errors encountered
        """
        self.debug("Processing telemetry results...")

        error_summary = [
            ErrorDetailsCount(error_details=error_details, count=count)
            for error_details, count in self._telemetry_state.error_counts.items()
        ]

        if not self._gpu_telemetry_accumulator:
            self.debug(
                "GPU telemetry accumulator not found, cannot process telemetry results"
            )
            return ProcessTelemetryResult(
                results=None,
            )

        # Get timing from profiling phase stats. Bound the aggregate window while
        # preserving the trailing scrape that often closes GPU counter deltas.
        # If start_ns/end_ns is None (no profiling phase), include all data.
        phase_stats = self._records_tracker.create_aggregate_stats_for_phase(
            CreditPhase.PROFILING
        )
        profiling_end_ns = (
            phase_stats.requests_end_ns + Environment.GPU.FINAL_SCRAPE_GRACE_NS
            if phase_stats.requests_end_ns is not None
            else None
        )
        telemetry_export_data = await self._gpu_telemetry_accumulator.export_results(
            ExportContext(
                start_ns=phase_stats.start_ns,
                end_ns=profiling_end_ns,
                phase=CreditPhase.PROFILING,
                error_summary=error_summary,
            )
        )

        return ProcessTelemetryResult(
            results=telemetry_export_data,
        )

    async def _publish_telemetry_results(self, phase: CreditPhase) -> None:
        """Publish telemetry results independently from inference results.

        Processes and publishes telemetry data via ProcessTelemetryResultMessage.
        Called at the end of _process_results to keep telemetry separate from
        inference metrics in the results pipeline.
        """
        telemetry_result = await self._process_telemetry_results()
        await self.publish(
            ProcessTelemetryResultMessage(
                service_id=self.service_id,
                telemetry_result=telemetry_result,
            )
        )

    async def _publish_accuracy_results(self, phase: CreditPhase) -> None:
        """Publish phase-scoped accuracy results on the dedicated accuracy channel.

        Mirrors ``_publish_telemetry_results``: exports the phase-scoped accuracy
        summary from the AccuracyAccumulator and publishes it independently from
        inference results.

        Exactly-once contract: this method only runs when accuracy is enabled and
        the phase is PROFILING. It attempts to publish exactly one
        ``ProcessAccuracyResultMessage``; the SystemController clears
        ``_should_wait_for_accuracy`` only on receipt of that message. A summary
        that fails to export still publishes a terminal ``results=None`` message so
        the gate is released. The publish itself is the only unrecoverable point:
        if the message bus raises here the gate cannot be released from this side.
        We log it at error level and return rather than re-raising -- propagating
        would skip the subsequent ``ProcessAllResultsMessage`` publish, and the
        caller already logs it. (Note: only the CANCEL path has a bounded wait on
        this gate; on normal completion an unreleased gate stalls shutdown, so this
        error log is the primary diagnostic for that case.)
        """
        summary: AccuracySummary | None = None
        if self._accuracy_accumulator is not None:
            try:
                summary = await self._accuracy_accumulator.export_results(
                    ExportContext(phase=phase)
                )
            except Exception as e:  # noqa: BLE001 - must still publish a terminal message
                self.exception(f"Accuracy summary export failed: {e!r}")
                summary = None
        try:
            await self.publish(
                ProcessAccuracyResultMessage(
                    service_id=self.service_id,
                    accuracy_result=ProcessAccuracyResult(results=summary),
                )
            )
        except Exception as e:  # noqa: BLE001
            self.error(
                "Failed to publish ProcessAccuracyResultMessage; the controller's "
                f"accuracy shutdown gate may not release: {e!r}"
            )

    async def _process_server_metrics_results(self) -> ProcessServerMetricsResult:
        """Process server metrics results by exporting the accumulated server metrics data.

        Returns:
            ProcessServerMetricsResult: Contains ServerMetricsResults with server metrics data hierarchy and any errors encountered
        """
        self.debug("Processing server metrics results...")

        error_summary = [
            ErrorDetailsCount(error_details=error_details, count=count)
            for error_details, count in self._server_metrics_state.error_counts.items()
        ]

        if not self._server_metrics_accumulator:
            return ProcessServerMetricsResult(
                results=None,
                error_summary=error_summary,
            )

        # Get timing from profiling phase stats (warmup is automatically excluded)
        # TimeFilter will be constructed per-endpoint in accumulator with per-endpoint end times
        phase_stats = RecordsManager._create_result_stats_for_phase(
            self, CreditPhase.PROFILING
        )
        profiling_start_ns = phase_stats.start_ns or time.time_ns()
        profiling_end_ns = phase_stats.requests_end_ns or time.time_ns()
        warmup_phase_stats = RecordsManager._create_result_stats_for_phase(
            self, CreditPhase.WARMUP
        )

        server_metrics_export_data = (
            await self._server_metrics_accumulator.export_results(
                ExportContext(
                    start_ns=profiling_start_ns,
                    end_ns=profiling_end_ns,
                    error_summary=error_summary,
                    warmup_start_ns=warmup_phase_stats.start_ns,
                    warmup_end_ns=warmup_phase_stats.requests_end_ns,
                )
            )
        )

        return ProcessServerMetricsResult(
            results=server_metrics_export_data,
            error_summary=error_summary,
        )

    async def _publish_server_metrics_results(self) -> None:
        """Publish server metrics results independently from inference results.

        Processes and publishes server metrics data via ProcessServerMetricsResultMessage.
        Called at the end of _process_results to keep server metrics separate from
        inference metrics in the results pipeline.
        """
        self.debug(
            "_publish_server_metrics_results: calling _process_server_metrics_results..."
        )
        try:
            server_metrics_result = await self._process_server_metrics_results()
        except Exception as e:  # noqa: BLE001
            self.exception(f"Failed to process server metrics results: {e!r}")
            server_metrics_result = ProcessServerMetricsResult(results=None)
        self.debug(
            "_publish_server_metrics_results: publishing ProcessServerMetricsResultMessage..."
        )
        await self.publish(
            ProcessServerMetricsResultMessage(
                service_id=self.service_id,
                server_metrics_result=server_metrics_result,
            )
        )
        self.debug(
            "_publish_server_metrics_results: published ProcessServerMetricsResultMessage"
        )


def main() -> None:
    """Main entry point for the records manager."""

    from aiperf.common.bootstrap import bootstrap_and_run_service
    from aiperf.plugin.enums import ServiceType

    bootstrap_and_run_service(ServiceType.RECORDS_MANAGER)


if __name__ == "__main__":
    main()
