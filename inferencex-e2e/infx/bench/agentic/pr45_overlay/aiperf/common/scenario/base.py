# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field

from aiperf.common.enums import CacheBustTarget
from aiperf.common.models import AIPerfBaseModel
from aiperf.plugin.enums import TimingMode


class ScenarioSpec(AIPerfBaseModel):
    """Frozen declaration of a benchmark scenario's invariants."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    name: str = Field(description="Scenario identifier, e.g. 'inferencex-agentx-mvp'.")
    cli_defaults: dict[str, Any] = Field(
        default_factory=dict,
        description="Defaults for omitted CLI flags, applied before config conversion.",
    )
    environment_defaults: dict[str, dict[str, str | int | float | bool]] = Field(
        default_factory=dict,
        description="Runtime defaults grouped by Environment subsystem; explicit settings win.",
    )
    timing_mode: TimingMode = Field(
        description="Required timing mode for this scenario."
    )
    require_ignore_eos: bool = Field(
        description="Inject ignore_eos=true into extra_inputs; error on explicit false."
    )
    require_use_think_time_only: bool = Field(
        default=False,
        description="Force --use-think-time-only=true to exclude response time from inter-turn delays.",
    )
    require_streaming: bool = Field(
        default=False,
        description=(
            "Force --streaming=true (auto-enabled when unset; error on explicit "
            "--no-streaming). Streaming is required for the per-token latency "
            "metrics (TTFT, ITL) that are core to this benchmark; without it a "
            "run would silently report no first-token signal."
        ),
    )
    forbid_ignore_trace_delays: bool = Field(
        default=False,
        description=(
            "Reject --ignore-trace-delays. The scenario replays recorded trace "
            "timing; --ignore-trace-delays nulls every per-turn timestamp/delay "
            "in the loader, dispatching all turns back-to-back and falsifying the "
            "workload while the run would otherwise still report "
            "submission_valid=true."
        ),
    )
    forbid_input_truncation: bool = Field(
        description=(
            "Reject client-side input-length truncation. Currently checks "
            "`--synthesis-max-isl` (which drops traces whose input length "
            "exceeds the cap)."
        )
    )
    require_loader: str | tuple[str, ...] = Field(
        description=(
            "Required loader plugin name (e.g. 'weka_trace'), or a tuple of "
            "equivalent loader names. The detected loader must match any one "
            "of them — useful when several loader plugins produce byte-identical "
            "data (e.g. file-based vs HF-hosted variants)."
        )
    )
    min_benchmark_duration_seconds: int = Field(
        gt=0, description="Floor on --benchmark-duration in seconds."
    )
    default_benchmark_duration_seconds: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Value auto-filled into --benchmark-duration when the user leaves "
            "it unset. Explicit user values are honored (subject to the "
            "min_benchmark_duration_seconds floor). None disables auto-fill."
        ),
    )
    default_trajectory_start_min_ratio: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Value auto-filled into --trajectory-start-min-ratio when the user "
            "leaves it unset. Explicit user values are honored. None disables "
            "auto-fill."
        ),
    )
    default_trajectory_start_max_ratio: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Value auto-filled into --trajectory-start-max-ratio when the user "
            "leaves it unset. Explicit user values are honored. None disables "
            "auto-fill."
        ),
    )
    inter_turn_delay_cap_seconds: float | None = Field(
        default=None,
        ge=0,
        description="Hard ceiling for trace inter-turn delays in seconds. None disables.",
    )
    trace_idle_gap_cap_seconds: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Hard ceiling (seconds) for idle gaps within each root trace. For "
            "Weka, parent + subagent request-start timestamps are compressed "
            "per-trace before per-turn delays are derived. Takes precedence over "
            "inter_turn_delay_cap_seconds and supersedes use_think_time_only."
        ),
    )
    system_idle_gap_cap_seconds: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Hard ceiling in seconds for globally idle replay time. Pending "
            "request timers shift uniformly only when no request is active."
        ),
    )
    forbid_trace_idle_gap_cap: bool = Field(
        default=False,
        description=(
            "Reject trace_idle_gap_cap_seconds so the scenario preserves each "
            "trace's original request timeline."
        ),
    )
    forbid_inter_turn_delay_cap: bool = Field(
        default=False,
        description=(
            "Reject inter_turn_delay_cap_seconds so the scenario preserves "
            "each trace's original inter-turn timing."
        ),
    )
    require_cache_bust: CacheBustTarget | None = Field(
        default=None,
        description=(
            "When set, prompt.cache_bust.target must equal this value. "
            "Mismatch is rejected unless --unsafe-override is also set "
            "(which stamps submission_valid=false)."
        ),
    )
    minimum_profile_metric_coverage_ratio: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description=(
            "Minimum fraction of each duration-based profiling phase that must "
            "reach a TTFT, inter-token-latency, or parsed streaming content observation. A run that "
            "falls below the threshold exits non-zero and is not a valid scenario "
            "submission. None disables the post-run check."
        ),
    )


class ScenarioViolation(AIPerfBaseModel):
    """A single conflict between user config and a locked scenario invariant."""

    flag: str = Field(
        description="The user-facing flag or config field that conflicts."
    )
    current_value: Any = Field(description="The value the user provided.")
    required_value: Any = Field(description="The value the scenario requires.")
    message: str = Field(description="Human-readable explanation of the conflict.")

    def __str__(self) -> str:
        return (
            f"{self.flag}: got {self.current_value!r}, "
            f"required {self.required_value!r} ({self.message})"
        )


class ScenarioOutcome(AIPerfBaseModel):
    """Result of applying a scenario lock against a resolved ``BenchmarkRun``.

    Produced by ``aiperf.common.scenario.apply_scenario`` and stored on
    ``run.resolved.scenario_outcome``. ``submission_valid`` is ``None`` when no
    scenario is set, ``True`` when all invariants are satisfied (after
    auto-fills), and ``False`` under ``--unsafe-override`` when violations were
    downgraded to warnings.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    scenario_name: str | None = Field(
        default=None,
        description="The applied scenario name, or None when no --scenario was set.",
    )
    applied_locks: list[str] = Field(
        default_factory=list,
        description="Short tags for each invariant lock that was applied "
        "(auto-filled or validated), e.g. 'timing_mode', 'streaming', "
        "'cache_bust'. Order reflects application order.",
    )
    violations: list[ScenarioViolation] = Field(
        default_factory=list,
        description="All scenario invariant conflicts collected in one pass. "
        "Non-empty only under --unsafe-override (otherwise a ScenarioLockError "
        "is raised).",
    )
    submission_valid: bool | None = Field(
        default=None,
        description="True when the scenario lock is satisfied, False under "
        "--unsafe-override with violations, None when no scenario is set.",
    )
    submission_invalid_reasons: list[str] = Field(
        default_factory=list,
        description="Short tags explaining why submission_valid is False "
        "(e.g. 'unsafe_override').",
    )


class ScenarioLockError(ValueError):
    """Raised when a scenario lock is violated and cannot proceed.

    By default the message suggests ``--unsafe-override``. Pass
    ``bypassable=False`` when the violations include hard locks that the
    override cannot downgrade (e.g. AgentX on a synthetic default dataset).
    """

    def __init__(
        self,
        violations: list[ScenarioViolation],
        *,
        bypassable: bool = True,
    ) -> None:
        self.violations = violations
        self.bypassable = bypassable
        joined = "\n  - ".join(str(v) for v in violations)
        if bypassable:
            hint = (
                "Pass --unsafe-override to convert to warnings "
                "(run will be marked submission_valid=false)."
            )
        else:
            hint = (
                "These violations cannot be bypassed with --unsafe-override. "
                "Provide --public-dataset (e.g. semianalysis_cc_traces_weka_062126) "
                "or --hf-weka-dataset semianalysisai/cc-traces-weka-062126."
            )
        super().__init__(
            f"Scenario invariants violated ({len(violations)} conflict"
            f"{'s' if len(violations) != 1 else ''}):\n  - {joined}\n"
            f"{hint}"
        )


class EmptyTracePoolError(RuntimeError):
    """Raised when the loader produces 0 valid traces and the scenario requires a non-empty pool."""


class TrajectoryWarmupFailedError(RuntimeError):
    """Raised when WARMUP has terminal failures across trajectories and PROFILING cannot honestly start."""

    def __init__(self, failed_trace_ids: list[str]) -> None:
        self.failed_trace_ids = failed_trace_ids
        super().__init__(
            f"Trajectory warmup failed for {len(failed_trace_ids)} trace(s): "
            f"{', '.join(failed_trace_ids)}. Run aborted to preserve metrics integrity."
        )


class UnknownScenarioError(ValueError):
    """Raised when --scenario references a name not in the registry."""
