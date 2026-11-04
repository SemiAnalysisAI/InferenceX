"""DCGM power eligibility of the srt-slurm multi-node lanes.

A matrix point asks for DCGM power with its master ``power`` field; each lane lists the
requests that may have it. A ``default_on`` lane measures GPU and Grace CPU power on every
benchmark point whose recipe keeps etcd and NATS off a dedicated node, and a point without
``power`` keeps its result when telemetry fails.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from infx.launch.context import LaunchError
from infx.launch.policy import LaunchPath, Match, any_of

if TYPE_CHECKING:
    from infx.launch.request import MultiNodeRequest


class PowerPolicyError(LaunchError):
    """The point asks for DCGM power on a lane that does not support it."""


@dataclass(frozen=True)
class PowerRule:
    """One allowed power combination; ``require`` makes the client fail invalid AgentX power."""

    when: Match
    agentx: bool
    adapter: bool = False
    require: bool = False


@dataclass(frozen=True)
class PowerLane:
    """Ordered power rules for one lane; first match wins."""

    rules: tuple[PowerRule, ...]
    error: str = ""
    agentic_error: str | None = None
    default_on: bool = False


POWER_LANES: dict[tuple[str, LaunchPath], PowerLane] = {
    ("gb200-nv", LaunchPath.SRT_MULTI): PowerLane(rules=(), default_on=True),
    ("gb300-nv", LaunchPath.SRT_MULTI): PowerLane(rules=(), default_on=True),
    ("b200-nscale", LaunchPath.SRT_NATIVE): PowerLane(
        rules=(
            PowerRule(
                Match(any_of("glm5.2"), any_of("fp4"), any_of("dynamo-sglang"), agentic=True),
                agentx=True,
            ),
        ),
        error="B200 nscale dcgm-power requires GLM-5.2 FP4 AgentX dynamo-sglang",
    ),
    ("b200-nscale", LaunchPath.SRT_MULTI): PowerLane(
        rules=(
            PowerRule(
                Match(any_of("qwen3.5"), any_of("fp8"), any_of("dynamo-sglang"), agentic=True),
                agentx=True,
            ),
        ),
        error="B200 Nscale dcgm-power requires Qwen3.5 FP8 AgentX dynamo-sglang",
    ),
    ("h200-dgxc", LaunchPath.SRT_MULTI): PowerLane(
        rules=(
            PowerRule(
                Match(any_of("glm5.2"), any_of("fp8"), any_of("dynamo-sglang"), agentic=True),
                agentx=False,
                adapter=True,
            ),
        ),
        error="H200 dcgm-power requires AgentX dynamo-sglang glm5.2 FP8",
    ),
}


@dataclass(frozen=True)
class PowerDecision:
    """Outcome of power eligibility for one launch."""

    dcgm: bool
    agentx: bool
    adapter: bool = False
    require: bool = False
    # The cluster's Grace CPU exporter is bound.
    cpu: bool = False
    # The point selected cpu-power, so a required point fails without a valid CPU leg.
    expect_cpu: bool = False
    # Invalid AgentX power leaves the benchmark result standing.
    best_effort: bool = False


NO_POWER = PowerDecision(dcgm=False, agentx=False)


def decide_power(
    cluster_id: str,
    path: LaunchPath,
    request: MultiNodeRequest,
    dedicated_discovery: Callable[[], bool] = lambda: False,
) -> PowerDecision:
    """Apply the lane's rules when the point's ``power`` field is on, or every benchmark point of
    a ``default_on`` lane except one whose recipe runs ``dedicated_discovery``.

    Raises ``PowerPolicyError`` for a combination the lane does not allow.
    """
    if request.al_gate:
        return NO_POWER
    lane = POWER_LANES.get((cluster_id, path))
    if lane is not None and lane.default_on:
        if request.eval_only or (not request.power and dedicated_discovery()):
            return NO_POWER
        required = request.power and any(r.require and r.when(request) for r in lane.rules)
        return PowerDecision(
            dcgm=True,
            agentx=request.is_agentic,
            require=required,
            cpu=True,
            expect_cpu=request.cpu_power,
            best_effort=not (request.power or request.require_power),
        )
    if not request.power:
        return NO_POWER
    if lane is None:
        raise PowerPolicyError(f"cluster {cluster_id!r} measures no DCGM power on {path}")
    for rule in lane.rules:
        if rule.when(request):
            return PowerDecision(
                dcgm=True,
                agentx=rule.agentx,
                adapter=rule.adapter,
                require=rule.require,
                cpu=request.cpu_power,
                expect_cpu=request.cpu_power,
            )
    message = lane.agentic_error if request.is_agentic and lane.agentic_error else lane.error
    raise PowerPolicyError(message)
