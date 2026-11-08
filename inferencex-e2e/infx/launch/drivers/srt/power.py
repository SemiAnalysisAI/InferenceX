"""Power eligibility of the srt-slurm multi-node lanes.

Every multi-node benchmark point on a cluster with a GPU exporter measures GPU power, and CPU
power where the cluster declares a CPU exporter, unless its recipe runs etcd and NATS on a
dedicated node. A point without ``power`` keeps its result when telemetry fails. A lane rule
the point matches changes how its AgentX power is measured.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from infx.launch.context import LaunchError
from infx.launch.policy import LaunchPath, Match, any_of

if TYPE_CHECKING:
    from infx.clusters.slurm import SrtSlurmSettings
    from infx.launch.request import MultiNodeRequest


class PowerPolicyError(LaunchError):
    """The point asks for power its cluster cannot measure."""


@dataclass(frozen=True)
class PowerRule:
    """How matching points measure AgentX power; ``require`` makes the client fail invalid
    AgentX power.
    """

    when: Match
    agentx: bool
    adapter: bool = False
    require: bool = False


@dataclass(frozen=True)
class PowerLane:
    """Ordered power rules for one lane; first match wins."""

    rules: tuple[PowerRule, ...]


POWER_LANES: dict[tuple[str, LaunchPath], PowerLane] = {
    ("h200-dgxc", LaunchPath.SRT_MULTI): PowerLane(
        rules=(
            PowerRule(
                Match(any_of("glm5.2"), any_of("fp8"), any_of("dynamo-sglang"), agentic=True),
                agentx=False,
                adapter=True,
            ),
        ),
    ),
}


@dataclass(frozen=True)
class PowerDecision:
    """Outcome of power eligibility for one launch."""

    # GPU power is measured: DCGM, or the cluster's own exporter where it sets no DCGM port.
    # The name is the job-event schema's (``event.Power.dcgm``).
    dcgm: bool
    agentx: bool
    adapter: bool = False
    require: bool = False
    # The cluster's CPU exporter is bound.
    cpu: bool = False
    # The point selected cpu-power, so a required point fails without a valid CPU leg.
    expect_cpu: bool = False
    # Invalid AgentX power leaves the benchmark result standing.
    best_effort: bool = False


NO_POWER = PowerDecision(dcgm=False, agentx=False)


def decide_power(
    cluster_id: str,
    path: LaunchPath,
    srt: SrtSlurmSettings,
    request: MultiNodeRequest,
    dedicated_discovery: Callable[[], bool] = lambda: False,
) -> PowerDecision:
    """Measure power on every benchmark point of a cluster with a GPU exporter, except one
    without ``power`` whose recipe runs ``dedicated_discovery``.

    Raises ``PowerPolicyError`` when the point asks for power the cluster cannot measure.
    """
    if request.eval_only:
        return NO_POWER
    if srt.gpu_exporter_image is None:
        if request.power:
            raise PowerPolicyError(f"cluster {cluster_id!r} measures no GPU power on {path}")
        return NO_POWER
    if request.cpu_power and srt.cpu_power_exporter is None:
        raise PowerPolicyError(
            f"cluster {cluster_id!r} declares no srt-slurm.cpu-power-exporter for cpu-power"
        )
    if not request.power and dedicated_discovery():
        return NO_POWER
    lane = POWER_LANES.get((cluster_id, path), PowerLane(rules=()))
    rule = next((rule for rule in lane.rules if rule.when(request)), None)
    return PowerDecision(
        dcgm=True,
        agentx=rule.agentx if rule else request.is_agentic,
        adapter=rule is not None and rule.adapter,
        require=request.power and rule is not None and rule.require,
        cpu=srt.cpu_power_exporter is not None,
        expect_cpu=request.cpu_power,
        best_effort=not (request.power or request.require_power),
    )
