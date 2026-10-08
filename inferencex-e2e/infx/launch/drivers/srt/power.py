"""DCGM power eligibility of the srt-slurm multi-node lanes.

A matrix point asks for DCGM power with its master ``power`` field; each lane lists the
requests that may have it.
"""

from __future__ import annotations

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
    """One allowed power combination."""

    when: Match
    agentx: bool
    adapter: bool = False


@dataclass(frozen=True)
class PowerLane:
    """Ordered power rules for one lane; first match wins."""

    rules: tuple[PowerRule, ...]
    error: str
    agentic_error: str | None = None


POWER_LANES: dict[tuple[str, LaunchPath], PowerLane] = {
    ("gb200-nv", LaunchPath.SRT_MULTI): PowerLane(
        rules=(
            PowerRule(
                Match(any_of("glm5.2"), any_of("fp4"), any_of("dynamo-sglang"), agentic=True),
                agentx=True,
            ),
            PowerRule(
                Match(any_of("kimik3"), any_of("fp4"), any_of("dynamo-vllm"), agentic=True),
                agentx=True,
            ),
            PowerRule(Match(frameworks=any_of("dynamo-sglang"), agentic=False), agentx=False),
        ),
        error="dcgm-power requires dynamo-sglang or the supported Kimi-K3 AgentX route",
        agentic_error="AgentX dcgm-power requires GLM-5.2 FP4 dynamo-sglang or Kimi-K3 FP4 dynamo-vllm",
    ),
    ("gb300-nv", LaunchPath.SRT_MULTI): PowerLane(
        rules=(
            PowerRule(
                Match(any_of("kimik3"), any_of("fp4"), any_of("dynamo-vllm"), agentic=True),
                agentx=True,
            ),
            PowerRule(Match(frameworks=any_of("dynamo-sglang")), agentx=False),
        ),
        error="dcgm-power requires dynamo-sglang or the supported Kimi-K3 AgentX route",
    ),
    ("b300-dsxe", LaunchPath.SRT_MULTI): PowerLane(
        rules=(
            PowerRule(
                Match(
                    any_of("dsv4"),
                    any_of("fp4"),
                    any_of("dynamo-sglang", "dynamo-vllm"),
                    agentic=False,
                ),
                agentx=False,
            ),
        ),
        error="B300 dcgm-power is limited to fixed-sequence DSV4 FP4 dynamo-sglang/vllm",
    ),
    ("b200-nscale", LaunchPath.SRT_NATIVE): PowerLane(
        rules=(
            PowerRule(
                Match(any_of("kimik3"), any_of("fp4"), any_of("dynamo-vllm"), agentic=True),
                agentx=True,
            ),
            PowerRule(
                Match(
                    any_of("dsv4"),
                    any_of("fp4"),
                    any_of("dynamo-sglang", "dynamo-vllm"),
                    agentic=False,
                ),
                agentx=False,
            ),
        ),
        error="B200 nscale dcgm-power requires a supported fixed-sequence lane or Kimi-K3 AgentX vLLM",
    ),
    ("b200-nscale", LaunchPath.SRT_MULTI): PowerLane(
        rules=(
            PowerRule(
                Match(any_of("qwen3.5"), any_of("fp8"), any_of("dynamo-sglang"), agentic=True),
                agentx=True,
            ),
            PowerRule(
                Match(any_of("dsv4"), any_of("fp4"), any_of("dynamo-vllm"), agentic=False),
                agentx=False,
            ),
        ),
        error="B200 Nscale dcgm-power requires fixed-sequence DSV4 FP4 dynamo-vllm or Qwen3.5 FP8 AgentX dynamo-sglang",
    ),
    ("h200-dgxc", LaunchPath.SRT_MULTI): PowerLane(
        rules=(
            PowerRule(
                Match(any_of("kimik3"), any_of("fp4"), any_of("vllm"), agentic=True), agentx=True
            ),
            PowerRule(
                Match(
                    any_of("glm5.2", "dsv4"), any_of("fp8"), any_of("dynamo-sglang"), agentic=True
                ),
                agentx=False,
                adapter=True,
            ),
        ),
        error="H200 dcgm-power requires AgentX dynamo-sglang glm5.2/dsv4 FP8 or Kimi-K3 vLLM FP4",
    ),
}


@dataclass(frozen=True)
class PowerDecision:
    """Outcome of power eligibility for one launch."""

    dcgm: bool
    agentx: bool
    adapter: bool = False


NO_POWER = PowerDecision(dcgm=False, agentx=False)


def decide_power(cluster_id: str, path: LaunchPath, request: MultiNodeRequest) -> PowerDecision:
    """Apply the lane's rules when the point's ``power`` field is on.

    Raises ``PowerPolicyError`` for a combination the lane does not allow.
    """
    if not request.power:
        return NO_POWER
    lane = POWER_LANES.get((cluster_id, path))
    if lane is None:
        raise PowerPolicyError(f"cluster {cluster_id!r} measures no DCGM power on {path}")
    for rule in lane.rules:
        if rule.when(request):
            return PowerDecision(dcgm=True, agentx=rule.agentx, adapter=rule.adapter)
    message = lane.agentic_error if request.is_agentic and lane.agentic_error else lane.error
    raise PowerPolicyError(message)
