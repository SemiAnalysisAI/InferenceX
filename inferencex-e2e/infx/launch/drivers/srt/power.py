"""DCGM power eligibility of the srt-slurm multi-node lanes.

A recipe asks for DCGM power in its top-level ``telemetry:`` mapping; each lane lists the
requests and recipes that may have it.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import TYPE_CHECKING

import yaml

from infx.launch.context import LaunchError
from infx.launch.drivers.srt.recipe import recipe_mirror_path, recipe_relpath
from infx.launch.policy import LaunchPath, Match, any_of

if TYPE_CHECKING:
    from infx.launch.request import LaunchRequest


def recipe_enables_dcgm_power(text: str) -> bool:
    """Whether the recipe's top-level ``telemetry`` is ``enabled: true`` with a ``dcgm_exporter``."""
    try:
        recipe = yaml.safe_load(text)
    except yaml.YAMLError:
        return False
    telemetry = recipe.get("telemetry") if isinstance(recipe, dict) else None
    return (
        isinstance(telemetry, dict)
        and "dcgm_exporter" in telemetry
        and telemetry.get("enabled") is True
    )


class PowerPolicyError(LaunchError):
    """The recipe enables DCGM power on a lane that does not support it."""


@dataclass(frozen=True)
class PowerRule:
    """One allowed power combination."""

    when: Match
    agentx: bool
    recipe_glob: str | None = None
    adapter: bool = False


@dataclass(frozen=True)
class PowerLane:
    """Ordered power rules for one lane; first match wins."""

    rules: tuple[PowerRule, ...]
    error: str
    agentic_error: str | None = None
    eval_recipe_when_eval_only: bool = False


POWER_LANES: dict[tuple[str, LaunchPath], PowerLane] = {
    ("gb200-nv", LaunchPath.SRT_MULTI): PowerLane(
        rules=(
            PowerRule(
                Match(any_of("glm5.2"), any_of("fp4"), any_of("dynamo-sglang"), agentic=True),
                agentx=True,
                recipe_glob="recipes/glm5.2/sglang/gb200-fp4/agentx/agg.yaml",
            ),
            PowerRule(
                Match(any_of("kimik3"), any_of("fp4"), any_of("dynamo-vllm"), agentic=True),
                agentx=True,
                recipe_glob="recipes/kimik3/vllm/gb200-fp4/agentx/*",
            ),
            PowerRule(Match(frameworks=any_of("dynamo-sglang"), agentic=False), agentx=False),
        ),
        error="dcgm-power requires dynamo-sglang or the supported Kimi-K3 AgentX route",
        agentic_error="AgentX dcgm-power requires the GLM-5.2 aggregate or supported Kimi-K3 recipe",
    ),
    ("gb300-nv", LaunchPath.SRT_MULTI): PowerLane(
        rules=(
            PowerRule(
                Match(any_of("kimik3"), any_of("fp4"), any_of("dynamo-vllm"), agentic=True),
                agentx=True,
                recipe_glob="recipes/kimik3/vllm/*/agentx/*",
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
        eval_recipe_when_eval_only=True,
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
        eval_recipe_when_eval_only=True,
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


def decide_power(
    cluster_id: str, path: LaunchPath, *, dcgm: bool, request: LaunchRequest, recipe: str
) -> PowerDecision:
    """Apply the lane's rules to the inspected ``recipe`` (``recipes/...``) when it enables dcgm.

    Raises ``PowerPolicyError`` for a combination the lane does not allow.
    """
    lane = POWER_LANES.get((cluster_id, path))
    if not dcgm or lane is None:
        return NO_POWER
    for rule in lane.rules:
        if rule.when(request) and (
            rule.recipe_glob is None or fnmatch.fnmatchcase(recipe, rule.recipe_glob)
        ):
            return PowerDecision(dcgm=True, agentx=rule.agentx, adapter=rule.adapter)
    message = lane.agentic_error if request.is_agentic and lane.agentic_error else lane.error
    raise PowerPolicyError(message)


def resolve_power(cluster_id: str, path: LaunchPath, request: LaunchRequest) -> PowerDecision:
    """Detect dcgm in the workspace mirror of the recipe the lane inspects, and decide.

    A recipe only upstream (no mirror) stays non-power.
    """
    lane = POWER_LANES.get((cluster_id, path))
    if lane is None:
        return NO_POWER
    config_file = request.config_file
    if lane.eval_recipe_when_eval_only and request.eval_only and request.eval_config_file:
        config_file = request.eval_config_file
    if not config_file:
        return NO_POWER
    mirror = recipe_mirror_path(request.workspace, config_file)
    dcgm = mirror.is_file() and recipe_enables_dcgm_power(mirror.read_text())
    return decide_power(
        cluster_id, path, dcgm=dcgm, request=request, recipe=recipe_relpath(config_file)
    )
