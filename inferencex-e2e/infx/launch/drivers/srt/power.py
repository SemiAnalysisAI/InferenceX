"""DCGM power eligibility of the srt-slurm multi-node lanes.

A recipe runs with DCGM power when its top-level ``telemetry:`` mapping is enabled and
configures the dcgm exporter. Each (cluster, lane) lists the requests and recipes that
may, and which of those are AgentX power lanes, whose job failure is deferred until the
power audit is staged.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from infx.launch.drivers.srt.recipe import recipe_mirror_path, recipe_relpath
from infx.launch.policy import LaunchPath, Match, any_of

if TYPE_CHECKING:
    from infx.launch.request import LaunchRequest


def recipe_enables_dcgm_power(text: str) -> bool:
    """Return True iff the recipe's top-level ``telemetry`` enables a ``dcgm_exporter``.

    That mapping must hold ``enabled: true`` and a ``dcgm_exporter`` key; ``enabled``
    under another key or under the exporter itself does not count. Unparseable YAML
    enables nothing.
    """
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


def uses_dcgm_power(workspace: Path, config_file: str | None) -> bool:
    """Return True iff ``config_file`` is set and its workspace mirror enables dcgm.

    Recipes that exist only upstream (no mirror) stay non-power.
    """
    if not config_file:
        return False
    path = recipe_mirror_path(workspace, config_file)
    return path.is_file() and recipe_enables_dcgm_power(path.read_text())


class PowerPolicyError(ValueError):
    """The recipe enables DCGM power on a lane that does not support it."""


@dataclass(frozen=True)
class PowerRule:
    """One allowed power combination: the requests ``when`` matches whose inspected
    recipe path (``recipes/...``) matches ``recipe_glob`` (fnmatch; ``*`` also matches
    ``/``; None: any).

    ``agentx`` marks an AgentX power lane. ``adapter`` marks a DCGM AgentX lane that is
    not one: each concurrency is validated by the power adapter instead.
    """

    when: Match
    agentx: bool
    recipe_glob: str | None = None
    adapter: bool = False


@dataclass(frozen=True)
class PowerLane:
    """Ordered power rules for one lane; first match wins."""

    rules: tuple[PowerRule, ...]
    error: str
    agentic_error: str | None = None  # reported instead for agentic misses
    eval_recipe_when_eval_only: bool = False  # inspect EVAL_CONFIG_FILE on eval-only runs


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
    # The Kimi-K3 vLLM rule is the AgentX power lane.
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

    dcgm: bool  # the recipe enables DCGM power on a lane that allows it
    agentx: bool  # an AgentX power lane
    adapter: bool = False  # per-concurrency power adapter (see PowerRule.adapter)


NO_POWER = PowerDecision(dcgm=False, agentx=False)


def power_config_file(lane: PowerLane, request: LaunchRequest) -> str | None:
    """Return the recipe the lane inspects (EVAL_CONFIG_FILE on eval-only where it says so)."""
    if lane.eval_recipe_when_eval_only and request.eval_only and request.eval_config_file:
        return request.eval_config_file
    return request.config_file


def decide_power(
    cluster_id: str, path: LaunchPath, *, dcgm: bool, request: LaunchRequest, recipe: str
) -> PowerDecision:
    """Apply ``POWER_LANES[(cluster_id, path)]`` to a detected dcgm recipe.

    ``recipe`` is ``recipe_relpath`` of the inspected config file. Lanes with
    no table entry never run power. Raises ``PowerPolicyError`` when dcgm is
    enabled on an unsupported combination.
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
    """Detect dcgm from the workspace recipe mirror and apply the lane's policy."""
    lane = POWER_LANES.get((cluster_id, path))
    if lane is None:
        return NO_POWER
    config_file = power_config_file(lane, request)
    dcgm = uses_dcgm_power(request.workspace, config_file)
    return decide_power(
        cluster_id,
        path,
        dcgm=dcgm,
        request=request,
        recipe=recipe_relpath(config_file) if config_file else "",
    )
