"""Cluster power defaults and legacy recipe opt-ins for srt-slurm launches."""

from __future__ import annotations

import fnmatch
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import yaml

from infx.launch.context import LaunchError
from infx.launch.drivers.srt.recipe import recipe_mirror_path, recipe_relpath
from infx.launch.policy import LaunchPath, Match, any_of

if TYPE_CHECKING:
    from infx.clusters.slurm import PowerTelemetrySettings
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
    expected_cpu_source: str | None = None
    require_power: bool = False


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
            return PowerDecision(
                dcgm=True,
                agentx=rule.agentx,
                adapter=rule.adapter,
                require_power=request.require_power or rule.agentx,
            )
    message = lane.agentic_error if request.is_agentic and lane.agentic_error else lane.error
    raise PowerPolicyError(message)


def resolve_power(
    cluster_id: str,
    path: LaunchPath,
    request: LaunchRequest,
    *,
    settings: PowerTelemetrySettings | None = None,
) -> PowerDecision:
    """Resolve recipe power policy, then apply cluster defaults while preserving strictness."""
    lane = POWER_LANES.get((cluster_id, path))
    if settings is not None and request.eval_only:
        return NO_POWER
    if lane is None and settings is None:
        return NO_POWER
    config_file = request.config_file
    if lane and lane.eval_recipe_when_eval_only and request.eval_only and request.eval_config_file:
        config_file = request.eval_config_file
    mirror = recipe_mirror_path(request.workspace, config_file) if config_file else None
    text = mirror.read_text() if mirror and mirror.is_file() else ""
    dcgm = recipe_enables_dcgm_power(text)
    try:
        decision = decide_power(
            cluster_id,
            path,
            dcgm=dcgm,
            request=request,
            recipe=recipe_relpath(config_file) if config_file else "",
        )
    except PowerPolicyError:
        if settings is None:
            raise
        decision = NO_POWER
    if settings is not None:
        return PowerDecision(
            dcgm=True,
            agentx=request.is_agentic,
            expected_cpu_source=settings.cpu_source,
            require_power=decision.require_power or request.require_power,
        )
    if decision.dcgm:
        telemetry = yaml.safe_load(text)["telemetry"]
        cpu = telemetry.get("cpu_power_exporter")
        source = cpu.get("source") if isinstance(cpu, dict) else None
        if source in {"acpi", "dcgm"}:
            decision = replace(decision, expected_cpu_source=source)
    return decision


def telemetry_arguments(
    settings: PowerTelemetrySettings, concurrencies: Sequence[int]
) -> list[str]:
    """Bind cluster sensors to the selected native recipe, after variant expansion."""
    if not concurrencies or any(value <= 0 for value in concurrencies):
        raise PowerPolicyError("power telemetry requires positive benchmark concurrencies")
    values = {
        "telemetry.enabled": True,
        "telemetry.collect_interval_ms": 1000,
        "telemetry.storage_subdir": "power",
        "telemetry.startup_timeout_seconds": 120,
        "telemetry.request_timeout_seconds": 2,
        "telemetry.dcgm_exporter.container_image": "dcgm-exporter",
        "telemetry.dcgm_exporter.port": settings.dcgm_port,
        "telemetry.dcgm_exporter.command": (
            "dcgm-exporter --collect-interval=100 --address :{port} "
            "-f /configs/dcgm-counters-noprof.csv"
        ),
        "telemetry.cpu_power_exporter.port": settings.cpu_port,
        "telemetry.cpu_power_exporter.source": settings.cpu_source,
        "benchmark.concurrencies": list(concurrencies),
    }
    return [arg for key, value in values.items() for arg in ("--set", f"{key}={json.dumps(value)}")]
