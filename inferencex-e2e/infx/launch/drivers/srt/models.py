"""Which checkpoint a job serves.

MODEL resolves to its ``models.entries`` record, keyed by its basename or
``<basename>@<root>``, a node-local copy first; ``OVERRIDES`` holds the exceptions. An
srt-slurm job serves that checkpoint unless the point's additional-settings name a
MODEL_PATH, and every ``model.path`` alias of its recipe (anything but an ``hf:`` id or
absolute path) maps to what it serves.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from infx.clusters.slurm import slurm_settings
from infx.launch.context import LaunchError
from infx.launch.drivers.srt.config import volume_path
from infx.launch.drivers.srt.recipe import recipe_mirror_path
from infx.launch.policy import Match, any_of, point_settings

if TYPE_CHECKING:
    from infx.clusters import Cluster
    from infx.launch.request import LaunchRequest


@dataclass(frozen=True)
class Override:
    """What differs for the requests ``when`` matches; per field, the first such row wins."""

    when: Match
    entry: str | None = None
    served_name: str | None = None
    require_config: bool = False


OVERRIDES: dict[str, tuple[Override, ...]] = {
    "b300-dsxe": (
        Override(
            Match(frameworks=any_of("vllm"), model_glob="*/DeepSeek-V4-Pro-0813"),
            entry="DeepSeek-V4-Pro-0813@scratch",
        ),
        Override(Match(model_glob="*/DeepSeek-V4-Pro-0813"), entry="DeepSeek-V4-Pro-0813"),
    ),
    "gb200-nv": (
        Override(
            Match(any_of("dsv4"), any_of("fp4"), any_of("llmd-vllm")),
            entry="DeepSeek-V4-Pro@numa1",
            served_name="deepseek-ai/DeepSeek-V4-Pro",
        ),
        Override(
            Match(any_of("dsr1"), any_of("fp4"), any_of("dynamo-sglang")),
            entry="deepseek-r1-0528-fp4-v2",
        ),
        Override(
            Match(any_of("dsr1"), any_of("fp8"), any_of("dynamo-sglang")), entry="deepseek-r1-0528"
        ),
        Override(Match(any_of("dsv4"), frameworks=any_of("dynamo-vllm")), entry="DeepSeek-V4-Pro"),
        Override(
            Match(any_of("dsr1"), any_of("fp4"), any_of("dynamo-trt")),
            served_name="deepseek-r1-fp4",
        ),
        Override(
            Match(any_of("dsr1"), any_of("fp8"), any_of("dynamo-trt")),
            served_name="deepseek-r1-fp8",
        ),
    ),
    "gb300-nv": (
        Override(Match(any_of("dsr1"), any_of("fp4")), served_name="deepseek-r1-fp4"),
        Override(Match(any_of("dsr1"), any_of("fp8")), served_name="deepseek-r1-fp8"),
        Override(
            Match(any_of("glm5.2"), any_of("fp4"), any_of("dynamo-trt")),
            served_name="GLM-5.2-NVFP4",
        ),
    ),
    "h100-dgxc": (
        Override(Match(), require_config=True),
        Override(
            Match(any_of("dsr1"), any_of("fp8"), any_of("dynamo-trt")),
            served_name="DeepSeek-R1-0528",
        ),
    ),
    "h200-dgxc": (
        Override(Match(any_of("dsv4")), require_config=True),
        Override(
            Match(any_of("dsr1"), any_of("fp8"), any_of("dynamo-trt")),
            served_name="DeepSeek-R1-0528",
        ),
    ),
}


def _override(cluster: Cluster, request: LaunchRequest, field: str) -> str | bool | None:
    """The first matching override's value of ``field``, if any row sets it."""
    for row in OVERRIDES.get(cluster.id, ()):
        if getattr(row, field) and row.when(request):
            return getattr(row, field)
    return None


@dataclass(frozen=True)
class Checkpoint:
    """Checkpoint directory ``dir`` of volume ``volume``, on any scheduler."""

    volume: str
    dir: str
    node_local: bool


def checkpoint(cluster: Cluster, request: LaunchRequest) -> Checkpoint | None:
    """MODEL's staged checkpoint on ``cluster``, or None when the cluster stages none.

    None too when the point's additional-settings name a MODEL_PATH of its own. Raises
    ``LaunchError`` when the checkpoint must be readable here and is not.
    """
    if "MODEL_PATH" in point_settings(request):
        return None
    volumes = cluster.scheduler_settings.volumes
    entries = cluster.models.entries

    def node_local(key: str) -> bool:
        return volumes[entries[key].root].visibility == "node-local"

    key = _override(cluster, request, "entry")
    if key is None:
        basename = (request.model or "").rsplit("/", 1)[-1]
        copies = [name for name in entries if name.partition("@")[0] == basename]
        key = min(copies, key=lambda name: not node_local(name), default=None)
    if key is None:
        return None
    entry = entries[str(key)]
    model = Checkpoint(entry.root, entry.dir, node_local(str(key)))
    if _override(cluster, request, "require_config"):
        config = host_path(cluster, model) / "config.json"
        if not os.access(config, os.R_OK):
            raise LaunchError(f"model checkpoint is unavailable: no readable {config}")
    return model


def host_path(cluster: Cluster, model: Checkpoint) -> Path:
    """Where a Slurm cluster's hosts and jobs see ``model``."""
    return volume_path(cluster, model.volume) / model.dir


def served_path(cluster: Cluster, request: LaunchRequest, model: Checkpoint | None) -> str | None:
    """What an srt-slurm job serves: ``model``, else the point's own MODEL_PATH setting.

    A point with a MODEL_PATH setting has no ``model``: ``checkpoint`` stages none for it.
    """
    if model is not None:
        return str(host_path(cluster, model))
    if "MODEL_PATH" in point_settings(request):
        return request.env.get("MODEL_PATH") or None
    return None


def recipe_aliases(recipe: Path) -> set[str]:
    """The aliases a recipe's ``model.path`` names, in any variant of an override bundle.

    A launch serves one MODEL, so every variant's alias maps to the same checkpoint.
    """
    raw = yaml.safe_load(recipe.read_text())
    blocks = raw.values() if isinstance(raw, dict) and "base" in raw else [raw]
    paths: set[object] = set()
    for block in blocks:
        model = block.get("model") if isinstance(block, dict) else None
        value = model.get("path") if isinstance(model, dict) else None
        paths.update(value if isinstance(value, list) else [value])
    return {path for path in paths if isinstance(path, str) and not path.startswith(("hf:", "/"))}


def model_paths(
    cluster: Cluster,
    request: LaunchRequest,
    config_file: str,
    served: str | None,
) -> dict[str, str]:
    """srtslurm.yaml ``model_paths``: each recipe alias mapped to ``served``."""
    recipe = recipe_mirror_path(request.workspace, config_file)
    if not recipe.is_file():
        raise LaunchError(f"CONFIG_FILE {config_file} is not in the recipe mirror: {recipe}")
    aliases = recipe_aliases(recipe)
    if aliases and served is None:
        raise LaunchError(
            f"cluster {cluster.id!r} stages no checkpoint for MODEL={request.model}, "
            f"which recipe aliases {sorted(aliases)} name"
        )
    return dict.fromkeys(sorted(aliases), served) if served is not None else {}


def job_env(cluster: Cluster, request: LaunchRequest, served: str | None) -> dict[str, str]:
    """MODEL_PATH and SERVED_MODEL_NAME, for the job's benchmark and eval clients."""
    env: dict[str, str] = {}
    if served is not None:
        env["MODEL_PATH"] = served
    if name := _override(cluster, request, "served_name"):
        env["SERVED_MODEL_NAME"] = str(name)
    return env


def single_node_model_path(cluster: Cluster, request: LaunchRequest) -> str:
    """What the single-node recipe's ``hf:<MODEL>`` serves: the staged checkpoint, or the Hub."""
    srt = slurm_settings(cluster).srt_slurm
    staged = srt is not None and srt.single_node_models == "staged"
    model = checkpoint(cluster, request) if staged else None
    return str(host_path(cluster, model)) if model is not None else f"hf:{request.model}"


SHARED_HF_CACHE_LANES: dict[str, tuple[Match, ...]] = {
    "mi300x-amd": (Match(model_glob="zai-org/GLM-5.3"),),
    "mi355x-amds": (
        Match(agentic=True, model_glob="MiniMaxAI/MiniMax-M3*"),
        Match(agentic=True, model_glob="amd/MiniMax-M3*"),
        Match(agentic=True, model_glob="zai-org/GLM-5.2-FP8"),
        Match(agentic=True, model_glob="deepseek-ai/DeepSeek-V4.1-Flash"),
        Match(
            frameworks=any_of("vllm", "atom"),
            agentic=True,
            model_glob="deepseek-ai/DeepSeek-V4-Pro",
        ),
        Match(
            frameworks=any_of("vllm", "atom"),
            agentic=True,
            model_glob="deepseek-ai/DeepSeek-V4-Pro-0813",
        ),
    ),
}


def single_node_hf_cache(cluster: Cluster, request: LaunchRequest) -> Path:
    """The HF hub cache a single-node job mounts at HF_HUB_CACHE."""
    shared = any(rule(request) for rule in SHARED_HF_CACHE_LANES.get(cluster.id, ()))
    return volume_path(cluster, "shared-hf-hub-cache" if shared else "hf-hub-cache")
