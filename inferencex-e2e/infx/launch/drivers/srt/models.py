"""Which checkpoint an srt-slurm job serves.

A recipe's ``model.path`` is an ``hf:`` id or an absolute path, which srtctl reads as given,
or an alias. Every alias the submitted recipe declares maps to the cluster's checkpoint for
MODEL: the ``models.entries`` record keyed by MODEL's basename, or ``<basename>@<root>`` for
another copy, where a node-local copy wins. ``OVERRIDES`` holds the exceptions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from infx.clusters.slurm import model_path, slurm_settings
from infx.launch.context import LaunchError
from infx.launch.drivers.srt.recipe import recipe_mirror_path
from infx.launch.policy import Match, any_of

if TYPE_CHECKING:
    from infx.clusters import Cluster
    from infx.launch.request import LaunchRequest


@dataclass(frozen=True)
class Override:
    """What differs for the requests ``when`` matches; per field, the first such row wins."""

    when: Match
    entry: str | None = None  # the models.entries key served instead
    served_name: str | None = None  # SERVED_MODEL_NAME, when the frontend registers another name
    require_config: bool = False  # the checkpoint's config.json must be readable before submission


OVERRIDES: dict[str, tuple[Override, ...]] = {
    # The NVMe copy is not on every node; only vLLM reads it.
    "b300-dsxe": (
        Override(
            Match(frameworks=any_of("sglang"), model_glob="*/DeepSeek-V4-Pro-0813"),
            entry="DeepSeek-V4-Pro-0813",
        ),
    ),
    "gb200-nv": (
        # dsr1 SGLang and DSV4 vLLM read the Lustre copies; TRT and the DSV4 power lane NVMe.
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
    # Operators stage these checkpoints; a missing one fails before submission.
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
    """A staged checkpoint: its host path, and whether each node holds its own copy."""

    path: Path
    node_local: bool


def checkpoint(cluster: Cluster, request: LaunchRequest) -> Checkpoint | None:
    """MODEL's checkpoint on ``cluster``, or None when the cluster stages none.

    Raises ``LaunchError`` when it must be readable here and is not.
    """
    volumes = slurm_settings(cluster).volumes
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
    path = model_path(cluster, str(key))
    required = _override(cluster, request, "require_config")
    if required and not os.access(path / "config.json", os.R_OK):
        raise LaunchError(f"model checkpoint is unavailable: no readable {path}/config.json")
    return Checkpoint(path, node_local(str(key)))


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
    cluster: Cluster, request: LaunchRequest, config_file: str, model: Checkpoint | None
) -> dict[str, str]:
    """srtslurm.yaml ``model_paths``: each alias of ``config_file``'s recipe mapped to ``model``."""
    recipe = recipe_mirror_path(request.workspace, config_file)
    if not recipe.is_file():
        raise LaunchError(f"CONFIG_FILE {config_file} is not in the recipe mirror: {recipe}")
    aliases = recipe_aliases(recipe)
    if aliases and model is None:
        raise LaunchError(
            f"cluster {cluster.id!r} stages no checkpoint for MODEL={request.model}, "
            f"which recipe aliases {sorted(aliases)} name"
        )
    return dict.fromkeys(sorted(aliases), str(model.path)) if model is not None else {}


def job_env(cluster: Cluster, request: LaunchRequest, model: Checkpoint | None) -> dict[str, str]:
    """MODEL_PATH and SERVED_MODEL_NAME, for the job's benchmark and eval clients."""
    env: dict[str, str] = {}
    if model is not None:
        env["MODEL_PATH"] = str(model.path)
    if served := _override(cluster, request, "served_name"):
        env["SERVED_MODEL_NAME"] = str(served)
    return env


def single_node_model_path(cluster: Cluster, request: LaunchRequest) -> str:
    """What the single-node recipe's ``hf:<MODEL>`` serves: the staged checkpoint, or the Hub."""
    srt = slurm_settings(cluster).srt_slurm
    staged = srt is not None and srt.single_node_models == "staged"
    model = checkpoint(cluster, request) if staged else None
    return str(model.path) if model is not None else f"hf:{request.model}"


# AgentX checkpoints these single-node jobs read from the shared Hub cache (the
# ``shared-hf-hub-cache`` volume) rather than the node-local one.
SHARED_HF_CACHE_LANES: dict[str, tuple[Match, ...]] = {
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
    """Return the HF hub cache mounted at HF_HUB_CACHE for a single-node job."""
    shared = any(rule(request) for rule in SHARED_HF_CACHE_LANES.get(cluster.id, ()))
    volume = "shared-hf-hub-cache" if shared else "hf-hub-cache"
    path = slurm_settings(cluster).path(volume)
    if path is None:
        raise LaunchError(f"cluster {cluster.id!r} has no {volume} volume")
    return path
