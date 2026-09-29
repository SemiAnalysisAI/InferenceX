"""Which checkpoint an srt-slurm job serves, and under which recipe aliases.

Multi-node lanes resolve ``ModelRule`` tables into srtslurm.yaml ``model_paths``;
single-node points bind the recipe's ``hf:<MODEL>`` alias to a staged checkpoint or the
Hugging Face cache. Checkpoints are ``models.entries`` keys of the cluster record. The
aliases here depend on the request; the cluster's ``srt-slurm.model-aliases`` name the
checkpoints every recipe there may use, and no alias has both owners.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from infx.clusters.slurm import model_path, slurm_settings
from infx.launch.policy import Match, any_of

if TYPE_CHECKING:
    from infx.clusters import Cluster
    from infx.launch.request import LaunchRequest


@dataclass(frozen=True)
class ModelRule:
    """One MODEL_PATH / SRT_SLURM_MODEL_PREFIX case of a lane.

    ``alias`` is the recipe ``model.path`` served (None exports no alias); ``entry``
    names the ``models.entries`` checkpoint (None serves MODEL, an HF id). ``when`` may
    constrain the launch's DCGM power decision.
    """

    when: Match
    alias: str | None
    entry: str | None
    env_path: bool = False  # an exported MODEL_PATH (additional-settings) wins
    env_path_if_dir: bool = False  # ... only when it is a directory on this host
    served_name: str | None = None  # SERVED_MODEL_NAME
    extra_aliases: tuple[str, ...] = ()
    hf_fallback: str | None = None  # serve hf:<id> when the checkpoint is absent here
    require_config: bool = False  # fail unless <path>/config.json is readable here

    @property
    def aliases(self) -> tuple[str, ...]:
        """Every recipe alias the rule maps to its checkpoint."""
        return tuple(alias for alias in (self.alias, *self.extra_aliases) if alias)


@dataclass(frozen=True)
class ResolvedModel:
    """The checkpoint a job serves and the recipe aliases that name it."""

    path: str  # checkpoint path, hf:<id>, or MODEL itself
    aliases: tuple[str, ...]
    served_name: str | None


def resolve_model(
    cluster: Cluster, rules: Sequence[ModelRule], request: LaunchRequest, *, dcgm: bool = False
) -> ResolvedModel | None:
    """Apply the first rule matching ``request`` and ``dcgm``; ``None`` when no rule does.

    Raises ``ValueError`` when a rule requires a readable config.json that is missing.
    """
    for rule in rules:
        if not rule.when(request, dcgm=dcgm):
            continue
        path = str(model_path(cluster, rule.entry)) if rule.entry else (request.model or "")
        exported = request.env.get("MODEL_PATH", "")
        if exported and (rule.env_path or (rule.env_path_if_dir and Path(exported).is_dir())):
            path = exported
        if rule.hf_fallback and not Path(path).is_dir():
            path = f"hf:{rule.hf_fallback}"
        if rule.require_config and not os.access(Path(path) / "config.json", os.R_OK):
            raise ValueError(f"model checkpoint is unavailable: no readable {path}/config.json")
        return ResolvedModel(path, rule.aliases, rule.served_name)
    return None


# Shared by b200-nscale native single-node points and its multinode lane.
B200_NSCALE_MODELS = (
    ModelRule(
        Match(any_of("dsv41flash"), any_of("fp4"), any_of("vllm", "sglang"), multinode=False),
        None,
        None,
    ),
    ModelRule(Match(any_of("dsr1"), any_of("fp4")), "dsr1", "DeepSeek-R1-0528-NVFP4-v2"),
    ModelRule(Match(any_of("dsr1"), any_of("fp8")), "dsr1-fp8", "DeepSeek-R1-0528"),
    ModelRule(
        Match(any_of("dsv4"), any_of("fp4"), model_glob="deepseek-ai/DeepSeek-V4-Pro-0813"),
        None,
        "DeepSeek-V4-Pro-0813",
        env_path=True,
    ),
    # Node-local weights are not visible on the runner/login node.
    ModelRule(Match(any_of("dsv4"), any_of("fp4")), "deepseek-v4-pro", "DeepSeek-V4-Pro-NVFP4"),
    ModelRule(Match(any_of("qwen3.5"), any_of("fp8")), "qwen3.5-fp8", "Qwen3.5-397B-A17B-FP8"),
    # SGLang keys moved to NVFP4-V2 while the TRT configs still declare plain NVFP4.
    ModelRule(
        Match(any_of("qwen3.5"), any_of("fp4"), model_glob="*NVFP4-V2"),
        "qwen3.5-fp4",
        "Qwen3.5-397B-A17B-NVFP4-V2",
    ),
    ModelRule(Match(any_of("qwen3.5"), any_of("fp4")), "qwen3.5-fp4", "Qwen3.5-397B-A17B-NVFP4"),
    ModelRule(Match(any_of("glm5.1"), any_of("fp8")), "glm5.1-fp8", "GLM-5.1-FP8", env_path=True),
    ModelRule(Match(any_of("glm5.2"), any_of("fp4")), "glm5.2-fp4", "GLM-5.2-NVFP4", env_path=True),
    ModelRule(Match(any_of("glm5.2"), any_of("fp8")), "glm5.2-fp8", "GLM-5.2-FP8", env_path=True),
    ModelRule(Match(any_of("minimaxm3"), any_of("fp8")), "minimax-m3-mxfp8", "MiniMax-M3-MXFP8"),
    ModelRule(
        Match(any_of("minimaxm3"), any_of("fp4")), "nvidia/MiniMax-M3-NVFP4", "MiniMax-M3-NVFP4"
    ),
    ModelRule(Match(any_of("kimik3"), any_of("fp4")), "kimik3", "Kimi-K3"),
    # No pool setting names this checkpoint; default to the node-local copy.
    ModelRule(
        Match(any_of("qwen3.8next"), any_of("fp4")),
        "qwen3.8next-fp4",
        "Qwen3.8-Flash-Next-NVFP4",
        env_path_if_dir=True,
    ),
)

# b200-nscale native lanes. Aliases must match the checked-in recipes.
B200_NSCALE_NATIVE_MODELS = (
    ModelRule(
        Match(any_of("dsv4"), any_of("fp4"), model_glob="deepseek-ai/DeepSeek-V4-Pro-0813"),
        "deepseek-v4-pro-0813",
        "DeepSeek-V4-Pro-0813",
    ),
    ModelRule(
        Match(any_of("dsv4"), any_of("fp4")), "deepseek-v4-pro", "DeepSeek-V4-Pro", env_path=True
    ),
    ModelRule(Match(any_of("kimik3"), any_of("fp4")), "kimik3", "Kimi-K3", env_path=True),
    ModelRule(
        Match(any_of("glm5.2"), any_of("fp4")), "glm-5.2-fp4", "GLM-5.2-NVFP4", env_path=True
    ),
    ModelRule(Match(any_of("glm5.1"), any_of("fp8")), "glm5.1-fp8", "GLM-5.1-FP8", env_path=True),
)

GB200_NV_MODELS = (
    # The DSV4 power lane reads the checkpoint staged on compute-node NVMe.
    ModelRule(
        Match(any_of("dsv4"), any_of("fp4"), any_of("dynamo-sglang"), agentic=False, dcgm=True),
        "deepseek-v4-pro",
        "DeepSeek-V4-Pro@numa1",
    ),
    ModelRule(
        Match(any_of("dsr1"), any_of("fp8"), any_of("dynamo-sglang")),
        "dsr1-fp8",
        "deepseek-r1-0528",
    ),
    ModelRule(
        Match(any_of("dsr1"), any_of("fp4"), any_of("dynamo-sglang")),
        "dsr1-fp4",
        "deepseek-r1-0528-fp4-v2",
    ),
    ModelRule(
        Match(any_of("dsv4"), any_of("fp4"), any_of("dynamo-sglang")),
        "deepseek-v4-pro",
        "deepseek-v4-pro",
    ),
    ModelRule(
        Match(any_of("qwen3.5"), any_of("fp8"), any_of("dynamo-sglang")),
        "qwen3.5-fp8",
        "Qwen3.5-397B-A17B-FP8",
    ),
    ModelRule(
        Match(any_of("qwen3.5"), any_of("fp4"), any_of("dynamo-sglang")),
        "qwen3.5-fp4",
        "Qwen3.5-397B-A17B-NVFP4-V2",
    ),
    ModelRule(
        Match(any_of("glm5.2"), any_of("fp4"), any_of("dynamo-sglang")),
        "glm-5.2-fp4",
        "GLM-5.2-NVFP4",
    ),
    ModelRule(
        Match(any_of("dsr1"), any_of("fp4"), any_of("dynamo-trt")),
        "dsr1",
        "DeepSeek-R1-0528-NVFP4-v2",
        served_name="deepseek-r1-fp4",
    ),
    ModelRule(
        Match(any_of("dsr1"), any_of("fp8"), any_of("dynamo-trt")),
        "dsr1-fp8",
        "DeepSeek-R1-0528",
        served_name="deepseek-r1-fp8",
    ),
    ModelRule(
        Match(any_of("minimaxm3"), any_of("fp4"), any_of("dynamo-trt")),
        "minimax-m3-nvfp4",
        "MiniMax-M3-NVFP4",
        served_name="nvidia/MiniMax-M3-NVFP4",
    ),
    ModelRule(Match(any_of("kimik3"), any_of("fp4"), any_of("dynamo-vllm")), "kimi-k3", "Kimi-K3"),
    # Base DeepSeek-V4-Pro, not the -NVFP4 re-quant: the pinned vLLM deepseek_v4
    # loader lacks the NVFP4 export's extra quant params, so the MXFP4 alias
    # points at the base checkpoint too.
    ModelRule(
        Match(any_of("dsv4"), any_of("fp4"), any_of("dynamo-vllm")),
        "deepseek-v4-pro",
        "DeepSeek-V4-Pro",
        extra_aliases=("deepseek-v4-pro-mxfp4",),
    ),
    ModelRule(
        Match(any_of("minimaxm3"), any_of("fp8"), any_of("dynamo-vllm")),
        "minimax-m3-mxfp8",
        "MiniMax-M3-MXFP8",
    ),
    ModelRule(
        Match(any_of("minimaxm3"), any_of("fp4"), any_of("dynamo-vllm")),
        "minimax-m3-nvfp4",
        "MiniMax-M3-NVFP4",
    ),
)

# TRT serves DSV4 under its HF id.
GB300_NV_MODELS = (
    ModelRule(
        Match(any_of("dsv41flash"), any_of("fp4"), any_of("vllm", "sglang"), multinode=False),
        None,
        None,
    ),
    ModelRule(
        Match(any_of("dsr1"), any_of("fp4")),
        "dsr1",
        "DeepSeek-R1-0528-NVFP4-v2",
        served_name="deepseek-r1-fp4",
    ),
    ModelRule(
        Match(any_of("dsr1"), any_of("fp8")),
        "dsr1-fp8",
        "DeepSeek-R1-0528",
        served_name="deepseek-r1-fp8",
    ),
    ModelRule(
        Match(
            any_of("dsv4"),
            any_of("fp4"),
            any_of("dynamo-trt"),
            model_glob="deepseek-ai/DeepSeek-V4-Pro-0813",
        ),
        "deepseek-ai/DeepSeek-V4-Pro",
        "DeepSeek-V4-Pro-0813",
    ),
    ModelRule(
        Match(any_of("dsv4"), any_of("fp4"), model_glob="deepseek-ai/DeepSeek-V4-Pro-0813"),
        "deepseek-v4-pro-0813",
        "DeepSeek-V4-Pro-0813",
    ),
    ModelRule(
        Match(any_of("dsv4"), any_of("fp4"), any_of("dynamo-trt")),
        "deepseek-ai/DeepSeek-V4-Pro",
        "DeepSeek-V4-Pro",
    ),
    ModelRule(Match(any_of("dsv4"), any_of("fp4")), "deepseek-v4-pro", "DeepSeek-V4-Pro"),
    ModelRule(
        Match(any_of("glm5.2"), any_of("fp4"), any_of("dynamo-trt")),
        "nvidia/GLM-5.2-NVFP4",
        "GLM-5.2-NVFP4",
        served_name="GLM-5.2-NVFP4",
    ),
    ModelRule(Match(any_of("glm5.2"), any_of("fp4")), "glm-5.2-fp4", "GLM-5.2-NVFP4"),
    ModelRule(
        Match(any_of("minimaxm3"), any_of("fp4")), "nvidia/MiniMax-M3-NVFP4", "MiniMax-M3-NVFP4"
    ),
    ModelRule(Match(any_of("minimaxm3"), any_of("fp8")), "minimax-m3-mxfp8", "MiniMax-M3-MXFP8"),
    ModelRule(Match(any_of("kimik3"), any_of("fp4")), "moonshotai/Kimi-K3", "Kimi-K3"),
    ModelRule(Match(any_of("qwen3.5"), any_of("fp4")), "qwen3.5-fp4", "Qwen3.5-397B-A17B-NVFP4-V2"),
    ModelRule(Match(any_of("qwen3.5"), any_of("fp8")), "qwen3.5-fp8", "Qwen3.5-397B-A17B-FP8"),
)

# The staged checkpoint must be readable before submission.
H100_DGXC_MODELS = (
    ModelRule(
        Match(any_of("dsr1"), any_of("fp8"), any_of("dynamo-sglang")),
        "dsr1-fp8",
        "dsr1-fp8",
        require_config=True,
    ),
    ModelRule(
        Match(any_of("dsr1"), any_of("fp8"), any_of("dynamo-trt")),
        "DeepSeek-R1-0528",
        "dsr1-fp8",
        served_name="DeepSeek-R1-0528",
        require_config=True,
    ),
)

H200_DGXC_MODELS = (
    ModelRule(
        Match(any_of("dsv4"), any_of("fp8"), any_of("dynamo-sglang")),
        "deepseek-v4-pro-0813",
        "DeepSeek-V4-Pro",
        require_config=True,
    ),
    ModelRule(
        Match(any_of("dsr1"), any_of("fp8"), any_of("dynamo-sglang")),
        "dsr1-fp8",
        "DeepSeek-R1-0528",
    ),
    # Absent copies fall back to the Hub.
    ModelRule(
        Match(any_of("glm5.2"), any_of("fp8"), any_of("dynamo-sglang")),
        "glm5.2-fp8",
        "GLM-5.2-FP8",
        hf_fallback="zai-org/GLM-5.2-FP8",
    ),
    ModelRule(
        Match(any_of("dsr1"), any_of("fp8"), any_of("dynamo-trt")),
        "DeepSeek-R1-0528",
        "DeepSeek-R1-0528",
        served_name="DeepSeek-R1-0528",
    ),
    ModelRule(Match(any_of("kimik3"), any_of("fp4"), any_of("vllm")), "kimik3", "Kimi-K3"),
)


# --------------------------------------------------------------------------
# Single-node points
# --------------------------------------------------------------------------

# Clusters whose single-node SRT_MODEL_PATH is a staged checkpoint rather than
# hf:<MODEL>; paths that are not absolute fall back to the HF cache.
SINGLE_NODE_MODELS: dict[str, tuple[ModelRule, ...]] = {"b200-nscale": B200_NSCALE_MODELS}


@dataclass(frozen=True)
class StagedBasenames:
    """Single-node checkpoints found by HF basename.

    ``hf_models`` download into the HF cache; the first matching ``copies`` rule
    names the ``models.entries`` copy a request reads. Other basenames with a
    ``models.entries`` record use it; the rest read ``<fallback_volume>/<basename>``.
    """

    fallback_volume: str
    hf_models: frozenset[str]
    copies: tuple[tuple[Match, str], ...] = ()


SINGLE_NODE_BASENAMES: dict[str, StagedBasenames] = {
    "b300-dsxe": StagedBasenames(
        "data-models",
        any_of("RadixArk/Qwen3.8-Flash-Next-NVFP4"),
        # vLLM reads the node-local NVMe copy, other engines the shared one.
        copies=(
            (
                Match(frameworks=any_of("vllm"), model_glob="*/DeepSeek-V4-Pro-0813"),
                "DeepSeek-V4-Pro-0813@scratch",
            ),
        ),
    ),
}

# AgentX checkpoints the jobs read from the shared Hub cache (the ``shared-hf-hub-cache``
# volume) rather than the node-local one.
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


def single_node_model_path(cluster: Cluster, request: LaunchRequest) -> str:
    """Return SRT_MODEL_PATH, the checkpoint behind the recipe's ``hf:<MODEL>`` alias."""
    model = request.model or ""
    rules = SINGLE_NODE_MODELS.get(cluster.id)
    if rules is not None:
        resolved = resolve_model(cluster, rules, request)
        if resolved is None:
            raise ValueError(
                f"unsupported model prefix/precision: {request.model_prefix}/{request.precision}"
            )
        return resolved.path if resolved.path.startswith("/") else f"hf:{model}"
    staged = SINGLE_NODE_BASENAMES.get(cluster.id)
    if staged is not None:
        basename = model.rsplit("/", 1)[-1]
        if model in staged.hf_models:
            return f"hf:{model}"
        for when, entry in staged.copies:
            if when(request):
                return str(model_path(cluster, entry))
        if basename in cluster.models.entries:
            return str(model_path(cluster, basename))
        return str(slurm_settings(cluster).volumes[staged.fallback_volume].path / basename)
    return f"hf:{model}"


def single_node_hf_cache(cluster: Cluster, request: LaunchRequest) -> Path:
    """Return the HF hub cache mounted at HF_HUB_CACHE for a single-node job."""
    shared = any(rule(request) for rule in SHARED_HF_CACHE_LANES.get(cluster.id, ()))
    volume = "shared-hf-hub-cache" if shared else "hf-hub-cache"
    path = slurm_settings(cluster).path(volume)
    if path is None:
        raise ValueError(f"cluster {cluster.id!r} has no {volume} volume")
    return path
