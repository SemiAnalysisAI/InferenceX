"""Each cluster's multi-node srt-slurm lanes: one ``SRT_LANES`` row per (cluster, launch path)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from infx.launch.context import LaunchError
from infx.launch.policy import LaunchPath, Match, any_of, salloc_time_limit
from infx.launch.request import RequestError

if TYPE_CHECKING:
    from infx.clusters.slurm import SrtSlurmSettings
    from infx.launch.request import LaunchRequest, MultiNodeRequest, SrtRequest


@dataclass(frozen=True)
class LaneMount:
    """A cluster volume the lane mounts for the requests ``when`` matches."""

    when: Match
    volume: str
    target: str | None = None
    world_writable: bool = False


@dataclass(frozen=True)
class SrtLane:
    """How one cluster's multi-node srt-slurm lane differs from the others."""

    frameworks: frozenset[str] | None = None
    rejects: tuple[tuple[Match, str], ...] = ()
    setup_scripts: Mapping[str, str] = field(default_factory=dict)
    mounts: tuple[LaneMount, ...] = ()
    shared_run_root: tuple[Match, ...] = ()
    eval_unsets: tuple[str, ...] = ()
    real_verification: Match | None = None
    head_frontend: Match | None = None
    write_eval_meta: bool = False
    time_limit: str | None = None
    long_time_limit: str | None = None
    long_time: Match | None = None
    # Where the AgentX client writes its per-concurrency artifacts in the container.
    agentic_result_dir: str = "/logs/agentic"


_DYNAMO = any_of("dynamo-sglang", "dynamo-trt", "dynamo-vllm")

SRT_LANES: dict[tuple[str, LaunchPath], SrtLane] = {
    ("b200-nscale", LaunchPath.SRT_NATIVE): SrtLane(),
    ("b200-nscale", LaunchPath.SRT_MULTI): SrtLane(
        frameworks=_DYNAMO,
        rejects=(
            (
                Match(any_of("dsv4"), frameworks=any_of("dynamo-sglang", "dynamo-trt")),
                "multinode dsv4 supports only dynamo-vllm",
            ),
        ),
    ),
    ("b300-dsxe", LaunchPath.SRT_MULTI): SrtLane(frameworks=_DYNAMO),
    ("gb200-nv", LaunchPath.SRT_MULTI): SrtLane(
        frameworks=_DYNAMO,
        setup_scripts={"dynamo-sglang": "install-torchao.sh"},
        mounts=(
            LaneMount(
                Match(any_of("glm5.2"), any_of("fp4"), any_of("dynamo-sglang"), agentic=True),
                "dynamo-wheels",
                "/configs/dynamo-wheels",
                world_writable=True,
            ),
        ),
        shared_run_root=(
            Match(any_of("minimaxm3", "kimik3", "qwen3.5", "glm5.2")),
            Match(any_of("dsv4"), frameworks=any_of("dynamo-vllm")),
            Match(any_of("dsv4"), frameworks=any_of("dynamo-sglang"), agentic=True),
        ),
    ),
    ("gb300-nv", LaunchPath.SRT_MULTI): SrtLane(
        real_verification=Match(frameworks=any_of("dynamo-trt"), agentic=True),
        head_frontend=Match(any_of("dsv4")),
        write_eval_meta=True,
        time_limit="4:00:00",
        long_time_limit="8:00:00",
        long_time=Match(
            any_of("dsv4"), frameworks=any_of("dynamo-sglang", "dynamo-trt"), agentic=True
        ),
    ),
    ("h100-dgxc", LaunchPath.SRT_MULTI): SrtLane(frameworks=any_of("dynamo-sglang", "dynamo-trt")),
    ("h200-dgxc", LaunchPath.SRT_MULTI): SrtLane(
        frameworks=any_of("dynamo-sglang", "dynamo-trt", "vllm"),
        time_limit="4:00:00",
        long_time_limit="8:00:00",
        long_time=Match(any_of("dsv4"), frameworks=any_of("dynamo-sglang"), agentic=True),
    ),
    ("mi355x-amds", LaunchPath.SRT_MULTI): SrtLane(
        mounts=(
            LaneMount(Match(), "aiperf-cache", "/aiperf_mmap_cache"),
            LaneMount(Match(frameworks=any_of("tilert")), "it-share-data", "/models"),
        ),
        eval_unsets=(
            "roles.prefill.args.ep-dispatch-algorithm",
            "roles.decode.args.ep-dispatch-algorithm",
        ),
        time_limit="01:00:00",
        agentic_result_dir="/infmax-workspace/LOGS/agentic",
    ),
}


def srt_lane(cluster_id: str, path: LaunchPath) -> SrtLane:
    try:
        return SRT_LANES[(cluster_id, path)]
    except KeyError:
        raise LaunchError(f"cluster {cluster_id!r} has no {path} srt-slurm lane") from None


def check_request(lane: SrtLane, request: SrtRequest) -> None:
    """Refuse requests the lane does not run, before any setup."""
    framework = request.framework
    if lane.frameworks is not None and framework not in lane.frameworks:
        supported = ", ".join(sorted(lane.frameworks))
        raise LaunchError(
            f"Unsupported framework: {framework}. Supported frameworks are: {supported}"
        )
    for match, message in lane.rejects:
        if match(request):
            raise LaunchError(f"{message} (FRAMEWORK={framework})")


def srt_recipe(request: MultiNodeRequest) -> str:
    """SRT_RECIPE, or on an eval-only run its real-verification EVAL_SRT_RECIPE."""
    if request.eval_only and request.eval_srt_recipe:
        print(
            f"EVAL_ONLY=true: selecting real-verification recipe {request.eval_srt_recipe}",
            flush=True,
        )
        return request.eval_srt_recipe
    return request.srt_recipe


def srt_time_limit(
    cluster_id: str, request: LaunchRequest, lane: SrtLane | None, srt: SrtSlurmSettings
) -> str:
    """A job's srtslurm.yaml ``default_time_limit`` (``lane`` None: single-node).

    A fixed profile limit wins; table_problems rejects a bump or lane limit it would shadow.
    """
    if srt.default_time_limit is not None:
        return srt.default_time_limit
    if lane is None:
        minutes = srt.single_node_time_limit or salloc_time_limit(cluster_id, request)
    elif lane.long_time is not None and lane.long_time_limit and lane.long_time(request):
        return lane.long_time_limit
    elif lane.time_limit is not None:
        return lane.time_limit
    else:
        minutes = request.salloc_time_limit
    if minutes is None:
        raise RequestError.missing("SALLOC_TIME_LIMIT")
    return str(minutes)
