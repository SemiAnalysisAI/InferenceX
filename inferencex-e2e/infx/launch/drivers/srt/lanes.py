"""Each cluster's multi-node srt-slurm lanes: one ``SRT_LANES`` row per (cluster, launch path).

The rows are the ``srt-lanes`` of each ``configs/platforms`` record.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from infx.clusters.platforms import SrtLane
from infx.launch.context import LaunchError
from infx.launch.policy import PLATFORMS, LaunchPath, salloc_time_limit
from infx.launch.request import RequestError

if TYPE_CHECKING:
    from infx.clusters.slurm import SrtSlurmSettings
    from infx.launch.request import LaunchRequest, MultiNodeRequest, SrtRequest


SRT_LANES: dict[tuple[str, LaunchPath], SrtLane] = {
    (cluster_id, LaunchPath(path)): lane
    for cluster_id, platform in PLATFORMS.items()
    for path, lane in platform.srt_lanes.items()
}


def srt_lane(cluster_id: str, path: LaunchPath) -> SrtLane:
    """Return the srt-slurm lane for a cluster and launch path, or raise LaunchError if none."""
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

    A fixed profile limit wins; ``Platform.problems`` rejects a bump or lane limit it would
    shadow.
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
