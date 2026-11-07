"""Workload policy the launch drivers share: tables keyed by cluster id, and functions over them.

Every table is a view of the platform records in ``configs/platforms``; tables one driver
reads alone live beside it. Cluster facts without a workload predicate belong in the
cluster record.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from enum import StrEnum
from typing import TYPE_CHECKING

from infx.clusters.platforms import Match, TimeBump, load_platforms
from infx.launch.context import LaunchError
from infx.launch.request import RequestError

if TYPE_CHECKING:
    from infx.clusters import Cluster
    from infx.launch.request import LaunchRequest

PLATFORMS = load_platforms()


class LaunchPath(StrEnum):
    """How ``python -m infx.launch run`` executes one request; ``drivers.ROUTES`` has each driver."""

    SRT_SINGLE = "srt-single"
    SRT_MULTI = "srt-multi"
    SRT_NATIVE = "srt-native"
    SRT_BATCH = "srt-batch"
    SRT_VIDEO = "srt-video"
    SCRIPT = "script"


NATIVE_SRT_LANES: dict[str, tuple[Match, ...]] = {
    cluster_id: platform.native_srt_lanes
    for cluster_id, platform in PLATFORMS.items()
    if platform.native_srt_lanes
}

BATCH_WRAPPED_LANES: dict[str, Match] = {
    cluster_id: platform.batch_wrapped_lane
    for cluster_id, platform in PLATFORMS.items()
    if platform.batch_wrapped_lane is not None
}


VIDEO_SCENARIO = "video-generation"


def launch_path(cluster_id: str, request: LaunchRequest) -> LaunchPath:
    """Return the launch path that runs ``request`` on the cluster ``cluster_id``."""
    if request.scenario_type == VIDEO_SCENARIO:
        if request.is_multinode or request.is_agentic:
            raise LaunchError("video-generation points are single-node and not agentic")
        return LaunchPath.SRT_VIDEO
    if request.is_multinode:
        if any(lane(request) for lane in NATIVE_SRT_LANES.get(cluster_id, ())):
            return LaunchPath.SRT_NATIVE
        return LaunchPath.SRT_MULTI
    wrapped = BATCH_WRAPPED_LANES.get(cluster_id)
    if wrapped is not None and wrapped(request) and not request.batch_reentry:
        return LaunchPath.SRT_BATCH
    if request.bench_script_override:
        return LaunchPath.SCRIPT
    return LaunchPath.SRT_SINGLE


SALLOC_TIME_BUMPS: dict[str, TimeBump] = {
    cluster_id: platform.salloc_time_bump
    for cluster_id, platform in PLATFORMS.items()
    if platform.salloc_time_bump is not None
}


def salloc_time_limit(cluster_id: str, request: LaunchRequest) -> int | None:
    """SALLOC_TIME_LIMIT in minutes: the cluster's bump, or as requested (always for eval-only)."""
    bump = SALLOC_TIME_BUMPS.get(cluster_id)
    if (
        bump is not None
        and bump.when(request)
        and not request.eval_only
        and request.conc is not None
        and request.conc >= bump.min_conc
    ):
        return bump.minutes
    return request.salloc_time_limit


def runtime_env(
    cluster: Cluster, request: LaunchRequest, *settings: Mapping[str, str]
) -> dict[str, str]:
    """The launch environment with the cluster's runtime settings applied over it.

    In increasing precedence: the cluster's ``env``, then each of ``settings``. They all
    beat the runner host's values, but never a name the point's additional-settings set.
    """
    merged = {k: v for source in (cluster.env, *settings) for k, v in source.items()}
    chosen = point_settings(request)
    return {**request.env, **{k: v for k, v in merged.items() if k not in chosen}}


def point_settings(request: LaunchRequest) -> frozenset[str]:
    """The names the point's NAME=value additional-settings set; the workflow exports them.

    A null value, which the workflow sends for a role without additional-settings, sets none.
    Raises ``RequestError`` if a value is not a JSON list.
    """
    names: set[str] = set()
    for name in ("PREFILL_ADDITIONAL_SETTINGS", "DECODE_ADDITIONAL_SETTINGS"):
        try:
            values = json.loads(request.env.get(name) or "[]")
        except ValueError:
            continue
        if values is None:
            continue
        if not isinstance(values, list):
            kind = type(values).__name__
            raise RequestError(f"{name} must be a JSON list of NAME=value strings, got {kind}")
        names.update(v.partition("=")[0] for v in values if isinstance(v, str))
    return frozenset(names)
