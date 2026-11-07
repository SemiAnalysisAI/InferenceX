"""Launch drivers, and the one map from a launch path to the driver that runs it.

A driver runs one benchmark point and returns its exit code, registering cleanups on the
launch's Lifecycle. Drivers say what runs; the cluster's backend runs it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from infx.launch import policy
from infx.launch.backends import backend_class
from infx.launch.context import Launch, LaunchError
from infx.launch.drivers import script, srt
from infx.launch.policy import LaunchPath, launch_path

if TYPE_CHECKING:
    from infx.clusters import Cluster
    from infx.launch.lifecycle import Lifecycle
    from infx.launch.request import LaunchRequest


@dataclass(frozen=True)
class Route:
    """A launch path's driver and the scheduler it requires (None: any)."""

    scheduler: str | None
    run: Callable[[Launch], int]


ROUTES: dict[LaunchPath, Route] = {
    LaunchPath.SRT_SINGLE: Route("slurm", srt.run_single_node),
    LaunchPath.SRT_BATCH: Route("slurm", srt.run_batch),
    LaunchPath.SRT_MULTI: Route("slurm", srt.run_multinode),
    LaunchPath.SRT_NATIVE: Route("slurm", srt.run_multinode),
    LaunchPath.SRT_VIDEO: Route("slurm", srt.run_video),
    LaunchPath.SCRIPT: Route(None, script.run),
}


def check_platform(cluster: Cluster) -> None:
    """Raise ``LaunchError`` naming every row of the cluster's platform its record contradicts."""
    platform = policy.PLATFORMS.get(cluster.id)
    if platform is not None and (problems := platform.problems(cluster)):
        raise LaunchError(
            f"configs/platforms/{cluster.id}.yaml disagrees with the cluster record:\n  "
            + "\n  ".join(problems)
        )


def run(cluster: Cluster, request: LaunchRequest, life: Lifecycle) -> int:
    """Run ``request`` with the driver its launch path selects.

    Fails before any work if the cluster's platform rows contradict its record, or if the
    path needs another scheduler than the cluster's.
    """
    check_platform(cluster)
    path = launch_path(cluster.id, request)
    life.event.set(launch_path=path.value)
    route = ROUTES[path]
    if route.scheduler not in (None, cluster.scheduler):
        runnable = sorted(p for p, r in ROUTES.items() if r.scheduler in (None, cluster.scheduler))
        raise LaunchError(
            f"launch path {path} needs a {route.scheduler} cluster; {cluster.id!r} uses "
            f"scheduler {cluster.scheduler!r}, which runs only: {', '.join(runnable)}"
        )
    backend = backend_class(cluster.scheduler)(cluster, request, life)
    return route.run(Launch(cluster, backend, request, life, path))
