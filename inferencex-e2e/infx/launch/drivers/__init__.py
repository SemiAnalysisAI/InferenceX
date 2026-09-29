"""Launch drivers, the one map from a launch path to the driver that runs it, and dispatch.

A driver function runs one benchmark and returns the process exit code, registering its
cleanups on the launch's Lifecycle. Drivers choose *what* runs; the cluster's backend
runs it. ``ROUTES`` names each path's driver and the scheduler it needs (None: any).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from infx.launch import policy
from infx.launch.backends import backend_class
from infx.launch.context import Launch, LaunchError
from infx.launch.drivers import legacy, script, srt
from infx.launch.policy import LaunchPath, launch_path

if TYPE_CHECKING:
    from infx.clusters import Cluster
    from infx.launch.lifecycle import Lifecycle
    from infx.launch.request import LaunchRequest


@dataclass(frozen=True)
class Route:
    """The driver of one launch path and the scheduler it requires (None: any)."""

    scheduler: str | None
    run: Callable[[Launch], int]


ROUTES: dict[LaunchPath, Route] = {
    LaunchPath.SRT_SINGLE: Route("slurm", srt.run_single_node),
    LaunchPath.SRT_BATCH: Route("slurm", srt.run_batch),
    LaunchPath.SRT_MULTI: Route("slurm", srt.run_multinode),
    LaunchPath.SRT_NATIVE: Route("slurm", srt.run_multinode),
    LaunchPath.SCRIPT: Route(None, script.run),
    LaunchPath.LEGACY_TILERT: Route("slurm", legacy.run_tilert),
    LaunchPath.LEGACY_AMD_UTILS: Route("slurm", legacy.run_amd_utils),
}


def check_tables(clusters: Mapping[str, Cluster], only: str | None = None) -> None:
    """Raise ``LaunchError`` listing every workload-table row the cluster inventory contradicts.

    ``only`` limits the check to the rows keyed by that cluster id.
    """
    problems = [*policy.table_problems(clusters, only), *srt.table_problems(clusters, only)]
    if problems:
        raise LaunchError(
            "launch policy tables disagree with the cluster inventory:\n  " + "\n  ".join(problems)
        )


def run(cluster: Cluster, request: LaunchRequest, life: Lifecycle) -> int:
    """Run ``request`` on ``cluster`` with the driver its launch path selects.

    Before any work, the table rows keyed by the cluster are checked against its record,
    so a checkpoint, volume or profile a row names but the record lacks fails with every
    such row named; and a path whose driver requires another scheduler than the
    cluster's fails, naming the paths that cluster can run.
    """
    check_tables({cluster.id: cluster}, only=cluster.id)
    path = launch_path(cluster.id, request)
    route = ROUTES[path]
    if route.scheduler not in (None, cluster.scheduler):
        runnable = sorted(p for p, r in ROUTES.items() if r.scheduler in (None, cluster.scheduler))
        raise LaunchError(
            f"launch path {path} needs a {route.scheduler} cluster; {cluster.id!r} uses "
            f"scheduler {cluster.scheduler!r}, which runs only: {', '.join(runnable)}"
        )
    backend = backend_class(cluster.scheduler)(cluster, request, life)
    return route.run(Launch(cluster, backend, request, life, path))
