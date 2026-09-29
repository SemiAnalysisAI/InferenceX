"""``python -m infx.launch {run,cleanup}``: the workflow entry points for benchmark launches.

``run`` resolves the cluster that owns ``RUNNER_NAME``, runs the driver its launch path
selects on the cluster's backend, and exits with its return code (130/143/129 after a
SIGINT/SIGTERM/SIGHUP). ``cleanup`` has the cluster's backend remove what earlier
launches on the runner left behind.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

from infx.clusters import Cluster, resolve_cluster
from infx.launch import drivers
from infx.launch.backends import BACKENDS, backend_class
from infx.launch.backends.base import BackendError
from infx.launch.context import LaunchError
from infx.launch.lifecycle import Lifecycle
from infx.launch.request import LaunchRequest, RequestError


def launch(cluster: Cluster, request: LaunchRequest) -> int:
    """Run ``request`` on ``cluster``; report a launch error as one ``ERROR:`` line."""
    with Lifecycle() as life:
        try:
            life.record(drivers.run(cluster, request, life))
        except (LaunchError, BackendError, RequestError) as error:
            print(f"ERROR: {error}", file=sys.stderr)
            life.record(1)
    return life.returncode


def _run(runner_config: Path | None) -> int:
    """Run one launch described by the environment and return its exit code."""
    try:
        request = LaunchRequest.from_env()
    except RequestError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return launch(resolve_cluster(request.runner_name, runner_config), request)


def _cleanup(runner_config: Path | None) -> int:
    """Have the runner's backend remove its leftovers and wait for them to go."""
    runner = os.environ.get("RUNNER_NAME", "")
    if not runner:
        print("ERROR: RUNNER_NAME is required", file=sys.stderr)
        return 1
    try:
        cluster = resolve_cluster(runner, runner_config)
    except (OSError, ValueError, yaml.YAMLError) as error:
        # Cleanup must not fail the job over config drift. Without the record, only
        # backends that find their jobs by runner name alone can clean up.
        print(f"WARNING: {error}", file=sys.stderr)
        for scheduler in BACKENDS:
            backend_class(scheduler).cleanup(None, runner)
        return 0
    backend_class(cluster.scheduler).cleanup(cluster.scheduler_settings, runner)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse the subcommand and dispatch."""
    parser = argparse.ArgumentParser(prog="python -m infx.launch", description=__doc__)
    parser.add_argument(
        "--runner-config",
        type=Path,
        default=None,
        help="runners.yaml to resolve clusters from (default: configs/runners.yaml)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run", help="launch the benchmark described by the environment")
    commands.add_parser("cleanup", help="remove this runner's leftover jobs and wait for them")
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args.runner_config)
    return _cleanup(args.runner_config)


if __name__ == "__main__":
    sys.exit(main())
