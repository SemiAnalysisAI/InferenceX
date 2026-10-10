"""Run the benchmark point the workflow environment describes, or clean up after earlier ones.

``run`` exits with the point's return code (128 + N after signal N) and leaves its record,
``job_event.json``, in the workspace; ``cleanup`` has the runner's cluster backend remove what
earlier launches on the runner left behind.
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
from infx.launch.event import JobEventBuilder
from infx.launch.lifecycle import Lifecycle
from infx.launch.request import LaunchRequest, RequestError


def launch(cluster: Cluster, request: LaunchRequest, event: JobEventBuilder | None = None) -> int:
    """Run ``request`` on ``cluster``; a launch error is one ``ERROR:`` line, not a traceback."""
    with Lifecycle(event) as life:
        try:
            rc = drivers.run(cluster, request, life)
        except (LaunchError, BackendError, RequestError) as error:
            life.event.error(error)
            rc = 1
        life.event.exited(rc)
        life.record(rc)
    return life.returncode


def _run(runner_config: Path | None) -> int:
    event = JobEventBuilder.begin(os.environ)
    rc = 1
    try:
        request = LaunchRequest.from_env()
        cluster = resolve_cluster(request.runner_name, runner_config)
        event.set(cluster=cluster.id)
        rc = launch(cluster, request, event)
    except RequestError as error:
        event.error(error)
    except SystemExit as stop:
        rc = 0 if stop.code is None else stop.code if isinstance(stop.code, int) else 1
        raise
    except BaseException as error:
        event.error(error, report=False)
        raise
    finally:
        event.emit(rc)
    return rc


def _cleanup(runner_config: Path | None) -> int:
    runner = os.environ.get("RUNNER_NAME", "")
    if not runner:
        print("ERROR: RUNNER_NAME is required", file=sys.stderr)
        return 1
    try:
        cluster = resolve_cluster(runner, runner_config)
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"WARNING: {error}", file=sys.stderr)
        for scheduler in BACKENDS:
            backend_class(scheduler).cleanup(None, runner)
        return 0
    backend_class(cluster.scheduler).cleanup(cluster.scheduler_settings, runner)
    return 0


def main(argv: list[str] | None = None) -> int:
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
