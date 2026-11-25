from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime

from infx.bench.proc import echo
from infx.launch.backends.slurm.cli import SlurmError

_HOST = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
_HOSTLIST = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.,\[\]-]*")
_AVAILABLE = {"IDLE", "ALLOCATED", "MIXED", "COMPLETING"}
_UNHEALTHY = {
    "DRAIN",
    "DRAINED",
    "DRAINING",
    "DOWN",
    "FAIL",
    "FAILING",
    "NOT_RESPONDING",
    "INVALID_REG",
    "UNKNOWN",
    "FUTURE",
    "POWER_DOWN",
    "POWERED_DOWN",
    "POWERING_DOWN",
}
_QUERY_TIMEOUT_S = 15


@dataclass(frozen=True)
class NodeState:
    name: str
    state: str
    reason: str


@dataclass(frozen=True)
class NodeHealth:
    job_id: str
    observed_utc: str
    expected: tuple[str, ...]
    nodes: tuple[NodeState, ...]
    errors: tuple[str, ...]


def _query(argv: list[str]) -> str:
    echo(argv)
    try:
        result = subprocess.run(
            argv,
            env={**os.environ, "LC_ALL": "C"},
            capture_output=True,
            text=True,
            timeout=_QUERY_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SlurmError(f"could not query node health: {error}") from error
    if result.returncode:
        raise SlurmError(f"node health query exited {result.returncode}: {result.stderr.strip()}")
    return result.stdout


def inspect_nodes(job_id: str, hostlist: str | None) -> NodeHealth:
    expected: tuple[str, ...] = ()
    nodes: list[NodeState] = []
    errors: list[str] = []
    try:
        if not hostlist or not _HOSTLIST.fullmatch(hostlist):
            raise SlurmError(f"Slurm job {job_id} has no verifiable node placement: {hostlist!r}")
        expected = tuple(_query(["scontrol", "show", "hostnames", hostlist]).splitlines())
        if (
            not expected
            or len(set(expected)) != len(expected)
            or any(not _HOST.fullmatch(name) for name in expected)
        ):
            raise SlurmError(f"invalid node expansion for job {job_id}: {expected!r}")
        output = _query(["scontrol", "show", "node", ",".join(expected), "--oneliner"])
        for line in output.splitlines():
            fields = dict(re.findall(r"(?:^|\s)(NodeName|State)=(\S+)", line))
            if set(fields) != {"NodeName", "State"}:
                raise SlurmError(f"incomplete node health record: {line!r}")
            reason = re.search(r"\bReason=(.*)", line)
            node = NodeState(fields["NodeName"], fields["State"], reason[1] if reason else "")
            nodes.append(node)
            flags = set(node.state.split("+"))
            if not flags & _AVAILABLE or flags & _UNHEALTHY:
                errors.append(f"{node.name}: state={node.state}, reason={node.reason}")
        names = [node.name for node in nodes]
        if len(set(names)) != len(names) or set(names) != set(expected):
            errors.append(f"node health coverage mismatch: expected={expected!r}, got={names!r}")
    except SlurmError as error:
        errors.append(str(error))
    return NodeHealth(job_id, datetime.now(UTC).isoformat(), expected, tuple(nodes), tuple(errors))
