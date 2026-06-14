#!/usr/bin/env python3
"""Probe and validate GB300 NVLink Fabric membership."""

from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SUMMARY_PREFIX = "__FABRIC_SUMMARY__ "
GPU_LINE = re.compile(r"^GPU [0-9]+:")
INVALID_FABRIC_VALUES = {"", "N/A", "Not Supported", "Unknown"}


@dataclass(frozen=True)
class FabricGpu:
    """Fabric identity and health fields for one GPU."""

    state: str
    status: str
    clique_id: str
    cluster_uuid: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def parse_nvidia_smi_query(output: str) -> list[FabricGpu]:
    """Extract every GPU's Fabric block from full ``nvidia-smi -q``."""
    rows: list[FabricGpu] = []
    current: dict[str, str] | None = None
    field_names = {
        "State": "state",
        "Status": "status",
        "CliqueId": "clique_id",
        "ClusterUUID": "cluster_uuid",
    }
    for line in output.splitlines():
        stripped = line.strip()
        if stripped == "Fabric":
            current = {}
            continue
        if current is None:
            continue
        name, separator, value = stripped.partition(":")
        name = name.replace(" ", "")
        if not separator or name not in field_names:
            continue
        current[field_names[name]] = value.strip()
        if name == "ClusterUUID":
            rows.append(
                FabricGpu(
                    state=current.get("state", ""),
                    status=current.get("status", ""),
                    clique_id=current.get("clique_id", ""),
                    cluster_uuid=current.get("cluster_uuid", ""),
                )
            )
            current = None
    return rows


def _valid_cluster_uuids(rows: list[FabricGpu]) -> list[str]:
    return sorted(
        {
            row.cluster_uuid
            for row in rows
            if row.cluster_uuid not in INVALID_FABRIC_VALUES
        }
    )


def node_summary(
    host: str,
    gpu_count: int,
    rows: list[FabricGpu],
) -> dict[str, Any]:
    """Build the machine-readable result for one physical node."""
    return {
        "host": host,
        "gpus": gpu_count,
        "fabric_rows": len(rows),
        "completed": sum(row.state == "Completed" for row in rows),
        "success": sum(row.status == "Success" for row in rows),
        "cluster_uuids": _valid_cluster_uuids(rows),
        "clique_ids": sorted(
            {
                row.clique_id
                for row in rows
                if row.clique_id not in INVALID_FABRIC_VALUES
            }
        ),
    }


def validate_node_summary(
    summary: dict[str, Any],
    expected_gpus: int,
) -> None:
    """Require every local GPU to be healthy in one Fabric domain."""
    problems: list[str] = []
    for field in ("gpus", "fabric_rows", "completed", "success"):
        if int(summary[field]) != expected_gpus:
            problems.append(
                f"{field}={summary[field]} expected={expected_gpus}"
            )
    cluster_uuids = list(summary["cluster_uuids"])
    if len(cluster_uuids) != 1:
        problems.append(
            "expected one non-empty ClusterUUID, got "
            f"{cluster_uuids!r}"
        )
    if problems:
        raise RuntimeError(
            f"{summary['host']} Fabric validation failed: "
            + "; ".join(problems)
        )


def validate_topology_summaries(
    summaries: list[dict[str, Any]],
    *,
    expected_nodes: int,
    expected_gpus_per_node: int,
) -> dict[str, Any]:
    """Require all physical nodes to report one shared ClusterUUID."""
    if len(summaries) != expected_nodes:
        raise RuntimeError(
            f"Fabric log has {len(summaries)} node summaries; "
            f"expected {expected_nodes}"
        )
    hosts = [str(summary["host"]) for summary in summaries]
    if len(set(hosts)) != expected_nodes:
        raise RuntimeError(f"Fabric log has duplicate hosts: {hosts!r}")
    for summary in summaries:
        validate_node_summary(summary, expected_gpus_per_node)
    cluster_uuids = sorted(
        {
            str(cluster_uuid)
            for summary in summaries
            for cluster_uuid in summary["cluster_uuids"]
        }
    )
    if len(cluster_uuids) != 1:
        raise RuntimeError(
            "The allocated nodes are not in one NVLink Fabric domain: "
            f"{cluster_uuids!r}"
        )
    return {
        "cluster_uuid": cluster_uuids[0],
        "hosts": sorted(hosts),
        "physical_nodes": expected_nodes,
        "gpus_per_node": expected_gpus_per_node,
        "active_gpu_count": expected_nodes * expected_gpus_per_node,
    }


def load_topology_summaries(path: Path) -> list[dict[str, Any]]:
    """Read prefixed JSON node summaries from a topology artifact."""
    summaries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        marker = line.find(SUMMARY_PREFIX)
        if marker < 0:
            continue
        summaries.append(
            json.loads(line[marker + len(SUMMARY_PREFIX) :])
        )
    return summaries


def run_node_probe(expected_gpus: int) -> int:
    """Run ``nvidia-smi`` once and emit a validated node summary."""
    gpu_list = subprocess.run(
        ["nvidia-smi", "-L"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    query = subprocess.run(
        ["nvidia-smi", "-q"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    host = socket.getfqdn()
    gpu_count = sum(
        bool(GPU_LINE.match(line))
        for line in gpu_list.splitlines()
    )
    rows = parse_nvidia_smi_query(query)
    summary = node_summary(host, gpu_count, rows)

    print(f"===== {host} =====")
    print(gpu_list.rstrip())
    for index, row in enumerate(rows):
        print(
            "__FABRIC_GPU__ "
            + json.dumps(
                {
                    "host": host,
                    "gpu_index": index,
                    **row.to_dict(),
                },
                sort_keys=True,
            )
        )
    print(SUMMARY_PREFIX + json.dumps(summary, sort_keys=True))
    validate_node_summary(summary, expected_gpus)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe = subparsers.add_parser("probe")
    probe.add_argument("--expected-gpus", type=int, required=True)

    validate = subparsers.add_parser("validate-log")
    validate.add_argument("path", type=Path)
    validate.add_argument("--expected-nodes", type=int, required=True)
    validate.add_argument(
        "--expected-gpus-per-node",
        type=int,
        required=True,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "probe":
            return run_node_probe(args.expected_gpus)
        summaries = load_topology_summaries(args.path)
        result = validate_topology_summaries(
            summaries,
            expected_nodes=args.expected_nodes,
            expected_gpus_per_node=args.expected_gpus_per_node,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"GB300 Fabric validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
