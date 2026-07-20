#!/usr/bin/env python3
"""Discover logical srt-slurm worker metrics endpoints for custom benchmarks."""

from __future__ import annotations

import argparse
import re


_NUMERIC_RANGE = re.compile(r"^(\d+)-(\d+)(?::(\d+))?$")


def _split_top_level(value: str) -> list[str]:
    """Split a Slurm hostlist on commas outside bracket expressions."""
    parts: list[str] = []
    start = 0
    depth = 0
    for index, character in enumerate(value):
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth < 0:
                raise ValueError(f"unmatched closing bracket in Slurm nodelist: {value!r}")
        elif character == "," and depth == 0:
            parts.append(value[start:index])
            start = index + 1
    if depth != 0:
        raise ValueError(f"unmatched opening bracket in Slurm nodelist: {value!r}")
    parts.append(value[start:])
    if any(not part for part in parts):
        raise ValueError(f"empty component in Slurm nodelist: {value!r}")
    return parts


def _expand_bracket_values(value: str) -> list[str]:
    """Expand the comma-separated numeric values inside one bracket pair."""
    expanded: list[str] = []
    for component in value.split(","):
        match = _NUMERIC_RANGE.fullmatch(component)
        if match is None:
            if not component.isdigit():
                raise ValueError(f"unsupported Slurm bracket component: {component!r}")
            expanded.append(component)
            continue

        start_text, end_text, step_text = match.groups()
        start = int(start_text)
        end = int(end_text)
        step = int(step_text or "1")
        if step == 0:
            raise ValueError(f"Slurm range step must be positive: {component!r}")
        direction = 1 if end >= start else -1
        width = max(len(start_text), len(end_text))
        stop = end + direction
        expanded.extend(f"{number:0{width}d}" for number in range(start, stop, direction * step))
    return expanded


def _expand_component(component: str) -> list[str]:
    """Recursively expand all bracket expressions in one hostlist component."""
    opening = component.find("[")
    if opening < 0:
        if "]" in component:
            raise ValueError(f"unmatched closing bracket in Slurm nodelist component: {component!r}")
        return [component]

    depth = 0
    closing = -1
    for index in range(opening, len(component)):
        if component[index] == "[":
            depth += 1
        elif component[index] == "]":
            depth -= 1
            if depth == 0:
                closing = index
                break
    if closing < 0:
        raise ValueError(f"unmatched opening bracket in Slurm nodelist component: {component!r}")

    prefix = component[:opening]
    values = _expand_bracket_values(component[opening + 1 : closing])
    suffixes = _expand_component(component[closing + 1 :])
    return [f"{prefix}{value}{suffix}" for value in values for suffix in suffixes]


def expand_slurm_nodelist(nodelist: str) -> list[str]:
    """Expand a Slurm nodelist without relying on ``scontrol`` in the container."""
    if not nodelist:
        raise ValueError("Slurm nodelist must not be empty")
    hosts: list[str] = []
    for component in _split_top_level(nodelist):
        hosts.extend(_expand_component(component))
    return hosts


def parse_worker_node_counts(value: str) -> list[int]:
    """Parse comma-separated physical node counts for logical workers."""
    try:
        counts = [int(component) for component in value.split(",")]
    except ValueError as error:
        raise ValueError(f"worker node counts must be comma-separated integers: {value!r}") from error
    if not counts or any(count < 1 for count in counts):
        raise ValueError(f"every logical worker must use at least one node: {value!r}")
    return counts


def build_worker_metrics_urls(
    nodelist: str,
    worker_node_counts: list[int],
    infra_node_count: int,
    system_port_base: int,
) -> list[str]:
    """Map logical worker leaders to upstream srt-slurm system ports."""
    if infra_node_count < 0:
        raise ValueError("infra node count must not be negative")
    if system_port_base < 1 or system_port_base > 65535:
        raise ValueError(f"invalid system port base: {system_port_base}")

    allocated_hosts = expand_slurm_nodelist(nodelist)
    expected_nodes = infra_node_count + sum(worker_node_counts)
    if len(allocated_hosts) != expected_nodes:
        raise ValueError(
            "Slurm allocation does not match the custom benchmark topology: "
            f"expanded {len(allocated_hosts)} hosts, expected {expected_nodes} "
            f"({infra_node_count} infra + {sum(worker_node_counts)} worker nodes)"
        )

    worker_hosts = allocated_hosts[infra_node_count:]
    urls: list[str] = []
    process_offset = 0
    for node_count in worker_node_counts:
        port = system_port_base + process_offset
        if port > 65535:
            raise ValueError(f"derived system port exceeds 65535: {port}")
        urls.append(f"http://{worker_hosts[process_offset]}:{port}/metrics")
        process_offset += node_count
    return urls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodelist", required=True)
    parser.add_argument("--worker-node-counts", required=True)
    parser.add_argument("--infra-node-count", required=True, type=int)
    parser.add_argument("--system-port-base", required=True, type=int)
    args = parser.parse_args()

    urls = build_worker_metrics_urls(
        nodelist=args.nodelist,
        worker_node_counts=parse_worker_node_counts(args.worker_node_counts),
        infra_node_count=args.infra_node_count,
        system_port_base=args.system_port_base,
    )
    print(",".join(urls))


if __name__ == "__main__":
    main()
