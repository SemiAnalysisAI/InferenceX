"""Tests for upstream-only srt-slurm custom benchmark metrics discovery."""

from pathlib import Path

import pytest
import yaml

from utils.agentic.srt_metrics_endpoints import (
    build_worker_metrics_urls,
    expand_slurm_nodelist,
    parse_worker_node_counts,
)


def test_expand_slurm_nodelist_preserves_order_and_zero_padding() -> None:
    assert expand_slurm_nodelist("gb300-[001-003,007],login9") == [
        "gb300-001",
        "gb300-002",
        "gb300-003",
        "gb300-007",
        "login9",
    ]


def test_expand_slurm_nodelist_supports_multiple_bracket_groups_and_steps() -> None:
    assert expand_slurm_nodelist("rack[1-2]-node[02-04:2]") == [
        "rack1-node02",
        "rack1-node04",
        "rack2-node02",
        "rack2-node04",
    ]


def test_build_worker_metrics_urls_uses_logical_leaders_and_process_offsets() -> None:
    assert build_worker_metrics_urls(
        nodelist="gb300-[00-10]",
        worker_node_counts=[2, 2, 2, 4],
        infra_node_count=1,
        system_port_base=7500,
    ) == [
        "http://gb300-01:7500/metrics",
        "http://gb300-03:7502/metrics",
        "http://gb300-05:7504/metrics",
        "http://gb300-07:7506/metrics",
    ]


def test_build_worker_metrics_urls_rejects_allocation_mismatch() -> None:
    with pytest.raises(ValueError, match="expanded 4 hosts, expected 5"):
        build_worker_metrics_urls(
            nodelist="gb300-[00-03]",
            worker_node_counts=[2, 2],
            infra_node_count=1,
            system_port_base=7500,
        )


def test_agentic_recipes_describe_their_upstream_worker_topology() -> None:
    recipe_directory = (
        Path(__file__).parents[2]
        / "benchmarks/multi_node/srt-slurm-recipes/sglang/deepseek-v4/agentic"
    )
    recipe_paths = list(recipe_directory.glob("*.yaml"))
    assert recipe_paths
    for recipe_path in recipe_paths:
        recipe = yaml.safe_load(recipe_path.read_text())
        resources = recipe["resources"]
        env = recipe["benchmark"]["env"]
        actual_counts = parse_worker_node_counts(env["AIPERF_SRT_WORKER_NODE_COUNTS"])

        if resources.get("agg_workers"):
            assert resources["agg_nodes"] % resources["agg_workers"] == 0
            nodes_per_worker = resources["agg_nodes"] // resources["agg_workers"]
            expected_counts = [nodes_per_worker] * resources["agg_workers"]
        else:
            assert resources["prefill_nodes"] % resources["prefill_workers"] == 0
            assert resources["decode_nodes"] % resources["decode_workers"] == 0
            prefill_nodes_per_worker = resources["prefill_nodes"] // resources["prefill_workers"]
            decode_nodes_per_worker = resources["decode_nodes"] // resources["decode_workers"]
            expected_counts = [prefill_nodes_per_worker] * resources["prefill_workers"]
            expected_counts += [decode_nodes_per_worker] * resources["decode_workers"]

        assert actual_counts == expected_counts, recipe_path.name
        assert env["AIPERF_SRT_INFRA_NODE_COUNT"] == "1"
        assert env["AIPERF_SRT_SYSTEM_PORT_BASE"] == "7500"


@pytest.mark.parametrize("value", ["", "2,0,1", "2,nope,1"])
def test_parse_worker_node_counts_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        parse_worker_node_counts(value)
