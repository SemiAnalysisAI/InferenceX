import json

import pytest

from bench_offline.gb300_fabric import (
    SUMMARY_PREFIX,
    load_topology_summaries,
    node_summary,
    parse_nvidia_smi_query,
    validate_node_summary,
    validate_topology_summaries,
)


def fabric_block(
    *,
    state="Completed",
    status="Success",
    clique_id="2",
    cluster_uuid="cluster-a",
):
    return f"""
    Fabric
        State                             : {state}
        Status                            : {status}
        CliqueId                          : {clique_id}
        ClusterUUID                       : {cluster_uuid}
        Health
            Bandwidth                     : Full
"""


def test_parse_full_nvidia_smi_query_extracts_only_fabric_blocks():
    output = (
        "State : unrelated\n"
        + fabric_block()
        + "\n"
        + fabric_block(clique_id="3")
    )
    rows = parse_nvidia_smi_query(output)
    assert [row.state for row in rows] == ["Completed", "Completed"]
    assert [row.status for row in rows] == ["Success", "Success"]
    assert [row.cluster_uuid for row in rows] == [
        "cluster-a",
        "cluster-a",
    ]


def test_node_summary_requires_every_gpu_and_one_cluster_uuid():
    rows = parse_nvidia_smi_query(fabric_block() * 4)
    summary = node_summary("node-1", 4, rows)
    validate_node_summary(summary, 4)
    bad_rows = parse_nvidia_smi_query(
        fabric_block() * 3
        + fabric_block(cluster_uuid="cluster-b")
    )
    with pytest.raises(RuntimeError, match="one non-empty ClusterUUID"):
        validate_node_summary(
            node_summary("node-1", 4, bad_rows),
            4,
        )


def test_topology_validation_requires_one_cluster_across_four_nodes():
    summaries = [
        node_summary(
            f"node-{index}",
            4,
            parse_nvidia_smi_query(fabric_block() * 4),
        )
        for index in range(4)
    ]
    result = validate_topology_summaries(
        summaries,
        expected_nodes=4,
        expected_gpus_per_node=4,
    )
    assert result["cluster_uuid"] == "cluster-a"
    assert result["active_gpu_count"] == 16

    summaries[-1]["cluster_uuids"] = ["cluster-b"]
    with pytest.raises(RuntimeError, match="one NVLink Fabric domain"):
        validate_topology_summaries(
            summaries,
            expected_nodes=4,
            expected_gpus_per_node=4,
        )


def test_load_topology_summaries_accepts_srun_prefixes(tmp_path):
    path = tmp_path / "topology.log"
    summary = {
        "host": "node-1",
        "gpus": 4,
        "fabric_rows": 4,
        "completed": 4,
        "success": 4,
        "cluster_uuids": ["cluster-a"],
        "clique_ids": ["2"],
    }
    path.write_text(
        "noise\n0: " + SUMMARY_PREFIX + json.dumps(summary) + "\n",
        encoding="utf-8",
    )
    assert load_topology_summaries(path) == [summary]
