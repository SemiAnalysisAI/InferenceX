import json
from types import SimpleNamespace

import pytest

from aggregate_rack import build_result


def replica_result(index: int) -> dict:
    return {
        "schema_version": 2,
        "status": "success",
        "started_at": "2026-06-15T00:00:00+00:00",
        "benchmark": {
            "experiment_id": f"rack-test-replica{index:02d}",
            "benchmark_profile": "rack-tp8-mtp1-engine",
            "active_gpu_count": 8,
            "global_batch_size": 8,
            "engine_max_batch_size": 512,
            "max_num_tokens": 32768,
            "engine_warmup_max_tokens": 32768,
            "input_tokens": 8192,
            "generated_output_tokens": 1024,
            "max_seq_len": 9256,
        },
        "config": {
            "name": "replica",
            "profile_key": "rack-tp8-mtp1-engine",
            "parallelism": "DEP8",
            "active_gpu_count": 8,
            "tensor_parallel_size": 8,
            "moe_expert_parallel_size": 8,
        },
        "provenance": {
            "image": "test-image",
            "git_revision": "deadbeef",
            "fabric_cluster_uuid": "cluster",
            "fabric_clique_id": "7",
            "slurm_nodes": f"node{index * 2},node{index * 2 + 1}",
        },
        "aggregate": {
            "selected_round_latencies_ms": [20.0 + index] * 256,
            "measured_decode_rounds": 256,
            "raw_proposed_draft_tokens": 2048,
            "raw_accepted_draft_tokens": 1024,
            "raw_generation_slots": 2048,
            "decode_round_tpot_ms": 20.0 + index,
            "output_tput_per_gpu": 100.0,
            "observed_tokens_per_step": 1.5,
            "schedule_validation": {
                "selected_first_iter": 2,
                "selected_last_iter": 257,
            },
            "request_samples": 8,
            "output_sequence_sha256": f"{index:064x}",
            "total_output_tokens": 8192,
        },
        "final": {
            "runtime_environment": {
                "rack_synchronization": {
                    "enabled": True,
                    "replica_count": 9,
                    "replica_index": index,
                }
            },
            "measured_pass": {
                "started_at": "2026-06-15T01:00:00+00:00",
                "finished_at": "2026-06-15T01:01:00+00:00",
            },
        },
    }


def test_build_rack_result_publishes_one_nvl72_row(tmp_path):
    replica_root = tmp_path / "replicas"
    for index in range(9):
        directory = replica_root / f"r{index:02d}"
        directory.mkdir(parents=True)
        (directory / f"offline_result_replica{index:02d}.json").write_text(
            json.dumps(replica_result(index)),
            encoding="utf-8",
        )

    result = build_result(
        SimpleNamespace(
            replica_root=replica_root,
            output=tmp_path / "rack.json",
            experiment_id="rack-tp8x9-mtp1-gbs72",
            global_batch_size=72,
            allocation_job_id="123",
            slurm_nodes=" ".join(f"node{index}" for index in range(18)),
            fabric_cluster_uuid="cluster",
            fabric_clique_id="7",
            rank_map_artifact="rank-map.tsv",
            topology_artifact="topology.log",
        )
    )

    assert result["status"] == "success"
    assert result["benchmark"]["active_gpu_count"] == 72
    assert result["benchmark"]["replica_count"] == 9
    assert result["benchmark"]["local_batch_size"] == 1
    assert result["benchmark"]["max_num_tokens"] == 32768
    assert result["config"]["allowed_global_batch_sizes"] == [
        72,
        288,
        576,
        30960,
        36864,
    ]
    assert result["aggregate"]["decode_round_tpot_ms"] == pytest.approx(28.0)
    assert result["aggregate"]["request_samples"] == 72
    assert result["aggregate"]["wall_seconds"] == pytest.approx(60.0)
    assert result["aggregate"]["measured_start_skew_seconds"] == 0.0
    assert result["huawei"]["reference_global_batch_size"] == 16


def test_build_rack_result_rejects_unsynchronized_measured_pass(tmp_path):
    replica_root = tmp_path / "replicas"
    for index in range(9):
        directory = replica_root / f"r{index:02d}"
        directory.mkdir(parents=True)
        result = replica_result(index)
        if index == 8:
            result["final"]["measured_pass"]["started_at"] = (
                "2026-06-15T01:00:11+00:00"
            )
        (directory / f"offline_result_replica{index:02d}.json").write_text(
            json.dumps(result),
            encoding="utf-8",
        )

    with pytest.raises(
        RuntimeError,
        match="barrier start skew exceeded",
    ):
        build_result(
            SimpleNamespace(
                replica_root=replica_root,
                output=tmp_path / "rack.json",
                experiment_id="rack-tp8x9-mtp1-gbs72",
                global_batch_size=72,
                allocation_job_id="123",
                slurm_nodes=" ".join(
                    f"node{index}" for index in range(18)
                ),
                fabric_cluster_uuid="cluster",
                fabric_clique_id="7",
                rank_map_artifact="rank-map.tsv",
                topology_artifact="topology.log",
            )
        )
