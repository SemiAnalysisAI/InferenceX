#!/usr/bin/env python3
"""Aggregate synchronized GB300 TP8 replicas into one NVL72 result."""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from io_utils import read_json, write_json
from metrics import (
    aggregate_replicated_decode_rounds,
    huawei_scaled_local_batch_comparison,
)


RACK_PROFILE = "rack-tp8x9-mtp1"
REPLICA_PROFILE = "rack-tp8-mtp1-engine"
RACK_ACTIVE_GPUS = 72
RACK_PHYSICAL_NODES = 18
REPLICA_ACTIVE_GPUS = 8
REPLICA_COUNT = 9
RACK_GLOBAL_BATCH_SIZES = (72, 288, 576, 30960, 36864)
MTP_DRAFT_TOKENS = 1
LATENCY_ROUNDS_TO_SKIP = 8
MAX_MEASURED_START_SKEW_SECONDS = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replica-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--allocation-job-id", required=True)
    parser.add_argument("--slurm-nodes", required=True)
    parser.add_argument("--fabric-cluster-uuid", required=True)
    parser.add_argument("--fabric-clique-id", required=True)
    parser.add_argument("--rank-map-artifact", required=True)
    parser.add_argument("--topology-artifact", required=True)
    return parser.parse_args()


def parse_timestamp(value: str) -> datetime:
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError(f"Timestamp lacks a timezone: {value}")
    return timestamp


def combined_sequence_sha256(aggregates: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for aggregate in aggregates:
        digest.update(
            str(aggregate["output_sequence_sha256"]).encode("ascii")
        )
    return digest.hexdigest()


def discover_replica_results(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    paths = sorted(root.glob("r*/offline_result_*.json"))
    if len(paths) != REPLICA_COUNT:
        raise RuntimeError(
            f"Expected {REPLICA_COUNT} replica results under {root}, "
            f"found {len(paths)}"
        )
    return [(path, read_json(path)) for path in paths]


def validate_replicas(
    discovered: list[tuple[Path, dict[str, Any]]],
    *,
    rack_global_batch_size: int,
    fabric_cluster_uuid: str,
    fabric_clique_id: str,
) -> tuple[list[dict[str, Any]], int]:
    if rack_global_batch_size % REPLICA_COUNT != 0:
        raise ValueError(
            f"Rack GBS {rack_global_batch_size} is not divisible by "
            f"{REPLICA_COUNT}"
        )
    engine_global_batch = rack_global_batch_size // REPLICA_COUNT
    results: list[dict[str, Any]] = []
    seen_indices: list[int] = []
    for path, result in discovered:
        if result.get("status") != "success":
            raise RuntimeError(
                f"Replica result failed: {path}: "
                f"{result.get('error', 'unknown error')}"
            )
        benchmark = result.get("benchmark") or {}
        if benchmark.get("benchmark_profile") != REPLICA_PROFILE:
            raise RuntimeError(
                f"Replica profile mismatch in {path}: "
                f"{benchmark.get('benchmark_profile')!r}"
            )
        if int(benchmark.get("active_gpu_count", -1)) != REPLICA_ACTIVE_GPUS:
            raise RuntimeError(f"Replica GPU count mismatch in {path}")
        if int(benchmark.get("global_batch_size", -1)) != engine_global_batch:
            raise RuntimeError(f"Replica GBS mismatch in {path}")

        provenance = result.get("provenance") or {}
        if provenance.get("fabric_cluster_uuid") != fabric_cluster_uuid:
            raise RuntimeError(f"Replica fabric UUID mismatch in {path}")
        if str(provenance.get("fabric_clique_id")) != fabric_clique_id:
            raise RuntimeError(f"Replica fabric clique mismatch in {path}")

        synchronization = (
            ((result.get("final") or {}).get("runtime_environment") or {})
            .get("rack_synchronization")
            or {}
        )
        if synchronization.get("enabled") is not True:
            raise RuntimeError(f"Replica rack barrier was not enabled: {path}")
        if int(synchronization.get("replica_count", -1)) != REPLICA_COUNT:
            raise RuntimeError(f"Replica barrier count mismatch in {path}")
        seen_indices.append(int(synchronization["replica_index"]))
        results.append(result)

    if sorted(seen_indices) != list(range(REPLICA_COUNT)):
        raise RuntimeError(
            f"Replica barrier indices are not 0..{REPLICA_COUNT - 1}: "
            f"{sorted(seen_indices)}"
        )
    return results, engine_global_batch


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    discovered = discover_replica_results(args.replica_root)
    replicas, engine_global_batch = validate_replicas(
        discovered,
        rack_global_batch_size=args.global_batch_size,
        fabric_cluster_uuid=args.fabric_cluster_uuid,
        fabric_clique_id=args.fabric_clique_id,
    )
    replica_aggregates = [result["aggregate"] for result in replicas]
    aggregate = aggregate_replicated_decode_rounds(
        replica_aggregates,
        global_batch_size=args.global_batch_size,
        num_gpus=RACK_ACTIVE_GPUS,
        skip_rounds=LATENCY_ROUNDS_TO_SKIP,
        mtp_draft_tokens=MTP_DRAFT_TOKENS,
    )

    measured_passes = [
        (result.get("final") or {}).get("measured_pass") or {}
        for result in replicas
    ]
    measured_starts = [
        parse_timestamp(str(item["started_at"]))
        for item in measured_passes
    ]
    measured_finishes = [
        parse_timestamp(str(item["finished_at"]))
        for item in measured_passes
    ]
    measured_started_at = min(measured_starts)
    measured_finished_at = max(measured_finishes)
    rack_wall_seconds = (
        measured_finished_at - measured_started_at
    ).total_seconds()
    measured_start_skew_seconds = (
        max(measured_starts) - min(measured_starts)
    ).total_seconds()
    if rack_wall_seconds <= 0:
        raise RuntimeError(
            f"Invalid synchronized rack wall time: {rack_wall_seconds}"
        )
    if measured_start_skew_seconds > MAX_MEASURED_START_SKEW_SECONDS:
        raise RuntimeError(
            "Rack measured-pass barrier start skew exceeded "
            f"{MAX_MEASURED_START_SKEW_SECONDS:.1f}s: "
            f"{measured_start_skew_seconds:.3f}s"
        )
    total_output_tokens = sum(
        int(item["total_output_tokens"])
        for item in replica_aggregates
    )
    aggregate.update(
        {
            "request_samples": sum(
                int(item["request_samples"])
                for item in replica_aggregates
            ),
            "output_sequence_sha256": combined_sequence_sha256(
                replica_aggregates
            ),
            "wall_seconds": rack_wall_seconds,
            "wall_output_tput_per_gpu": (
                total_output_tokens
                / rack_wall_seconds
                / RACK_ACTIVE_GPUS
            ),
            "total_output_tokens": total_output_tokens,
            "request_perf_metrics_collected": False,
            "pass_count": 1,
            "measured_started_at": measured_started_at.isoformat(),
            "measured_finished_at": measured_finished_at.isoformat(),
            "measured_start_skew_seconds": measured_start_skew_seconds,
        }
    )

    first = replicas[0]
    first_benchmark = first["benchmark"]
    first_provenance = first["provenance"]
    replica_config = dict(first["config"])
    config = {
        **replica_config,
        "name": "pr-1689-rack-tp8x9-ep8x9-mtp1",
        "profile_key": RACK_PROFILE,
        "parallelism": "9xDEP8",
        "active_gpu_count": RACK_ACTIVE_GPUS,
        "replica_count": REPLICA_COUNT,
        "replica_active_gpu_count": REPLICA_ACTIVE_GPUS,
        "replica_global_batch_size": engine_global_batch,
        "allowed_global_batch_sizes": list(RACK_GLOBAL_BATCH_SIZES),
    }
    local_batch = args.global_batch_size // RACK_ACTIVE_GPUS
    comparable_to_huawei = local_batch in {1, 4, 8}
    result: dict[str, Any] = {
        "schema_version": 2,
        "status": "success",
        "started_at": min(
            str(result["started_at"]) for result in replicas
        ),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": {
            "mode": (
                "offline_scaled_huawei_fixed_local_batch_decode"
                if comparable_to_huawei
                else "offline_rack_decode_saturation"
            ),
            "execution": (
                "nine_fresh_tp8_engines_one_synchronized_measured_pass"
            ),
            "experiment_id": args.experiment_id,
            "engine": "TensorRT-LLM PyTorch backend",
            "request_path": "direct LLM.generate; no server or HTTP",
            "hardware": "GB300 NVL72",
            "hardware_profile": "gb300",
            "benchmark_profile": RACK_PROFILE,
            "renderer_hw": "gb300-nv",
            "world_size": RACK_ACTIVE_GPUS,
            "active_gpu_count": RACK_ACTIVE_GPUS,
            "physical_nodes": RACK_PHYSICAL_NODES,
            "gpus_per_node": 4,
            "is_multinode": True,
            "effective_parallelism": "9xDEP8",
            "replica_count": REPLICA_COUNT,
            "replica_world_size": REPLICA_ACTIVE_GPUS,
            "global_batch_size": args.global_batch_size,
            "concurrency": args.global_batch_size,
            "engine_global_batch_size": engine_global_batch,
            "local_batch_size": local_batch,
            "engine_max_batch_size": first_benchmark[
                "engine_max_batch_size"
            ],
            "max_num_tokens": first_benchmark.get("max_num_tokens"),
            "engine_warmup_max_tokens": first_benchmark.get(
                "engine_warmup_max_tokens"
            ),
            "input_tokens": first_benchmark["input_tokens"],
            "generated_output_tokens": first_benchmark[
                "generated_output_tokens"
            ],
            "mtp_max_draft_len": MTP_DRAFT_TOKENS,
            "max_seq_len": first_benchmark["max_seq_len"],
            "warmup_decode_rounds": 0,
            "measured_decode_rounds": aggregate[
                "measured_decode_rounds"
            ],
            "request_warmup_enabled": False,
            "perfect_router_enabled": False,
            "timing_source": aggregate["timing_source"],
            "latency_rounds_to_skip": LATENCY_ROUNDS_TO_SKIP,
            "headline_tpot": (
                "mean slowest-replica rank-0 TRT host_step_time across "
                "256 logical rack decode rounds after startup-round skip "
                "and upper-IQR filtering"
            ),
            "headline_throughput": (
                "rack_global_batch_size / decode_round_tpot_seconds / 72"
            ),
        },
        "config": config,
        "provenance": {
            **first_provenance,
            "slurm_job_id": args.allocation_job_id,
            "slurm_nodes": args.slurm_nodes,
            "rank_map_artifact": args.rank_map_artifact,
            "topology_artifact": args.topology_artifact,
            "fabric_cluster_uuid": args.fabric_cluster_uuid,
            "fabric_clique_id": args.fabric_clique_id,
            "reference_run": (
                "https://github.com/SemiAnalysisAI/InferenceX/actions/"
                "runs/27164980476/attempts/14"
            ),
        },
        "aggregate": aggregate,
        "rack": {
            "replica_count": REPLICA_COUNT,
            "replica_profile": REPLICA_PROFILE,
            "replica_active_gpu_count": REPLICA_ACTIVE_GPUS,
            "replica_global_batch_size": engine_global_batch,
            "round_alignment": (
                "measured passes share one start barrier; logical round "
                "latency is the maximum same-index TP8 host step"
            ),
            "replicas": [
                {
                    "index": index,
                    "source": str(path),
                    "experiment_id": result["benchmark"][
                        "experiment_id"
                    ],
                    "nodes": result["provenance"].get("slurm_nodes"),
                    "decode_round_tpot_ms": result["aggregate"][
                        "decode_round_tpot_ms"
                    ],
                    "output_tput_per_gpu": result["aggregate"][
                        "output_tput_per_gpu"
                    ],
                }
                for index, ((path, _), result) in enumerate(
                    zip(discovered, replicas, strict=True)
                )
            ],
        },
    }
    if comparable_to_huawei:
        result["huawei"] = huawei_scaled_local_batch_comparison(
            args.global_batch_size,
            aggregate,
            hardware_key="gb300-rack",
            hardware_label="GB300 NVL72",
        )
        result["benchmark"]["huawei_method_match"] = {
            "fixed_global_batch": True,
            "scaled_to_same_local_batch": True,
            "prefill_decode_barrier_required": True,
            "measured_decode_rounds": aggregate[
                "measured_decode_rounds"
            ],
            "upper_iqr_outlier_filter": True,
            "mtp_yield_reported_separately": True,
            "mtp_depth_match": False,
        }
    return result


def main() -> int:
    args = parse_args()
    write_json(args.output, build_result(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
