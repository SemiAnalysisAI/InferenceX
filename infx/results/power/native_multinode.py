"""Validate native SMI traces from each node of one fixed-sequence deployment.

This format is owned by InferenceX. It does not claim the srt-slurm/DCGM wire
contract. The launcher records real serving-device membership and synchronized
host clocks; the normal result processor binds every trace to the client window.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import socket
import time
from datetime import timezone
from pathlib import Path

from . import ALL_POWER_METRIC_KEYS
from .common import (
    _load_benchmark_data, _write_json_atomic, benchmark_window_payload,
    patch_power_metrics,
)
from .single_node import _derived_metrics, _detect_columns, _parse_timestamp, integrate_power


def _identity(path: Path, vendor: str) -> dict[str, str]:
    """Map the vendor enumeration index to a physical UUID; never use PCI alone."""
    if vendor == "nvidia":
        with path.open(newline="") as stream:
            rows = []
            for row in csv.DictReader(stream, skipinitialspace=True):
                if any(not isinstance(k, str) or not isinstance(v, str) for k, v in row.items()):
                    raise ValueError("invalid_device_identity")
                rows.append({k.strip().lower(): v.strip() for k, v in row.items()})
    elif vendor == "amd":
        payload = json.loads(path.read_text())
        rows = []

        def visit(value: object) -> None:
            if isinstance(value, list):
                for item in value:
                    visit(item)
            elif isinstance(value, dict):
                row = {str(k).lower(): v for k, v in value.items()}
                if "gpu" in row and "uuid" in row:
                    rows.append(row)
                else:
                    for item in value.values():
                        visit(item)

        visit(payload)
    else:
        raise ValueError("unsupported_native_vendor")
    identities = {}
    for row in rows:
        index = str(row.get("index", row.get("gpu", ""))).strip()
        uuid = str(row.get("uuid", "")).strip()
        if not index.isdigit() or not uuid or uuid.lower() in {"n/a", "none", "null"}:
            raise ValueError("invalid_device_identity")
        if index in identities or uuid in identities.values():
            raise ValueError("duplicate_device_identity")
        identities[index] = uuid
    if not identities:
        raise ValueError("device_identity_missing")
    return identities


def record_begin(directory: Path, *, vendor: str, node: str, rank: int,
                 role: str, gpu_indices: list[int], num_nodes: int, job_id: str,
                 clock_synchronized: bool, revision: str) -> None:
    if (role not in {"prefill", "decode", "aggregate"} or rank < 0 or
            num_nodes <= rank or not gpu_indices or min(gpu_indices) < 0 or
            len(set(gpu_indices)) != len(gpu_indices)):
        raise ValueError("invalid_native_topology")
    directory.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(directory / "manifest.json", {
        "schema_version": 1, "collector": "inferencex-native-smi",
        "vendor": vendor, "node": node, "rank": rank, "role": role,
        "selected_gpu_indices": gpu_indices, "expected_num_nodes": num_nodes,
        "job_id": job_id, "collector_revision": revision,
        "clock_source": "utc_ntp", "clock_synchronized": clock_synchronized,
        "clock_observation": "timedatectl NTPSynchronized on serving host; no measured clock offset",
        "collection_start_unix": time.time(), "lifecycle": "collecting",
    })


def record_end(directory: Path, *, collector_exit_code: int) -> None:
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest.update(collection_end_unix=time.time(),
                    collector_exit_code=collector_exit_code,
                    lifecycle="complete" if collector_exit_code == 0 else "failed")
    _write_json_atomic(directory / "manifest.json", manifest)


def run(power_dir: Path, bench_result: Path, agg_result: Path, *,
        expected_prefill_gpus: int, expected_decode_gpus: int, expected_aggregate_gpus: int = 0,
        validation_result: Path | None = None, require_power: bool = False) -> int:
    validation_result = validation_result or bench_result.with_name(
        f"power_validation_{bench_result.stem}.json")
    benchmark, reasons = _load_benchmark_data(bench_result)
    expected_gpus = expected_prefill_gpus + expected_decode_gpus + expected_aggregate_gpus
    if expected_aggregate_gpus and (expected_prefill_gpus or expected_decode_gpus):
        reasons.append("native_mixed_aggregate_role_topology")
    roles: dict[str, list[str]] = {"prefill": [], "decode": [], "aggregate": []}
    receipts = []
    node_errors = []
    samples = []
    ranks = []
    nodes = []
    jobs = set()
    revisions = set()
    expected_nodes = set()
    paths = sorted(power_dir.glob("node-*/manifest.json"))
    if not paths:
        reasons.append("native_manifests_missing")
    if expected_gpus <= 0 or min(expected_prefill_gpus, expected_decode_gpus, expected_aggregate_gpus) < 0:
        reasons.append("invalid_expected_gpu_count")
    for path in paths:
        try:
            manifest = json.loads(path.read_text())
            rank = manifest["rank"]
            role = manifest["role"]
            if (manifest.get("schema_version") != 1 or
                    manifest.get("collector") != "inferencex-native-smi" or
                    type(rank) is not int or rank < 0 or role not in roles or
                    path.parent.name != f"node-{rank}"):
                raise ValueError("invalid_native_manifest")
            if (not isinstance(manifest.get("node"), str) or not manifest["node"] or
                    not isinstance(manifest.get("job_id"), str) or
                    not isinstance(manifest.get("collector_revision"), str) or
                    type(manifest.get("expected_num_nodes")) is not int or
                    manifest["expected_num_nodes"] <= 0):
                raise ValueError("invalid_native_run_identity")
            ranks.append(rank)
            nodes.append(manifest["node"])
            expected_nodes.add(manifest["expected_num_nodes"])
            jobs.add(manifest["job_id"])
            revisions.add(manifest["collector_revision"])
            if (manifest.get("lifecycle") != "complete" or
                    manifest.get("collector_exit_code") != 0):
                reasons.append("native_collector_incomplete")
            if (manifest.get("clock_source") != "utc_ntp" or
                    manifest.get("clock_synchronized") is not True):
                reasons.append("native_clock_not_synchronized")
            start, end = manifest["collection_start_unix"], manifest["collection_end_unix"]
            if (not all(type(x) in (int, float) and math.isfinite(x) for x in (start, end))
                    or end <= start or (benchmark is not None and
                    (start > benchmark.start_unix or end < benchmark.end_unix))):
                reasons.append("native_collection_window_mismatch")
            selected = manifest["selected_gpu_indices"]
            if (not isinstance(selected, list) or not selected or
                    any(type(i) is not int or i < 0 for i in selected) or
                    len(set(selected)) != len(selected)):
                raise ValueError("invalid_native_gpu_selection")
            suffix = "csv" if manifest["vendor"] == "nvidia" else "json"
            stem = "gpu_metrics_identity" if suffix == "csv" else "gpu_metrics_devices"
            first_path = path.parent / f"{stem}.{suffix}"
            last_path = path.parent / f"{stem}_end.{suffix}"
            first = _identity(first_path, manifest["vendor"])
            last = _identity(last_path, manifest["vendor"])
            selected_ids = {str(i): first[str(i)] for i in selected}
            if any(last.get(i) != uuid for i, uuid in selected_ids.items()):
                reasons.append("native_device_identity_changed")
            previous = {uuid for members in roles.values() for uuid in members}
            if previous.intersection(selected_ids.values()):
                reasons.append("native_duplicate_physical_gpu")
            roles[role].extend(selected_ids.values())
            csv_path = path.parent / "gpu_metrics.csv"
            with csv_path.open(newline="") as stream:
                reader = csv.DictReader(stream, skipinitialspace=True)
                reader.fieldnames = [c.strip() for c in (reader.fieldnames or [])]
                t_col, p_col, g_col = _detect_columns(reader.fieldnames)
                if not all((t_col, p_col, g_col)):
                    raise ValueError("native_telemetry_columns_missing")
                for row in reader:
                    gpu = (row.get(g_col) or "").strip()
                    if gpu not in first:
                        reasons.append("native_unknown_device_index")
                        continue
                    if gpu not in selected_ids:
                        continue
                    timestamp = _parse_timestamp((row.get(t_col) or ""), naive_timezone=timezone.utc)
                    # Invalid rows are retained for the common validator to reject.
                    samples.append((timestamp, selected_ids[gpu], row.get(p_col) or ""))
            receipts.append({**manifest, "physical_gpu_ids": selected_ids,
                             "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "telemetry_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                             "identity_sha256": hashlib.sha256(first_path.read_bytes()).hexdigest(),
                             "identity_end_sha256": hashlib.sha256(last_path.read_bytes()).hexdigest()})
        except (OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
            reasons.append("native_node_invalid")
            node_errors.append({"node": path.parent.name, "type": type(exc).__name__, "detail": str(exc)})
    if (len(expected_nodes) != 1 or not expected_nodes or
            type(next(iter(expected_nodes), None)) is not int or
            set(ranks) != set(range(next(iter(expected_nodes), 0))) or
            len(set(nodes)) != len(nodes) or len(ranks) != len(set(ranks))):
        reasons.append("native_node_topology_mismatch")
    if len(jobs) != 1 or "" in jobs or len(revisions) != 1 or "" in revisions:
        reasons.append("native_run_identity_mismatch")
    if expected_aggregate_gpus:
        if roles["prefill"] or roles["decode"] or len(roles["aggregate"]) != expected_aggregate_gpus:
            reasons.append("native_role_gpu_count_mismatch")
    elif (roles["aggregate"] or len(roles["prefill"]) != expected_prefill_gpus or
          len(roles["decode"]) != expected_decode_gpus):
        reasons.append("native_role_gpu_count_mismatch")
    combined_path = validation_result.with_name(f"{validation_result.stem}_native.csv")
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with combined_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp", "gpu", "power"])
        writer.writerows(samples)
    integration = None
    if benchmark is not None:
        integration = integrate_power(combined_path, start_unix=benchmark.start_unix,
                                      end_unix=benchmark.end_unix, expected_num_gpus=expected_gpus)
        reasons.extend(integration.invalid_reasons)
    metrics = {}
    if not reasons and integration is not None and benchmark is not None:
        metrics = _derived_metrics(integration, benchmark)
        if roles["prefill"]:
            prefill = sum(integration.per_gpu_energy_j[uuid] for uuid in roles["prefill"])
            metrics.update(prefill_gpu_energy_j=prefill,
                           prefill_avg_power_w=prefill / benchmark.integration_duration_s / expected_prefill_gpus,
                           prefill_joules_per_input_token=prefill / benchmark.total_input_tokens)
        if roles["decode"]:
            decode = sum(integration.per_gpu_energy_j[uuid] for uuid in roles["decode"])
            metrics.update(decode_gpu_energy_j=decode,
                           decode_avg_power_w=decode / benchmark.integration_duration_s / expected_decode_gpus,
                           decode_joules_per_output_token=decode / benchmark.total_output_tokens)
    valid = not reasons
    try:
        patch_power_metrics(agg_result, metric_keys=ALL_POWER_METRIC_KEYS, power_valid=valid, metrics=metrics)
    except (OSError, ValueError):
        reasons.append("aggregate_result_unwritable")
        valid, metrics = False, {}
    audit = {"schema_version": 1, "telemetry_kind": "native_multinode_smi",
             "power_valid": valid, "reasons": list(dict.fromkeys(reasons)),
             "benchmark_result": str(bench_result),
             "benchmark_result_sha256": hashlib.sha256(bench_result.read_bytes()).hexdigest() if bench_result.is_file() else None,
             "benchmark_window": benchmark_window_payload(benchmark),
             "expected_gpu_count": expected_gpus, "nodes": receipts,
             "node_errors": node_errors,
             "observed_gpu_count": integration.observed_num_gpus if integration else 0,
             "per_gpu_role": {uuid: role for role, uuids in roles.items() for uuid in uuids},
             "per_gpu_sample_counts": integration.per_gpu_sample_counts if integration else {},
             "boundary_degenerate_rows": integration.boundary_degenerate_rows if integration else {},
             "per_gpu_max_sample_gap_s": integration.per_gpu_max_sample_gap_s if integration else {},
             "producer": {"name": "inferencex-native-smi", "revisions": sorted(revisions),
                          "producer_git_commit": next(iter(revisions)) if len(revisions) == 1 else None},
             "integration_method": "per_device_trapezoidal_with_linear_boundary_interpolation",
             "power_percentile_method": "time_weighted_synchronized_total_piecewise_linear",
             "metrics": metrics}
    _write_json_atomic(validation_result, audit)
    return int(require_power and not valid)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    begin = sub.add_parser("begin")
    begin.add_argument("--directory", type=Path, required=True)
    begin.add_argument("--vendor", choices=("amd", "nvidia"), required=True)
    begin.add_argument("--node", default=os.environ.get("POWERX_NODE_NAME", socket.gethostname()))
    begin.add_argument("--rank", type=int, required=True)
    begin.add_argument("--role", choices=("prefill", "decode", "aggregate"), required=True)
    begin.add_argument("--gpu-indices", required=True)
    begin.add_argument("--num-nodes", type=int, required=True)
    begin.add_argument("--job-id", default=os.environ.get("SLURM_JOB_ID", ""))
    begin.add_argument("--revision", default=os.environ.get("POWERX_COLLECTOR_REVISION", ""))
    begin.add_argument("--clock-synchronized", choices=("true", "false"), default="false")
    end = sub.add_parser("end")
    end.add_argument("--directory", type=Path, required=True)
    end.add_argument("--collector-exit-code", type=int, required=True)
    args = vars(parser.parse_args())
    action = args.pop("action")
    if action == "begin":
        args["gpu_indices"] = [int(i) for i in args["gpu_indices"].split(",")]
        args["clock_synchronized"] = args["clock_synchronized"] == "true"
        record_begin(**args)
    else:
        record_end(**args)


if __name__ == "__main__":
    main()
