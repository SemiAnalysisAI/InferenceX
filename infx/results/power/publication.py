"""Build the required-power publication contract from retained measurement evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from infx.results.agentic.validate_power import validate_power
from infx.results.power.native_multinode import _identity
from infx.results.topology import Parallelism

SCHEMA_VERSION = 2


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _positive(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _artifact(path: Path, root: Path) -> dict[str, str]:
    if not path.is_file() or path.is_symlink() or not path.stat().st_size:
        raise ValueError(f"Missing required artifact: {path}")
    relative = path.resolve().relative_to(root.resolve())
    return {
        "path": relative.as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "validation_state": "valid",
    }


def _devices(audit: dict, sidecar: Path, *, multinode: bool) -> tuple[list[dict], list[Path]]:
    energies = audit.get("per_gpu_energy_j")
    if not isinstance(energies, dict) or not energies:
        raise ValueError(f"Missing per-device energy: {sidecar}")
    artifacts = []
    if multinode:
        if audit.get("telemetry_kind") == "native_multinode_smi":
            identities = {
                uuid: [node["node"], uuid]
                for node in audit["nodes"]
                for uuid in node["physical_gpu_ids"].values()
            }
        else:
            identities = {key: key.rsplit("/", 1) for key in energies}
        roles = audit.get("per_gpu_role", {})
        if any(len(parts) != 2 for parts in identities.values()):
            raise ValueError(f"Missing physical node/GPU identity: {sidecar}")
    else:
        node_path = sidecar.with_name("power_node.txt")
        node = node_path.read_text().strip()
        if not node:
            raise ValueError(f"Missing power node identity: {node_path}")
        identity_path = sidecar.with_name("gpu_metrics_identity.csv")
        vendor = "nvidia"
        if not identity_path.is_file():
            identity_path = sidecar.with_name("gpu_metrics_devices.json")
            vendor = "amd"
        physical = _identity(identity_path, vendor)
        identities = {key: [node, physical[key]] for key in energies}
        roles = dict.fromkeys(energies, "aggregate")
        artifacts = [node_path, identity_path]
    devices = []
    for key, energy in energies.items():
        role = roles.get(key)
        if role == "agg":
            role = "aggregate"
        if role not in {"aggregate", "prefill", "decode"} or not _positive(energy):
            raise ValueError(f"Invalid physical GPU role/energy: {key}")
        node, uuid = identities[key]
        if not node or not uuid:
            raise ValueError(f"Missing physical node/GPU identity: {key}")
        devices.append({"node": node, "gpu_uuid": uuid, "role": role, "energy_j": energy})
    if len({device["gpu_uuid"] for device in devices}) != len(devices):
        raise ValueError("Duplicate physical GPU identity")
    return devices, artifacts


def _validate_topology(row: dict, result: dict, *, multinode: bool) -> int:
    total = 0
    for role in ("prefill", "decode") if multinode else ("",):
        worker = row[role] if role else row
        count = worker["num-worker"] if role else 1
        layout = Parallelism(
            tp=worker["tp"],
            pp=worker.get("pp", 1),
            dcp_size=worker.get("dcp-size", 1),
            pcp_size=worker.get("pcp-size", 1),
            ep=worker.get("ep", 1),
        )
        gpus = layout.gpus_per_worker * count
        total += gpus
        prefix = f"{role}_" if role else ""
        if role and (
            result.get(f"{role}_num_workers") != count or result.get(f"num_{role}_gpu") != gpus
        ):
            raise ValueError(f"Required {role} worker/GPU count differs from aggregate")
        if role == "decode":
            layout = layout.for_decode(gpus)
        for field, expected in layout.fields(prefix).items():
            default = 1 if field.endswith(("pp", "pcp_size", "dcp_size")) else None
            if result.get(field, default) != expected:
                raise ValueError(f"Required topology {field} differs from aggregate")
        dp = result.get(f"{prefix}dp_attention", False)
        if dp not in (True, False, "true", "false") or (dp in (True, "true")) != worker.get(
            "dp-attn", False
        ):
            raise ValueError(f"Required topology {prefix}dp_attention differs from aggregate")
    return total


def _point(row: dict, result_path: Path, root: Path, *, multinode: bool) -> dict:
    result = _read(result_path)
    errors = validate_power(result_path, disagg=row.get("disagg", False))
    if errors:
        raise ValueError(f"Invalid required power in {result_path}: {errors}")
    agentic = row.get("scenario-type") == "agentic-coding"
    required_gpu_count = _validate_topology(row, result, multinode=multinode)
    artifact_name = result_path.relative_to(root).parts[0]
    suffix = artifact_name.removeprefix("bmk_agentic_" if agentic else "bmk_")
    evidence = root / (f"agentic_{suffix}" if agentic else f"power_audit_{suffix}")
    candidates = list(evidence.rglob("power_validation*.json"))
    if len(candidates) != 1:
        candidates = [
            path
            for path in candidates
            if Path(_read(path).get("benchmark_result", "")).name
            == result_path.name.removeprefix("agg_")
            or f"concurrency_{result['conc']}.json" in _read(path).get("benchmark_result", "")
            or f"conc_{result['conc']}" in path.parts
        ]
    if len(candidates) != 1:
        raise ValueError(f"Expected one power validation sidecar for {result_path}")
    sidecar = candidates[0]
    audit = _read(sidecar)
    window = audit.get("benchmark_window") or {}
    start, end = window.get("start_time_unix"), window.get("end_time_unix")
    if (
        audit.get("power_valid") is not True
        or not _positive(start)
        or not _positive(end)
        or end <= start
    ):
        raise ValueError(f"Invalid power validation/window: {sidecar}")
    devices, identity_files = _devices(audit, sidecar, multinode=multinode)
    if audit.get("observed_gpu_count") != len(devices) or audit.get("expected_gpu_count") != len(
        devices
    ):
        raise ValueError(f"Incomplete physical GPU coverage: {sidecar}")
    expected_gpus = result.get("num_gpus")
    if expected_gpus is None:
        expected_gpus = (
            result["num_prefill_gpu"] + result["num_decode_gpu"]
            if multinode
            else result["tp"] * result.get("pp", 1) * result.get("pcp_size", 1)
        )
    if expected_gpus != len(devices) or required_gpu_count != len(devices):
        raise ValueError(f"Aggregate GPU count differs from evidence: {result_path}")
    for role, count in (
        ("prefill", result.get("num_prefill_gpu")),
        ("decode", result.get("num_decode_gpu")),
    ):
        if row.get("disagg") and (
            not _positive(count) or sum(device["role"] == role for device in devices) != count
        ):
            raise ValueError(f"Missing {role} GPU evidence: {sidecar}")
        if row.get("disagg") and not math.isclose(
            sum(device["energy_j"] for device in devices if device["role"] == role),
            result[f"{role}_gpu_energy_j"],
            rel_tol=1e-4,
            abs_tol=0.01,
        ):
            raise ValueError(f"Aggregate {role} energy differs from device evidence: {result_path}")
    if multinode and len({device["node"] for device in devices}) != row.get("node-count"):
        raise ValueError(f"Physical node count differs from required topology: {result_path}")
    if not row.get("disagg") and any(device["role"] != "aggregate" for device in devices):
        raise ValueError(f"Unexpected split GPU roles: {sidecar}")
    if not math.isclose(
        sum(device["energy_j"] for device in devices),
        result["total_gpu_energy_j"],
        rel_tol=1e-4,
        abs_tol=0.01,
    ):
        raise ValueError(f"Aggregate energy differs from device evidence: {result_path}")
    if multinode:
        audit_root = root / f"power_audit_{suffix}"
        if audit.get("telemetry_kind") == "native_multinode_smi":
            telemetry = list(audit_root.rglob("gpu_metrics.csv"))
            if len(telemetry) != len(audit["nodes"]):
                raise ValueError(f"Missing native multinode telemetry: {audit_root}")
            telemetry = [
                path for csv in telemetry for path in csv.parent.rglob("*") if path.is_file()
            ]
        else:
            telemetry = list(audit_root.rglob("samples.csv"))
            if len(telemetry) != 1 or not telemetry[0].with_name("manifest.json").is_file():
                raise ValueError(f"Missing central multinode telemetry: {audit_root}")
            selected = audit.get("selected_window") or {}
            relative_window = selected.get("window_file")
            if not isinstance(relative_window, str):
                raise ValueError("Missing selected central measurement window")
            window_path = telemetry[0].parent / relative_window
            window_path.resolve().relative_to(telemetry[0].parent.resolve())
            central_window = _read(window_path)
            if (
                central_window.get("status") != "completed"
                or central_window.get("concurrency") != result["conc"]
                or central_window.get("benchmark_start_time_unix") != start
                or central_window.get("benchmark_end_time_unix") != end
            ):
                raise ValueError("Central measurement window differs from required point")
            telemetry = [path for path in telemetry[0].parent.rglob("*") if path.is_file()]
    else:
        telemetry = [sidecar.with_name("gpu_metrics.csv")]
    files = [result_path, sidecar, *identity_files, *telemetry]
    topology_fields = (
        "disagg",
        "is_multinode",
        "num_gpus",
        "num_prefill_gpu",
        "num_decode_gpu",
        "tp",
        "pp",
        "dcp_size",
        "pcp_size",
        "ep",
        "dp_attention",
        "prefill_tp",
        "prefill_pp",
        "prefill_dcp_size",
        "prefill_pcp_size",
        "prefill_dp_attention",
        "prefill_ep",
        "prefill_num_workers",
        "decode_tp",
        "decode_pp",
        "decode_dcp_size",
        "decode_pcp_size",
        "decode_dp_attention",
        "decode_ep",
        "decode_num_workers",
    )
    return {
        "config_key": row["exp-name"],
        "identity": {
            "model": result["infmax_model_prefix"],
            "hardware": result["hw"],
            "framework": result["framework"],
            "precision": result["precision"],
            "recipe_fingerprint": result["recipe_fingerprint"],
            "benchmark_type": "agentic_traces" if agentic else "single_turn",
            "isl": None if agentic else result["isl"],
            "osl": None if agentic else result["osl"],
            "concurrency": result["conc"],
        },
        "topology": {
            **{key: result[key] for key in topology_fields if key in result},
            "num_gpus": expected_gpus,
        },
        "measurement_window": {"start_time_unix": start, "end_time_unix": end},
        "devices": devices,
        "artifacts": [_artifact(path, root) for path in sorted(set(files))],
    }


def build_manifest(sweep: dict, artifacts: Path, *, publication: dict | None = None) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", str(sweep.get("head", ""))):
        raise ValueError("Invalid tested commit SHA")
    if any(type(sweep.get(key)) is not int or sweep[key] <= 0 for key in ("run-id", "run-attempt")):
        raise ValueError("Invalid source run identity")
    results = [(path, _read(path)) for path in sorted(artifacts.glob("bmk*/*.json"))]
    points = []
    seen = set()
    for topology in ("single_node", "multi_node"):
        for rows in sweep["matrix"].get(topology, {}).values():
            for row in rows:
                if row.get("require-power") is not True or row.get("eval-only") is True:
                    continue
                concs = row["conc"] if isinstance(row["conc"], list) else [row["conc"]]
                for conc in concs:
                    key = (row.get("recipe-fingerprint"), conc)
                    if not re.fullmatch(r"[0-9a-f]{64}", str(key[0])) or key in seen:
                        raise ValueError(f"Missing/duplicate recipe point identity: {key}")
                    seen.add(key)
                    matches = [
                        path
                        for path, result in results
                        if result.get("recipe_fingerprint") == key[0]
                        and result.get("conc") == conc
                        and result.get("infmax_model_prefix") == row["model-prefix"]
                        and result.get("hw") == row["runner"]
                        and result.get("framework") == row["framework"]
                        and result.get("precision") == row["precision"]
                        and result.get("disagg") is row.get("disagg", False)
                        and result.get("is_multinode") is (topology == "multi_node")
                    ]
                    if len(matches) != 1:
                        raise ValueError(
                            f"Expected one required result for {row['exp-name']} concurrency {conc}"
                        )
                    points.append(
                        _point(row, matches[0], artifacts, multinode=topology == "multi_node")
                    )
    if not points:
        raise ValueError("Required-power manifest has no required points")
    if publication is None:
        publication = {"mode": "incremental", "replacement_scope": []}
    if publication.get("mode") not in {"incremental", "replacement"} or not isinstance(
        publication.get("replacement_scope"), list
    ):
        raise ValueError("Invalid explicit publication policy")
    scopes = publication["replacement_scope"]
    if (publication["mode"] == "incremental" and scopes) or (
        publication["mode"] == "replacement" and not scopes
    ):
        raise ValueError("Replacement policy must identify the exact replacement scope")
    for scope in scopes:
        if (
            not isinstance(scope, dict)
            or not isinstance(scope.get("curve_scope"), str)
            or not scope["curve_scope"]
            or type(scope.get("previous_snapshot_workflow_run_id")) is not int
            or scope["previous_snapshot_workflow_run_id"] <= 0
            or scope["previous_snapshot_workflow_run_id"] > 2**53 - 1
            or not isinstance(scope.get("removed_point_identities"), list)
            or not scope["removed_point_identities"]
            or any(
                not isinstance(identity, str) or not identity
                for identity in scope["removed_point_identities"]
            )
            or len(set(scope["removed_point_identities"])) != len(scope["removed_point_identities"])
        ):
            raise ValueError("Replacement policy must name a snapshot and exact removed identities")
    if len({scope["curve_scope"] for scope in scopes}) != len(scopes):
        raise ValueError("Duplicate replacement curve scope")
    return {
        **sweep,
        "schema-version": SCHEMA_VERSION,
        "publication": publication,
        "points": points,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publication-policy", type=Path)
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--expected-run-id", type=int)
    parser.add_argument("--expected-head")
    parser.add_argument("--expected-run-attempt", type=int)
    args = parser.parse_args()
    if args.verify_existing and args.publication_policy:
        parser.error("A reused manifest's publication policy cannot be overwritten")
    sweep = _read(args.sweep)
    if args.expected_run_id is not None and sweep.get("run-id") != args.expected_run_id:
        raise ValueError("Required manifest source run does not match dispatch")
    if args.expected_head is not None and sweep.get("head") != args.expected_head:
        raise ValueError("Required manifest tested head does not match source")
    if args.expected_run_attempt is not None and (
        type(sweep.get("run-attempt")) is not int
        or not 0 < sweep["run-attempt"] <= args.expected_run_attempt
    ):
        raise ValueError("Required manifest attempt is outside the source run")
    if args.verify_existing and sweep.get("schema-version") != SCHEMA_VERSION:
        raise ValueError("Reused required-power manifest has an incompatible schema")
    publication = _read(args.publication_policy) if args.publication_policy else None
    if args.verify_existing:
        publication = sweep.get("publication")
    manifest = build_manifest(sweep, args.artifacts, publication=publication)
    if args.verify_existing and (
        manifest["points"] != sweep.get("points")
        or manifest["publication"] != sweep.get("publication")
    ):
        raise ValueError("Reused required-power manifest disagrees with retained evidence")
    args.output.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
