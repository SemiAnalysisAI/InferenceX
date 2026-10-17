"""Build the required-power publication contract from retained measurement evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from infx.results.topology import Parallelism

SCHEMA_VERSION = 2


@dataclass(frozen=True)
class SourceIdentity:
    run_id: int
    run_attempt: int
    head_sha: str

    @classmethod
    def parse(cls, value: dict) -> SourceIdentity:
        if not re.fullmatch(r"[0-9a-f]{40}", str(value.get("head", ""))):
            raise ValueError("Invalid tested commit SHA")
        if any(
            type(value.get(key)) is not int or not 0 < value[key] <= 2**53 - 1
            for key in ("run-id", "run-attempt")
        ):
            raise ValueError("Invalid source run identity")
        return cls(value["run-id"], value["run-attempt"], value["head"])


@dataclass(frozen=True)
class PointKey:
    fingerprint: str
    concurrency: int
    benchmark_type: str
    isl: int | None
    osl: int | None


def required_points(matrix: dict) -> list[tuple[dict, bool, PointKey]]:
    points = []
    seen = set()
    for topology in ("single_node", "multi_node"):
        for scenario, rows in matrix.get(topology, {}).items():
            for row in rows:
                if row.get("require-power") is not True or row.get("eval-only") is True:
                    continue
                agentic = row.get("scenario-type") == "agentic-coding"
                if agentic != (scenario == "agentic"):
                    raise ValueError("Required point scenario differs from matrix bucket")
                isl, osl = (None, None) if agentic else (row["isl"], row["osl"])
                if not agentic and any(type(n) is not int or n <= 0 for n in (isl, osl)):
                    raise ValueError("Invalid fixed sequence lengths")
                concs = row["conc"] if isinstance(row["conc"], list) else [row["conc"]]
                if not concs:
                    raise ValueError("Empty required concurrency list")
                for conc in concs:
                    key = PointKey(
                        row.get("recipe-fingerprint"),
                        conc,
                        "agentic_traces" if agentic else "single_turn",
                        isl,
                        osl,
                    )
                    if (
                        not re.fullmatch(r"[0-9a-f]{64}", str(key.fingerprint))
                        or type(conc) is not int
                        or conc <= 0
                        or key in seen
                    ):
                        raise ValueError(f"Missing/duplicate recipe point identity: {key}")
                    seen.add(key)
                    points.append((row, topology == "multi_node", key))
    return points


def compare_required_scope(source: dict, merged: dict) -> None:
    def scope(matrix: dict) -> dict[PointKey, dict]:
        return {
            key: {
                name: value
                for name, value in row.items()
                if name not in {"priority", "queue-token", "run-eval", "eval-conc"}
            }
            for row, _, key in required_points(matrix)
        }

    if scope(source) != scope(merged):
        raise ValueError("Reused required-power scope differs from the merged plan")


def _legacy_identity(path: Path) -> dict[str, str]:
    if path.suffix == ".csv":
        with path.open(newline="") as stream:
            rows = [
                {k.strip().lower(): v.strip() for k, v in row.items()}
                for row in csv.DictReader(stream)
            ]
        pairs = [(row["index"], row["uuid"]) for row in rows]
    else:
        pairs = []

        def visit(value: Any) -> None:
            if isinstance(value, list):
                for child in value:
                    visit(child)
            elif isinstance(value, dict):
                if "gpu" in value and "uuid" in value:
                    pairs.append((str(value["gpu"]), value["uuid"]))
                else:
                    for child in value.values():
                        visit(child)

        visit(_read(path))
    if (
        not pairs
        or any(
            not str(index).isdigit()
            or not isinstance(uuid, str)
            or not uuid.strip()
            or uuid.lower() in {"n/a", "none", "null"}
            for index, uuid in pairs
        )
        or len(dict(pairs)) != len(pairs)
        or len({u for _, u in pairs}) != len(pairs)
    ):
        raise ValueError("Invalid physical GPU identity")
    return dict(pairs)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _positive(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _artifact(path: Path, root: Path) -> dict[str, str]:
    relative = path.absolute().relative_to(root.absolute())
    candidate = root.absolute()
    for part in relative.parts:
        candidate /= part
        if candidate.is_symlink():
            raise ValueError(f"Symlink in required artifact path: {path}")
    if not path.is_file() or not path.stat().st_size:
        raise ValueError(f"Missing required artifact: {path}")
    relative = path.resolve().relative_to(root.resolve())
    return {
        "path": relative.as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "validation_state": "valid",
    }


def _source_receipt(evidence: Path, source: SourceIdentity) -> tuple[Path, int]:
    path = evidence / "power_source.json"
    receipt = _read(path)
    measured = SourceIdentity.parse(receipt)
    if (
        measured.run_id != source.run_id
        or measured.head_sha != source.head_sha
        or measured.run_attempt > source.run_attempt
    ):
        raise ValueError("Required artifact source run, head or attempt differs from the sweep")
    checkout = receipt.get("checkout-sha")
    parents = receipt.get("checkout-parents", [])
    if (
        not re.fullmatch(r"[0-9a-f]{40}", str(checkout))
        or not re.fullmatch(r"[0-9a-f]{40}", str(receipt.get("checkout-tree")))
        or not isinstance(parents, list)
        or any(not re.fullmatch(r"[0-9a-f]{40}", str(parent)) for parent in parents)
        or (checkout != source.head_sha and (len(parents) != 2 or source.head_sha not in parents))
    ):
        raise ValueError("Required artifact checkout does not contain the tested head")
    return path, measured.run_attempt


def _evidence_root(result_path: Path, root: Path) -> Path:
    suffix = result_path.parent.name.removeprefix("bmk_agentic_").removeprefix("bmk_")
    audit_root = root / f"power_audit_{suffix}"
    return audit_root if audit_root.is_dir() else root / f"agentic_{suffix}"


def _devices(audit: dict, sidecar: Path, *, native: bool) -> tuple[list[dict], list[Path]]:
    energies = audit.get("per_gpu_energy_j")
    if not isinstance(energies, dict) or not energies:
        raise ValueError(f"Missing per-device energy: {sidecar}")
    artifacts = []
    if native:
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
        if not identity_path.is_file():
            identity_path = sidecar.with_name("gpu_metrics_devices.json")
        physical = _legacy_identity(identity_path)
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
            dwdp_size=worker.get("dwdp-size", 1),
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


def _point(
    row: dict, result_path: Path, root: Path, source: SourceIdentity, *, multinode: bool
) -> dict:
    result = _read(result_path)
    fields = [
        "avg_power_w",
        "avg_total_gpu_power_w",
        "total_gpu_energy_j",
        "joules_per_output_token",
    ]
    if row.get("disagg"):
        fields += [
            "prefill_gpu_energy_j",
            "decode_gpu_energy_j",
            "prefill_joules_per_input_token",
            "decode_joules_per_output_token",
        ]
    if (
        result.get("power_valid") != 1
        or result.get("power_metric_schema_version") != 2
        or any(not _positive(result.get(field)) for field in fields)
    ):
        raise ValueError(f"Invalid required power metrics: {result_path}")
    agentic = row.get("scenario-type") == "agentic-coding"
    if agentic and result.get("users") != result.get("conc"):
        raise ValueError("AgentX users differs from concurrency")
    required_gpu_count = _validate_topology(row, result, multinode=multinode)
    artifact_name = result_path.relative_to(root).parts[0]
    suffix = artifact_name.removeprefix("bmk_agentic_" if agentic else "bmk_")
    audit_root = root / f"power_audit_{suffix}"
    evidence = _evidence_root(result_path, root)
    reference = (result.get("power_audit") or {}).get("source")
    candidates = sorted(evidence.rglob("power_validation*.json"))
    if reference:
        matched = [
            path
            for path in candidates
            if path.relative_to(evidence).as_posix() == reference
            or path.as_posix().endswith("/" + reference)
        ]
        if matched:
            candidates = matched
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
    native = bool(audit.get("selected_window"))
    receipt, _ = _source_receipt(evidence, source)
    window = audit.get("benchmark_window") or {}
    start, end = window.get("start_time_unix"), window.get("end_time_unix")
    if (
        audit.get("power_valid") is not True
        or not _positive(start)
        or not _positive(end)
        or end <= start
    ):
        raise ValueError(f"Invalid power validation/window: {sidecar}")
    devices, identity_files = _devices(audit, sidecar, native=native)
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
    if multinode:
        compute_nodes = row.get("compute-node-count")
        if (
            type(compute_nodes) is not int
            or not 0 < compute_nodes <= row["node-count"]
            or len({device["node"] for device in devices}) != compute_nodes
        ):
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
    if native:
        telemetry = list(audit_root.rglob("samples.csv"))
        if len(telemetry) != 1 or not telemetry[0].with_name("manifest.json").is_file():
            raise ValueError(f"Missing native telemetry: {audit_root}")
        package = telemetry[0].parent
        selected = audit["selected_window"]
        relative_window = selected.get("window_file")
        if not isinstance(relative_window, str):
            raise ValueError("Missing selected native measurement window")
        window_path = package / relative_window
        window_path.resolve().relative_to(package.resolve())
        central_window = _read(window_path)
        if (
            central_window.get("status") != "completed"
            or central_window.get("concurrency") != result["conc"]
            or central_window.get("benchmark_start_time_unix") != start
            or central_window.get("benchmark_end_time_unix") != end
        ):
            raise ValueError("Native measurement window differs from required point")
        telemetry = [path for path in package.rglob("*") if path.is_file()]
    else:
        if multinode:
            raise ValueError("Multi-node required power needs a native measurement package")
        telemetry = [sidecar.with_name("gpu_metrics.csv")]
    files = [result_path, sidecar, receipt, *identity_files, *telemetry]
    topology_fields = (
        "disagg",
        "is_multinode",
        "num_gpus",
        "num_aggregate_gpu",
        "num_prefill_gpu",
        "num_decode_gpu",
        "tp",
        "pp",
        "dcp_size",
        "pcp_size",
        "ep",
        "dwdp_size",
        "dp_attention",
        "prefill_tp",
        "prefill_pp",
        "prefill_dcp_size",
        "prefill_pcp_size",
        "prefill_dp_attention",
        "prefill_ep",
        "prefill_dwdp_size",
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
    source = SourceIdentity.parse(sweep)
    results = {}
    for path in sorted(artifacts.glob("bmk_*/*.json")):
        result = _read(path)
        agentic = path.parent.name.startswith("bmk_agentic_")
        key = PointKey(
            result.get("recipe_fingerprint"),
            result.get("users") if agentic else result.get("conc"),
            "agentic_traces" if agentic else "single_turn",
            None if agentic else result.get("isl"),
            None if agentic else result.get("osl"),
        )
        results.setdefault(key, []).append((path, result))
    points = []
    for row, multinode, key in required_points(sweep["matrix"]):
        matches = [
            (path, result)
            for path, result in results.get(key, [])
            if result.get("infmax_model_prefix") == row["model-prefix"]
            and result.get("hw") == row["runner"]
            and result.get("framework") == row["framework"]
            and result.get("precision") == row["precision"]
            and result.get("disagg") is row.get("disagg", False)
            and result.get("is_multinode") is multinode
        ]
        if matches:
            attempts = [
                (path, result, _source_receipt(_evidence_root(path, artifacts), source)[1])
                for path, result in matches
            ]
            latest = max(attempt for _, _, attempt in attempts)
            matches = [(path, result) for path, result, attempt in attempts if attempt == latest]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one required result for {row['exp-name']} concurrency {key.concurrency}"
            )
        points.append(_point(row, matches[0][0], artifacts, source, multinode=multinode))
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
    parser.add_argument("--expected-matrix", type=Path)
    parser.add_argument("--source-plan", type=Path)
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
    if (
        args.expected_run_attempt is not None
        and sweep.get("run-attempt") != args.expected_run_attempt
    ):
        raise ValueError("Required manifest attempt differs from the source run")
    if args.expected_matrix:
        compare_required_scope(sweep["matrix"], _read(args.expected_matrix))
    if args.source_plan:
        plan = _read(args.source_plan)
        if SourceIdentity.parse(plan) != SourceIdentity.parse(sweep):
            raise ValueError("Required manifest source differs from retained plan")
        compare_required_scope(sweep["matrix"], plan["matrix"])
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
