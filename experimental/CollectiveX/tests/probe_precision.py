#!/usr/bin/env python3
"""Bounded real-hardware capability probe for provisional CollectiveX precision cells."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import inspect
import json
import os
import platform
import re
import socket
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]

import artifact_safety  # noqa: E402
import capability  # noqa: E402
import ep_harness  # noqa: E402


FORMAT = "collectivex.precision-probe.v1"
PLAN_FORMAT = "collectivex.precision-probe-plan.v1"
CONTROL_FORMAT = "collectivex.precision-probe-control.v1"
RECORD_TYPE = "precision-capability-probe"
PROBE_CONTRACT = "bounded-native-cell-v1"
FENCE_CONTRACT = "caller-event-cross-stream-v1"
SUPPORTED_REASON = "native-probe-passed"
UNSUPPORTED_REASONS = frozenset({
    "backend-construction-failed",
    "completion-fence-failed",
    "cross-rank-evidence-mismatch",
    "native-operation-failed",
    "precision-contract-mismatch",
    "runtime-identity-mismatch",
    "target-not-provisional",
    "transport-fallback-detected",
    "unsupported-native-api",
    "unverified-execution-identity",
})
BACKENDS = frozenset({
    "deepep", "deepep-v2", "deepep-hybrid", "mori", "uccl",
})
SHA40 = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")
IMAGE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


class ProbeError(RuntimeError):
    """A provisional cell did not produce complete native runtime evidence."""

    def __init__(self, reason: str):
        if reason not in UNSUPPORTED_REASONS:
            raise ValueError(f"unknown precision probe reason {reason!r}")
        super().__init__(reason)
        self.reason = reason


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _exact_keys(value: Any, expected: set[str], path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{path} fields differ from {FORMAT}")
    return value


def _text(value: Any, path: str, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 4096
        or any(ord(character) < 0x20 or ord(character) > 0x7E for character in value)
    ):
        raise ValueError(f"{path} is not bounded printable ASCII")
    return value


def _boolean(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{path} is not boolean")
    return value


def _integer(value: Any, path: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{path} is not an integer >= {minimum}")
    return value


def validate_manifest(document: Any) -> dict[str, Any]:
    """Validate the closed probe format without extending publication schemas."""
    doc = _exact_keys(document, {
        "evidence", "format", "generated_at", "privacy", "probe_contract",
        "record_type", "result", "schema_version", "target", "topology",
    }, "probe")
    if (
        doc["format"] != FORMAT
        or doc["record_type"] != RECORD_TYPE
        or doc["schema_version"] != 1
        or doc["probe_contract"] != PROBE_CONTRACT
    ):
        raise ValueError("probe format, record type, schema, or contract differs")
    _text(doc["generated_at"], "probe.generated_at")
    target = _exact_keys(doc["target"], {
        "backend", "basis", "ep", "mode", "precision_profile", "registry_disposition",
        "sku",
    }, "probe.target")
    if (
        target["backend"] not in BACKENDS
        or target["registry_disposition"] != "provisional"
        or target["mode"] not in {"normal", "low-latency"}
    ):
        raise ValueError("probe target is not a provisional native adapter cell")
    for field in ("basis", "precision_profile", "sku"):
        _text(target[field], f"probe.target.{field}")
    _integer(target["ep"], "probe.target.ep", 1)
    topology = _exact_keys(doc["topology"], {
        "gpus_per_node", "nodes", "placement_valid", "scale_up_domain",
        "scale_up_transport", "scale_out_transport", "scope", "topology_class",
        "transport", "world_size",
    }, "probe.topology")
    for field in ("gpus_per_node", "nodes", "scale_up_domain", "world_size"):
        _integer(topology[field], f"probe.topology.{field}", 1)
    for field in ("scale_up_transport", "scope", "topology_class", "transport"):
        _text(topology[field], f"probe.topology.{field}")
    _text(topology["scale_out_transport"], "probe.topology.scale_out_transport", nullable=True)
    _boolean(topology["placement_valid"], "probe.topology.placement_valid")
    result = _exact_keys(doc["result"], {
        "disposition", "reason", "registry_mutation", "runtime_executed",
        "static_inspection_sufficient",
    }, "probe.result")
    if result["disposition"] not in {"supported", "unsupported"}:
        raise ValueError("probe result disposition is invalid")
    expected_reason = (
        SUPPORTED_REASON if result["disposition"] == "supported" else result["reason"]
    )
    if result["reason"] != expected_reason or (
        result["disposition"] == "unsupported"
        and result["reason"] not in UNSUPPORTED_REASONS
    ):
        raise ValueError("probe result reason is invalid")
    if result["registry_mutation"] is not False or result["static_inspection_sufficient"] is not False:
        raise ValueError("probe must never mutate or statically promote the registry")
    _boolean(result["runtime_executed"], "probe.result.runtime_executed")
    privacy = _exact_keys(doc["privacy"], {"contract", "sanitized"}, "probe.privacy")
    if privacy != {"contract": "artifact-safety-v1", "sanitized": True}:
        raise ValueError("probe privacy contract differs")
    if result["disposition"] == "supported":
        _validate_evidence(doc["evidence"])
    elif doc["evidence"] is not None:
        _validate_evidence(doc["evidence"])
    artifact_safety.assert_publication_safe([doc])
    return doc


def _validate_evidence(value: Any) -> None:
    evidence = _exact_keys(value, {
        "api", "completion", "identity", "precision", "transport",
    }, "probe.evidence")
    api = _exact_keys(evidence["api"], {"calls", "signature_sha256"}, "probe.evidence.api")
    if not isinstance(api["calls"], list) or not api["calls"]:
        raise ValueError("probe API calls are empty")
    for index, call in enumerate(api["calls"]):
        item = _exact_keys(call, {"name", "signature"}, f"probe.evidence.api.calls[{index}]")
        _text(item["name"], "probe API name")
        _text(item["signature"], "probe API signature")
    if not isinstance(api["signature_sha256"], str) or not SHA256.fullmatch(api["signature_sha256"]):
        raise ValueError("probe API signature digest is invalid")
    completion = _exact_keys(evidence["completion"], {
        "caller_event_complete", "contract", "mode", "output_finite",
        "verifier_stream_complete",
    }, "probe.evidence.completion")
    if completion["contract"] != FENCE_CONTRACT:
        raise ValueError("probe completion contract differs")
    _text(completion["mode"], "probe completion mode")
    if not all(
        _boolean(completion[field], f"probe completion {field}")
        for field in ("caller_event_complete", "output_finite", "verifier_stream_complete")
    ):
        raise ValueError("probe completion evidence did not pass")
    identity_record = _exact_keys(evidence["identity"], {
        "backend_components", "backend_provenance_sha256", "image_digest",
        "image_digest_verified", "image_reference", "source_sha",
    }, "probe.evidence.identity")
    if not SHA40.fullmatch(str(identity_record["source_sha"])):
        raise ValueError("probe source SHA is invalid")
    if not IMAGE_DIGEST.fullmatch(str(identity_record["image_digest"])):
        raise ValueError("probe image digest is invalid")
    _text(identity_record["image_reference"], "probe image reference")
    if identity_record["image_digest_verified"] is not True:
        raise ValueError("probe image digest is unverified")
    if not SHA256.fullmatch(str(identity_record["backend_provenance_sha256"])):
        raise ValueError("probe backend provenance digest is invalid")
    components = identity_record["backend_components"]
    if not isinstance(components, list) or not components:
        raise ValueError("probe backend component identity is empty")
    for component in components:
        item = _exact_keys(component, {"revision", "role", "version"}, "probe backend component")
        _text(item["role"], "probe backend component role")
        _text(item["revision"], "probe backend component revision", nullable=True)
        _text(item["version"], "probe backend component version", nullable=True)
        if item["revision"] is None and item["version"] is None:
            raise ValueError("probe backend component has no identity")
    precision = _exact_keys(evidence["precision"], {
        "combine", "correctness", "dispatch", "profile_id",
    }, "probe.evidence.precision")
    _text(precision["profile_id"], "probe precision profile")
    for direction in ("dispatch", "combine"):
        axis = _exact_keys(precision[direction], {
            "accumulator_dtype", "accumulator_evidence", "api_input_dtype",
            "api_output_dtype", "communication_format", "runtime_input",
            "runtime_output", "scale_contract", "semantic_output",
        }, f"probe.evidence.precision.{direction}")
        for field in (
            "accumulator_dtype", "accumulator_evidence", "api_input_dtype",
            "api_output_dtype", "communication_format",
        ):
            _text(axis[field], f"probe precision {direction} {field}")
        _validate_tensor_summary(axis["runtime_input"], f"probe precision {direction} input")
        _validate_tensor_summary(axis["runtime_output"], f"probe precision {direction} output")
        _validate_tensor_summary(axis["semantic_output"], f"probe precision {direction} semantic")
        scale = _exact_keys(axis["scale_contract"], {
            "alignment", "dtype", "finite", "group_size", "layout", "padding",
            "positive", "runtime_shapes", "runtime_storage_dtype",
        }, f"probe precision {direction} scales")
        for field in ("alignment", "layout", "padding"):
            _text(scale[field], f"probe precision {direction} scale {field}")
        _text(scale["dtype"], "probe scale dtype", nullable=True)
        _text(scale["runtime_storage_dtype"], "probe scale storage", nullable=True)
        if scale["group_size"] is not None:
            _integer(scale["group_size"], "probe scale group", 1)
        for field in ("finite", "positive"):
            if scale[field] is not None:
                _boolean(scale[field], f"probe scale {field}")
        if not isinstance(scale["runtime_shapes"], list):
            raise ValueError("probe scale shapes are invalid")
    correctness = precision["correctness"]
    if not isinstance(correctness, dict) or correctness.get("passed") is not True:
        raise ValueError("probe precision correctness did not pass")
    transport = _exact_keys(evidence["transport"], {
        "evidence", "fallback_used", "native_backend", "requested", "runtime_route",
    }, "probe.evidence.transport")
    for field in ("native_backend", "requested", "runtime_route"):
        _text(transport[field], f"probe transport {field}")
    if transport["fallback_used"] is not False:
        raise ValueError("probe transport fallback is present")
    if not isinstance(transport["evidence"], list) or not transport["evidence"]:
        raise ValueError("probe transport evidence is empty")
    for item in transport["evidence"]:
        _text(item, "probe transport evidence item")


def _validate_tensor_summary(value: Any, path: str) -> None:
    summary = _exact_keys(value, {"finite", "rank", "shapes", "storage_dtype"}, path)
    _text(summary["storage_dtype"], f"{path}.storage_dtype")
    _integer(summary["rank"], f"{path}.rank", 0)
    if summary["finite"] is not True:
        raise ValueError(f"{path} is not finite")
    if not isinstance(summary["shapes"], list) or not summary["shapes"]:
        raise ValueError(f"{path} shapes are empty")
    for shape in summary["shapes"]:
        if not isinstance(shape, list) or any(type(item) is not int or item < 0 for item in shape):
            raise ValueError(f"{path} shape is invalid")


def provisional_targets() -> list[dict[str, Any]]:
    """Return deterministic probe cells without changing their dispositions."""
    return sorted(
        capability.provisional_precision_targets(),
        key=lambda item: (
            item["sku"], item["backend"], item["ep"], item["mode"],
            item["precision_profile"],
        ),
    )


def _probe_id(target: dict[str, Any]) -> str:
    return f"probe-{_sha({key: target[key] for key in ('backend', 'sku', 'ep', 'mode', 'precision_profile')})[:20]}"


def _workflow_row(target: dict[str, Any]) -> dict[str, Any]:
    topology = capability.topology_for(target["sku"], target["ep"])
    if topology is None:
        raise ValueError("precision probe target has no registered topology")
    return {
        "backend": target["backend"],
        "basis": target["basis"],
        "disposition": target["disposition"],
        "ep": target["ep"],
        "execution_weight": target["ep"],
        "gpus_per_node": topology["gpus_per_node"],
        "id": _probe_id(target),
        "launcher": capability.PLATFORMS[target["sku"]]["launcher"],
        "mode": target["mode"],
        "n": 1,
        "nodes": topology["nodes"],
        "precision_profile": target["precision_profile"],
        "scale_up_domain": topology["scale_up_domain"],
        "sku": target["sku"],
    }


def workflow_plan(
    *, backend: str = "all", only_sku: str = "", min_nodes: int = 0,
    max_nodes: int = 0,
) -> dict[str, Any]:
    if min_nodes < 0 or max_nodes < 0 or (min_nodes and max_nodes and min_nodes > max_nodes):
        raise ValueError("precision probe node filters are invalid")
    targets = [
        target for target in provisional_targets()
        if (backend == "all" or target["backend"] == backend)
        and (not only_sku or target["sku"] == only_sku)
        and (
            not min_nodes
            or capability.topology_for(target["sku"], target["ep"])["nodes"] >= min_nodes
        )
        and (
            not max_nodes
            or capability.topology_for(target["sku"], target["ep"])["nodes"] <= max_nodes
        )
    ]
    if backend != "all" and backend not in BACKENDS:
        raise ValueError("precision probe backend is not registered")
    if only_sku and only_sku not in capability.PLATFORMS:
        raise ValueError("precision probe SKU is not registered")
    if not targets:
        raise ValueError("precision probe filters select no provisional cells")
    return {
        "format": PLAN_FORMAT,
        "include": [_workflow_row(target) for target in targets],
        "schema_version": 1,
    }


def validate_workflow_plan(document: Any) -> dict[str, Any]:
    plan = _exact_keys(document, {"format", "include", "schema_version"}, "probe plan")
    if plan["format"] != PLAN_FORMAT or plan["schema_version"] != 1:
        raise ValueError("precision probe plan format differs")
    if not isinstance(plan["include"], list):
        raise ValueError("precision probe plan include is not a list")
    expected = {_probe_id(target): _workflow_row(target) for target in provisional_targets()}
    seen: set[str] = set()
    for row in plan["include"]:
        if not isinstance(row, dict) or row.get("id") not in expected or row != expected[row["id"]]:
            raise ValueError("precision probe plan row differs from the capability registry")
        if row["id"] in seen:
            raise ValueError("precision probe plan contains a duplicate row")
        seen.add(row["id"])
    return plan


def extract_control(
    plan: Any, *, probe_id: str, sku: str, backend: str, nodes: int,
) -> dict[str, Any]:
    rows = [row for row in validate_workflow_plan(plan)["include"] if row["id"] == probe_id]
    if len(rows) != 1:
        raise ValueError("precision probe ID is not unique in the plan")
    row = rows[0]
    if (row["sku"], row["backend"], row["nodes"]) != (sku, backend, nodes):
        raise ValueError("precision probe control differs from the workflow matrix")
    target = select_target(
        backend=row["backend"], sku=row["sku"], ep=row["ep"], mode=row["mode"],
        precision_profile=row["precision_profile"],
    )
    topology = capability.topology_for(row["sku"], row["ep"])
    if topology is None:
        raise ValueError("precision probe control has no topology")
    return {
        "format": CONTROL_FORMAT,
        "id": row["id"],
        "launcher": row["launcher"],
        "schema_version": 1,
        "target": target,
        "topology": topology,
    }


def validate_control(
    document: Any, *, sku: str, backend: str, nodes: int,
) -> dict[str, Any]:
    control = _exact_keys(
        document, {"format", "id", "launcher", "schema_version", "target", "topology"},
        "probe control",
    )
    if control["format"] != CONTROL_FORMAT or control["schema_version"] != 1:
        raise ValueError("precision probe control format differs")
    expected = extract_control(
        {"format": PLAN_FORMAT, "include": [_workflow_row(control["target"])], "schema_version": 1},
        probe_id=control["id"], sku=sku, backend=backend, nodes=nodes,
    )
    if control != expected:
        raise ValueError("precision probe control differs from the capability registry")
    return control


def validate_bundle(plan: Any, manifests: list[Any]) -> None:
    rows = validate_workflow_plan(plan)["include"]
    expected = {
        (row["backend"], row["sku"], row["ep"], row["mode"], row["precision_profile"])
        for row in rows
    }
    observed = []
    for manifest in manifests:
        target = validate_manifest(manifest)["target"]
        observed.append((
            target["backend"], target["sku"], target["ep"], target["mode"],
            target["precision_profile"],
        ))
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("precision probe bundle does not cover the exact workflow plan")


def select_target(
    *, backend: str, sku: str, ep: int, mode: str, precision_profile: str
) -> dict[str, Any]:
    matches = [
        item for item in provisional_targets()
        if item["backend"] == backend and item["sku"] == sku and item["ep"] == ep
        and item["mode"] == mode and item["precision_profile"] == precision_profile
    ]
    if len(matches) != 1:
        raise ProbeError("target-not-provisional")
    return matches[0]


def _target_record(target: dict[str, Any]) -> dict[str, Any]:
    return {
        "backend": target["backend"],
        "basis": target["basis"],
        "ep": target["ep"],
        "mode": target["mode"],
        "precision_profile": target["precision_profile"],
        "registry_disposition": "provisional",
        "sku": target["sku"],
    }


def build_manifest(
    *, target: dict[str, Any], topology: dict[str, Any], disposition: str,
    reason: str, runtime_executed: bool, evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    document = {
        "evidence": evidence,
        "format": FORMAT,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "privacy": {"contract": "artifact-safety-v1", "sanitized": True},
        "probe_contract": PROBE_CONTRACT,
        "record_type": RECORD_TYPE,
        "result": {
            "disposition": disposition,
            "reason": reason,
            "registry_mutation": False,
            "runtime_executed": runtime_executed,
            "static_inspection_sufficient": False,
        },
        "schema_version": 1,
        "target": _target_record(target),
        "topology": topology,
    }
    return validate_manifest(document)


def _write_atomic(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical(document) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _dtype_name(dtype: Any) -> str:
    return str(dtype).removeprefix("torch.")


def _local_tensor_summary(torch_module, tensor) -> dict[str, Any]:
    return {
        "finite": bool(torch_module.isfinite(tensor.float()).all().item()),
        "rank": int(tensor.ndim),
        "shape": [int(item) for item in tensor.shape],
        "storage_dtype": _dtype_name(tensor.dtype),
    }


def _aggregate_tensor_summaries(records: list[dict[str, Any]]) -> dict[str, Any]:
    dtypes = {record["storage_dtype"] for record in records}
    ranks = {record["rank"] for record in records}
    if len(dtypes) != 1 or len(ranks) != 1:
        raise ProbeError("cross-rank-evidence-mismatch")
    return {
        "finite": all(record["finite"] for record in records),
        "rank": ranks.pop(),
        "shapes": [
            list(shape) for shape in sorted({tuple(record["shape"]) for record in records})
        ],
        "storage_dtype": dtypes.pop(),
    }


def _scale_contract(torch_module, axis: dict[str, Any], scales) -> dict[str, Any]:
    return {
        "alignment": axis["alignment_contract"],
        "dtype": axis["scale_dtype"],
        "finite": (
            bool(torch_module.isfinite(scales.float()).all().item())
            if scales is not None else None
        ),
        "group_size": axis["scale_group_size"],
        "layout": axis["scale_layout"],
        "padding": axis["padding_contract"],
        "positive": bool((scales > 0).all().item()) if scales is not None else None,
        "runtime_shape": [int(item) for item in scales.shape] if scales is not None else None,
        "runtime_storage_dtype": _dtype_name(scales.dtype) if scales is not None else None,
    }


def _aggregate_scale_contracts(records: list[dict[str, Any]]) -> dict[str, Any]:
    fixed_fields = (
        "alignment", "dtype", "group_size", "layout", "padding", "runtime_storage_dtype",
    )
    result: dict[str, Any] = {}
    for field in fixed_fields:
        values = {_canonical(record[field]) for record in records}
        if len(values) != 1:
            raise ProbeError("cross-rank-evidence-mismatch")
        result[field] = records[0][field]
    for field in ("finite", "positive"):
        values = [record[field] for record in records]
        result[field] = None if all(value is None for value in values) else all(value is True for value in values)
    result["runtime_shapes"] = [
        list(shape)
        for shape in sorted({tuple(record["runtime_shape"] or ()) for record in records})
    ] if any(record["runtime_shape"] is not None for record in records) else []
    return result


def _signature(callable_object: Any, name: str) -> dict[str, str]:
    try:
        signature = str(inspect.signature(callable_object))
    except (TypeError, ValueError) as exc:
        raise ProbeError("unsupported-native-api") from exc
    _text(signature, f"native API {name} signature")
    return {"name": name, "signature": signature}


def _api_evidence(backend_name: str, backend) -> dict[str, Any]:
    if backend_name in {"deepep", "uccl"}:
        native = type(backend.buffer)
        dispatch_name = "low_latency_dispatch" if backend.mode == "low-latency" else "dispatch"
        combine_name = "low_latency_combine" if backend.mode == "low-latency" else "combine"
        calls = [
            _signature(native.__init__, f"{native.__name__}.__init__"),
            _signature(getattr(native, dispatch_name), f"{native.__name__}.{dispatch_name}"),
            _signature(getattr(native, combine_name), f"{native.__name__}.{combine_name}"),
        ]
    elif backend_name in {"deepep-v2", "deepep-hybrid"}:
        native = type(backend.buffer)
        calls = [
            _signature(native.__init__, f"{native.__name__}.__init__"),
            _signature(native.dispatch, f"{native.__name__}.dispatch"),
            _signature(native.combine, f"{native.__name__}.combine"),
        ]
    elif backend_name == "mori":
        native = type(backend.op)
        calls = [
            _signature(type(backend.config).__init__, "EpDispatchCombineConfig.__init__"),
            _signature(native.dispatch, f"{native.__name__}.dispatch"),
            _signature(native.combine, f"{native.__name__}.combine"),
        ]
    else:  # pragma: no cover - guarded by target registry
        raise ProbeError("unsupported-native-api")
    return {"calls": calls, "signature_sha256": _sha(calls)}


def _completion_mode(backend_name: str, mode: str) -> str:
    if backend_name in {"deepep", "uccl"}:
        return "async_finish=false;return_recv_hook=false"
    if backend_name == "deepep-v2":
        return "async_with_compute_stream=false;do_cpu_sync=true"
    if backend_name == "deepep-hybrid":
        return "metadata-nonblocking=false;caller-stream-ordered"
    if backend_name == "mori":
        return "current-stream-ordered"
    raise ProbeError("unsupported-native-api")


def _transport_evidence(backend_name: str, backend, args) -> dict[str, Any]:
    provenance = backend.backend_provenance
    fallback = False
    facts: list[str]
    if backend_name == "deepep":
        if args.scope == "scale-out" and int(provenance["num_rdma_bytes"]) <= 0:
            fallback = True
        if args.scale_up_transport == "mnnvl" and provenance["mnnvl_comm"] != "explicit-allow-mnnvl":
            fallback = True
        route = f"deepep-{backend.mode}"
        facts = [
            f"mnnvl={provenance['mnnvl_comm']}",
            f"nvl-buffer={int(provenance['num_nvl_bytes']) > 0}",
            f"rdma-buffer={int(provenance['num_rdma_bytes']) > 0}",
        ]
    elif backend_name == "uccl":
        scratch_location = str(backend.buffer.scratch.device.type)
        rdma_active = int(provenance["num_rdma_bytes"]) > 0
        fallback = rdma_active and scratch_location != "cuda"
        route = f"uccl-proxy-{backend.mode}"
        facts = [f"rdma-buffer={rdma_active}", f"rdma-memory={scratch_location}"]
    elif backend_name == "deepep-v2":
        expected_gin = args.scope == "scale-out"
        fallback = bool(provenance["gin_enabled"]) != expected_gin
        route = str(provenance["communication_backend"])
        facts = [
            f"gin-enabled={bool(provenance['gin_enabled'])}",
            f"nccl-communicator={provenance['nccl_communicator']}",
        ]
    elif backend_name == "deepep-hybrid":
        route = str(provenance["transport"])
        expected_build = "multinode-doca" if args.scope == "scale-out" else "intradomain"
        realized_build = os.environ.get("DEEPEP_HYBRID_BUILD_MODE")
        fallback = realized_build != expected_build
        facts = [f"build-mode={realized_build or 'missing'}", f"domains={backend.communication_domains}"]
    elif backend_name == "mori":
        route = str(backend.kernel_generation)
        expected_kernel = (
            "inter-node-v1" if args.scope == "scale-out"
            else "async-ll" if args.runner == "mi325x" else "intranode"
        )
        fallback = route != expected_kernel
        facts = [
            f"kernel={route}",
            f"external-input={bool(provenance['use_external_inp_buf'])}",
            f"qps={int(provenance['num_qps'])}",
        ]
    else:  # pragma: no cover - guarded by registry
        raise ProbeError("unsupported-native-api")
    return {
        "evidence": sorted(facts),
        "fallback_used": fallback,
        "native_backend": backend_name,
        "requested": str(args.transport),
        "runtime_route": route,
    }


def _component_identities(backend_name: str, provenance: dict[str, Any]) -> list[dict[str, Any]]:
    if backend_name == "deepep":
        values = [("deepep", provenance.get("deepep_commit"), provenance.get("deepep_version"))]
    elif backend_name == "deepep-v2":
        values = [
            ("deepep-v2", provenance.get("deepep_commit"), provenance.get("deepep_version")),
            ("deepep-tree", provenance.get("deepep_tree"), None),
            ("fmt", provenance.get("fmt_commit"), None),
        ]
    elif backend_name == "deepep-hybrid":
        values = [
            ("deepep-hybrid", provenance.get("deepep_commit"), None),
            ("deepep-tree", provenance.get("deepep_tree"), None),
        ]
    elif backend_name == "uccl":
        values = [
            ("uccl", provenance.get("uccl_commit"), provenance.get("uccl_version")),
            ("uccl-wrapper", provenance.get("uccl_wrapper_commit"), None),
        ]
    elif backend_name == "mori":
        values = [("mori", provenance.get("mori_commit"), None)]
    else:  # pragma: no cover
        raise ProbeError("unsupported-native-api")
    result = [
        {"revision": revision, "role": role, "version": version}
        for role, revision, version in values
    ]
    for item in result:
        if item["revision"] is None and item["version"] is None:
            raise ProbeError("unverified-execution-identity")
    return result


def _execution_identity(backend_name: str, backend) -> dict[str, Any]:
    source_sha = os.environ.get("COLLECTIVEX_SOURCE_SHA") or os.environ.get("GITHUB_SHA")
    image_reference = os.environ.get("COLLECTIVEX_IMAGE")
    image_digest = os.environ.get("COLLECTIVEX_IMAGE_DIGEST")
    verified = os.environ.get("COLLECTIVEX_IMAGE_DIGEST_VERIFIED") == "1"
    if (
        not isinstance(source_sha, str) or not SHA40.fullmatch(source_sha)
        or not isinstance(image_reference, str) or not image_reference
        or not isinstance(image_digest, str) or not IMAGE_DIGEST.fullmatch(image_digest)
        or not verified
    ):
        raise ProbeError("unverified-execution-identity")
    provenance = backend.backend_provenance
    return {
        "backend_components": _component_identities(backend_name, provenance),
        "backend_provenance_sha256": _sha(provenance),
        "image_digest": image_digest,
        "image_digest_verified": True,
        "image_reference": image_reference,
        "source_sha": source_sha,
    }


def _correctness_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    profile_ids = {record["profile_id"] for record in records}
    if len(profile_ids) != 1:
        raise ProbeError("cross-rank-evidence-mismatch")
    result: dict[str, Any] = {"profile_id": profile_ids.pop()}
    for direction in ("dispatch", "combine"):
        axes = [record[direction] for record in records]
        scale_finite = [axis["scales_finite"] for axis in axes]
        scale_positive = [axis["scales_positive"] for axis in axes]
        result[direction] = {
            "dequantized_semantics": all(axis["dequantized_semantics"] for axis in axes),
            "encoded_payload_valid": all(axis["encoded_payload_valid"] for axis in axes),
            "max_abs_error": max(float(axis["max_abs_error"]) for axis in axes),
            "max_rel_error": max(float(axis["max_rel_error"]) for axis in axes),
            "passed": all(axis["passed"] for axis in axes),
            "saturation_count": sum(int(axis["saturation_count"]) for axis in axes),
            "saturation_rate": max(float(axis["saturation_rate"]) for axis in axes),
            "scales_finite": (
                None if all(value is None for value in scale_finite)
                else all(value is True for value in scale_finite)
            ),
            "scales_positive": (
                None if all(value is None for value in scale_positive)
                else all(value is True for value in scale_positive)
            ),
        }
    result["passed"] = all(record["passed"] for record in records) and all(
        result[direction]["passed"] for direction in ("dispatch", "combine")
    )
    return result


def _topology_record(topology: dict[str, Any], placement_valid: bool) -> dict[str, Any]:
    return {
        "gpus_per_node": topology["gpus_per_node"],
        "nodes": topology["nodes"],
        "placement_valid": placement_valid,
        "scale_up_domain": topology["scale_up_domain"],
        "scale_up_transport": topology["scale_up_transport"],
        "scale_out_transport": topology["scale_out_transport"],
        "scope": topology["scope"],
        "topology_class": topology["topology_class"],
        "transport": topology["transport"],
        "world_size": topology["nodes"] * topology["gpus_per_node"],
    }


def _backend_class(name: str):
    if name == "deepep":
        from ep_deepep import DeepEPBackend
        return DeepEPBackend
    if name == "deepep-v2":
        from ep_deepep_v2 import DeepEPV2Backend
        return DeepEPV2Backend
    if name == "deepep-hybrid":
        from ep_deepep_hybrid import DeepEPHybridBackend
        return DeepEPHybridBackend
    if name == "uccl":
        from ep_uccl import UCCLBackend
        return UCCLBackend
    if name == "mori":
        from ep_mori import MoRIBackend
        return MoRIBackend
    raise ProbeError("unsupported-native-api")


def _runtime_args(target: dict[str, Any], topology: dict[str, Any], fingerprint: dict[str, Any]):
    return SimpleNamespace(
        backend=target["backend"],
        eplb=False,
        experts=256,
        gpus_per_node=topology["gpus_per_node"],
        hidden=7168,
        mode=target["mode"],
        num_logical_experts=256,
        num_sms=24,
        phase="decode",
        precision_profile=target["precision_profile"],
        runner=target["sku"],
        runtime_fingerprint=fingerprint,
        scale_out_transport=topology["scale_out_transport"] or "",
        scale_up_domain=topology["scale_up_domain"],
        scale_up_transport=topology["scale_up_transport"],
        scope=topology["scope"],
        tokens_ladder="8",
        topk=8,
        topology_class=topology["topology_class"],
        transport=topology["transport"],
    )


def _init_distributed(torch_module, dist, backend_name: str, device, rank: int, world_size: int) -> None:
    if dist.is_initialized():
        return
    if backend_name == "mori":
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl", rank=rank, world_size=world_size, device_id=device
        )
    elif backend_name == "deepep-v2":
        dist.init_process_group("nccl", device_id=device)
    else:
        dist.init_process_group("nccl")


def _runtime_context(torch_module, dist, target: dict[str, Any], device, local_rank: int):
    import run_ep

    world_size = dist.get_world_size()
    topology = capability.topology_for(target["sku"], target["ep"])
    if topology is None or world_size != target["ep"]:
        raise ProbeError("runtime-identity-mismatch")
    machine = {"x86_64": "amd64", "aarch64": "arm64"}.get(
        platform.machine(), platform.machine()
    )
    properties = torch_module.cuda.get_device_properties(device)
    if torch_module.version.hip:
        vendor = "amd"
        arch = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
    else:
        vendor = "nvidia"
        major, minor = torch_module.cuda.get_device_capability(device)
        arch = f"sm{major}{minor}"
    fingerprint = run_ep._runtime_fingerprint(
        torch_module, device, machine=machine, vendor=vendor, arch=arch
    )
    issues = capability.runtime_identity_issues(
        target["sku"], vendor=vendor, arch=arch, machine=machine,
        device_name=torch_module.cuda.get_device_name(device),
        device_count=torch_module.cuda.device_count(), world_size=world_size,
    )
    records: list[Any] = [None] * world_size
    dist.all_gather_object(records, (socket.gethostname(), local_rank, fingerprint, issues))
    if any(record[3] for record in records):
        raise ProbeError("runtime-identity-mismatch")
    placement = run_ep._summarize_realized_placement(
        [(record[0], record[1]) for record in records],
        expected_nodes=topology["nodes"],
        expected_gpus_per_node=topology["gpus_per_node"],
        expected_world_size=world_size,
    )
    common_fingerprint = run_ep._common_runtime_fingerprint([record[2] for record in records])
    return topology, placement, common_fingerprint


def _local_probe(torch_module, dist, target: dict[str, Any], backend, args, rank: int):
    import routing

    tokens = 8
    global_idx, global_weights = routing.build_global_routing(
        tokens * target["ep"], args.experts, args.topk, "uniform", ep_harness.ROUTING_SEED
    )
    local_idx, local_weights = routing.rank_slice(
        global_idx, global_weights, rank, tokens
    )
    x = routing.rank_activations(
        tokens, args.hidden, ep_harness.ROUTING_SEED, rank,
        torch_module.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"),
        torch_module.bfloat16,
    )
    problem = backend.make_problem(
        tokens, local_idx.to(x.device), local_weights.to(x.device), x
    )
    oracle = ep_harness._run_expert_oracle(
        torch_module, routing, backend, problem, global_idx, global_weights, rank,
        args.experts // target["ep"], ep_harness.ROUTING_SEED,
    )
    if not oracle["passed"] or not oracle["_precision"]["passed"]:
        raise ProbeError("precision-contract-mismatch")

    caller = torch_module.cuda.Stream(device=x.device)
    verifier = torch_module.cuda.Stream(device=x.device)
    completion_event = torch_module.cuda.Event()
    with torch_module.cuda.stream(caller):
        handle = backend.dispatch(problem)
        problem.recv_tokens = backend.recv_tokens(handle)
        view = (
            backend.inspect_expert_dispatch(problem, handle)
            if target["mode"] == "low-latency"
            else backend.inspect_dispatch(problem, handle)
        )
        backend.stage(problem, handle)
        combined = backend.combine(problem, handle)
        completion_event.record(caller)
    with torch_module.cuda.stream(verifier):
        verifier.wait_event(completion_event)
        verifier_sentinel = combined.float().abs().sum()
    verifier.synchronize()
    completion = {
        "caller_event_complete": bool(completion_event.query()),
        "contract": FENCE_CONTRACT,
        "mode": _completion_mode(target["backend"], target["mode"]),
        "output_finite": bool(torch_module.isfinite(combined.float()).all().item()),
        "verifier_stream_complete": bool(torch_module.isfinite(verifier_sentinel).item()),
    }
    if not all(
        completion[field]
        for field in ("caller_event_complete", "output_finite", "verifier_stream_complete")
    ):
        raise ProbeError("completion-fence-failed")

    deferred = getattr(backend, "capture_deferred_provenance", None)
    if deferred is not None:
        deferred()
    dispatch_input = problem.dispatch_x[0] if isinstance(problem.dispatch_x, tuple) else problem.dispatch_x
    dispatch_input_scales = (
        problem.dispatch_x[1] if isinstance(problem.dispatch_x, tuple)
        else getattr(problem, "dispatch_scales", None)
        or getattr(problem, "scales", None)
    )
    dispatch_axis = backend.communication_precision["dispatch"]
    combine_axis = backend.communication_precision["combine"]
    local = {
        "api": _api_evidence(target["backend"], backend),
        "completion": completion,
        "identity": _execution_identity(target["backend"], backend),
        "precision": {
            "profile_id": backend.precision_profile_id,
            "correctness": oracle["_precision"],
            "dispatch": {
                "accumulator_dtype": "not-applicable",
                "accumulator_evidence": "not-applicable",
                "api_input_dtype": dispatch_axis["api_input_dtype"],
                "api_output_dtype": dispatch_axis["api_output_dtype"],
                "communication_format": dispatch_axis["communication_format"],
                "runtime_input": _local_tensor_summary(torch_module, dispatch_input),
                "runtime_output": _local_tensor_summary(torch_module, view.encoded_payload),
                "scale_contract": _scale_contract(
                    torch_module, dispatch_axis,
                    view.scales if view.scales is not None else dispatch_input_scales,
                ),
                "semantic_output": _local_tensor_summary(torch_module, view.payload),
            },
            "combine": {
                "accumulator_dtype": "fp32",
                "accumulator_evidence": "pinned-source-image-plus-runtime-oracle",
                "api_input_dtype": combine_axis["api_input_dtype"],
                "api_output_dtype": combine_axis["api_output_dtype"],
                "communication_format": combine_axis["communication_format"],
                "runtime_input": _local_tensor_summary(torch_module, handle.combine_input),
                "runtime_output": _local_tensor_summary(torch_module, combined),
                "scale_contract": _scale_contract(torch_module, combine_axis, None),
                "semantic_output": _local_tensor_summary(torch_module, combined),
            },
        },
        "transport": _transport_evidence(target["backend"], backend, args),
    }
    if local["transport"]["fallback_used"]:
        raise ProbeError("transport-fallback-detected")
    return local


def _aggregate_local(records: list[dict[str, Any]]) -> dict[str, Any]:
    for field in ("api", "completion", "identity", "transport"):
        values = {_canonical(record[field]) for record in records}
        if len(values) != 1:
            raise ProbeError("cross-rank-evidence-mismatch")
    profile_ids = {record["precision"]["profile_id"] for record in records}
    if len(profile_ids) != 1:
        raise ProbeError("cross-rank-evidence-mismatch")
    precision: dict[str, Any] = {
        "profile_id": profile_ids.pop(),
        "correctness": _correctness_aggregate([
            record["precision"]["correctness"] for record in records
        ]),
    }
    for direction in ("dispatch", "combine"):
        axes = [record["precision"][direction] for record in records]
        fixed = (
            "accumulator_dtype", "accumulator_evidence", "api_input_dtype",
            "api_output_dtype", "communication_format",
        )
        result: dict[str, Any] = {}
        for field in fixed:
            if len({axis[field] for axis in axes}) != 1:
                raise ProbeError("cross-rank-evidence-mismatch")
            result[field] = axes[0][field]
        for field in ("runtime_input", "runtime_output", "semantic_output"):
            result[field] = _aggregate_tensor_summaries([axis[field] for axis in axes])
        result["scale_contract"] = _aggregate_scale_contracts([
            axis["scale_contract"] for axis in axes
        ])
        precision[direction] = result
    return {
        "api": records[0]["api"],
        "completion": records[0]["completion"],
        "identity": records[0]["identity"],
        "precision": precision,
        "transport": records[0]["transport"],
    }


def _finalize(backend, dist) -> None:
    if backend is not None:
        backend.finalize(0)
        return
    if dist.is_initialized():
        dist.destroy_process_group()


def run_target(target: dict[str, Any], output: Path) -> int:
    try:
        import torch
        import torch.distributed as dist
    except Exception as exc:  # pragma: no cover - diagnostic runtime requirement
        raise ProbeError("runtime-identity-mismatch") from exc
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    _init_distributed(torch, dist, target["backend"], device, rank, world_size)
    backend = None
    topology = capability.topology_for(target["sku"], target["ep"])
    topology_record = _topology_record(topology, False) if topology is not None else {
        "gpus_per_node": 1, "nodes": target["ep"], "placement_valid": False,
        "scale_up_domain": 1, "scale_up_transport": "unknown",
        "scale_out_transport": None, "scope": "scale-out",
        "topology_class": "unknown", "transport": "unknown", "world_size": target["ep"],
    }
    try:
        topology, placement, fingerprint = _runtime_context(
            torch, dist, target, device, local_rank
        )
        topology_record = _topology_record(topology, bool(placement["valid"]))
        args = _runtime_args(target, topology, fingerprint)
        try:
            backend = _backend_class(target["backend"])(
                args, rank, world_size, local_rank, device
            )
            construction = {"ok": True}
        except Exception:
            construction = {"ok": False, "reason": "backend-construction-failed"}
        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(gathered, construction)
        if not all(record.get("ok") is True for record in gathered):
            manifest = build_manifest(
                target=target, topology=topology_record, disposition="unsupported",
                reason="backend-construction-failed", runtime_executed=True, evidence=None,
            )
        else:
            try:
                local = {"ok": True, "evidence": _local_probe(
                    torch, dist, target, backend, args, rank
                )}
            except ProbeError as exc:
                local = {"ok": False, "reason": exc.reason}
            except Exception:
                local = {"ok": False, "reason": "native-operation-failed"}
            gathered = [None] * world_size
            dist.all_gather_object(gathered, local)
            if not all(record.get("ok") is True for record in gathered):
                reasons = {record.get("reason") for record in gathered}
                reason = reasons.pop() if len(reasons) == 1 else "cross-rank-evidence-mismatch"
                manifest = build_manifest(
                    target=target, topology=topology_record, disposition="unsupported",
                    reason=reason, runtime_executed=True, evidence=None,
                )
            else:
                evidence = _aggregate_local([record["evidence"] for record in gathered])
                manifest = build_manifest(
                    target=target, topology=topology_record, disposition="supported",
                    reason=SUPPORTED_REASON, runtime_executed=True, evidence=evidence,
                )
    except ProbeError as exc:
        manifest = build_manifest(
            target=target, topology=topology_record, disposition="unsupported",
            reason=exc.reason, runtime_executed=False, evidence=None,
        )
    if rank == 0:
        _write_atomic(output, manifest)
        print(json.dumps(manifest, allow_nan=False, sort_keys=True, separators=(",", ":")))
    dist.barrier()
    _finalize(backend, dist)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-targets", action="store_true")
    parser.add_argument("--workflow-plan", action="store_true")
    parser.add_argument("--extract-from", type=Path)
    parser.add_argument("--probe-id")
    parser.add_argument("--validate-control", type=Path)
    parser.add_argument("--validate-manifest", type=Path, nargs="+")
    parser.add_argument("--validate-bundle", type=Path)
    parser.add_argument("--backend", choices=sorted(BACKENDS | {"all"}))
    parser.add_argument("--sku")
    parser.add_argument("--only-sku", default="")
    parser.add_argument("--min-nodes", type=int, default=0)
    parser.add_argument("--max-nodes", type=int, default=0)
    parser.add_argument("--expect-sku")
    parser.add_argument("--expect-backend")
    parser.add_argument("--expect-nodes", type=int)
    parser.add_argument("--ep", type=int)
    parser.add_argument("--mode", choices=("normal", "low-latency"))
    parser.add_argument("--precision-profile")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.list_targets:
        print(json.dumps(provisional_targets(), allow_nan=False, sort_keys=True, separators=(",", ":")))
        return 0
    if args.workflow_plan:
        plan = workflow_plan(
            backend=args.backend or "all", only_sku=args.only_sku,
            min_nodes=args.min_nodes, max_nodes=args.max_nodes,
        )
        if args.out is None:
            print(json.dumps(plan, allow_nan=False, sort_keys=True, separators=(",", ":")))
        else:
            _write_atomic(args.out, plan)
        return 0
    if args.extract_from is not None:
        if None in (args.probe_id, args.expect_sku, args.expect_backend, args.expect_nodes, args.out):
            parser.error("probe extraction requires ID, expected placement, and --out")
        control = extract_control(
            json.loads(args.extract_from.read_text()), probe_id=args.probe_id,
            sku=args.expect_sku, backend=args.expect_backend, nodes=args.expect_nodes,
        )
        _write_atomic(args.out, control)
        return 0
    if args.validate_control is not None:
        if None in (args.expect_sku, args.expect_backend, args.expect_nodes):
            parser.error("control validation requires expected placement")
        validate_control(
            json.loads(args.validate_control.read_text()), sku=args.expect_sku,
            backend=args.expect_backend, nodes=args.expect_nodes,
        )
        return 0
    if args.validate_bundle is not None:
        if not args.validate_manifest:
            parser.error("bundle validation requires manifest paths")
        validate_bundle(
            json.loads(args.validate_bundle.read_text()),
            [json.loads(path.read_text()) for path in args.validate_manifest],
        )
        return 0
    if args.validate_manifest is not None:
        for path in args.validate_manifest:
            validate_manifest(json.loads(path.read_text()))
        return 0
    if any(
        value is None
        for value in (args.backend, args.sku, args.ep, args.mode, args.precision_profile, args.out)
    ):
        parser.error("one exact --backend/--sku/--ep/--mode/--precision-profile/--out cell is required")
    try:
        target = select_target(
            backend=args.backend, sku=args.sku, ep=args.ep, mode=args.mode,
            precision_profile=args.precision_profile,
        )
        return run_target(target, args.out)
    except ProbeError as exc:
        parser.error(exc.reason)


if __name__ == "__main__":
    raise SystemExit(main())
