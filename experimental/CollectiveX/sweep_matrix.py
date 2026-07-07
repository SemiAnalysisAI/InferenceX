#!/usr/bin/env python3
"""Resolve CollectiveX v1 suites and extract validated execution shards.

Mode changes measurement semantics and therefore participates in case identity.
Precision sensitivity uses allowlisted communication profiles; provisional native
paths remain outside the executable matrix until their probes are resolved.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tests"))

try:  # Shard extraction on GPU runners is intentionally stdlib-only.
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised by the workflow environment
    yaml = None

import capability as cap  # noqa: E402
import contracts  # noqa: E402
import ep_harness  # noqa: E402
import identity  # noqa: E402


EP_TIMING_PROFILE = (
    f"{ep_harness.TIMED_ITERS_PER_TRIAL}:{ep_harness.TRIALS_PER_POINT}:"
    f"{ep_harness.WARMUP_ITERS_PER_TRIAL}"
)
V1_WORKLOAD = ("deepseek-v3-v1", 7168, 8, 256)
V1_SUITE_CONTRACTS = {
    "ep-core-v1": {
        "mode": "normal",
        "publication": "official",
        "coordinates": {
            ("normal", "decode", "uniform", False),
            ("normal", "prefill", "uniform", False),
        },
        "ladders": {
            "decode": tuple(ep_harness.DECODE_LADDER),
            "prefill": (256, 512),
        },
    },
    "ep-low-latency-v1": {
        "mode": "low-latency",
        "publication": "official",
        "backends": {"deepep", "uccl"},
        "coordinates": {("low-latency", "decode", "uniform", False)},
        "ladders": {"decode": tuple(ep_harness.DECODE_LADDER)},
    },
    "ep-precision-normal-v1": {
        "mode": "normal",
        "publication": "comparable-experimental",
        "backends": {"deepep", "deepep-v2", "uccl", "deepep-hybrid", "mori"},
        "precision_profiles": identity.V1_NORMAL_PRECISION_PROFILE_IDS,
        "coordinates": {
            ("normal", "decode", "uniform", False),
            ("normal", "prefill", "uniform", False),
        },
        "ladders": {"decode": (128,), "prefill": (512,)},
    },
    "ep-precision-low-latency-v1": {
        "mode": "low-latency",
        "publication": "comparable-experimental",
        "backends": {"deepep", "uccl"},
        "precision_profiles": identity.V1_LOW_LATENCY_PRECISION_PROFILE_IDS,
        "coordinates": {("low-latency", "decode", "uniform", False)},
        "ladders": {"decode": (128,)},
    },
}
IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9.-]*")
SUITE_FIELDS = {
    "backends", "ep_degrees", "eplb", "mode", "phases", "platforms",
    "precision_profiles", "provisional", "required_publication", "routings",
    "token_points", "token_points_decode", "token_points_prefill", "workloads",
}
SUITE_REQUIRED = {
    "ep_degrees", "mode", "phases", "platforms", "required_publication", "routings",
    "workloads",
}
TOPOLOGY_FIELDS = (
    "nodes", "gpus_per_node", "scale_up_domain", "scope", "scale_up_transport",
    "scale_out_transport", "transport", "topology_class",
)
QUALIFICATION_INDICES = range(1, 4)
CANONICAL_V1_EXECUTION_PLAN_SHA256 = {
    1: "71faf69466bf3f35dac2defa04d4ea85434fbbde248631a9204fc8dcd8e2e37f",
    2: "8eae0bb33aacc56811e74e2ad80b24f3ea8c55e84bb2e50e76d4b23f3648f1c6",
    3: "9ac32155b789d3a9067d5c7fe2e23e8574431eed5041bb243a57ff886ae3e940",
}


class MatrixError(ValueError):
    """A matrix or shard-control document violates the execution contract."""


if yaml is not None:
    class _UniqueKeyLoader(yaml.SafeLoader):
        pass

    def _unique_mapping(loader: Any, node: Any, deep: bool = False) -> dict[Any, Any]:
        result: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in result:
                raise SystemExit(f"duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    _UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
    )


def _load(name: str) -> dict[str, Any]:
    if yaml is None:
        raise SystemExit("matrix generation requires PyYAML; shard extraction does not")
    try:
        with (HERE / "configs" / name).open() as fh:
            document = yaml.load(fh, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise SystemExit(f"configs/{name} is not valid YAML: {exc}") from exc
    if not isinstance(document, dict):
        raise SystemExit(f"configs/{name} must contain a YAML object")
    return document


def _workload_registry(workloads: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: cfg
        for section in ("synthetic", "model_derived")
        for name, cfg in (workloads.get(section) or {}).items()
    }


def _fields(value: Any, path: str, allowed: set[str], required: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SystemExit(f"{path} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise SystemExit(f"{path} field names must be strings")
    unknown, missing = set(value) - allowed, required - set(value)
    if unknown or missing:
        raise SystemExit(f"{path} fields: unknown={sorted(unknown)}, missing={sorted(missing)}")
    return value


def _list(value: Any, path: str, item_type: type, allowed: set[Any] | None = None) -> list[Any]:
    if (not isinstance(value, list) or not value
            or any(type(item) is not item_type for item in value)
            or len(value) != len(set(value))
            or (allowed is not None and any(item not in allowed for item in value))):
        raise SystemExit(f"{path} must be a non-empty unique list of valid {item_type.__name__}s")
    return value


def validate_config_documents(
    suites_document: dict[str, Any], workloads: dict[str, Any]
) -> None:
    """Reject configuration that is ambiguous, unused, or outside the v1 grid."""
    _fields(
        suites_document, "configs/suites.yaml",
        {"schema_version", "suites"}, {"schema_version", "suites"},
    )
    _fields(
        workloads, "configs/workloads.yaml",
        {"schema_version", "synthetic", "model_derived"}, {"schema_version"},
    )
    if type(suites_document["schema_version"]) is not int or suites_document["schema_version"] != 1:
        raise SystemExit("configs/suites.yaml schema_version must be integer 1")
    if type(workloads["schema_version"]) is not int or workloads["schema_version"] != 1:
        raise SystemExit("configs/workloads.yaml schema_version must be integer 1")
    registry: dict[str, dict[str, Any]] = {}
    for section, expert_field in (
        ("synthetic", "experts"),
        ("model_derived", "routed_experts"),
    ):
        entries = workloads.get(section, {})
        if not isinstance(entries, dict):
            raise SystemExit(f"workloads.{section} must be an object")
        for name, value in entries.items():
            if not isinstance(name, str) or not IDENTIFIER.fullmatch(name) or name in registry:
                raise SystemExit(f"workloads.{section} has invalid or duplicate name {name!r}")
            fields = {"hidden", "topk", expert_field, "verified_against"}
            config = _fields(value, f"workload {name}", fields, fields - {"verified_against"})
            dimensions = [config[key] for key in ("hidden", "topk", expert_field)]
            if any(type(item) is not int or item <= 0 for item in dimensions):
                raise SystemExit(f"workload {name} dimensions must be positive integers")
            if dimensions[1] > dimensions[2]:
                raise SystemExit(f"workload {name}.topk exceeds its expert count")
            source = config.get("verified_against")
            if source is not None and (not isinstance(source, str) or not source.strip()):
                raise SystemExit(f"workload {name}.verified_against must be a non-empty string")
            registry[name] = config
    if not registry:
        raise SystemExit("configs/workloads.yaml must define at least one workload")

    suites = suites_document["suites"]
    if not isinstance(suites, dict) or not suites:
        raise SystemExit("configs/suites.yaml suites must be a non-empty object")
    referenced: set[str] = set()
    for name, value in suites.items():
        if not isinstance(name, str) or not IDENTIFIER.fullmatch(name):
            raise SystemExit(f"invalid suite name {name!r}")
        suite = _fields(value, f"suite {name}", SUITE_FIELDS, SUITE_REQUIRED)
        contract = V1_SUITE_CONTRACTS.get(name)
        if contract is None:
            raise SystemExit(f"suite {name} is outside the frozen v1 catalog")
        mode = suite["mode"]
        if mode not in identity.V1_CASE_PROFILES or mode != contract["mode"]:
            raise SystemExit(f"suite {name}.mode differs from the frozen v1 catalog")
        suite_backends = _list(
            suite.get("backends", list(cap.SWEEP_BACKENDS)),
            f"suite {name}.backends",
            str,
            set(cap.SWEEP_BACKENDS),
        )
        expected_backends = contract.get("backends")
        if expected_backends is not None and set(suite_backends) != expected_backends:
            raise SystemExit(f"suite {name}.backends differs from the frozen v1 catalog")
        if expected_backends is None and "backends" in suite:
            raise SystemExit(f"suite {name}.backends must be omitted")
        expected_profiles = contract.get("precision_profiles")
        if expected_profiles is None:
            if "precision_profiles" in suite or "provisional" in suite:
                raise SystemExit(
                    f"suite {name} cannot add precision fields to a baseline suite"
                )
            precision_profiles: list[str] = []
        else:
            precision_profiles = _list(
                suite.get("precision_profiles"),
                f"suite {name}.precision_profiles",
                str,
                set(identity.V1_PRECISION_PROFILES),
            )
            if tuple(precision_profiles) != expected_profiles:
                raise SystemExit(
                    f"suite {name}.precision_profiles differs from the frozen v1 catalog"
                )
            if identity.V1_CONTROL_PRECISION_PROFILE in precision_profiles:
                raise SystemExit(
                    f"suite {name} must reference existing BF16 evidence, not duplicate it"
                )
            if any(
                mode not in identity.V1_PRECISION_PROFILES[profile]["modes"]
                for profile in precision_profiles
            ):
                raise SystemExit(f"suite {name} contains a precision profile for another mode")
            if type(suite.get("provisional")) is not bool:
                raise SystemExit(f"suite {name}.provisional must be a boolean")
            unresolved = cap.provisional_precision_targets(precision_profiles)
            if suite["provisional"] != bool(unresolved):
                raise SystemExit(
                    f"suite {name}.provisional must track unresolved capability targets"
                )
            candidates = cap.precision_targets(precision_profiles)
            covered_candidates = [
                target for target in candidates
                if target["backend"] in suite_backends
                and target["sku"] in suite["platforms"]
                and target["ep"] in suite["ep_degrees"]
                and target["mode"] == mode
            ]
            if covered_candidates != candidates:
                raise SystemExit(
                    f"suite {name} does not cover every declared precision target"
                )
        suite_workloads = _list(suite["workloads"], f"suite {name}.workloads", str)
        unknown = sorted(set(suite_workloads) - set(registry))
        if unknown:
            raise SystemExit(f"suite {name}: unknown workloads {unknown}")
        referenced.update(suite_workloads)
        platforms = _list(
            suite["platforms"], f"suite {name}.platforms", str, set(cap.PLATFORMS)
        )
        phases = _list(suite["phases"], f"suite {name}.phases", str, {"decode", "prefill"})
        routings = _list(suite["routings"], f"suite {name}.routings", str, {"uniform"})
        eplb = _list(suite.get("eplb", [False]), f"suite {name}.eplb", bool)
        if True in eplb:
            raise SystemExit(f"suite {name}: EPLB is unavailable for v1 uniform routing")
        if suite["required_publication"] not in {"official", "comparable-experimental"}:
            raise SystemExit(f"suite {name}.required_publication is invalid")
        if suite["required_publication"] != contract["publication"]:
            raise SystemExit(
                f"suite {name}.required_publication differs from the frozen v1 catalog"
            )
        if suite["required_publication"] == "official":
            unverified = [item for item in suite_workloads if not registry[item].get("verified_against")]
            if unverified:
                raise SystemExit(f"suite {name}: official workloads need verified_against: {unverified}")
        degrees = _list(suite["ep_degrees"], f"suite {name}.ep_degrees", int)
        if degrees != [8, 16]:
            raise SystemExit(f"suite {name}.ep_degrees must be exactly [8, 16]")
        for platform in platforms:
            if not set(degrees).issubset(cap.PLATFORMS[platform]["ep_degrees"]):
                raise SystemExit(f"suite {name}: invalid EP degree for {platform}")
        for phase in {"decode", "prefill"} - set(phases):
            if f"token_points_{phase}" in suite:
                raise SystemExit(f"suite {name}.token_points_{phase} is unreachable")
        if "token_points" in suite and all(
            f"token_points_{phase}" in suite for phase in phases
        ):
            raise SystemExit(f"suite {name}.token_points is unreachable")
        for phase in phases:
            _ladder(suite, phase)
        coordinates = {
            (mode, phase, routing, enabled)
            for phase, routing, enabled in itertools.product(phases, routings, eplb)
        }
        if coordinates != contract["coordinates"] or any(
            tuple(map(int, _ladder(suite, phase).split())) != contract["ladders"][phase]
            for phase in phases
        ):
            raise SystemExit(f"suite {name} coordinates differ from the frozen v1 catalog")
    unused = sorted(set(registry) - referenced)
    if unused:
        raise SystemExit(f"unreferenced workloads: {unused}")


def _dims(workloads: dict[str, Any], name: str) -> tuple[int, int, int]:
    config = _workload_registry(workloads)[name]
    values = (
        config.get("hidden"),
        config.get("topk"),
        config.get("experts", config.get("routed_experts")),
    )
    return values  # type: ignore[return-value]


def _ladder(suite: dict[str, Any], phase: str) -> str:
    points = suite.get(f"token_points_{phase}", suite.get("token_points"))
    if points is None:
        points = ep_harness.DECODE_LADDER if phase == "decode" else ep_harness.PREFILL_LADDER
    if (not isinstance(points, list) or not points
            or any(isinstance(point, bool) or not isinstance(point, int) or point <= 0
                   for point in points)
            or points != sorted(set(points))):
        raise SystemExit(f"invalid {phase} token ladder: {points!r}")
    return " ".join(map(str, points))


def _v1_requested_ladder(case: dict[str, Any]) -> str:
    """Bind extracted controls to the frozen v1 suite and workload catalog."""
    suite = V1_SUITE_CONTRACTS.get(case.get("suite"))
    expected_profiles = None if suite is None else suite.get("precision_profiles")
    precision_profile = case.get("precision_profile")
    coordinate = (
        case.get("mode"), case.get("phase"), case.get("routing"), case.get("eplb")
    )
    if (
        suite is None
        or coordinate not in suite["coordinates"]
        or case.get("required_publication") != suite["publication"]
        or (
            case.get("workload"), case.get("hidden"), case.get("topk"), case.get("experts")
        ) != V1_WORKLOAD
        or (expected_profiles is None and precision_profile is not None)
        or (expected_profiles is not None and precision_profile not in expected_profiles)
    ):
        raise MatrixError("case differs from the frozen v1 suite/workload catalog")
    return " ".join(map(str, suite["ladders"][case["phase"]]))


def _expected_disposition(
    sku: str, case: dict[str, Any]
) -> tuple[str, str | None, str | None]:
    requested_ladder = _v1_requested_ladder(case)
    precision_profile = case.get("precision_profile")
    if precision_profile is not None and not cap.precision_target_declared(
        precision_profile,
        sku=sku,
        backend=case["backend"],
        ep=case["ep"],
        mode=case["mode"],
    ):
        raise MatrixError("precision case is not an exact native capability target")
    disposition, detail = cap.resolve_disposition(
        sku, case["backend"], ep=case["ep"], nodes=case["nodes"],
        routing=case["routing"], eplb=case["eplb"], mode=case["mode"],
        precision_profile=precision_profile,
    )
    if disposition == "supported":
        if case["ladder"] != requested_ladder:
            raise MatrixError("case ladder differs from the frozen v1 suite catalog")
        return "runnable", None, None
    if case["ladder"] != requested_ladder:
        raise MatrixError("unsupported case ladder differs from the frozen v1 suite catalog")
    if disposition == "unsupported":
        reason = (
            "precision-profile-unsupported"
            if precision_profile is not None
            else "backend-platform-unsupported"
        )
        return "unsupported", reason, detail
    if disposition == "provisional":
        raise MatrixError("provisional precision target entered the executable matrix")
    raise MatrixError("not-applicable precision tuple entered the requested matrix")


def _case_id(sku: str, case: dict[str, Any]) -> str:
    return identity.case_id(
        sku=sku, profile=identity.profile_for_case(case), case=case
    )


def _semantic_points(sku: str, case: dict[str, Any]) -> list[str]:
    execution = {
        key: value for key, value in case.items()
        if key not in {"canonical", "case_id", "ladder", "required_publication", "suite", "workload"}
    }
    return [
        json.dumps(
            {"sku": sku, "tokens_per_rank": int(point), **execution},
            sort_keys=True,
            separators=(",", ":"),
        )
        for point in case["ladder"].split()
    ]


def _select_backends(backend: str, backends: str) -> list[str]:
    available = list(cap.SWEEP_BACKENDS)
    if backend and backends:
        raise SystemExit("--backend and --backends are mutually exclusive")
    if backends:
        names = available if backends == "all" else [
            value.strip() for value in backends.split(",") if value.strip()
        ]
    else:
        names = [backend or "deepep"]
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise SystemExit(f"unknown backend values {unknown}; have {available}")
    if len(names) != len(set(names)):
        raise SystemExit("backend selection contains duplicates")
    return names


def resolve_matrix(
    suites: str = "all",
    backend: str = "",
    backends: str = "",
    only_sku: str = "",
    min_nodes: int = 0,
    max_nodes: int = 0,
    max_cases: int = 128,
) -> dict[str, Any]:
    """Resolve suite configuration into allocation-sized workflow shards."""
    if max_cases <= 0:
        raise SystemExit("--max-cases must be positive")
    if min_nodes < 0 or max_nodes < 0 or (min_nodes and max_nodes and min_nodes > max_nodes):
        raise SystemExit("invalid node bounds")
    if only_sku and only_sku not in cap.PLATFORMS:
        raise SystemExit(f"unknown --only-sku {only_sku!r}; have {sorted(cap.PLATFORMS)}")

    workloads = _load("workloads.yaml")
    suites_document = _load("suites.yaml")
    validate_config_documents(suites_document, workloads)
    registry = suites_document["suites"]
    select_all = suites == "all"
    names = (
        [name for name, suite in registry.items() if not suite.get("provisional", False)]
        if select_all
        else [value.strip() for value in suites.split(",") if value.strip()]
    )
    if not names or len(names) != len(set(names)):
        raise SystemExit("suite selection must be non-empty and unique")
    unknown = sorted(set(names) - set(registry))
    if unknown:
        raise SystemExit(f"unknown suites {unknown}; have {sorted(registry)}")
    blocked = [name for name in names if registry[name].get("provisional", False)]
    if blocked:
        unresolved = sum(
            len(cap.provisional_precision_targets(registry[name]["precision_profiles"]))
            for name in blocked
        )
        raise SystemExit(
            f"provisional precision suites cannot be scheduled: {blocked}; "
            f"resolve {unresolved} capability targets first"
        )
    targets = _select_backends(backend, backends)

    shards: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    requested_cases: list[dict[str, Any]] = []
    scheduled: set[str] = set()
    for suite_name in names:
        suite = registry[suite_name]
        mode = suite["mode"]
        phases = suite["phases"]
        routings = suite["routings"]
        eplb_values = suite.get("eplb", [False])
        precision_profiles = suite.get("precision_profiles", [None])
        suite_backends = set(suite.get("backends", cap.SWEEP_BACKENDS))
        suite_targets = [target for target in targets if target in suite_backends]
        if not suite_targets:
            continue
        for platform_name in suite["platforms"]:
            if only_sku and platform_name != only_sku:
                continue
            ep_degrees = suite["ep_degrees"]
            for workload, ep, phase, routing, eplb, target, precision_profile in itertools.product(
                suite["workloads"], ep_degrees, phases, routings, eplb_values,
                suite_targets, precision_profiles,
            ):
                if precision_profile is not None and not cap.precision_target_declared(
                    precision_profile,
                    sku=platform_name,
                    backend=target,
                    ep=ep,
                    mode=mode,
                ):
                    continue
                topology = cap.topology_for(platform_name, ep)
                if topology is None:
                    raise SystemExit(
                        f"suite {suite_name}: {platform_name} EP{ep} is not registered"
                    )
                nodes = int(topology["nodes"])
                if min_nodes and nodes < min_nodes:
                    continue
                if max_nodes and nodes > max_nodes:
                    continue
                capability_disposition, capability_detail = cap.resolve_disposition(
                    platform_name,
                    target,
                    ep=ep,
                    nodes=nodes,
                    routing=routing,
                    eplb=bool(eplb),
                    mode=mode,
                    precision_profile=precision_profile,
                )
                hidden, topk, experts = _dims(workloads, workload)

                def add_case(
                    case_ladder: str,
                    disposition: str,
                    reason: str | None,
                    detail: str | None,
                ) -> None:
                    case: dict[str, Any] = {
                        "suite": suite_name,
                        "workload": workload,
                        "required_publication": suite["required_publication"],
                        "backend": target,
                        "routing": routing,
                        "phase": phase,
                        "ep": ep,
                        "eplb": eplb,
                        "hidden": hidden,
                        "topk": topk,
                        "experts": experts,
                        "samples_per_point": ep_harness.TIMED_SAMPLES_PER_POINT,
                        "warmup_semantics": ep_harness.WARMUP_SEMANTICS,
                        "ladder": case_ladder,
                        "mode": mode,
                        "timing": EP_TIMING_PROFILE,
                        "canonical": True,
                        **{field: topology[field] for field in TOPOLOGY_FIELDS},
                    }
                    if precision_profile is not None:
                        case["precision_profile"] = precision_profile
                    for signature in _semantic_points(platform_name, case):
                        if signature in scheduled:
                            raise SystemExit(
                                f"suite {suite_name}: duplicate semantic point for {platform_name}"
                            )
                        scheduled.add(signature)
                    case["case_id"] = _case_id(platform_name, case)
                    requested_cases.append(
                        {
                            "sku": platform_name,
                            "case": case,
                            "disposition": disposition,
                            "reason": reason,
                            "detail": detail,
                        }
                    )
                    if disposition == "runnable":
                        shards.setdefault((platform_name, target, nodes), []).append(case)

                requested_ladder = _ladder(suite, phase)
                if capability_disposition == "not-applicable":
                    continue
                if capability_disposition == "provisional":
                    raise SystemExit(
                        f"suite {suite_name}: provisional target escaped its suite gate: "
                        f"{precision_profile} {target} {platform_name} EP{ep}"
                    )
                if capability_disposition == "unsupported":
                    add_case(
                        requested_ladder,
                        "unsupported",
                        (
                            "precision-profile-unsupported"
                            if precision_profile is not None
                            else "backend-platform-unsupported"
                        ),
                        capability_detail,
                    )
                    continue
                if capability_disposition != "supported":
                    raise SystemExit(
                        f"suite {suite_name}: invalid capability disposition "
                        f"{capability_disposition!r}"
                    )
                add_case(requested_ladder, "runnable", None, None)

    shards_by_sku: dict[str, list[dict[str, Any]]] = {}
    for (sku, target, nodes), cases in sorted(shards.items()):
        chunk_size = max_cases
        for offset in range(0, len(cases), chunk_size):
            chunk = cases[offset:offset + chunk_size]
            part = offset // chunk_size
            shard_id = f"{sku}-{target}-n{nodes}"
            if len(cases) > chunk_size:
                shard_id += f"-p{part}"
            shards_by_sku.setdefault(sku, []).append({
                "id": shard_id,
                "sku": sku,
                "backend": target,
                "launcher": cap.PLATFORMS[sku]["launcher"],
                **{field: chunk[0][field] for field in TOPOLOGY_FIELDS},
                "n": len(chunk),
                "execution_weight": execution_weight(chunk),
                "case_ids": [case["case_id"] for case in chunk],
            })
    include = [
        shards_by_sku[sku][round_index]
        for round_index in range(max(map(len, shards_by_sku.values()), default=0))
        for sku in sorted(shards_by_sku)
        if round_index < len(shards_by_sku[sku])
    ]
    return {
        "format": "collectivex.matrix.v1",
        "schema_version": 1,
        "requested_cases": requested_cases,
        "include": include,
    }


def _strict_json_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise MatrixError(f"non-finite JSON number {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MatrixError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    if not path.is_file():
        raise MatrixError(f"matrix does not exist: {path}")
    if path.stat().st_size == 0:
        raise MatrixError(f"matrix is empty: {path}")
    try:
        with path.open() as fh:
            return json.load(
                fh, parse_constant=reject_constant, object_pairs_hook=reject_duplicates
            )
    except (OSError, json.JSONDecodeError) as exc:
        raise MatrixError(f"matrix is not valid JSON: {exc}") from exc


def _positive_int(value: Any, field: str) -> int:
    if type(value) is not int:
        raise MatrixError(f"{field} must be a positive integer")
    if value <= 0:
        raise MatrixError(f"{field} must be a positive integer")
    return value


def _qualification_index(value: Any, field: str = "qualification_index") -> int:
    if type(value) is not int or value not in QUALIFICATION_INDICES:
        raise MatrixError(f"{field} must be an integer in 1..3")
    return value


def _requested_qualification_index(value: int | None = None) -> int:
    if value is not None:
        return _qualification_index(value)
    raw = os.environ.get("CX_QUALIFICATION_INDEX", "1")
    if raw not in {"1", "2", "3"}:
        raise MatrixError("CX_QUALIFICATION_INDEX must be an integer in 1..3")
    return int(raw)


def _case_precision_profile(case: dict[str, Any]) -> str:
    profile = case.get("precision_profile", identity.V1_CONTROL_PRECISION_PROFILE)
    if not isinstance(profile, str) or profile not in identity.V1_PRECISION_PROFILES:
        raise MatrixError("qualification case has an invalid precision profile")
    return profile


def _qualification_digest(
    shard_id: str,
    case_id: str,
    profile_id: str,
    qualification_index: int,
) -> bytes:
    return hashlib.sha256(
        "\0".join((shard_id, case_id, profile_id, str(qualification_index))).encode()
    ).digest()


def _rotate(values: list[Any], offset: int) -> list[Any]:
    if not values:
        return []
    position = offset % len(values)
    return values[position:] + values[:position]


def _seeded_qualification_order(
    shard_id: str,
    cases: list[dict[str, Any]],
    qualification_index: int,
) -> list[dict[str, Any]]:
    by_profile: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        by_profile.setdefault(_case_precision_profile(case), []).append(case)
    profiles = sorted(
        by_profile,
        key=lambda profile: _qualification_digest(
            shard_id, "profile-order", profile, 1
        ),
    )
    profiles = _rotate(profiles, qualification_index - 1)
    groups: dict[str, list[dict[str, Any]]] = {}
    for profile in profiles:
        group = sorted(
            by_profile[profile],
            key=lambda case: _qualification_digest(
                shard_id, case["case_id"], profile, qualification_index
            ),
        )
        groups[profile] = _rotate(group, qualification_index - 1)
    interleaved = [
        groups[profile][position]
        for position in range(max(map(len, groups.values())))
        for profile in profiles
        if position < len(groups[profile])
    ]
    return interleaved


def qualification_execution_order(
    shard_id: str,
    cases: list[dict[str, Any]],
    qualification_index: int,
) -> list[dict[str, Any]]:
    """Return one deterministic, repeat-specific permutation of a shard's cases."""
    index = _qualification_index(qualification_index)
    if not isinstance(shard_id, str) or not IDENTIFIER.fullmatch(shard_id):
        raise MatrixError("qualification shard ID is invalid")
    if not isinstance(cases, list) or not cases:
        raise MatrixError("qualification planning requires at least one case")
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for current in range(1, index + 1):
        candidate = _seeded_qualification_order(shard_id, cases, current)
        signature = tuple(case["case_id"] for case in candidate)
        if len(cases) >= current and signature in seen:
            for offset in range(1, len(candidate)):
                rotated = _rotate(candidate, offset)
                rotated_signature = tuple(case["case_id"] for case in rotated)
                if rotated_signature not in seen:
                    candidate = rotated
                    signature = rotated_signature
                    break
        seen.add(signature)
        selected = candidate
    return selected


def execution_plan_sha256(cases: list[dict[str, Any]]) -> str:
    """Bind an execution plan to only its ordered case and precision-profile IDs."""
    plan = [
        [case["case_id"], _case_precision_profile(case)]
        for case in cases
    ]
    payload = json.dumps(plan, ensure_ascii=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def execution_weight(cases: list[dict[str, Any]]) -> int:
    """Return deterministic GPU-point work used to bound workflow parallelism."""
    if not isinstance(cases, list) or not cases:
        raise MatrixError("execution weight requires at least one case")
    weight = 0
    for case in cases:
        ep = _positive_int(case.get("ep"), "execution-weight.ep")
        ladder = case.get("ladder")
        if not isinstance(ladder, str) or not ladder.split():
            raise MatrixError("execution weight requires a token ladder")
        weight += ep * len(ladder.split())
    return weight


def qualification_execution_plan_sha256(
    matrix: dict[str, Any], qualification_index: int
) -> str:
    """Bind one qualification repeat to every shard's ordered case plan."""
    index = _qualification_index(qualification_index)
    document = validate_matrix_document(matrix)
    requested = {
        item["case"]["case_id"]: item["case"]
        for item in document["requested_cases"]
    }
    plan = []
    for shard in sorted(document["include"], key=lambda item: item["id"]):
        ordered = qualification_execution_order(
            shard["id"],
            [requested[case_id] for case_id in shard["case_ids"]],
            index,
        )
        plan.append([
            shard["id"], shard["execution_weight"], execution_plan_sha256(ordered),
        ])
    payload = json.dumps(
        {"qualification_index": index, "shards": plan},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_shard_control(
    shard: dict[str, Any],
    *,
    sku: str,
    backend: str,
    nodes: int,
    require_runnable: bool = True,
    qualification_index: int | None = None,
) -> None:
    """Validate one shard against the workflow cell that requested it."""
    if not isinstance(shard, dict):
        raise MatrixError("shard must be a JSON object")
    if sku not in cap.PLATFORMS or backend not in cap.SWEEP_BACKENDS:
        raise MatrixError("shard platform/backend is not registered")
    top_fields = {
        "schema_version", "id", "sku", "backend", "nodes", "n", "cases",
        "qualification_index", "execution_plan_sha256", "execution_weight",
    }
    if (
        set(shard) != top_fields
        or type(shard.get("schema_version")) is not int
        or shard["schema_version"] != 1
    ):
        raise MatrixError("shard fields or schema version differ from v1 contract")
    if not isinstance(shard.get("id"), str) or not IDENTIFIER.fullmatch(shard["id"]):
        raise MatrixError("shard has invalid id")
    observed_qualification = _qualification_index(
        shard.get("qualification_index"), "shard.qualification_index"
    )
    if (
        qualification_index is not None
        and observed_qualification != _qualification_index(qualification_index)
    ):
        raise MatrixError("shard qualification_index differs from the requested repeat")
    if (
        not isinstance(shard.get("execution_plan_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", shard["execution_plan_sha256"]) is None
    ):
        raise MatrixError("shard execution_plan_sha256 is invalid")
    for field, expected in (("sku", sku), ("backend", backend)):
        if shard.get(field) != expected:
            raise MatrixError(
                f"shard {field} mismatch: expected {expected!r}, got {shard.get(field)!r}"
            )
    if _positive_int(shard.get("nodes"), "shard.nodes") != nodes:
        raise MatrixError(
            f"shard nodes mismatch: expected {nodes}, got {shard.get('nodes')!r}"
        )
    cases = shard.get("cases")
    if not isinstance(cases, list) or not cases:
        raise MatrixError("shard must contain at least one case")
    if _positive_int(shard.get("n"), "shard.n") != len(cases):
        raise MatrixError("shard.n does not match the number of cases")
    seen: set[str] = set()
    base_required = {
        "case_id", "suite", "workload", "required_publication", "backend", "routing",
        "mode", "phase", "ep", "eplb", "hidden", "topk", "experts",
        "samples_per_point",
        "warmup_semantics", "ladder", "timing", "canonical",
    } | set(TOPOLOGY_FIELDS)
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise MatrixError(f"case {index} must be a JSON object")
        suite_contract = V1_SUITE_CONTRACTS.get(case.get("suite"))
        required = base_required | (
            {"precision_profile"}
            if suite_contract is not None and "precision_profiles" in suite_contract
            else set()
        )
        fields = set(case)
        if fields != required:
            raise MatrixError(
                f"case {index} fields differ from v1 contract: "
                f"missing={sorted(required - fields)}, extra={sorted(fields - required)}"
            )
        case_id = case["case_id"]
        if not identity.is_typed_id(case_id, "case"):
            raise MatrixError(f"case {index} has invalid case_id")
        if case_id in seen:
            raise MatrixError(f"duplicate case_id {case_id}")
        seen.add(case_id)
        string_fields = [
            "suite", "workload", "required_publication", "backend", "mode", "routing",
            "phase", "warmup_semantics", "ladder", "timing",
        ]
        if "precision_profile" in required:
            string_fields.append("precision_profile")
        for field in string_fields:
            if not isinstance(case[field], str) or not case[field]:
                raise MatrixError(f"case {index}.{field} must be a non-empty string")
        identifier_fields = [
            "suite", "workload", "required_publication", "backend", "routing", "phase",
        ]
        if "precision_profile" in required:
            identifier_fields.append("precision_profile")
        for field in identifier_fields:
            if not IDENTIFIER.fullmatch(case[field]):
                raise MatrixError(f"case {index}.{field} is not a safe identifier")
        if case["required_publication"] not in {"official", "comparable-experimental"}:
            raise MatrixError(f"case {index} has invalid publication requirement")
        case_identity = {key: value for key, value in case.items() if key != "case_id"}
        if case_id != _case_id(sku, case_identity):
            raise MatrixError(f"case {index} case_id does not match its contents")
        if case["backend"] != backend:
            raise MatrixError(f"case {index} backend does not match shard")
        if case["mode"] not in identity.V1_CASE_PROFILES:
            raise MatrixError(f"case {index} mode is invalid")
        if _positive_int(case["nodes"], f"case {index}.nodes") != nodes:
            raise MatrixError(f"case {index} nodes does not match shard")
        ep = _positive_int(case["ep"], f"case {index}.ep")
        gpus_per_node = _positive_int(
            case["gpus_per_node"], f"case {index}.gpus_per_node"
        )
        topology = cap.topology_for(sku, ep)
        if topology is None or any(case[field] != topology[field] for field in TOPOLOGY_FIELDS):
            raise MatrixError(f"case {index} differs from the platform registry")
        if ep != nodes * gpus_per_node:
            raise MatrixError(f"case {index} ep does not equal nodes * gpus_per_node")
        if case["samples_per_point"] != ep_harness.TIMED_SAMPLES_PER_POINT:
            raise MatrixError(f"case {index} violates fixed-512-v1")
        if case["timing"] != EP_TIMING_PROFILE:
            raise MatrixError(f"case {index} has invalid timing profile")
        if case["warmup_semantics"] != ep_harness.WARMUP_SEMANTICS:
            raise MatrixError(f"case {index} has invalid warmup semantics")
        if case["phase"] not in {"decode", "prefill"}:
            raise MatrixError(f"case {index} has invalid phase")
        if case["routing"] != "uniform":
            raise MatrixError(f"case {index} has invalid routing")
        if not isinstance(case["eplb"], bool) or case["eplb"]:
            raise MatrixError(f"case {index} has invalid EPLB setting")
        if not isinstance(case["canonical"], bool) or not case["canonical"]:
            raise MatrixError(f"case {index} must use a canonical workload")
        for field in ("ep", "nodes", "gpus_per_node", "hidden", "topk", "experts",
                      "samples_per_point", "scale_up_domain"):
            if isinstance(case[field], bool) or not isinstance(case[field], int):
                raise MatrixError(f"case {index}.{field} must be an integer")
            _positive_int(case[field], f"case {index}.{field}")
        scale_up_domain = _positive_int(
            case["scale_up_domain"], f"case {index}.scale_up_domain"
        )
        expected_scope = "scale-up" if ep <= scale_up_domain else "scale-out"
        if case["scope"] != expected_scope or (
            expected_scope == "scale-out" and ep % scale_up_domain
        ):
            raise MatrixError(f"case {index} has invalid scale-up/scale-out geometry")
        try:
            ladder = [int(value) for value in case["ladder"].split()]
        except (AttributeError, ValueError) as exc:
            raise MatrixError(f"case {index} has invalid token ladder") from exc
        if (not ladder or any(value <= 0 for value in ladder)
                or ladder != sorted(set(ladder))
                or case["ladder"] != " ".join(map(str, ladder))):
            raise MatrixError(f"case {index} has invalid token ladder")
        if require_runnable:
            disposition, reason, _ = _expected_disposition(sku, case)
            if disposition != "runnable":
                raise MatrixError(f"case {index} violates capability registry: {reason}")
        else:
            _v1_requested_ladder(case)
    if _positive_int(
        shard.get("execution_weight"), "shard.execution_weight"
    ) != execution_weight(cases):
        raise MatrixError("shard execution_weight differs from its cases")
    expected_order = qualification_execution_order(
        shard["id"], cases, observed_qualification
    )
    if [case["case_id"] for case in cases] != [
        case["case_id"] for case in expected_order
    ]:
        raise MatrixError("shard cases differ from the qualification execution order")
    if shard["execution_plan_sha256"] != execution_plan_sha256(cases):
        raise MatrixError("shard execution_plan_sha256 differs from its ordered cases")


def validate_matrix_document(document: Any) -> dict[str, Any]:
    """Validate the complete requested grid and its runnable shard partition."""
    if not isinstance(document, dict) or set(document) != {
        "format", "schema_version", "requested_cases", "include"
    }:
        raise MatrixError("matrix fields differ from the v1 contract")
    if (
        document["format"] != "collectivex.matrix.v1"
        or type(document["schema_version"]) is not int
        or document["schema_version"] != 1
    ):
        raise MatrixError("matrix format/schema differs from v1")
    requested = document["requested_cases"]
    include = document["include"]
    if not isinstance(requested, list) or not requested:
        raise MatrixError("matrix.requested_cases must be non-empty")
    if not isinstance(include, list):
        raise MatrixError("matrix.include must be an array")

    cases_by_id: dict[str, dict[str, Any]] = {}
    runnable_ids: set[str] = set()
    semantic_points: set[str] = set()
    for index, value in enumerate(requested):
        path = f"matrix.requested_cases[{index}]"
        if not isinstance(value, dict) or set(value) != {
            "sku", "case", "disposition", "reason", "detail"
        }:
            raise MatrixError(f"{path} fields differ from the v1 contract")
        sku = value["sku"]
        case = value["case"]
        disposition = value["disposition"]
        if sku not in cap.PLATFORMS:
            raise MatrixError(f"{path}.sku is unknown")
        if disposition not in {"runnable", "unsupported"}:
            raise MatrixError(f"{path}.disposition is invalid")
        if disposition == "runnable":
            if value["reason"] is not None or value["detail"] is not None:
                raise MatrixError(f"{path} runnable cases cannot have a reason")
        else:
            if (
                not isinstance(value["reason"], str)
                or not IDENTIFIER.fullmatch(value["reason"])
                or not isinstance(value["detail"], str)
                or not value["detail"]
            ):
                raise MatrixError(f"{path} unsupported cases need a public reason and detail")
        if not isinstance(case, dict):
            raise MatrixError(f"{path}.case must be an object")
        backend = case.get("backend")
        nodes = case.get("nodes")
        if not isinstance(backend, str) or type(nodes) is not int:
            raise MatrixError(f"{path}.case backend/nodes are invalid")
        requested_case_plan = [case]
        validate_shard_control(
            {
                "schema_version": 1,
                "id": "requested-case",
                "sku": sku,
                "backend": backend,
                "nodes": nodes,
                "n": 1,
                "execution_weight": execution_weight(requested_case_plan),
                "qualification_index": 1,
                "execution_plan_sha256": execution_plan_sha256(
                    requested_case_plan
                ),
                "cases": requested_case_plan,
            },
            sku=sku,
            backend=backend,
            nodes=nodes,
            require_runnable=disposition == "runnable",
        )
        case_id = case["case_id"]
        if case_id in cases_by_id:
            raise MatrixError(f"duplicate requested case_id {case_id}")
        for signature in _semantic_points(sku, case):
            if signature in semantic_points:
                raise MatrixError(f"{path} duplicates a semantic token point")
            semantic_points.add(signature)
        cases_by_id[case_id] = value
        expected = _expected_disposition(sku, case)
        if (disposition, value["reason"], value["detail"]) != expected:
            raise MatrixError(f"{path} disposition differs from the frozen v1 catalog")
        if disposition == "runnable":
            runnable_ids.add(case_id)

    shard_ids: set[str] = set()
    assigned: list[str] = []
    for index, shard in enumerate(include):
        path = f"matrix.include[{index}]"
        expected = {
            "id", "sku", "backend", "launcher", "n", "execution_weight", "case_ids",
        } | set(TOPOLOGY_FIELDS)
        if not isinstance(shard, dict) or set(shard) != expected:
            raise MatrixError(f"{path} fields differ from the v1 contract")
        shard_id = shard["id"]
        if not isinstance(shard_id, str) or not IDENTIFIER.fullmatch(shard_id):
            raise MatrixError(f"{path}.id is invalid")
        if shard_id in shard_ids:
            raise MatrixError(f"duplicate shard id {shard_id}")
        shard_ids.add(shard_id)
        sku = shard["sku"]
        if sku not in cap.PLATFORMS:
            raise MatrixError(f"{path}.sku is unknown")
        platform = cap.PLATFORMS[sku]
        if shard["launcher"] != platform["launcher"]:
            raise MatrixError(f"{path}.launcher differs from the platform registry")
        case_ids = shard["case_ids"]
        if not isinstance(case_ids, list) or not case_ids or len(case_ids) != len(set(case_ids)):
            raise MatrixError(f"{path}.case_ids must be a non-empty unique array")
        if _positive_int(shard["n"], f"{path}.n") != len(case_ids):
            raise MatrixError(f"{path}.n differs from case_ids")
        nodes = _positive_int(shard["nodes"], f"{path}.nodes")
        for case_id in case_ids:
            wrapper = cases_by_id.get(case_id)
            if wrapper is None or wrapper["disposition"] != "runnable":
                raise MatrixError(f"{path} references a missing or unsupported case")
            case = wrapper["case"]
            if (
                wrapper["sku"] != sku
                or case["backend"] != shard["backend"]
                or case["nodes"] != nodes
                or any(shard[field] != case[field] for field in TOPOLOGY_FIELDS)
            ):
                raise MatrixError(f"{path} case does not match shard coordinates")
            assigned.append(case_id)
        if shard["execution_weight"] != execution_weight(
            [cases_by_id[case_id]["case"] for case_id in case_ids]
        ):
            raise MatrixError(f"{path}.execution_weight differs from its cases")
    if len(assigned) != len(set(assigned)):
        raise MatrixError("a runnable case is assigned to more than one shard")
    if set(assigned) != runnable_ids:
        raise MatrixError("runnable requested cases and shard assignments differ")
    return document


def extract_shard(
    matrix_path: str | os.PathLike[str],
    shard_id: str,
    output_path: str | os.PathLike[str],
    *,
    sku: str,
    backend: str,
    nodes: int,
    qualification_index: int | None = None,
) -> dict[str, Any]:
    """Extract one strictly matched shard control file, writing it atomically."""
    qualification = _requested_qualification_index(qualification_index)
    document = validate_matrix_document(_strict_json_load(Path(matrix_path)))
    include = document["include"]
    matches = [item for item in include if isinstance(item, dict) and item.get("id") == shard_id]
    if len(matches) != 1:
        raise MatrixError(f"expected exactly one shard {shard_id!r}, found {len(matches)}")
    source = matches[0]
    requested = {
        item["case"]["case_id"]: item
        for item in document["requested_cases"]
    }
    cases = qualification_execution_order(
        source["id"],
        [requested[case_id]["case"] for case_id in source["case_ids"]],
        qualification,
    )
    control = {
        "schema_version": 1,
        "id": source.get("id"),
        "sku": source.get("sku"),
        "backend": source.get("backend"),
        "nodes": source.get("nodes"),
        "n": source.get("n"),
        "execution_weight": source.get("execution_weight"),
        "qualification_index": qualification,
        "execution_plan_sha256": execution_plan_sha256(cases),
        "cases": cases,
    }
    validate_shard_control(
        control,
        sku=sku,
        backend=backend,
        nodes=nodes,
        qualification_index=qualification,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w") as fh:
            json.dump(control, fh, sort_keys=True, separators=(",", ":"))
            fh.write("\n")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return control


def emit_unsupported(
    matrix_path: str | os.PathLike[str], output_dir: str | os.PathLike[str]
) -> list[Path]:
    """Materialize one strict terminal outcome for each unsupported requested case."""
    source = Path(matrix_path)
    document = validate_matrix_document(_strict_json_load(source))
    control_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        qualification_index = int(os.environ.get("CX_QUALIFICATION_INDEX", "1"))
    except ValueError as exc:
        raise MatrixError("CX_QUALIFICATION_INDEX must be an integer in 1..3") from exc
    if qualification_index not in range(1, 4):
        raise MatrixError("CX_QUALIFICATION_INDEX must be in 1..3")
    git_run = {
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "qualification_index": qualification_index,
        "ref": os.environ.get("GITHUB_REF_NAME") or os.environ.get("GITHUB_REF"),
        "source_sha": os.environ.get("COLLECTIVEX_SOURCE_SHA") or os.environ.get("GITHUB_SHA"),
        "repo": os.environ.get("GITHUB_REPOSITORY"),
        "job": os.environ.get("GITHUB_JOB"),
        "artifact": os.environ.get("COLLECTIVEX_ARTIFACT_NAME"),
    }
    allocation_factors = {
        "artifact": git_run["artifact"],
        "execution_id": os.environ.get("COLLECTIVEX_EXECUTION_ID"),
        "job": git_run["job"],
        "qualification_index": qualification_index,
        "repo": git_run["repo"],
        "run_attempt": git_run["run_attempt"],
        "run_id": git_run["run_id"],
        "runner": "capability-resolver",
        "source_sha": git_run["source_sha"],
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for wrapper in document["requested_cases"]:
        if wrapper["disposition"] != "unsupported":
            continue
        scheduled = wrapper["case"]
        case = {key: value for key, value in scheduled.items() if key != "case_id"}
        case_factors = {
            "case": case,
            "profile": identity.profile_for_case(case),
            "sku": wrapper["sku"],
        }
        case_id = identity.digest("case", case_factors)
        if case_id != scheduled["case_id"]:
            raise MatrixError(f"unsupported case identity differs for {scheduled['case_id']}")
        attempt_ordinal = 1
        record = contracts.make_terminal_document(
            allocation_factors=allocation_factors,
            attempt_ordinal=attempt_ordinal,
            case=case,
            case_factors=case_factors,
            control_sha256=control_sha256,
            failure_mode="capability",
            generated_at=generated_at,
            git_run=git_run,
            reason=wrapper["reason"],
            return_code=5,
            source="matrix-capability-resolver",
            status="unsupported",
            expected_case_id=case_id,
        )
        path = destination / f"unsupported_{case_id}.json"
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        try:
            with temporary.open("x") as handle:
                json.dump(record, handle, allow_nan=False, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        written.append(path)
    return written


def frontend_catalog(matrix: dict[str, Any]) -> dict[str, Any]:
    """Project the validated requested graph into a compact frontend test fixture."""
    document = validate_matrix_document(matrix)
    matrix_bytes = contracts.canonical_json_bytes(document) + b"\n"
    cases = []
    precision_profiles: dict[str, dict[str, Any]] = {}
    for wrapper in document["requested_cases"]:
        case = wrapper["case"]
        precision_profile = case.get(
            "precision_profile", identity.V1_CONTROL_PRECISION_PROFILE
        )
        precision = identity.precision_profile(precision_profile)
        precision_profiles[precision_profile] = {
            "dispatch": precision["dispatch"],
            "combine": precision["combine"],
        }
        cases.append({
            "backend": case["backend"],
            "backend_generation": None,
            "case_id": case["case_id"],
            "disposition": wrapper["disposition"],
            "eplb": case["eplb"],
            "label": (
                f"{wrapper['sku'].upper()} / {case['backend']} / EP{case['ep']} / "
                f"{case['mode']} / {case['phase']} / {case['routing']}"
            ),
            "mode": case["mode"],
            "phase": case["phase"],
            "precision_profile": precision_profile,
            "publication_tier": case["required_publication"],
            "reason": wrapper["reason"],
            "required": True,
            "resource": {
                "mode": "fixed-profile",
                "profile": None,
                "comm_units_kind": None,
                "configured_units": None,
            },
            "routing": case["routing"],
            "sku": wrapper["sku"],
            "suite": case["suite"],
            "topology": {
                "ep_size": case["ep"],
                **{field: case[field] for field in TOPOLOGY_FIELDS},
            },
            "workload": case["workload"],
            "points": [
                {
                    "global_tokens": int(token) * case["ep"],
                    "tokens_per_rank": int(token),
                }
                for token in case["ladder"].split()
            ],
        })
    return {
        "case_count": len(cases),
        "format": "collectivex.frontend-catalog.v1",
        "matrix_sha256": hashlib.sha256(matrix_bytes).hexdigest(),
        "point_count": sum(len(case["points"]) for case in cases),
        "precision_profiles": dict(sorted(precision_profiles.items())),
        "schema_version": 1,
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="CollectiveX v1 matrix resolver")
    parser.add_argument("--suites", default="all", help="'all' or comma-list of suites")
    parser.add_argument("--backend", default="", help="select one EP backend")
    parser.add_argument("--backends", default="", help="'all' or comma-list of EP backends")
    parser.add_argument("--only-sku", default="")
    parser.add_argument("--min-nodes", type=int, default=0)
    parser.add_argument("--max-nodes", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=128)
    parser.add_argument("--extract-from", default="", metavar="MATRIX")
    parser.add_argument("--validate-control", default="", metavar="SHARD")
    parser.add_argument("--emit-unsupported-from", default="", metavar="MATRIX")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--frontend-catalog", action="store_true")
    parser.add_argument("--shard-id", default="")
    parser.add_argument("--expect-sku", default="")
    parser.add_argument("--expect-backend", default="")
    parser.add_argument("--expect-nodes", type=int, default=0)
    parser.add_argument("--qualification-index", type=int, default=None)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    if args.emit_unsupported_from:
        if not args.out_dir:
            parser.error("unsupported outcome emission requires --out-dir")
        try:
            written = emit_unsupported(args.emit_unsupported_from, args.out_dir)
        except MatrixError as exc:
            parser.error(str(exc))
        print(f"emitted {len(written)} unsupported terminal outcomes", file=sys.stderr)
        return 0

    if args.validate_control:
        if not all((args.expect_sku, args.expect_backend, args.expect_nodes)):
            parser.error(
                "control validation requires --expect-sku, --expect-backend, and --expect-nodes"
            )
        try:
            control = _strict_json_load(Path(args.validate_control))
            qualification = _requested_qualification_index(
                args.qualification_index
            )
            validate_shard_control(
                control,
                sku=args.expect_sku,
                backend=args.expect_backend,
                nodes=args.expect_nodes,
                qualification_index=qualification,
            )
        except MatrixError as exc:
            parser.error(str(exc))
        print(f"validated {control.get('id')}: {control['n']} cases", file=sys.stderr)
        return 0

    if args.extract_from:
        if not all((args.shard_id, args.expect_sku, args.expect_backend, args.expect_nodes, args.out)):
            parser.error(
                "shard extraction requires --shard-id, --expect-sku, --expect-backend, "
                "--expect-nodes, and --out"
            )
        try:
            control = extract_shard(
                args.extract_from,
                args.shard_id,
                args.out,
                sku=args.expect_sku,
                backend=args.expect_backend,
                nodes=args.expect_nodes,
                qualification_index=args.qualification_index,
            )
        except MatrixError as exc:
            parser.error(str(exc))
        print(f"extracted {control['id']}: {control['n']} cases", file=sys.stderr)
        print(json.dumps(control, separators=(",", ":")))
        return 0

    matrix = resolve_matrix(
        suites=args.suites,
        backend=args.backend,
        backends=args.backends,
        only_sku=args.only_sku,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        max_cases=args.max_cases,
    )
    try:
        validate_matrix_document(matrix)
    except MatrixError as exc:
        parser.error(str(exc))
    output_document = frontend_catalog(matrix) if args.frontend_catalog else matrix
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(output_document, fh, sort_keys=True, separators=(",", ":"))
            fh.write("\n")
    runnable = sum(
        item["disposition"] == "runnable" for item in matrix["requested_cases"]
    )
    unsupported = len(matrix["requested_cases"]) - runnable
    print(
        f"resolved {len(matrix['include'])} shard-cells, "
        f"{runnable} runnable and {unsupported} unsupported cases",
        file=sys.stderr,
    )
    print(json.dumps(output_document))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
