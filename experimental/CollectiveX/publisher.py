#!/usr/bin/env python3
"""Fail-closed filesystem publisher for CollectiveX EP v1 artifacts."""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import statistics
import sys
import tempfile
from typing import Any, Iterator, Sequence
import zipfile

import jsonschema
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import artifact_safety  # noqa: E402
import capability  # noqa: E402
import contracts  # noqa: E402
import identity  # noqa: E402
import sweep_matrix  # noqa: E402

FORMAT_BUNDLE = "collectivex.private.bundle.v1"
FORMAT_PUBLIC = "collectivex.public.v1"
FORMAT_CHANNEL = "collectivex.channel.v1"
POLICY = "collectivex-decision-grade-v1"
PUBLISHER_POLICY = "collectivex-publisher-v1"
OUTCOMES = ("success", "unsupported", "failed", "invalid", "diagnostic")
REQUIRED_ALLOCATIONS = 3
REQUIRED_COHORT_KINDS = ("library", "chip", "system")
PRECISION_COHORT_KINDS = (
    "dispatch-precision", "combine-precision", "precision-pair",
)
REQUIRED_PROMOTION_COHORT_COUNTS = {"library": 24, "system": 4}
CANONICAL_FULL_V1_MATRIX_SHA256 = (
    "ba09fb548bae7d4fcc1994b1461d15a9feeb241930366f027ca4ac5582f8e5eb"
)
CANONICAL_FULL_V1_CASE_CATALOG_SHA256 = (
    "7bda7073a07c05952c0cabdea88161bd39c07552e6add49dd39c9d19e0d8d850"
)
P50_STABILITY_LIMIT = 1.10
P99_STABILITY_LIMIT = 1.25
TRIAL_DRIFT_RATIO_LIMIT = 1.10
TRIAL_OUTLIER_FRACTION_LIMIT = 0.05
TRIAL_OUTLIER_MAD_MULTIPLIER = 6.0
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_EQUIVALENCE_BAND = 0.05
BOOTSTRAP_POLICY = "hierarchical-run-trial-p99-ratio-v1"
BOOTSTRAP_CHUNK_SIZE = 250
MAX_ARCHIVE_MEMBERS = 20_000
MAX_ARCHIVE_MEMBER_BYTES = 2 * 1024**3
MAX_ARCHIVE_TOTAL_BYTES = 16 * 1024**3
MAX_PUBLIC_DATASET_BYTES = 32 * 1024**2
HEX64 = re.compile(r"[0-9a-f]{64}")
SAFE_ID = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}")
REASON = re.compile(r"[a-z0-9][a-z0-9.-]{0,95}")
ARTIFACT_NAME = re.compile(
    r"cx(?:unsupported|shard-[a-z0-9][a-z0-9_.-]{0,127})-[1-9][0-9]*-[1-9][0-9]*"
)
COVERAGE_TOPOLOGY_FIELDS = (
    "ep_size", "nodes", "gpus_per_node", "scale_up_domain", "scope",
    "scale_up_transport", "scale_out_transport", "transport", "topology_class",
)
CHANNEL_PATH = re.compile(r"datasets/([0-9a-f]{64})/dataset\.json")
SCHEMA_DIR = HERE / "schemas"
_SCHEMAS: dict[str, jsonschema.protocols.Validator] = {}
_BOOTSTRAP_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}


class PublisherError(ValueError):
    """Input or stored state violates the publication contract."""


strict_load = contracts.strict_load
_canonical = contracts.canonical_json_bytes


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latest_timestamp(values: Sequence[str]) -> str:
    """Return the latest evidence timestamp without introducing publisher wall time."""
    if not values:
        raise PublisherError("cannot derive a timestamp without evidence")

    def parsed(value: str) -> dt.datetime:
        try:
            timestamp = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise PublisherError("evidence timestamp is not ISO-8601") from exc
        if timestamp.tzinfo is None:
            raise PublisherError("evidence timestamp must include a timezone")
        return timestamp.astimezone(dt.timezone.utc)

    return max(values, key=lambda value: (parsed(value), value))


def _schema(name: str, value: Any) -> None:
    validator = _SCHEMAS.get(name)
    if validator is None:
        schema = strict_load(SCHEMA_DIR / name)
        jsonschema.Draft202012Validator.check_schema(schema)
        validator = jsonschema.Draft202012Validator(
            schema, format_checker=jsonschema.FormatChecker()
        )
        _SCHEMAS[name] = validator
    errors = sorted(validator.iter_errors(value), key=lambda error: list(error.absolute_path))
    if errors:
        error = errors[0]
        location = ".".join(map(str, error.absolute_path)) or "$"
        raise PublisherError(f"{name}:{location}: {error.message}")
def _exact(obj: Any, fields: set[str], path: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise PublisherError(f"{path} must be an object")
    actual = set(obj)
    if actual != fields:
        raise PublisherError(
            f"{path} fields differ: missing={sorted(fields - actual)}, "
            f"extra={sorted(actual - fields)}"
        )
    return obj
def _array(value: Any, path: str, *, nonempty: bool = False) -> list[Any]:
    if not isinstance(value, list) or (nonempty and not value):
        qualifier = "a nonempty" if nonempty else "an"
        raise PublisherError(f"{path} must be {qualifier} array")
    return value


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PublisherError(f"{path} must be an integer >= {minimum}")
    return value


def _unique(values: Sequence[Any], path: str) -> None:
    serialized = [_canonical(value) for value in values]
    if len(serialized) != len(set(serialized)):
        raise PublisherError(f"{path} contains duplicates")

def _eligibility(value: dict[str, Any], path: str) -> dict[str, Any]:
    allocations = value["allocation_ids"]
    p50 = value["p50_max_min_ratio"]
    p99 = value["p99_max_min_ratio"]
    gates = (
        len(allocations) >= REQUIRED_ALLOCATIONS,
        value["complete"], value["correct"], value["measured_roundtrip_p99"],
        value["stable_p50"], value["stable_p99"], value["stable_ordering"],
        p50 is not None and p50 <= P50_STABILITY_LIMIT,
        p99 is not None and p99 <= P99_STABILITY_LIMIT,
    )
    if value["decision_grade"] != (all(gates) and not value["reasons"]):
        raise PublisherError(f"{path}.decision_grade does not match promotion gates")
    if value["decision_grade"] == bool(value["reasons"]):
        raise PublisherError(f"{path}.reasons does not match decision status")
    return value


def validate_channel(doc: Any, *, expected_channel: str | None = None) -> dict[str, Any]:
    _schema("channel-v1.schema.json", doc)
    if expected_channel and doc["channel"] != expected_channel:
        raise PublisherError("channel name does not match its file")
    target = doc["dataset"]
    match = CHANNEL_PATH.fullmatch(target["path"]) if isinstance(target["path"], str) else None
    if not match or match.group(1) != target["sha256"]:
        raise PublisherError("channel dataset path and sha256 do not agree")
    return doc


def _metric_value(series: dict[str, Any], metric: dict[str, Any]) -> tuple[str, float, str]:
    point = next(
        (point for point in series["points"] if point["tokens_per_rank"] == metric["tokens_per_rank"]),
        None,
    )
    if point is None or series["phase"] != metric["phase"]:
        raise PublisherError("decision metric references an unavailable point")
    component = point["components"]["roundtrip"]
    if metric["measure"] == "latency_us":
        value = component["latency_us"][metric["statistic"]]
        unit = "us"
    else:
        rates = component[metric["measure"]]
        if rates is None:
            raise PublisherError("data-rate decision has no byte accounting contract")
        value = rates[metric["statistic"]]
        unit = "GB/s"
    return point["point_id"], value, unit


def _validate_metric(metric: dict[str, Any]) -> None:
    expected = "min" if metric["measure"] == "latency_us" else "max"
    if metric["objective"] != expected:
        raise PublisherError(f"{metric['measure']} objective must be {expected}")


def _metric_label(measure: str, statistic: str) -> str:
    if measure == "latency_us":
        return f"{statistic} latency"
    label = (
        "activation data rate"
        if measure == "activation_data_rate_gbps_at_latency_percentile"
        else "total logical data rate"
    )
    return f"{label} at {statistic} latency"


def _public_case_factors(series: dict[str, Any]) -> dict[str, Any]:
    workload = series["workload"]
    system = series["system"]
    measurement = series["measurement"]
    ep_size = system["ep_size"]
    case = {
        "backend": series["backend"]["id"],
        "canonical": True,
        "eplb": workload["eplb"],
        "ep": ep_size,
        "experts": workload["experts"],
        "gpus_per_node": system["gpus_per_node"],
        "hidden": workload["hidden"],
        "ladder": " ".join(str(point["tokens_per_rank"]) for point in series["points"]),
        "mode": series["mode"],
        "nodes": system["nodes"],
        "phase": series["phase"],
        "required_publication": series["publication_tier"],
        "routing": workload["routing"],
        "samples_per_point": measurement["samples_per_component"],
        "scale_out_transport": system["scale_out_transport"],
        "scale_up_domain": system["scale_up_domain"],
        "scale_up_transport": system["scale_up_transport"],
        "scope": system["scope"],
        "suite": series["suite"],
        "timing": (
            f"{measurement['iters']}:{measurement['trials']}:"
            f"{measurement['warmups']}"
        ),
        "topk": workload["top_k"],
        "topology_class": system["topology_class"],
        "transport": system["transport"],
        "warmup_semantics": sweep_matrix.ep_harness.WARMUP_SEMANTICS,
        "workload": series["model"],
    }
    if workload["precision_profile"] != identity.V1_CONTROL_PRECISION_PROFILE:
        case["precision_profile"] = workload["precision_profile"]
    return {
        "case": case,
        "profile": identity.profile_for_case(case),
        "sku": system["sku"],
    }


def _coverage_topology(case: dict[str, Any]) -> dict[str, Any]:
    """Project exact fabric placement without exposing private runner details."""
    return {
        "ep_size": case.get("ep_size", case.get("ep")),
        **{field: case[field] for field in COVERAGE_TOPOLOGY_FIELDS if field != "ep_size"},
    }


def _coverage_coordinates(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "sku": case["sku"], "backend": case["backend"],
        "mode": case["mode"], "phase": case["phase"],
        "topology": _coverage_topology(case),
    }


@lru_cache(maxsize=1)
def _canonical_coverage_cases() -> dict[str, dict[str, Any]]:
    matrix = sweep_matrix.resolve_matrix(suites="all", max_cases=128, backends="all")
    return {
        item["case"]["case_id"]: {
            "sku": item["sku"],
            **item["case"],
            "disposition": item["disposition"],
            "reason": item["reason"],
        }
        for item in matrix["requested_cases"]
    }


def _public_series_config(series: dict[str, Any]) -> dict[str, Any]:
    return {
        "backend": {
            "generation": series["backend"]["generation"],
            "version": series["backend"]["version"],
        },
        "resource": series["resource"],
        "system": {"label": series["system"]["label"]},
    }


def _public_cohort_factors(kind: str, item: dict[str, Any]) -> tuple[Any, Any]:
    workload = item["workload"]
    build = item["build"]
    shape = {
        key: workload[key]
        for key in (
            "hidden", "top_k", "experts", "precision_profile", "dispatch_precision",
            "combine_precision", "activation_profile",
        )
    }
    common = {
        "model": item["model"], "mode": item["mode"], "phase": item["phase"],
        "shape": shape, "measurement": item["measurement"],
        "ep_size": item["system"]["ep_size"],
    }
    if kind == "library":
        return (
            {**common, "system": item["system"], "workload": workload,
             "resource_mode": item["resource"]["mode"], "source": build["source_sha"]},
            item["backend"]["id"],
        )
    if kind == "chip":
        return (
            {**common, "backend": item["backend"], "workload": workload,
             "resource_mode": item["resource"]["mode"], "source": build["source_sha"]},
            item["system"],
        )
    if kind == "system":
        return {**common, "workload": workload, "source": build["source_sha"]}, [
            item["system"]["sku"], item["backend"]["id"], item["resource"]["profile"]
        ]
    if kind in PRECISION_COHORT_KINDS:
        static_shape = {
            key: workload[key]
            for key in ("hidden", "top_k", "experts", "activation_profile")
        }
        control = {
            "backend": item["backend"],
            "build": {
                key: build[key]
                for key in (
                    "image_digest", "runtime_fingerprint_sha256", "source_sha",
                    "squash_sha256",
                )
            },
            "measurement": item["measurement"],
            "mode": item["mode"],
            "model": item["model"],
            "phase": item["phase"],
            "resource": item["resource"],
            "shape": static_shape,
            "system": item["system"],
            "workload": {
                "eplb": workload["eplb"],
                "routing": workload["routing"],
            },
        }
        if kind == "dispatch-precision":
            control["combine_precision"] = workload["combine_precision"]
            variant = workload["dispatch_precision"]
        elif kind == "combine-precision":
            control["dispatch_precision"] = workload["dispatch_precision"]
            variant = workload["combine_precision"]
        else:
            control.pop("resource")
            variant = {
                "combine_precision": workload["combine_precision"],
                "dispatch_precision": workload["dispatch_precision"],
                "precision_profile": workload["precision_profile"],
                "resource": item["resource"],
            }
        return control, variant
    raise PublisherError(f"unknown cohort kind {kind}")


def _case_disposition_catalog_sha256(coverage: Sequence[dict[str, Any]]) -> str:
    catalog = [
        {"case_id": item["case_id"], "disposition": item["disposition"]}
        for item in sorted(coverage, key=lambda item: item["case_id"])
    ]
    return _sha_bytes(_canonical(catalog))


def validate_public_dataset(doc: Any) -> dict[str, Any]:
    _schema("public-dataset-v1.schema.json", doc)
    if len(_canonical(doc)) + 1 > MAX_PUBLIC_DATASET_BYTES:
        raise PublisherError("public dataset exceeds the serving size limit")
    try:
        artifact_safety.assert_publication_safe([doc])
    except artifact_safety.ArtifactSafetyError as exc:
        raise PublisherError(str(exc)) from exc
    if doc["source_bundle_ids"] != sorted(doc["source_bundle_ids"]):
        raise PublisherError("source bundle IDs are not canonical")
    for field, key in (
        ("coverage", "case_id"), ("attempts", "attempt_id"),
        ("series", "series_id"), ("cohorts", "cohort_id"),
        ("rankings", "ranking_id"), ("recommendations", "recommendation_id"),
        ("sensitivities", "sensitivity_id"),
    ):
        if doc[field] != sorted(doc[field], key=lambda item: item[key]):
            raise PublisherError(f"{field} are not in canonical identity order")
    promotion = doc["promotion"]
    quarantined = promotion["status"] == "quarantined"
    if quarantined != (promotion["reason"] is not None) or quarantined != (
        promotion["matrix_id"] is None
    ):
        raise PublisherError("promotion reason/matrix identity differs from status")
    attempts = {item["attempt_id"]: item for item in doc["attempts"]}
    if len(attempts) != len(doc["attempts"]):
        raise PublisherError("dataset has duplicate attempt IDs")
    evidence = [
        value["evidence_id"] for item in doc["attempts"] for value in item["evidence"]
    ]
    _unique(evidence, "dataset attempt evidence")
    series = {item["series_id"]: item for item in doc["series"]}
    if len(series) != len(doc["series"]):
        raise PublisherError("dataset has duplicate series IDs")
    allocation_ids = set(promotion["allocation_ids"])
    case_ids = {item["case_id"] for item in doc["coverage"]}
    if len(case_ids) != len(doc["coverage"]):
        raise PublisherError("dataset has duplicate case coverage")
    coverage_by_case = {item["case_id"]: item for item in doc["coverage"]}
    series_case_ids = {
        case_id for item in doc["series"] for case_id in item["case_ids"]
    }
    canonical_cases = _canonical_coverage_cases()
    for item in doc["coverage"]:
        topology = item["topology"]
        registered = capability.topology_for(item["sku"], topology["ep_size"])
        if (
            item["sku"] not in capability.PLATFORMS
            or item["backend"] not in capability.BACKENDS
            or registered is None
            or any(
                topology[field] != registered[field]
                for field in COVERAGE_TOPOLOGY_FIELDS if field != "ep_size"
            )
        ):
            raise PublisherError("coverage topology differs from the capability registry")
        canonical = canonical_cases.get(item["case_id"])
        if canonical is not None:
            precision_profile = canonical.get(
                "precision_profile", identity.V1_CONTROL_PRECISION_PROFILE
            )
            precision = identity.precision_profile(precision_profile)
            expected_projection = {
                "sku": canonical["sku"],
                "suite": canonical["suite"],
                "workload": canonical["workload"],
                "publication_tier": canonical["required_publication"],
                "backend": canonical["backend"],
                "mode": canonical["mode"],
                "phase": canonical["phase"],
                "routing": canonical["routing"],
                "eplb": canonical["eplb"],
                "precision_profile": precision_profile,
                "dispatch_precision": precision["dispatch"],
                "combine_precision": precision["combine"],
                "topology": _coverage_topology(canonical),
                "disposition": canonical["disposition"],
            }
            if any(item[field] != value for field, value in expected_projection.items()):
                raise PublisherError("coverage dimensions differ from its case identity")
            expected_tokens = [int(value) for value in canonical["ladder"].split()]
            if [point["tokens_per_rank"] for point in item["points"]] != expected_tokens:
                raise PublisherError("coverage points differ from the requested token ladder")
        if canonical is None and item["case_id"] not in series_case_ids:
            raise PublisherError("coverage case identity is outside the v1 catalog")
        for point in item["points"]:
            if point["global_tokens"] != point["tokens_per_rank"] * topology["ep_size"]:
                raise PublisherError("coverage point global token count differs")
            if (point["terminal_status"] == "measured") != (point["reason"] is None):
                raise PublisherError("coverage point terminal reason differs from status")
    for item in doc["attempts"]:
        if item["case_id"] not in case_ids or item["allocation_id"] not in allocation_ids:
            raise PublisherError("attempt references undeclared coverage or allocation")
        if item["series_id"] is not None and item["series_id"] not in series:
            raise PublisherError("attempt references unknown series")
        if (item["outcome"] == "success") != (item["reason"] is None):
            raise PublisherError("attempt reason must be null exactly for success")
        if item["outcome"] == "success" and item["failure_mode"] is not None:
            raise PublisherError("successful attempt cannot have a failure mode")
        if (item["outcome"] == "success" and item["selected"]) != (
            item["series_id"] is not None
        ):
            raise PublisherError("attempt series must be present exactly for selected success")
    if {item["allocation_id"] for item in doc["attempts"]} != allocation_ids:
        raise PublisherError("promotion allocation catalog differs from attempts")
    attempt_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in doc["attempts"]:
        attempt_groups.setdefault((item["case_id"], item["allocation_id"]), []).append(item)
    for (case_id, allocation_id), group in attempt_groups.items():
        ordinals = sorted(item["attempt_index"] for item in group)
        if ordinals != list(range(1, len(group) + 1)):
            raise PublisherError("public retries must retain contiguous attempt indexes")
        if any(
            item["attempt_id"] != identity.attempt_id(
                allocation=allocation_id, case=case_id, ordinal=item["attempt_index"]
            )
            for item in group
        ):
            raise PublisherError("public retry identity differs from its case/allocation/index")
        selected = [item for item in group if item["selected"]]
        if len(selected) != 1 or selected[0]["attempt_index"] != ordinals[-1]:
            raise PublisherError("publisher must select the latest retry per case/allocation")
    selected_by_series: dict[str, list[dict[str, Any]]] = {}
    for item in doc["attempts"]:
        if item["selected"] and item["outcome"] == "success":
            selected_by_series.setdefault(item["series_id"], []).append(item)
    terminal = 0
    for item in doc["coverage"]:
        listed = set(item["attempt_ids"])
        selected = item["selected_attempt_id"]
        expected_attempts = {
            attempt_id for attempt_id, attempt in attempts.items()
            if attempt["case_id"] == item["case_id"]
        }
        if listed != expected_attempts:
            raise PublisherError("coverage references attempts from another case")
        if selected is not None:
            terminal += 1
            if (selected not in listed or not attempts[selected]["selected"]
                    or any(attempts[selected][field] != item[field]
                           for field in ("outcome", "failure_mode", "reason"))):
                raise PublisherError("coverage selected outcome differs")
            selected_candidates = [attempts[value] for value in listed if attempts[value]["selected"]]
            latest = max(
                selected_candidates,
                key=lambda attempt: (
                    int(attempt["run_id"]), attempt["run_attempt"],
                    attempt["attempt_index"], attempt["attempt_id"]
                ),
            )
            if selected != latest["attempt_id"]:
                raise PublisherError("coverage does not select the latest canonical allocation")
            expected_status = (
                "measured" if attempts[selected]["outcome"] == "success"
                else attempts[selected]["outcome"]
            )
            if any(point["terminal_status"] != expected_status for point in item["points"]):
                raise PublisherError("coverage point status differs from selected attempt")
            if expected_status == "measured":
                selected_series = series.get(attempts[selected]["series_id"])
                if selected_series is None:
                    raise PublisherError("measured coverage points lack a public series")
                public_points = {
                    point["tokens_per_rank"]: point for point in selected_series["points"]
                }
                if any(
                    point["series_id"] != selected_series["series_id"]
                    or point["point_id"]
                    != public_points.get(point["tokens_per_rank"], {}).get("point_id")
                    for point in item["points"]
                ):
                    raise PublisherError("coverage point identities differ from series")
    measured_cases = sum(
        all(point["terminal_status"] == "measured" for point in item["points"])
        for item in doc["coverage"]
    )
    unsupported_cases = sum(
        all(point["terminal_status"] == "unsupported" for point in item["points"])
        for item in doc["coverage"]
    )
    requested_points = sum(len(item["points"]) for item in doc["coverage"])
    measured_points = sum(
        point["terminal_status"] == "measured"
        for item in doc["coverage"] for point in item["points"]
    )
    unsupported_points = sum(
        point["terminal_status"] == "unsupported"
        for item in doc["coverage"] for point in item["points"]
    )
    expected_counts = {
        "requested_cases": len(doc["coverage"]),
        "terminal_cases": terminal,
        "measured_cases": measured_cases,
        "unsupported_cases": unsupported_cases,
        "requested_points": requested_points,
        "terminal_points": requested_points,
        "measured_points": measured_points,
        "unsupported_points": unsupported_points,
    }
    if any(promotion[field] != value for field, value in expected_counts.items()):
        raise PublisherError("promotion coverage counts differ")
    selected_evidence: dict[tuple[str, str], set[str]] = {}
    for attempt in doc["attempts"]:
        if attempt["selected"] and attempt["series_id"] is not None:
            for value in attempt["evidence"]:
                selected_evidence.setdefault(
                    (attempt["series_id"], value["point_id"]), set()
                ).add(value["evidence_id"])
    for item in doc["series"]:
        eligibility = _eligibility(item["eligibility"], f"series {item['series_id']}")
        workload = item["workload"]
        model, hidden, top_k, experts = sweep_matrix.V1_WORKLOAD
        suite_contract = sweep_matrix.V1_SUITE_CONTRACTS.get(item["suite"])
        coordinate = (
            item["mode"], item["phase"], workload["routing"], workload["eplb"]
        )
        profile_case = {"mode": item["mode"]}
        if workload["precision_profile"] != identity.V1_CONTROL_PRECISION_PROFILE:
            profile_case["precision_profile"] = workload["precision_profile"]
        profile = identity.profile_for_case(profile_case)
        communication_precision = identity.precision_profile(workload["precision_profile"])
        if (
            item["model"] != model
            or (workload["hidden"], workload["top_k"], workload["experts"])
            != (hidden, top_k, experts)
            or suite_contract is None
            or coordinate not in suite_contract["coordinates"]
            or (
                suite_contract.get("backends") is not None
                and item["backend"]["id"] not in suite_contract["backends"]
            )
            or item["publication_tier"] != suite_contract["publication"]
            or item["measurement"]["contract"] != profile["contract"]
            or item["measurement"]["component_order_contract"]
            != profile["component_order_contract"]
            or item["measurement"]["combine_semantics"] != profile["combine_semantics"]
            or item["measurement"]["payload_unit"] != profile["payload_unit"]
            or workload["dispatch_precision"] != communication_precision["dispatch"]
            or workload["combine_precision"] != communication_precision["combine"]
            or item["measurement"]["qualification_indices"]
            != sorted(item["measurement"]["qualification_indices"])
            or len(set(item["measurement"]["qualification_indices"]))
            != len(item["measurement"]["qualification_indices"])
        ):
            raise PublisherError("series differs from the frozen v1 workload/suite profile")
        backend_id = item["backend"]["id"]
        expected_role = "reference" if backend_id == "nccl-ep" else "library"
        if (
            backend_id not in capability.BACKENDS
            or item["backend"]["label"] != BACKEND_LABELS[backend_id]
            or item["backend"]["role"] != expected_role
            or item["backend"]["version"] is None
        ):
            raise PublisherError("series backend projection differs from v1")
        sku = item["system"]["sku"]
        platform = capability.PLATFORMS.get(sku)
        ep_size = item["system"]["ep_size"]
        registered_topology = capability.topology_for(sku, ep_size)
        if platform is None or registered_topology is None:
            raise PublisherError("series system projection differs from v1")
        disposition, _ = capability.resolve_disposition(
            sku, backend_id, ep=ep_size, nodes=item["system"]["nodes"],
            routing=workload["routing"], eplb=workload["eplb"],
            mode=item["mode"],
            precision_profile=(
                workload["precision_profile"]
                if workload["precision_profile"] != identity.V1_CONTROL_PRECISION_PROFILE
                else None
            ),
        )
        if (
            disposition != "supported"
            or item["system"]["vendor"] != platform["vendor"]
            or any(
                item["system"][field] != registered_topology[field]
                for field in (
                    "nodes", "gpus_per_node", "scale_up_domain", "scope",
                    "scale_up_transport", "scale_out_transport", "transport",
                    "topology_class",
                )
            )
            or item["system"]["world_size"] != ep_size
            or platform["product"] not in set(
                re.findall(r"[a-z]+\d+[a-z]*", item["system"]["label"].lower())
            )
        ):
            raise PublisherError("series system projection differs from v1")
        if contracts.public_series_config_sha256(_public_series_config(item)) != item[
            "build"
        ]["public_config_sha256"]:
            raise PublisherError("public series configuration differs from its commitment")
        covered = [coverage_by_case.get(case_id) for case_id in item["case_ids"]]
        if not covered or any(
            case is None
            or {
                "sku": case["sku"], "backend": case["backend"],
                "mode": case["mode"], "phase": case["phase"],
                "topology": case["topology"],
            }
            != {
                "sku": sku, "backend": backend_id,
                "mode": item["mode"], "phase": item["phase"],
                "topology": _coverage_topology(item["system"]),
            }
            for case in covered
        ):
            raise PublisherError("series projection differs from its case coverage")
        if (
            item["eplb"]["enabled"] != item["workload"]["eplb"]
            or item["eplb"]["logical_experts"] != item["workload"]["experts"]
        ):
            raise PublisherError("series EPLB descriptor differs from its workload")
        eplb = item["eplb"]
        expected_physical = eplb["logical_experts"] + eplb["redundant_experts"]
        nullable_eplb = (
            "planner", "mapping_sha256", "reference_tokens_per_rank", "max_replicas",
            "imbalance_before", "imbalance_after", "calibration_workload_id",
            "calibration_trace_sha256", "calibration_window", "calibration_token_offset",
        )
        if eplb["enabled"]:
            if (
                item["workload"]["routing"] != "zipf"
                or any(eplb[field] is None for field in nullable_eplb)
                or eplb["planner"] != "greedy-rank-major-v1"
                or eplb["reference_tokens_per_rank"] != 2048
                or eplb["redundant_experts"] != 32
                or eplb["redundant_experts"] % ep_size != 0
                or eplb["physical_experts"] != expected_physical
                or eplb["logical_experts"] % ep_size != 0
                or eplb["physical_experts"] % ep_size != 0
                or not 1 <= eplb["replicated_experts"] <= min(
                    eplb["logical_experts"], eplb["redundant_experts"]
                )
                or not 2 <= eplb["max_replicas"] <= 1 + eplb["redundant_experts"]
                or not 1 <= eplb["imbalance_after"] <= eplb["imbalance_before"] <= ep_size
            ):
                raise PublisherError("enabled EPLB descriptor is incomplete")
            expected_plan, calibration = contracts._expected_eplb_calibration(
                workload["routing"], workload["hidden"], workload["top_k"],
                eplb["logical_experts"], eplb["physical_experts"], ep_size,
                identity.V1_CASE_PROFILE["seed"],
                identity.V1_CASE_PROFILE["eplb_reference_tokens_per_rank"],
            )
            expected_eplb = {
                **calibration,
                "enabled": True,
                "planner": identity.V1_CASE_PROFILE["eplb_planner"],
                "mapping_sha256": contracts.eplb_contract.mapping_hash(expected_plan),
                "logical_experts": eplb["logical_experts"],
                "physical_experts": eplb["physical_experts"],
                "redundant_experts": identity.V1_CASE_PROFILE["eplb_redundant_experts"],
                "reference_tokens_per_rank": identity.V1_CASE_PROFILE[
                    "eplb_reference_tokens_per_rank"
                ],
                "replicated_experts": expected_plan["replicated_experts"],
                "max_replicas": expected_plan["max_replicas"],
                "imbalance_before": expected_plan["imbalance_before"],
                "imbalance_after": expected_plan["imbalance_after"],
            }
            if eplb != expected_eplb:
                raise PublisherError("enabled EPLB descriptor differs from deterministic plan")
        elif (
            any(eplb[field] is not None for field in nullable_eplb)
            or eplb["physical_experts"] != expected_physical
            or eplb["redundant_experts"] != 0
            or eplb["replicated_experts"] != 0
        ):
            raise PublisherError("disabled EPLB descriptor claims a plan")
        if item["backend"]["id"] == "nccl-ep":
            expected_generation = (
                "nccl" if item["system"]["vendor"] == "nvidia" else "rccl"
            )
            if item["backend"]["generation"] != expected_generation:
                raise PublisherError("NCCL/RCCL reference generation differs from system vendor")
        if (item["status"] == "decision-grade") != eligibility["decision_grade"]:
            raise PublisherError("series status differs from eligibility")
        if (
            set(eligibility["allocation_ids"]) != set(item["allocation_ids"])
            or eligibility["correct"] != all(
                point["correctness"]["semantic_pass"]
                and point["correctness"]["precision"]["passed"]
                for point in item["points"]
            )
        ):
            raise PublisherError("series eligibility differs from its evidence")
        selected_attempts = selected_by_series.get(item["series_id"], [])
        if (
            set(item["case_ids"]) != {attempt["case_id"] for attempt in selected_attempts}
            or set(item["allocation_ids"])
            != {attempt["allocation_id"] for attempt in selected_attempts}
            or item["measurement"]["qualification_indices"]
            != sorted({attempt["qualification_index"] for attempt in selected_attempts})
        ):
            raise PublisherError("series case/allocation catalog differs from selected attempts")
        if item["eligibility"]["decision_grade"] and len(
            {attempt["run_id"] for attempt in selected_attempts}
        ) < REQUIRED_ALLOCATIONS:
            raise PublisherError("decision-grade series lacks independent workflow runs")
        tokens = [point["tokens_per_rank"] for point in item["points"]]
        if tokens != sorted(set(tokens)):
            raise PublisherError("series points are not in unique ascending token order")
        if len(item["case_ids"]) != 1:
            raise PublisherError("public series must represent exactly one v1 case")
        case_id = item["case_ids"][0]
        if identity.digest("case", _public_case_factors(item)) != case_id:
            raise PublisherError("public series projection differs from its case identity")
        build = item["build"]
        expected_series_id = identity.series_id({
            "backend": backend_id,
            "case_id": case_id,
            "image_digest": build["image_digest"],
            "implementation_contract_sha256": build[
                "implementation_contract_sha256"
            ],
            "public_config_sha256": build["public_config_sha256"],
            "routing_control_sha256": build["routing_control_sha256"],
            "runtime_fingerprint_sha256": build["runtime_fingerprint_sha256"],
            "source_sha": build["source_sha"],
            "squash_sha256": build["squash_sha256"],
            "workload_id": workload["workload_id"],
        })
        if item["series_id"] != expected_series_id:
            raise PublisherError("public series identity differs from its committed factors")
        for point in item["points"]:
            if point["point_id"] != identity.point_id(series=item["series_id"], tokens_per_rank=point["tokens_per_rank"]):
                raise PublisherError("point identity differs")
            if point["global_tokens"] != point["tokens_per_rank"] * item["system"]["ep_size"]:
                raise PublisherError("global_tokens must use EP size")
            routing = point["routing"]
            max_fanout = min(item["workload"]["top_k"], item["system"]["ep_size"])
            if (
                routing["routed_copies"] < point["global_tokens"]
                or routing["routed_copies"] > point["global_tokens"] * max_fanout
                or routing["recv_tokens_max"] > routing["routed_copies"]
                or routing["recv_tokens_max"] * item["system"]["ep_size"]
                < routing["routed_copies"]
                or not math.isclose(
                    routing["fanout_mean"],
                    routing["routed_copies"] / point["global_tokens"],
                    rel_tol=1e-12,
                )
                or routing["hotspot_ratio"] < 1
                or routing["empty_expert_count"] >= eplb["physical_experts"]
                or routing["empty_rank_count"] >= item["system"]["ep_size"]
            ):
                raise PublisherError("point routing/load facts are internally inconsistent")
            expected_evidence = selected_evidence.get(
                (item["series_id"], point["point_id"]), set()
            )
            if set(point["evidence_ids"]) != expected_evidence:
                raise PublisherError("point evidence differs from selected series attempts")
            point_correctness = point["correctness"]
            if (
                point_correctness["precision"]["profile_id"]
                != workload["precision_profile"]
                or (
                    point_correctness["semantic_pass"]
                    and not point_correctness["precision"]["passed"]
                )
                or point["stability"]["qualification_indices"]
                != item["measurement"]["qualification_indices"]
            ):
                raise PublisherError("point correctness/stability differs from series evidence")
            diagnostics = point["trial_diagnostics"]
            diagnostic_reasons = set(diagnostics["reasons"])
            component_reasons: set[str] = set()
            for name, summary in diagnostics["components"].items():
                if summary is None:
                    if point["components"][name] is not None:
                        raise PublisherError("trial diagnostics omit a measured component")
                    continue
                if point["components"][name] is None:
                    raise PublisherError("trial diagnostics describe an unavailable component")
                if summary["drift_flagged"] != (
                    summary["first_last_median_ratio"] > TRIAL_DRIFT_RATIO_LIMIT
                ) or summary["outlier_flagged"] != (
                    summary["robust_outlier_fraction"] > TRIAL_OUTLIER_FRACTION_LIMIT
                ):
                    raise PublisherError("trial diagnostic flags differ from their thresholds")
                if summary["drift_flagged"]:
                    component_reasons.add("trial-drift")
                if summary["outlier_flagged"]:
                    component_reasons.add("trial-outliers")
            if (
                diagnostic_reasons != component_reasons
                or diagnostics["flagged"] != bool(diagnostic_reasons)
                or not diagnostic_reasons.issubset(point["anomalies"])
            ):
                raise PublisherError("trial diagnostic summary is inconsistent")
            components = point["components"]
            if (components["dispatch"] is None) != (components["combine"] is None):
                raise PublisherError("dispatch/combine availability differs")
            for name, component in components.items():
                if component is None:
                    continue
                expected_origin = "derived" if name == "isolated_sum" else "measured"
                expected_samples = None if name == "isolated_sum" else 512
                if component["origin"] != expected_origin or component["sample_count"] != expected_samples:
                    raise PublisherError(f"{name} origin or sample count differs")
                rate_fields = (
                    "activation_data_rate_gbps_at_latency_percentile",
                    "total_logical_data_rate_gbps_at_latency_percentile",
                )
                if name == "isolated_sum" and any(component[field] is not None for field in rate_fields):
                    raise PublisherError("isolated_sum cannot publish a derived data rate")
                if name != "isolated_sum" and any(component[field] is None for field in rate_fields):
                    raise PublisherError(f"{name} measured data rates are missing")
                latency = component["latency_us"]
                if list(latency.values()) != sorted(latency.values()):
                    raise PublisherError("latency percentiles are not ordered")
                byte_provenance = component["byte_provenance"]
                if byte_provenance["total_logical_bytes"] != (
                    byte_provenance["activation_data_bytes"] + byte_provenance["scale_bytes"]
                ):
                    raise PublisherError("component byte accounting does not reconcile")
                for field, byte_field in (
                    ("activation_data_rate_gbps_at_latency_percentile", "activation_data_bytes"),
                    ("total_logical_data_rate_gbps_at_latency_percentile", "total_logical_bytes"),
                ):
                    if component[field] is not None:
                        for statistic, rate in component[field].items():
                            expected = byte_provenance[byte_field] / (latency[statistic] * 1000.0)
                            if not math.isclose(rate, expected, rel_tol=1e-9, abs_tol=1e-12):
                                raise PublisherError("component GB/s formula differs")
            if components["roundtrip"] is None or components["roundtrip"]["origin"] != "measured":
                raise PublisherError("roundtrip must be measured")
            for statistic, throughput in point["roundtrip_token_rate_at_latency_percentile"].items():
                expected = point["global_tokens"] / (
                    components["roundtrip"]["latency_us"][statistic] * 1e-6
                )
                if not math.isclose(throughput, expected, rel_tol=1e-9):
                    raise PublisherError("roundtrip token throughput formula differs")
            if components["dispatch"] is not None:
                derived = components["isolated_sum"]
                if derived is None or any(not math.isclose(
                    derived["latency_us"][statistic],
                    components["dispatch"]["latency_us"][statistic]
                    + (
                        components["stage"]["latency_us"][statistic]
                        if components["stage"] is not None else 0.0
                    )
                    + components["combine"]["latency_us"][statistic], rel_tol=1e-12
                ) for statistic in ("p50", "p90", "p95", "p99")):
                    raise PublisherError("isolated_sum is not the component percentile sum")
            elif components["isolated_sum"] is not None:
                raise PublisherError("isolated_sum requires measured dispatch/combine components")
        if any(point["trial_diagnostics"]["flagged"] for point in item["points"]) != (
            "unresolved-trial-diagnostic" in item["eligibility"]["reasons"]
        ):
            raise PublisherError("series trial diagnostic eligibility is inconsistent")
    cohorts = {item["cohort_id"]: item for item in doc["cohorts"]}
    if len(cohorts) != len(doc["cohorts"]):
        raise PublisherError("dataset has duplicate cohort IDs")
    for item in doc["cohorts"]:
        if not set(item["series_ids"]).issubset(series):
            raise PublisherError("cohort references unknown series")
        members = [series[series_id] for series_id in item["series_ids"]]
        expected_tier = (
            "comparable-experimental"
            if any(member["publication_tier"] == "comparable-experimental" for member in members)
            else "official"
        )
        if item["publication_tier"] != expected_tier:
            raise PublisherError("cohort publication tier differs from its members")
        if f"/ {members[0]['mode']} /" not in item["label"]:
            raise PublisherError("cohort label omits its controlled mode")
        roles = {member["backend"]["role"] for member in members}
        if item["kind"] == "library" and roles != {"library"}:
            raise PublisherError("library cohort contains non-library evidence")
        if item["kind"] == "system" and roles != {"reference"}:
            raise PublisherError("system cohort is not a portable reference comparison")
        if item["kind"] in {"chip", *PRECISION_COHORT_KINDS} and len(
            {_canonical(member["backend"]) for member in members}
        ) != 1:
            raise PublisherError(f"{item['kind']} cohort mixes backend implementations")
        public_factors = [_public_cohort_factors(item["kind"], member) for member in members]
        if len({_canonical(value[0]) for value in public_factors}) != 1:
            raise PublisherError(f"{item['kind']} cohort does not control its public factors")
        if len({_canonical(value[1]) for value in public_factors}) < 2:
            raise PublisherError(f"{item['kind']} cohort does not vary its declared contrast")
        if item["kind"] in PRECISION_COHORT_KINDS:
            if item["publication_tier"] != "comparable-experimental":
                raise PublisherError("precision cohorts must be experimental")
            if item["kind"] in {"dispatch-precision", "combine-precision"}:
                axis = (
                    "dispatch"
                    if item["kind"] == "dispatch-precision"
                    else "combine"
                )
                field = f"{axis}_precision"
                bf16 = identity.precision_profile(
                    identity.V1_CONTROL_PRECISION_PROFILE
                )[axis]
                has_baseline = sum(
                    _canonical(member["workload"][field]) == _canonical(bf16)
                    for member in members
                ) == 1
                missing_reason = (
                    "missing-bf16-precision-baseline"
                    in item["eligibility"]["reasons"]
                )
                if has_baseline == missing_reason:
                    raise PublisherError(
                        "precision baseline and eligibility reason disagree"
                    )
        expected_id = _derived_id("cxcohort-v1-", {
            "kind": item["kind"], "series_ids": item["series_ids"],
            "controlled_factors": item["controlled_factors"],
            "varying_factors": item["varying_factors"],
        })
        if item["cohort_id"] != expected_id:
            raise PublisherError("cohort ID differs from its public factors")
        expected_factors = {
            "library": (
                ["system", "workload", "mode", "phase", "measurement", "resource.mode", "source"],
                ["backend", "resource"],
            ),
            "chip": (
                ["backend", "source", "workload", "mode", "phase", "measurement", "resource.mode"],
                ["system", "resource"],
            ),
            "system": (
                ["workload", "mode", "phase", "measurement", "source"],
                ["system", "backend", "resource"],
            ),
            "dispatch-precision": (
                [
                    "backend", "implementation-static-build", "system", "model-shape",
                    "mode", "phase", "workload.routing", "workload.eplb",
                    "measurement", "resource", "combine-precision",
                ],
                ["dispatch-precision"],
            ),
            "combine-precision": (
                [
                    "backend", "implementation-static-build", "system", "model-shape",
                    "mode", "phase", "workload.routing", "workload.eplb",
                    "measurement", "resource", "dispatch-precision",
                ],
                ["combine-precision"],
            ),
            "precision-pair": (
                [
                    "backend", "implementation-static-build", "system", "model-shape",
                    "mode", "phase", "workload.routing", "workload.eplb",
                    "measurement",
                ],
                [
                    "dispatch-precision", "combine-precision", "precision-profile",
                    "resource",
                ],
            ),
        }[item["kind"]]
        member_allocations = {
            allocation for series_id in item["series_ids"]
            for allocation in series[series_id]["allocation_ids"]
        }
        if (
            (item["controlled_factors"], item["varying_factors"]) != expected_factors
            or set(item["eligibility"]["allocation_ids"]) != member_allocations
        ):
            raise PublisherError("cohort factors or allocations differ from its members")
        _eligibility(item["eligibility"], f"cohort {item['cohort_id']}")
    expected_ranking_keys: set[tuple[str, str, str, int]] = set()
    for cohort in doc["cohorts"]:
        if not cohort["eligibility"]["decision_grade"]:
            continue
        members = [series[series_id] for series_id in cohort["series_ids"]]
        tokens = set.intersection(*(
            {point["tokens_per_rank"] for point in member["points"]}
            for member in members
        ))
        expected_ranking_keys.update(
            (cohort["cohort_id"], measure, statistic, token)
            for token in tokens
            for measure in (
                "latency_us", "activation_data_rate_gbps_at_latency_percentile",
                "total_logical_data_rate_gbps_at_latency_percentile",
            )
            for statistic in ("p50", "p99")
        )
    ranking_top: dict[
        tuple[str, str, str, int], dict[str, Any] | None
    ] = {}
    ranking_ids: set[str] = set()
    for ranking in doc["rankings"]:
        cohort = cohorts.get(ranking["cohort_id"])
        if (
            cohort is None
            or not cohort["eligibility"]["decision_grade"]
            or ranking["eligibility"] != cohort["eligibility"]
            or ranking["publication_tier"] != cohort["publication_tier"]
        ):
            raise PublisherError("ranking references an ineligible cohort")
        entries = ranking["entries"]
        _validate_metric(ranking["metric"])
        if cohort["kind"] == "library" and any(
            series[series_id]["backend"]["role"] == "reference"
            for series_id in cohort["series_ids"]
        ):
            raise PublisherError("reference evidence cannot drive a library ranking")
        if {entry["series_id"] for entry in entries} != set(cohort["series_ids"]):
            raise PublisherError("ranking does not cover its cohort")
        for entry in entries:
            point_id, value, unit = _metric_value(series[entry["series_id"]], ranking["metric"])
            if entry["point_id"] != point_id or entry["unit"] != unit or not math.isclose(entry["value"], value, rel_tol=1e-12):
                raise PublisherError("ranking entry differs from series data")
        reverse = ranking["metric"]["objective"] == "max"
        expected = sorted(entries, key=lambda entry: (entry["value"], entry["series_id"]), reverse=reverse)
        metric = ranking["metric"]
        ranks = [entry["rank"] for entry in entries]
        if metric["measure"] == "latency_us" and metric["statistic"] == "p99":
            tied_first = sum(rank == 1 for rank in ranks)
            expected_ranks = [1] * tied_first + list(
                range(tied_first + 1, len(entries) + 1)
            )
        else:
            expected_ranks = list(range(1, len(entries) + 1))
        if entries != expected or not ranks or ranks != expected_ranks:
            raise PublisherError("ranking order differs")
        expected_id = _derived_id("cxranking-v1-", {
            "cohort_id": ranking["cohort_id"], "metric": metric,
        })
        if ranking["ranking_id"] != expected_id or expected_id in ranking_ids:
            raise PublisherError("ranking ID is duplicate or differs")
        ranking_ids.add(expected_id)
        ranking_top[(ranking["cohort_id"], metric["measure"], metric["statistic"], metric["tokens_per_rank"])] = (
            entries[0] if ranks.count(1) == 1 else None
        )
    if set(ranking_top) != expected_ranking_keys:
        raise PublisherError("rankings do not cover every eligible cohort metric")
    objective = {
        "min-p50-latency": ("latency_us", "p50"), "min-p99-latency": ("latency_us", "p99"),
        "max-activation-data-rate-at-p50-latency": (
            "activation_data_rate_gbps_at_latency_percentile", "p50"
        ),
        "max-activation-data-rate-at-p99-latency": (
            "activation_data_rate_gbps_at_latency_percentile", "p99"
        ),
        "max-total-logical-data-rate-at-p50-latency": (
            "total_logical_data_rate_gbps_at_latency_percentile", "p50"
        ),
        "max-total-logical-data-rate-at-p99-latency": (
            "total_logical_data_rate_gbps_at_latency_percentile", "p99"
        ),
    }
    recommendation_ids: set[str] = set()
    for item in doc["recommendations"]:
        if item["objective"] != "min-p99-latency":
            raise PublisherError("recommendation is not a unique p99 latency winner")
        measure, statistic = objective[item["objective"]]
        candidates = [top for key, top in ranking_top.items()
                      if key[:3] == (item["cohort_id"], measure, statistic)
                      and top is not None and top["point_id"] == item["point_id"]]
        if len(candidates) != 1 or any(item[field] != candidates[0][field] for field in ("series_id", "point_id", "value", "unit")):
            raise PublisherError("recommendation is not a ranking winner")
        matching_ranking = next(
            ranking for ranking in doc["rankings"]
            if ranking["cohort_id"] == item["cohort_id"]
            and ranking["metric"]["measure"] == measure
            and ranking["metric"]["statistic"] == statistic
            and ranking["entries"][0]["point_id"] == item["point_id"]
        )
        expected_id = _derived_id("cxrecommendation-v1-", {
            "objective": item["objective"], "ranking_id": matching_ranking["ranking_id"],
        })
        cohort = cohorts[item["cohort_id"]]
        if (item["recommendation_id"] != expected_id or expected_id in recommendation_ids
                or cohort["publication_tier"] != "official"
                or item["publication_tier"] != "official"
                or item["eligibility"] != cohort["eligibility"]):
            raise PublisherError("recommendation ID/eligibility differs")
        recommendation_ids.add(expected_id)
    expected_recommendations = sum(
        cohorts[ranking["cohort_id"]]["publication_tier"] == "official"
        and ranking["metric"]["measure"] == "latency_us"
        and ranking["metric"]["statistic"] == "p99"
        and sum(entry["rank"] == 1 for entry in ranking["entries"]) == 1
        for ranking in doc["rankings"]
    )
    if len(doc["recommendations"]) != expected_recommendations:
        raise PublisherError("recommendations do not cover every actionable ranking")
    sensitivity_ids: set[str] = set()
    sensitivity_keys: set[tuple[str, str, str, str, str, int]] = set()
    for item in doc["sensitivities"]:
        cohort = cohorts.get(item["cohort_id"])
        if (
            cohort is None
            or cohort["kind"] not in {
                "dispatch-precision", "combine-precision",
            }
            or not cohort["eligibility"]["decision_grade"]
            or item["publication_tier"] != cohort["publication_tier"]
            or item["eligibility"] != cohort["eligibility"]
        ):
            raise PublisherError("sensitivity references an ineligible contrast cohort")
        if (
            item["baseline_series_id"] == item["candidate_series_id"]
            or not {item["baseline_series_id"], item["candidate_series_id"]}.issubset(cohort["series_ids"])
        ):
            raise PublisherError("sensitivity series differ from its cohort")
        _validate_metric(item["metric"])
        baseline_series = series[item["baseline_series_id"]]
        axis = (
            "dispatch"
            if cohort["kind"] == "dispatch-precision"
            else "combine"
        )
        field = f"{axis}_precision"
        bf16 = identity.precision_profile(
            identity.V1_CONTROL_PRECISION_PROFILE
        )[axis]
        if _canonical(baseline_series["workload"][field]) != _canonical(bf16):
            raise PublisherError("precision sensitivity baseline is not BF16")
        _, baseline, _ = _metric_value(series[item["baseline_series_id"]], item["metric"])
        _, candidate, _ = _metric_value(series[item["candidate_series_id"]], item["metric"])
        if not math.isclose(item["signed_change_ratio"], (candidate - baseline) / baseline, rel_tol=1e-12):
            raise PublisherError("sensitivity ratio differs")
        expected_id = _derived_id("cxsensitivity-v1-", {
            "baseline": item["baseline_series_id"],
            "candidate": item["candidate_series_id"],
            "cohort": item["cohort_id"], "metric": item["metric"],
        })
        if item["sensitivity_id"] != expected_id or expected_id in sensitivity_ids:
            raise PublisherError("sensitivity ID is duplicate or differs")
        sensitivity_ids.add(expected_id)
        sensitivity_keys.add((
            item["cohort_id"], item["baseline_series_id"], item["candidate_series_id"],
            item["metric"]["measure"], item["metric"]["statistic"],
            item["metric"]["tokens_per_rank"],
        ))
    expected_sensitivity_keys: set[tuple[str, str, str, str, str, int]] = set()
    for cohort in doc["cohorts"]:
        if (
            cohort["kind"] not in {
                "dispatch-precision", "combine-precision",
            }
            or not cohort["eligibility"]["decision_grade"]
        ):
            continue
        members = [series[series_id] for series_id in cohort["series_ids"]]
        axis = (
            "dispatch"
            if cohort["kind"] == "dispatch-precision"
            else "combine"
        )
        field = f"{axis}_precision"
        bf16 = identity.precision_profile(
            identity.V1_CONTROL_PRECISION_PROFILE
        )[axis]
        baseline = next((
            member for member in members
            if _canonical(member["workload"][field]) == _canonical(bf16)
        ), None)
        if baseline is None:
            continue
        tokens = set.intersection(*(
            {point["tokens_per_rank"] for point in member["points"]}
            for member in members
        ))
        expected_sensitivity_keys.update(
            (cohort["cohort_id"], baseline["series_id"], candidate["series_id"],
             measure, statistic, token)
            for candidate in members if candidate is not baseline
            for token in tokens
            for measure in (
                "latency_us", "activation_data_rate_gbps_at_latency_percentile",
                "total_logical_data_rate_gbps_at_latency_percentile",
            )
            for statistic in ("p50", "p99")
        )
    if sensitivity_keys != expected_sensitivity_keys:
        raise PublisherError("sensitivities do not cover every declared contrast metric")
    observed_qualification_indices = sorted({
        item["qualification_index"] for item in doc["attempts"] if item["selected"]
    })
    if promotion["qualification_indices"] != observed_qualification_indices:
        raise PublisherError("promotion qualification index catalog differs from attempts")
    if promotion["status"] == "promoted":
        run_ids = {item["run_id"] for item in doc["attempts"] if item["selected"]}
        repeated_cases = all(
            {
                attempts[attempt_id]["qualification_index"]
                for attempt_id in coverage["attempt_ids"]
                if attempts[attempt_id]["selected"]
            } == {1, 2, 3}
            for coverage in doc["coverage"]
        )
        if promotion["matrix_id"] != CANONICAL_FULL_V1_MATRIX_SHA256:
            raise PublisherError("promotion requires the canonical full-v1 matrix")
        if (
            _case_disposition_catalog_sha256(doc["coverage"])
            != CANONICAL_FULL_V1_CASE_CATALOG_SHA256
        ):
            raise PublisherError("promotion requires the canonical case/disposition catalog")
        if (
            terminal != len(doc["coverage"])
            or promotion["qualification_indices"] != [1, 2, 3]
            or promotion["measured_cases"] + promotion["unsupported_cases"]
            != promotion["requested_cases"]
            or promotion["measured_points"] + promotion["unsupported_points"]
            != promotion["requested_points"]
            or promotion["terminal_points"] != promotion["requested_points"]
            or len(doc["source_bundle_ids"]) != REQUIRED_ALLOCATIONS
            or len(run_ids) != REQUIRED_ALLOCATIONS
            or not repeated_cases
        ):
            raise PublisherError("promoted dataset lacks complete coverage")
        expected_outcomes = {
            item["case_id"]: (
                "success" if item["disposition"] == "runnable" else "unsupported"
            )
            for item in doc["coverage"]
        }
        if any(
            item["selected"]
            and item["outcome"] != expected_outcomes[item["case_id"]]
            for item in doc["attempts"]
        ):
            raise PublisherError("promoted outcomes differ from requested dispositions")
        runnable_cases = {
            item["case_id"] for item in doc["coverage"]
            if item["disposition"] == "runnable"
        }
        if any(
            item["case_id"] in runnable_cases and item["outcome"] != "success"
            for item in doc["attempts"]
        ):
            raise PublisherError(
                "promotion rejects runnable cases with failed, invalid, or diagnostic retries"
            )
        _require_promotion_series(doc["series"])
        _require_promotion_cohorts(doc["cohorts"], doc["series"])
        if not doc["rankings"]:
            raise PublisherError("promoted dataset lacks eligible rankings")
    if promotion["status"] == "quarantined" and any((
        doc["source_bundle_ids"], promotion["allocation_ids"], doc["coverage"],
        doc["attempts"], doc["series"], doc["cohorts"], doc["rankings"],
        doc["recommendations"], doc["sensitivities"],
    )):
        raise PublisherError("quarantined dataset exposes unvalidated evidence")
    return doc


def _file_record(value: Any, path: str) -> dict[str, Any]:
    item = _exact(value, {"path", "sha256", "bytes"}, path)
    if not isinstance(item["path"], str) or PurePosixPath(item["path"]).is_absolute() or ".." in PurePosixPath(item["path"]).parts:
        raise PublisherError(f"{path}.path is unsafe")
    if not isinstance(item["sha256"], str) or HEX64.fullmatch(item["sha256"]) is None:
        raise PublisherError(f"{path}.sha256 is invalid")
    _integer(item["bytes"], f"{path}.bytes", minimum=1)
    return item

def validate_bundle_manifest(doc: Any) -> dict[str, Any]:
    _schema("private-bundle-v1.schema.json", doc)
    attempts = {item["attempt_id"]: item for item in doc["attempts"]}
    if len(attempts) != len(doc["attempts"]):
        raise PublisherError("bundle has duplicate attempt IDs")
    selections = doc["coverage"]["selections"]
    if len({item["case_id"] for item in selections}) != len(selections):
        raise PublisherError("bundle has duplicate selected cases")
    counts = {name: 0 for name in OUTCOMES}
    for selection in selections:
        attempt = attempts.get(selection["selected_attempt_id"])
        if attempt is None or not attempt["selected"] or attempt["case_id"] != selection["case_id"] or attempt["outcome"] != selection["outcome"]:
            raise PublisherError("bundle selection differs from retained attempt")
        counts[selection["outcome"]] += 1
    coverage = doc["coverage"]
    if coverage["terminal_cases"] != len(selections) or coverage["outcome_counts"] != counts:
        raise PublisherError("bundle terminal counts differ")
    if coverage["complete"] != (coverage["expected_cases"] == len(selections)):
        raise PublisherError("bundle completeness differs from coverage")
    fingerprints: dict[str, set[str]] = {}
    for attempt in doc["attempts"]:
        value = attempt["runtime_fingerprint_sha256"]
        if value:
            fingerprints.setdefault(attempt["allocation_id"], set()).add(value)
    if any(len(values) != 1 for values in fingerprints.values()):
        raise PublisherError("bundle runtime is heterogeneous within an allocation")
    return doc


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes(path: Path, data: bytes, *, mode: int) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(descriptor, view):]


def _write_json(path: Path, value: Any, *, mode: int) -> bytes:
    data = _canonical(value) + b"\n"
    _write_bytes(path, data, mode=mode)
    return data


def _file_metadata(path: Path, relative_to: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": _sha_file(path),
        "bytes": path.stat().st_size,
    }


def _tree_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and not path.is_symlink() and path.name != "COMPLETE"
    )


def _verify_regular_file(path: Path, expected_mode: int) -> None:
    _reject_symlinked_path(path.parent)
    try:
        metadata = os.lstat(path)
    except FileNotFoundError as exc:
        raise PublisherError(f"required file is missing: {path.name}") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != expected_mode
    ):
        raise PublisherError(
            f"file is not an owned regular {expected_mode:o} object: {path.name}"
        )


def _verify_frozen_tree(root: Path, *, private: bool) -> None:
    _reject_symlinked_path(root)
    directory_mode = 0o500 if private else 0o555
    file_mode = 0o400 if private else 0o444
    try:
        root_metadata = os.lstat(root)
    except OSError as exc:
        raise PublisherError(f"cannot inspect immutable object: {root.name}") from exc
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise PublisherError(f"immutable object is not a real directory: {root.name}")
    try:
        entries = [root, *root.rglob("*")]
    except OSError as exc:
        raise PublisherError(f"cannot inspect immutable object: {root.name}") from exc
    for path in entries:
        metadata = os.lstat(path)
        if metadata.st_uid != os.getuid():
            raise PublisherError(f"immutable object has the wrong owner: {path.name}")
        if stat.S_ISDIR(metadata.st_mode):
            expected = directory_mode
        elif stat.S_ISREG(metadata.st_mode):
            expected = file_mode
        else:
            raise PublisherError(f"immutable object contains a linked or special entry: {path.name}")
        if stat.S_IMODE(metadata.st_mode) != expected:
            raise PublisherError(
                f"immutable object mode differs for {path.name}: expected {expected:o}"
            )


def _freeze_tree(root: Path, *, private: bool) -> None:
    files: list[Path] = []
    directories = [root]
    for path in root.rglob("*"):
        metadata = os.lstat(path)
        if stat.S_ISDIR(metadata.st_mode):
            directories.append(path)
        elif stat.S_ISREG(metadata.st_mode):
            files.append(path)
        else:
            raise PublisherError(f"immutable object contains a linked or special entry: {path.name}")
    for path in files:
        os.chmod(path, 0o400 if private else 0o444)
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        os.chmod(path, 0o500 if private else 0o555)
        _fsync_dir(path)
    _verify_frozen_tree(root, private=private)


def _reject_symlinked_path(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            break
        if stat.S_ISLNK(metadata.st_mode):
            raise PublisherError("COLLECTIVEX_STORE_ROOT must not traverse a symlinked parent")
        if not stat.S_ISDIR(metadata.st_mode):
            raise PublisherError(f"store path component is not a directory: {current}")


class Store:
    """Atomic private/public directory operations on one operator filesystem."""

    def __init__(self, root: str | os.PathLike[str]):
        candidate = Path(os.path.abspath(os.path.expanduser(root)))
        _reject_symlinked_path(candidate)
        candidate.mkdir(parents=True, exist_ok=True, mode=0o750)
        resolved = candidate.resolve()
        if candidate != resolved:
            raise PublisherError(
                "COLLECTIVEX_STORE_ROOT must not traverse a symlinked parent"
            )
        root_metadata = candidate.stat()
        if root_metadata.st_uid != os.getuid() or stat.S_IMODE(root_metadata.st_mode) & 0o022:
            raise PublisherError(
                "COLLECTIVEX_STORE_ROOT must be owned by this user and not group/world writable"
            )
        os.chmod(candidate, 0o750)
        if stat.S_IMODE(candidate.stat().st_mode) != 0o750:
            raise PublisherError("COLLECTIVEX_STORE_ROOT mode must be 750")
        self.root = resolved
        raw = self.root
        self.private = raw / "private"
        self.incoming = self.private / "incoming"
        self.bundles = self.private / "bundles"
        self.quarantine = self.private / "quarantine"
        self.public = raw / "public"
        self.datasets = self.public / "datasets"
        self.channels = self.public / "channels"
        self.locks = raw / "locks"
        for path, mode in (
            (self.private, 0o700), (self.incoming, 0o700), (self.bundles, 0o700),
            (self.quarantine, 0o700), (self.public, 0o755), (self.datasets, 0o755),
            (self.channels, 0o755), (self.locks, 0o700),
        ):
            path.mkdir(parents=True, exist_ok=True, mode=mode)
            if path.is_symlink() or not path.is_dir():
                raise PublisherError(f"store path is not a real directory: {path}")
            os.chmod(path, mode)

    @contextlib.contextmanager
    def locked(self) -> Iterator[None]:
        lock_path = self.locks / "publisher.lock"
        descriptor = os.open(
            lock_path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            os.fchmod(descriptor, 0o600)
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600
            ):
                raise PublisherError("publisher lock is not an owned regular 600 file")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    @contextlib.contextmanager
    def staging(self, parent: Path, *, private: bool) -> Iterator[Path]:
        stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=parent))
        os.chmod(stage, 0o700 if private else 0o755)
        try:
            yield stage
        finally:
            if stage.exists():
                for path in stage.rglob("*"):
                    metadata = os.lstat(path)
                    if stat.S_ISDIR(metadata.st_mode):
                        os.chmod(path, 0o700)
                    elif stat.S_ISREG(metadata.st_mode):
                        os.chmod(path, 0o600)
                os.chmod(stage, 0o700)
            shutil.rmtree(stage, ignore_errors=True)

    @staticmethod
    def complete(stage: Path, value: str, *, private: bool) -> None:
        _write_bytes(stage / "COMPLETE", (value + "\n").encode(), mode=0o600 if private else 0o644)
        _fsync_dir(stage)

    @staticmethod
    def install(stage: Path, destination: Path, *, private: bool) -> None:
        if destination.is_symlink():
            raise PublisherError(f"immutable destination is a symlink: {destination.name}")
        if destination.exists():
            _verify_frozen_tree(destination, private=private)
            marker = destination / "COMPLETE"
            if not marker.is_file() or marker.read_text().strip() != destination.name:
                raise PublisherError(f"immutable destination is incomplete: {destination.name}")
            return
        _freeze_tree(stage, private=private)
        os.rename(stage, destination)
        _fsync_dir(destination.parent)
        _verify_frozen_tree(destination, private=private)

    def install_dataset(self, dataset: dict[str, Any]) -> tuple[str, int]:
        validate_public_dataset(dataset)
        payload = _canonical(dataset) + b"\n"
        if len(payload) > MAX_PUBLIC_DATASET_BYTES:
            raise PublisherError("public dataset exceeds the serving size limit")
        digest = _sha_bytes(payload)
        destination = self.datasets / digest
        with self.staging(self.datasets, private=False) as stage:
            _write_bytes(stage / "dataset.json", payload, mode=0o644)
            self.complete(stage, digest, private=False)
            self.install(stage, destination, private=False)
        stored = destination / "dataset.json"
        marker = destination / "COMPLETE"
        if (not marker.is_file() or marker.read_text().strip() != digest
                or _sha_file(stored) != digest or stored.stat().st_size != len(payload)):
            raise PublisherError("stored dataset checksum differs after installation")
        return digest, len(payload)

    def update_channel(self, channel: str, digest: str, size: int, generated_at: str) -> None:
        if size > MAX_PUBLIC_DATASET_BYTES:
            raise PublisherError("channel dataset exceeds the serving size limit")
        _verify_frozen_tree(self.datasets / digest, private=False)
        marker = self.datasets / digest / "COMPLETE"
        if not marker.is_file() or marker.read_text().strip() != digest:
            raise PublisherError("cannot advance a channel to an incomplete dataset")
        dataset_path = self.datasets / digest / "dataset.json"
        dataset = validate_public_dataset(strict_load(dataset_path))
        if (
            _sha_file(dataset_path) != digest
            or dataset_path.stat().st_size != size
            or dataset["generated_at"] != generated_at
        ):
            raise PublisherError("channel metadata differs from its stored dataset")
        if channel == "dev-latest" and dataset["promotion"]["status"] != "promoted":
            raise PublisherError("dev-latest may only reference a promoted dataset")
        pointer = {
            "format": FORMAT_CHANNEL,
            "channel": channel,
            "dataset": {
                "path": f"datasets/{digest}/dataset.json",
                "sha256": digest,
                "bytes": size,
            },
            "generated_at": generated_at,
        }
        validate_channel(pointer, expected_channel=channel)
        destination = self.channels / f"{channel}.json"
        temporary = self.channels / f".{channel}.tmp-{os.getpid()}"
        try:
            data = _canonical(pointer) + b"\n"
            _write_bytes(temporary, data, mode=0o644)
            os.replace(temporary, destination)
            _fsync_dir(self.channels)
        finally:
            temporary.unlink(missing_ok=True)

    def verify_channel(self, channel: str) -> dict[str, Any]:
        channel_path = self.channels / f"{channel}.json"
        _verify_regular_file(channel_path, 0o644)
        pointer = validate_channel(strict_load(channel_path), expected_channel=channel)
        target = self.public / pointer["dataset"]["path"]
        _verify_frozen_tree(target.parent, private=False)
        if target.stat().st_size != pointer["dataset"]["bytes"] or _sha_file(target) != pointer["dataset"]["sha256"]:
            raise PublisherError(f"channel {channel} dataset checksum differs")
        marker = target.parent / "COMPLETE"
        if not marker.is_file() or marker.read_text().strip() != pointer["dataset"]["sha256"]:
            raise PublisherError(f"channel {channel} dataset is incomplete")
        dataset = validate_public_dataset(strict_load(target))
        if pointer["generated_at"] != dataset["generated_at"]:
            raise PublisherError(f"channel {channel} metadata differs from its dataset")
        if channel == "dev-latest" and dataset["promotion"]["status"] != "promoted":
            raise PublisherError("dev-latest points to a non-promoted dataset")
        return pointer


def _copy_source(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file() or not stat.S_ISREG(source.stat().st_mode):
        raise PublisherError(f"source must be a regular non-symlink file: {source}")
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        output = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                _write_all(output, chunk)
            os.fsync(output)
        finally:
            os.close(output)
    finally:
        os.close(descriptor)


def _archive_download_directory(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_dir():
        raise PublisherError(f"artifact directory is invalid: {source}")
    files: list[Path] = []
    for path in source.rglob("*"):
        if path.is_symlink():
            raise PublisherError("artifact directory contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise PublisherError("artifact directory contains a non-regular entry")
        files.append(path)
    files.sort()
    if not files or len(files) > MAX_ARCHIVE_MEMBERS:
        raise PublisherError("artifact directory has an invalid file count")
    total = 0
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_STORED) as archive:
        for path in files:
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(descriptor, "rb") as handle:
                metadata = os.fstat(handle.fileno())
                if not stat.S_ISREG(metadata.st_mode):
                    raise PublisherError("artifact directory member changed type")
                size = metadata.st_size
                total += size
                if size > MAX_ARCHIVE_MEMBER_BYTES or total > MAX_ARCHIVE_TOTAL_BYTES:
                    raise PublisherError("artifact directory exceeds size limits")
                relative = path.relative_to(source).as_posix()
                _safe_member(relative)
                info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = (stat.S_IFREG | 0o600) << 16
                with archive.open(info, "w") as output:
                    written = 0
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        output.write(chunk)
                        written += len(chunk)
                    if written != size:
                        raise PublisherError("artifact directory member changed size")
    descriptor = os.open(destination, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _artifact_name(source: Path) -> str:
    name = source.name if source.is_dir() else source.name.removesuffix(".zip")
    if (
        not source.is_dir() and source.suffix != ".zip"
        or ARTIFACT_NAME.fullmatch(name) is None
    ):
        raise PublisherError(f"artifact source has an invalid GHA name: {source.name}")
    return name


def archive_incoming(
    store: Store,
    matrix: Path,
    artifacts: Sequence[Path],
    run: dict[str, Any],
) -> tuple[str, Path, list[dict[str, Any]]]:
    """Copy exact delivery bytes into immutable incoming before any JSON/ZIP parse."""
    if not artifacts:
        raise PublisherError("at least one GitHub artifact archive is required")
    with store.staging(store.incoming, private=True) as stage:
        sources = stage / "sources"
        sources.mkdir(mode=0o700)
        copied: list[dict[str, Any]] = []
        named_artifacts = sorted(
            ((_artifact_name(path), path) for path in artifacts), key=lambda item: item[0]
        )
        artifact_names = [name for name, _ in named_artifacts]
        if len(artifact_names) != len(set(artifact_names)):
            raise PublisherError("artifact delivery contains duplicate GHA names")
        inputs = [("matrix.json", matrix, "matrix", None)] + [
            (f"artifact-{index:04d}.zip", path, "artifact", artifact_name)
            for index, (artifact_name, path) in enumerate(named_artifacts)
        ]
        for name, source, kind, artifact_name in inputs:
            destination = sources / name
            if source.is_dir():
                _archive_download_directory(source, destination)
            else:
                if source != matrix and source.stat().st_size > MAX_ARCHIVE_TOTAL_BYTES:
                    raise PublisherError("artifact archive exceeds the size limit")
                _copy_source(source, destination)
            copied.append({
                **_file_metadata(destination, stage),
                "kind": kind,
                "artifact_name": artifact_name,
            })
        ingest_id = _sha_bytes(_canonical({"run": run, "sources": copied}))
        incoming_manifest = {
            "format": "collectivex.incoming.v1",
            "schema_version": 1,
            "ingest_id": ingest_id,
            "run": run,
            "sources": copied,
        }
        _write_json(stage / "incoming.json", incoming_manifest, mode=0o600)
        store.complete(stage, ingest_id, private=True)
        destination = store.incoming / ingest_id
        store.install(stage, destination, private=True)
    installed = store.incoming / ingest_id
    if strict_load(installed / "incoming.json") != incoming_manifest:
        raise PublisherError("existing incoming object differs from archived delivery")
    for record in copied:
        _resolve_bundle_file(installed, record)
    return ingest_id, installed, copied


def _safe_member(name: str) -> PurePosixPath:
    if "\\" in name or "\0" in name:
        raise PublisherError("archive member has an unsafe separator")
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise PublisherError("archive member path escapes its artifact")
    return path


def extract_archive(archive: Path, destination: Path) -> list[Path]:
    """Extract a bounded regular-file ZIP without trusting member paths or links."""
    try:
        handle = zipfile.ZipFile(archive)
    except (OSError, zipfile.BadZipFile) as exc:
        raise PublisherError("artifact is not a valid ZIP archive") from exc
    extracted: list[Path] = []
    seen: set[str] = set()
    total = 0
    with handle:
        members = handle.infolist()
        if not members or len(members) > MAX_ARCHIVE_MEMBERS:
            raise PublisherError("artifact has an invalid member count")
        for member in members:
            path = _safe_member(member.filename.rstrip("/"))
            key = path.as_posix()
            if key in seen:
                raise PublisherError("artifact contains duplicate member paths")
            seen.add(key)
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode) or (mode and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))):
                raise PublisherError("artifact contains a non-regular member")
            if member.flag_bits & 0x1:
                raise PublisherError("encrypted artifact members are not accepted")
            if member.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                raise PublisherError("artifact member exceeds the size limit")
            total += member.file_size
            if total > MAX_ARCHIVE_TOTAL_BYTES:
                raise PublisherError("artifact exceeds the expanded size limit")
            target = destination.joinpath(*path.parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True, mode=0o700)
                continue
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            output = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with handle.open(member, "r") as source:
                    written = 0
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        _write_all(output, chunk)
                        written += len(chunk)
                    if written != member.file_size:
                        raise PublisherError("artifact member size changed during extraction")
                os.fsync(output)
            finally:
                os.close(output)
            extracted.append(target)
    return extracted


def validate_matrix(document: Any) -> list[dict[str, Any]]:
    try:
        artifact_safety.assert_publication_safe([document])
        matrix = sweep_matrix.validate_matrix_document(document)
    except (SystemExit, ValueError, artifact_safety.ArtifactSafetyError) as exc:
        raise PublisherError(f"requested matrix is invalid: {exc}") from exc
    return [
        {
            "sku": item["sku"],
            **item["case"],
            "_disposition": item["disposition"],
            "_reason": item["reason"],
        }
        for item in matrix["requested_cases"]
    ]


def _expected_deliveries(
    matrix: dict[str, Any], cases: Sequence[dict[str, Any]], run: dict[str, Any]
) -> dict[str, tuple[str, str, str]]:
    shard_by_case: dict[str, str] = {}
    for shard in matrix["include"]:
        for case_id in shard["case_ids"]:
            if case_id in shard_by_case:
                raise PublisherError("requested case appears in two runnable shards")
            shard_by_case[case_id] = shard["id"]
    suffix = f"{run['run_id']}-{run['run_attempt']}"
    deliveries: dict[str, tuple[str, str, str]] = {}
    for case in cases:
        case_id = case["case_id"]
        if case["_disposition"] == "unsupported":
            deliveries[case_id] = (
                f"cxunsupported-{suffix}", "setup",
                f"{run['run_id']}_{run['run_attempt']}_unsupported",
            )
            continue
        shard_id = shard_by_case.get(case_id)
        if shard_id is None:
            raise PublisherError("runnable case has no matrix shard")
        deliveries[case_id] = (
            f"cxshard-{shard_id}-{suffix}", "sweep",
            f"{run['run_id']}_{run['run_attempt']}_{shard_id}",
        )
    return deliveries


def _document_git_run(document: dict[str, Any]) -> dict[str, Any] | None:
    provenance = document.get("provenance")
    if not isinstance(provenance, dict):
        return None
    value = provenance.get("git_run", provenance)
    return value if isinstance(value, dict) else None


def _run_matches(document: dict[str, Any], run: dict[str, Any]) -> bool:
    git_run = _document_git_run(document)
    if git_run is None:
        return False
    return (
        str(git_run.get("run_id")) == run["run_id"]
        and str(git_run.get("run_attempt")) == str(run["run_attempt"])
        and git_run.get("qualification_index") == run["qualification_index"]
        and git_run.get("source_sha") == run["source_sha"]
        and (git_run.get("repo") or git_run.get("repository")) == run["repository"]
    )


def _case_matches(document: dict[str, Any], expected: dict[str, Any]) -> bool:
    scheduled = {
        key: value for key, value in expected.items()
        if key not in {"sku", "case_id"} and not key.startswith("_")
    }
    return document.get("identity", {}).get("case_factors") == {
        "case": scheduled,
        "profile": identity.profile_for_case(scheduled),
        "sku": expected["sku"],
    }


def _outcome(document: dict[str, Any]) -> tuple[str, str | None]:
    status = document["outcome"]["status"]
    if status == "success":
        return status, None
    native = document["outcome"].get("reason")
    reason = native if isinstance(native, str) and REASON.fullmatch(native) else {
        "unsupported": "unsupported-capability", "failed": "execution-failed",
        "invalid": "validation-failed", "diagnostic": "diagnostic-evidence",
    }.get(status)
    if reason is None:
        raise PublisherError(f"unsupported native outcome {status!r}")
    return status, reason


def _attempt_record(
    document: dict[str, Any], path: Path, root: Path, *, selected: bool
) -> dict[str, Any]:
    normalized = contracts.normalize_attempt(document)
    runtime = normalized["runtime_fingerprint"]
    runtime_sha = _sha_bytes(_canonical(runtime)) if runtime is not None else None
    sample_record = None
    evidence_ids: list[str] = []
    series_ids: list[str] = []
    if document["format"] == contracts.RAW_FORMAT:
        sample_path = path.with_name(document["sample_artifact"]["path"])
        sample_record = _file_metadata(sample_path, root)
        evidence_ids = [row["evidence_id"] for row in document["measurement"]["rows"]]
        series_ids = [document["identity"]["series_id"]]
        declared = document["identity"]["series_factors"]["runtime_fingerprint_sha256"]
        if runtime_sha != declared:
            raise PublisherError("runtime fingerprint checksum differs from series identity")
    status, reason = _outcome(document)
    return {
        "attempt_id": normalized["attempt_id"],
        "allocation_id": normalized["allocation_id"],
        "case_id": normalized["case_id"],
        "outcome": status,
        "reason": reason,
        "selected": selected,
        "document": _file_metadata(path, root),
        "samples": sample_record,
        "runtime_fingerprint_sha256": runtime_sha,
        "series_ids": series_ids,
        "evidence_ids": evidence_ids,
    }


def _validate_delivery_binding(
    document: dict[str, Any], path: Path, raw_root: Path,
    artifact_by_root: dict[str, str], expected_by_id: dict[str, dict[str, Any]],
    expected_deliveries: dict[str, tuple[str, str, str]], run: dict[str, Any],
) -> str:
    case_id = document["identity"]["case_id"]
    if case_id not in expected_by_id:
        raise PublisherError("artifact contains an extra case outcome")
    expected = expected_by_id[case_id]
    if not _case_matches(document, expected):
        raise PublisherError("attempt case coordinates differ from the requested matrix")
    unsupported = document["outcome"]["status"] == "unsupported"
    if (expected["_disposition"] == "unsupported") != unsupported:
        raise PublisherError("terminal outcome differs from requested capability disposition")
    if unsupported and document["outcome"]["reason"] != expected["_reason"]:
        raise PublisherError("unsupported outcome reason differs from requested matrix")
    if not _run_matches(document, run):
        raise PublisherError("attempt provenance differs from publisher run metadata")
    relative = path.relative_to(raw_root)
    if len(relative.parts) < 2:
        raise PublisherError("attempt document is outside a delivered artifact")
    delivered_name = artifact_by_root.get(relative.parts[0])
    expected_name, expected_job, expected_execution = expected_deliveries[case_id]
    git_run = _document_git_run(document)
    allocation = document["identity"]["allocation_factors"]
    if (
        git_run is None
        or delivered_name != expected_name
        or git_run["artifact"] != delivered_name
        or git_run["job"] != expected_job
        or allocation["execution_id"] != expected_execution
    ):
        raise PublisherError("attempt provenance differs from its delivered GHA shard")
    return case_id


def _parse_extracted(root: Path) -> tuple[list[tuple[Path, dict[str, Any]]], set[Path]]:
    attempts: list[tuple[Path, dict[str, Any]]] = []
    consumed_samples: set[Path] = set()
    json_paths = sorted(path for path in root.rglob("*.json") if path.is_file())
    for path in json_paths:
        if path in consumed_samples:
            continue
        try:
            document = contracts.strict_load(path)
            artifact_safety.assert_publication_safe([document])
            format_name = document.get("format") if isinstance(document, dict) else None
            if format_name == contracts.SAMPLES_FORMAT:
                _schema("samples-v1.schema.json", document)
                # It must be claimed by a raw document; orphan checking happens after the scan.
                continue
            if format_name == contracts.RAW_FORMAT:
                _schema("raw-case-v1.schema.json", document)
                sample_path = path.with_name(document["sample_artifact"]["path"])
                sample_document = contracts.strict_load(sample_path)
                artifact_safety.assert_publication_safe([sample_document])
                _schema("samples-v1.schema.json", sample_document)
                validated = contracts.load_raw_attempt(path)
                consumed_samples.add(sample_path)
            elif format_name == contracts.TERMINAL_FORMAT:
                _schema("terminal-outcome-v1.schema.json", document)
                validated = contracts.validate_terminal_document(document)
            else:
                raise PublisherError(f"artifact contains unknown JSON document {path.name}")
        except (
            contracts.ContractError, artifact_safety.ArtifactSafetyError,
            jsonschema.ValidationError, OSError,
        ) as exc:
            raise PublisherError(f"native contract rejected {path.name}: {exc}") from exc
        attempts.append((path, validated))
    orphan_samples = [
        path for path in json_paths
        if isinstance((doc := contracts.strict_load(path)), dict)
        and doc.get("format") == contracts.SAMPLES_FORMAT
        and path not in consumed_samples
    ]
    if orphan_samples:
        raise PublisherError("artifact contains an orphan samples document")
    if not attempts:
        raise PublisherError("artifact contains zero native attempt documents")
    return attempts, consumed_samples


def build_bundle(
    store: Store,
    incoming_id: str,
    incoming_path: Path,
    run: dict[str, Any],
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    """Validate one exact workflow delivery and install its immutable private bundle."""
    incoming_manifest = strict_load(incoming_path / "incoming.json")
    _exact(
        incoming_manifest,
        {"format", "schema_version", "ingest_id", "run", "sources"},
        "incoming",
    )
    artifact_safety.assert_publication_safe([incoming_manifest])
    if (
        incoming_manifest["format"] != "collectivex.incoming.v1"
        or incoming_manifest["schema_version"] != 1
        or incoming_manifest["ingest_id"] != incoming_id
        or incoming_manifest["run"] != run
        or _sha_bytes(_canonical({"run": run, "sources": incoming_manifest["sources"]}))
        != incoming_id
    ):
        raise PublisherError("incoming manifest identity differs from archived delivery")
    incoming_sources = _array(incoming_manifest["sources"], "incoming.sources", nonempty=True)
    for index, record in enumerate(incoming_sources):
        _exact(
            record,
            {"path", "sha256", "bytes", "kind", "artifact_name"},
            f"incoming.sources[{index}]",
        )
        _resolve_bundle_file(incoming_path, record)
    matrix_records = [record for record in incoming_sources if record["kind"] == "matrix"]
    artifact_records = [record for record in incoming_sources if record["kind"] == "artifact"]
    if (
        len(matrix_records) != 1
        or matrix_records[0]["artifact_name"] is not None
        or not artifact_records
        or any(ARTIFACT_NAME.fullmatch(record["artifact_name"] or "") is None
               for record in artifact_records)
        or len({record["artifact_name"] for record in artifact_records}) != len(artifact_records)
    ):
        raise PublisherError("incoming source catalog is invalid")
    matrix_source = _resolve_bundle_file(incoming_path, matrix_records[0])
    matrix_document = strict_load(matrix_source)
    expected_cases = validate_matrix(matrix_document)
    expected_by_id = {case["case_id"]: case for case in expected_cases}
    expected_deliveries = _expected_deliveries(matrix_document, expected_cases, run)
    if {record["artifact_name"] for record in artifact_records} != {
        delivery[0] for delivery in expected_deliveries.values()
    }:
        raise PublisherError("incoming artifact archive set differs from requested matrix shards")
    with store.staging(store.bundles, private=True) as stage:
        source_copy = stage / "source"
        raw_root = stage / "raw"
        source_copy.mkdir(mode=0o700)
        raw_root.mkdir(mode=0o700)
        matrix_path = stage / "matrix.json"
        _copy_source(matrix_source, matrix_path)
        source_records: list[dict[str, Any]] = []
        artifact_by_root: dict[str, str] = {}
        for index, source_record in enumerate(artifact_records):
            archive = _resolve_bundle_file(incoming_path, source_record)
            copied = source_copy / f"artifact-{index:04d}.zip"
            _copy_source(archive, copied)
            source_records.append({
                **_file_metadata(copied, stage),
                "artifact_name": source_record["artifact_name"],
            })
            artifact_root = raw_root / f"artifact-{index:04d}"
            artifact_root.mkdir(mode=0o700)
            artifact_by_root[artifact_root.name] = source_record["artifact_name"]
            extract_archive(copied, artifact_root)
        parsed, consumed_samples = _parse_extracted(raw_root)
        created_at = _latest_timestamp(
            [document["generated_at"] for _, document in parsed]
        )
        consumed_files = {path for path, _ in parsed} | consumed_samples
        extracted_files = {
            path for path in raw_root.rglob("*")
            if path.is_file() and not path.is_symlink()
        }
        if consumed_files != extracted_files:
            raise PublisherError("artifact contains an unconsumed non-native member")
        by_case: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
        for path, document in parsed:
            case_id = _validate_delivery_binding(
                document, path, raw_root, artifact_by_root, expected_by_id,
                expected_deliveries, run,
            )
            by_case.setdefault(case_id, []).append((path, document))
        missing = set(expected_by_id) - set(by_case)
        if missing:
            raise PublisherError(f"artifact is missing {len(missing)} requested case outcomes")
        attempt_records: list[dict[str, Any]] = []
        selections: list[dict[str, Any]] = []
        selected_documents: list[dict[str, Any]] = []
        runtime_hashes: set[str] = set()
        outcome_counts = {name: 0 for name in OUTCOMES}
        for case_id in sorted(expected_by_id):
            case_attempts = by_case[case_id]
            ordinals = [document["identity"]["attempt_ordinal"] for _, document in case_attempts]
            allocations_for_case = {
                document["identity"]["allocation_id"] for _, document in case_attempts
            }
            if len(allocations_for_case) != 1 or sorted(ordinals) != list(
                range(1, len(ordinals) + 1)
            ):
                raise PublisherError(
                    "case retries must retain contiguous ordinals in one allocation"
                )
            _, selected_document = max(
                case_attempts, key=lambda item: item[1]["identity"]["attempt_ordinal"]
            )
            selected_id = selected_document["identity"]["attempt_id"]
            selected_documents.append(selected_document)
            selected_status, _ = _outcome(selected_document)
            selections.append({
                "case_id": case_id,
                "selected_attempt_id": selected_id,
                "outcome": selected_status,
            })
            outcome_counts[selected_status] += 1
            for path, document in sorted(
                case_attempts, key=lambda item: item[1]["identity"]["attempt_ordinal"]
            ):
                normalized = contracts.normalize_attempt(document)
                if document["format"] == contracts.RAW_FORMAT:
                    sample_path = path.with_name(document["sample_artifact"]["path"])
                    if sample_path not in consumed_samples:
                        raise PublisherError("validated raw attempt lost its samples document")
                record = _attempt_record(
                    document, path, stage,
                    selected=normalized["attempt_id"] == selected_id,
                )
                if record["runtime_fingerprint_sha256"]:
                    runtime_hashes.add(record["runtime_fingerprint_sha256"])
                attempt_records.append(record)
        # Every extracted byte is covered; the bundle manifest anchors this checksum catalog.
        payload_records = [_file_metadata(path, stage) for path in _tree_files(stage)]
        checksum_document = {
            "format": "collectivex.checksums.v1",
            "files": payload_records,
        }
        checksum_path = stage / "checksums.json"
        _write_json(checksum_path, checksum_document, mode=0o600)
        bundle = {
            "format": FORMAT_BUNDLE,
            "schema_version": 1,
            "created_at": created_at,
            "ingest_id": incoming_id,
            "run": run,
            "matrix": _file_metadata(matrix_path, stage),
            "sources": source_records,
            "attempts": attempt_records,
            "coverage": {
                "expected_cases": len(expected_cases),
                "terminal_cases": len(selections),
                "complete": len(selections) == len(expected_cases),
                "outcome_counts": outcome_counts,
                "selections": selections,
            },
            "runtime_fingerprints": sorted(runtime_hashes),
            "checksums": _file_metadata(checksum_path, stage),
            "validation": {
                "policy": PUBLISHER_POLICY,
                "passed": True,
                "checks": [
                    "archive-safety", "checksums", "exact-coverage", "identity",
                    "native-schema", "privacy", "runtime-homogeneity", "terminal-outcomes",
                ],
            },
        }
        validate_bundle_manifest(bundle)
        # Runtime homogeneity is scoped to a realized allocation, not across unlike SKUs.
        by_allocation: dict[str, set[str]] = {}
        for attempt in attempt_records:
            fingerprint = attempt["runtime_fingerprint_sha256"]
            if fingerprint:
                by_allocation.setdefault(attempt["allocation_id"], set()).add(fingerprint)
        if any(len(values) != 1 for values in by_allocation.values()):
            raise PublisherError("runtime fingerprint is heterogeneous within an allocation")
        bundle_bytes = _canonical(bundle) + b"\n"
        bundle_id = _sha_bytes(bundle_bytes)
        _write_bytes(stage / "bundle.json", bundle_bytes, mode=0o600)
        store.complete(stage, bundle_id, private=True)
        store.install(stage, store.bundles / bundle_id, private=True)
    installed = load_bundle(store, bundle_id)
    if installed["manifest"] != bundle:
        raise PublisherError("existing bundle differs from validated manifest")
    return bundle_id, bundle, selected_documents


def _slug(value: Any, fallback: str = "unknown") -> str:
    text = re.sub(r"[^a-z0-9_.-]+", "-", str(value or "").lower()).strip("-.")
    return text[:128] if text and SAFE_ID.fullmatch(text[:128]) else fallback


def _derived_id(prefix: str, value: Any) -> str:
    return f"{prefix}{_sha_bytes(_canonical(value))}"


def _git_run(document: dict[str, Any]) -> dict[str, Any]:
    return _document_git_run(document) or {}


def _public_attempt(document: dict[str, Any], *, selected: bool = False) -> dict[str, Any]:
    normalized = contracts.normalize_attempt(document)
    run = _git_run(document)
    evidence = (
        [{"evidence_id": row["evidence_id"], "point_id": row["point_id"]}
         for row in document["measurement"]["rows"]]
        if document["format"] == contracts.RAW_FORMAT else []
    )
    status, reason = _outcome(document)
    failure_mode = document["outcome"].get("failure_mode")
    if not isinstance(failure_mode, str) or REASON.fullmatch(failure_mode) is None:
        failure_mode = None if status == "success" else reason
    series_id = normalized["series_id"] if status == "success" and selected else None
    return {
        "attempt_id": normalized["attempt_id"],
        "evidence": evidence,
        "case_id": normalized["case_id"],
        "allocation_id": normalized["allocation_id"],
        "run_id": str(run["run_id"]),
        "run_attempt": int(run["run_attempt"]),
        "qualification_index": int(run["qualification_index"]),
        "attempt_index": document["identity"]["attempt_ordinal"],
        "selected": selected,
        "outcome": status,
        "failure_mode": failure_mode,
        "reason": reason,
        "series_id": series_id,
        "completed_at": document["generated_at"],
    }


def _ratio(values: Sequence[float]) -> float | None:
    return max(values) / min(values) if len(values) >= REQUIRED_ALLOCATIONS and min(values) > 0 else None


def _private_trial_components(sample_document: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Copy validated trial blocks into publisher-private memory without fixing component names."""
    points: dict[int, dict[str, Any]] = {}
    for point in sample_document["points"]:
        token = point["tokens_per_rank"]
        components: dict[str, Any] = {}
        for name, component in point["components"].items():
            availability = component["availability"]
            if availability in {"unavailable", "not-applicable"}:
                components[name] = None
                continue
            if availability != "measured":
                raise PublisherError(f"private sample component {name} has invalid availability")
            trials = component["trials"]
            if (
                not isinstance(trials, list)
                or len(trials) != 64
                or any(not isinstance(trial, list) or len(trial) != 8 for trial in trials)
            ):
                raise PublisherError(f"private sample component {name} is not 64x8")
            copied = tuple(
                tuple(float(sample) for sample in trial)
                for trial in trials
            )
            if any(
                not math.isfinite(sample) or sample < 0
                for trial in copied for sample in trial
            ):
                raise PublisherError(f"private sample component {name} is not finite")
            components[name] = copied
        points[token] = components
    return points


def _trial_diagnostics(
    trial_blocks: dict[str, dict[int, dict[str, Any]]], token: int,
) -> dict[str, Any]:
    components: dict[str, Any] = {}
    reasons: set[str] = set()
    for name in ("dispatch", "stage", "combine", "roundtrip"):
        values = [trial_blocks[run_id][token][name] for run_id in sorted(trial_blocks)]
        if all(value is None for value in values):
            components[name] = None
            continue
        if any(value is None for value in values):
            raise PublisherError(f"{name} trial availability differs across qualification runs")
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (REQUIRED_ALLOCATIONS, 64, 8) or not np.isfinite(array).all():
            raise PublisherError(f"{name} trial diagnostics require three finite 64x8 runs")
        medians = np.median(array, axis=2)
        first = np.median(medians[:, :8], axis=1)
        last = np.median(medians[:, -8:], axis=1)
        if np.any(first <= 0) or np.any(last <= 0):
            raise PublisherError(f"{name} trial diagnostics require positive latency")
        drift_ratio = float(np.max(np.maximum(first / last, last / first)))
        center = float(np.median(medians))
        mad = float(np.median(np.abs(medians - center)))
        if mad == 0:
            outliers = np.abs(medians - center) > 0
        else:
            outliers = np.abs(medians - center) > (
                TRIAL_OUTLIER_MAD_MULTIPLIER * 1.4826 * mad
            )
        outlier_fraction = float(np.count_nonzero(outliers) / medians.size)
        drift_flagged = drift_ratio > TRIAL_DRIFT_RATIO_LIMIT
        outlier_flagged = outlier_fraction > TRIAL_OUTLIER_FRACTION_LIMIT
        if drift_flagged:
            reasons.add("trial-drift")
        if outlier_flagged:
            reasons.add("trial-outliers")
        components[name] = {
            "drift_flagged": drift_flagged,
            "first_last_median_ratio": drift_ratio,
            "outlier_flagged": outlier_flagged,
            "robust_outlier_fraction": outlier_fraction,
            "trial_count": int(medians.size),
        }
    return {
        "flagged": bool(reasons),
        "reasons": sorted(reasons),
        "components": components,
    }


def _nearest_rank_p99(blocks: Sequence[Sequence[float]]) -> float:
    samples = sorted(float(sample) for block in blocks for sample in block)
    if len(samples) != 512 or samples[0] < 0 or not all(map(math.isfinite, samples)):
        raise PublisherError("p99 bootstrap input must contain 512 finite samples")
    return samples[math.ceil(0.99 * len(samples)) - 1]


def _roundtrip_trial_array(
    internal: dict[str, Any], token: int
) -> tuple[tuple[str, ...], np.ndarray]:
    trial_blocks = internal.get("trial_blocks")
    if not isinstance(trial_blocks, dict):
        raise PublisherError("series is missing private trial blocks")
    run_ids = tuple(sorted(trial_blocks, key=lambda value: (int(value), value)))
    if len(run_ids) != REQUIRED_ALLOCATIONS:
        raise PublisherError("p99 bootstrap requires exactly three run blocks")
    values = []
    for run_id in run_ids:
        point = trial_blocks[run_id].get(token)
        blocks = point.get("roundtrip") if isinstance(point, dict) else None
        if blocks is None:
            raise PublisherError("p99 bootstrap requires measured roundtrip blocks")
        if len(blocks) != 64 or any(len(block) != 8 for block in blocks):
            raise PublisherError("p99 bootstrap roundtrip blocks must be 64x8")
        values.append(blocks)
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (REQUIRED_ALLOCATIONS, 64, 8):
        raise PublisherError("p99 bootstrap trial array shape differs")
    if not np.isfinite(array).all() or np.any(array <= 0):
        raise PublisherError("p99 bootstrap latencies must be finite and positive")
    return run_ids, array


def _bootstrap_seed(
    dataset_binding: str, baseline_series_id: str, candidate_series_id: str, token: int
) -> tuple[str, int]:
    payload = _canonical({
        "policy": BOOTSTRAP_POLICY,
        "resamples": BOOTSTRAP_RESAMPLES,
        "confidence": BOOTSTRAP_CONFIDENCE,
        "equivalence_band": BOOTSTRAP_EQUIVALENCE_BAND,
        "dataset_binding": dataset_binding,
        "baseline_series_id": baseline_series_id,
        "candidate_series_id": candidate_series_id,
        "tokens_per_rank": token,
    })
    digest = hashlib.sha256(payload).digest()
    return digest.hex(), int.from_bytes(digest[:16], "big")


def _hierarchical_p99_ratio(
    baseline_series_id: str,
    candidate_series_id: str,
    token: int,
    internals: dict[str, dict[str, Any]],
    dataset_binding: str,
) -> dict[str, Any]:
    """Bootstrap candidate/baseline p99 across runs, then 64 trial blocks."""
    baseline_runs, baseline = _roundtrip_trial_array(
        internals[baseline_series_id], token
    )
    candidate_runs, candidate = _roundtrip_trial_array(
        internals[candidate_series_id], token
    )
    if baseline_runs != candidate_runs:
        raise PublisherError("p99 bootstrap run blocks are not aligned")
    seed_sha256, seed = _bootstrap_seed(
        dataset_binding, baseline_series_id, candidate_series_id, token
    )
    cache_key = (
        seed_sha256,
        _sha_bytes(baseline.tobytes()),
        _sha_bytes(candidate.tobytes()),
    )
    cached = _BOOTSTRAP_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    baseline_run_p99 = np.asarray(
        [_nearest_rank_p99(run) for run in baseline], dtype=np.float64
    )
    candidate_run_p99 = np.asarray(
        [_nearest_rank_p99(run) for run in candidate], dtype=np.float64
    )
    run_ratios = candidate_run_p99 / baseline_run_p99
    point_ratio = float(np.median(candidate_run_p99) / np.median(baseline_run_p99))

    rng = np.random.Generator(np.random.PCG64(seed))
    ratios = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    p99_index = math.ceil(0.99 * 512) - 1
    for start in range(0, BOOTSTRAP_RESAMPLES, BOOTSTRAP_CHUNK_SIZE):
        size = min(BOOTSTRAP_CHUNK_SIZE, BOOTSTRAP_RESAMPLES - start)
        sampled_runs = rng.integers(0, REQUIRED_ALLOCATIONS, size=(size, 3))
        sampled_blocks = rng.integers(0, 64, size=(size, 3, 64))
        run_index = sampled_runs[:, :, None]
        baseline_sample = baseline[run_index, sampled_blocks].reshape(size, 3, 512)
        candidate_sample = candidate[run_index, sampled_blocks].reshape(size, 3, 512)
        baseline_p99 = np.partition(baseline_sample, p99_index, axis=2)[:, :, p99_index]
        candidate_p99 = np.partition(candidate_sample, p99_index, axis=2)[:, :, p99_index]
        ratios[start:start + size] = (
            np.median(candidate_p99, axis=1) / np.median(baseline_p99, axis=1)
        )
    ratios.sort()
    tail = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    lower_index = max(0, math.ceil(tail * BOOTSTRAP_RESAMPLES) - 1)
    upper_index = min(
        BOOTSTRAP_RESAMPLES - 1,
        math.ceil((1.0 - tail) * BOOTSTRAP_RESAMPLES) - 1,
    )
    ci = [float(ratios[lower_index]), float(ratios[upper_index])]
    threshold = 1.0 + BOOTSTRAP_EQUIVALENCE_BAND
    baseline_wins = ci[0] > threshold and bool(np.all(run_ratios > threshold))
    result = {
        "policy": BOOTSTRAP_POLICY,
        "resamples": BOOTSTRAP_RESAMPLES,
        "confidence": BOOTSTRAP_CONFIDENCE,
        "equivalence_band": BOOTSTRAP_EQUIVALENCE_BAND,
        "seed_sha256": seed_sha256,
        "point_ratio": point_ratio,
        "ci95": ci,
        "run_ratios": [float(value) for value in run_ratios],
        "all_runs_agree": bool(np.all(run_ratios > threshold)),
        "baseline_wins": baseline_wins,
        "tie": not baseline_wins,
    }
    _BOOTSTRAP_CACHE[cache_key] = result
    return dict(result)


def _bootstrap_inputs_ready(
    members: Sequence[dict[str, Any]],
    internals: dict[str, dict[str, Any]],
    tokens: Sequence[int],
) -> bool:
    try:
        expected_runs: tuple[str, ...] | None = None
        for member in members:
            for token in tokens:
                run_ids, _ = _roundtrip_trial_array(internals[member["series_id"]], token)
                if expected_runs is None:
                    expected_runs = run_ids
                elif run_ids != expected_runs:
                    return False
        return expected_runs is not None
    except (KeyError, PublisherError, TypeError, ValueError):
        return False


def _eligibility_record(
    allocations: Sequence[str],
    *,
    complete: bool,
    correct: bool,
    measured: bool,
    stable_ordering: bool,
    p50_ratio: float | None,
    p99_ratio: float | None,
    extra_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    ids = sorted(set(allocations))
    stable_p50 = p50_ratio is not None and p50_ratio <= P50_STABILITY_LIMIT
    stable_p99 = p99_ratio is not None and p99_ratio <= P99_STABILITY_LIMIT
    reasons = list(extra_reasons)
    for condition, reason in (
        (len(ids) >= REQUIRED_ALLOCATIONS, "insufficient-allocations"),
        (complete, "incomplete-repeat-coverage"),
        (correct, "correctness-failed"),
        (measured, "missing-measured-roundtrip-p99"),
        (stable_p50, "unstable-p50"),
        (stable_p99, "unstable-p99"),
        (stable_ordering, "unstable-ordering"),
    ):
        if not condition:
            reasons.append(reason)
    reasons = sorted(set(reasons))
    decision = not reasons
    return {
        "decision_grade": decision,
        "allocation_ids": ids,
        "complete": complete,
        "correct": correct,
        "measured_roundtrip_p99": measured,
        "stable_p50": stable_p50,
        "stable_p99": stable_p99,
        "stable_ordering": stable_ordering,
        "p50_max_min_ratio": p50_ratio,
        "p99_max_min_ratio": p99_ratio,
        "reasons": reasons,
    }


def _aggregate_percentiles(values: Sequence[dict[str, Any]]) -> dict[str, float]:
    return {
        name: float(statistics.median(float(value[name]) for value in values))
        for name in ("p50", "p90", "p95", "p99")
    }


def _aggregate_component(
    rows: Sequence[dict[str, Any]], name: str
) -> dict[str, Any] | None:
    components = [row["components"][name] for row in rows]
    if all(component["availability"] == "unavailable" for component in components):
        return None
    if any(component["availability"] == "unavailable" for component in components):
        raise PublisherError("component availability differs across repeat allocations")
    latency = _aggregate_percentiles([component["percentiles_us"] for component in components])
    if name == "isolated_sum":
        byte_provenance = {
            "accounting_contract": "activation-data-plus-scales-v1",
            "activation_data_bytes": 0,
            "scale_bytes": 0,
            "total_logical_bytes": 0,
        }
        return {
            "origin": "derived",
            "latency_us": latency,
            "byte_provenance": byte_provenance,
            "activation_data_rate_gbps_at_latency_percentile": None,
            "total_logical_data_rate_gbps_at_latency_percentile": None,
            "sample_count": None,
        }
    byte_provenance = _exact_repeat_value(
        [row["byte_provenance"][name] for row in rows],
        f"{name} byte accounting",
    )
    activation_rates = {
        statistic: byte_provenance["activation_data_bytes"] / (latency[statistic] * 1000.0)
        for statistic in latency
    }
    total_rates = {
        statistic: byte_provenance["total_logical_bytes"] / (latency[statistic] * 1000.0)
        for statistic in latency
    }
    return {
        "origin": "measured",
        "latency_us": latency,
        "byte_provenance": byte_provenance,
        "activation_data_rate_gbps_at_latency_percentile": activation_rates,
        "total_logical_data_rate_gbps_at_latency_percentile": total_rates,
        "sample_count": 512,
    }


def _exact_repeat_value(values: Sequence[Any], label: str) -> Any:
    if not values or len({_canonical(value) for value in values}) != 1:
        raise PublisherError(f"{label} differs across repeat allocations")
    return values[0]


def _eplb_descriptor(document: dict[str, Any]) -> dict[str, Any]:
    value = document["case"]["eplb"]
    return {
        "enabled": value["enabled"],
        "calibration_workload_id": value["calibration_workload_id"],
        "calibration_trace_sha256": value["calibration_trace_sha256"],
        "calibration_window": value["calibration_window"],
        "calibration_token_offset": value["calibration_token_offset"],
        "planner": value["planner"],
        "mapping_sha256": value["mapping_hash"],
        "logical_experts": value["num_logical_experts"],
        "physical_experts": value["num_physical_experts"],
        "redundant_experts": value["num_redundant"],
        "reference_tokens_per_rank": value["reference_tokens_per_rank"],
        "replicated_experts": value["replicated_experts"],
        "max_replicas": value["max_replicas"],
        "imbalance_before": value["imbalance_before"],
        "imbalance_after": value["imbalance_after"],
    }


def _routing_facts(row: dict[str, Any]) -> dict[str, Any]:
    routing = row["routing"]
    return {
        "fanout_mean": routing["fanout_mean"],
        "recv_tokens_max": row["receive"]["max"],
        "expert_load_cv": routing["expert_load_cv"],
        "payload_rank_cv": routing["payload_rank_cv"],
        "hotspot_ratio": routing["hotspot_ratio"],
        "empty_expert_count": routing["empty_expert_count"],
        "empty_rank_count": routing["empty_rank_count"],
        "routed_copies": routing["routed_copies"],
    }


def _aggregate_precision_evidence(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values = [row["correctness"]["precision"] for row in rows]
    profile_ids = {value["profile_id"] for value in values}
    if len(profile_ids) != 1:
        raise PublisherError("precision evidence profile differs across qualification runs")
    result: dict[str, Any] = {"profile_id": profile_ids.pop()}
    for direction in ("dispatch", "combine"):
        axes = [value[direction] for value in values]
        finite = [axis["scales_finite"] for axis in axes]
        positive = [axis["scales_positive"] for axis in axes]
        result[direction] = {
            "encoded_payload_valid": all(axis["encoded_payload_valid"] for axis in axes),
            "scales_finite": None if all(value is None for value in finite) else all(
                value is True for value in finite
            ),
            "scales_positive": None if all(value is None for value in positive) else all(
                value is True for value in positive
            ),
            "dequantized_semantics": all(axis["dequantized_semantics"] for axis in axes),
            "saturation_count": max(axis["saturation_count"] for axis in axes),
            "saturation_rate": max(axis["saturation_rate"] for axis in axes),
            "max_abs_error": max(axis["max_abs_error"] for axis in axes),
            "max_rel_error": max(axis["max_rel_error"] for axis in axes),
            "passed": all(axis["passed"] for axis in axes),
        }
    result["passed"] = result["dispatch"]["passed"] and result["combine"]["passed"]
    return result


def _series_extra_reasons(documents: Sequence[dict[str, Any]]) -> list[str]:
    reasons: set[str] = set()
    for document in documents:
        validity = document["outcome"]["validity"]
        rows = document["measurement"]["rows"]
        if validity.get("provenance_complete") is not True:
            reasons.add("incomplete-provenance")
        if validity.get("workload_source") != "canonical-serialized":
            reasons.add("noncanonical-workload")
        if validity.get("anomaly_free") is not True or any(row["anomalies"] for row in rows):
            reasons.add("unresolved-anomaly")
        if validity.get("semantic_correctness") != "pass":
            reasons.add("semantic-correctness-failed")
        if validity.get("measurement_conformance") != "conformant" or validity.get("sampling_conformance") != "conformant":
            reasons.add("measurement-nonconformant")
        profile = identity.case_profile(document["case"]["mode"])
        scopes = {row["correctness"].get("scope") for row in rows}
        if scopes != {profile["correctness_scope"]}:
            reasons.add("expert-oracle-incomplete")
    return sorted(reasons)


BACKEND_LABELS = {
    "deepep": "DeepEP V1",
    "deepep-v2": "DeepEP V2",
    "deepep-hybrid": "DeepEP Hybrid",
    "uccl": "UCCL",
    "mori": "MoRI",
    "nccl-ep": "NCCL/RCCL reference",
}


def _build_series(
    series_id: str,
    documents: Sequence[dict[str, Any]],
    sample_documents: Sequence[dict[str, Any]],
    expected_repeats: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not documents:
        raise PublisherError("cannot aggregate an empty series")
    first = documents[0]
    if any(document["identity"]["series_id"] != series_id for document in documents):
        raise PublisherError("series aggregation mixed identities")
    if len(sample_documents) != len(documents):
        raise PublisherError("series aggregation lost private sample documents")
    allocations = [document["identity"]["allocation_id"] for document in documents]
    if len(allocations) != len(set(allocations)):
        raise PublisherError("series repeats reuse an allocation identity")
    row_maps = [
        {row["tokens_per_rank"]: row for row in document["measurement"]["rows"]}
        for document in documents
    ]
    token_sets = {tuple(sorted(rows)) for rows in row_maps}
    if len(token_sets) != 1:
        raise PublisherError("series token coverage differs across allocations")
    tokens = list(next(iter(token_sets)))
    qualification_indices = sorted(
        document["measurement"]["qualification_index"] for document in documents
    )
    p50_ratios = [
        _ratio([rows[token]["components"]["roundtrip"]["percentiles_us"]["p50"] for rows in row_maps])
        for token in tokens
    ]
    p99_ratios = [
        _ratio([rows[token]["components"]["roundtrip"]["percentiles_us"]["p99"] for rows in row_maps])
        for token in tokens
    ]
    p50_ratio = max((value for value in p50_ratios if value is not None), default=None)
    p99_ratio = max((value for value in p99_ratios if value is not None), default=None)
    correct = all(
        row["correctness"]["passed"]
        for document in documents for row in document["measurement"]["rows"]
    )
    measured = all(
        row["components"]["roundtrip"]["availability"] == "measured"
        and row["components"]["roundtrip"]["percentiles_us"].get("p99") is not None
        for document in documents for row in document["measurement"]["rows"]
    )
    extra_reasons = _series_extra_reasons(documents)
    case = first["case"]
    shape = case["shape"]
    topology = first["topology"]
    runtime = first["runtime_fingerprint"]
    workload_id = first["workload"]["workload_id"]
    if not identity.is_typed_id(workload_id, "workload"):
        raise PublisherError("raw workload is not canonical")
    backend_id = case["backend"]
    resource_raw = first["implementation"]["resource_profile"]
    public_config = contracts.public_series_config(
        kernel_generation=first["implementation"]["kernel_generation"],
        provenance=first["implementation"]["provenance"],
        resource_profile=resource_raw,
        resource_mode=case["resource_mode"],
        device_product=topology["device_product"],
    )
    resource_profile = public_config["resource"]["profile"]
    configured_units = public_config["resource"]["configured_units"]
    units_kind = public_config["resource"]["comm_units_kind"]
    resource_label = (
        f"{configured_units} {str(units_kind).upper()}"
        if configured_units is not None and units_kind
        else resource_profile
    )
    eplb = _exact_repeat_value(
        [_eplb_descriptor(document) for document in documents], "EPLB descriptor"
    )
    points: list[dict[str, Any]] = []
    run_metrics: dict[str, dict[int, dict[str, float]]] = {}
    trial_blocks: dict[str, dict[int, dict[str, Any]]] = {}
    for document, sample_document, rows in zip(
        documents, sample_documents, row_maps, strict=True
    ):
        if any(
            sample_document[field] != document["identity"][field]
            for field in ("allocation_id", "attempt_id", "case_id", "series_id")
        ):
            raise PublisherError("private samples differ from their selected raw attempt")
        if sample_document["qualification_index"] != document["measurement"]["qualification_index"]:
            raise PublisherError("private sample qualification index differs from raw attempt")
        run_id = str(_git_run(document)["run_id"])
        if run_id in run_metrics:
            raise PublisherError("series has two allocations from one workflow run")
        trial_blocks[run_id] = _private_trial_components(sample_document)
        run_metrics[run_id] = {}
        for token in tokens:
            latency = rows[token]["components"]["roundtrip"]["percentiles_us"]
            byte_provenance = rows[token]["byte_provenance"]["roundtrip"]
            run_metrics[run_id][token] = {
                "latency_us": {statistic: latency[statistic] for statistic in ("p50", "p99")},
                "activation_data_rate_gbps_at_latency_percentile": {
                    statistic: byte_provenance["activation_data_bytes"]
                    / (latency[statistic] * 1000.0)
                    for statistic in ("p50", "p99")
                },
                "total_logical_data_rate_gbps_at_latency_percentile": {
                    statistic: byte_provenance["total_logical_bytes"]
                    / (latency[statistic] * 1000.0)
                    for statistic in ("p50", "p99")
                },
            }
    for token in tokens:
        rows = [row_map[token] for row_map in row_maps]
        diagnostics = _trial_diagnostics(trial_blocks, token)
        if diagnostics["flagged"]:
            extra_reasons.append("unresolved-trial-diagnostic")
        routing = _exact_repeat_value(
            [_routing_facts(row) for row in rows], "routing/load facts"
        )
        components = {
            name: _aggregate_component(rows, name)
            for name in ("dispatch", "stage", "combine", "roundtrip")
        }
        if components["dispatch"] is None:
            components["isolated_sum"] = None
        else:
            latency = {
                statistic: components["dispatch"]["latency_us"][statistic]
                + (
                    components["stage"]["latency_us"][statistic]
                    if components["stage"] is not None else 0.0
                )
                + components["combine"]["latency_us"][statistic]
                for statistic in ("p50", "p90", "p95", "p99")
            }
            components["isolated_sum"] = {
                "origin": "derived",
                "latency_us": latency,
                "byte_provenance": components["roundtrip"]["byte_provenance"],
                "activation_data_rate_gbps_at_latency_percentile": None,
                "total_logical_data_rate_gbps_at_latency_percentile": None,
                "sample_count": None,
            }
        points.append({
            "point_id": rows[0]["point_id"],
            "tokens_per_rank": token,
            "global_tokens": token * case["ep_size"],
            "correctness": {
                "semantic_pass": all(row["correctness"]["passed"] for row in rows),
                "precision": _aggregate_precision_evidence(rows),
            },
            "anomalies": sorted({
                anomaly["type"].replace("_", "-")
                for row in rows for anomaly in row["anomalies"]
            } | set(diagnostics["reasons"])),
            "stability": {
                "complete": qualification_indices == [1, 2, 3],
                "qualification_indices": qualification_indices,
                "p50_max_min_ratio": p50_ratios[tokens.index(token)]
                if qualification_indices == [1, 2, 3] else None,
                "p99_max_min_ratio": p99_ratios[tokens.index(token)]
                if qualification_indices == [1, 2, 3] else None,
                "stable_p50": bool(
                    qualification_indices == [1, 2, 3]
                    and p50_ratios[tokens.index(token)] is not None
                    and p50_ratios[tokens.index(token)] <= P50_STABILITY_LIMIT
                ),
                "stable_p99": bool(
                    qualification_indices == [1, 2, 3]
                    and p99_ratios[tokens.index(token)] is not None
                    and p99_ratios[tokens.index(token)] <= P99_STABILITY_LIMIT
                ),
            },
            "trial_diagnostics": diagnostics,
            "routing": routing,
            "components": components,
            "roundtrip_token_rate_at_latency_percentile": {
                statistic: (token * case["ep_size"])
                / (components["roundtrip"]["latency_us"][statistic] * 1e-6)
                for statistic in ("p50", "p90", "p95", "p99")
            },
            "evidence_ids": [row["evidence_id"] for row in rows],
        })
    eligibility = _eligibility_record(
        allocations,
        complete=len(documents) == expected_repeats,
        correct=correct,
        measured=measured,
        # Ordering is defined only across alternatives in a controlled cohort.
        stable_ordering=True,
        p50_ratio=p50_ratio,
        p99_ratio=p99_ratio,
        extra_reasons=sorted(set(extra_reasons)),
    )
    series = {
        "series_id": series_id,
        "label": (
            f"{case['runner'].upper()} / {BACKEND_LABELS.get(backend_id, backend_id)} / "
            f"EP{case['ep_size']} / {topology['nodes']} node"
            f"{'s' if topology['nodes'] != 1 else ''} / {topology['scope']} / "
            f"{case['mode']} / {case['phase']} / {shape['routing']}"
            f"{' + EPLB' if case['eplb']['enabled'] else ''} / {resource_label}"
        ),
        "status": "decision-grade" if eligibility["decision_grade"] else "diagnostic",
        "case_ids": sorted({document["identity"]["case_id"] for document in documents}),
        "allocation_ids": sorted(allocations),
        "model": _slug(case["workload_name"]),
        "suite": _slug(case["suite"]),
        "mode": case["mode"],
        "phase": case["phase"],
        "publication_tier": case["required_publication"],
        "backend": {
            "id": _slug(backend_id),
            "label": BACKEND_LABELS.get(backend_id, backend_id),
            "role": "reference" if backend_id == "nccl-ep" else "library",
            **public_config["backend"],
        },
        "build": {
            "implementation_contract_sha256": first["identity"]["series_factors"][
                "implementation_contract_sha256"
            ],
            "public_config_sha256": first["identity"]["series_factors"][
                "public_config_sha256"
            ],
            "routing_control_sha256": first["identity"]["series_factors"][
                "routing_control_sha256"
            ],
            "runtime_fingerprint_sha256": first["identity"]["series_factors"][
                "runtime_fingerprint_sha256"
            ],
            "image_digest": first["identity"]["series_factors"]["image_digest"],
            "source_sha": first["identity"]["series_factors"]["source_sha"],
            "squash_sha256": first["identity"]["series_factors"]["squash_sha256"],
        },
        "system": {
            "sku": _slug(case["runner"]),
            "label": public_config["system"]["label"],
            "vendor": runtime["vendor"],
            "topology_class": _slug(topology["topology_class"]),
            "transport": _slug(topology["transport"]),
            "scale_up_transport": _slug(topology["scale_up_transport"]),
            "scale_out_transport": (
                _slug(topology["scale_out_transport"])
                if topology["scale_out_transport"] is not None
                else None
            ),
            "scope": topology["scope"],
            "nodes": topology["nodes"],
            "gpus_per_node": topology["gpus_per_node"],
            "scale_up_domain": topology["scale_up_domain"],
            "world_size": topology["world_size"],
            "ep_size": case["ep_size"],
            "placement": topology["placement"],
        },
        "workload": {
            "workload_id": workload_id,
            "hidden": shape["hidden"],
            "top_k": shape["topk"],
            "experts": case["eplb"]["num_logical_experts"],
            "routing": shape["routing"],
            "eplb": case["eplb"]["enabled"],
            "precision_profile": shape["precision_profile"],
            "dispatch_precision": shape["dispatch_precision"],
            "combine_precision": shape["combine_precision"],
            "activation_profile": shape["activation_profile"],
        },
        "eplb": eplb,
        "resource": public_config["resource"],
        "measurement": {
            "contract": first["measurement"]["contract"],
            "component_order_contract": first["measurement"]["component_order_contract"],
            "combine_semantics": identity.case_profile(case["mode"])["combine_semantics"],
            "payload_unit": identity.case_profile(case["mode"])["payload_unit"],
            "sampling_contract": first["measurement"]["sampling"]["contract"],
            "iters": first["measurement"]["sampling"]["iterations_per_trial"],
            "trials": first["measurement"]["sampling"]["trials"],
            "warmups": first["measurement"]["sampling"]["warmup_iterations"],
            "samples_per_component": first["measurement"]["sampling"]["samples_per_component"],
            "qualification_indices": qualification_indices,
            "headline_component": "roundtrip",
            "headline_percentile": "p99",
        },
        "points": points,
        "eligibility": eligibility,
    }
    internal = {
        "documents": list(documents),
        "run_metrics": run_metrics,
        "trial_blocks": trial_blocks,
        "series_factors": first["identity"]["series_factors"],
    }
    return series, internal


def _resolve_bundle_file(root: Path, record: dict[str, Any]) -> Path:
    path = root.joinpath(*PurePosixPath(record["path"]).parts)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PublisherError("bundle record escapes its directory") from exc
    if path.resolve() != path or path.is_symlink() or not path.is_file():
        raise PublisherError("bundle record points to a missing or linked file")
    if path.stat().st_size != record["bytes"] or _sha_file(path) != record["sha256"]:
        raise PublisherError("bundle file checksum differs from its manifest")
    return path


def load_bundle(store: Store, bundle_id: str) -> dict[str, Any]:
    if HEX64.fullmatch(bundle_id) is None:
        raise PublisherError("bundle ID must be a SHA-256 digest")
    root = store.bundles / bundle_id
    if root.is_symlink() or not (root / "COMPLETE").is_file():
        raise PublisherError(f"bundle {bundle_id} is missing or incomplete")
    _verify_frozen_tree(root, private=True)
    if (root / "COMPLETE").read_text().strip() != bundle_id:
        raise PublisherError("bundle COMPLETE marker differs")
    manifest_path = root / "bundle.json"
    if _sha_file(manifest_path) != bundle_id:
        raise PublisherError("bundle directory digest differs from bundle.json")
    manifest = validate_bundle_manifest(strict_load(manifest_path))
    checksum_path = _resolve_bundle_file(root, manifest["checksums"])
    checksum_document = strict_load(checksum_path)
    checksum_document = _exact(checksum_document, {"format", "files"}, "checksums")
    if checksum_document["format"] != "collectivex.checksums.v1":
        raise PublisherError("bundle checksum format is invalid")
    records = [_file_record(value, f"checksums.files[{index}]")
               for index, value in enumerate(_array(checksum_document["files"], "checksums.files"))]
    _unique([record["path"] for record in records], "checksums.files[].path")
    for record in records:
        _resolve_bundle_file(root, record)
    expected_paths = {
        path.relative_to(root).as_posix() for path in _tree_files(root)
        if path.name not in {"bundle.json", "checksums.json"}
    }
    if {record["path"] for record in records} != expected_paths:
        raise PublisherError("bundle checksum catalog does not cover its payload exactly")
    artifact_by_root: dict[str, str] = {}
    for index, source in enumerate(manifest["sources"]):
        _resolve_bundle_file(root, source)
        archive_key = f"artifact-{index:04d}"
        if source["path"] != f"source/{archive_key}.zip":
            raise PublisherError("bundle source catalog order/path differs")
        artifact_by_root[archive_key] = source["artifact_name"]
    if len(set(artifact_by_root.values())) != len(artifact_by_root):
        raise PublisherError("bundle source catalog repeats an artifact name")
    matrix_path = _resolve_bundle_file(root, manifest["matrix"])
    matrix_document = strict_load(matrix_path)
    cases = validate_matrix(matrix_document)
    expected_by_id = {case["case_id"]: case for case in cases}
    expected_deliveries = _expected_deliveries(
        matrix_document, cases, manifest["run"]
    )
    if {item["case_id"] for item in manifest["coverage"]["selections"]} != set(expected_by_id):
        raise PublisherError("bundle selected coverage differs from requested matrix")
    documents: dict[str, dict[str, Any]] = {}
    sample_documents: dict[str, dict[str, Any]] = {}
    runtime_fingerprints: set[str] = set()
    for attempt in manifest["attempts"]:
        document_path = _resolve_bundle_file(root, attempt["document"])
        document = contracts.strict_load(document_path)
        artifact_safety.assert_publication_safe([document])
        if document.get("format") == contracts.RAW_FORMAT:
            _schema("raw-case-v1.schema.json", document)
            sample_path = document_path.with_name(document["sample_artifact"]["path"])
            if attempt["samples"] is None:
                raise PublisherError("raw attempt is missing its sample manifest record")
            manifest_sample_path = _resolve_bundle_file(root, attempt["samples"])
            if manifest_sample_path != sample_path:
                raise PublisherError("sample manifest record points to the wrong raw evidence")
            sample_document = contracts.strict_load(sample_path)
            artifact_safety.assert_publication_safe([sample_document])
            _schema("samples-v1.schema.json", sample_document)
            sample_document = contracts.validate_samples_document(sample_document)
            document = contracts.load_raw_attempt(document_path)
            sample_documents[attempt["attempt_id"]] = sample_document
        else:
            if attempt["samples"] is not None:
                raise PublisherError("terminal attempt unexpectedly names a sample artifact")
            _schema("terminal-outcome-v1.schema.json", document)
            document = contracts.validate_terminal_document(document)
        _validate_delivery_binding(
            document, document_path, root / "raw", artifact_by_root,
            expected_by_id, expected_deliveries, manifest["run"],
        )
        expected_record = _attempt_record(
            document, document_path, root, selected=attempt["selected"]
        )
        if expected_record != attempt:
            raise PublisherError("bundle attempt record differs from native document")
        if attempt["runtime_fingerprint_sha256"]:
            runtime_fingerprints.add(attempt["runtime_fingerprint_sha256"])
        documents[attempt["attempt_id"]] = document
    if sorted(runtime_fingerprints) != manifest["runtime_fingerprints"]:
        raise PublisherError("bundle runtime fingerprint catalog differs from attempts")
    selected = {
        selection["case_id"]: documents[selection["selected_attempt_id"]]
        for selection in manifest["coverage"]["selections"]
    }
    return {
        "id": bundle_id,
        "root": root,
        "manifest": manifest,
        "cases": cases,
        "documents": documents,
        "sample_documents": sample_documents,
        "selected": selected,
    }


def _cohort_control(
    kind: str, series: dict[str, Any], internal: dict[str, Any]
) -> tuple[dict[str, Any], list[str], list[str], Any]:
    binary_build = series["build"]
    source = binary_build["source_sha"]
    workload = series["workload"]
    shape = {
        key: workload[key]
        for key in (
            "hidden", "top_k", "experts", "precision_profile", "dispatch_precision",
            "combine_precision", "activation_profile",
        )
    }
    common = {
        "model": series["model"], "mode": series["mode"],
        "phase": series["phase"], "shape": shape,
        "measurement": series["measurement"], "ep_size": series["system"]["ep_size"],
    }
    if kind == "library":
        control = {**common, "system": series["system"], "workload": workload,
                   "resource_mode": series["resource"]["mode"], "source": source}
        return control, ["system", "workload", "mode", "phase", "measurement", "resource.mode", "source"], ["backend", "resource"], series["backend"]["id"]
    if kind == "chip":
        control = {**common, "backend": series["backend"], "source": source,
                   "workload": workload, "resource_mode": series["resource"]["mode"]}
        return control, ["backend", "source", "workload", "mode", "phase", "measurement", "resource.mode"], ["system", "resource"], series["system"]
    if kind == "system":
        control = {**common, "workload": workload, "source": source}
        varying = [series["system"]["sku"], series["backend"]["id"], series["resource"]["profile"]]
        return control, ["workload", "mode", "phase", "measurement", "source"], ["system", "backend", "resource"], varying
    if kind in PRECISION_COHORT_KINDS:
        control, variant = _public_cohort_factors(kind, series)
        if kind == "dispatch-precision":
            controlled = [
                "backend", "implementation-static-build", "system", "model-shape",
                "mode", "phase", "workload.routing", "workload.eplb", "measurement",
                "resource", "combine-precision",
            ]
            varying = ["dispatch-precision"]
        elif kind == "combine-precision":
            controlled = [
                "backend", "implementation-static-build", "system", "model-shape",
                "mode", "phase", "workload.routing", "workload.eplb", "measurement",
                "resource", "dispatch-precision",
            ]
            varying = ["combine-precision"]
        else:
            controlled = [
                "backend", "implementation-static-build", "system", "model-shape",
                "mode", "phase", "workload.routing", "workload.eplb", "measurement",
            ]
            varying = [
                "dispatch-precision", "combine-precision", "precision-profile", "resource",
            ]
        return control, controlled, varying, variant
    raise PublisherError(f"unknown cohort kind {kind}")


def _cohort_ordering(
    members: Sequence[dict[str, Any]], internals: dict[str, dict[str, Any]], tokens: Sequence[int]
) -> tuple[bool, int]:
    run_ids = set.intersection(*(
        set(internals[member["series_id"]]["run_metrics"]) for member in members
    ))
    if len(run_ids) < REQUIRED_ALLOCATIONS:
        return False, len(run_ids)
    orders: list[tuple[str, str, int, str, tuple[str, ...]]] = []
    for run_id in sorted(run_ids):
        for token in tokens:
            for measure in (
                "latency_us", "activation_data_rate_gbps_at_latency_percentile",
                "total_logical_data_rate_gbps_at_latency_percentile",
            ):
                for statistic in ("p50", "p99"):
                    ordered = tuple(
                        member["series_id"]
                        for member in sorted(
                            members,
                            key=lambda item: (
                                internals[item["series_id"]]["run_metrics"][run_id][token][measure][statistic],
                                item["series_id"],
                            ),
                            reverse=measure != "latency_us",
                        )
                    )
                    orders.append((measure, statistic, token, run_id, ordered))
    for token in tokens:
        for measure in (
            "latency_us", "activation_data_rate_gbps_at_latency_percentile",
            "total_logical_data_rate_gbps_at_latency_percentile",
        ):
            for statistic in ("p50", "p99"):
                observed = {
                    entry[4]
                    for entry in orders
                    if entry[0] == measure and entry[1] == statistic and entry[2] == token
                }
                if len(observed) != 1:
                    return False, len(run_ids)
    return True, len(run_ids)


def _p99_top_tie_ids(
    members: Sequence[dict[str, Any]],
    internals: dict[str, dict[str, Any]],
    token: int,
    dataset_binding: str,
    cohort_id: str,
) -> set[str]:
    metric = {
        "operation": "roundtrip",
        "statistic": "p99",
        "measure": "latency_us",
        "objective": "min",
        "tokens_per_rank": token,
        "phase": members[0]["phase"],
    }
    ordered = sorted(
        members,
        key=lambda member: (
            _metric_value(member, metric)[1], member["series_id"]
        ),
    )
    baseline_id = ordered[0]["series_id"]
    comparisons: dict[str, dict[str, Any]] = {}
    tie_ids = {baseline_id}
    for candidate in ordered[1:]:
        candidate_id = candidate["series_id"]
        result = _hierarchical_p99_ratio(
            baseline_id, candidate_id, token, internals, dataset_binding
        )
        comparisons[candidate_id] = result
        if not result["baseline_wins"]:
            tie_ids.add(candidate_id)
    internals[baseline_id].setdefault("decision_statistics", {})[
        f"{cohort_id}:p99:{token}"
    ] = {
        "baseline_series_id": baseline_id,
        "comparisons": comparisons,
        "tie_series_ids": sorted(tie_ids),
    }
    return tie_ids


def build_decisions(
    series: Sequence[dict[str, Any]],
    internals: dict[str, dict[str, Any]],
    *,
    dataset_binding: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if dataset_binding is None:
        dataset_binding = _sha_bytes(_canonical({
            "series_ids": sorted(item["series_id"] for item in series),
        }))
    cohorts: list[dict[str, Any]] = []
    for kind in (*REQUIRED_COHORT_KINDS, *PRECISION_COHORT_KINDS):
        groups: dict[bytes, list[tuple[dict[str, Any], Any, list[str], list[str]]]] = {}
        for item in series:
            if kind == "library" and item["backend"]["role"] != "library":
                continue
            if kind == "system" and item["backend"]["role"] != "reference":
                continue
            control, controlled, varying, variant = _cohort_control(kind, item, internals[item["series_id"]])
            groups.setdefault(_canonical(control), []).append((item, variant, controlled, varying))
        for entries in groups.values():
            variants = {_canonical(entry[1]) for entry in entries}
            if len(entries) < 2 or len(variants) < 2:
                continue
            members = sorted((entry[0] for entry in entries), key=lambda item: item["series_id"])
            token_sets = [set(point["tokens_per_rank"] for point in member["points"]) for member in members]
            tokens = sorted(set.intersection(*token_sets))
            same_points = len({tuple(sorted(values)) for values in token_sets}) == 1
            ordering, aligned_runs = _cohort_ordering(members, internals, tokens) if tokens else (False, 0)
            allocations = sorted({value for member in members for value in member["allocation_ids"]})
            p50_ratio = max(
                (member["eligibility"]["p50_max_min_ratio"] for member in members
                 if member["eligibility"]["p50_max_min_ratio"] is not None), default=None
            )
            p99_ratio = max(
                (member["eligibility"]["p99_max_min_ratio"] for member in members
                 if member["eligibility"]["p99_max_min_ratio"] is not None), default=None
            )
            extra = {
                reason for member in members for reason in member["eligibility"]["reasons"]
                if reason not in {"unstable-ordering"}
            }
            if aligned_runs < REQUIRED_ALLOCATIONS:
                extra.add("incomplete-aligned-repeats")
            if tokens and not _bootstrap_inputs_ready(members, internals, tokens):
                extra.add("missing-trial-blocks")
            endpoint_contrast = kind in PRECISION_COHORT_KINDS
            if not tokens or (not endpoint_contrast and not same_points):
                extra.add("unmatched-token-coverage")
            if kind in {"dispatch-precision", "combine-precision"}:
                axis = "dispatch" if kind == "dispatch-precision" else "combine"
                field = f"{axis}_precision"
                bf16 = identity.precision_profile(
                    identity.V1_CONTROL_PRECISION_PROFILE
                )[axis]
                if sum(
                    _canonical(member["workload"][field]) == _canonical(bf16)
                    for member in members
                ) != 1:
                    extra.add("missing-bf16-precision-baseline")
            eligibility = _eligibility_record(
                allocations,
                complete=all(member["eligibility"]["complete"] for member in members)
                and bool(tokens) and (endpoint_contrast or same_points),
                correct=all(member["eligibility"]["correct"] for member in members),
                measured=all(member["eligibility"]["measured_roundtrip_p99"] for member in members),
                stable_ordering=ordering,
                p50_ratio=p50_ratio,
                p99_ratio=p99_ratio,
                extra_reasons=sorted(extra),
            )
            member_ids = [member["series_id"] for member in members]
            publication_tier = (
                "comparable-experimental"
                if any(member["publication_tier"] == "comparable-experimental" for member in members)
                else "official"
            )
            controlled, varying = entries[0][2], entries[0][3]
            cohort_id = _derived_id("cxcohort-v1-", {
                "kind": kind, "series_ids": member_ids,
                "controlled_factors": controlled, "varying_factors": varying,
            })
            kind_label = {
                "chip": "Platform",
                "dispatch-precision": "Dispatch precision",
                "combine-precision": "Combine precision",
                "precision-pair": "Precision profile",
            }.get(kind, kind.title())
            first = members[0]
            routing_label = first["workload"]["routing"] + (
                "+EPLB" if first["workload"]["eplb"] else ""
            )
            context = {
                "library": (
                    f"{first['system']['sku'].upper()} EP{first['system']['ep_size']} / "
                    f"{first['mode']} / {first['phase']} / {routing_label}"
                ),
                "chip": (
                    f"{first['backend']['label']} EP{first['system']['ep_size']} / "
                    f"{first['mode']} / {first['phase']} / {routing_label}"
                ),
                "system": (
                    f"Reference EP{first['system']['ep_size']} / {first['mode']} / "
                    f"{first['phase']} / {routing_label}"
                ),
                "dispatch-precision": (
                    f"{first['system']['sku'].upper()} / {first['backend']['label']} / "
                    f"EP{first['system']['ep_size']} / {first['mode']} / {first['phase']}"
                ),
                "combine-precision": (
                    f"{first['system']['sku'].upper()} / {first['backend']['label']} / "
                    f"EP{first['system']['ep_size']} / {first['mode']} / {first['phase']}"
                ),
                "precision-pair": (
                    f"{first['system']['sku'].upper()} / {first['backend']['label']} / "
                    f"EP{first['system']['ep_size']} / {first['mode']} / {first['phase']}"
                ),
            }[kind]
            cohorts.append({
                "cohort_id": cohort_id,
                "kind": kind,
                "label": f"{context} / {kind_label} contrast ({len(members)} series)",
                "description": (
                    "Publisher-controlled NCCL/RCCL system comparison"
                    if kind == "system"
                    else (
                        "Descriptive configured-stack precision comparison; no isolated axis claim"
                        if kind == "precision-pair"
                        else f"Publisher-controlled {kind_label.lower()} comparison"
                    )
                ),
                "series_ids": member_ids,
                "controlled_factors": controlled,
                "varying_factors": varying,
                "publication_tier": publication_tier,
                "eligibility": eligibility,
            })
    cohorts.sort(key=lambda item: item["cohort_id"])
    series_by_id = {item["series_id"]: item for item in series}
    rankings: list[dict[str, Any]] = []
    recommendations: list[dict[str, Any]] = []
    sensitivities: list[dict[str, Any]] = []
    for cohort in cohorts:
        if not cohort["eligibility"]["decision_grade"]:
            continue
        if cohort["kind"] == "precision-pair":
            continue
        members = [series_by_id[series_id] for series_id in cohort["series_ids"]]
        tokens = sorted(set.intersection(*(
            {point["tokens_per_rank"] for point in member["points"]} for member in members
        )))
        for token in tokens:
            p99_tie_ids = _p99_top_tie_ids(
                members, internals, token, dataset_binding, cohort["cohort_id"]
            )
            for measure, objective, unit in (
                ("latency_us", "min", "us"),
                ("activation_data_rate_gbps_at_latency_percentile", "max", "GB/s"),
                ("total_logical_data_rate_gbps_at_latency_percentile", "max", "GB/s"),
            ):
                for statistic in ("p50", "p99"):
                    metric = {
                        "operation": "roundtrip", "statistic": statistic,
                        "measure": measure, "objective": objective,
                        "tokens_per_rank": token, "phase": members[0]["phase"],
                    }
                    entries = []
                    for member in members:
                        point_id, value, observed_unit = _metric_value(member, metric)
                        if observed_unit != unit:
                            raise PublisherError("publisher metric unit differs")
                        entries.append({
                            "rank": 0, "series_id": member["series_id"], "point_id": point_id,
                            "value": value, "unit": unit,
                        })
                    entries.sort(key=lambda item: (item["value"], item["series_id"]), reverse=objective == "max")
                    for rank, entry in enumerate(entries, 1):
                        entry["rank"] = (
                            1
                            if measure == "latency_us"
                            and statistic == "p99"
                            and entry["series_id"] in p99_tie_ids
                            else rank
                        )
                    ranking_id = _derived_id("cxranking-v1-", {
                        "cohort_id": cohort["cohort_id"], "metric": metric,
                    })
                    metric_label = _metric_label(measure, statistic)
                    rankings.append({
                        "ranking_id": ranking_id, "cohort_id": cohort["cohort_id"],
                        "label": f"{cohort['kind'].title()} {metric_label} T={token}",
                        "metric": metric, "entries": entries,
                        "publication_tier": cohort["publication_tier"],
                        "eligibility": cohort["eligibility"],
                    })
                    if (
                        cohort["publication_tier"] != "official"
                        or measure != "latency_us"
                        or statistic != "p99"
                        or sum(entry["rank"] == 1 for entry in entries) != 1
                    ):
                        continue
                    objective_name = "min-p99-latency"
                    top = entries[0]
                    recommendation_id = _derived_id("cxrecommendation-v1-", {
                        "objective": objective_name, "ranking_id": ranking_id,
                    })
                    recommendations.append({
                        "recommendation_id": recommendation_id,
                        "cohort_id": cohort["cohort_id"],
                        "label": f"Best {metric_label} at T={token}",
                        "objective": objective_name,
                        "series_id": top["series_id"], "point_id": top["point_id"],
                        "value": top["value"], "unit": top["unit"],
                        "rationale": (
                            "Unique p99 winner after deterministic hierarchical bootstrap "
                            "and all-run agreement"
                        ),
                        "publication_tier": cohort["publication_tier"],
                        "eligibility": cohort["eligibility"],
                    })
        if cohort["kind"] in {"dispatch-precision", "combine-precision"}:
            axis = (
                "dispatch"
                if cohort["kind"] == "dispatch-precision"
                else "combine"
            )
            field = f"{axis}_precision"
            bf16 = identity.precision_profile(
                identity.V1_CONTROL_PRECISION_PROFILE
            )[axis]
            baseline = next(
                member for member in members
                if _canonical(member["workload"][field]) == _canonical(bf16)
            )
            for candidate in members:
                if candidate is baseline:
                    continue
                for token in tokens:
                    for measure, objective in (
                        ("latency_us", "min"),
                        ("activation_data_rate_gbps_at_latency_percentile", "max"),
                        ("total_logical_data_rate_gbps_at_latency_percentile", "max"),
                    ):
                        for statistic in ("p50", "p99"):
                            metric = {
                                "operation": "roundtrip",
                                "statistic": statistic,
                                "measure": measure,
                                "objective": objective,
                                "tokens_per_rank": token,
                                "phase": baseline["phase"],
                            }
                            _, base_value, _ = _metric_value(baseline, metric)
                            _, candidate_value, _ = _metric_value(candidate, metric)
                            sensitivity_id = _derived_id("cxsensitivity-v1-", {
                                "baseline": baseline["series_id"],
                                "candidate": candidate["series_id"],
                                "cohort": cohort["cohort_id"],
                                "metric": metric,
                            })
                            sensitivities.append({
                                "sensitivity_id": sensitivity_id,
                                "cohort_id": cohort["cohort_id"],
                                "label": (
                                    f"{axis.title()} precision sensitivity: "
                                    f"{_metric_label(measure, statistic)} T={token}"
                                ),
                                "baseline_series_id": baseline["series_id"],
                                "candidate_series_id": candidate["series_id"],
                                "metric": metric,
                                "signed_change_ratio": (
                                    candidate_value - base_value
                                ) / base_value,
                                "publication_tier": cohort["publication_tier"],
                                "eligibility": cohort["eligibility"],
                            })
    rankings.sort(key=lambda item: item["ranking_id"])
    recommendations.sort(key=lambda item: item["recommendation_id"])
    sensitivities.sort(key=lambda item: item["sensitivity_id"])
    return cohorts, rankings, recommendations, sensitivities


def _require_runnable_promotion_success(
    bundles: Sequence[dict[str, Any]], cases: dict[str, dict[str, Any]]
) -> None:
    for bundle in bundles:
        for case_id, case in cases.items():
            if case["_disposition"] != "runnable":
                continue
            status, _ = _outcome(bundle["selected"][case_id])
            if status != "success":
                raise PublisherError(
                    "promotion requires every runnable matrix case to succeed "
                    "in every selected bundle"
                )
            prior_statuses = {
                _outcome(document)[0]
                for document in bundle["documents"].values()
                if document["identity"]["case_id"] == case_id
            }
            if prior_statuses != {"success"}:
                raise PublisherError(
                    "promotion rejects runnable cases with failed, invalid, or diagnostic retries"
                )


def _expected_chip_cohort_count(series: Sequence[dict[str, Any]]) -> int:
    groups: dict[bytes, set[bytes]] = {}
    for item in series:
        control, variant = _public_cohort_factors("chip", item)
        groups.setdefault(_canonical(control), set()).add(_canonical(variant))
    return sum(len(variants) >= 2 for variants in groups.values())


def _require_promotion_cohorts(
    cohorts: Sequence[dict[str, Any]], series: Sequence[dict[str, Any]]
) -> None:
    eligible_kinds = {
        cohort["kind"]
        for cohort in cohorts
        if cohort["eligibility"]["decision_grade"]
    }
    required_kinds = list(REQUIRED_COHORT_KINDS)
    if any(
        item["workload"].get(
            "precision_profile", identity.V1_CONTROL_PRECISION_PROFILE
        )
        != identity.V1_CONTROL_PRECISION_PROFILE
        for item in series
    ):
        required_kinds.extend(PRECISION_COHORT_KINDS)
    missing = [kind for kind in required_kinds if kind not in eligible_kinds]
    if missing:
        raise PublisherError(
            "promotion lacks decision-grade cohort kinds: " + ", ".join(missing)
        )
    for kind, expected in REQUIRED_PROMOTION_COHORT_COUNTS.items():
        members = [cohort for cohort in cohorts if cohort["kind"] == kind]
        if len(members) != expected or any(
            not cohort["eligibility"]["decision_grade"] for cohort in members
        ):
            raise PublisherError(
                f"promotion requires exactly {expected} decision-grade {kind} cohorts"
            )

    chip_cohorts = [cohort for cohort in cohorts if cohort["kind"] == "chip"]
    expected_chips = _expected_chip_cohort_count(series)
    if len(chip_cohorts) != expected_chips or any(
        not cohort["eligibility"]["decision_grade"] for cohort in chip_cohorts
    ):
        raise PublisherError(
            f"promotion requires all {expected_chips} derived chip cohorts to be decision-grade"
        )


def _require_promotion_series(series: Sequence[dict[str, Any]]) -> None:
    if not series or any(item["status"] != "decision-grade" for item in series):
        raise PublisherError("promotion has unstable or incomplete required series")


def build_dataset(
    store: Store,
    bundle_ids: Sequence[str],
    *,
    promote: bool,
) -> dict[str, Any]:
    if not bundle_ids or len(bundle_ids) != len(set(bundle_ids)):
        raise PublisherError("dataset requires unique explicit bundle IDs")
    loaded = [load_bundle(store, bundle_id) for bundle_id in bundle_ids]
    loaded.sort(key=lambda bundle: (
        int(bundle["manifest"]["run"]["run_id"]),
        bundle["manifest"]["run"]["run_attempt"],
        bundle["id"],
    ))
    matrix_ids = {bundle["manifest"]["matrix"]["sha256"] for bundle in loaded}
    case_sets = [{case["case_id"] for case in bundle["cases"]} for bundle in loaded]
    if len(matrix_ids) != 1 or len({tuple(sorted(values)) for values in case_sets}) != 1:
        raise PublisherError("dataset bundles do not share one exact requested matrix")
    run_ids = [bundle["manifest"]["run"]["run_id"] for bundle in loaded]
    qualification_indices = sorted(
        bundle["manifest"]["run"]["qualification_index"] for bundle in loaded
    )
    if promote and (
        len(loaded) != REQUIRED_ALLOCATIONS
        or len(run_ids) != len(set(run_ids))
        or qualification_indices != [1, 2, 3]
        or any(bundle["manifest"]["run"]["run_attempt"] != 1 for bundle in loaded)
    ):
        raise PublisherError(
            "promotion requires qualification indices 1, 2, and 3 from first-attempt runs"
        )
    if promote and matrix_ids != {CANONICAL_FULL_V1_MATRIX_SHA256}:
        raise PublisherError("promotion requires the canonical full-v1 matrix")
    cases = {case["case_id"]: case for case in loaded[0]["cases"]}
    if promote:
        _require_runnable_promotion_success(loaded, cases)
    all_documents = [
        document for bundle in loaded for document in bundle["documents"].values()
    ]
    selected_ids = {
        selection["selected_attempt_id"]
        for bundle in loaded for selection in bundle["manifest"]["coverage"]["selections"]
    }
    public_attempts = [
        _public_attempt(
            document, selected=document["identity"]["attempt_id"] in selected_ids
        )
        for document in all_documents
    ]
    _unique([attempt["attempt_id"] for attempt in public_attempts], "dataset attempts")
    selected_by_case: dict[str, list[dict[str, Any]]] = {
        case_id: [bundle["selected"][case_id] for bundle in loaded]
        for case_id in sorted(cases)
    }
    samples_by_attempt = {
        attempt_id: sample_document
        for bundle in loaded
        for attempt_id, sample_document in bundle["sample_documents"].items()
    }
    coverage: list[dict[str, Any]] = []
    for case_id, case in sorted(cases.items()):
        attempts = sorted(
            (attempt for attempt in public_attempts if attempt["case_id"] == case_id),
            key=lambda attempt: (
                int(attempt["run_id"]), attempt["run_attempt"],
                attempt["attempt_index"], attempt["attempt_id"],
            ),
        )
        selected_document = selected_by_case[case_id][-1]
        selected = _public_attempt(selected_document, selected=True)
        precision_profile = case.get(
            "precision_profile", identity.V1_CONTROL_PRECISION_PROFILE
        )
        precision = identity.precision_profile(precision_profile)
        selected_raw = (
            selected_document
            if selected_document["format"] == contracts.RAW_FORMAT
            and selected_document["outcome"]["status"] == "success"
            else None
        )
        if selected_raw is not None:
            backend_generation = selected_raw["implementation"]["kernel_generation"]
            projected = contracts.public_series_config(
                kernel_generation=backend_generation,
                provenance=selected_raw["implementation"]["provenance"],
                resource_profile=selected_raw["implementation"]["resource_profile"],
                resource_mode=selected_raw["case"]["resource_mode"],
                device_product=selected_raw["topology"]["device_product"],
            )
            resource = projected["resource"]
            rows_by_token = {
                row["tokens_per_rank"]: row for row in selected_raw["measurement"]["rows"]
            }
            series_id = selected_raw["identity"]["series_id"]
        else:
            backend_generation = None
            resource = {
                "mode": "fixed-profile",
                "profile": None,
                "comm_units_kind": None,
                "configured_units": None,
            }
            rows_by_token = {}
            series_id = None
        point_status = (
            "measured" if selected["outcome"] == "success" else selected["outcome"]
        )
        point_reason = (
            None
            if point_status == "measured"
            else case["_reason"]
            if point_status == "unsupported"
            else selected["reason"]
        )
        token_ladder = [int(value) for value in case["ladder"].split()]
        coverage_points = []
        for token in token_ladder:
            row = rows_by_token.get(token)
            coverage_points.append({
                "point_id": row["point_id"] if row is not None else None,
                "series_id": series_id if row is not None else None,
                "tokens_per_rank": token,
                "global_tokens": token * case["ep"],
                "terminal_status": point_status,
                "reason": point_reason,
            })
        coverage.append({
            "case_id": case_id,
            "label": (
                f"{case['sku'].upper()} / {case['backend']} / EP{case['ep']} / "
                f"{case['mode']} / {case['phase']} / {case['routing']}"
            ),
            "required": True,
            "sku": _slug(case["sku"]),
            "suite": _slug(case["suite"]),
            "workload": _slug(case["workload"]),
            "publication_tier": case["required_publication"],
            "backend": _slug(case["backend"]),
            "backend_generation": backend_generation,
            "mode": case["mode"],
            "phase": case["phase"],
            "routing": case["routing"],
            "eplb": case["eplb"],
            "precision_profile": precision_profile,
            "dispatch_precision": precision["dispatch"],
            "combine_precision": precision["combine"],
            "resource": resource,
            "topology": _coverage_topology(case),
            "points": coverage_points,
            "disposition": case["_disposition"],
            "selected_attempt_id": selected["attempt_id"],
            "outcome": selected["outcome"],
            "failure_mode": selected["failure_mode"],
            "reason": case["_reason"] if case["_disposition"] == "unsupported" else selected["reason"],
            "attempt_ids": [attempt["attempt_id"] for attempt in attempts],
        })
    by_series: dict[str, list[dict[str, Any]]] = {}
    for case_documents in selected_by_case.values():
        for document in case_documents:
            if (
                document["format"] == contracts.RAW_FORMAT
                and document["outcome"]["status"] == "success"
            ):
                by_series.setdefault(document["identity"]["series_id"], []).append(document)
    series: list[dict[str, Any]] = []
    internals: dict[str, dict[str, Any]] = {}
    for series_id, documents in sorted(by_series.items()):
        try:
            sample_documents = [
                samples_by_attempt[document["identity"]["attempt_id"]]
                for document in documents
            ]
        except KeyError as exc:
            raise PublisherError(
                "selected raw evidence is missing its private sample document"
            ) from exc
        item, internal = _build_series(
            series_id, documents, sample_documents, len(loaded)
        )
        series.append(item)
        internals[series_id] = internal
    dataset_binding = _sha_bytes(_canonical({
        "matrix_id": next(iter(matrix_ids)),
        "source_bundle_ids": sorted(bundle_ids),
    }))
    cohorts, rankings, recommendations, sensitivities = build_decisions(
        series, internals, dataset_binding=dataset_binding
    )
    allocation_ids = sorted({attempt["allocation_id"] for attempt in public_attempts})
    qualification_indices = sorted({int(value) for value in qualification_indices})
    measured_cases = sum(
        all(point["terminal_status"] == "measured" for point in item["points"])
        for item in coverage
    )
    unsupported_cases = sum(
        all(point["terminal_status"] == "unsupported" for point in item["points"])
        for item in coverage
    )
    requested_points = sum(len(item["points"]) for item in coverage)
    measured_points = sum(
        point["terminal_status"] == "measured"
        for item in coverage for point in item["points"]
    )
    unsupported_points = sum(
        point["terminal_status"] == "unsupported"
        for item in coverage for point in item["points"]
    )
    status = "promoted" if promote else "diagnostic"
    dataset = {
        "format": FORMAT_PUBLIC,
        "schema_version": 1,
        "generated_at": _latest_timestamp(
            [bundle["manifest"]["created_at"] for bundle in loaded]
        ),
        "source_bundle_ids": sorted(bundle_ids),
        "promotion": {
            "status": status,
            "reason": None,
            "matrix_id": next(iter(matrix_ids)),
            "allocation_ids": allocation_ids,
            "required_allocations": REQUIRED_ALLOCATIONS,
            "qualification_indices": qualification_indices,
            "requested_cases": len(coverage),
            "terminal_cases": len(coverage),
            "measured_cases": measured_cases,
            "unsupported_cases": unsupported_cases,
            "requested_points": requested_points,
            "terminal_points": requested_points,
            "measured_points": measured_points,
            "unsupported_points": unsupported_points,
            "policy": POLICY,
        },
        "coverage": coverage,
        "attempts": sorted(public_attempts, key=lambda attempt: attempt["attempt_id"]),
        "series": series,
        "cohorts": cohorts,
        "rankings": rankings,
        "recommendations": recommendations,
        "sensitivities": sensitivities,
    }
    if promote:
        _require_promotion_series(series)
        _require_promotion_cohorts(cohorts, series)
    validate_public_dataset(dataset)
    return dataset


def quarantine_incoming(
    store: Store, ingest_id: str, reason: str, generated_at: str
) -> str:
    if REASON.fullmatch(reason) is None:
        raise PublisherError("quarantine reason must be a machine code")
    public_reason = f"{reason}-{ingest_id}"
    if REASON.fullmatch(public_reason) is None:
        raise PublisherError("quarantine reason and incoming ID exceed the public reason contract")
    manifest = {
        "format": "collectivex.quarantine.v1",
        "schema_version": 1,
        "created_at": generated_at,
        "incoming_id": ingest_id,
        "reason": reason,
    }
    digest = _sha_bytes(_canonical(manifest))
    with store.staging(store.quarantine, private=True) as stage:
        _write_json(stage / "quarantine.json", manifest, mode=0o600)
        store.complete(stage, digest, private=True)
        store.install(stage, store.quarantine / digest, private=True)
    if _sha_bytes(_canonical(strict_load(store.quarantine / digest / "quarantine.json"))) != digest:
        raise PublisherError("existing quarantine object differs")
    return digest


def _store_from_args(args: argparse.Namespace) -> Store:
    root = args.store_root or os.environ.get("COLLECTIVEX_STORE_ROOT")
    if not root:
        raise PublisherError("COLLECTIVEX_STORE_ROOT or --store-root is required")
    if not Path(root).is_absolute():
        raise PublisherError("COLLECTIVEX_STORE_ROOT must be an absolute path")
    return Store(root)


def _run_metadata(args: argparse.Namespace) -> dict[str, Any]:
    """Validate offline operator assertions about a completed successful GHA run.

    The publisher deliberately performs no network access. The caller must preflight workflow
    identity and conclusion against GitHub before supplying these values; artifact-internal
    provenance is then required to match them exactly.
    """
    run = {
        "repository": args.repository,
        "run_id": args.run_id,
        "run_attempt": args.run_attempt,
        "qualification_index": args.qualification_index,
        "source_sha": args.source_sha,
    }
    # Reuse the authoritative private schema constraints before any filesystem mutation.
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", run["repository"] or ""):
        raise PublisherError("--repository must be owner/name")
    if not re.fullmatch(r"[1-9][0-9]*", run["run_id"] or ""):
        raise PublisherError("--run-id must be a positive decimal string")
    if type(run["run_attempt"]) is not int or run["run_attempt"] < 1:
        raise PublisherError("--run-attempt must be positive")
    if type(run["qualification_index"]) is not int or run["qualification_index"] not in range(1, 4):
        raise PublisherError("--qualification-index must be 1, 2, or 3")
    if not re.fullmatch(r"[0-9a-f]{40}", run["source_sha"] or ""):
        raise PublisherError("--source-sha must be a 40-character lowercase Git SHA")
    return run


def _ingest_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Path, list[Path]]:
    run = _run_metadata(args)
    matrix = Path(args.matrix).absolute()
    if matrix.is_symlink() or not matrix.is_file():
        raise PublisherError("--matrix must be a regular non-symlink file")
    artifacts = [Path(value).absolute() for value in args.artifact]
    if not artifacts:
        raise PublisherError("at least one --artifact is required")
    names = [_artifact_name(path) for path in artifacts]
    if len(names) != len(set(names)):
        raise PublisherError("--artifact contains duplicate GHA names")
    for path in artifacts:
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise PublisherError("--artifact must be a regular ZIP or real directory")
    return run, matrix, artifacts


def _bundle_ids(values: Sequence[str], *, promote: bool) -> list[str]:
    bundle_ids = list(values)
    if (
        not bundle_ids
        or len(bundle_ids) != len(set(bundle_ids))
        or any(HEX64.fullmatch(value) is None for value in bundle_ids)
    ):
        raise PublisherError("bundle IDs must be unique SHA-256 digests")
    if promote and len(bundle_ids) != REQUIRED_ALLOCATIONS:
        raise PublisherError("promotion requires exactly three explicit bundle IDs")
    return bundle_ids


def ingest_command(args: argparse.Namespace) -> dict[str, Any]:
    run, matrix, artifacts = _ingest_inputs(args)
    store = _store_from_args(args)
    with store.locked():
        ingest_id, incoming, _ = archive_incoming(
            store, matrix, artifacts, run
        )
        try:
            bundle_id, _, _ = build_bundle(store, ingest_id, incoming, run)
            return {
                "status": "accepted", "incoming_id": ingest_id,
                "bundle_id": bundle_id,
            }
        except (
            PublisherError, contracts.ContractError, artifact_safety.ArtifactSafetyError,
            jsonschema.ValidationError,
        ) as exc:
            # Invalid delivery bytes provide no trusted timestamp. A fixed sentinel keeps
            # repeated quarantine of the same immutable incoming object content-idempotent.
            generated_at = "1970-01-01T00:00:00Z"
            quarantine_id = quarantine_incoming(
                store, ingest_id, "artifact-validation-failed", generated_at
            )
            raise PublisherError(
                f"incoming {ingest_id} quarantined as {quarantine_id}: {exc}"
            ) from exc


def promote_command(args: argparse.Namespace) -> dict[str, Any]:
    bundle_ids = _bundle_ids(args.bundle, promote=True)
    store = _store_from_args(args)
    with store.locked():
        dataset = build_dataset(store, bundle_ids, promote=True)
        digest, size = store.install_dataset(dataset)
        store.update_channel("dev-latest", digest, size, dataset["generated_at"])
        store.verify_channel("dev-latest")
    return {
        "status": "promoted", "bundle_ids": bundle_ids,
        "dataset_sha256": digest, "channel": "dev-latest",
    }


def verify_command(args: argparse.Namespace) -> dict[str, Any]:
    bundle_ids = _bundle_ids(args.bundle, promote=False) if args.bundle else []
    channels = args.channel or ["dev-latest"]
    if any(channel != "dev-latest" for channel in channels):
        raise PublisherError("unknown channel")
    store = _store_from_args(args)
    with store.locked():
        pointers = {channel: store.verify_channel(channel) for channel in channels}
        bundles = [load_bundle(store, bundle_id)["id"] for bundle_id in bundle_ids]
    return {"status": "verified", "channels": pointers, "bundle_ids": bundles}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CollectiveX isolated filesystem publisher")
    parser.add_argument("--store-root", help="defaults to COLLECTIVEX_STORE_ROOT")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest = subparsers.add_parser("ingest", help="archive and validate one complete GHA run")
    ingest.add_argument("--matrix", required=True)
    ingest.add_argument("--artifact", action="append", required=True)
    ingest.add_argument("--repository", required=True)
    ingest.add_argument("--run-id", required=True)
    ingest.add_argument("--run-attempt", required=True, type=int)
    ingest.add_argument("--qualification-index", required=True, type=int)
    ingest.add_argument("--source-sha", required=True)
    promote = subparsers.add_parser("promote", help="publish explicit independent bundles")
    promote.add_argument("--bundle", action="append", required=True)
    verify = subparsers.add_parser("verify", help="verify immutable targets and pointers")
    verify.add_argument("--channel", action="append", choices=["dev-latest"])
    verify.add_argument("--bundle", action="append", default=[])
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "ingest":
            result = ingest_command(args)
        elif args.command == "promote":
            result = promote_command(args)
        elif args.command == "verify":
            result = verify_command(args)
        else:
            raise PublisherError(f"unknown command {args.command!r}")
    except (
        PublisherError, contracts.ContractError, artifact_safety.ArtifactSafetyError,
        jsonschema.ValidationError, OSError,
    ) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
