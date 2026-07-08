#!/usr/bin/env python3
"""Canonical cross-runtime identities for CollectiveX.

A case identity is a readable, factor-derived join between a scheduled case and
its emitted result: a human-readable body (SKU, backend, workload, mode, phase,
EP size, routing) plus a short digest of the exact case factors. The suffix keeps
the identity collision-proof and tamper-evident without hiding what the case is.
Attempt and point identities are readable derivations of the case identity.

Content SHA-256 survives only where it identifies separately-stored bytes: the
workload manifest whose exact routing/gate data a result consumes. Detached
sample files, source commits, container images, and generated kernels carry their
own SHA-256 fields in the artifacts; this module does not mint identifiers for
catalogs, series, evidence, allocations, or any other publisher-side grouping.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

MAX_SAFE_INTEGER = (1 << 53) - 1
WORKLOAD_PREFIX = "cxwork-v1-"
_CASE_SUFFIX_LEN = 12
_NON_SLUG = re.compile(r"[^a-z0-9]+")
_CASE_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]*-[0-9a-f]{%d}$" % _CASE_SUFFIX_LEN)
_WORKLOAD_ID = re.compile(r"^cxwork-v1-[0-9a-f]{64}$")

V1_NORMAL_CASE_PROFILE = {
    "activation_generator": "collectivex-activation-counter-v4",
    "activation_profile": "canonical-counter-source-v4",
    "combine_dtype": "bf16",
    "combine_semantics": "activation-only",
    "component_order_contract": "qualification-hash-rotated-components-v1",
    "conditioning_contract": "fixed-phase-ramp-8-roundtrips-v1",
    "contract": "layout-and-dispatch-v1",
    "correctness_scope": "dispatch-metadata-and-transformed-combine",
    "dtype": "bf16",
    "eplb_planner": "greedy-rank-major-v1",
    "eplb_redundant_experts": 32,
    "eplb_reference_tokens_per_rank": 2048,
    "mode": "normal",
    "oracle_contract": "expert-specific-transform-v1",
    "payload_unit": "token-rank",
    "placement": "packed",
    "percentile_method": "nearest-rank",
    "rank_reduction": "cross-rank-max-per-iteration",
    "resource_mode": "fixed-profile",
    "routing_generator": "collectivex-routing-counter-v3",
    "sampling_contract": "fixed-512-v1",
    "seed": 67,
    "source_identity_contract": "bounded-sign-bit-source-v1",
}

V1_LOW_LATENCY_CASE_PROFILE = {
    **V1_NORMAL_CASE_PROFILE,
    "component_order_contract": "qualification-hash-rotated-components-v1",
    "combine_semantics": "gate-weighted",
    "contract": "expert-packed-weighted-combine-v1",
    "correctness_scope": "expert-assignment-and-weighted-combine",
    "mode": "low-latency",
    "oracle_contract": "expert-assignment-transform-v1",
    "payload_unit": "token-expert",
}

# Compatibility alias for normal-mode callers. New scheduling and validation
# must select a profile from the explicit case mode.
V1_CASE_PROFILE = V1_NORMAL_CASE_PROFILE
V1_CASE_PROFILES = {
    "normal": V1_NORMAL_CASE_PROFILE,
    "low-latency": V1_LOW_LATENCY_CASE_PROFILE,
}


class IdentityError(ValueError):
    """An identity payload cannot be represented consistently across runtimes."""


def case_profile(mode: str) -> dict[str, Any]:
    """Return the immutable measurement profile for one scheduled mode."""
    try:
        return V1_CASE_PROFILES[mode]
    except KeyError as exc:
        raise IdentityError(f"unknown CollectiveX case mode {mode!r}") from exc


def profile_for_case(case: dict[str, Any]) -> dict[str, Any]:
    """Resolve a scheduled case's explicit mode to its identity profile.

    Dispatch and combine are fixed BF16 benchmark facts, so a case's identity
    profile is fully determined by its measurement mode.
    """
    mode = case.get("mode")
    if not isinstance(mode, str):
        raise IdentityError("scheduled case mode is missing")
    return case_profile(mode)


def _validate(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, str):
        if any(ord(character) < 0x20 or ord(character) > 0x7E for character in value):
            raise IdentityError(f"{path}: string must contain printable ASCII only")
        return
    if type(value) is int:
        if abs(value) > MAX_SAFE_INTEGER:
            raise IdentityError(f"{path}: integer exceeds the cross-runtime safe range")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise IdentityError(f"{path}: object key is not a string")
            if any(ord(character) < 0x20 or ord(character) > 0x7E for character in key):
                raise IdentityError(f"{path}: object key must contain printable ASCII only")
            _validate(item, f"{path}.{key}")
        return
    raise IdentityError(f"{path}: unsupported identity value {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    """Return compact UTF-8 JSON after enforcing the portable value subset."""
    _validate(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _slug(value: Any) -> str:
    text = _NON_SLUG.sub("-", str(value).strip().lower()).strip("-")
    if not text:
        raise IdentityError("case factor has no printable identity component")
    return text


def _case_body(sku: str, case: dict[str, Any]) -> str:
    parts = [
        _slug(sku),
        _slug(case["backend"]),
        _slug(case.get("workload") or "manual"),
        _slug(case["mode"]),
        _slug(case["phase"]),
        "ep%d" % int(case["ep"]),
        _slug(case["routing"]),
    ]
    if case.get("eplb"):
        parts.append("eplb")
    return "-".join(parts)


def case_id(*, sku: str, profile: dict[str, Any], case: dict[str, Any]) -> str:
    """Readable case identity: factor body plus a digest of the exact factors."""
    factors = {"case": case, "profile": profile, "sku": sku}
    return f"{_case_body(sku, case)}-{_content_sha256(factors)[:_CASE_SUFFIX_LEN]}"


def case_id_from_factors(factors: dict[str, Any]) -> str:
    """Recompute a case identity from an emitted `case_factors` object."""
    return case_id(sku=factors["sku"], profile=factors["profile"], case=factors["case"])


def attempt_id(*, case: str, ordinal: int) -> str:
    if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 1:
        raise IdentityError("attempt ordinal must be a positive integer")
    return f"{case}-a{ordinal:02d}"


def point_id(*, case: str, tokens_per_rank: int) -> str:
    if not isinstance(tokens_per_rank, int) or isinstance(tokens_per_rank, bool) or tokens_per_rank < 1:
        raise IdentityError("tokens_per_rank must be a positive integer")
    return f"{case}-t{tokens_per_rank}"


def is_case_id(value: Any) -> bool:
    return bool(isinstance(value, str) and _CASE_ID.fullmatch(value))


def workload_id(value: dict[str, Any]) -> str:
    """Content identity of a workload manifest's canonical routing/gate data."""
    return WORKLOAD_PREFIX + _content_sha256(
        {"kind": "workload", "value": value, "version": 1}
    )


def is_workload_id(value: Any) -> bool:
    return bool(isinstance(value, str) and _WORKLOAD_ID.fullmatch(value))
