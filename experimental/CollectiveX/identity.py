#!/usr/bin/env python3
"""Canonical, cross-runtime identities for CollectiveX v1."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

IDENTITY_VERSION = 1
MAX_SAFE_INTEGER = (1 << 53) - 1
PREFIXES = {
    "case": "cxcase-v1-",
    "workload": "cxwork-v1-",
    "series": "cxseries-v1-",
    "point": "cxpoint-v1-",
    "evidence": "cxevidence-v1-",
    "allocation": "cxallocation-v1-",
    "attempt": "cxattempt-v1-",
}
V1_NORMAL_CASE_PROFILE = {
    "activation_generator": "collectivex-activation-counter-v4",
    "activation_profile": "canonical-counter-source-v4",
    "combine_dtype": "bf16",
    "combine_quant_mode": "none",
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
    "oracle_tolerances": "codec-specific-combine-v1",
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


class IdentityError(ValueError):
    """An identity payload cannot be represented consistently across runtimes."""


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


def digest(kind: str, value: Any) -> str:
    """Hash a typed v1 identity payload and return its typed identifier."""
    try:
        prefix = PREFIXES[kind]
    except KeyError as exc:
        raise IdentityError(f"unknown identity kind {kind!r}") from exc
    body = {"kind": kind, "value": value, "version": IDENTITY_VERSION}
    return prefix + hashlib.sha256(canonical_bytes(body)).hexdigest()


def is_typed_id(value: Any, kind: str) -> bool:
    prefix = PREFIXES.get(kind)
    return bool(
        isinstance(value, str)
        and prefix
        and re.fullmatch(re.escape(prefix) + r"[0-9a-f]{64}", value)
    )


def case_id(*, sku: str, profile: dict[str, Any], case: dict[str, Any]) -> str:
    return digest("case", {"case": case, "profile": profile, "sku": sku})


def workload_id(value: dict[str, Any]) -> str:
    return digest("workload", value)


def series_id(value: dict[str, Any]) -> str:
    return digest("series", value)


def point_id(*, series: str, tokens_per_rank: int) -> str:
    return digest("point", {"series_id": series, "tokens_per_rank": tokens_per_rank})


def allocation_id(value: dict[str, Any]) -> str:
    return digest("allocation", value)


def attempt_id(*, allocation: str, case: str, ordinal: int) -> str:
    return digest(
        "attempt", {"allocation_id": allocation, "case_id": case, "ordinal": ordinal}
    )


def evidence_id(
    *, point: str, allocation: str, attempt: str, sample_sha256: str
) -> str:
    return digest(
        "evidence",
        {
            "allocation_id": allocation,
            "attempt_id": attempt,
            "point_id": point,
            "sample_sha256": sample_sha256,
        },
    )


IDENTITY_TEST_VECTOR = {
    "payload": {"backend": "deepep", "ep": 8, "shape": [7168, 8, 256]},
    "series_id": "cxseries-v1-a79bf758488e3edd50f5531f3af825f371bf42aae7c4097e461fd2a32615af81",
}


def verify_test_vector() -> None:
    observed = series_id(IDENTITY_TEST_VECTOR["payload"])
    if observed != IDENTITY_TEST_VECTOR["series_id"]:
        raise IdentityError(
            f"identity implementation differs: {observed} != {IDENTITY_TEST_VECTOR['series_id']}"
        )


if __name__ == "__main__":
    verify_test_vector()
    print(IDENTITY_TEST_VECTOR["series_id"])
