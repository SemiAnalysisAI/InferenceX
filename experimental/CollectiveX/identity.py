#!/usr/bin/env python3
"""Canonical, cross-runtime identities for CollectiveX v1."""
from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
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

V1_CONTROL_PRECISION_PROFILE = "d-bf16.c-bf16"
V1_NORMAL_PRECISION_PROFILE_IDS = (
    "d-fp8-e4m3fn-b128-f32-prequantized.c-bf16",
    "d-bf16.c-fp8-e4m3fn-direct-cast-noscale",
    "d-fp8-e4m3fn-b128-f32-prequantized.c-fp8-e4m3fn-direct-cast-noscale",
    "d-fp8-e4m3fnuz-b128-f32-prequantized.c-bf16",
    "d-bf16.c-fp8-e4m3fnuz-direct-cast-noscale",
    "d-fp8-e4m3fnuz-b128-f32-prequantized.c-fp8-e4m3fnuz-direct-cast-noscale",
)
V1_LOW_LATENCY_PRECISION_PROFILE_IDS = (
    "d-fp8-e4m3fn-b128-f32-fused.c-bf16",
)


def _communication_axis(
    *,
    api_input_dtype: str,
    api_output_dtype: str,
    communication_format: str,
    scale_dtype: str | None,
    scale_layout: str,
    scale_group_size: int | None,
    padding_contract: str,
    alignment_contract: str,
    quantization_origin: str,
    conversion_boundary: str,
) -> dict[str, Any]:
    return {
        "api_input_dtype": api_input_dtype,
        "api_output_dtype": api_output_dtype,
        "communication_format": communication_format,
        "scale_dtype": scale_dtype,
        "scale_layout": scale_layout,
        "scale_group_size": scale_group_size,
        "padding_contract": padding_contract,
        "alignment_contract": alignment_contract,
        "quantization_origin": quantization_origin,
        "conversion_boundary": conversion_boundary,
    }


_BF16_AXIS = _communication_axis(
    api_input_dtype="bf16",
    api_output_dtype="bf16",
    communication_format="bf16",
    scale_dtype=None,
    scale_layout="none",
    scale_group_size=None,
    padding_contract="none",
    alignment_contract="native-bf16-vector-alignment",
    quantization_origin="none",
    conversion_boundary="none",
)
_FP8_E4M3FN_PREQUANTIZED_DISPATCH = _communication_axis(
    api_input_dtype="fp8-e4m3fn-with-f32-scale",
    api_output_dtype="fp8-e4m3fn-with-f32-scale",
    communication_format="fp8-e4m3fn",
    scale_dtype="f32",
    scale_layout="per-token-hidden-block",
    scale_group_size=128,
    padding_contract="right-zero-pad-hidden-to-128",
    alignment_contract="hidden-block-128",
    quantization_origin="caller-prequantized",
    conversion_boundary="before-dispatch-timing",
)
_FP8_E4M3FNUZ_PREQUANTIZED_DISPATCH = _communication_axis(
    api_input_dtype="fp8-e4m3fnuz-with-f32-scale",
    api_output_dtype="fp8-e4m3fnuz-with-f32-scale",
    communication_format="fp8-e4m3fnuz",
    scale_dtype="f32",
    scale_layout="per-token-hidden-block",
    scale_group_size=128,
    padding_contract="right-zero-pad-hidden-to-128",
    alignment_contract="hidden-block-128",
    quantization_origin="caller-prequantized",
    conversion_boundary="before-dispatch-timing",
)
_FP8_E4M3FN_FUSED_DISPATCH = _communication_axis(
    api_input_dtype="bf16",
    api_output_dtype="fp8-e4m3fn-with-f32-scale",
    communication_format="fp8-e4m3fn",
    scale_dtype="f32",
    scale_layout="per-token-hidden-block",
    scale_group_size=128,
    padding_contract="right-zero-pad-hidden-to-128",
    alignment_contract="hidden-block-128",
    quantization_origin="backend-fused",
    conversion_boundary="inside-dispatch-timing",
)
_FP8_E4M3FN_DIRECT_CAST_COMBINE = _communication_axis(
    api_input_dtype="bf16",
    api_output_dtype="bf16",
    communication_format="fp8-e4m3fn",
    scale_dtype=None,
    scale_layout="none",
    scale_group_size=None,
    padding_contract="none",
    alignment_contract="native-fp8-vector-alignment",
    quantization_origin="backend-internal-direct-cast",
    conversion_boundary="inside-combine-timing",
)
_FP8_E4M3FNUZ_DIRECT_CAST_COMBINE = _communication_axis(
    api_input_dtype="bf16",
    api_output_dtype="bf16",
    communication_format="fp8-e4m3fnuz",
    scale_dtype=None,
    scale_layout="none",
    scale_group_size=None,
    padding_contract="none",
    alignment_contract="native-fp8-vector-alignment",
    quantization_origin="backend-internal-direct-cast",
    conversion_boundary="inside-combine-timing",
)

V1_PRECISION_PROFILES: dict[str, dict[str, Any]] = {
    V1_CONTROL_PRECISION_PROFILE: {
        "modes": ["normal", "low-latency"],
        "dispatch": _BF16_AXIS,
        "combine": _BF16_AXIS,
    },
    "d-fp8-e4m3fn-b128-f32-prequantized.c-bf16": {
        "modes": ["normal"],
        "dispatch": _FP8_E4M3FN_PREQUANTIZED_DISPATCH,
        "combine": _BF16_AXIS,
    },
    "d-fp8-e4m3fn-b128-f32-fused.c-bf16": {
        "modes": ["low-latency"],
        "dispatch": _FP8_E4M3FN_FUSED_DISPATCH,
        "combine": _BF16_AXIS,
    },
    "d-bf16.c-fp8-e4m3fn-direct-cast-noscale": {
        "modes": ["normal"],
        "dispatch": _BF16_AXIS,
        "combine": _FP8_E4M3FN_DIRECT_CAST_COMBINE,
    },
    "d-fp8-e4m3fn-b128-f32-prequantized.c-fp8-e4m3fn-direct-cast-noscale": {
        "modes": ["normal"],
        "dispatch": _FP8_E4M3FN_PREQUANTIZED_DISPATCH,
        "combine": _FP8_E4M3FN_DIRECT_CAST_COMBINE,
    },
    "d-fp8-e4m3fnuz-b128-f32-prequantized.c-bf16": {
        "modes": ["normal"],
        "dispatch": _FP8_E4M3FNUZ_PREQUANTIZED_DISPATCH,
        "combine": _BF16_AXIS,
    },
    "d-bf16.c-fp8-e4m3fnuz-direct-cast-noscale": {
        "modes": ["normal"],
        "dispatch": _BF16_AXIS,
        "combine": _FP8_E4M3FNUZ_DIRECT_CAST_COMBINE,
    },
    "d-fp8-e4m3fnuz-b128-f32-prequantized.c-fp8-e4m3fnuz-direct-cast-noscale": {
        "modes": ["normal"],
        "dispatch": _FP8_E4M3FNUZ_PREQUANTIZED_DISPATCH,
        "combine": _FP8_E4M3FNUZ_DIRECT_CAST_COMBINE,
    },
}

V1_COMBINE_ORACLE_TOLERANCES = {
    "bf16": {"atol": 2e-2, "rtol": 5e-2},
    "fp8-direct-cast": {"atol": 4e-2, "rtol": 8e-2},
}


def combine_oracle_tolerances(communication_precision: dict[str, Any]) -> dict[str, float]:
    """Return the frozen combine-oracle gate for one exact native codec."""
    combine = communication_precision["combine"]
    communication_format = combine["communication_format"]
    if communication_format == "bf16":
        key = "bf16"
    elif (
        communication_format in {"fp8-e4m3fn", "fp8-e4m3fnuz"}
        and combine["quantization_origin"] == "backend-internal-direct-cast"
    ):
        key = "fp8-direct-cast"
    else:
        raise ValueError("precision profile has no frozen combine-oracle tolerance")
    return dict(V1_COMBINE_ORACLE_TOLERANCES[key])


def case_profile(mode: str) -> dict[str, Any]:
    """Return the immutable measurement profile for one scheduled mode."""
    try:
        return V1_CASE_PROFILES[mode]
    except KeyError as exc:
        raise IdentityError(f"unknown CollectiveX case mode {mode!r}") from exc


def precision_profile(name: str) -> dict[str, Any]:
    """Return one exact dispatch/combine communication-format profile."""
    try:
        profile = V1_PRECISION_PROFILES[name]
    except KeyError as exc:
        raise IdentityError(f"unknown CollectiveX precision profile {name!r}") from exc
    return {"profile_id": name, **deepcopy(profile)}


def profile_for_case(case: dict[str, Any]) -> dict[str, Any]:
    """Resolve a scheduled case's explicit mode to its identity profile."""
    mode = case.get("mode")
    if not isinstance(mode, str):
        raise IdentityError("scheduled case mode is missing")
    base = case_profile(mode)
    precision_name = case.get("precision_profile")
    if precision_name is None:
        return base
    if not isinstance(precision_name, str):
        raise IdentityError("scheduled case precision_profile must be a string")
    precision = precision_profile(precision_name)
    if mode not in precision["modes"]:
        raise IdentityError(
            f"precision profile {precision_name!r} is not valid in mode {mode!r}"
        )
    return {**base, "communication_precision": precision}


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
