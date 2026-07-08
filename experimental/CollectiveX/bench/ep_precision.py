#!/usr/bin/env python3
"""Native communication-precision helpers for CollectiveX EP adapters."""
from __future__ import annotations

import inspect
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable


class PrecisionError(RuntimeError):
    """A requested precision profile cannot be realized by the pinned API."""


# ---------------------------------------------------------------------------
# Native communication-precision catalog.
#
# Dispatch and combine are fixed BF16 benchmark facts for the default sweep, so
# no scheduled case selects an FP8 profile. The FP8 axes below stay callable so
# an adapter can still exercise its native codec directly; they are dormant, not
# removed. This catalog lives with the codec (not in ``identity``) because it is
# a property of the communication kernels, not of the neutral artifact identity.
# ---------------------------------------------------------------------------


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

V1_CONTROL_PRECISION_PROFILE = "d-bf16.c-bf16"
V1_NORMAL_PRECISION_PROFILE_IDS = (
    "d-fp8-e4m3fn-b128-f32-prequantized.c-bf16",
    "d-fp8-e4m3fnuz-b128-f32-prequantized.c-fp8-e4m3fnuz-direct-cast-noscale",
)
V1_LOW_LATENCY_PRECISION_PROFILE_IDS = (
    "d-fp8-e4m3fn-b128-f32-fused.c-bf16",
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


def precision_profile(name: str) -> dict[str, Any]:
    """Return one exact dispatch/combine communication-format profile."""
    try:
        profile = V1_PRECISION_PROFILES[name]
    except KeyError as exc:
        raise PrecisionError(f"unknown CollectiveX precision profile {name!r}") from exc
    return {"profile_id": name, **deepcopy(profile)}


def combine_oracle_tolerances(communication_precision: dict[str, Any]) -> dict[str, float]:
    """Return the frozen combine-oracle gate for one exact native codec."""
    combine = communication_precision["combine"]
    fmt = combine["communication_format"]
    if fmt == "bf16":
        key = "bf16"
    elif (
        fmt in {"fp8-e4m3fn", "fp8-e4m3fnuz"}
        and combine["quantization_origin"] == "backend-internal-direct-cast"
    ):
        key = "fp8-direct-cast"
    else:
        raise PrecisionError("precision profile has no frozen combine-oracle tolerance")
    return dict(V1_COMBINE_ORACLE_TOLERANCES[key])


@dataclass(frozen=True)
class DispatchEncoding:
    """One dispatch input plus its post-codec semantic representation."""

    native_input: Any
    encoded_payload: Any | None
    scales: Any | None
    semantic: Any
    evidence: dict[str, Any]


def resolve_precision(
    args,
    *,
    backend: str,
    mode: str,
    supported_profiles: Iterable[str],
) -> tuple[str, dict[str, Any]]:
    """Resolve and validate the exact profile requested for one adapter."""
    profile_id = (
        getattr(args, "precision_profile", "")
        or V1_CONTROL_PRECISION_PROFILE
    )
    profile = precision_profile(profile_id)  # raises PrecisionError on unknown
    if mode not in profile["modes"]:
        raise PrecisionError(
            f"precision profile {profile_id!r} is not valid in mode {mode!r}"
        )
    supported = frozenset(supported_profiles)
    if profile_id not in supported:
        raise PrecisionError(
            f"{backend} does not realize precision profile {profile_id!r} in mode {mode!r}"
        )
    return profile_id, profile


def require_keyword(callable_object, keyword: str, *, api: str) -> None:
    """Fail closed when a pinned Python API does not expose a required control."""
    try:
        parameters = inspect.signature(callable_object).parameters
    except (TypeError, ValueError) as exc:
        raise PrecisionError(f"cannot inspect required precision API {api}") from exc
    if keyword not in parameters:
        raise PrecisionError(f"required precision API {api} omits {keyword!r}")


def communication_format(profile: dict[str, Any], component: str) -> str:
    """Return the exact wire format for dispatch or combine."""
    return str(profile[component]["communication_format"])


def is_low_precision_dispatch(profile: dict[str, Any]) -> bool:
    return communication_format(profile, "dispatch").startswith("fp8-")


def is_caller_prequantized(profile: dict[str, Any]) -> bool:
    return profile["dispatch"]["quantization_origin"] == "caller-prequantized"


def uses_logfmt_combine(profile: dict[str, Any]) -> bool:
    return communication_format(profile, "combine") == "logfmt10"


def uses_direct_cast_combine(profile: dict[str, Any]) -> bool:
    return profile["combine"]["quantization_origin"] == "backend-internal-direct-cast"


def _fp8_dtype(torch_module, axis: dict[str, Any]):
    fmt = axis["communication_format"]
    attribute = {
        "fp8-e4m3fn": "float8_e4m3fn",
        "fp8-e4m3fnuz": "float8_e4m3fnuz",
    }.get(fmt)
    if attribute is None:
        raise PrecisionError(f"unsupported FP8 communication format {fmt!r}")
    dtype = getattr(torch_module, attribute, None)
    if dtype is None:
        raise PrecisionError(f"active torch build does not expose torch.{attribute}")
    return dtype


def _axis_evidence(
    *,
    dequantized_semantics: bool,
    encoded_payload_valid: bool,
    max_abs_error: float,
    max_rel_error: float,
    saturation_count: int,
    saturation_rate: float,
    scales_finite: bool | None,
    scales_positive: bool | None,
    passed: bool,
) -> dict[str, Any]:
    return {
        "encoded_payload_valid": bool(encoded_payload_valid),
        "scales_finite": scales_finite,
        "scales_positive": scales_positive,
        "dequantized_semantics": bool(dequantized_semantics),
        "saturation_count": int(saturation_count),
        "saturation_rate": float(saturation_rate),
        "max_abs_error": float(max_abs_error),
        "max_rel_error": float(max_rel_error),
        "passed": bool(passed),
    }


def exact_axis_evidence() -> dict[str, Any]:
    """Evidence for an unquantized BF16 communication axis."""
    return _axis_evidence(
        encoded_payload_valid=True,
        scales_finite=None,
        scales_positive=None,
        dequantized_semantics=True,
        saturation_count=0,
        saturation_rate=0.0,
        max_abs_error=0.0,
        max_rel_error=0.0,
        passed=True,
    )


def _quantize_fp8(torch_module, x, axis: dict[str, Any]) -> DispatchEncoding:
    group_size = axis["scale_group_size"]
    if group_size != 128 or axis["scale_dtype"] != "f32":
        raise PrecisionError("v1 FP8 dispatch requires block-128 FP32 scales")
    if x.ndim != 2 or x.shape[1] % group_size:
        raise PrecisionError(
            "v1 native FP8 dispatch requires a 2D hidden dimension divisible by 128"
        )
    dtype = _fp8_dtype(torch_module, axis)
    fp8_max = float(torch_module.finfo(dtype).max)
    blocks = x.float().reshape(x.shape[0], x.shape[1] // group_size, group_size)
    amax = blocks.abs().amax(dim=-1)
    # Match the pinned DeepEP/HybridEP block codec, including its nonzero scale floor.
    scales = (amax.clamp_min(1e-4) / fp8_max).to(torch_module.float32)
    normalized = blocks / scales.unsqueeze(-1)
    saturation_mask = normalized.abs() > fp8_max
    encoded = normalized.clamp(min=-fp8_max, max=fp8_max).to(dtype).reshape_as(x).contiguous()
    semantic = dequantize_dispatch(
        torch_module, encoded, scales, axis
    ).to(x.dtype).contiguous()
    absolute = (semantic.float() - x.float()).abs()
    max_abs = float(absolute.max().item()) if absolute.numel() else 0.0
    reference_max = float(x.float().abs().max().item()) if x.numel() else 0.0
    max_rel = max_abs / (reference_max + 1e-6)
    saturation_count = int(saturation_mask.sum().item())
    saturation_rate = saturation_count / max(1, int(x.numel()))
    finite = bool(torch_module.isfinite(scales).all().item())
    positive = bool((scales > 0).all().item())
    semantic_ok = bool(
        torch_module.isfinite(semantic.float()).all().item()
        and torch_module.allclose(
            semantic.float(), x.float(), rtol=0.05, atol=0.02
        )
    )
    valid = encoded.dtype == dtype and encoded.shape == x.shape
    evidence = _axis_evidence(
        encoded_payload_valid=valid,
        scales_finite=finite,
        scales_positive=positive,
        dequantized_semantics=semantic_ok,
        saturation_count=saturation_count,
        saturation_rate=saturation_rate,
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        passed=valid and finite and positive and semantic_ok,
    )
    return DispatchEncoding(
        native_input=(encoded, scales),
        encoded_payload=encoded,
        scales=scales,
        semantic=semantic,
        evidence=evidence,
    )


def encode_dispatch(torch_module, x, profile: dict[str, Any]) -> DispatchEncoding:
    """Build caller-prequantized input or a fused-codec oracle outside timing."""
    axis = profile["dispatch"]
    origin = axis["quantization_origin"]
    if origin == "none":
        return DispatchEncoding(
            native_input=x,
            encoded_payload=None,
            scales=None,
            semantic=x,
            evidence=exact_axis_evidence(),
        )
    if origin not in {"caller-prequantized", "backend-fused"}:
        raise PrecisionError(f"unsupported dispatch quantization origin {origin!r}")
    encoded = _quantize_fp8(torch_module, x, axis)
    if origin == "backend-fused":
        return DispatchEncoding(
            native_input=x,
            encoded_payload=encoded.encoded_payload,
            scales=encoded.scales,
            semantic=encoded.semantic,
            evidence=encoded.evidence,
        )
    return encoded


def dequantize_dispatch(
    torch_module,
    encoded_payload,
    scales,
    axis: dict[str, Any],
    *,
    uint8_storage: bool = False,
):
    """Decode one native block-scaled FP8 payload to BF16 semantics."""
    group_size = axis["scale_group_size"]
    if group_size != 128 or scales is None:
        raise PrecisionError("FP8 dispatch payload is missing block-128 scales")
    dtype = _fp8_dtype(torch_module, axis)
    payload = encoded_payload.view(dtype) if uint8_storage else encoded_payload
    if payload.dtype != dtype or payload.ndim < 2 or payload.shape[-1] % group_size:
        raise PrecisionError("native FP8 dispatch payload has an invalid dtype or shape")
    expected_scale_shape = (*payload.shape[:-1], payload.shape[-1] // group_size)
    if tuple(scales.shape) != expected_scale_shape or scales.dtype != torch_module.float32:
        raise PrecisionError("native FP8 dispatch scales have an invalid dtype or shape")
    values = payload.float().reshape(
        *payload.shape[:-1], payload.shape[-1] // group_size, group_size
    )
    return (values * scales.float().reshape(*expected_scale_shape, 1)).reshape(
        payload.shape
    ).to(torch_module.bfloat16).contiguous()


def validate_received_encoding(
    torch_module,
    *,
    encoded_payload,
    scales,
    semantic,
    axis: dict[str, Any],
    uint8_storage: bool = False,
) -> bool:
    """Validate that received bytes/scales exactly decode to the semantic view."""
    try:
        decoded = dequantize_dispatch(
            torch_module,
            encoded_payload,
            scales,
            axis,
            uint8_storage=uint8_storage,
        )
    except PrecisionError:
        return False
    return bool(torch_module.equal(decoded, semantic))


def dequantize_expert_prefixes(
    torch_module,
    encoded_payload,
    scales,
    axis: dict[str, Any],
    counts: tuple[int, ...],
    output,
):
    """Decode only valid expert-packed prefixes into a reusable BF16 workspace."""
    if encoded_payload.ndim != 3 or len(counts) != encoded_payload.shape[0]:
        raise PrecisionError("expert-packed FP8 receive counts have an invalid shape")
    if output.shape != encoded_payload.shape or output.dtype != torch_module.bfloat16:
        raise PrecisionError("expert-packed BF16 stage workspace has an invalid shape")
    capacity = encoded_payload.shape[1]
    for expert, count in enumerate(counts):
        if count < 0 or count > capacity:
            raise PrecisionError("expert-packed FP8 receive count exceeds capacity")
        if count:
            output[expert, :count].copy_(dequantize_dispatch(
                torch_module,
                encoded_payload[expert, :count],
                scales[expert, :count],
                axis,
            ))
    return output


def _direct_cast_saturation(torch_module, profile: dict[str, Any], view) -> tuple[int, float]:
    """Count values clipped in the exact native BF16-to-FP8 combine input."""
    if not hasattr(view, "combine_input"):
        return 0, 0.0
    transformed = view.combine_input.float()
    dtype = _fp8_dtype(torch_module, profile["combine"])
    fp8_max = float(torch_module.finfo(dtype).max)
    count = int((transformed.abs() > fp8_max).sum().item())
    return count, count / max(1, int(transformed.numel()))


def precision_evidence(
    torch_module,
    *,
    profile_id: str,
    profile: dict[str, Any],
    problem,
    view=None,
    uint8_storage: bool = False,
) -> dict[str, Any]:
    """Return schema-shaped codec evidence; the harness adds combine-oracle errors."""
    dispatch = deepcopy(problem.dispatch_precision_evidence)
    if (
        is_low_precision_dispatch(profile)
        and view is not None
        and all(hasattr(view, field) for field in ("encoded_payload", "scales", "payload"))
    ):
        valid = validate_received_encoding(
            torch_module,
            encoded_payload=view.encoded_payload,
            scales=view.scales,
            semantic=view.payload,
            axis=profile["dispatch"],
            uint8_storage=uint8_storage,
        )
        dispatch["encoded_payload_valid"] = (
            dispatch["encoded_payload_valid"] and valid
        )
        dispatch["passed"] = dispatch["passed"] and valid
    combine = exact_axis_evidence()
    # Internal quantizers are validated by native configuration here. The harness
    # replaces the error fields and pass bit with the transformed-combine oracle.
    if communication_format(profile, "combine") != "bf16":
        combine["scales_finite"] = None
        combine["scales_positive"] = None
    if uses_direct_cast_combine(profile) and view is not None:
        count, rate = _direct_cast_saturation(torch_module, profile, view)
        combine["saturation_count"] = count
        combine["saturation_rate"] = rate
        combine["passed"] = combine["passed"] and count == 0
    return {
        "profile_id": profile_id,
        "dispatch": dispatch,
        "combine": combine,
        "passed": bool(dispatch["passed"] and combine["passed"]),
    }
