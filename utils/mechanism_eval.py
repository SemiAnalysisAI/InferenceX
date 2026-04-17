"""ISB1 mechanism_eval schema: env-driven mechanism fields + registry validation.

This module extends the ISB1 replay result schema with a backward-compatible set
of optional fields that classify every row by the *mechanism* it exercises
(baseline, KV quantization, KV compression, compressed attention, speculative
decoding). It also loads the mechanism_variant and quality_eval registries and
exposes helpers used by process_result_isb1.py and gate_isb1.py.

The schema is strictly additive: every new field defaults to None so existing
consumers are unaffected until they opt into the mechanism_eval vocabulary.

Hard gate: any row claiming support_status == "supported" with a compression
mechanism (kv_quantization, kv_compression, compressed_attention) must carry a
registered quality_eval_id and quality_eval_status == "completed". gate_isb1.py
enforces this.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional


# Ordered list of optional mechanism_eval fields surfaced on every processed
# ISB1 row. Each field is driven by an environment variable of the same
# (upper-cased) name and defaults to None when the variable is unset.
MECHANISM_FIELDS: tuple[tuple[str, str, str], ...] = (
    # (row_key, env_var, kind)  — kind is one of str, float, int, bool
    ("mechanism", "MECHANISM", "str"),
    ("mechanism_variant", "MECHANISM_VARIANT", "str"),
    ("compression_method", "COMPRESSION_METHOD", "str"),
    ("compression_scope", "COMPRESSION_SCOPE", "str"),
    ("compression_ratio", "COMPRESSION_RATIO", "float"),
    ("compression_overhead_ms", "COMPRESSION_OVERHEAD_MS", "float"),
    ("decompression_overhead_ms", "DECOMPRESSION_OVERHEAD_MS", "float"),
    ("quality_eval_id", "QUALITY_EVAL_ID", "str"),
    ("quality_eval_status", "QUALITY_EVAL_STATUS", "str"),
    ("quality_delta_summary", "QUALITY_DELTA_SUMMARY", "str"),
    ("draft_model_id", "DRAFT_MODEL_ID", "str"),
    ("speculative_acceptance_rate", "SPECULATIVE_ACCEPTANCE_RATE", "float"),
    ("speculative_wasted_tokens", "SPECULATIVE_WASTED_TOKENS", "int"),
    ("mechanism_notes", "MECHANISM_NOTES", "str"),
)

# Default values when the mechanism env vars are absent. `mechanism` defaults
# to "baseline" so unclassified rows are never silently treated as compressed.
_DEFAULTS: dict[str, Any] = {"mechanism": "baseline"}

COMPRESSION_MECHANISMS: frozenset[str] = frozenset(
    {"kv_quantization", "kv_compression", "compressed_attention"}
)
SPECULATIVE_MECHANISMS: frozenset[str] = frozenset({"speculative_decoding"})
VALID_QUALITY_STATUSES: frozenset[str] = frozenset(
    {"pending", "completed", "failed", "not_required"}
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MECHANISM_REGISTRY_PATH = REPO_ROOT / "datasets/isb1/registry/mechanism_variant_registry.json"
QUALITY_REGISTRY_PATH = REPO_ROOT / "datasets/isb1/registry/quality_eval_registry.json"


def _coerce(value: Optional[str], kind: str) -> Any:
    if value is None or value == "":
        return None
    try:
        if kind == "float":
            return float(value)
        if kind == "int":
            return int(float(value))
        if kind == "bool":
            return value.lower() in {"1", "true", "yes", "on"}
    except (TypeError, ValueError):
        return None
    return value


def build_mechanism_fields(
    env: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Return the mechanism_eval field dict derived from environment variables.

    Unset or blank environment variables yield None for every field except
    `mechanism`, which defaults to "baseline" so rows are never silently
    unclassified.
    """
    env = os.environ if env is None else env
    result: dict[str, Any] = {}
    for row_key, env_var, kind in MECHANISM_FIELDS:
        raw = env.get(env_var)
        coerced = _coerce(raw, kind)
        if coerced is None:
            coerced = _DEFAULTS.get(row_key)
        result[row_key] = coerced
    return result


def load_mechanism_registry(path: Optional[Path] = None) -> dict[str, Any]:
    registry_path = path or MECHANISM_REGISTRY_PATH
    payload = json.loads(registry_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Mechanism registry at {registry_path} is not a JSON object.")
    return payload


def load_quality_registry(path: Optional[Path] = None) -> dict[str, Any]:
    registry_path = path or QUALITY_REGISTRY_PATH
    payload = json.loads(registry_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Quality registry at {registry_path} is not a JSON object.")
    return payload


def registered_variant_keys(registry: dict[str, Any]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for entry in registry.get("variants", []) or []:
        mechanism = entry.get("mechanism")
        variant = entry.get("mechanism_variant")
        if mechanism and variant:
            keys.add((mechanism, variant))
    return keys


def registered_quality_ids(registry: dict[str, Any]) -> set[str]:
    return {
        entry.get("quality_eval_id")
        for entry in registry.get("eval_harnesses", []) or []
        if entry.get("quality_eval_id")
    }


def validate_mechanism_fields(
    fields: dict[str, Any],
    *,
    mechanism_registry: Optional[dict[str, Any]] = None,
    quality_registry: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Return a validation record describing registration + coherence issues.

    The record never raises: it is additive metadata attached to the processed
    row and consumed by gate_isb1.py. Unregistered mechanism/variant pairs
    yield `mechanism_eval_registered=False`; an unregistered quality_eval_id
    yields `quality_eval_registered=False`.
    """
    mechanism_registry = mechanism_registry or load_mechanism_registry()
    quality_registry = quality_registry or load_quality_registry()

    mechanism = fields.get("mechanism")
    variant = fields.get("mechanism_variant") or "none"
    quality_eval_id = fields.get("quality_eval_id")
    quality_eval_status = fields.get("quality_eval_status")

    variant_key = (mechanism, variant)
    variant_registered = variant_key in registered_variant_keys(mechanism_registry)

    if quality_eval_id is None:
        quality_registered: Optional[bool] = None
    else:
        quality_registered = quality_eval_id in registered_quality_ids(quality_registry)

    status_known = (
        quality_eval_status is None or quality_eval_status in VALID_QUALITY_STATUSES
    )

    issues: list[str] = []
    if not variant_registered and mechanism != "baseline":
        issues.append(
            f"mechanism/mechanism_variant pair ({mechanism!r}, {variant!r}) "
            "is not registered in mechanism_variant_registry.json"
        )
    if quality_eval_id is not None and quality_registered is False:
        issues.append(
            f"quality_eval_id={quality_eval_id!r} is not registered in quality_eval_registry.json"
        )
    if not status_known:
        issues.append(
            f"quality_eval_status={quality_eval_status!r} is outside the accepted set "
            f"{sorted(VALID_QUALITY_STATUSES)}"
        )
    if mechanism in SPECULATIVE_MECHANISMS and not fields.get("draft_model_id"):
        issues.append(
            "speculative_decoding mechanism requires draft_model_id to be set"
        )

    return {
        "mechanism_eval_registered": variant_registered,
        "quality_eval_registered": quality_registered,
        "quality_eval_status_known": status_known,
        "issues": issues,
    }


def row_requires_completed_quality_eval(
    mechanism: Optional[str], support_status: Optional[str]
) -> bool:
    """Hard rule: supported tier × compression mechanism ⇒ completed quality eval."""
    if support_status != "supported":
        return False
    return mechanism in COMPRESSION_MECHANISMS


__all__ = [
    "MECHANISM_FIELDS",
    "COMPRESSION_MECHANISMS",
    "SPECULATIVE_MECHANISMS",
    "VALID_QUALITY_STATUSES",
    "MECHANISM_REGISTRY_PATH",
    "QUALITY_REGISTRY_PATH",
    "build_mechanism_fields",
    "load_mechanism_registry",
    "load_quality_registry",
    "registered_variant_keys",
    "registered_quality_ids",
    "validate_mechanism_fields",
    "row_requires_completed_quality_eval",
]
