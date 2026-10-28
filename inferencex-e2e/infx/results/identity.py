"""Benchmark identity names as InferenceX-app ingestion stores them."""

from __future__ import annotations

FRAMEWORK_ALIASES = {
    "atom-disagg": "mooncake-atom",
    "sglang-disagg": "mori-sglang",
    "trtllm": "trt",
    "dynamo-trtllm": "dynamo-trt",
}
_LONGEST_FIRST = sorted(FRAMEWORK_ALIASES.items(), key=lambda alias: -len(alias[0]))


def public_framework(framework: str) -> str:
    """The framework name the database and public API hold for a config's ``framework``.

    The aliases mirror InferenceX-app's FRAMEWORK_ALIASES used at ingestion.
    """
    framework = framework.lower()
    return FRAMEWORK_ALIASES.get(framework, framework)


def public_config_key(key: str) -> str:
    """A changelog config key as ingestion stores it: every alias replaced, longest first."""
    for alias, canonical in _LONGEST_FIRST:
        key = key.replace(alias, canonical)
    return key
