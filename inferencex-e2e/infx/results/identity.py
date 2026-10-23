"""Benchmark identity names as InferenceX-app ingestion stores them."""

from __future__ import annotations

# InferenceX-app FRAMEWORK_ALIASES: ingestion stores these config frameworks under other names.
FRAMEWORK_ALIASES = {
    "atom-disagg": "mooncake-atom",
    "sglang-disagg": "mori-sglang",
}


def public_framework(framework: str) -> str:
    """The framework name the database and public API hold for a config's ``framework``."""
    framework = framework.lower()
    return FRAMEWORK_ALIASES.get(framework, framework)
