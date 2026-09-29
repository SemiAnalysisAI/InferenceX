"""Verify a custom AgentX model's context metadata without executing model code."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def native_context_length(config: dict[str, Any]) -> int:
    text = config.get("text_config", config.get("language_config", config))
    if not isinstance(text, dict):
        raise ValueError("Model text configuration must be an object")  # noqa: TRY004
    values = [
        text.get(name)
        for name in (
            "max_position_embeddings",
            "max_sequence_length",
            "seq_length",
            "n_positions",
        )
    ]
    confirmed = [value for value in values if type(value) is int and value > 0]
    if not confirmed:
        raise ValueError(
            "config.json does not declare a positive native context length"
        )
    # Do not guess a RoPE extension or trust tokenizer sentinel limits. When
    # aliases disagree, only their common supported range is confirmed.
    return min(confirmed)


def verify_model_config(
    path: Path, expected_sha256: str, native: int, maximum: int
) -> None:
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != expected_sha256:
        raise ValueError("Custom AgentX model config changed after recipe resolution")
    config = json.loads(contents)
    if not isinstance(config, dict) or native_context_length(config) != native:
        raise ValueError(
            "Custom AgentX native context metadata does not match its recipe"
        )
    if not 0 < maximum <= native:
        raise ValueError(
            "Custom AgentX replay context must be within the native model context"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--native-context", type=int, required=True)
    parser.add_argument("--max-context", type=int, required=True)
    args = parser.parse_args()
    try:
        verify_model_config(
            args.config, args.expected_sha256, args.native_context, args.max_context
        )
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Custom AgentX model metadata rejected: {exc}\n")


if __name__ == "__main__":
    main()
