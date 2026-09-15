"""Build native srtctl arguments without modifying a recipe on disk."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RecipeOverride:
    """One native ``--set`` or ``--unset`` argument pair.

    Values use JSON syntax, which srtctl's YAML value parser also accepts.
    Quoting strings explicitly preserves numeric environment variables and
    engine JSON strings instead of coercing them into other YAML types.
    """

    path: str
    value: Any = None
    unset: bool = False

    def argv(self) -> list[str]:
        if self.unset:
            return ["--unset", self.path]
        return [
            "--set",
            f"{self.path}={json.dumps(self.value, separators=(',', ':'), allow_nan=False)}",
        ]


def override_argv(overrides: list[RecipeOverride]) -> list[str]:
    """Return argument boundaries for direct subprocess or Bash-array use."""
    return [argument for override in overrides for argument in override.argv()]
