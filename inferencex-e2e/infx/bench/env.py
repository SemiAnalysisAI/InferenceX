"""Caller-supplied inputs: a missing or malformed one fails, naming every offender."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping


class BenchError(Exception):
    """Reported by ``python3 -m infx.bench`` as one ``ERROR:`` line, exit code 1."""


class InputError(BenchError):
    """A required caller input is missing or malformed."""


def require(*names: str, env: Mapping[str, str] = os.environ) -> dict[str, str]:
    """Return the named values; unset and empty both count as missing."""
    missing = [name for name in names if not env.get(name)]
    if missing:
        listed = "\n".join(f"  - {name}" for name in missing)
        raise InputError(f"The following required environment variables are not set:\n{listed}")
    return {name: env[name] for name in names}


def optional(name: str, env: Mapping[str, str] = os.environ) -> str | None:
    """Return a deliberately optional input, or ``None`` when unset or empty."""
    return env.get(name) or None


def flag(name: str, env: Mapping[str, str] = os.environ) -> bool:
    """Parse a required ``true``/``false`` input."""
    value = require(name, env=env)[name]
    if value not in {"true", "false"}:
        raise InputError(f"{name} must be true or false, got {value!r}")
    return value == "true"


def positive_int(name: str, env: Mapping[str, str] = os.environ) -> int:
    """Parse a required positive decimal integer input."""
    return parse_positive_int(name, require(name, env=env)[name])


def parse_positive_int(name: str, value: str) -> int:
    """Parse ``value`` as a positive decimal integer named ``name`` in errors."""
    if not re.fullmatch(r"[1-9][0-9]*", value):
        raise InputError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def non_negative_int(name: str, env: Mapping[str, str] = os.environ) -> int:
    """Parse a required non-negative decimal integer input."""
    value = require(name, env=env)[name]
    if not re.fullmatch(r"[0-9]+", value):
        raise InputError(f"{name} must be a non-negative integer, got {value!r}")
    return int(value)
