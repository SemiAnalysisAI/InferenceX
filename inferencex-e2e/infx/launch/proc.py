"""Echoed subprocess execution, like bash xtrace, with secrets redacted from the echo."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

REDACTED = "***"
_SECRET_NAME = re.compile(r"TOKEN|SECRET", re.IGNORECASE)
# Shorter values are not credentials and would mangle unrelated argv text.
_MIN_SECRET_LENGTH = 4


def secret_values(env: Mapping[str, str] | None = None) -> list[str]:
    """Return values of secret-named variables (HF_TOKEN, MODAL_TOKEN_*, *TOKEN*, *SECRET*).

    Both the process environment and ``env`` are inspected, longest value first so a
    secret containing another secret is redacted whole.
    """
    values = {
        value
        for source in (os.environ, env or {})
        for name, value in source.items()
        if _SECRET_NAME.search(name) and len(value) >= _MIN_SECRET_LENGTH
    }
    return sorted(values, key=len, reverse=True)


def redact(text: str, env: Mapping[str, str] | None = None) -> str:
    """Replace every secret value from :func:`secret_values` in ``text``."""
    for value in secret_values(env):
        text = text.replace(value, REDACTED)
    return text


def echo(argv: Sequence[str | os.PathLike[str]], env: Mapping[str, str] | None = None) -> None:
    """Print ``+ <argv>`` to stderr with secrets redacted, like bash xtrace."""
    sys.stdout.flush()
    print(f"+ {redact(shlex.join(map(os.fspath, argv)), env)}", file=sys.stderr, flush=True)


def run(
    argv: Sequence[str | os.PathLike[str]],
    *,
    env: Mapping[str, str] | None = None,
    check: bool = False,
    capture: bool = False,
    cwd: str | Path | None = None,
    input: str | None = None,  # noqa: A002 - mirrors subprocess.run
    echo_command: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Echo ``argv`` and run it.

    ``env`` replaces the child environment (subprocess semantics); pass
    ``{**os.environ, ...}`` to extend it. Output streams straight to this process's
    stdout/stderr unless ``capture`` is set, in which case both are returned as text.
    ``input`` is fed to stdin (``""`` detaches stdin like ``</dev/null``); otherwise
    stdin is inherited. ``echo_command=False`` silences the echo for tight poll loops.
    A missing executable raises ``FileNotFoundError``; ``check`` raises
    ``CalledProcessError`` on a nonzero exit.
    """
    if echo_command:
        echo(argv, env)
    sys.stderr.flush()
    return subprocess.run(
        [os.fspath(arg) for arg in argv],
        env=None if env is None else dict(env),
        check=check,
        capture_output=capture,
        text=True,
        cwd=cwd,
        input=input,
    )
