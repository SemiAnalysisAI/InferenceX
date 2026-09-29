"""Subprocesses echoed like bash xtrace, with secret values masked in the echo."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

_SECRET_NAME = re.compile(r"TOKEN|SECRET", re.IGNORECASE)


def echo(argv: Sequence[str | os.PathLike[str]], env: Mapping[str, str] | None = None) -> None:
    """Print ``+ <argv>`` to stderr, masking the values of ``*TOKEN*`` and ``*SECRET*`` variables."""
    text = shlex.join(map(os.fspath, argv))
    # Longest first, so a secret containing another is masked whole. Values shorter than
    # four characters are not credentials and would mangle unrelated text.
    secrets = {
        value
        for source in (os.environ, env or {})
        for name, value in source.items()
        if _SECRET_NAME.search(name) and len(value) >= 4
    }
    for value in sorted(secrets, key=len, reverse=True):
        text = text.replace(value, "***")
    sys.stdout.flush()
    print(f"+ {text}", file=sys.stderr, flush=True)


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
    """Echo ``argv`` and run it. ``env`` replaces the child environment, as in subprocess.

    Output streams through unless ``capture`` is set; ``input=""`` detaches stdin.
    """
    if echo_command:
        echo(argv, env)
    sys.stderr.flush()
    return subprocess.run(
        [os.fspath(arg) for arg in argv],
        env=env,
        check=check,
        capture_output=capture,
        text=True,
        cwd=cwd,
        input=input,
    )
