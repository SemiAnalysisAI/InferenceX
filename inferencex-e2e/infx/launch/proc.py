"""Subprocesses echoed like bash xtrace (``infx.bench.proc.echo`` masks their secrets)."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from infx.bench.proc import echo


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
