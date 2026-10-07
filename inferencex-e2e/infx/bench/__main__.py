"""``python3 -m infx.bench <command>``: container-side clients, each loaded on first use."""

from __future__ import annotations

import importlib
import os
import sys

from infx.bench.env import BenchError

COMMANDS = {
    "wait": "infx.bench.server",
    "fixed-seq": "infx.bench.fixed_seq",
    "agentic": "infx.bench.agentic.run",
    "eval": "infx.bench.eval",
}


def main(argv: list[str]) -> int:
    """Dispatch ``argv[0]``; a ``BenchError`` is one ``ERROR:`` line, not a traceback."""
    if not argv or argv[0] not in COMMANDS:
        print(f"usage: python3 -m infx.bench {{{','.join(COMMANDS)}}} [args]", file=sys.stderr)
        return 2
    # Only this interpreter needs it; child scripts that import their siblings break under it.
    os.environ.pop("PYTHONSAFEPATH", None)
    command = importlib.import_module(COMMANDS[argv[0]])
    try:
        return command.main(argv[1:])
    except BenchError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
