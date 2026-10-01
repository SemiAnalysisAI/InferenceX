"""Fail a GPU dispatch whose measured checkout predates ``python -m infx.launch``.

GPU jobs run the measured revision's own launcher; without one they would hold a runner
only to fail.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ENTRY_POINT = "inferencex-e2e/infx/launch/__main__.py"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path, help="repository checkout of the measured revision")
    checkout = parser.parse_args(argv).checkout
    if (checkout / ENTRY_POINT).is_file():
        return 0
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    print(
        f"::error::Revision {commit or checkout} predates the Python launcher (infx/launch); "
        "GPU jobs launch only revisions that contain it. Choose a newer ref.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
