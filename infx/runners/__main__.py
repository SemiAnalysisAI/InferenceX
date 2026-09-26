"""CLI entry point: ``python3 -m infx.runners <cluster>``.

Dispatches to the Python-ported launcher for the given cluster name (the
``RUNNER_NAME%%_*`` prefix).  Exits non-zero on unknown clusters or launcher
failures.

Usage::

    python3 -m infx.runners h100-dgxc-slurm
"""

from __future__ import annotations

import importlib
import logging
import sys

from infx.runners.clusters import PORTED_CLUSTERS

# TODO: pyslurm build integration — before any pyslurm call, ensure the
# extension is importable.  The companion ``infx/runners/pyslurm_build.py``
# (being added on the ``klaud/runners-python-pyslurm`` branch) compiles
# pyslurm from ``third_party/pyslurm/`` using vendored Slurm headers.
# Accept ``INFERENCEX_PYSLURM_PATH`` to prepend to ``sys.path``; when set,
# do ``sys.path.insert(0, path)`` before any import that reaches pyslurm.
_PYSLURM_PATH_VAR = "INFERENCEX_PYSLURM_PATH"


def _setup_pyslurm_path() -> None:
    """Insert the pyslurm build output directory into sys.path if configured."""
    import os

    path = os.environ.get(_PYSLURM_PATH_VAR, "")
    if path and path not in sys.path:
        sys.path.insert(0, path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: python3 -m infx.runners <cluster>", file=sys.stderr)
        print(f"Ported clusters: {', '.join(sorted(PORTED_CLUSTERS))}", file=sys.stderr)
        sys.exit(1)

    cluster = sys.argv[1]
    module_name = PORTED_CLUSTERS.get(cluster)
    if module_name is None:
        print(
            f"ERROR: cluster {cluster!r} is not ported to Python. "
            f"Ported clusters: {', '.join(sorted(PORTED_CLUSTERS))}",
            file=sys.stderr,
        )
        sys.exit(1)

    _setup_pyslurm_path()

    module = importlib.import_module(module_name)
    launch_fn = getattr(module, "launch", None)
    if launch_fn is None:
        print(
            f"ERROR: module {module_name} does not define a launch() function",
            file=sys.stderr,
        )
        sys.exit(1)

    rc = launch_fn()
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
