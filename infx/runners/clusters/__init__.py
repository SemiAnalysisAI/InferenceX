"""Per-cluster launcher modules.

Each module exposes a ``launch()`` function that mirrors the bash launcher for
that Slurm pool.  The cluster name (the ``RUNNER_NAME%%_*`` prefix) maps to a
Python module via :data:`PORTED_CLUSTERS`.
"""

from __future__ import annotations

# Clusters that have been ported to Python.  The workflow switch checks this
# mapping to decide whether to run ``python3 -m infx.runners <cluster>`` or
# fall back to the existing bash launcher.
PORTED_CLUSTERS: dict[str, str] = {
    "h100-dgxc-slurm": "infx.runners.clusters.h100_dgxc_slurm",
}
