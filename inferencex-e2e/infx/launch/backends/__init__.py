"""Scheduler backends, keyed by the ``scheduler:`` value of a cluster record.

Each entry names the :class:`~infx.launch.backends.base.Backend` implementation as
``module:Class``, imported on first use: reading cluster records needs only the
settings models in :data:`infx.clusters.SCHEDULERS`, so it never imports a backend
(or the client libraries a backend imports).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from infx.launch.backends.base import Backend

BACKENDS: dict[str, str] = {"slurm": "infx.launch.backends.slurm:SlurmBackend"}


def backend_class(scheduler: str) -> type[Backend]:
    """Import and return the backend registered for ``scheduler``."""
    module, _, name = BACKENDS[scheduler].partition(":")
    return getattr(importlib.import_module(module), name)
