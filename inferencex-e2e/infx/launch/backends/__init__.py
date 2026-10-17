"""Scheduler backends by ``scheduler:`` name.

Imported on first use, so reading cluster records never imports a backend or its clients.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from infx.launch.backends.base import Backend

BACKENDS: dict[str, str] = {"slurm": "infx.launch.backends.slurm:SlurmBackend"}


def backend_class(scheduler: str) -> type[Backend]:
    module, _, name = BACKENDS[scheduler].partition(":")
    return getattr(importlib.import_module(module), name)
