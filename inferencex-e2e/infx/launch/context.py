"""What a driver receives: the launch it runs, and the error it reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from infx.clusters import Cluster
    from infx.launch.backends.base import Backend
    from infx.launch.lifecycle import Lifecycle
    from infx.launch.policy import LaunchPath
    from infx.launch.request import LaunchRequest


class LaunchError(RuntimeError):
    """A launch input or cluster precondition is wrong; reported without a traceback."""


@dataclass(frozen=True)
class Launch:
    """One launch: what runs (``request`` on ``path``), where (``cluster`` through
    ``backend``), and the owner of its cleanups (``life``)."""

    cluster: Cluster
    backend: Backend
    request: LaunchRequest
    life: Lifecycle
    path: LaunchPath
