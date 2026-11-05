"""The ``native:`` sub-record of a ``scheduler: native`` cluster, and its image references."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from infx.clusters.base import HostPath, HostVolumeSettings

if TYPE_CHECKING:
    from pathlib import Path

    from infx.clusters import Cluster

DIGEST_SEPARATOR = "@sha256:"


def split_image(image: str) -> tuple[str, str]:
    """The URL and sha256 hex digest of a native image reference, ``<archive url>@sha256:<hex>``.

    Raises ValueError for any other reference.
    """
    url, separator, digest = image.rpartition(DIGEST_SEPARATOR)
    if not separator or not url or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"native images are <archive url>{DIGEST_SEPARATOR}<hex>: {image!r}")
    return url, digest


class NativeSettings(HostVolumeSettings):
    """Host directories of a cluster whose jobs run on the runner host without containers."""

    root: HostPath

    @property
    def images(self) -> Path:
        """The directory holding one unpacked image per archive digest."""
        return self.root / "images"

    @property
    def jobs(self) -> Path:
        """The directory holding each runner's job record and job directory, and the host lock."""
        return self.root / "jobs"


def native_settings(cluster: Cluster) -> NativeSettings:
    """The native sub-record of ``cluster``; a cluster on another scheduler is a caller bug."""
    if not isinstance(cluster.scheduler_settings, NativeSettings):
        raise TypeError(f"cluster {cluster.id!r} is not a native cluster")
    return cluster.scheduler_settings
