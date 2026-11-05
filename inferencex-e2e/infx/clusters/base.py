"""Bases of every cluster record, shared by the neutral fields and each scheduler's sub-record."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

Visibility = Literal["shared", "node-local"]


def _absolute(path: Path) -> Path:
    """Expand a runner-home ``~`` and require an absolute host path."""
    path = path.expanduser()
    if not path.is_absolute():
        raise ValueError(f"cluster path must be absolute: {path}")
    return path


HostPath = Annotated[Path, AfterValidator(_absolute)]


class Record(BaseModel):
    """Base of every runners.yaml record: unknown keys fail, values are frozen."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class Volume(Record):
    """A named checkpoint root or cache; each scheduler's subclass says how its jobs reach it."""

    visibility: Visibility = "shared"


class HostVolume(Volume):
    """A directory on the cluster's storage, at ``path`` on the launching host and in jobs."""

    path: HostPath


class SchedulerSettings(Record):
    """Base of a ``<scheduler>:`` sub-record; subclasses narrow ``volumes`` to their own Volume."""

    volumes: Mapping[str, Volume] = Field(default_factory=dict)


class HostVolumeSettings(SchedulerSettings):
    """Scheduler facts of a cluster whose volumes are directories on the launching host."""

    volumes: dict[str, HostVolume] = Field(default_factory=dict)

    def path(self, volume: str) -> Path | None:
        """Host path of volume ``volume``, or None when the cluster declares no such volume."""
        declared = self.volumes.get(volume)
        return None if declared is None else declared.path
