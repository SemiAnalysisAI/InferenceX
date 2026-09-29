"""Bases of every cluster record, shared by the neutral fields and each scheduler's sub-record."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ``shared`` storage holds the same files on every node; ``node-local`` storage is a
# per-node copy that one node can lack.
Visibility = Literal["shared", "node-local"]


class Record(BaseModel):
    """Base of every runners.yaml record: unknown keys fail, values are frozen."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class Volume(Record):
    """One named cluster volume, in its scheduler's terms (``<scheduler>.volumes.<name>``).

    Schedulers extend it with how their jobs reach the volume: a host path, a
    persistent volume claim, ...
    """

    visibility: Visibility = "shared"


class SchedulerSettings(Record):
    """Base of a scheduler sub-record (``<scheduler>:`` in a cluster record).

    ``volumes`` are the cluster's checkpoint roots and caches by name; model entries and
    drivers refer to them only by name. Subclasses narrow the value type to their own
    :class:`Volume`.
    """

    volumes: Mapping[str, Volume] = Field(default_factory=dict)

    def model_references(self) -> Mapping[str, str]:
        """``models.entries`` keys this record names, keyed by where it names them."""
        return {}
