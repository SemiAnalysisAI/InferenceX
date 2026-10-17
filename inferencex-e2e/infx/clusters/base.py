"""Bases of every cluster record, shared by the neutral fields and each scheduler's sub-record."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Visibility = Literal["shared", "node-local"]


class Record(BaseModel):
    """Base of every runners.yaml record: unknown keys fail, values are frozen."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class Volume(Record):
    """A named checkpoint root or cache; each scheduler's subclass says how its jobs reach it."""

    visibility: Visibility = "shared"


class SchedulerSettings(Record):
    """Base of a ``<scheduler>:`` sub-record; subclasses narrow ``volumes`` to their own Volume."""

    volumes: Mapping[str, Volume] = Field(default_factory=dict)
