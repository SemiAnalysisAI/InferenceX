"""Versioned local contracts, deliberately separate from external API schemas."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc(value: str | datetime) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if result.tzinfo is None:
        raise ValueError("timestamp requires a timezone")
    return result.astimezone(timezone.utc)


def stamp(value: datetime) -> str:
    return utc(value).isoformat().replace("+00:00", "Z")


def identity(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True,
                              alias_generator=lambda key: key.replace("_", "-"))


class Policy(Contract):
    # Response freshness is separate from the API's configured cluster staleness policy.
    response_max_age_seconds: int = Field(default=120, gt=0, le=120)
    # Bounds reuse of a local fetch, not the server/CDN's Age header.
    public_max_age_seconds: int = Field(default=7200, gt=0)
    clock_skew_seconds: int = Field(default=5, ge=0, le=30)


class CandidateReview(Contract):
    candidate_id: str = Field(pattern=r"^[0-9a-f]{16}-[0-9a-f]{16}$")
    decision: Literal["proceed", "duplicate", "uncertain"]
    family: str | None = Field(pattern=r"^configs/[^/:]+-master\.yaml:[^\s:]+$")
    telemetry_clusters: list[Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9._+-]{0,63}$")]]
    pull_requests: list[Annotated[int, Field(gt=0)]]
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def consistent_decision(self) -> CandidateReview:
        if self.decision == "proceed" and (not self.family or not self.telemetry_clusters or self.pull_requests):
            raise ValueError("proceed requires a resolved family, exact telemetry clusters and no overlapping PRs")
        if len(set(self.telemetry_clusters)) != len(self.telemetry_clusters):
            raise ValueError("telemetry clusters must be distinct")
        if self.decision == "duplicate" and not self.pull_requests:
            raise ValueError("duplicate requires an overlapping PR number")
        return self


class PRReview(Contract):
    decisions: list[CandidateReview]


class Feed(Contract):
    url: str
    retrieved_at: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload: Any
    headers: dict[str, str] = Field(default_factory=dict)
    error: str | None = None

    @field_validator("retrieved_at")
    @classmethod
    def valid_timestamp(cls, value: str) -> str:
        utc(value)
        return value


class PublicRow(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)
    model: str = Field(min_length=1)
    hardware: str = Field(min_length=1)
    framework: str = Field(min_length=1)
    precision: str = Field(min_length=1)
    spec_method: str = Field(min_length=1)
    disagg: bool
    benchmark_type: str = Field(min_length=1)
    isl: int | None = Field(ge=1)
    osl: int | None = Field(ge=1)
    image: str = Field(min_length=1)
    date: str

    @field_validator("date")
    @classmethod
    def valid_date(cls, value: str) -> str:
        from datetime import date
        if date.fromisoformat(value).isoformat() != value:
            raise ValueError("expected YYYY-MM-DD")
        return value
