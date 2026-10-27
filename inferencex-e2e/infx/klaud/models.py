"""Versioned local contracts, deliberately separate from external API schemas."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator


def utc(value: str | datetime) -> datetime:
    """Return an ISO timestamp or aware datetime as a UTC datetime, raising ValueError if naive."""
    result = (
        datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    )
    if result.tzinfo is None:
        raise ValueError("timestamp requires a timezone")
    return result.astimezone(UTC)


def stamp(value: datetime) -> str:
    """Return an aware datetime as an ISO UTC string ending in ``Z``."""
    return utc(value).isoformat().replace("+00:00", "Z")


def identity(value: Any) -> str:
    """Return the SHA-256 hex digest of ``value`` as canonical JSON."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def normalized_image(value: str) -> str:
    """Return image ``value`` with a ``registry#path`` separator turned into ``/``.

    Tags, digests and repositories are left unchanged.
    """
    registry, separator, remainder = value.partition("#")
    if separator and remainder and ("." in registry or ":" in registry):
        return registry + "/" + remainder
    return value


def family_keys(family: str) -> list[str]:
    """Return the master config keys listed by ``family`` (``configs/x-master.yaml:KEY,...``)."""
    return family.split(":", 1)[1].split(",")


def key_families(family: str) -> set[str]:
    """Return the single-key family of each key listed by ``family``."""
    source = family.split(":", 1)[0]
    return {f"{source}:{key}" for key in family_keys(family)}


def sorted_keys(family: str) -> str:
    """Return ``family`` unchanged, raising ValueError unless its keys are sorted and distinct."""
    keys = family_keys(family)
    if keys != sorted(set(keys)):
        raise ValueError("family keys must be sorted and distinct")
    return family


Family = Annotated[
    str,
    Field(pattern=r"^configs/[^/:]+-master\.yaml:[^\s:,]+(?:,[^\s:,]+)*$"),
    AfterValidator(sorted_keys),
]


class Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        populate_by_name=True,
        alias_generator=lambda key: key.replace("_", "-"),
    )


class Policy(Contract):
    response_max_age_seconds: int = Field(default=120, gt=0, le=120)
    public_max_age_seconds: int = Field(default=7200, gt=0)
    clock_skew_seconds: int = Field(default=5, ge=0, le=30)
    scheduler_max_age_seconds: int = Field(default=300, gt=0, le=900)


class CandidateReview(Contract):
    candidate_id: str = Field(pattern=r"^[0-9a-f]{16}-[0-9a-f]{16}$")
    decision: Literal["proceed", "duplicate", "uncertain"]
    family: Family | None
    pull_requests: list[Annotated[int, Field(gt=0)]]
    baseline_model: str | None = Field(default=None, min_length=1)
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def consistent_decision(self) -> CandidateReview:
        """Return the review, raising ValueError if the decision contradicts its fields."""
        if self.decision == "proceed" and (
            not self.family or self.pull_requests or not self.baseline_model
        ):
            raise ValueError(
                "proceed requires a resolved family, baseline model and no overlapping PRs"
            )
        if self.decision == "duplicate" and not self.pull_requests:
            raise ValueError("duplicate requires an overlapping PR number")
        return self


class PRReview(Contract):
    decisions: list[CandidateReview]


class OwnedCandidate(Contract):
    id: str = Field(pattern=r"^[0-9a-f]{16}-[0-9a-f]{16}$")
    family: Family
    base: str = Field(pattern=r"^[0-9a-f]{40}$")


class Ownership(Contract):
    run_id: int = Field(gt=0)
    candidates: list[OwnedCandidate] = Field(max_length=256)


ASSET_REASONS = ("weights-missing", "route-mismatch")
INFRA_REASONS = ("infra-registry-rate-limit", "infra-hf-rate-limit", "infra-node-fault")

LEGACY_OUTCOMES = {
    "readiness-blocked": "missing-assets",
    "uncertain": "inconclusive",
    "unexpected-error": "no-outcome",
    "capacity-deferred": "no-capacity",
    "aborted": "closed-by-maintainer",
}


class CandidateOutcome(Contract):
    outcome: Literal[
        "validated",
        "no-capacity",
        "missing-assets",
        "incompatible",
        "duplicate",
        "retired",
        "already-updated",
        "inconclusive",
        "failed",
        "closed-by-maintainer",
        "no-outcome",
    ]
    phase: Literal["resolve", "baseline", "targeted", "final-sweep", "cleanup", "unknown"]
    pull_request: Annotated[int, Field(gt=0)] | None
    run_ids: list[Annotated[int, Field(gt=0)]] = Field(max_length=256)
    repairs_used: int | None = Field(ge=0)
    reason_code: (
        Literal[
            "baseline-preflight-mismatch",
            "baseline-provenance-unverified",
            "baseline-point-mismatch",
            "baseline-eval-unverified",
            "baseline-api-unavailable",
            "baseline-other",
            "weights-missing",
            "route-mismatch",
            "infra-registry-rate-limit",
            "infra-hf-rate-limit",
            "infra-node-fault",
        ]
        | None
    ) = None

    @model_validator(mode="before")
    @classmethod
    def current_names(cls, data: Any) -> Any:
        """Return ``data`` with an outcome name from older PR comments and artifacts renamed."""
        if isinstance(data, dict) and data.get("outcome") in LEGACY_OUTCOMES:
            return {**data, "outcome": LEGACY_OUTCOMES[data["outcome"]]}
        return data

    @model_validator(mode="after")
    def consistent_outcome(self) -> CandidateOutcome:
        """Return the outcome, raising ValueError if phase, PR, runs or reason code disagree."""
        if self.outcome != "no-outcome" and self.phase == "unknown":
            raise ValueError("Completed outcomes require a known phase")
        if self.outcome in ("validated", "closed-by-maintainer") and self.pull_request is None:
            raise ValueError("This outcome requires a PR")
        if self.outcome == "validated" and (not self.run_ids or self.phase != "final-sweep"):
            raise ValueError("Validated requires final-sweep evidence")
        if len(set(self.run_ids)) != len(self.run_ids):
            raise ValueError("Run IDs must be distinct")
        if self.reason_code in ASSET_REASONS:
            if self.outcome != "missing-assets":
                raise ValueError("Asset reason codes require a missing-assets outcome")
        elif self.reason_code in INFRA_REASONS:
            if self.outcome not in ("failed", "inconclusive") or self.phase not in (
                "resolve",
                "targeted",
                "final-sweep",
            ):
                raise ValueError("Infrastructure reason codes require a failed attempt phase")
        elif self.reason_code is not None and (
            self.outcome != "failed" or self.phase != "baseline"
        ):
            raise ValueError("Baseline reason codes require a failed baseline outcome")
        return self


class SessionEnd(Contract):
    """How a candidate session ended: finished with a verified outcome, or parked on its runs."""

    status: Literal["finished", "parked"]
    outcome: CandidateOutcome | None = None

    @model_validator(mode="after")
    def outcome_only_when_finished(self) -> SessionEnd:
        """Return the record, raising ValueError unless only a finished session has an outcome."""
        if (self.status == "finished") != (self.outcome is not None):
            raise ValueError("A finished session reports its outcome; a parked one reports none")
        return self


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
        """Return ``value`` unchanged, raising ValueError unless it is an aware ISO timestamp."""
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
        """Return ``value`` unchanged, raising ValueError unless it is a YYYY-MM-DD date."""
        from datetime import date

        if date.fromisoformat(value).isoformat() != value:
            raise ValueError("expected YYYY-MM-DD")
        return value
