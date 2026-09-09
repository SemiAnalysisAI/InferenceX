"""Repository roles determine who may request each protected operation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Tier(IntEnum):
    PUBLIC = 0
    COLLABORATOR = 1
    MAINTAINER = 2


_ROLES = {
    "none": Tier.PUBLIC,
    "read": Tier.PUBLIC,
    "triage": Tier.PUBLIC,
    "write": Tier.COLLABORATOR,
    "maintain": Tier.MAINTAINER,
    "admin": Tier.MAINTAINER,
}
_OPERATIONS = {"stage-results", "trusted-external-sweep"}


@dataclass(frozen=True)
class Decision:
    allowed: bool
    tier: Tier | None
    reason: str


def authorize(operation: str, permission: str, role_name: str | None = None) -> Decision:
    """Require both the original base permission and a supported effective role."""
    tier = _ROLES.get(role_name)
    if operation not in _OPERATIONS:
        return Decision(False, tier, "unknown-operation")
    if tier is None:
        return Decision(False, None, "unknown-role")
    allowed = tier >= Tier.COLLABORATOR and permission in ("admin", "maintain", "write")
    return Decision(allowed, tier, "allowed" if allowed else "insufficient-role")
