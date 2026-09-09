"""Authorization shared by trusted GitHub workflow entrypoints."""

from .authorization import Decision, Tier, authorize

__all__ = ["Decision", "Tier", "authorize"]
