"""Workflow adapter: JSON on stdout; exit 0 allowed, 1 denied, 2 unavailable."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping

from .authorization import authorize
from .github import PermissionLookupError, get_repository_permission


def requester(operation: str, event: dict, env: Mapping[str, str]) -> str:
    """Use the original requester, never the person re-running a job."""
    expected_event, action = {
        "stage-results": ("issue_comment", "created"),
        "trusted-external-sweep": ("pull_request_target", "labeled"),
    }[operation]
    if env.get("GITHUB_EVENT_NAME") != expected_event or event.get("action") != action:
        raise ValueError("unexpected-event")
    actor = (
        event["comment"]["user"]["login"]
        if operation == "stage-results"
        else env["GITHUB_ACTOR"]
    )
    if not isinstance(actor, str) or not actor.strip():
        raise ValueError("missing-actor")
    return actor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["authorize"])
    parser.add_argument("operation")
    args = parser.parse_args(argv)
    decision = authorize(args.operation, "none")
    if decision.reason == "unknown-operation":
        print(json.dumps({"allowed": False, "reason": decision.reason}))
        return 1
    try:
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        actor = requester(args.operation, event, os.environ)
        permission = get_repository_permission(
            os.environ["GITHUB_REPOSITORY"], actor, os.environ.get("GITHUB_TOKEN", ""),
        )
    except PermissionLookupError as exc:
        print(json.dumps({"allowed": False, "reason": str(exc)}))
        return 2
    except (OSError, KeyError, ValueError, TypeError, AttributeError):
        print(json.dumps({"allowed": False, "reason": "invalid-workflow-context"}))
        return 2
    decision = authorize(args.operation, permission.permission, permission.role_name)
    print(json.dumps({
        "allowed": decision.allowed,
        "reason": decision.reason,
        "tier": decision.tier.name if decision.tier is not None else None,
        "actor": actor,
        "role": permission.role_name,
        "permission": permission.permission,
    }))
    return 0 if decision.allowed else 1


if __name__ == "__main__":
    raise SystemExit(main())
