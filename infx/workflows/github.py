"""One repository metadata read using the runner's existing GitHub CLI."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from urllib.parse import quote


class PermissionLookupError(RuntimeError):
    """The requester's repository access could not be established."""


@dataclass(frozen=True)
class RepositoryPermission:
    permission: str
    role_name: str


def get_repository_permission(
    repository: str, actor: str, token: str,
) -> RepositoryPermission:
    """Use GITHUB_TOKEN's metadata access, without falling back to local auth."""
    parts = repository.split("/")
    if len(parts) != 2 or any(part in ("", ".", "..") for part in parts):
        raise PermissionLookupError("invalid-repository")
    if not actor or actor in (".", ".."):
        raise PermissionLookupError("missing-actor")
    if not token:
        raise PermissionLookupError("missing-token")
    owner, repo = (quote(part, safe="") for part in parts)
    endpoint = f"repos/{owner}/{repo}/collaborators/{quote(actor, safe='')}/permission"
    try:
        response = subprocess.run(
            ["gh", "api", "--hostname", "github.com", endpoint],
            env={**os.environ, "GH_TOKEN": token},
            capture_output=True, text=True, check=True, timeout=15,
        )
        data = json.loads(response.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        # gh stderr and transport errors can contain credentials or response data.
        raise PermissionLookupError("repository-permission-lookup-failed") from exc
    if not isinstance(data, dict) or any(
        not isinstance(data.get(key), str) or not data[key]
        for key in ("role_name", "permission")
    ):
        raise PermissionLookupError("invalid-repository-permission-response")
    return RepositoryPermission(data["permission"], data["role_name"])
