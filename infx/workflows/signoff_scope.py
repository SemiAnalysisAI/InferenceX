from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

from codeowners import CodeOwners

from infx import github
from infx.workflows.reuse import write_outputs


def required_owners(repo: str, pr: dict[str, Any], token: str) -> list[str]:
    files = github.paginate(repo, f"/pulls/{pr['number']}/files", token, "")
    # GitHub lists a symlink-to-file change as removed and added entries for
    # one path, while changed_files counts that path once. Keep every entry
    # below so ownership checks still include all previous_filename values.
    if len({file["filename"] for file in files}) != pr["changed_files"]:
        raise RuntimeError("Incomplete changed-file list; cannot determine sign-off scope")
    # A PR's recorded base SHA can predate ownership fixes on its target branch.
    # Resolve the current trusted base branch once, then pin both reads to it.
    branch = github.api(repo, f"/branches/{quote(pr['base']['ref'], safe='')}", token)
    params = {"ref": branch["commit"]["sha"]}
    errors = github.api(repo, "/codeowners/errors", token, params)
    if errors["errors"]:
        raise RuntimeError("The base CODEOWNERS file has errors; cannot determine sign-off scope")
    content = github.api(repo, "/contents/.github/CODEOWNERS", token, params)
    if content.get("type") != "file" or content.get("encoding") != "base64":
        raise RuntimeError("The base CODEOWNERS file is unavailable")
    text = base64.b64decode(content["content"]).decode("utf-8")
    if not text.strip():
        raise RuntimeError("The base CODEOWNERS file is empty")
    owners = CodeOwners(text)
    matched = set()
    for file in files:
        paths = [file["filename"]]
        if file.get("previous_filename"):
            paths.append(file["previous_filename"])
        for path in paths:
            matched.update((kind, name.lower()) for kind, name in owners.of(path))
    required = []
    for kind, name in sorted(matched):
        if name == "@semianalysisai/core":
            continue
        if kind == "USERNAME":
            permission = github.api(
                repo, f"/collaborators/{quote(name[1:], safe='')}/permission", token
            )
            if permission.get("permission") == permission.get("role_name") == "admin":
                continue
        required.append(name)
    return required


def check_scope(repo: str, number: int, token: str) -> dict[str, str]:
    pr = github.api(repo, f"/pulls/{number}", token)
    status = "error"
    try:
        required = required_owners(repo, pr, token)
        current = github.api(repo, f"/pulls/{number}", token)
        if (
            current["head"]["sha"],
            current["base"]["sha"],
            current["base"]["ref"],
            current["changed_files"],
        ) != (
            pr["head"]["sha"],
            pr["base"]["sha"],
            pr["base"]["ref"],
            pr["changed_files"],
        ):
            raise RuntimeError(
                "PR changed while determining sign-off scope; retry on the current head"
            )
        status = None if required else "success"
    finally:
        if status:
            github.api(
                repo,
                f"/statuses/{pr['head']['sha']}",
                token,
                method="POST",
                data={
                    "context": "CODEOWNER sign-off",
                    "state": status,
                    "description": "N/A"
                    if status == "success"
                    else "Could not determine sign-off scope",
                },
            )
    return {
        "required": str(bool(required)).lower(),
        "pr-number": str(number),
        "head-sha": pr["head"]["sha"],
    }


def main() -> None:
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    number = (event.get("pull_request") or event.get("issue") or {}).get("number")
    if number is None:
        inputs = event.get("inputs", {})
        number = inputs.get("pr-number")
        match = re.search(r"/pull/(\d+)", inputs.get("comment_url", ""))
        if match is None or str(number) != match[1] or int(number) < 1:
            raise ValueError("pr-number must match the pull request in comment_url")
        number = int(number)
    outputs = check_scope(os.environ["GITHUB_REPOSITORY"], number, os.environ["GH_TOKEN"])
    write_outputs(os.environ["GITHUB_OUTPUT"], outputs)


if __name__ == "__main__":
    main()
