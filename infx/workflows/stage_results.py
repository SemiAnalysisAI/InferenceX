"""Authorize a staging request and select its source sweep."""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

from infx import github

from . import sweep_runs

FULL_SWEEP_LABELS = (
    "full-sweep-enabled",
    "non-canary-full-sweep-enabled",
    "full-sweep-fail-fast",
    "full-sweep-fail-fast-no-canary",
)
COMMAND_SPACE = "\t\n\v\f\r \u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"


def timestamp(value: str | None) -> float:
    try:
        parsed = datetime.fromisoformat(value or "")
        return (parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)).timestamp()
    except ValueError:
        return float("inf")


def full_sweep_at(events: list[dict[str, Any]], created_at: str) -> bool:
    created = timestamp(created_at)
    if created == float("inf"):
        return False
    active = set()
    for event in sorted(events, key=lambda item: timestamp(item.get("created_at"))):
        if timestamp(event.get("created_at")) > created:
            break
        name = (event.get("label") or {}).get("name")
        if name in FULL_SWEEP_LABELS:
            if event["event"] == "labeled":
                active.add(name)
            elif event["event"] == "unlabeled":
                active.discard(name)
    return bool(active)


def resolve_run(
    repo: str, pr_number: int, pull: dict[str, Any], token: str, requested_id: str | None
) -> dict[str, Any]:
    timeline = github.paginate(repo, f"/issues/{pr_number}/timeline", token)
    commits = sweep_runs.pr_commit_shas(repo, pr_number, token)

    def rejection(run: dict[str, Any]) -> str:
        if (
            run.get("path") != ".github/workflows/run-sweep.yml"
            or run.get("event") != "pull_request"
        ):
            return "not a pull-request run of run-sweep.yml"
        associated = any(pr["number"] == pr_number for pr in run.get("pull_requests") or [])
        if run.get("head_sha") not in commits and not (requested_id is not None and associated):
            return "its head commit is not in the PR commit list"
        if not full_sweep_at(timeline, run.get("created_at", "")):
            return "it was not created while a full-sweep label was applied"
        conclusion = run.get("conclusion")
        if run.get("status") != "completed" or conclusion not in {
            "success",
            "cancelled",
            "failure",
        }:
            return f"it is {run.get('status')}/{'null' if conclusion is None else conclusion}"
        names = sweep_runs.artifact_names(repo, run["id"], token)
        if "changelog-metadata" not in names:
            return "it has no unexpired changelog-metadata artifact"
        if not sweep_runs.has_reusable_result_artifacts(names):
            return "it has no unexpired benchmark result artifacts"
        return ""

    if requested_id is not None:
        run = github.api(repo, f"/actions/runs/{int(requested_id)}", token)
        if reason := rejection(run):
            raise RuntimeError(f"Run {requested_id} cannot be staged because {reason}")
        return run
    for run in sweep_runs.completed_pr_runs(repo, "run-sweep.yml", pull["head"]["ref"], token):
        if not rejection(run):
            return run
    raise RuntimeError(
        "No stageable completed run-sweep.yml run with unexpired staging artifacts "
        f"was found for PR #{pr_number}"
    )


def request(repo: str, event: dict[str, Any], token: str) -> dict[str, str]:
    pr_number = event["issue"]["number"]
    actor = event["comment"]["user"]["login"]

    def reject(reason: str, comment: str) -> NoReturn:
        github.api(
            repo, f"/issues/{pr_number}/comments", token, method="POST", data={"body": comment}
        )
        raise RuntimeError(reason)

    command = re.fullmatch(
        rf"/stage-results(?:[{COMMAND_SPACE}]+([0-9]+))?",
        event["comment"]["body"].strip(COMMAND_SPACE),
    )
    if not command:
        reject(
            "Unsupported /stage-results syntax",
            "Usage: `/stage-results` or `/stage-results <run-id>`.",
        )
    access = github.api(repo, f"/collaborators/{actor}/permission", token)
    if not isinstance(access, dict) or any(
        not isinstance(access.get(key), str) or not access[key]
        for key in ("permission", "role_name")
    ):
        raise RuntimeError("Invalid repository permission response.")
    permission, role = access["permission"], access["role_name"]
    if any(value not in {"admin", "maintain", "write"} for value in (permission, role)):
        reason = (
            f'@{actor} has repository permission "{permission}" and role "{role}"; '
            "both must be write, maintain, or admin to stage results."
        )
        reject(
            reason,
            f'{reason}\n\n@{actor} 的仓库权限为 "{permission}"，角色为 "{role}"；'
            "两者都必须是 write、maintain 或 admin 才能暂存结果。",
        )
    pull = github.api(repo, f"/pulls/{pr_number}", token)
    labels = {label if isinstance(label, str) else label.get("name") for label in pull["labels"]}
    if not labels.intersection(FULL_SWEEP_LABELS):
        reject(
            "PR does not have a full-sweep label",
            f"@{actor} `/stage-results` requires a completed run from a PR using one of: "
            + ", ".join(f"`{label}`" for label in FULL_SWEEP_LABELS)
            + ".",
        )
    run = resolve_run(repo, pr_number, pull, token, command[1])
    attempt = run.get("run_attempt")
    return {
        "run-id": str(run["id"]),
        "run-attempt": str(1 if attempt is None else attempt),
        "run-date": run["created_at"][:10],
        "requested-by": actor,
        "run-url": run["html_url"],
    }


def main() -> None:
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
    outputs = request(os.environ["GITHUB_REPOSITORY"], event, os.environ["GH_TOKEN"])
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as handle:
        handle.writelines(f"{key}={value}\n" for key, value in outputs.items())


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
