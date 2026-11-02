"""Publish staging progress and exact Pareto-line links on the source PR."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlsplit

from infx import github

STAGING_SITE = "https://inferencemax-app-git-staging-semianalysisai.vercel.app"


def chart_links(payload: dict[str, Any], run_id: str, run_date: str) -> list[str]:
    """Return markdown chart links from payload, or a default link for run_id on run_date.

    Raises RuntimeError if any link is malformed or not on the staging site.
    """
    links = payload.get("chart-links")
    if links is None:
        return [f"{STAGING_SITE}/inference?i_dates={quote(f'{run_date}~r{run_id}')}"]
    if not isinstance(links, list):
        raise RuntimeError("Callback chart-links must be a list")
    result = []
    for link in links:
        if not isinstance(link, dict):
            raise RuntimeError("Each chart link must contain a label and URL")
        label, url = link.get("label"), link.get("url")
        if not isinstance(label, str) or not label.strip() or any(c in label for c in "\r\n"):
            raise RuntimeError("Chart link labels must be nonempty single-line strings")
        if not isinstance(url, str):
            raise RuntimeError("Chart link URLs must be strings")
        parsed = urlsplit(url)
        query = parse_qs(parsed.query)
        if (
            f"{parsed.scheme}://{parsed.netloc}" != STAGING_SITE
            or parsed.path != "/inference"
            or parsed.fragment
            or not all(query.get(key) for key in ("g_model", "i_seq", "i_prec", "i_gpus"))
            or len(query.get("i_dates", [])) != 1
            or not re.fullmatch(rf"[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}~r{run_id}", query["i_dates"][0])
        ):
            raise RuntimeError("Chart link must select a Pareto line from the staged run")
        escaped = label.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
        result.append(f"- [{escaped}]({quote(url, safe=':/?&=%~+-._')})")
    return result


def report(repo: str, event: dict[str, Any], token: str, server_url: str) -> None:
    """Post or update the PR comment for a staging callback event.

    Raises RuntimeError if the payload is malformed or from an unexpected source.
    """
    payload = event["client_payload"]
    if payload.get("source-repository") != "SemiAnalysisAI/InferenceX-app":
        raise RuntimeError("Unsupported staging callback source")
    run_id, pr_number, app_run_id = (
        str(payload.get(key, "")) for key in ("run-id", "pr-number", "app-run-id")
    )
    if not all(re.fullmatch(r"[0-9]+", value) for value in (run_id, pr_number, app_run_id)):
        raise RuntimeError("Callback run and PR identifiers must be numeric")
    comment_id = str(payload.get("comment-id") or "")
    if comment_id and not re.fullmatch(r"[0-9]+", comment_id):
        raise RuntimeError("Callback comment identifier must be numeric")
    run_date = str(payload.get("run-date", ""))
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", run_date):
        raise RuntimeError("Callback run date must use YYYY-MM-DD")
    requester = str(payload.get("requested-by", ""))
    if not re.fullmatch(r"[A-Za-z0-9-]+", requester):
        raise RuntimeError("Callback requester is not a valid GitHub login")
    callback_type = event.get("action")
    if callback_type not in {"stage-results-started", "stage-results-completed"}:
        raise RuntimeError("Unsupported staging callback type")
    succeeded = payload.get("stage-succeeded")
    if callback_type == "stage-results-completed" and succeeded not in {"true", "false"}:
        raise RuntimeError("Callback stage-succeeded value must be true or false")

    app_url = f"{server_url}/SemiAnalysisAI/InferenceX-app/actions/runs/{app_run_id}"
    source_url = f"{server_url}/{repo}/actions/runs/{run_id}"
    if callback_type == "stage-results-started":
        body = (
            f"@{requester} staging [run {run_id}]({source_url}). "
            f"[InferenceX-app staging workflow]({app_url}) is now running. "
            "Existing staged runs will be preserved; this comment will be updated when staging completes."
        )
    elif succeeded == "true":
        links = chart_links(payload, run_id, run_date)
        body = "\n".join(
            [
                f"@{requester} staged [run {run_id}]({source_url}).",
                "",
                *(links or ["This run has no staged performance curves."]),
                "",
                (
                    "This run remains available across future `/stage-results` requests. "
                    "Staging the same run ID again updates its staged data. "
                    f"[Staging workflow]({app_url})"
                ),
            ]
        )
    else:
        body = (
            f"@{requester} staging run {run_id} failed. [Inspect the staging workflow]({app_url})."
        )
    path = f"/issues/comments/{comment_id}" if comment_id else f"/issues/{pr_number}/comments"
    github.api(repo, path, token, method="PATCH" if comment_id else "POST", data={"body": body})


def main() -> None:
    """Report the staging callback from the GitHub Actions environment."""
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
    report(
        os.environ["GITHUB_REPOSITORY"],
        event,
        os.environ["GH_TOKEN"],
        os.environ["GITHUB_SERVER_URL"],
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
