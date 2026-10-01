"""Publish the advisory CODEOWNER verdict associated with one sign-off."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from infx import github

VERIFIER_AUTHORS = {"github-actions[bot]", "Klaud-Cold"}
SIGNOFF_KEY = re.compile(r"(?:issuecomment-|pullrequestreview-|discussion_r)[1-9][0-9]*")
SUCCESS_HEADER = "## ✅✅✅ **Verdict: PASS** ✅✅✅"
REJECT = "## ❌❌❌ **REJECTED** ❌❌❌"
WARN = "## ⚠️ **Verdict: WARN** ⚠️"
CHECK_ROW = re.compile(
    r"(?:[-*+] )?(✅|❌|➖|⚠️) Check ([0-9]+) \([^\r\n]+\): (PASS|FAIL|N/A|WARN) — \S.*"  # noqa: RUF001 - verifier's N/A emoji
)
CHECK_PREFIX = re.compile(r"(?:[-*+] )?(?:✅|❌|➖|⚠️)\s*Check\b")  # noqa: RUF001 - verifier's N/A emoji
STATUS_EMOJI = {"PASS": "✅", "FAIL": "❌", "N/A": "➖", "WARN": "⚠️"}  # noqa: RUF001 - verifier's N/A emoji
ESCALATION = (
    "⚠️ Pareto coverage needs additional review: @functionstackx @cquil11 @Oseltamivir @adibarra. "
    "At least 5 points per affected throughput-versus-E2EL frontier are highly recommended. "
    "Below 5, or when coverage cannot be verified, merge only with an explicit, recorded admin bypass "
    "for the assessed commit; this advisory comment does not grant or enforce a bypass."
)
INVALID = (
    f"{REJECT}\n\nThe verifier did not produce a valid verdict. Retry the sign-off verification."
)


def check_statuses(lines: list[str]) -> dict[int, str]:
    statuses = {}
    for line in lines:
        match = CHECK_ROW.fullmatch(line.strip())
        if match is None:
            if CHECK_PREFIX.match(line.strip()):
                return {}
            continue
        emoji, number, status = match.groups()
        check = int(number)
        allowed = {"PASS", "N/A", "WARN"} if check in {4, 14} else {"PASS", "N/A", "FAIL"}
        if check in statuses or status not in allowed or emoji != STATUS_EMOJI[status]:
            return {}
        statuses[check] = status
    return statuses if statuses.keys() == set(range(15)) else {}


def marker(signoff_key: str) -> str:
    if SIGNOFF_KEY.fullmatch(signoff_key) is None:
        raise ValueError("Invalid sign-off key")
    return f"<!-- codeowner-signoff-verify signoff={signoff_key} -->"


def _is_verdict_for(comment: dict[str, Any], signoff_key: str) -> bool:
    author = comment.get("user") or {}
    body = comment.get("body") or ""
    return author.get("login") in VERIFIER_AUTHORS and body.startswith(f"{marker(signoff_key)}\n")


def verdict_body(verdict: str, head_sha: str, signoff_key: str) -> tuple[str, str]:
    lines = verdict.splitlines()
    headers = [line for line in lines if line in (SUCCESS_HEADER, REJECT, WARN)]
    checks = check_statuses(lines)
    warning = "WARN" in checks.values()
    expected = REJECT if "FAIL" in checks.values() else WARN if warning else SUCCESS_HEADER
    valid = bool(checks) and headers == [expected] and lines[0] == expected
    if not valid:
        verdict = INVALID
    elif checks.get(14) == "WARN":
        header, _, rest = verdict.partition("\n")
        verdict = f"{header}\n\n{ESCALATION}\n\n{rest}"
    status = (
        "success"
        if verdict.startswith(SUCCESS_HEADER)
        else "warning"
        if verdict.startswith(WARN)
        else "failure"
    )
    return f"{marker(signoff_key)}\n{verdict}\n\nAssessed commit: `{head_sha}`.\n", status


def publish(
    repo: str,
    token: str,
    pr_number: int,
    head_sha: str,
    signoff_key: str,
    verdict_path: Path,
    *,
    verification_succeeded: bool,
) -> None:
    verdict = ""
    if verification_succeeded and verdict_path.exists():
        verdict = verdict_path.read_text(encoding="utf-8").strip()
    body, status = verdict_body(verdict, head_sha, signoff_key)
    comments = github.paginate(repo, f"/issues/{pr_number}/comments", token)
    matches = [comment for comment in comments if _is_verdict_for(comment, signoff_key)]
    current = matches[-1] if matches else None
    if current is None:
        github.api(repo, f"/issues/{pr_number}/comments", token, method="POST", data={"body": body})
    elif current.get("body") != body:
        try:
            github.api(
                repo,
                f"/issues/comments/{current['id']}",
                token,
                method="PATCH",
                data={"body": body},
            )
        except github.APIError as exc:
            if exc.status != 404:
                raise
            github.api(
                repo, f"/issues/{pr_number}/comments", token, method="POST", data={"body": body}
            )
    print(f"CODEOWNER sign-off={status} for assessed commit {head_sha}")


def main() -> None:
    publish(
        os.environ["GITHUB_REPOSITORY"],
        os.environ["GH_TOKEN"],
        int(os.environ["PR_NUMBER"]),
        os.environ["HEAD_SHA"],
        os.environ["SIGNOFF_KEY"],
        Path(os.environ["VERDICT_PATH"]),
        verification_succeeded=os.environ["VERIFICATION_SUCCEEDED"] == "true",
    )


if __name__ == "__main__":
    main()
