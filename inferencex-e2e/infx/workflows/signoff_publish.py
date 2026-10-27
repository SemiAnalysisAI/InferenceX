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
    r"(?:[-*+] )?(✅|❌|➖|⚠️) Check ([0-9]+) \([^\r\n]+\): (PASS|FAIL|N/A|WARN) — \S.*"  # noqa: RUF001
)
CHECK_PREFIX = re.compile(r"(?:[-*+] )?(?:✅|❌|➖|⚠️)\s*Check\b")  # noqa: RUF001
STATUS_EMOJI = {"PASS": "✅", "FAIL": "❌", "N/A": "➖", "WARN": "⚠️"}  # noqa: RUF001
NA_ROW_WITH_PASS_EMOJI = re.compile(
    r"^([ \t]*(?:[-*+] )?)✅( Check [0-9]+ \([^\r\n]+\): N/A — \S.*)$", re.MULTILINE
)
ESCALATION = (
    "⚠️ Pareto coverage needs additional review: @functionstackx @cquil11 @Oseltamivir @adibarra. "
    "At least 5 points per affected throughput-versus-E2EL frontier are highly recommended. "
    "Below 5, or when coverage cannot be verified, merge only with an explicit, recorded admin bypass "
    "for the assessed commit; this advisory comment does not grant or enforce a bypass."
)
NVIDIA_OVERVIEW_WARNING = (
    "❗❗ WARNING ❗❗ NVIDIA official recipes overview: "
    "model/framework coverage is missing or unverified. See Check 16."
)
INVALID = (
    f"{REJECT}\n\nThe verifier did not produce a valid verdict. Retry the sign-off verification."
)


CHECKS = range(19)


def normalize_verdict(verdict: str) -> str:
    """Return verdict with pass emojis in N/A rows replaced by the neutral marker."""
    return NA_ROW_WITH_PASS_EMOJI.sub(r"\1➖\2", verdict)  # noqa: RUF001


def parse_verdict(verdict: str) -> tuple[dict[int, str], list[str]]:
    """Return the check statuses and every reason the verdict cannot be published."""
    lines = verdict.splitlines()
    statuses: dict[int, str] = {}
    errors = []
    for number, line in enumerate(lines, 1):
        row = line.strip()
        match = CHECK_ROW.fullmatch(row)
        if match is None:
            if CHECK_PREFIX.match(row):
                errors.append(
                    f"Line {number}: write each check on one line as "
                    f"`<emoji> Check N (<name>): <STATUS> — <reason>`: {row}"
                )
            continue
        emoji, check_number, status = match.groups()
        check = int(check_number)
        allowed = {"PASS", "N/A", "WARN"} if check in {4, 14, 15, 16} else {"PASS", "N/A", "FAIL"}
        if check == 13:
            allowed.add("WARN")
        if check not in CHECKS:
            errors.append(f"Line {number}: Check {check} does not exist; use Checks 0-18.")
        elif check in statuses:
            errors.append(f"Line {number}: Check {check} appears more than once.")
        elif status not in allowed:
            errors.append(
                f"Line {number}: Check {check} cannot be {status}; use {', '.join(sorted(allowed))}."
            )
        elif emoji != STATUS_EMOJI[status]:
            errors.append(f"Line {number}: {status} rows start with {STATUS_EMOJI[status]}.")
        statuses.setdefault(check, status)
    missing = [str(check) for check in CHECKS if check not in statuses]
    if missing:
        errors.append(f"Missing Check {', '.join(missing)}; include each of Checks 0-18 once.")
    expected = (
        REJECT
        if "FAIL" in statuses.values()
        else WARN
        if "WARN" in statuses.values()
        else SUCCESS_HEADER
    )
    headers = [line for line in lines if line in (SUCCESS_HEADER, REJECT, WARN)]
    if headers != [expected] or lines[:1] != [expected]:
        errors.append(f"The first line must be the only verdict header: {expected}")
    return statuses, errors


def verdict_errors(verdict: str) -> list[str]:
    """Return every reason the publisher would reject this verdict; empty means it publishes."""
    return parse_verdict(normalize_verdict(verdict))[1]


def marker(signoff_key: str) -> str:
    """Return the hidden HTML marker for signoff_key, raising ValueError if the key is invalid."""
    if SIGNOFF_KEY.fullmatch(signoff_key) is None:
        raise ValueError("Invalid sign-off key")
    return f"<!-- codeowner-signoff-verify signoff={signoff_key} -->"


def _is_verdict_for(comment: dict[str, Any], signoff_key: str) -> bool:
    """Return True if comment is a verifier bot's verdict for signoff_key."""
    author = comment.get("user") or {}
    body = comment.get("body") or ""
    return author.get("login") in VERIFIER_AUTHORS and body.startswith(f"{marker(signoff_key)}\n")


def verdict_body(verdict: str, head_sha: str, signoff_key: str) -> tuple[str, str]:
    """Build the PR comment body and status for a verdict on head_sha.

    Returns (body, status) where status is success, warning or failure. Invalid verdicts become a
    failure notice.
    """
    verdict = normalize_verdict(verdict)
    checks, errors = parse_verdict(verdict)
    if errors:
        verdict = INVALID
    else:
        notices = []
        if checks.get(14) == "WARN":
            notices.append(ESCALATION)
        if checks.get(16) == "WARN":
            notices.append(NVIDIA_OVERVIEW_WARNING)
        if notices:
            header, _, rest = verdict.partition("\n")
            notice = "\n\n".join(notices)
            verdict = f"{header}\n\n{notice}\n\n{rest}"
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
    """Post or update the PR verdict comment for signoff_key on head_sha.

    Uses the verdict file only when verification succeeded, and reposts if the old comment was
    deleted.
    """
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
    """Publish the sign-off verdict using settings from the environment."""
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
