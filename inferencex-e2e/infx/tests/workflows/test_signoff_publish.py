"""Exercise sign-off verdict publication against a controlled GitHub API."""

import json
import subprocess
from copy import deepcopy

import pytest

from infx import github
from infx.workflows import signoff_publish

PASS = "## ✅✅✅ **Verdict: PASS** ✅✅✅"
REJECT = "## ❌❌❌ **REJECTED** ❌❌❌"
WARN = "## ⚠️ **Verdict: WARN** ⚠️"
SHA = "abcdef1234567890abcdef1234567890abcdef1234"
KEY = "pullrequestreview-42"


def verdict(
    header=PASS, *, failure=False, warning=False, reuse_warning=False, prefix=""
):
    rows = [
        f"✅ Check {number} (Requirement): PASS — Verified." for number in range(14)
    ]
    rows[13] = "➖ Check 13 (Draft precision): N/A — No speculative changes."
    expanded = []
    if reuse_warning:
        rows.pop(4)
        expanded.append(
            "⚠️ Check 4 (Reuse command): WARN — No authorized reuse command."
        )
    if failure:
        rows.pop(1)
        expanded.append("❌ Check 1 (Sweep): FAIL — No passing sweep.")
    if warning:
        expanded.append(
            prefix
            + "⚠️ Check 14 (Pareto coverage): WARN — curve-a: 3/5; admin bypass not verified."
        )
    else:
        rows.append("✅ Check 14 (Pareto coverage): PASS — curve-a: 5/5.")
    return (
        header
        + "\n\n"
        + "\n".join(expanded)
        + "\n\n<details>\n<summary>Passing checks</summary>\n\n"
        + "\n".join(rows)
        + "\n</details>"
    )


def marker(key=KEY):
    return f"<!-- codeowner-signoff-verify signoff={key} -->"


@pytest.fixture
def publish(tmp_path, monkeypatch):
    def run(
        content,
        comments=(),
        *,
        signoff_key=KEY,
        succeeded=True,
        update_error=None,
        repeats=1,
    ):
        verdict_path = tmp_path / "verdict.md"
        if content is not None:
            verdict_path.write_text(content)
        else:
            verdict_path.unlink(missing_ok=True)
        current_comments = deepcopy(list(comments))
        writes = []

        def gh(args, **kwargs):
            method = args[args.index("--method") + 1]
            endpoint = next(arg for arg in args if arg.startswith("repos/"))
            if method == "GET":
                return subprocess.CompletedProcess(
                    args, 0, json.dumps([current_comments]), ""
                )
            body = json.loads(kwargs["input"])["body"]
            if method == "PATCH" and update_error:
                raise subprocess.CalledProcessError(
                    1, args, stderr=f"gh: GitHub write failed (HTTP {update_error})"
                )
            if method == "PATCH":
                comment_id = int(endpoint.rsplit("/", 1)[1])
                comment = next(
                    comment
                    for comment in current_comments
                    if comment["id"] == comment_id
                )
                comment["body"] = body
            else:
                comment = {
                    "id": 100 + len(writes),
                    "user": {"login": "github-actions[bot]"},
                    "body": body,
                }
                current_comments.append(comment)
            writes.append((method, endpoint))
            return subprocess.CompletedProcess(args, 0, json.dumps(comment), "")

        monkeypatch.setattr(github.subprocess, "run", gh)
        for name, value in {
            "GITHUB_REPOSITORY": "example/repo",
            "GH_TOKEN": "test-token",
            "PR_NUMBER": "7",
            "HEAD_SHA": SHA,
            "SIGNOFF_KEY": signoff_key,
            "VERDICT_PATH": str(verdict_path),
            "VERIFICATION_SUCCEEDED": str(succeeded).lower(),
        }.items():
            monkeypatch.setenv(name, value)
        for _ in range(repeats):
            signoff_publish.main()
        return {"comments": current_comments, "writes": writes}

    return run


def test_edit_updates_only_the_verdict_for_that_signoff(publish):
    comments = [
        {"id": 1, "user": {"login": "github-actions[bot]"}, "body": marker() + "\nOld"},
        {
            "id": 2,
            "user": {"login": "github-actions[bot]"},
            "body": marker("issuecomment-41") + "\nOlder sign-off",
        },
        {
            "id": 3,
            "user": {"login": "Klaud-Cold"},
            "body": "<!-- codeowner-signoff-verify -->\nLegacy",
        },
    ]
    result = publish(verdict(), comments)
    assert result["comments"][1:] == comments[1:]
    assert result["comments"][0]["body"] == (
        marker() + "\n" + verdict() + f"\n\nAssessed commit: `{SHA}`.\n"
    )
    assert result["writes"] == [("PATCH", "repos/example/repo/issues/comments/1")]


def test_missing_verdict_creates_one_without_touching_other_or_spoofed_comments(
    publish,
):
    comments = [
        {
            "id": 1,
            "user": {"login": "github-actions[bot]"},
            "body": marker("issuecomment-41") + "\nOlder sign-off",
        },
        {"id": 2, "user": {"login": "contributor"}, "body": marker() + "\nSpoof"},
    ]
    result = publish(verdict(), comments)
    assert result["comments"][:2] == comments
    assert result["comments"][2]["body"].startswith(marker() + "\n" + PASS)
    assert result["writes"] == [("POST", "repos/example/repo/issues/7/comments")]


def test_identical_replay_does_not_write_again(publish):
    result = publish(verdict(), repeats=2)
    assert len(result["comments"]) == 1
    assert result["writes"] == [("POST", "repos/example/repo/issues/7/comments")]


def test_replacement_signoff_gets_a_separate_verdict(publish):
    previous = publish(verdict(), signoff_key="issuecomment-41")["comments"]
    result = publish(verdict(), previous)
    assert result["comments"][0] == previous[0]
    assert result["comments"][0]["body"].startswith(marker("issuecomment-41"))
    assert result["comments"][1]["body"].startswith(marker())
    assert result["writes"] == [("POST", "repos/example/repo/issues/7/comments")]


def test_deleted_verdict_during_update_is_recreated(publish):
    comments = [
        {"id": 1, "user": {"login": "github-actions[bot]"}, "body": marker() + "\nOld"}
    ]
    result = publish(verdict(), comments, update_error=404)
    assert result["comments"][0] == comments[0]
    assert result["comments"][1]["body"].startswith(marker() + "\n" + PASS)
    assert result["writes"] == [("POST", "repos/example/repo/issues/7/comments")]


def test_update_errors_other_than_missing_comment_propagate(publish):
    comments = [
        {"id": 1, "user": {"login": "github-actions[bot]"}, "body": marker() + "\nOld"}
    ]
    with pytest.raises(github.APIError, match="GitHub write failed"):
        publish(verdict(), comments, update_error=403)


@pytest.mark.parametrize(
    "content,succeeded",
    [
        (None, True),
        ("Incomplete response", True),
        (verdict() + "\n" + REJECT, True),
        (verdict() + "\n" + PASS, True),
        (verdict(), False),
        (verdict(WARN, failure=True, warning=True), True),
        (verdict(PASS, warning=True, prefix="- "), True),
        (verdict(WARN), True),
        (verdict(PASS, reuse_warning=True), True),
        (verdict(REJECT, reuse_warning=True), True),
        (
            verdict().replace(
                "✅ Check 4 (Requirement): PASS", "❌ Check 4 (Requirement): FAIL"
            ),
            True,
        ),
        (verdict(REJECT), True),
        (
            verdict().replace(
                "✅ Check 14 (Pareto coverage): PASS — curve-a: 5/5.\n", ""
            ),
            True,
        ),
        (verdict() + "\n✅ Check 14 (Pareto coverage): PASS — duplicate", True),
        (verdict() + "\n✅ Check 15 (Unknown): PASS — extra", True),
        (
            verdict().replace("✅ Check 1 (Requirement)", "❌ Check 1 (Requirement)"),
            True,
        ),
        (
            verdict().replace(
                "✅ Check 1 (Requirement): PASS", "⚠️ Check 1 (Requirement): WARN"
            ),
            True,
        ),
        (
            verdict().replace(
                "✅ Check 14 (Pareto coverage): PASS",
                "❌ Check 14 (Pareto coverage): FAIL",
            ),
            True,
        ),
        (
            verdict() + "\n⚠️ Check 14 (Pareto coverage): UNKNOWN — malformed duplicate",
            True,
        ),
        (verdict().replace("PASS — curve-a: 5/5.", "PASS — "), True),
    ],
)
def test_invalid_or_failed_verification_publishes_rejection(
    publish, content, succeeded
):
    result = publish(content, succeeded=succeeded)
    assert result["comments"][0]["body"] == (
        marker() + "\n## ❌❌❌ **REJECTED** ❌❌❌\n\n"
        "The verifier did not produce a valid verdict. Retry the sign-off verification.\n\n"
        f"Assessed commit: `{SHA}`.\n"
    )


@pytest.mark.parametrize(
    "header,failure,prefix",
    [(WARN, False, ""), (REJECT, True, ""), (WARN, False, "- ")],
)
def test_coverage_warning_escalates_once_without_masking_other_failures(
    publish, header, failure, prefix
):
    result = publish(
        verdict(header, failure=failure, warning=True, prefix=prefix), repeats=2
    )
    body = result["comments"][0]["body"]
    assert body.startswith(marker() + "\n" + header)
    assert body.count("@functionstackx") == 1
    assert body.count("@cquil11") == 1
    assert body.count("@Oseltamivir") == 1
    assert body.count("@adibarra") == 1
    assert "does not grant or enforce a bypass" in body
    assert f"Assessed commit: `{SHA}`" in body
    assert len(result["writes"]) == 1


@pytest.mark.parametrize(
    "failure,coverage_warning",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_reuse_warning_preserves_verdict_and_only_coverage_escalates(
    publish, failure, coverage_warning
):
    header = REJECT if failure else WARN
    result = publish(
        verdict(header, failure=failure, warning=coverage_warning, reuse_warning=True)
    )
    body = result["comments"][0]["body"]
    assert body.startswith(marker() + "\n" + header)
    assert "⚠️ Check 4 (Reuse command): WARN — No authorized reuse command." in body
    assert ("Pareto coverage needs additional review" in body) == coverage_warning
    assert ("@functionstackx" in body) == coverage_warning
    assert ("❌ Check 1 (Sweep): FAIL" in body) == failure
    assert "did not produce a valid verdict" not in body


@pytest.mark.parametrize(
    "key", ["", "issuecomment-0", "pullrequestreview-x", "other-42"]
)
def test_invalid_signoff_key_is_rejected_before_github_write(publish, key):
    with pytest.raises(ValueError, match="sign-off key"):
        publish(verdict(), signoff_key=key)
