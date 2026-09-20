import json
import subprocess
from copy import deepcopy

import pytest

from infx import github
from infx.workflows import signoff_publish


PASS = "## ✅✅✅ **Verdict: PASS** ✅✅✅"
REJECT = "## ❌❌❌ **REJECTED** ❌❌❌"
WARN = "## ⚠️ **Verdict: WARN** ⚠️"


def verdict(header=PASS, *, failure=False, warning=False, prefix=""):
    rows = [f"✅ Check {number} (Requirement): PASS — Verified." for number in range(14)]
    rows[13] = "➖ Check 13 (Draft precision): N/A — No speculative changes."
    expanded = []
    if failure:
        rows.pop(1)
        expanded.append("❌ Check 1 (Sweep): FAIL — No passing sweep.")
    if warning:
        expanded.append(prefix + "⚠️ Check 14 (Pareto coverage): WARN — curve-a: 3/5; admin bypass not verified.")
    else:
        rows.append("✅ Check 14 (Pareto coverage): PASS — curve-a: 5/5.")
    return (
        header + "\n\n" + "\n".join(expanded)
        + "\n\n<details>\n<summary>Passing checks</summary>\n\n"
        + "\n".join(rows) + "\n</details>"
    )


@pytest.fixture
def publish(tmp_path, monkeypatch):
    def run(verdict, comments=(), *, succeeded=True, update_error=None, repeats=1):
        verdict_path = tmp_path / "verdict.md"
        if verdict is not None:
            verdict_path.write_text(verdict)
        else:
            verdict_path.unlink(missing_ok=True)
        comments = deepcopy(list(comments))
        writes = 0

        def gh(args, **kwargs):
            nonlocal writes
            method = args[args.index("--method") + 1]
            endpoint = next(arg for arg in args if arg.startswith("repos/"))
            if method == "GET":
                return subprocess.CompletedProcess(args, 0, json.dumps([comments]), "")
            body = json.loads(kwargs["input"])["body"]
            if method == "PATCH":
                comment_id = int(endpoint.rsplit("/", 1)[1])
                if update_error:
                    if update_error == 404:
                        comments[:] = [c for c in comments if c["id"] != comment_id]
                    raise subprocess.CalledProcessError(
                        1, args, stderr=f"gh: GitHub write failed (HTTP {update_error})"
                    )
                comment = next(c for c in comments if c["id"] == comment_id)
                comment["body"] = body
            else:
                comment = {"id": 100, "user": {"login": "github-actions[bot]"}, "body": body}
                comments.append(comment)
            writes += 1
            return subprocess.CompletedProcess(args, 0, json.dumps(comment), "")

        monkeypatch.setattr(github.subprocess, "run", gh)
        for name, value in {
            "GITHUB_REPOSITORY": "example/repo", "GH_TOKEN": "test-token", "PR_NUMBER": "7",
            "HEAD_SHA": "abcdef1234567890abcdef1234567890abcdef1234",
            "VERDICT_PATH": str(verdict_path), "VERIFICATION_SUCCEEDED": str(succeeded).lower(),
        }.items():
            monkeypatch.setenv(name, value)
        for _ in range(repeats):
            signoff_publish.main()
        return {"comments": comments, "writes": writes}
    return run


@pytest.mark.parametrize("author,marker", [
    ("github-actions[bot]", "<!-- codeowner-signoff-verify -->"),
    ("Klaud-Cold", "<!-- codeowner-signoff-verify sha=1111111111111111111111111111111111111111 -->"),
])
def test_verdict_reuses_bot_comment_and_records_only_assessed_commit(publish, author, marker):
    text = verdict()
    result = publish(text, [
        {"id": 1, "user": {"login": "contributor"}, "body": "<!-- codeowner-signoff-verify -->\nKeep my comment"},
        {"id": 2, "user": {"login": author}, "body": marker + "\nOld verdict"},
    ], repeats=2)
    assert result == {"comments": [
        {"id": 1, "user": {"login": "contributor"}, "body": "<!-- codeowner-signoff-verify -->\nKeep my comment"},
        {"id": 2, "user": {"login": author}, "body":
         "<!-- codeowner-signoff-verify -->\n" + text +
         "\n\nAssessed commit: `abcdef1234567890abcdef1234567890abcdef1234`.\n"},
    ], "writes": 1}


@pytest.mark.parametrize("verdict,succeeded", [
    (None, True),
    ("Incomplete response", True),
    (verdict() + "\n" + REJECT, True),
    (verdict() + "\n" + PASS, True),
    (verdict(), False),
    (verdict(WARN, failure=True, warning=True), True),
    (verdict(PASS, warning=True, prefix="- "), True),
    (verdict(WARN), True),
    (verdict(REJECT), True),
    (verdict().replace("✅ Check 14 (Pareto coverage): PASS — curve-a: 5/5.\n", ""), True),
    (verdict() + "\n✅ Check 14 (Pareto coverage): PASS — duplicate", True),
    (verdict() + "\n✅ Check 15 (Unknown): PASS — extra", True),
    (verdict().replace("✅ Check 1 (Requirement)", "❌ Check 1 (Requirement)"), True),
    (verdict().replace("✅ Check 1 (Requirement): PASS", "⚠️ Check 1 (Requirement): WARN"), True),
    (verdict().replace("✅ Check 14 (Pareto coverage): PASS", "❌ Check 14 (Pareto coverage): FAIL"), True),
    (verdict() + "\n⚠️ Check 14 (Pareto coverage): UNKNOWN — malformed duplicate", True),
    (verdict().replace("PASS — curve-a: 5/5.", "PASS — "), True),
])
def test_invalid_or_failed_verification_replaces_previous_pass(publish, verdict, succeeded):
    result = publish(verdict, [{"id": 1, "user": {"login": "github-actions[bot]"},
                               "body": "<!-- codeowner-signoff-verify -->\n## ✅✅✅ **Verdict: PASS** ✅✅✅"}],
                     succeeded=succeeded)
    assert len(result["comments"]) == 1
    assert result["comments"][0]["body"] == (
        "<!-- codeowner-signoff-verify -->\n## ❌❌❌ **REJECTED** ❌❌❌\n\n"
        "The verifier did not produce a valid verdict. Retry the sign-off verification.\n\n"
        "Assessed commit: `abcdef1234567890abcdef1234567890abcdef1234`.\n"
    )


@pytest.mark.parametrize("existing,update_error", [(False, None), (True, 404)])
def test_missing_or_deleted_verdict_is_created_without_editing_human_comment(publish, existing, update_error):
    comments = [{"id": 1, "user": {"login": "contributor"},
                 "body": "<!-- codeowner-signoff-verify -->\nMy comment"}]
    if existing:
        comments.append({"id": 2, "user": {"login": "github-actions[bot]"},
                         "body": "<!-- codeowner-signoff-verify -->\nOld verdict"})
    text = verdict(REJECT, failure=True)
    result = publish(text, comments,
                     update_error=update_error)
    assert len(result["comments"]) == 2
    assert result["comments"][0]["body"] == "<!-- codeowner-signoff-verify -->\nMy comment"
    assert result["comments"][1]["body"] == (
        "<!-- codeowner-signoff-verify -->\n" + text + "\n\n"
        "Assessed commit: `abcdef1234567890abcdef1234567890abcdef1234`.\n"
    )


def test_comment_update_errors_are_not_silently_replaced_with_duplicate_comments(publish):
    with pytest.raises(RuntimeError, match="GitHub write failed"):
        publish(verdict(), [
            {"id": 1, "user": {"login": "github-actions[bot]"},
             "body": "<!-- codeowner-signoff-verify -->\nOld verdict"},
        ], update_error=403)


@pytest.mark.parametrize("header,failure,prefix", [
    (WARN, False, ""),
    (REJECT, True, ""),
    (WARN, False, "- "),
])
def test_coverage_warning_escalates_once_without_masking_other_failures(publish, header, failure, prefix):
    result = publish(
        verdict(header, failure=failure, warning=True, prefix=prefix),
        repeats=2,
    )
    body = result["comments"][0]["body"]
    assert body.startswith("<!-- codeowner-signoff-verify -->\n" + header)
    assert body.count("@functionstackx") == 1
    assert body.count("@cquil11") == 1
    assert body.count("@Oseltamivir") == 1
    assert body.count("@adibarra") == 1
    assert "does not grant or enforce a bypass" in body
    assert "Assessed commit: `abcdef1234567890abcdef1234567890abcdef1234`" in body
    assert result["writes"] == 1


def test_successful_reassessment_removes_stale_warning_and_tags(publish):
    comments = publish(verdict(WARN, warning=True))["comments"]
    result = publish(verdict(), comments)
    assert len(result["comments"]) == 1
    assert "**Verdict: PASS**" in result["comments"][0]["body"]
    assert "@functionstackx" not in result["comments"][0]["body"]
    assert "@adibarra" not in result["comments"][0]["body"]
    assert "curve-a: 3/5" not in result["comments"][0]["body"]
