import json
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def publish(tmp_path):
    def run(verdict, comments=(), *, succeeded=True, update_error=None, repeats=1):
        verdict_path = tmp_path / "verdict.md"
        if verdict is not None:
            verdict_path.write_text(verdict)
        result = subprocess.run(
            ["node", "-e", """
const fs = require('node:fs');
const { publish } = require('./.github/scripts/codeowner-signoff.cjs');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const comments = input.comments;
let writes = 0;
const issues = {
  listComments: Symbol('listComments'),
  async updateComment({ comment_id, body }) {
    if (input.update_error) {
      if (input.update_error === 404) comments.splice(comments.findIndex(c => c.id === comment_id), 1);
      throw Object.assign(new Error('GitHub write failed'), { status: input.update_error });
    }
    const comment = comments.find(c => c.id === comment_id);
    comment.body = body;
    writes++;
    return { data: comment };
  },
  async createComment({ body }) {
    const comment = { id: 100, user: { login: 'github-actions[bot]' }, body };
    comments.push(comment);
    writes++;
    return { data: comment };
  },
};
const github = {
  rest: { issues },
  async paginate(endpoint) {
    if (endpoint !== issues.listComments) throw new Error('Unexpected API');
    return comments;
  },
};
(async () => {
  for (let n = 0; n < input.repeats; n++) {
    await publish({ github, context: { repo: { owner: 'example', repo: 'repo' } },
      core: { info() {} }, prNumber: 7, headSha: 'abcdef1234567890abcdef1234567890abcdef1234',
      verdictPath: input.path, verificationSucceeded: input.succeeded });
  }
  process.stdout.write(JSON.stringify({ comments, writes }));
})().catch(error => { console.error(error.message); process.exitCode = 1; });
"""],
            input=json.dumps({"path": str(verdict_path), "comments": comments,
                             "succeeded": succeeded, "update_error": update_error, "repeats": repeats}),
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        return json.loads(result.stdout)
    return run


@pytest.mark.parametrize("author,marker", [
    ("github-actions[bot]", "<!-- codeowner-signoff-verify -->"),
    ("Klaud-Cold", "<!-- codeowner-signoff-verify sha=1111111111111111111111111111111111111111 -->"),
])
def test_verdict_reuses_bot_comment_and_records_only_assessed_commit(publish, author, marker):
    result = publish("## ✅✅✅ **Verdict: PASS** ✅✅✅\n\nAll checks passed.", [
        {"id": 1, "user": {"login": "contributor"}, "body": "<!-- codeowner-signoff-verify -->\nKeep my comment"},
        {"id": 2, "user": {"login": author}, "body": marker + "\nOld verdict"},
    ], repeats=2)
    assert result == {"comments": [
        {"id": 1, "user": {"login": "contributor"}, "body": "<!-- codeowner-signoff-verify -->\nKeep my comment"},
        {"id": 2, "user": {"login": author}, "body":
         "<!-- codeowner-signoff-verify -->\n## ✅✅✅ **Verdict: PASS** ✅✅✅\n\nAll checks passed."
         "\n\nAssessed commit: `abcdef1234567890abcdef1234567890abcdef1234`.\n"},
    ], "writes": 1}


@pytest.mark.parametrize("verdict,succeeded", [
    (None, True),
    ("Incomplete response", True),
    ("## ✅✅✅ **Verdict: PASS** ✅✅✅\n## ❌❌❌ **REJECTED** ❌❌❌", True),
    ("## ✅✅✅ **Verdict: PASS** ✅✅✅", False),
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
    result = publish("## ❌❌❌ **REJECTED** ❌❌❌\n\nMissing sweep evidence.", comments,
                     update_error=update_error)
    assert len(result["comments"]) == 2
    assert result["comments"][0]["body"] == "<!-- codeowner-signoff-verify -->\nMy comment"
    assert result["comments"][1]["body"] == (
        "<!-- codeowner-signoff-verify -->\n## ❌❌❌ **REJECTED** ❌❌❌\n\nMissing sweep evidence.\n\n"
        "Assessed commit: `abcdef1234567890abcdef1234567890abcdef1234`.\n"
    )


def test_comment_update_errors_are_not_silently_replaced_with_duplicate_comments(publish):
    with pytest.raises(RuntimeError, match="GitHub write failed"):
        publish("## ✅✅✅ **Verdict: PASS** ✅✅✅", [
            {"id": 1, "user": {"login": "github-actions[bot]"},
             "body": "<!-- codeowner-signoff-verify -->\nOld verdict"},
        ], update_error=403)
