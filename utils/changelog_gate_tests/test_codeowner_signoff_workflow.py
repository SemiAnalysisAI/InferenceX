from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github/workflows/codeowner-signoff-verify.yml"
PROMPT_PATH = REPO_ROOT / ".github/codeowner-signoff-verify-prompt.md"


def load_workflow() -> dict[str, object]:
    """Load workflow YAML without coercing the top-level ``on`` key."""
    return yaml.load(WORKFLOW_PATH.read_text(), Loader=yaml.BaseLoader)


def step_script(job: dict[str, object], name: str) -> str:
    """Return the github-script body for a named workflow step."""
    steps = job["steps"]
    assert isinstance(steps, list)
    step = next(item for item in steps if item["name"] == name)
    return step["with"]["script"]


def test_verifier_tracks_pr_synchronization_with_pr_scoped_concurrency() -> None:
    workflow = load_workflow()

    assert workflow["on"]["pull_request_target"]["types"] == [
        "opened",
        "reopened",
        "ready_for_review",
        "synchronize",
    ]
    assert workflow["concurrency"]["cancel-in-progress"] == "true"
    assert "inputs.pr_number" in workflow["concurrency"]["group"]


def test_gate_creates_an_immediate_check_and_guards_signoff_reuse() -> None:
    workflow = load_workflow()
    gate = workflow["jobs"]["gate"]
    script = step_script(gate, "Resolve PR state and sign-off metadata")

    assert gate["permissions"]["checks"] == "write"
    assert "fingerprintDiff" in script
    assert "priorVerification" in script
    assert script.index("const checkRunId = await createCheck") < script.index(
        "const fingerprint = await fingerprintDiff"
    )
    assert "prior.state.fp !== fingerprint" in script
    assert "fresh CODEOWNER sign-off is required" in script
    assert "github.rest.checks.create" in script
    assert "'in_progress'" in script
    assert "'action_required'" in script
    assert "github.rest.repos.createCommitStatus" not in script


def test_verifier_uses_structured_output_and_updates_the_same_check_run() -> None:
    workflow = load_workflow()
    verify = workflow["jobs"]["verify"]
    complete_script = step_script(verify, "Complete CODEOWNER verification check")
    prompt = PROMPT_PATH.read_text()

    assert verify["permissions"]["checks"] == "write"
    assert "statuses" not in verify["permissions"]
    assert "VERDICT_FILE" in prompt
    assert '"checks"' in prompt
    assert "Do NOT post, edit, or delete any PR comment" in prompt
    assert "gh pr comment" not in prompt
    assert "numbers.length === 11" in complete_script
    assert "github.rest.checks.update" in complete_script
    assert "github.rest.repos.createCommitStatus" not in complete_script


def test_contributor_docs_describe_check_run_without_verdict_comment_spam() -> None:
    english = (REPO_ROOT / "CONTRIBUTING.md").read_text()
    chinese = (REPO_ROOT / "CONTRIBUTING_zh.md").read_text()

    assert "Detailed results live in the Check Run" in english
    assert "effective PR diff is unchanged" in english
    assert "详细结果会显示在 Check Run 中" in chinese
    assert "PR 的实际 diff 未改变" in chinese
