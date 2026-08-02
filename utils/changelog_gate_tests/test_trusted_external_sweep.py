"""Verify the trusted external-sweep control-plane boundary."""

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def _workflow(name: str) -> dict:
    """Load a workflow without YAML 1.1 coercing the ``on`` key."""
    return yaml.load(
        (REPO_ROOT / ".github/workflows" / name).read_text(),
        Loader=yaml.BaseLoader,
    )


def test_dispatcher_is_label_only_and_never_checks_out_pr_code() -> None:
    workflow = _workflow("trusted-external-sweep.yml")
    trigger = workflow["on"]
    assert set(trigger) == {"pull_request_target"}
    assert trigger["pull_request_target"]["types"] == ["labeled"]

    permissions = workflow["permissions"]
    assert permissions == {
        "actions": "write",
        "contents": "read",
        "issues": "write",
        "pull-requests": "read",
    }

    job = workflow["jobs"]["dispatch"]
    assert "head.repo.full_name != github.repository" in job["if"]
    step = job["steps"][0]
    assert step["uses"].startswith("actions/github-script@")
    script = step["with"]["script"]
    assert "getCollaboratorPermissionLevel" in script
    assert "pull.head.sha !== eventPull.head.sha" in script
    assert "createWorkflowDispatch" in script
    assert "workflow_id: 'e2e-tests.yml'" in script
    assert "ref: context.payload.repository.default_branch" in script
    assert "actions/checkout" not in script


def test_trusted_dispatch_pins_matrix_and_benchmark_revisions() -> None:
    workflow = _workflow("trusted-external-sweep.yml")
    script = workflow["jobs"]["dispatch"]["steps"][0]["with"]["script"]
    assert "'ref': pull.merge_commit_sha" in script
    assert "'changelog-base-ref': pull.base.sha" in script
    assert "'changelog-head-ref': pull.head.sha" in script

    e2e = _workflow("e2e-tests.yml")
    dispatch_inputs = e2e["on"]["workflow_dispatch"]["inputs"]
    call_inputs = e2e["on"]["workflow_call"]["inputs"]
    trusted_inputs = {
        "changelog-base-ref",
        "changelog-head-ref",
        "trim-conc",
        "all-evals",
        "evals-only",
        "fail-fast",
        "pr-labels-json",
    }
    assert trusted_inputs <= set(dispatch_inputs)
    assert trusted_inputs <= set(call_inputs)

    checkout = e2e["jobs"]["get-jobs"]["steps"][0]
    assert checkout["with"]["ref"] == "${{ inputs.ref }}"
    assert checkout["with"]["fetch-depth"] == "0"
    assert checkout["with"]["persist-credentials"] == "false"


def test_untrusted_fork_run_cannot_fan_out_to_gpu_jobs() -> None:
    run_sweep = _workflow("run-sweep.yml")
    setup_if = run_sweep["jobs"]["setup"]["if"]
    assert (
        "github.event.pull_request.head.repo.full_name == github.repository"
        in setup_if
    )
