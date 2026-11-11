"""Unattended workflows run only in SemiAnalysisAI/InferenceX."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = "github.repository == 'SemiAnalysisAI/InferenceX'"
UNATTENDED = {"schedule", "workflow_run"}
STATUS_FUNCTIONS = ("always()", "cancelled()", "failure()")


def triggers(workflow: dict) -> set[str]:
    on = workflow.get(True, workflow.get("on", {}))
    return {on} if isinstance(on, str) else set(on)


def test_unattended_jobs_require_the_upstream_repository():
    unguarded = []
    for path in sorted((ROOT.parent / ".github" / "workflows").glob("*.yml")):
        workflow = yaml.safe_load(path.read_text())
        if not UNATTENDED & triggers(workflow):
            continue
        for name, job in workflow["jobs"].items():
            condition = str(job.get("if", ""))
            starts_alone = "needs" not in job or any(f in condition for f in STATUS_FUNCTIONS)
            if starts_alone and UPSTREAM not in condition:
                unguarded.append(f"{path.name}:{name}")
    assert unguarded == []
