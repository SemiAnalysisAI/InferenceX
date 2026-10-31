"""Evaluate canary conditions with the documented Actions status-check semantics."""

import re
from pathlib import Path

import pytest
import yaml


WORKFLOW = Path(__file__).resolve().parents[4] / ".github/workflows/run-sweep.yml"
JOBS = yaml.safe_load(WORKFLOW.read_text())["jobs"]


def selected(job, *, labels=(), setup="success", selector="success", output="[{}]", cancelled=False):
    results = {"reuse-sweep-gate": "skipped", "setup": setup, "canary-select": selector}
    expression = JOBS[job]["if"].strip().removeprefix("${{").removesuffix("}}").strip()
    # Actions adds success() unless the condition contains a status function;
    # skipped ancestors remain in that status check after setup has succeeded.
    if not re.search(r"\b(?:always|cancelled|failure|success)\s*\(", expression):
        expression = f"success() && ({expression})"
    expression = expression.replace("github.event.pull_request.labels.*.name", "labels")
    expression = re.sub(
        r"needs\.([\w-]+)\.(result|outputs\.[\w-]+)",
        lambda match: repr(results[match[1]] if match[2] == "result" else output),
        expression,
    )
    expression = re.sub(r"!(?!=)", " not ", expression).replace("&&", " and ").replace("||", " or ")
    return eval(" ".join(expression.split()), {"__builtins__": {}}, {
        "labels": labels,
        "contains": lambda values, item: item in values,
        "cancelled": lambda: cancelled,
        "success": lambda: all(result == "success" for result in results.values()),
    })


@pytest.mark.parametrize("label", ["full-sweep-enabled", "full-sweep-fail-fast"])
def test_label_launch_selects_canary_after_intentionally_skipped_reuse_gate(label):
    assert selected("canary-select", labels=[label])


@pytest.mark.parametrize("context", [
    {"labels": []},
    {"labels": ["sweep-enabled"]},
    {"labels": ["full-sweep-enabled"], "setup": "failure"},
    {"labels": ["full-sweep-enabled"], "setup": "skipped"},
    {"labels": ["full-sweep-enabled"], "cancelled": True},
])
def test_selector_requires_full_label_successful_setup_and_active_workflow(context):
    assert not selected("canary-select", **context)


@pytest.mark.parametrize("job", ["canary-sweep", "canary-multi-node-sweep"])
def test_selected_canary_runs_after_intentionally_skipped_reuse_gate(job):
    assert selected(job)


@pytest.mark.parametrize("job", ["canary-sweep", "canary-multi-node-sweep"])
@pytest.mark.parametrize("context", [
    {"output": ""}, {"output": "[]"}, {"selector": "failure"},
    {"selector": "skipped"}, {"cancelled": True},
])
def test_canary_requires_successful_selection_nonempty_matrix_and_active_workflow(job, context):
    assert not selected(job, **context)
