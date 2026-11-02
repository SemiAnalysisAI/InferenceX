import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def jobs():
    return yaml.safe_load((ROOT / ".github/workflows/run-sweep.yml").read_text())["jobs"]


def read_outputs(path):
    return dict(line.split("=", 1) for line in path.read_text().splitlines())


def select(jobs, tmp_path, plan):
    step = jobs["canary-select"]["steps"][0]
    output = tmp_path / "selection"
    subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", step["run"]],
        env={**os.environ, "SEARCH_SPACE": json.dumps(plan), "GITHUB_OUTPUT": str(output)},
        check=True,
        capture_output=True,
        text=True,
    )
    return read_outputs(output)


def context(plan, selection):
    return {
        "setup": {"result": "success", "outputs": {"search-space-config": json.dumps(plan)}},
        "canary-select": {"result": "success", "outputs": selection},
        "canary-sweep": {
            "result": "success" if json.loads(selection["canary-config"]) else "skipped",
        },
        "canary-multi-node-sweep": {
            "result": "success" if json.loads(selection["multi-node-canary-config"]) else "skipped",
        },
    }


def run_gate(jobs, tmp_path, needs, labels=None):
    step = jobs["canary-gate"]["steps"][0]
    output = tmp_path / "gate-output"
    env = {}
    for name, expression in step["env"].items():
        if name == "SWEEP_LABELS":
            env[name] = json.dumps(["full-sweep-fail-fast"] if labels is None else labels)
            continue
        value = {"needs": needs}
        for key in expression.removeprefix("${{ ").removesuffix(" }}").split("."):
            value = value.get(key, {}) if isinstance(value, dict) else ""
        env[name] = value if isinstance(value, str) else ""
    result = subprocess.run(
        [sys.executable, "-c", step["run"]],
        env={
            **os.environ,
            **env,
            "GITHUB_OUTPUT": str(output),
        },
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        assert not output.exists()
        return result, None
    return result, json.loads(read_outputs(output)["search-space-config"])


def point(conc, mode="none", **extra):
    return {"conc": conc, "kv-offloading": mode, "model": "glm5.3flash", **extra}


@pytest.fixture
def agentic_plan():
    return {
        "single_node": {
            "agentic": [
                point(conc, mode) for mode in ["none", "dram"] for conc in [24, 32, 48, 64]
            ],
        },
        "multi_node": {},
        "agentic_evals": [point(64, "dram", **{"run-eval": True, "eval-only": True})],
    }


@pytest.mark.parametrize("label", ["full-sweep-fail-fast", "full-sweep-enabled"])
def test_actual_selector_and_gate_release_only_remaining_points(
    jobs,
    tmp_path,
    agentic_plan,
    label,
):
    original = copy.deepcopy(agentic_plan)
    selected = select(jobs, tmp_path, agentic_plan)
    assert json.loads(selected["canary-config"]) == [point(24)]
    assert json.loads(selected["multi-node-canary-config"]) == []
    needs = context(agentic_plan, selected)
    needs["reuse-sweep-gate"] = {"result": "skipped"}
    result, remaining = run_gate(jobs, tmp_path, needs, [label])
    assert result.returncode == 0, result.stderr
    assert remaining["single_node"]["agentic"] == original["single_node"]["agentic"][1:]
    assert remaining["agentic_evals"] == original["agentic_evals"]
    assert agentic_plan == original


@pytest.mark.parametrize("status", ["skipped", "failure", "cancelled"])
def test_required_selector_must_succeed(jobs, tmp_path, agentic_plan, status):
    needs = context(agentic_plan, select(jobs, tmp_path, agentic_plan))
    needs["canary-select"]["result"] = status
    result, _ = run_gate(jobs, tmp_path, needs)
    assert result.returncode != 0
    assert "Required canary selection did not succeed" in result.stderr


@pytest.mark.parametrize("status", ["skipped", "failure", "cancelled"])
@pytest.mark.parametrize("multi", [False, True])
def test_selected_canary_must_succeed(jobs, tmp_path, status, multi):
    plan = (
        {"multi_node": {"agentic": [point([8]), point([16])]}}
        if multi
        else {"single_node": {"agentic": [point(8), point(16)]}}
    )
    needs = context(plan, select(jobs, tmp_path, plan))
    needs["canary-multi-node-sweep" if multi else "canary-sweep"]["result"] = status
    result, _ = run_gate(jobs, tmp_path, needs)
    assert result.returncode != 0
    assert "canary did not succeed" in result.stderr


@pytest.mark.parametrize(
    "output",
    [
        "canary-config",
        "multi-node-canary-config",
        "remaining-search-space-config",
    ],
)
@pytest.mark.parametrize("bad", [None, "", "{invalid", "null", "{}"])
def test_missing_or_malformed_selector_outputs_fail_closed(
    jobs,
    tmp_path,
    agentic_plan,
    output,
    bad,
):
    needs = context(agentic_plan, select(jobs, tmp_path, agentic_plan))
    if bad is None:
        del needs["canary-select"]["outputs"][output]
    else:
        needs["canary-select"]["outputs"][output] = bad
    result, _ = run_gate(jobs, tmp_path, needs)
    assert result.returncode != 0


def test_forged_empty_selection_cannot_bypass_eligible_work(jobs, tmp_path, agentic_plan):
    needs = context(agentic_plan, select(jobs, tmp_path, agentic_plan))
    needs["canary-select"]["outputs"]["canary-config"] = "[]"
    needs["canary-sweep"]["result"] = "skipped"
    result, _ = run_gate(jobs, tmp_path, needs)
    assert result.returncode != 0
    assert "does not match the eligible setup plan" in result.stderr


@pytest.mark.parametrize("replacement", [[point(32)], [point(24), point(32)], [point(1)]])
def test_selector_must_choose_exactly_the_expected_minimum(
    jobs,
    tmp_path,
    agentic_plan,
    replacement,
):
    needs = context(agentic_plan, select(jobs, tmp_path, agentic_plan))
    needs["canary-select"]["outputs"]["canary-config"] = json.dumps(replacement)
    result, _ = run_gate(jobs, tmp_path, needs)
    assert result.returncode != 0


@pytest.mark.parametrize("mutation", ["retain-canary", "drop-another-point", "drop-eval"])
def test_remaining_matrix_cannot_duplicate_or_discard_work(
    jobs,
    tmp_path,
    agentic_plan,
    mutation,
):
    selected = select(jobs, tmp_path, agentic_plan)
    remaining = json.loads(selected["remaining-search-space-config"])
    if mutation == "retain-canary":
        remaining["single_node"]["agentic"].insert(0, point(24))
    elif mutation == "drop-another-point":
        remaining["single_node"]["agentic"].pop()
    else:
        remaining["agentic_evals"] = []
    selected["remaining-search-space-config"] = json.dumps(remaining)
    result, _ = run_gate(jobs, tmp_path, context(agentic_plan, selected))
    assert result.returncode != 0
    assert "remove only the selected canary" in result.stderr


@pytest.mark.parametrize(
    "plan",
    [
        {"single_node": {}, "multi_node": {}, "agentic_evals": [point(64, "dram")]},
        {"single_node": {"agentic": [point(8, **{"run-eval": True})]}},
        {},
    ],
)
def test_no_eligible_performance_requires_successful_explicit_empty_selection(jobs, tmp_path, plan):
    selected = select(jobs, tmp_path, plan)
    assert json.loads(selected["canary-config"]) == []
    assert json.loads(selected["multi-node-canary-config"]) == []
    result, remaining = run_gate(jobs, tmp_path, context(plan, selected))
    assert result.returncode == 0, result.stderr
    assert remaining == json.loads(selected["remaining-search-space-config"])


def test_single_node_has_precedence_over_lower_concurrency_multi_node(jobs, tmp_path):
    plan = {
        "single_node": {"agentic": [point(64), point(32)]},
        "multi_node": {"agentic": [point([1]), point([2])]},
    }
    selected = select(jobs, tmp_path, plan)
    assert json.loads(selected["canary-config"]) == [point(32)]
    assert json.loads(selected["multi-node-canary-config"]) == []
    result, remaining = run_gate(jobs, tmp_path, context(plan, selected))
    assert result.returncode == 0, result.stderr
    assert remaining["single_node"]["agentic"] == [point(64)]
    assert remaining["multi_node"]["agentic"] == plan["multi_node"]["agentic"]


def test_multi_node_agentic_is_selected_when_single_candidates_are_absent(jobs, tmp_path):
    plan = {"multi_node": {"agentic": [point([16]), point([4])]}}
    selected = select(jobs, tmp_path, plan)
    assert json.loads(selected["multi-node-canary-config"]) == [point([4])]
    result, remaining = run_gate(jobs, tmp_path, context(plan, selected))
    assert result.returncode == 0, result.stderr
    assert remaining["multi_node"]["agentic"] == [point([16])]


def test_multi_node_fixed_sequence_remains_outside_existing_canary_pool(jobs, tmp_path):
    plan = {"multi_node": {"1k1k": [point([8])], "8k1k": [point([16])]}}
    selected = select(jobs, tmp_path, plan)
    assert json.loads(selected["multi-node-canary-config"]) == []
    result, remaining = run_gate(jobs, tmp_path, context(plan, selected))
    assert result.returncode == 0, result.stderr
    assert remaining["multi_node"]["1k1k"] == plan["multi_node"]["1k1k"]
    assert remaining["multi_node"]["8k1k"] == plan["multi_node"]["8k1k"]


def test_duplicate_identical_points_remove_one_occurrence(jobs, tmp_path):
    plan = {"single_node": {"agentic": [point(8), point(8), point(16)]}}
    selected = select(jobs, tmp_path, plan)
    result, remaining = run_gate(jobs, tmp_path, context(plan, selected))
    assert result.returncode == 0, result.stderr
    assert remaining["single_node"]["agentic"] == [point(8), point(16)]


def test_large_remaining_plan_passes_as_separate_raw_environment_values(jobs, tmp_path):
    retained = point(16)
    metadata = {"audit_padding": "x" * 80_000}
    plan = {
        "single_node": {"agentic": [point(8), retained]},
        "changelog_metadata": metadata,
    }
    selected = select(jobs, tmp_path, plan)
    result, remaining = run_gate(jobs, tmp_path, context(plan, selected))
    assert result.returncode == 0, result.stderr
    assert remaining["single_node"]["agentic"] == [retained]
    assert remaining["changelog_metadata"] == metadata


def test_identical_points_in_multiple_groups_preserve_selector_removal_behavior(jobs, tmp_path):
    plan = {
        "single_node": {
            "1k1k": [point(8), point(8)],
            "8k1k": [point(8), point(16)],
            "agentic": [point(32)],
        },
    }
    selected = select(jobs, tmp_path, plan)
    result, remaining = run_gate(jobs, tmp_path, context(plan, selected))
    assert result.returncode == 0, result.stderr
    assert remaining["single_node"] == {
        "1k1k": [point(8)],
        "8k1k": [point(16)],
        "agentic": [point(32)],
    }


@pytest.mark.parametrize("single,multi", [(None, None), ({}, None), (None, {})])
def test_null_families_keep_existing_empty_selection_behavior(jobs, tmp_path, single, multi):
    plan = {"single_node": single, "multi_node": multi}
    selected = select(jobs, tmp_path, plan)
    result, remaining = run_gate(jobs, tmp_path, context(plan, selected))
    assert result.returncode == 0, result.stderr
    assert remaining == {
        "single_node": {"1k1k": [], "8k1k": [], "agentic": []},
        "multi_node": {"agentic": []},
    }


def test_intentional_non_canary_mode_preserves_complete_plan(jobs, tmp_path, agentic_plan):
    needs = {
        "setup": {
            "result": "success",
            "outputs": {"search-space-config": json.dumps(agentic_plan)},
        },
        "canary-select": {"result": "skipped", "outputs": {}},
        "canary-sweep": {"result": "skipped"},
        "canary-multi-node-sweep": {"result": "skipped"},
    }
    result, remaining = run_gate(jobs, tmp_path, needs, ["non-canary-full-sweep-enabled"])
    assert result.returncode == 0, result.stderr
    assert remaining == agentic_plan


@pytest.mark.parametrize(
    "labels",
    [
        [],
        ["agentx-fast"],
        ["full-sweep-enabled", "non-canary-full-sweep-enabled"],
    ],
)
def test_removed_or_conflicting_primary_labels_fail_closed(jobs, tmp_path, agentic_plan, labels):
    needs = context(agentic_plan, select(jobs, tmp_path, agentic_plan))
    result, _ = run_gate(jobs, tmp_path, needs, labels)
    assert result.returncode != 0
    assert "Exactly one primary sweep label" in result.stderr


@pytest.mark.parametrize("status", ["skipped", "failure", "cancelled"])
def test_setup_must_succeed(jobs, tmp_path, agentic_plan, status):
    needs = context(agentic_plan, select(jobs, tmp_path, agentic_plan))
    needs["setup"]["result"] = status
    result, _ = run_gate(jobs, tmp_path, needs)
    assert result.returncode != 0
    assert "Setup did not succeed" in result.stderr
