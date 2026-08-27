"""Static contracts for priority-scheduled workflow node demand."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"


def workflow(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def test_every_priority_scheduled_gpu_workflow_emits_node_demand() -> None:
    expected_node_expression = {
        "benchmark-tmpl.yml": "toJSON('nodes:1')",
        "benchmark-multinode-tmpl.yml": (
            "toJSON(format('nodes:{0}', inputs.node-count))"
        ),
        "collectivex-sweep.yml": "toJSON(format('nodes:{0}', matrix.nodes))",
        "speedbench-al.yml": "toJSON('nodes:1')",
    }

    priority_workflows = {
        path.name
        for path in WORKFLOWS.glob("*.yml")
        if "ci-job-" in path.read_text(encoding="utf-8")
    }
    assert priority_workflows == set(expected_node_expression)

    for name, expression in expected_node_expression.items():
        contents = workflow(name)
        assert "vars.NODE_SLOT_SCHEDULER_ENABLED == 'true'" in contents
        assert expression in contents


def test_profile_routes_validated_node_demand_through_reusable_workflows() -> None:
    contents = workflow("profile.yml")

    assert contents.count("uses: ./.github/workflows/benchmark-tmpl.yml") == 2
    assert "uses: ./.github/workflows/benchmark-multinode-tmpl.yml" in contents
    assert "node_count = entry.get('node-count', 1)" in contents
    assert "or node_count != 1" in contents
    assert "node_count = entry.get('node-count')" in contents
    assert "or node_count <= 0" in contents
    assert "node-count: ${{ matrix.config['node-count'] }}" in contents


def test_multinode_workflow_never_suppresses_an_empty_node_request() -> None:
    contents = workflow("benchmark-multinode-tmpl.yml")

    node_input = contents.split("      node-count:", 1)[1].split(
        "      priority:", 1
    )[0]
    assert "required: true" in node_input
    assert "type: number" in node_input
    assert "inputs.node-count != ''" not in contents
    assert "inputs.node-count == ''" not in contents
