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
        "profile.yml": (
            "toJSON(format('nodes:{0}', matrix.config['node-count']))"
        ),
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


def test_multinode_workflow_never_suppresses_an_empty_node_request() -> None:
    contents = workflow("benchmark-multinode-tmpl.yml")

    node_input = contents.split("      node-count:", 1)[1].split(
        "      priority:", 1
    )[0]
    assert "required: true" in node_input
    assert "type: number" in node_input
    assert "inputs.node-count != ''" not in contents
    assert "inputs.node-count == ''" not in contents


def test_b200_nscale_launcher_discovers_a_live_slurm_association() -> None:
    contents = (REPO_ROOT / "runners" / "launch_b200-nscale-slurm.sh").read_text(
        encoding="utf-8"
    )

    assert 'SLURM_ACCOUNT="restricted"' in contents
    assert 'SLURM_PARTITION="batch_2"' in contents
    assert "sacctmgr -nP show assoc" in contents
    assert "sbatch --test-only" in contents
    assert "--gpus-per-node=8" in contents
    assert 'SLURM_ACCOUNT="$account"' in contents
    assert 'SLURM_PARTITION="$partition"' in contents


def test_b200_dsv4_agentx_uses_the_0813_bundled_dspark_checkpoint() -> None:
    launcher = (REPO_ROOT / "runners" / "launch_b200-nscale-slurm.sh").read_text(
        encoding="utf-8"
    )
    assert "$NSCALE_MODEL_ROOT/DeepSeek-V4-Pro-0813" in launcher
    assert 'SRT_SLURM_MODEL_PREFIX="deepseek-v4-pro-0813"' in launcher

    recipe_dir = (
        REPO_ROOT
        / "benchmarks"
        / "multi_node"
        / "srt-slurm-recipes"
        / "sglang"
        / "deepseek-v4"
        / "agentic"
    )
    recipes = sorted(recipe_dir.glob("*b200*-mtp-*.yaml"))
    assert len(recipes) == 8

    for recipe in recipes:
        contents = recipe.read_text(encoding="utf-8")
        assert 'path: "deepseek-v4-pro-0813"' in contents
        assert "speculative-algorithm: DSPARK" in contents
        assert "speculative-draft-model-path" not in contents
        assert 'dynamo: "1.5.0.dev20260914"' in contents
        assert 'cpus-per-task: "192"' in contents
        assert 'mem: "0"' in contents
