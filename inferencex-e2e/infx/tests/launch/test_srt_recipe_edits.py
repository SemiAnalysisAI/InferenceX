"""Edits the multi-node lanes apply to the bound recipe."""

import pytest
import yaml

from infx.launch.drivers.srt.lanes import srt_lane
from infx.launch.drivers.srt.recipe import eval_overrides, prepare_recipe
from infx.launch.policy import LaunchPath
from infx.launch.request import LaunchRequest


def test_the_job_names_the_run_and_gives_each_watchdog_role_the_dist_timeout(tmp_path):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump({
        "schema": 2, "name": "upstream",
        "roles": {
            "prefill": {"name": "keep-me", "args": {"watchdog-timeout": 600, "tp": 4}},
            "decode": {"args": {"tp": 4}},
        },
        "benchmark": {"type": "sa-bench"},
    }, sort_keys=False))  # fmt: skip

    prepare_recipe(recipe, "runner_00", 1800)

    assert yaml.safe_load(recipe.read_text()) == {
        "schema": 2, "name": "runner_00",
        "roles": {
            "prefill": {"name": "keep-me", "args": {"watchdog-timeout": 600, "tp": 4, "dist-timeout": 1800}},
            "decode": {"args": {"tp": 4}},
        },
        "benchmark": {"type": "sa-bench"},
    }  # fmt: skip


@pytest.mark.parametrize(("cluster", "prefix", "framework", "agentic", "eval_only", "expected"), [
    ("b300-dsxe", "dsv41flash", "dynamo-vllm", "1", "true", ["--set", "frontend.placement.node=head"]),
    ("b300-dsxe", "dsv41flash", "dynamo-vllm", "1", "false", []),
    ("b300-dsxe", "dsv41flash", "dynamo-sglang", "1", "true", []),
    ("b300-dsxe", "kimik3", "dynamo-vllm", "1", "true", []),
    ("b300-dsxe", "dsv41flash", "dynamo-vllm", "0", "true", []),
    ("gb300-nv", "dsv4", "dynamo-trt", "1", "true", ["--set", "frontend.placement.node=head"]),
    ("gb300-nv", "dsv4", "dynamo-vllm", "1", "true", []),
])  # fmt: skip
def test_eval_frontend_overrides_preserve_benchmark_and_other_workloads(
    tmp_path, cluster, prefix, framework, agentic, eval_only, expected
):
    request = LaunchRequest.from_env({
        "RUNNER_NAME": "r_0", "GITHUB_WORKSPACE": str(tmp_path),
        "MODEL_PREFIX": prefix, "FRAMEWORK": framework,
        "IS_AGENTIC": agentic, "IS_MULTINODE": "true", "EVAL_ONLY": eval_only,
        "CONC": "1", "CONC_LIST": "1",
    })

    assert eval_overrides(tmp_path, srt_lane(cluster, LaunchPath.SRT_MULTI), request) == expected
