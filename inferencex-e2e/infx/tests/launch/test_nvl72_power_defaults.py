"""NVL72 native telemetry follows the cluster, not a model or recipe allowlist."""

from pathlib import Path

import pytest

from infx.launch.drivers.srt.power import PowerPolicyError, decide_power
from infx.launch.policy import LaunchPath
from infx.launch.request import MultiNodeRequest
from infx.srt_slurm.workload import compose_recipe

ROOT = Path(__file__).resolve().parents[3]


def request(**values):
    return MultiNodeRequest.from_env(
        {
            "RUNNER_NAME": "fixture_0",
            "IMAGE": "image",
            "SPEC_DECODING": "none",
            "RESULT_FILENAME": "result",
            "RUN_EVAL": "false",
            "EVAL_ONLY": "false",
            "GITHUB_WORKSPACE": "/tmp",
            "SRT_RECIPE": "recipe.yaml",
            "IS_AGENTIC": "1",
            "MODEL_PREFIX": "future-model",
            "PRECISION": "fp4",
            "FRAMEWORK": "dynamo-vllm",
            "CONC": "4",
            "CONC_LIST": "4",
            **values,
        }
    )


@pytest.mark.parametrize("cluster", ["gb200-nv", "gb300-nv"])
def test_default_power_for_new_agentic_model(cluster):
    decision = decide_power(cluster, LaunchPath.SRT_MULTI, request(POWER="0"))
    assert decision.dcgm and decision.agentx and decision.cpu
    assert not decision.require
    evaluation = decide_power(cluster, LaunchPath.SRT_MULTI, request(EVAL_ONLY="true"))
    assert not evaluation.dcgm


def test_other_cluster_keeps_explicit_power_policy():
    assert not decide_power("h200-dgxc", LaunchPath.SRT_MULTI, request(POWER="0")).dcgm


def test_existing_strict_agentx_rule_stays_strict():
    decision = decide_power(
        "gb200-nv",
        LaunchPath.SRT_MULTI,
        request(POWER="1", MODEL_PREFIX="kimik3", FRAMEWORK="dynamo-vllm"),
    )
    assert decision.dcgm and decision.agentx and decision.cpu and decision.require


def test_explicit_requirement_uses_default_nvl72_collector_and_skips_evals():
    strict = request(REQUIRE_POWER="1", POWER="0")
    decision = decide_power("gb200-nv", LaunchPath.SRT_MULTI, strict)
    assert decision.dcgm and decision.agentx and decision.cpu and decision.require
    evaluation = decide_power("gb200-nv", LaunchPath.SRT_MULTI,
                              request(REQUIRE_POWER="1", EVAL_ONLY="true"))  # fmt: skip
    assert not evaluation.dcgm and not evaluation.require


def test_explicit_requirement_without_collector_fails():
    with pytest.raises(PowerPolicyError, match="eligible power collector"):
        decide_power("h200-dgxc", LaunchPath.SRT_MULTI, request(REQUIRE_POWER="1", POWER="0"))


@pytest.mark.parametrize("hardware", ["gb200", "gb300"])
@pytest.mark.parametrize("precision", ["fp4", "fp8"])
def test_composed_dsr1_recipe_places_power_client_on_head(hardware, precision):
    path = (
        ROOT
        / "benchmarks/multi_node/srt-slurm-recipes/dsr1/trtllm"
        / (f"{hardware}-{precision}/8k1k/disagg-variants.yaml")
    )
    recipe = compose_recipe(path, agentic=False, multinode=True, root=ROOT, power_port=9401,
                            cpu_power_source="acpi")  # fmt: skip
    base = recipe["base"]
    assert base["telemetry"]["enabled"]
    assert base["telemetry"]["cpu_power_exporter"]["source"] == "acpi"
    assert base["frontend"]["placement"]["node"] == "head"
    assert base["benchmark"]["placement"]["node"] == "head"
    assert all(service["placement"]["node"] == "head" for service in base["services"])
