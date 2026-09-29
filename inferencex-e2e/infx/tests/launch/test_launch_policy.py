"""Power eligibility, dcgm detection, allocation-time policy, and the tables' agreement
with the cluster inventory."""

import json

import pytest

from infx.clusters import load_clusters, load_inventory
from infx.launch import drivers, policy
from infx.launch.__main__ import launch
from infx.launch.context import LaunchError
from infx.launch.drivers import check_tables
from infx.launch.drivers.srt import models
from infx.launch.drivers.srt.power import (
    PowerPolicyError,
    decide_power,
    recipe_enables_dcgm_power,
    resolve_power,
)
from infx.launch.lifecycle import Lifecycle
from infx.launch.policy import LaunchPath, Match, salloc_time_limit
from infx.launch.request import LaunchRequest

DCGM, AGENTX, ADAPTER, ERR = "dcgm", "agentx", "adapter", "error"
MULTI, NATIVE = LaunchPath.SRT_MULTI, LaunchPath.SRT_NATIVE


def _request(**env):
    return LaunchRequest.from_env({"RUNNER_NAME": "r_0", **env})


GB200_GLM = "recipes/glm5.2/sglang/gb200-fp4/agentx/agg.yaml"
KIMI_GB200 = "recipes/kimik3/vllm/gb200-fp4/agentx/k.yaml"
KIMI_GB300 = "recipes/kimik3/vllm/gb300-fp4/agentx/deep/k.yaml"
OTHER = "recipes/other.yaml"

CASES = [
    ("gb200-nv", MULTI, "1", "glm5.2", "fp4", "dynamo-sglang", GB200_GLM, AGENTX),
    ("gb200-nv", MULTI, "1", "glm5.2", "fp4", "dynamo-sglang", OTHER, ERR),
    ("gb200-nv", MULTI, "1", "kimik3", "fp4", "dynamo-vllm", KIMI_GB200, AGENTX),
    ("gb200-nv", MULTI, "1", "kimik3", "fp4", "dynamo-vllm", KIMI_GB300, ERR),
    ("gb200-nv", MULTI, "0", "dsr1", "fp8", "dynamo-sglang", OTHER, DCGM),
    ("gb200-nv", MULTI, "0", "dsv4", "fp4", "dynamo-vllm", OTHER, ERR),
    ("gb300-nv", MULTI, "1", "kimik3", "fp4", "dynamo-vllm", KIMI_GB300, AGENTX),
    ("gb300-nv", MULTI, "1", "kimik3", "fp4", "dynamo-vllm", OTHER, ERR),
    ("gb300-nv", MULTI, "1", "glm5.2", "fp4", "dynamo-sglang", OTHER, DCGM),
    ("gb300-nv", MULTI, "0", "qwen3.5", "fp8", "dynamo-trt", OTHER, ERR),
    ("b300-dsxe", MULTI, "0", "dsv4", "fp4", "dynamo-vllm", OTHER, DCGM),
    ("b300-dsxe", MULTI, "0", "dsv4", "fp8", "dynamo-sglang", OTHER, ERR),
    ("b300-dsxe", MULTI, "1", "dsv4", "fp4", "dynamo-sglang", OTHER, ERR),
    ("b200-nscale", NATIVE, "1", "kimik3", "fp4", "dynamo-vllm", OTHER, AGENTX),
    ("b200-nscale", NATIVE, "0", "dsv4", "fp4", "dynamo-sglang", OTHER, DCGM),
    ("b200-nscale", NATIVE, "0", "glm5.2", "fp4", "dynamo-vllm", OTHER, ERR),
    ("b200-nscale", MULTI, "1", "qwen3.5", "fp8", "dynamo-sglang", OTHER, AGENTX),
    ("b200-nscale", MULTI, "1", "kimik3", "fp4", "dynamo-vllm", OTHER, ERR),
    ("b200-nscale", MULTI, "0", "dsv4", "fp4", "dynamo-vllm", OTHER, DCGM),
    ("b200-nscale", MULTI, "0", "dsv4", "fp4", "dynamo-sglang", OTHER, ERR),
    ("h200-dgxc", MULTI, "1", "kimik3", "fp4", "vllm", OTHER, AGENTX),
    ("h200-dgxc", MULTI, "1", "dsv4", "fp8", "dynamo-sglang", OTHER, ADAPTER),
    ("h200-dgxc", MULTI, "0", "dsv4", "fp8", "dynamo-sglang", OTHER, ERR),
    ("h200-dgxc", MULTI, "1", "kimik3", "fp4", "dynamo-vllm", OTHER, ERR),
]


@pytest.mark.parametrize(
    ("cluster", "path", "agentic", "prefix", "precision", "framework", "recipe", "expected"), CASES
)
def test_power_eligibility_of_each_cluster_lane(
    cluster, path, agentic, prefix, precision, framework, recipe, expected
):
    request = _request(
        IS_AGENTIC=agentic, MODEL_PREFIX=prefix, PRECISION=precision, FRAMEWORK=framework
    )
    if expected == ERR:
        with pytest.raises(PowerPolicyError):
            decide_power(cluster, path, dcgm=True, request=request, recipe=recipe)
        return
    decision = decide_power(cluster, path, dcgm=True, request=request, recipe=recipe)
    assert (decision.dcgm, decision.agentx, decision.adapter) == (
        True, expected == AGENTX, expected == ADAPTER,
    )  # fmt: skip
    assert not decide_power(cluster, path, dcgm=False, request=request, recipe=recipe).dcgm


def test_gb200_reports_agentic_misses_distinctly():
    request = _request(IS_AGENTIC="1", MODEL_PREFIX="dsv4", PRECISION="fp4", FRAMEWORK="dynamo-sglang")
    with pytest.raises(PowerPolicyError, match="AgentX dcgm-power requires the GLM-5.2"):
        decide_power("gb200-nv", MULTI, dcgm=True, request=request, recipe=OTHER)


@pytest.mark.parametrize(
    ("text", "enabled"),
    [
        ("telemetry:\n  dcgm_exporter:\n    image: x\n  enabled: true\n", True),
        ("telemetry:\n  dcgm_exporter:\nbenchmark:\n  enabled: true\n", False),
        ("benchmark:\n  dcgm_exporter:\n  enabled: true\n", False),
        ("telemetry:\n  dcgm_exporter:\n    enabled: true\n", False),
        ("telemetry: [unclosed\n  enabled: true\n", False),
    ],
)
def test_dcgm_detection_is_scoped_to_telemetry_block(text, enabled):
    assert recipe_enables_dcgm_power(text) is enabled


def _mirror(tmp_path, rel, text):
    path = tmp_path / "benchmarks/multi_node/srt-slurm-recipes" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


POWER_RECIPE = "telemetry:\n  dcgm_exporter:\n  enabled: true\n"


def test_nscale_eval_only_inspects_eval_recipe(tmp_path):
    _mirror(tmp_path, "dsv4/eval.yaml", POWER_RECIPE)
    _mirror(tmp_path, "dsv4/bench.yaml", "model: {}\n")
    env = dict(
        GITHUB_WORKSPACE=str(tmp_path), IS_AGENTIC="0", MODEL_PREFIX="dsv4", PRECISION="fp4",
        FRAMEWORK="dynamo-vllm", CONFIG_FILE="recipes/dsv4/bench.yaml:zip",
        EVAL_CONFIG_FILE="recipes/dsv4/eval.yaml",
    )
    assert resolve_power("b200-nscale", MULTI, _request(**env, EVAL_ONLY="true")).dcgm
    assert not resolve_power("b200-nscale", MULTI, _request(**env, EVAL_ONLY="false")).dcgm
    assert not resolve_power("b300-dsxe", MULTI, _request(**env, EVAL_ONLY="true")).dcgm


def test_upstream_only_recipe_stays_non_power(tmp_path):
    request = _request(GITHUB_WORKSPACE=str(tmp_path), CONFIG_FILE="recipes/missing.yaml",
                       IS_AGENTIC="0", FRAMEWORK="dynamo-trt")
    assert not resolve_power("gb300-nv", MULTI, request).dcgm


AGENTX_FLASH = dict(MODEL_PREFIX="dsv41flash", FRAMEWORK="sglang", IS_MULTINODE="false",
                    IS_AGENTIC="1", EVAL_ONLY="false", SALLOC_TIME_LIMIT="480")


@pytest.mark.parametrize(
    ("cluster", "overrides", "expected"),
    [
        ("h200-dgxc", {"CONC": "64"}, 1440),
        ("h200-dgxc", {"CONC": "63"}, 480),
        ("h200-dgxc", {"CONC": "64", "EVAL_ONLY": "true"}, 480),
        ("h200-dgxc", {"CONC": "64", "FRAMEWORK": "vllm"}, 480),
        ("b300-dsxe", {"CONC": "64"}, 480),
    ],
)
def test_salloc_time_bumps(cluster, overrides, expected):
    assert salloc_time_limit(cluster, _request(**{**AGENTX_FLASH, **overrides})) == expected


def test_workload_tables_agree_with_the_checked_in_inventory():
    check_tables(load_clusters())


RECORD = {"gpus-per-node": 8, "arch": "x86_64", "scheduler": "slurm",
          "slurm": {"partition": "p", "exclusive": True}}  # fmt: skip


def test_every_row_the_inventory_contradicts_is_reported_at_once(monkeypatch):
    clusters = load_inventory({"labels": {"cluster:c": ["c_0"]}, "clusters": {"c": RECORD}}).clusters
    monkeypatch.setitem(policy.LEGACY_TILERT, "c", policy.LEGACY_TILERT["b200-nscale"])
    monkeypatch.setitem(models.OVERRIDES, "c", ())
    with pytest.raises(LaunchError) as raised:
        check_tables(clusters)
    problems = str(raised.value)
    assert "LEGACY_TILERT['c']" in problems and "OVERRIDES['c']" in problems


def test_a_launch_checks_its_own_cluster_rows_before_any_work(monkeypatch, capsys):
    labels = {"cluster:c": ["c_0"], "cluster:d": ["d_0"]}
    clusters = load_inventory({"labels": labels, "clusters": {"c": RECORD, "d": RECORD}}).clusters
    monkeypatch.setitem(policy.LEGACY_TILERT, "d", policy.LEGACY_TILERT["b200-nscale"])

    class ReachedBackend(Exception):
        pass

    def backend_class(_scheduler):
        raise ReachedBackend

    monkeypatch.setattr(drivers, "backend_class", backend_class)
    with pytest.raises(ReachedBackend), Lifecycle() as life:
        drivers.run(clusters["c"], _request(IS_MULTINODE="false"), life)

    monkeypatch.setitem(policy.LEGACY_TILERT, "c", policy.LEGACY_TILERT["b200-nscale"])
    assert launch(clusters["c"], _request(IS_MULTINODE="false")) == 1
    assert "LEGACY_TILERT['c']" in capsys.readouterr().err


HOST = {"TILERT": "host", "CLUSTER": "host", "LATER": "host", "SHARED": "host"}


@pytest.mark.parametrize(("framework", "chosen", "expected"), [
    ("tilert", {}, {"TILERT": "tilert", "CLUSTER": "cluster", "LATER": "later", "SHARED": "later"}),
    ("tilert", {"TILERT": "point", "SHARED": "point"},
     {"TILERT": "point", "CLUSTER": "cluster", "LATER": "later", "SHARED": "point"}),
    ("sglang", {}, {"TILERT": "host", "CLUSTER": "cluster", "LATER": "later", "SHARED": "later"}),
])  # fmt: skip
def test_cluster_settings_beat_the_host_but_never_the_points_settings(monkeypatch, framework, chosen, expected):
    record = {**RECORD, "env": {"CLUSTER": "cluster", "SHARED": "cluster"}}
    cluster = load_inventory({"labels": {"cluster:c": ["c_0"]}, "clusters": {"c": record}}).clusters["c"]
    monkeypatch.setitem(policy.TILERT_ENV, "c", {"TILERT": "tilert", "SHARED": "tilert"})
    settings = json.dumps([f"{name}={value}" for name, value in chosen.items()])
    request = _request(FRAMEWORK=framework, DECODE_ADDITIONAL_SETTINGS=settings, **{**HOST, **chosen})

    env = policy.runtime_env(cluster, request, {"LATER": "later", "SHARED": "later"})

    assert {name: env[name] for name in HOST} == expected
