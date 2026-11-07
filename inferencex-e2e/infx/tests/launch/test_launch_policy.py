"""Power eligibility, allocation-time policy, and the tables' agreement with the cluster inventory."""

import json

import pytest

from infx.clusters import load_inventory
from infx.clusters.slurm import SrtSlurmSettings
from infx.launch import drivers, policy
from infx.launch.__main__ import launch
from infx.launch.context import LaunchError
from infx.launch.drivers import check_tables
from infx.launch.drivers.srt import models, power
from infx.launch.drivers.srt.power import PowerPolicyError, decide_power
from infx.launch.lifecycle import Lifecycle
from infx.launch.policy import LaunchPath, salloc_time_limit
from infx.launch.request import LaunchRequest, MultiNodeRequest, RequestError

MULTI = LaunchPath.SRT_MULTI


def _request(**env):
    return LaunchRequest.from_env({"RUNNER_NAME": "r_0", **env})


def _multinode(**env):
    return MultiNodeRequest.from_env({
        "RUNNER_NAME": "r_0", "IMAGE": "i", "SPEC_DECODING": "none", "RESULT_FILENAME": "r",
        "RUN_EVAL": "false", "EVAL_ONLY": "false", "GITHUB_WORKSPACE": "/ws", "SRT_RECIPE": "r.yaml",
        **env,
    })


def _srt(**settings) -> SrtSlurmSettings:
    return SrtSlurmSettings.model_validate({"network-interface": "", **settings})


EXPORTER = {"extra": {"default_gpu_exporter": {"container_image": "exporter:test"}}}
GPU_ONLY = _srt(**EXPORTER, **{"power-exporter-port": 9401})
WITH_CPU = _srt(**EXPORTER, **{"power-exporter-port": 9401, "cpu-power-exporter": {"source": "acpi"}})


def _point(agentic: str, prefix: str = "dsv4", framework: str = "dynamo-vllm", **env):
    if agentic == "1":
        env.update(CONC="4", CONC_LIST="4", IS_MULTINODE="true")
    return _multinode(
        IS_AGENTIC=agentic, MODEL_PREFIX=prefix, PRECISION="fp8", FRAMEWORK=framework, **env
    )


@pytest.mark.parametrize(("srt", "cpu"), [(GPU_ONLY, False), (WITH_CPU, True)])
@pytest.mark.parametrize("agentic", ["0", "1"])
def test_a_cluster_with_a_gpu_exporter_measures_every_benchmark_point_without_failing_it(
    srt, cpu, agentic
):
    point = _point(agentic)
    assert decide_power("c", MULTI, srt, point) == power.PowerDecision(
        dcgm=True, agentx=agentic == "1", cpu=cpu, best_effort=True
    )
    skipped = decide_power("c", MULTI, srt, point, lambda: True)
    assert skipped == power.skip_power("dedicated_discovery")
    evaluation = _point(agentic, EVAL_ONLY="true")
    assert decide_power("c", MULTI, srt, evaluation) == power.skip_power("eval_only")


def test_power_the_cluster_cannot_measure_is_refused_only_when_the_point_asks_for_it():
    no_exporter = _srt()
    skipped = decide_power("c", MULTI, no_exporter, _point("1"))
    assert skipped == power.skip_power("no_gpu_exporter")
    with pytest.raises(PowerPolicyError, match="measures no GPU power"):
        decide_power("c", MULTI, no_exporter, _point("1", POWER="1"))
    with pytest.raises(PowerPolicyError, match="no srt-slurm.cpu-power-exporter"):
        decide_power("c", MULTI, GPU_ONLY, _point("1", POWER="1", CPU_POWER="1"))


def test_the_power_field_keeps_invalid_power_fatal():
    explicit = _point("1", POWER="1", CPU_POWER="1")
    assert decide_power("c", MULTI, WITH_CPU, explicit) == power.PowerDecision(
        dcgm=True, agentx=True, cpu=True, expect_cpu=True
    )
    assert decide_power("c", MULTI, WITH_CPU, _point("1", POWER="1"), lambda: True).dcgm
    workflow = _point("1", REQUIRE_POWER="1")
    assert not decide_power("c", MULTI, WITH_CPU, workflow).best_effort


def test_a_lane_rule_changes_how_its_matching_agentx_points_measure_power():
    matching = _point("1", prefix="glm5.2", framework="dynamo-sglang")
    decision = decide_power("h200-dgxc", MULTI, GPU_ONLY, matching)
    assert (decision.agentx, decision.adapter) == (False, True)
    other = decide_power("h200-dgxc", MULTI, GPU_ONLY, _point("1", framework="dynamo-sglang"))
    assert (other.agentx, other.adapter) == (True, False)


def test_a_power_lane_needs_its_clusters_gpu_exporter(monkeypatch):
    labels = {"cluster:c": ["c_0"]}
    srt = {"network-interface": ""}
    record = {**RECORD, "slurm": {**RECORD["slurm"], "srt-slurm": srt}}
    clusters = load_inventory({"labels": labels, "clusters": {"c": record}}).clusters
    monkeypatch.setitem(drivers.srt.lanes.SRT_LANES, ("c", MULTI), drivers.srt.lanes.SrtLane())
    monkeypatch.setitem(power.POWER_LANES, ("c", MULTI), power.PowerLane(rules=()))
    assert drivers.srt.table_problems(clusters, "c") == [
        "POWER_LANES['c', srt-multi]: no srt-slurm.extra.default_gpu_exporter"
    ]
    srt.update(EXPORTER)
    clusters = load_inventory({"labels": labels, "clusters": {"c": record}}).clusters
    assert drivers.srt.table_problems(clusters, "c") == []


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


RECORD = {"gpus-per-node": 8, "arch": "x86_64", "scheduler": "slurm",
          "slurm": {"partition": "p", "exclusive": True}}  # fmt: skip


def test_every_row_the_inventory_contradicts_is_reported_at_once(monkeypatch):
    clusters = load_inventory({"labels": {"cluster:c": ["c_0"]}, "clusters": {"c": RECORD}}).clusters
    monkeypatch.setitem(policy.SALLOC_TIME_BUMPS, "missing-cluster", None)
    monkeypatch.setitem(models.OVERRIDES, "c", ())
    with pytest.raises(LaunchError) as raised:
        check_tables(clusters)
    problems = str(raised.value)
    assert "SALLOC_TIME_BUMPS['missing-cluster']: no such cluster" in problems
    assert "OVERRIDES['c']" in problems


def test_a_launch_checks_its_own_cluster_rows_before_any_work(monkeypatch, capsys):
    labels = {"cluster:c": ["c_0"], "cluster:d": ["d_0"]}
    clusters = load_inventory({"labels": labels, "clusters": {"c": RECORD, "d": RECORD}}).clusters
    monkeypatch.setitem(models.OVERRIDES, "d", ())

    class ReachedBackend(Exception):
        pass

    def backend_class(_scheduler):
        raise ReachedBackend

    monkeypatch.setattr(drivers, "backend_class", backend_class)
    with pytest.raises(ReachedBackend), Lifecycle() as life:
        drivers.run(clusters["c"], _request(IS_MULTINODE="false"), life)

    monkeypatch.setitem(models.OVERRIDES, "c", ())
    assert launch(clusters["c"], _request(IS_MULTINODE="false")) == 1
    assert "OVERRIDES['c']" in capsys.readouterr().err


HOST = {"CLUSTER": "host", "LATER": "host", "SHARED": "host"}


@pytest.mark.parametrize(("chosen", "expected"), [
    ({}, {"CLUSTER": "cluster", "LATER": "later", "SHARED": "later"}),
    ({"SHARED": "point"}, {"CLUSTER": "cluster", "LATER": "later", "SHARED": "point"}),
])  # fmt: skip
def test_cluster_settings_beat_the_host_but_never_the_points_settings(chosen, expected):
    record = {**RECORD, "env": {"CLUSTER": "cluster", "SHARED": "cluster"}}
    cluster = load_inventory({"labels": {"cluster:c": ["c_0"]}, "clusters": {"c": record}}).clusters["c"]
    settings = json.dumps([f"{name}={value}" for name, value in chosen.items()])
    request = _request(DECODE_ADDITIONAL_SETTINGS=settings, **{**HOST, **chosen})

    env = policy.runtime_env(cluster, request, {"LATER": "later", "SHARED": "later"})

    assert {name: env[name] for name in HOST} == expected


def test_a_role_without_additional_settings_arrives_as_null_and_sets_no_names():
    record = {**RECORD, "env": {"CLUSTER": "cluster", "SHARED": "cluster"}}
    cluster = load_inventory({"labels": {"cluster:c": ["c_0"]}, "clusters": {"c": record}}).clusters["c"]
    request = _request(
        PREFILL_ADDITIONAL_SETTINGS='["SHARED=point"]', DECODE_ADDITIONAL_SETTINGS="null",
        **{**HOST, "SHARED": "point"},
    )

    env = policy.runtime_env(cluster, request, {"LATER": "later", "SHARED": "later"})

    assert {name: env[name] for name in HOST} == {"CLUSTER": "cluster", "LATER": "later", "SHARED": "point"}


@pytest.mark.parametrize(("settings", "kind"), [("5", "int"), ("true", "bool"), ('{"SHARED=point": 1}', "dict")])
def test_additional_settings_that_are_not_a_list_are_rejected(settings, kind):
    record = {**RECORD, "env": {"CLUSTER": "cluster", "SHARED": "cluster"}}
    cluster = load_inventory({"labels": {"cluster:c": ["c_0"]}, "clusters": {"c": record}}).clusters["c"]
    request = _request(DECODE_ADDITIONAL_SETTINGS=settings, **HOST)

    with pytest.raises(RequestError, match=f"^DECODE_ADDITIONAL_SETTINGS must be a JSON list .*, got {kind}$"):
        policy.runtime_env(cluster, request, {"LATER": "later", "SHARED": "later"})
