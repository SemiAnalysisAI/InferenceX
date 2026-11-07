"""Power eligibility, allocation-time policy, and the platforms' agreement with the cluster inventory."""

import json

import pytest
from pydantic import ValidationError

from infx.clusters import load_inventory
from infx.clusters.platforms import Match, Override, Platform
from infx.clusters.slurm import SrtSlurmSettings
from infx.launch import drivers, policy
from infx.launch.__main__ import launch
from infx.launch.drivers.srt import power
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
    assert decide_power("c", MULTI, srt, point, lambda: True) == power.NO_POWER
    evaluation = _point(agentic, EVAL_ONLY="true")
    assert decide_power("c", MULTI, srt, evaluation) == power.NO_POWER


def test_power_the_cluster_cannot_measure_is_refused_only_when_the_point_asks_for_it():
    no_exporter = _srt()
    assert decide_power("c", MULTI, no_exporter, _point("1")) == power.NO_POWER
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


def _cluster(srt=None, **record):
    slurm = {**RECORD["slurm"], **({"srt-slurm": srt} if srt is not None else {}), **record.pop("slurm", {})}
    inventory = {"labels": {"cluster:c": ["c_0"]}, "clusters": {"c": {**RECORD, **record, "slurm": slurm}}}
    return load_inventory(inventory).clusters["c"]


CONTRADICTED = Platform.model_validate({
    "srt-lanes": {"srt-multi": {
        "mounts": [{"when": {}, "volume": "wheels"}], "shared-run-root": [{}], "time-limit": "4:00:00",
    }},
    "salloc-time-bump": {"when": {}, "min-conc": 64, "minutes": 1440},
    "model-overrides": [{"when": {}, "entry": "M"}],
    "shared-hf-cache-lanes": [{"agentic": True}],
})


def test_every_platform_row_its_cluster_record_contradicts_is_reported_at_once():
    fixed = {"network-interface": "", "default-time-limit": "6:00:00"}
    assert CONTRADICTED.problems(_cluster(fixed)) == [
        "srt-lanes.srt-multi: no volume 'wheels'",
        "srt-lanes.srt-multi: no srt-slurm.shared-run-root",
        "srt-lanes.srt-multi: its time limit is shadowed by default-time-limit",
        "model-overrides: no models.entries 'M'",
        "shared-hf-cache-lanes: no volume 'shared-hf-hub-cache'",
        "salloc-time-bump: shadowed by the cluster's fixed srt-slurm default-time-limit or "
        "single-node-time-limit",
    ]
    assert CONTRADICTED.problems(_cluster()) == [
        "srt-lanes: no slurm.srt-slurm profile",
        "model-overrides: no slurm.srt-slurm profile",
        "shared-hf-cache-lanes: no slurm.srt-slurm profile",
    ]
    volumes = {"wheels": {"path": "/wheels"}, "shared-hf-hub-cache": {"path": "/hub"}}
    complete = _cluster(
        {"network-interface": "", "shared-run-root": "/runs"},
        models={"entries": {"M": {"root": "wheels", "dir": "m"}}},
        slurm={"volumes": volumes},
    )
    assert CONTRADICTED.problems(complete) == []


def test_native_rules_without_the_native_lane_fail_to_load():
    with pytest.raises(ValidationError, match="missing srt-lanes.srt-native"):
        Platform.model_validate({"srt-lanes": {"srt-multi": {}}, "native-srt-lanes": [{}]})


def test_a_launch_checks_its_own_cluster_rows_before_any_work(monkeypatch, capsys):
    labels = {"cluster:c": ["c_0"], "cluster:d": ["d_0"]}
    clusters = load_inventory({"labels": labels, "clusters": {"c": RECORD, "d": RECORD}}).clusters
    contradicted = Platform(model_overrides=(Override(when=Match()),))
    monkeypatch.setitem(policy.PLATFORMS, "d", contradicted)

    class ReachedBackend(Exception):
        pass

    def backend_class(_scheduler):
        raise ReachedBackend

    monkeypatch.setattr(drivers, "backend_class", backend_class)
    with pytest.raises(ReachedBackend), Lifecycle() as life:
        drivers.run(clusters["c"], _request(IS_MULTINODE="false"), life)

    monkeypatch.setitem(policy.PLATFORMS, "c", contradicted)
    assert launch(clusters["c"], _request(IS_MULTINODE="false")) == 1
    assert "configs/platforms/c.yaml disagrees" in capsys.readouterr().err


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
