"""Launch-path, model, cache and time-limit policy for srt-slurm lanes."""

import pytest

from infx.clusters import Cluster, load_inventory
from infx.clusters.slurm import SrtSlurmSettings
from infx.launch import policy
from infx.launch.context import LaunchError
from infx.launch.drivers.srt import models
from infx.launch.drivers.srt.lanes import SrtLane, srt_lane, srt_time_limit
from infx.launch.drivers.srt.models import (
    ModelRule,
    StagedBasenames,
    resolve_model,
    single_node_hf_cache,
    single_node_model_path,
)
from infx.launch.policy import LaunchPath, Match, TimeBump, any_of, launch_path, runtime_env
from infx.launch.request import LaunchRequest


def request(**env: str) -> LaunchRequest:
    """A launch request from ``env`` (RUNNER_NAME and a workspace preset)."""
    return LaunchRequest.from_env({"RUNNER_NAME": "r_0", "GITHUB_WORKSPACE": "/ws", **env})


def cluster(tmp_path) -> Cluster:
    """Cluster ``c``: checkpoint ``M`` on shared storage, a node-local copy, and two Hub caches."""
    record = {
        "gpus-per-node": 8, "arch": "x86_64", "scheduler": "slurm",
        "models": {"entries": {"M": {"root": "shared", "dir": "m"}, "M@nvme": {"root": "nvme", "dir": "m"}}},
        "slurm": {"partition": "p", "exclusive": True, "volumes": {
            "shared": {"path": str(tmp_path / "shared")},
            "nvme": {"path": str(tmp_path / "nvme"), "visibility": "node-local"},
            "hf-hub-cache": {"path": str(tmp_path / "hub"), "visibility": "node-local"},
            "shared-hf-hub-cache": {"path": str(tmp_path / "shared-hub")},
        }},
    }  # fmt: skip
    return load_inventory({"labels": {"cluster:c": ["c_0"]}, "clusters": {"c": record}}).clusters["c"]


MULTI = dict(IS_MULTINODE="true", CONFIG_FILE="recipes/x.yaml")
SINGLE = dict(IS_MULTINODE="false")


@pytest.mark.parametrize(("cluster_id", "env", "path"), [
    ("b200-nscale", dict(MULTI, MODEL_PREFIX="dsv4", PRECISION="fp4", FRAMEWORK="dynamo-sglang", SPEC_DECODING="mtp"), LaunchPath.SRT_NATIVE),
    ("b200-nscale", dict(MULTI, MODEL_PREFIX="dsv4", PRECISION="fp4", FRAMEWORK="dynamo-sglang", SPEC_DECODING="eagle"), LaunchPath.SRT_MULTI),
    ("b200-nscale", dict(MULTI, MODEL_PREFIX="kimik3", PRECISION="fp4", FRAMEWORK="dynamo-vllm"), LaunchPath.SRT_NATIVE),
    ("b200-nscale", dict(MULTI, MODEL_PREFIX="dsr1", PRECISION="fp8", FRAMEWORK="dynamo-trt"), LaunchPath.SRT_MULTI),
    ("b200-nscale", dict(MULTI, MODEL_PREFIX="glm5.1", PRECISION="fp8", FRAMEWORK="tilert", SPEC_DECODING="mtp", IS_AGENTIC="1"), LaunchPath.SRT_NATIVE),
    ("b200-nscale", dict(MULTI, MODEL_PREFIX="glm5.1", PRECISION="fp8", FRAMEWORK="tilert", SPEC_DECODING="mtp", IS_AGENTIC="0"), LaunchPath.LEGACY_TILERT),
    ("mi355x-amds", dict(IS_MULTINODE="true", FRAMEWORK="atom-disagg"), LaunchPath.LEGACY_AMD_UTILS),
    ("mi355x-amds", dict(MULTI, FRAMEWORK="sglang-disagg"), LaunchPath.SRT_MULTI),
    ("gb200-nv", dict(MULTI, FRAMEWORK="tilert"), LaunchPath.SRT_MULTI),
    ("b300-dsxe", dict(SINGLE, MODEL_PREFIX="dsv41flash", FRAMEWORK="sglang", IS_AGENTIC="1"), LaunchPath.SRT_BATCH),
    ("b300-dsxe", dict(SINGLE, MODEL_PREFIX="dsv41flash", FRAMEWORK="sglang", IS_AGENTIC="1", INFX_BATCH_REENTRY="1"), LaunchPath.SRT_SINGLE),
    ("b300-dsxe", dict(SINGLE, MODEL_PREFIX="dsv41flash", FRAMEWORK="vllm", IS_AGENTIC="1"), LaunchPath.SRT_SINGLE),
    ("gb200-nv", dict(SINGLE, MODEL_PREFIX="dsv41flash", FRAMEWORK="sglang", IS_AGENTIC="1"), LaunchPath.SRT_SINGLE),
    ("b300-dsxe", dict(SINGLE, BENCH_SCRIPT_OVERRIDE="benchmarks/single_node/speedbench/x.sh"), LaunchPath.SCRIPT),
    ("b300-dsxe", dict(MULTI, FRAMEWORK="dynamo-trt", BENCH_SCRIPT_OVERRIDE="x.sh"), LaunchPath.SRT_MULTI),
])  # fmt: skip
def test_each_cluster_routes_requests_to_its_launch_path(cluster_id, env, path):
    assert launch_path(cluster_id, request(**env)) is path


def test_a_path_without_a_lane_on_the_cluster_is_refused():
    with pytest.raises(LaunchError, match="cluster 'b200-cw' has no srt-multi srt-slurm lane"):
        srt_lane("b200-cw", LaunchPath.SRT_MULTI)


def test_the_first_matching_rule_names_the_checkpoint_and_its_aliases(tmp_path):
    c = cluster(tmp_path)
    rules = (
        ModelRule(Match(any_of("m"), dcgm=True), "m-power", "M@nvme"),
        ModelRule(Match(any_of("m")), "m", "M", served_name="org/M", extra_aliases=("m-alt",)),
    )
    power = resolve_model(c, rules, request(MODEL_PREFIX="m"), dcgm=True)
    plain = resolve_model(c, rules, request(MODEL_PREFIX="m"))

    assert (power.path, power.aliases) == (f"{tmp_path}/nvme/m", ("m-power",))
    assert (plain.path, plain.aliases, plain.served_name) == (f"{tmp_path}/shared/m", ("m", "m-alt"), "org/M")
    assert resolve_model(c, rules, request(MODEL_PREFIX="other")) is None


def test_a_rule_can_defer_to_an_exported_path_the_hub_or_a_readable_config(tmp_path):
    c = cluster(tmp_path)
    exported = tmp_path / "exported"
    exported.mkdir()
    rules = (
        ModelRule(Match(any_of("env")), None, "M", env_path=True),
        ModelRule(Match(any_of("dir")), None, "M", env_path_if_dir=True),
        ModelRule(Match(any_of("hub")), None, "M", hf_fallback="org/M"),
        ModelRule(Match(any_of("config")), None, "M", require_config=True),
    )

    def path(prefix: str, **env: str) -> str:
        return resolve_model(c, rules, request(MODEL_PREFIX=prefix, **env)).path

    staged = f"{tmp_path}/shared/m"
    assert path("env", MODEL_PATH="/absent") == "/absent"
    assert path("dir", MODEL_PATH=str(exported)) == str(exported)
    assert path("dir", MODEL_PATH="/absent") == staged
    assert path("hub") == "hf:org/M"
    with pytest.raises(ValueError, match="config.json"):
        path("config")
    (tmp_path / "shared/m").mkdir(parents=True)
    (tmp_path / "shared/m/config.json").write_text("{}")
    assert path("hub") == path("config") == staged


def test_single_node_points_read_staged_checkpoints_or_the_hub(tmp_path, monkeypatch):
    c = cluster(tmp_path)
    copies = ((Match(frameworks=any_of("vllm"), model_glob="*/M"), "M@nvme"),)
    monkeypatch.setitem(models.SINGLE_NODE_BASENAMES, "c", StagedBasenames("shared", any_of("org/Hub"), copies))

    def path(model: str, **env: str) -> str:
        return single_node_model_path(c, request(MODEL=model, **env))

    # Found by HF basename: a Hub download, a copy for some requests, the entry, or the fallback root.
    assert path("org/Hub") == "hf:org/Hub"
    assert path("org/M", FRAMEWORK="vllm") == f"{tmp_path}/nvme/m"
    assert path("org/M", FRAMEWORK="sglang") == f"{tmp_path}/shared/m"
    assert path("org/Other") == f"{tmp_path}/shared/Other"
    # Clusters with model rules serve a staged checkpoint, else hf:<MODEL>, and refuse the rest.
    rules = (ModelRule(Match(any_of("staged")), None, "M"), ModelRule(Match(any_of("hub")), None, None))
    monkeypatch.setitem(models.SINGLE_NODE_MODELS, "c", rules)
    assert path("org/M", MODEL_PREFIX="staged") == f"{tmp_path}/shared/m"
    assert path("org/M", MODEL_PREFIX="hub") == "hf:org/M"
    with pytest.raises(ValueError, match="unsupported model"):
        path("org/M", MODEL_PREFIX="other")


def test_matching_single_node_points_read_the_shared_hub_cache(tmp_path, monkeypatch):
    c = cluster(tmp_path)
    monkeypatch.setitem(models.SHARED_HF_CACHE_LANES, "c", (Match(agentic=True),))

    assert single_node_hf_cache(c, request(IS_AGENTIC="1")) == tmp_path / "shared-hub"
    assert single_node_hf_cache(c, request(IS_AGENTIC="0")) == tmp_path / "hub"


def test_runtime_settings_fill_what_the_workflow_left_unset(tmp_path, monkeypatch):
    c = cluster(tmp_path)
    entries = ((Match(multinode=True), "M@nvme"), (Match(), "M"))
    monkeypatch.setitem(policy.RUNTIME_MODEL_ENTRIES, "c", entries)
    monkeypatch.setitem(policy.TILERT_ENV, "c", {"UCX_NET_DEVICES": "mlx5_0:1"})

    tilert = runtime_env(c, request(IS_MULTINODE="true", FRAMEWORK="tilert"))
    single = runtime_env(c, request(IS_MULTINODE="false", FRAMEWORK="sglang"))

    assert (tilert["MODEL_PATH"], tilert["UCX_NET_DEVICES"]) == (f"{tmp_path}/nvme/m", "mlx5_0:1")
    assert single["MODEL_PATH"] == f"{tmp_path}/shared/m" and "UCX_NET_DEVICES" not in single
    # An exported MODEL_PATH (a master-config additional-setting) wins.
    assert runtime_env(c, request(MODEL_PATH="/exported"))["MODEL_PATH"] == "/exported"


# A controlled bump every AgentX point at CONC >= 64 would get, and a lane with a long
# limit for dsv4 AgentX requests.
BUMP = TimeBump(Match(agentic=True), min_conc=64, minutes=1440)
LANE = SrtLane(tag=None, time_limit="4:00:00", long_time_limit="8:00:00",
               long_time=Match(any_of("dsv4"), agentic=True))  # fmt: skip
BUMPED = dict(IS_AGENTIC="1", CONC="64")
LONG = dict(MODEL_PREFIX="dsv4", IS_AGENTIC="1")


@pytest.mark.parametrize(("profile", "lane", "env", "limit"), [
    ({}, None, {}, "480"),
    ({}, None, BUMPED, "1440"),
    ({"single-node-time-limit": 180}, None, BUMPED, "180"),
    ({}, LANE, LONG, "8:00:00"),
    ({}, LANE, dict(LONG, IS_AGENTIC="0"), "4:00:00"),
    ({}, SrtLane(tag=None), BUMPED, "480"),
    ({"default-time-limit": "6:00:00", "single-node-time-limit": 180}, None, BUMPED, "6:00:00"),
], ids=[
    "single-node-salloc", "single-node-bump", "single-node-profile-limit", "lane-long-limit",
    "lane-limit", "multi-node-salloc-is-never-bumped", "fixed-limit-wins",
])  # fmt: skip
def test_srt_time_limits(monkeypatch, profile, lane, env, limit):
    monkeypatch.setitem(policy.SALLOC_TIME_BUMPS, "c", BUMP)
    srt = SrtSlurmSettings.model_validate({"network-interface": "", **profile})
    point = request(SALLOC_TIME_LIMIT="480", EVAL_ONLY="false", **env)
    assert srt_time_limit("c", point, lane, srt) == limit
