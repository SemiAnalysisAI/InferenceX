"""Launch-path, model, cache and time-limit policy for srt-slurm lanes."""

import pytest

from infx.clusters import Cluster, load_inventory
from infx.clusters.slurm import SrtSlurmSettings
from infx.launch import policy
from infx.launch.context import LaunchError
from infx.launch.drivers.srt import models
from infx.launch.drivers.srt.lanes import SrtLane, srt_lane, srt_time_limit
from infx.launch.drivers.srt.models import (
    Override,
    checkpoint,
    host_path,
    job_env,
    model_paths,
    served_path,
    single_node_hf_cache,
    single_node_model_path,
)
from infx.launch.policy import LaunchPath, Match, TimeBump, any_of, launch_path
from infx.launch.request import LaunchRequest


def request(**env: str) -> LaunchRequest:
    """A launch request from ``env`` (RUNNER_NAME and a workspace preset)."""
    return LaunchRequest.from_env({"RUNNER_NAME": "r_0", "GITHUB_WORKSPACE": "/ws", **env})


def cluster(tmp_path, single_node_models: str = "staged") -> Cluster:
    """Cluster ``c``: ``M`` on shared storage with a node-local copy, ``S`` shared only, two Hub caches."""
    record = {
        "gpus-per-node": 8, "arch": "x86_64", "scheduler": "slurm",
        "models": {"entries": {
            "M": {"root": "shared", "dir": "m"}, "M@nvme": {"root": "nvme", "dir": "m"},
            "S": {"root": "shared", "dir": "s"},
        }},
        "slurm": {"partition": "p", "exclusive": True, "volumes": {
            "shared": {"path": str(tmp_path / "shared")},
            "nvme": {"path": str(tmp_path / "nvme"), "visibility": "node-local"},
            "hf-hub-cache": {"path": str(tmp_path / "hub"), "visibility": "node-local"},
            "shared-hf-hub-cache": {"path": str(tmp_path / "shared-hub")},
        }, "srt-slurm": {"network-interface": "", "single-node-models": single_node_models}},
    }  # fmt: skip
    return load_inventory({"labels": {"cluster:c": ["c_0"]}, "clusters": {"c": record}}).clusters["c"]


MULTI = dict(IS_MULTINODE="true", CONFIG_FILE="recipes/x.yaml")
SINGLE = dict(IS_MULTINODE="false")


@pytest.mark.parametrize(("cluster_id", "env", "path"), [
    ("b200-nscale", dict(MULTI, MODEL_PREFIX="dsv4", PRECISION="fp4", FRAMEWORK="dynamo-sglang", SPEC_DECODING="mtp"), LaunchPath.SRT_NATIVE),
    ("b200-nscale", dict(MULTI, MODEL_PREFIX="dsv4", PRECISION="fp4", FRAMEWORK="dynamo-sglang", SPEC_DECODING="eagle"), LaunchPath.SRT_MULTI),
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


OVERRIDES = (
    Override(Match(frameworks=any_of("sglang")), entry="M"),
    Override(Match(frameworks=any_of("trt")), served_name="served-m"),
    Override(Match(model_glob="*/S"), require_config=True),
)


@pytest.mark.parametrize(("env", "path", "node_local", "served"), [
    (dict(MODEL="org/M"), "nvme/m", True, None),
    (dict(MODEL="org/M", FRAMEWORK="sglang"), "shared/m", False, None),
    (dict(MODEL="org/M", FRAMEWORK="trt"), "nvme/m", True, "served-m"),
    (dict(MODEL="org/Unstaged"), None, None, None),
])  # fmt: skip
def test_models_resolve_by_basename_with_overrides(tmp_path, monkeypatch, env, path, node_local, served):
    monkeypatch.setitem(models.OVERRIDES, "c", OVERRIDES)
    c, point = cluster(tmp_path), request(**env)
    found = checkpoint(c, point)
    assert (found and (host_path(c, found), found.node_local)) == (path and (tmp_path / path, node_local))
    assert job_env(c, point, served_path(c, point, found)).get("SERVED_MODEL_NAME") == served
    assert single_node_model_path(c, point) == (str(tmp_path / path) if path else f"hf:{env['MODEL']}")
    assert single_node_model_path(cluster(tmp_path, "hub"), point) == f"hf:{env['MODEL']}"


def test_a_checkpoint_that_must_be_readable_fails_before_submission(tmp_path, monkeypatch):
    monkeypatch.setitem(models.OVERRIDES, "c", OVERRIDES)
    with pytest.raises(LaunchError, match="no readable .*/s/config.json"):
        checkpoint(cluster(tmp_path), request(MODEL="org/S"))
    (tmp_path / "shared/s").mkdir(parents=True)
    (tmp_path / "shared/s/config.json").write_text("{}")
    c = cluster(tmp_path)
    assert host_path(c, checkpoint(c, request(MODEL="org/S"))) == tmp_path / "shared/s"


def test_a_points_own_model_path_is_what_its_job_serves(tmp_path, monkeypatch):
    c = cluster(tmp_path)
    host = request(MODEL="org/M", MODEL_PATH="/host/m")
    assert served_path(c, host, checkpoint(c, host)) == str(tmp_path / "nvme/m")
    point = request(MODEL="org/M", MODEL_PATH="/point/m", PREFILL_ADDITIONAL_SETTINGS='["MODEL_PATH=/point/m"]')
    monkeypatch.setitem(models.OVERRIDES, "c", (Override(Match(), require_config=True),))
    served = served_path(c, point, checkpoint(c, point))
    assert served == "/point/m"
    assert job_env(c, point, served)["MODEL_PATH"] == "/point/m"


BUNDLE = """base:
  model: {path: alias-a}
override_x:
  model: {path: alias-b}
zip_override_y:
  model: {path: [alias-c, "hf:org/M"]}
"""


@pytest.mark.parametrize(("recipe", "model", "paths"), [
    (BUNDLE, "org/M", {"alias-a": "nvme/m", "alias-b": "nvme/m", "alias-c": "nvme/m"}),
    ("model: {path: /abs/m}\n", "org/Unstaged", {}),
    ("model: {path: alias-a}\n", "org/Unstaged", LaunchError),
])  # fmt: skip
def test_every_recipe_alias_maps_to_the_checkpoint_and_literals_pass_through(tmp_path, recipe, model, paths):
    mirror = tmp_path / "ws/benchmarks/multi_node/srt-slurm-recipes/r.yaml"
    mirror.parent.mkdir(parents=True)
    mirror.write_text(recipe)
    c = cluster(tmp_path)
    point = request(MODEL=model, GITHUB_WORKSPACE=str(tmp_path / "ws"))
    if paths is LaunchError:
        with pytest.raises(LaunchError, match="stages no checkpoint"):
            model_paths(c, point, "recipes/r.yaml:override_x", served_path(c, point, checkpoint(c, point)))
        return
    resolved = model_paths(c, point, "recipes/r.yaml:override_x", served_path(c, point, checkpoint(c, point)))
    assert resolved == {alias: str(tmp_path / path) for alias, path in paths.items()}


def test_only_a_fork_may_run_a_recipe_the_mirror_lacks(tmp_path):
    c, point = cluster(tmp_path), request(MODEL="org/M", GITHUB_WORKSPACE=str(tmp_path))
    assert model_paths(c, point, "recipes/fork-only.yaml", "/m", fork=True) == {}
    with pytest.raises(LaunchError, match="not in the recipe mirror"):
        model_paths(c, point, "recipes/fork-only.yaml", "/m")


def test_matching_single_node_points_read_the_shared_hub_cache(tmp_path, monkeypatch):
    c = cluster(tmp_path)
    monkeypatch.setitem(models.SHARED_HF_CACHE_LANES, "c", (Match(agentic=True),))

    assert single_node_hf_cache(c, request(IS_AGENTIC="1")) == tmp_path / "shared-hub"
    assert single_node_hf_cache(c, request(IS_AGENTIC="0")) == tmp_path / "hub"


BUMP = TimeBump(Match(agentic=True), min_conc=64, minutes=1440)
LANE = SrtLane(time_limit="4:00:00", long_time_limit="8:00:00", long_time=Match(any_of("dsv4"), agentic=True))
BUMPED = dict(IS_AGENTIC="1", CONC="64")
LONG = dict(MODEL_PREFIX="dsv4", IS_AGENTIC="1")


@pytest.mark.parametrize(("profile", "lane", "env", "limit"), [
    ({}, None, {}, "480"),
    ({}, None, BUMPED, "1440"),
    ({"single-node-time-limit": 180}, None, BUMPED, "180"),
    ({}, LANE, LONG, "8:00:00"),
    ({}, LANE, dict(LONG, IS_AGENTIC="0"), "4:00:00"),
    ({}, SrtLane(), BUMPED, "480"),
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
