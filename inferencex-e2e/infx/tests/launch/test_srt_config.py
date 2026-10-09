"""srtslurm.yaml rendering from controlled cluster records."""

import shutil
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import yaml

from infx.clusters import Cluster, load_inventory
from infx.launch.backends.slurm import SlurmBackend
from infx.launch.context import Launch
from infx.launch.drivers.srt.checkout import Checkout, compute_workspace
from infx.launch.drivers.srt.config import SrtJob, pyxis_spelling, render, write
from infx.launch.drivers.srt.recipe import HEALTH_ATTEMPTS, prepare_recipe
from infx.launch.drivers.srt.run import SrtRun
from infx.launch.lifecycle import Lifecycle
from infx.launch.policy import LaunchPath
from infx.launch.request import SrtRequest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "utils/srt-slurm/src"))
from srtctl.core.config import resolve_config_with_defaults  # noqa: E402
from srtctl.core.schema import ClusterConfig  # noqa: E402

# The exporter block every AMD cluster record carries: our fixed device-metrics-exporter
# build, read by srt-slurm's `kind: custom` power collector. The `scope` string keeps
# manifests comparable with earlier AMD runs.
AMD_EXPORTER = {
    "container_image": "ghcr.io#semianalysisai/amd-device-metrics-exporter@sha256:"
    "8a3fe70b8a848ca10a7fd90862d1d9e41e9c7b6314669a2e8340c646e6dae15c",
    "port": 19500,
    "command": "env AMD_GPU_GET_CACHE_TTL=0s /home/amd/tools/entrypoint.sh",
    "kind": "custom",
    "gpu_labels": {"index": "gpu_id", "identity": "serial_number"},
    "gpu_metrics": {
        "power": {
            "metric": "gpu_power_usage",
            "scope": "gpu_device_power_as_reported_by_amd_device_metrics_exporter",
        },
        "gpu_util": {"metric": "gpu_gfx_activity"},
        "temperature": {"metric": "gpu_junction_temperature"},
    },
}
DCGM_EXPORTER = {
    "container_image": "nvcr.io#nvidia/k8s/dcgm-exporter:4.6.0-4.8.3-distroless",
    "port": 9401,
    "command": "dcgm-exporter --collect-interval=1000 --address :{port} -f /configs/dcgm-counters-noprof.csv",
}


def inventory_cluster(cluster_id: str) -> Cluster:
    """The repository's own record of ``cluster_id``."""
    return load_inventory(yaml.safe_load((ROOT / "configs/runners.yaml").read_text())).clusters[cluster_id]


def cluster(slurm: dict | None = None, srt: dict | None = None, entries: dict | None = None) -> Cluster:
    """Cluster ``c`` with the given ``slurm`` fields, srt-slurm profile fields and model entries."""
    record = {
        "gpus-per-node": 8, "arch": "x86_64", "scheduler": "slurm",
        "models": {"entries": entries or {}},
        "slurm": {
            "partition": "batch", "exclusive": True, **(slurm or {}),
            "srt-slurm": {"network-interface": "", **(srt or {})},
        },
    }  # fmt: skip
    return load_inventory({"labels": {"cluster:c": ["c_0"]}, "clusters": {"c": record}}).clusters["c"]


def job(**overrides) -> SrtJob:
    """A job with staged images and a 480-minute limit."""
    values = dict(
        srtctl_root=Path("/ws/srt-slurm"), workspace=Path("/ws"), time_limit="480",
        image="lmsysorg/sglang:v1", container="/sq/image.sqsh", nginx="/sq/nginx.sqsh",
    )  # fmt: skip
    return SrtJob(**{**values, **overrides})


def test_the_profile_renders_its_facts_and_mounts_a_volume_at_a_second_target():
    record = cluster(
        slurm={"cpus-per-task": 128, "volumes": {"data": {"path": "/share/data"}, "hub": {"path": "/share/hub"}}},
        srt={
            "network-interface": "eno0",
            "outputs": "/share/outputs",
            "host-setup": {
                "script": "runners/hooks/setup.sh", "env": {"IBDEVICES": "rdma0,rdma1"},
                "timeout-s": 1200, "nodes": "all",
            },
            "volume-mounts": {"hub": "/hf_hub_cache/hub"},
            "mounts": {"/dev/kfd": "/dev/kfd"},
            "extra": {"visible_devices_env": "ROCR_VISIBLE_DEVICES", "default_gpu_exporter": None},
        },
        entries={"Model-A": {"root": "data", "dir": "Model-A"}},
    )  # fmt: skip
    config = render(record, job(mounts=[("/share/hub", "/mnt/hf_hub_cache/")], single_node=True))
    assert config["default_mounts"] == {
        "/share/hub": "/hf_hub_cache/hub",
        "/dev/kfd": "/dev/kfd",
        "/share/hub/": "/mnt/hf_hub_cache/",
    }
    assert config["default_host_setup"] == {
        "commands": ["IBDEVICES=rdma0,rdma1 bash /ws/runners/hooks/setup.sh"],
        "timeout_seconds": 1200,
        "nodes": "all",
    }
    assert config["output_dir"] == "/share/outputs"
    assert config["default_sbatch_directives"] == {"cpus-per-task": "128"}
    assert "model_paths" not in config
    assert (config["visible_devices_env"], config["default_gpu_exporter"]) == ("ROCR_VISIBLE_DEVICES", None)
    assert (config["network_interface"], config["use_exclusive_sbatch_directive"]) == ("eno0", True)
    assert config["cluster"] == "c"


@pytest.mark.parametrize("cluster_id", ["mi355x-amds", "mi325x-amd", "mi300x-amd"])
def test_amd_clusters_render_the_custom_exporter_and_mount_its_config(cluster_id):
    config = render(inventory_cluster(cluster_id), job(single_node=True))

    assert config["default_gpu_exporter"] == AMD_EXPORTER
    assert config["default_mounts"]["/ws/runners/srt-slurm/exporters/amd-power.json"] == "/etc/metrics/config.json"
    # The pinned srt-slurm reads this file; a key it does not know fails here.
    ClusterConfig.Schema().load(config)
    recipe = {"schema": 2, "name": "point", "telemetry": {"enabled": True}}
    assert resolve_config_with_defaults(recipe, config)["telemetry"]["dcgm_exporter"] == AMD_EXPORTER


def test_nvidia_clusters_keep_the_dcgm_exporter_and_mount_no_exporter_config():
    config = render(inventory_cluster("h200-cw"), job(single_node=True))

    assert config["default_gpu_exporter"] == DCGM_EXPORTER
    assert "/etc/metrics/config.json" not in config.get("default_mounts", {}).values()
    ClusterConfig.Schema().load(config)


def test_a_host_directory_cannot_be_mounted_at_three_targets():
    record = cluster(slurm={"volumes": {"hub": {"path": "/share/hub"}}}, srt={"volume-mounts": {"hub": "/hf_hub_cache"}})
    with pytest.raises(ValueError, match="conflicting container paths"):
        render(record, job(mounts=[("/share/hub", "/a"), ("/share/hub/", "/b")]))


def test_node_exclusions_cpus_and_image_aliases_are_rendered():
    record = cluster(
        slurm={"exclude": ["node-1", "node-2"], "cpus-per-task": 192,
               "volumes": {"scratch": {"path": "/scratch/models"}}},
        srt={"container-aliases": ["dynamo-sglang", "dynamo-vllm"], "nginx-aliases": ["nginx-sqsh"]},
    )  # fmt: skip
    config = render(record, job())
    assert config["default_sbatch_directives"] == {"exclude": "node-1,node-2", "cpus-per-task": "192"}
    assert config["containers"] == {
        "dynamo-sglang": "/sq/image.sqsh",
        "dynamo-vllm": "/sq/image.sqsh",
        "lmsysorg/sglang:v1": "/sq/image.sqsh",
        "nginx-sqsh": "/sq/nginx.sqsh",
    }
    assert config["default_partition"] == "batch"


@pytest.mark.parametrize(("image", "spelling"), [
    ("nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.8.1", "nvcr.io#nvidia/ai-dynamo/tensorrtllm-runtime:0.8.1"),
    ("nvcr.io#nvidia/tensorrt-llm/release:1.3.0rc24", "nvcr.io#nvidia/tensorrt-llm/release:1.3.0rc24"),
    ("lmsysorg/sglang:v0.5.8", "lmsysorg/sglang:v0.5.8"),
])  # fmt: skip
def test_registry_images_are_aliased_in_both_spellings(image, spelling):
    assert pyxis_spelling(image) == spelling
    containers = render(cluster(), job(image=image))["containers"]
    assert containers[image] == containers[spelling] == "/sq/image.sqsh"


def test_a_failed_write_keeps_the_previous_config(tmp_path):
    target = tmp_path / "srtslurm.yaml"
    write(target, {"default_partition": "old"})
    with pytest.raises(yaml.YAMLError):
        write(target, {"default_partition": object()})
    assert yaml.safe_load(target.read_text()) == {"default_partition": "old"}


@pytest.mark.parametrize(("declared", "exported", "users_default", "expected"), [
    ("bench", "exported", "team", "bench"),
    (None, "exported", "team", "exported"),
    (None, None, "team", "team"),
    (None, None, "", None),
])  # fmt: skip
def test_jobs_run_under_the_declared_else_exported_else_users_default_account(
    tmp_path, monkeypatch, declared, exported, users_default, expected
):
    sacctmgr = tmp_path / "sacctmgr"
    sacctmgr.write_text(f"#!/bin/sh\nprintf '%s' '{users_default}'\n")
    sacctmgr.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    record = cluster(slurm={"account": declared} if declared else None)
    request = SrtRequest.from_env({
        "RUNNER_NAME": "c_0", "GITHUB_WORKSPACE": str(tmp_path), "IMAGE": "i", "FRAMEWORK": "sglang",
        "MODEL_PREFIX": "m", "PRECISION": "fp8", "SPEC_DECODING": "none", "RESULT_FILENAME": "r",
        "IS_AGENTIC": "0", "RUN_EVAL": "false", "EVAL_ONLY": "false",
        **({"SLURM_ACCOUNT": exported} if exported else {}),
    })  # fmt: skip
    life = Lifecycle()
    launch = Launch(record, SlurmBackend(record, request, life), request, life, LaunchPath.SRT_MULTI)

    run = SrtRun.create(launch, request)

    assert render(record, job(account=run.account)).get("default_account") == expected


def test_selected_partition_overrides_inherited_and_point_environment(tmp_path):
    record = cluster(slurm={"partition": "batch_3", "account": "benchmark"})
    request = SrtRequest.from_env({
        "RUNNER_NAME": "c_0", "GITHUB_WORKSPACE": str(tmp_path), "IMAGE": "i", "FRAMEWORK": "sglang",
        "MODEL_PREFIX": "m", "PRECISION": "fp8", "SPEC_DECODING": "none", "RESULT_FILENAME": "r",
        "IS_AGENTIC": "0", "RUN_EVAL": "false", "EVAL_ONLY": "false",
        "SBATCH_PARTITION": "batch_1", "SLURM_PARTITION": "batch_1",
        "PREFILL_ADDITIONAL_SETTINGS": '["SBATCH_PARTITION=batch_1"]',
    })
    life = Lifecycle()
    launch = Launch(record, SlurmBackend(record, request, life), request, life, LaunchPath.SRT_MULTI)

    run = SrtRun.create(launch, request, {"SLURM_PARTITION": "another-pool"})

    assert run.env["SBATCH_PARTITION"] == "batch_3"
    assert run.env["SLURM_PARTITION"] == "batch_3"
    assert request.env["SBATCH_PARTITION"] == "batch_1"


@pytest.mark.parametrize(("health", "effective"), [
    (None, {"max_attempts": HEALTH_ATTEMPTS, "interval_seconds": 10}),
    ({"max_attempts": 100, "interval_seconds": 5}, {"max_attempts": HEALTH_ATTEMPTS, "interval_seconds": 5}),
    ({"max_attempts": 2160, "interval_seconds": 5}, {"max_attempts": 2160, "interval_seconds": 5}),
])  # fmt: skip
def test_multinode_jobs_wait_at_least_the_health_floor_for_their_server(tmp_path, health, effective):
    recipe = tmp_path / "recipes/r.yaml"
    recipe.parent.mkdir()
    recipe.write_text(yaml.safe_dump({"schema": 2, "name": "r", **({"health_check": health} if health else {})}))

    prepare_recipe(tmp_path, "recipes/r.yaml", "job", None, None)

    resolved = resolve_config_with_defaults(yaml.safe_load(recipe.read_text()), render(cluster(), job()))
    assert resolved["health_check"] == effective


def test_single_node_jobs_skip_the_segment_and_typed_gres_replaces_gpus_per_node():
    typed = cluster(slurm={"exclusive": False, "gres": "gpu:h100:{gpus}"}, srt={"gpus-per-node-directive": False})
    single = render(typed, job(single_node=True))
    assert single["use_segment_sbatch_directive"] is False
    assert single["use_exclusive_sbatch_directive"] is True
    assert single["use_gpus_per_node_directive"] is False
    assert single["default_sbatch_directives"]["gres"] == "gpu:h100:8"
    shared = cluster(srt={"single-node-exclusive": False})
    assert render(shared, job(single_node=True))["use_exclusive_sbatch_directive"] is False
    multi = render(cluster(slurm={"exclusive": False}, srt={"segment-directive": True}), job())
    assert (multi["use_segment_sbatch_directive"], multi["use_exclusive_sbatch_directive"]) == (True, False)
    assert "use_gpus_per_node_directive" not in multi and "default_sbatch_directives" not in multi


def test_the_cpu_request_follows_the_clusters_unit():
    assert render(cluster(slurm={"cpus-per-gpu": 24}), job())["default_sbatch_directives"] == {"cpus-per-gpu": "24"}
    with pytest.raises(ValueError, match="not both"):
        cluster(slurm={"cpus-per-task": 192, "cpus-per-gpu": 24})


@pytest.mark.skipif(shutil.which("rsync") is None, reason="needs rsync")
def test_the_staged_workspace_drops_run_artifacts_but_keeps_srt_slurm_sources(tmp_path):
    workspace = tmp_path / "workspace"
    kept = ["runners/srt-slurm/hooks/common.sh", "benchmarks/multi_node/srt-slurm-recipes/r.yaml"]
    dropped = ["srt-slurm-1-1-abc/recipe.yaml", "utils/srt-slurm/pyproject.toml", ".venv/bin/python",
               "outputs/42/log", "LOGS/sweep.log", "image.sqsh"]  # fmt: skip
    for name in kept + dropped:
        (workspace / name).parent.mkdir(parents=True, exist_ok=True)
        (workspace / name).write_text("")

    def rsync(source: Path, staging: Path, *, exclude: tuple[str, ...]) -> Path:
        staging.mkdir(parents=True)
        excludes = [f"--exclude={pattern}" for pattern in exclude]
        subprocess.run(["rsync", "-a", *excludes, f"{source}/", f"{staging}/"], check=True)
        return staging

    run = SimpleNamespace(workspace=workspace, backend=SimpleNamespace(stage_workspace=rsync))
    checkout = Checkout(tmp_path / "runs/srt-slurm-1-1-abc", "sha")
    staged = compute_workspace(run, checkout, shared=True)

    assert staged == tmp_path / "runs/infmax-workspace-1-1-abc"
    assert sorted(str(p.relative_to(staged)) for p in staged.rglob("*") if p.is_file()) == sorted(kept)
