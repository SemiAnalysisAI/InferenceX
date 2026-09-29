"""srtslurm.yaml rendering from controlled cluster records."""

from pathlib import Path

import pytest
import yaml

from infx.clusters import Cluster, load_inventory
from infx.launch.drivers.srt.config import SrtJob, pyxis_spelling, render, write


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
    config = render(record, job(mounts=[("/share/hub", "/mnt/hf_hub_cache/")], exclusive=True))
    # One host directory mounted twice: the cluster's Hub view and the job's HF_HUB_CACHE.
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
    assert "model_paths" not in config and "default_health_check" not in config
    assert (config["visible_devices_env"], config["default_gpu_exporter"]) == ("ROCR_VISIBLE_DEVICES", None)
    assert (config["network_interface"], config["use_exclusive_sbatch_directive"]) == ("eno0", True)
    assert "default_account" not in config


def test_a_host_directory_cannot_be_mounted_at_three_targets():
    record = cluster(slurm={"volumes": {"hub": {"path": "/share/hub"}}}, srt={"volume-mounts": {"hub": "/hf_hub_cache"}})
    with pytest.raises(ValueError, match="conflicting container paths"):
        render(record, job(mounts=[("/share/hub", "/a"), ("/share/hub/", "/b")]))


def test_exclusivity_follows_the_shape():
    shared = cluster(slurm={"exclusive": False}, srt={"segment-directive": False})
    multinode = render(shared, job())
    assert multinode["use_exclusive_sbatch_directive"] is False
    assert multinode["use_segment_sbatch_directive"] is False
    assert "use_gpus_per_node_directive" not in multinode
    assert render(shared, job(exclusive=True))["use_exclusive_sbatch_directive"] is True


def test_node_exclusions_cpus_and_image_aliases_are_rendered():
    record = cluster(
        slurm={"account": "bench", "exclude": ["node-1", "node-2"], "cpus-per-task": 192,
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
    assert (config["default_account"], config["default_partition"]) == ("bench", "batch")


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
