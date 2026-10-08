import copy

import pytest
import yaml
from pydantic import ValidationError

from infx.clusters import SCHEDULERS, load_inventory, resolve_cluster
from infx.launch.backends import BACKENDS
from infx.tests.launch.fake_backend import FakeSettings

CLUSTER = {
    "gpus-per-node": 8,
    "arch": "x86_64",
    "models": {"entries": {"Kimi-K3": {"root": "scratch", "dir": "Kimi-K3"}}},
    "scheduler": "slurm",
    "slurm": {
        "partition": "batch",
        "exclusive": True,
        "volumes": {"scratch": {"path": "/scratch/models", "visibility": "node-local"}},
        "squash": {"dir": "/shared/squash", "import": "pre-staged"},
        "srt-slurm": {"network-interface": ""},
    },
}


def inventory(**clusters):
    """Runner config where cluster ``<id>`` owns runners ``<id>_0`` and ``<id>_1``."""
    return {
        "labels": {
            "gpu": [f"{cluster_id}_0" for cluster_id in clusters],
            **{f"cluster:{cluster_id}": [f"{cluster_id}_0", f"{cluster_id}_1"] for cluster_id in clusters},
        },
        "clusters": clusters,
    }


def with_change(path: str, value):
    """Copy CLUSTER with one dotted key replaced (``None`` deletes it)."""
    cluster = copy.deepcopy(CLUSTER)
    *parents, leaf = path.split(".")
    target = cluster
    for parent in parents:
        target = target.setdefault(parent, {})
    if value is None:
        del target[leaf]
    else:
        target[leaf] = value
    return cluster


def test_resolve_cluster_accepts_a_path_and_names_the_owning_cluster(tmp_path):
    path = tmp_path / "runners.yaml"
    path.write_text(yaml.safe_dump(inventory(alpha=CLUSTER, beta=CLUSTER)))

    assert resolve_cluster("beta_1", path).id == "beta"
    with pytest.raises(ValueError, match="exactly one cluster label"):
        resolve_cluster("gamma_0", path)


def test_scheduler_record_errors_carry_the_record_name():
    with pytest.raises(ValidationError) as raised:
        load_inventory(inventory(alpha=with_change("slurm.gres", "gpu:8")))

    [error] = raised.value.errors()
    assert error["loc"] == ("clusters", "alpha", "slurm", "gres")


def test_runner_partition_resolution_preserves_cluster_and_shared_profile():
    data = inventory(alpha=with_change(
        "slurm.partitions", ["batch", "batch_1", "batch_3"]
    ))
    data["labels"].update({"partition:batch_1": ["alpha_0"], "partition:batch_3": ["alpha_1"]})
    loaded = load_inventory(data)
    first = loaded.cluster_for("alpha_0")
    third = loaded.cluster_for("alpha_1")
    assert first.id == third.id == "alpha"
    assert first.scheduler_settings.partition == "batch_1"
    assert third.scheduler_settings.partition == "batch_3"
    assert loaded.clusters["alpha"].scheduler_settings.partition == "batch"
    assert loaded.cluster_for("alpha_0").scheduler_settings.partition == "batch_1"
    assert str(third.scheduler_settings.squash.dir) == "/shared/squash"


@pytest.mark.parametrize("labels", [
    {"partition:batch_1": ["alpha_0"]},
    {"partition:batch_1": ["alpha_0", "alpha_1"], "partition:batch_3": ["alpha_1"]},
    {"partition:unknown": ["alpha_0", "alpha_1"]},
    {"partition:batch_1,batch_3": ["alpha_0", "alpha_1"]},
])
def test_partition_routes_reject_incomplete_unknown_or_multi_partition_targets(labels):
    data = inventory(alpha=with_change("slurm.partitions", ["batch", "batch_1", "batch_3"]))
    data["labels"].update(labels)
    with pytest.raises(ValidationError):
        load_inventory(data)


def test_a_registered_scheduler_parses_its_own_record_and_volumes(monkeypatch):
    monkeypatch.setitem(SCHEDULERS, "fake", FakeSettings)
    fake = {
        "gpus-per-node": 8, "arch": "x86_64",
        "models": {"entries": {"Kimi-K3": {"root": "models", "dir": "Kimi-K3"}}, "download-root": "models"},
        "scheduler": "fake",
        "fake": {"root": "/sandbox", "namespace": "bench", "volumes": {"models": {"claim": "ckpt"}}},
    }  # fmt: skip
    cluster = load_inventory(inventory(alpha=fake)).clusters["alpha"]

    assert cluster.scheduler_settings.volumes["models"].claim == "ckpt"
    fake["fake"]["volumes"]["models"] = {"path": "/models"}
    with pytest.raises(ValidationError, match="claim"):
        load_inventory(inventory(alpha=fake))
    with pytest.raises(ValidationError, match="records for schedulers other than 'slurm'"):
        load_inventory(inventory(alpha={**CLUSTER, "fake": {"root": "/sandbox", "namespace": "bench"}}))


def test_every_scheduler_a_record_can_name_has_a_backend():
    assert BACKENDS.keys() == SCHEDULERS.keys()


@pytest.mark.parametrize(
    "runner_config,message",
    [
        (
            {"labels": {"cluster:a": ["x_0"], "cluster:b": ["x_0"]}, "clusters": {"a": CLUSTER, "b": CLUSTER}},
            "several clusters",
        ),
        (
            {"labels": {"cluster:a": ["a_0"], "gpu": ["stray_0"]}, "clusters": {"a": CLUSTER}},
            "without a cluster:<id> label: \\['stray_0'\\]",
        ),
        ({"labels": {"cluster:a": ["a_0"]}, "clusters": {}}, "without a clusters entry"),
        ({"labels": {"gpu": ["a_0"]}, "clusters": {"a": CLUSTER}}, "clusters without a cluster:<id> label"),
        ({**inventory(a=CLUSTER), "hardware": {}}, "hardware"),
    ],
    ids=["shared-runner", "orphan-runner", "label-without-record", "record-without-label", "legacy-hardware"],
)
def test_inventory_rejects_inconsistent_labels(runner_config, message):
    with pytest.raises(ValidationError, match=message):
        load_inventory(runner_config)


@pytest.mark.parametrize(
    "path,value,message",
    [
        ("gpus-per-node", 0, "greater than 0"),
        ("env", {"UCX_NET_DEVICES": "mlx5_0:1,mlx5_1:1"}, "cannot contain ','"),
        ("scheduler", "no-such-scheduler", "scheduler must be one of"),
        ("slurm", None, "needs its 'slurm' record"),
        ("slurm.gres", "gpu:8", "placeholder"),
        ("slurm.gres", "gpu:{count}", "placeholder"),
        ("slurm.srun-args", ["container-remap-root"], "long option"),
        ("slurm.squash", {"dir": "/nvme/squash", "visibility": "node-local", "import": "pre-staged",
                          "helper-dirs": {"nginx": {"import": "submit-host"}}}, "cannot populate node-local"),
        ("slurm.volumes.scratch", {"path": "models"}, "absolute"),
        ("models.entries", {"Kimi-K3": {"root": "nvme", "dir": "Kimi-K3"}}, "unknown volume 'nvme'"),
        ("models.entries", {"Kimi-K3": {"root": "scratch", "dir": "../Kimi-K3"}}, "relative to its root"),
        ("models.download-root", "writable", "unknown volume 'writable'"),
        ("models.download-root", "scratch", "must be a shared volume"),
        # Fabric values are lists, rendered with commas only at bind time.
        ("slurm.srt-slurm.fabric", {"nccl-ib-hca": "mlx5_0,mlx5_1"}, "valid tuple"),
        ("slurm.srt-slurm.fabric", {"nccl-ib-hca": ["mlx5_0,mlx5_1"]}, "should match pattern"),
        ("slurm.srt-slurm.volume-mounts", {"hf-hub-cache": "/hf_hub_cache"}, "unknown volumes"),
        ("slurm.srt-slurm.host-setup", {"script": "/opt/setup.sh"}, "repository-relative"),
        ("partition", "batch", "Extra inputs"),
    ],
)
def test_invalid_cluster_is_rejected(path, value, message):
    with pytest.raises(ValidationError, match=message):
        load_inventory(inventory(alpha=with_change(path, value)))
