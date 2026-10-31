import pytest

from infx.clusters import load_inventory

NODES = {
    "hcm-hgx01": ("hgx-h200-01", 4, "gpu:nvidia_h200:{gpus}", 4),
    "hcm-hgx02": ("hgx-h200-02", 8, "gpu:nvidia_h200:{gpus}", 4),
    "hcm-rtx01": ("hoanq3-rtx6000x2-slurm-1", 2, "gpu:nvidia_rtx6000pro:{gpus}", 2),
    "hcm-rtx02": ("hoanq3-rtx6000x2-slurm-2", 2, "gpu:nvidia_rtx6000pro:{gpus}", 2),
    "han-h200-01": ("hoanq3-h200-1x-han-1", 1, "gpu:{gpus}", 1),
    "han-h200-02": ("hoanq3-h200-1x-han-2", 1, "gpu:{gpus}", 1),
    "han-h200-04": ("hoanq3-h200-1x-han-4", 1, "gpu:{gpus}", 1),
}


@pytest.mark.parametrize("cluster_id", NODES)
def test_vng_node_is_one_cluster_pinned_and_shared(cluster_id):
    node, gpus, gres, slots = NODES[cluster_id]
    inventory = load_inventory()
    runners = inventory.labels[f"cluster:{cluster_id}"]
    assert runners == [f"{cluster_id}_{index:02d}" for index in range(slots)]
    cluster = inventory.cluster_for(runners[0])
    settings = cluster.scheduler_settings
    assert cluster.gpus_per_node == gpus
    assert settings.exclusive is False
    assert settings.gres == gres
    assert f"--nodelist={node}" in settings.salloc_args
    assert "--cpus-per-gpu=16" in settings.salloc_args
    assert any(arg.startswith("--mem-per-gpu=") for arg in settings.salloc_args)
    assert "hf-hub-cache" in settings.volumes
    assert settings.squash.import_mode == "submit-host" and settings.squash.single_node_import


def test_pool_labels_list_every_slot():
    labels = load_inventory().labels
    assert labels["h200-hcm"] == labels["cluster:hcm-hgx01"] + labels["cluster:hcm-hgx02"]
    assert labels["rtx6000pro-hcm"] == labels["cluster:hcm-rtx01"] + labels["cluster:hcm-rtx02"]
    assert labels["h200-han"] == labels["cluster:han-h200-01"] + labels["cluster:han-h200-02"] + labels["cluster:han-h200-04"]
