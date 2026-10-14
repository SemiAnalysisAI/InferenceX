"""Matrix generation rejects a recipe's fabric reference before any cluster resolves it."""

import pytest
import yaml

from infx.matrix.generate import generate_config_matrix
from infx.matrix.validation import load_config_files

ROLE = {"num-worker": 1, "tp": 8, "ep": 1, "dp-attn": False}
MASTER = {"fixture": {
    "image": "org/image:1", "model": "org/model", "model-prefix": "dsr1", "precision": "fp8",
    "framework": "dynamo-sglang", "runner": "h200", "multinode": True, "disagg": True,
    "kv-p2p-transfer": "nixl", "srt-recipe-dir": "fixture",
    "scenarios": {"fixed-seq-len": [{"isl": 1024, "osl": 128, "search-space": [
        {"prefill": ROLE, "decode": ROLE, "conc-list": [4], "srt-recipe": "bundle.yaml:override_wide"},
    ]}]},
}}  # fmt: skip


def generate(project, reference: str, fabrics: dict[str, dict], runner: str = "h200") -> list[dict]:
    """The fixture rows on ``runner``, the selected variant's prefill UCX devices set to ``reference``."""
    recipe = {
        "base": {"schema": 2, "engine": "sglang", "roles": {"prefill": {"nodes": 1}, "decode": {"nodes": 1}}},
        "override_narrow": {"roles": {"prefill": {"env": {"UCX_NET_DEVICES": "@fabric.typo"}}}},
        "override_wide": {"roles": {"prefill": {"env": {"UCX_NET_DEVICES": reference}}}},
    }  # fmt: skip
    files = {
        "benchmarks/multi_node/srt-slurm-recipes/fixture/bundle.yaml": recipe,
        "configs/master.yaml": {"fixture": {**MASTER["fixture"], "runner": runner}},
    }
    for path, data in files.items():
        (project / path).parent.mkdir(parents=True, exist_ok=True)
        (project / path).write_text(yaml.safe_dump(data))
    runners = {
        "labels": {
            "h200": [f"{cluster}_0" for cluster in fabrics],
            **{f"cluster:{cluster}": [f"{cluster}_0"] for cluster in fabrics},
        },
        "clusters": {cluster: {
            "gpus-per-node": 8, "arch": "x86_64", "scheduler": "slurm", "slurm": {
                "partition": "p", "exclusive": True,
                "srt-slurm": {"network-interface": "", "fabric": fabric},
            },
        } for cluster, fabric in fabrics.items()},
    }  # fmt: skip
    master = load_config_files([str(project / "configs/master.yaml")])
    return generate_config_matrix(["fixture"], master, runners, eval_mode="none", root=project)


DEVICES = {"rdma-device-ports": ["mlx5_0:1"]}


def test_a_reference_every_reachable_cluster_sets_generates_the_row_as_written(tmp_path):
    assert len(generate(tmp_path, "@fabric.rdma-device-ports", {"h200-a": DEVICES, "h200-b": DEVICES})) == 1


@pytest.mark.parametrize(("reference", "fabrics", "message"), [
    ("@fabric.rdma-device-port", {"h200-a": DEVICES},
     r"override_wide: roles\.prefill\.env\.UCX_NET_DEVICES: '@fabric\.rdma-device-port' is not a whole"),
    ("@fabric.rdma-device-ports", {"h200-a": DEVICES, "h200-b": {}},
     r"this cluster sets no srt-slurm\.fabric\.rdma-device-ports \(cluster 'h200-b'\)"),
])  # fmt: skip
def test_a_reference_no_fabric_field_or_reachable_cluster_serves_fails_generation(
    tmp_path, reference, fabrics, message
):
    with pytest.raises(ValueError, match=message):
        generate(tmp_path, reference, fabrics)


def test_a_runner_no_cluster_label_lists_still_has_its_reference_names_checked(tmp_path):
    with pytest.raises(ValueError, match=r"'@fabric\.rdma-device-port' is not a whole .* \(runner 'h200-a'\)"):
        generate(tmp_path, "@fabric.rdma-device-port", {"h200-a": DEVICES}, runner="h200-a")
