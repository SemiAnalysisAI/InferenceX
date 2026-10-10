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


def generate(project, reference: str, fabrics: dict[str, dict]) -> list[dict]:
    """The fixture rows, the selected variant's prefill UCX devices set to ``reference``."""
    recipe = {
        "base": {"schema": 2, "engine": "sglang", "roles": {"prefill": {"nodes": 1}, "decode": {"nodes": 1}}},
        # The unselected variant's typo does not concern this row.
        "override_narrow": {"roles": {"prefill": {"env": {"UCX_NET_DEVICES": "@fabric.typo"}}}},
        "override_wide": {"roles": {"prefill": {"env": {"UCX_NET_DEVICES": reference}}}},
    }  # fmt: skip
    files = {
        "benchmarks/multi_node/srt-slurm-recipes/fixture/bundle.yaml": recipe,
        "configs/master.yaml": MASTER,
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


DEVICES = {"ucx-net-devices": ["mlx5_0:1"]}


def test_a_reference_every_reachable_cluster_sets_generates_the_row_as_written(tmp_path):
    assert len(generate(tmp_path, "@fabric.ucx-net-devices", {"h200-a": DEVICES, "h200-b": DEVICES})) == 1


@pytest.mark.parametrize(("reference", "fabrics", "message"), [
    ("@fabric.ucx-net-device", {"h200-a": DEVICES},
     r"override_wide: roles\.prefill\.env\.UCX_NET_DEVICES: '@fabric\.ucx-net-device' is not a whole"),
    ("mlx5_0:1,@fabric.ucx-net-devices", {"h200-a": DEVICES}, "is not a whole '@fabric"),
    ("@fabric.ucx-net-devices", {"h200-a": DEVICES, "h200-b": {}},
     r"runner 'h200' reaches clusters \['h200-b'\] that set no srt-slurm\.fabric\.ucx-net-devices"),
])  # fmt: skip
def test_a_reference_no_fabric_field_or_reachable_cluster_serves_fails_generation(
    tmp_path, reference, fabrics, message
):
    with pytest.raises(ValueError, match=message):
        generate(tmp_path, reference, fabrics)
