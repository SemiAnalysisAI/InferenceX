"""Behavioral checks for the offload study and its standard matrix path."""

import argparse
import json
from pathlib import Path

import pytest
import runtime
from runtime import connector_config, sysfs_backing_devices, verify_nvme_backing

from infx.matrix.generate import generate_test_config_sweep
from infx.matrix.validation import SingleNodeMasterConfigEntry
from runners.patch_vllm_simple_kv_offload import (
    NATIVE_REGIONS,
    NATIVE_SETUP,
    patch_worker,
)


def test_hbm_has_no_connector():
    assert connector_config("none", 4, 0, 0, Path("/cache")) is None


def test_current_upstream_simple_cpu_layout_needs_no_patch(tmp_path):
    worker = tmp_path / "worker.py"
    source = NATIVE_SETUP + NATIVE_REGIONS
    worker.write_text(source)

    assert patch_worker(worker) is False
    assert worker.read_text() == source


def test_dram_uses_aggregate_decimal_bytes():
    config = connector_config("dram", 4, 739_000_000_000, 0, Path("/cache"))
    assert config["kv_connector_extra_config"] == {
        "kv_offload_backend": "cpu",
        "cpu_bytes_to_use": 739_000_000_000,
        "lazy_offload": True,
    }


def test_nvme_divides_disk_capacity_without_enabling_page_cache():
    config = connector_config("nvme", 4, 0, 1024, Path("/cache"))
    assert config["kv_connector_extra_config"] == {
        "kv_offload_backend": "disk",
        "cpu_bytes_to_use": 1024,
        "lazy_offload": True,
        "disk_path": "/cache/cache.bin",
        "disk_capacity_bytes": 256,
        "disk_buffer_slots": 4,
        "use_page_cache": False,
    }


def test_combined_keeps_dram_and_secondary_fs():
    config = connector_config("dram-nvme", 4, 1024, 2048, Path("/cache"))
    assert config["kv_connector"] == "OffloadingConnector"
    assert config["kv_connector_extra_config"]["cpu_bytes_to_use"] == 1024
    assert config["kv_connector_extra_config"]["secondary_tiers"] == [
        {
            "type": "fs",
            "root_dir": "/cache",
            "n_read_threads": 32,
            "n_write_threads": 16,
            "locality": "LOCAL",
        }
    ]


def test_sysfs_storage_proof_resolves_raid_without_device_node(tmp_path):
    sys_block = tmp_path / "sys" / "class" / "block"
    (sys_block / "md0" / "slaves").mkdir(parents=True)
    for name in ("nvme0n1p1", "nvme1n1p1"):
        (sys_block / "md0" / "slaves" / name).mkdir()
        rotational = sys_block / name / "queue" / "rotational"
        rotational.parent.mkdir(parents=True)
        rotational.write_text("0\n")
        (sys_block / name / "slaves").mkdir()

    assert sysfs_backing_devices("/dev/md0[/offload-scratch]", sys_block) == {
        "method": "sysfs",
        "blockdevices": [
            {
                "name": "md0",
                "children": [
                    {"name": "nvme0n1p1", "rota": False, "tran": "nvme"},
                    {"name": "nvme1n1p1", "rota": False, "tran": "nvme"},
                ],
            }
        ],
    }
    verify_nvme_backing(sysfs_backing_devices("/dev/md0", sys_block))


@pytest.mark.parametrize("name,rotational", [("sda", "0\n"), ("nvme0n1", "1\n")])
def test_sysfs_storage_proof_rejects_non_nvme_or_rotational_leaf(
    tmp_path, name, rotational
):
    sys_block = tmp_path / "sys" / "class" / "block"
    (sys_block / name / "slaves").mkdir(parents=True)
    rota = sys_block / name / "queue" / "rotational"
    rota.parent.mkdir(parents=True)
    rota.write_text(rotational)

    with pytest.raises(RuntimeError, match="Could not verify NVMe"):
        verify_nvme_backing(sysfs_backing_devices(f"/dev/{name}", sys_block))


@pytest.mark.parametrize(
    "arm,tp,dram,nvme",
    [
        ("none", 4, 1, 0),
        ("dram", 4, 0, 0),
        ("nvme", 4, 1, 1024),
        ("nvme", 4, 0, 1025),
        ("dram-nvme", 4, 0, 1024),
        ("mystery", 4, 0, 0),
    ],
)
def test_invalid_tier_capacities_fail(arm, tp, dram, nvme):
    with pytest.raises(ValueError):
        connector_config(arm, tp, dram, nvme, Path("/cache"))


@pytest.mark.parametrize(
    "arm,expected", [("none", 0), ("nvme", 0), ("dram", 2), ("dram-nvme", 2)]
)
def test_standard_matrix_preserves_tier_and_experiment(arm, expected):
    entry = {"tp": 4, "spec-decoding": "mtp", "kv-offloading": arm, "conc-list": [7]}
    if arm != "none":
        entry["kv-offload-backend"] = {
            "name": "vllm-native" if arm == "dram-nvme" else "vllm-simple"
        }
    config = SingleNodeMasterConfigEntry.model_validate(
        {
            "experiment": "agentx-offload",
            "image": "fixture",
            "model": "nvidia/MiniMax-M3-NVFP4",
            "model-prefix": "minimaxm3",
            "precision": "fp4",
            "framework": "vllm",
            "runner": "cluster:b200-nscale",
            "multinode": False,
            "scenarios": {
                "agentic-coding": [{"dram-utilization": 0.5, "search-space": [entry]}]
            },
        }
    ).model_dump(by_alias=True, exclude_none=True)
    runners = {
        "labels": {"cluster:b200-nscale": ["fixture_0"]},
        "hardware": {
            "cluster:b200-nscale": {"gpus-per-node": 8, "available-cpu-dram-mib": 8192}
        },
    }
    rows = generate_test_config_sweep(
        argparse.Namespace(config_keys=["fixture"], conc=[7]),
        {"fixture": config},
        runners,
    )
    assert len(rows) == 1
    assert rows[0]["experiment"] == "agentx-offload"
    assert rows[0]["kv-offloading"] == arm
    # 8192 MiB * 50% utilization * 4/8 GPUs = 2 GiB -> floor(2.147 decimal GB).
    assert rows[0]["total-cpu-dram-gb"] == expected
    assert rows[0]["conc"] == 7
    assert rows[0]["duration"] == 3600


@pytest.mark.parametrize(
    "mismatch,guard", [(False, False), (True, False), (False, True)]
)
def test_cleanup_requires_ownership_and_preserves_evidence(
    tmp_path, monkeypatch, mismatch, guard
):
    scratch_root = tmp_path / "nvme"
    scratch = scratch_root / "owned-run"
    cache = scratch / "cache"
    cache.mkdir(parents=True)
    (cache / "kv.bin").write_bytes(b"disposable cache")
    unrelated = scratch_root / "shared-model"
    unrelated.write_bytes(b"keep")
    result = tmp_path / "results"
    result.mkdir()
    owner = {"study": "test", "run": "123", "attempt": "1", "job": "456"}
    cfg = {
        "study": {"study": "test"},
        "scratch": str(scratch),
        **{k: owner[k] for k in ("run", "attempt", "job")},
    }
    runtime.write_json(result / "offload_config.json", cfg)
    runtime.write_json(
        scratch / "owner.json", {**owner, "run": "999"} if mismatch else owner
    )
    if guard:
        runtime.write_json(result / "offload-guard.json", {"reason": "test guard"})
    monkeypatch.setenv("RESULT_DIR", str(result))
    monkeypatch.setattr(runtime, "SCRATCH_ROOT", scratch_root)
    if mismatch or guard:
        with pytest.raises(
            RuntimeError, match="identity mismatch" if mismatch else "invalidated"
        ):
            runtime.finish(0)
    else:
        runtime.finish(0)
    assert unrelated.read_bytes() == b"keep"
    assert scratch.exists() == mismatch
    assert (result / "offload_config.json").exists()
    if not mismatch:
        receipt = json.loads((result / "offload_cleanup.json").read_text())
        assert receipt["deleted"] is True
        assert receipt["usage"]["logical_bytes"] == 16
