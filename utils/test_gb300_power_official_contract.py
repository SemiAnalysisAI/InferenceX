"""Static contract for the official GB300 dcgm-power lane.

Shares helpers with the GB200 contract; GB300 specifics are the literal
shared squash cache path (no SQUASH_DIR var in this launcher), the 19401
exporter port, and the preserved v1.0.25/sa-submission non-power refs.
"""

import yaml

from test_gb200_power_official_contract import (
    REPO_ROOT,
    assert_exporter_provisioning,
    assert_pinned_clone_contract,
    assert_recipe_driven_detection,
)

RECIPE_PATH = REPO_ROOT / "benchmarks/multi_node/srt-slurm-recipes/sglang/qwen3.5/gb300-fp8/8k1k/1p1d-tp4-tp4.yaml"
LAUNCHER_PATH = REPO_ROOT / "runners/launch_gb300-nv.sh"


def test_recipe_declares_enabled_dcgm_power_lane():
    recipe = yaml.safe_load(RECIPE_PATH.read_text())

    telemetry = recipe["telemetry"]
    assert telemetry["enabled"] is True
    assert telemetry["provider"] == "dcgm-power"
    assert telemetry["required"] is True
    assert telemetry["dcgm_exporter"]["container_image"] == "dcgm-exporter"
    # 9401 is already bound by the cluster-level exporter on im-gb300 nodes.
    assert telemetry["dcgm_exporter"]["port"] == 19401

    assert recipe["benchmark"]["concurrencies"] == "1x2x4x8x16x32x64x128"
    assert recipe["resources"]["prefill_nodes"] == 1
    assert recipe["resources"]["decode_nodes"] == 1
    assert recipe["resources"]["gpus_per_node"] == 4


def test_launcher_detects_power_lane_from_recipe():
    assert_recipe_driven_detection(LAUNCHER_PATH.read_text())


def test_launcher_provisions_exporter_through_shared_squash_path():
    launcher = LAUNCHER_PATH.read_text()
    assert_exporter_provisioning(launcher)
    # No SQUASH_DIR var here; the /data/ mount avoids the /home NFS ELOOP bug.
    assert 'DCGM_EXPORTER_SQSH="/data/home/sa-shared/gharunners/squash/' in launcher
    assert 'srun --partition=$SLURM_PARTITION --exclusive --time=30 bash -c "unsquashfs -l' in launcher


def test_launcher_pins_power_producer():
    assert_pinned_clone_contract(LAUNCHER_PATH.read_text())


def test_non_power_lane_keeps_existing_ref_logic():
    launcher = LAUNCHER_PATH.read_text()
    assert 'git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"' in launcher
    assert "git checkout v1.0.25" in launcher
    assert "git checkout sa-submission-q2-2026" in launcher
