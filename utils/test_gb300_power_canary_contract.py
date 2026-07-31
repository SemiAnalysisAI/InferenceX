"""Static contract for the one-shot GB300 PR2 dcgm-power canary staging."""

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPE_PATH = REPO_ROOT / "benchmarks/multi_node/srt-slurm-recipes/sglang/qwen3.5/gb300-fp8/8k1k/1p1d-tp4-tp4.yaml"
MASTER_PATH = REPO_ROOT / "configs/nvidia-master.yaml"


def test_recipe_is_single_point_dcgm_power():
    recipe = yaml.safe_load(RECIPE_PATH.read_text())

    assert recipe["benchmark"]["concurrencies"] == "4"
    assert recipe["resources"]["prefill_nodes"] == 1
    assert recipe["resources"]["decode_nodes"] == 1
    assert recipe["resources"]["gpus_per_node"] == 4

    telemetry = recipe["telemetry"]
    assert telemetry["enabled"] is True
    assert telemetry["provider"] == "dcgm-power"
    assert telemetry["required"] is True
    assert telemetry["dcgm_exporter"]["container_image"] == "dcgm-exporter"


def test_master_stages_exactly_one_pinned_canary_job():
    master = yaml.safe_load(MASTER_PATH.read_text())

    # The sweep filter matches section keys via startswith("qwen3.5"); the MTP
    # sibling must stay staged out or it adds rows to the canary matrix.
    assert "qwen3.5-fp8-gb300-dynamo-sglang-mtp" not in master

    scenarios = master["qwen3.5-fp8-gb300-dynamo-sglang"]["scenarios"]["fixed-seq-len"]

    assert len(scenarios) == 1
    assert scenarios[0]["isl"] == 8192
    assert scenarios[0]["osl"] == 1024

    search_space = scenarios[0]["search-space"]
    assert len(search_space) == 1
    assert search_space[0]["conc-list"] == [4]
    assert search_space[0]["prefill"]["num-worker"] == 1
    assert search_space[0]["prefill"]["tp"] == 4
    assert search_space[0]["decode"]["num-worker"] == 1
    assert search_space[0]["decode"]["tp"] == 4

    settings = search_space[0]["prefill"]["additional-settings"]
    assert "ENABLE_DCGM_POWER_CANARY=1" in settings
    assert "SRT_SLURM_REPO_URL=https://github.com/edwingao28/srt-slurm.git" in settings

    refs = [s for s in settings if s.startswith("SRT_SLURM_REF=")]
    assert len(refs) == 1
    # The producer pin must be a literal 40-hex SHA, never a branch name.
    assert re.fullmatch(r"SRT_SLURM_REF=[0-9a-f]{40}", refs[0])
