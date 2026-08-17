"""High-signal contracts for ATOM/Infera validation on MI355X."""

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPE_ROOT = (
    REPO_ROOT
    / "benchmarks/multi_node/srt-slurm-recipes/atom/qwen3-0.6b/mi355x"
)
AGG_RECIPE = RECIPE_ROOT / "agg-2w-fixed-seq.yaml"
DISAGG_RECIPE = RECIPE_ROOT / "disagg-1p1d-fixed-seq.yaml"
MASTER_CONFIG = REPO_ROOT / "configs/amd-master.yaml"
LAUNCHER = REPO_ROOT / "runners/launch_mi355x-amds-srt.sh"


def test_aggregate_exercises_two_atom_workers_and_kv_aware_infera():
    recipe = yaml.safe_load(AGG_RECIPE.read_text())

    assert recipe["resources"] == {
        "gpu_type": "mi355x",
        "gpus_per_node": 8,
        "agg_nodes": 1,
        "agg_workers": 2,
        "gpus_per_agg": 1,
    }
    assert recipe["frontend"]["type"] == "infera"
    assert recipe["frontend"]["args"]["router-policy"] == "kv-aware"
    assert recipe["backend"]["type"] == "atom"
    assert recipe["backend"]["enable_kv_events"] is True
    assert recipe["backend"]["aggregated_environment"]["PYTHONPATH"] == (
        "/atom-source:/infera-source"
    )
    command = recipe["benchmark"]["command"]
    assert "--random-prefix-len 96" in command
    assert "--endpoint /v1/completions" in command


def test_disaggregate_uses_one_prefill_one_decode_and_mooncake():
    recipe = yaml.safe_load(DISAGG_RECIPE.read_text())

    assert recipe["resources"]["prefill_workers"] == 1
    assert recipe["resources"]["decode_workers"] == 1
    assert recipe["resources"]["prefill_nodes"] == 1
    assert recipe["resources"]["decode_nodes"] == 1
    assert recipe["backend"]["type"] == "atom"
    assert recipe["backend"]["connector"] == "mooncake"
    assert recipe["backend"]["mooncake_protocol"] == "tcp"
    assert recipe["backend"]["enable_kv_events"] is True
    assert recipe["backend"]["prefill_environment"]["PYTHONPATH"] == (
        "/atom-source:/infera-source"
    )


def test_matrix_rows_route_only_through_the_atom_srt_launcher():
    configs = yaml.safe_load(MASTER_CONFIG.read_text())
    launcher = LAUNCHER.read_text()

    agg = configs["qwen3-0.6b-fp16-mi355x-atom-infera-srt-agg"]
    disagg = configs["qwen3-0.6b-fp16-mi355x-atom-infera-srt-disagg"]
    assert agg["framework"] == "atom"
    assert agg["router"] == {"name": "infera", "version": "0.1.1"}
    assert agg["scenarios"]["fixed-seq-len"][0]["search-space"][0][
        "prefill"
    ]["additional-settings"] == [
        "CONFIG_FILE=recipes/atom/qwen3-0.6b/mi355x/agg-2w-fixed-seq.yaml"
    ]
    assert disagg["framework"] == "atom-disagg"
    assert disagg["kv-p2p-transfer"] == "mooncake"
    assert disagg["scenarios"]["fixed-seq-len"][0]["search-space"][0][
        "prefill"
    ]["additional-settings"] == [
        "CONFIG_FILE=recipes/atom/qwen3-0.6b/mi355x/disagg-1p1d-fixed-seq.yaml"
    ]
    assert "5ecfb13d1ba0960045482f1ef006312d8729d37a" in launcher
    assert "8ed8f1728c745d4e91ba9eaa09ed81159aa57e41" in launcher
    assert "2ab42bc2c64d1ad04f698c396da48473e71a6dbb" in launcher
    assert "ensure_git_checkout()" in launcher
    assert 'exec {lock_fd}>"\\${target}.lock"' in launcher
    assert 'flock -w 2400 "\\$lock_fd"' in launcher
    assert 'mv -T "\\$temporary" "\\$target"' in launcher
    assert "scancel" not in launcher
