"""High-signal contract checks for the MI300X srt-slurm bring-up lane."""

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPE_PATH = (
    REPO_ROOT
    / "benchmarks/multi_node/srt-slurm-recipes/vllm/qwen3-0.6b/mi300x/agg-fixed-seq.yaml"
)
CLUSTER_PATH = (
    REPO_ROOT
    / "benchmarks/multi_node/srt-slurm-recipes/cluster-configs/mi300x-amds.yaml"
)


def test_mi300x_cluster_uses_the_rocm_slurm_contract():
    cluster = yaml.safe_load(CLUSTER_PATH.read_text())

    assert cluster["accelerator_vendor"] == "amd"
    assert cluster["gpu_sbatch_directive"] == "gres"
    assert cluster["use_segment_sbatch_directive"] is False
    assert cluster["default_mounts"]["/dev/kfd"] == "/dev/kfd"
    assert cluster["default_mounts"]["/dev/dri"] == "/dev/dri"


def test_fixed_sequence_recipe_uses_inferencex_custom_benchmark():
    recipe = yaml.safe_load(RECIPE_PATH.read_text())
    benchmark = recipe["benchmark"]
    command = benchmark["command"]

    assert benchmark["type"] == "custom"
    assert "/infmax-workspace/utils/bench_serving/benchmark_serving.py" in command
    assert "--backend openai-chat" in command
    assert "--endpoint /v1/chat/completions" in command
    assert "--random-input-len 128" in command
    assert "--random-output-len 32" in command
    assert "--random-range-ratio 1.0" in command
    assert "best-of" not in command
    assert "sa-bench" not in command
