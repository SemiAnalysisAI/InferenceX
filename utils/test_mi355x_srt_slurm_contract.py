"""High-signal contract checks for the MI355X srt-slurm bring-up lanes."""

import os
import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPE_ROOT = (
    REPO_ROOT
    / "benchmarks/multi_node/srt-slurm-recipes/sglang/qwen3-0.6b/mi355x"
)
AGG_RECIPE = RECIPE_ROOT / "agg-fixed-seq.yaml"
DISAGG_RECIPE = RECIPE_ROOT / "disagg-1p1d-fixed-seq.yaml"
PRODUCTION_DISAGG_RECIPE = (
    REPO_ROOT
    / "benchmarks/multi_node/srt-slurm-recipes/sglang/qwen3.5/mi355x/disagg-1p1d-tp8-fixed-seq.yaml"
)
CLUSTER_PROFILE = (
    REPO_ROOT
    / "benchmarks/multi_node/srt-slurm-recipes/cluster-configs/mi355x-amds.yaml"
)
MASTER_CONFIG = REPO_ROOT / "configs/amd-master.yaml"
LAUNCHER = REPO_ROOT / "runners/launch_mi355x-amds-srt.sh"
LEGACY_LAUNCHER = REPO_ROOT / "runners/launch_mi355x-amds.sh"


def test_cluster_profile_matches_the_mi355x_rocm_slurm_contract():
    cluster = yaml.safe_load(CLUSTER_PROFILE.read_text())

    assert cluster["accelerator_vendor"] == "amd"
    assert cluster["network_interface"] == "eno0"
    assert cluster["gpu_sbatch_directive"] == "gres"
    assert cluster["use_segment_sbatch_directive"] is False
    assert cluster["use_exclusive_sbatch_directive"] is False
    assert cluster["runtime_config_transport"] == "shared-filesystem"
    assert cluster["default_mounts"]["/dev/kfd"] == "/dev/kfd"
    assert cluster["default_mounts"]["/dev/dri"] == "/dev/dri"
    assert cluster["containers"]["sglang-rocm-v0.5.16-mi35x"].endswith(
        "/sglang-rocm-v0.5.16-mi35x-20260728.sqsh"
    )
    assert cluster["output_dir"].startswith("/it-share/gharunners2/srt-slurm/")
    assert cluster["containers"]["sglang-rocm-v0.5.16-mi35x"].startswith(
        "/it-share/gharunners2/srt-slurm/"
    )


def test_recipes_use_native_sglang_router_and_only_disagg_uses_mori():
    agg = yaml.safe_load(AGG_RECIPE.read_text())
    disagg = yaml.safe_load(DISAGG_RECIPE.read_text())

    assert agg["resources"] == {
        "gpu_type": "mi355x",
        "gpus_per_node": 1,
        "agg_nodes": 1,
        "agg_workers": 1,
        "gpus_per_agg": 1,
    }
    assert agg["frontend"]["type"] == "sgl-router"
    assert agg["backend"]["type"] == "sglang"
    assert disagg["frontend"]["type"] == "sgl-router"
    assert disagg["backend"]["type"] == "sglang"
    for role in ("prefill", "decode"):
        assert disagg["backend"]["sglang_config"][role][
            "disaggregation-transfer-backend"
        ] == "mori"
        assert "rdma0" in disagg["backend"][f"{role}_environment"]["IBDEVICES"]

    assert "mori" not in AGG_RECIPE.read_text().lower()
    for recipe in (AGG_RECIPE, DISAGG_RECIPE, PRODUCTION_DISAGG_RECIPE):
        text = recipe.read_text().lower()
        assert "dynamo" not in text
        assert "nixl" not in text
        assert "nats" not in text
        assert "etcd" not in text


def test_matrix_rows_explicitly_select_the_srt_recipes():
    master = yaml.safe_load(MASTER_CONFIG.read_text())
    expected = {
        "qwen3-0.6b-fp16-mi355x-sglang-srt-agg": (
            False,
            "recipes/sglang/qwen3-0.6b/mi355x/agg-fixed-seq.yaml",
        ),
        "qwen3-0.6b-fp16-mi355x-sglang-srt-disagg": (
            True,
            "recipes/sglang/qwen3-0.6b/mi355x/disagg-1p1d-fixed-seq.yaml",
        ),
        "qwen3.5-fp8-mi355x-sglang-srt-disagg": (
            True,
            "recipes/sglang/qwen3.5/mi355x/disagg-1p1d-tp8-fixed-seq.yaml",
        ),
    }
    for name, (is_disagg, recipe) in expected.items():
        config = master[name]
        search = config["scenarios"]["fixed-seq-len"][0]["search-space"][0]
        assert config["runner"] == "cluster:mi355x-amds"
        assert config["multinode"] is True
        assert config["disagg"] is is_disagg
        assert search["prefill"]["additional-settings"] == [
            f"CONFIG_FILE={recipe}"
        ]


def test_launcher_pins_runtime_and_preserves_legacy_default():
    launcher = LAUNCHER.read_text()
    legacy = LEGACY_LAUNCHER.read_text()

    assert "25e1e4e71dc8e7383b1857041decff1a9ae0339e" in launcher
    assert "v0.5.16-rocm720-mi35x-20260728" in launcher
    assert "make setup-compute ARCH=x86_64" in launcher
    assert "SRTCTL_RUNTIME_SOURCE_DIR" in launcher
    assert 'SHARED_BASE="/it-share/gharunners2/srt-slurm"' in launcher
    assert "/it-share/inferencex" not in launcher
    assert "scancel" not in launcher
    assert 'snapshot_download(os.environ["MODEL_REPO"])' in launcher
    assert 'if [[ -n "${CONFIG_FILE:-}" ]]; then' in legacy
    assert "launch_mi355x-amds-srt.sh" in legacy


def test_fixed_sequence_commands_execute_with_attached_arguments(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        "#!/bin/bash\n"
        'printf "%s\\n" "$@" >> "$FAKE_ARGS_LOG"\n'
        'printf "%s\\n" --CALL-END-- >> "$FAKE_ARGS_LOG"\n'
    )
    fake_python.chmod(0o755)

    for recipe_path in (AGG_RECIPE, DISAGG_RECIPE):
        command = yaml.safe_load(recipe_path.read_text())["benchmark"]["command"]
        result_dir = tmp_path / recipe_path.stem
        command = command.replace(
            'result_root="/results/${SLURM_JOB_ID}"',
            f'result_root="{result_dir}"',
        )
        args_log = tmp_path / f"{recipe_path.stem}.args"
        env = {
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "FAKE_ARGS_LOG": str(args_log),
            "SRT_FRONTEND_HOST": "127.0.0.1",
            "SRT_FRONTEND_PORT": "8000",
            "SLURM_JOB_ID": "123",
        }

        subprocess.run(["bash", "-n"], input=command, text=True, check=True)
        subprocess.run(["bash", "-c", command], env=env, check=True)
        calls = args_log.read_text().split("--CALL-END--\n")
        calls = [[arg for arg in call.splitlines() if arg] for call in calls if call]

        assert len(calls) == 2
        assert [call[call.index("--num-prompts") + 1] for call in calls] == [
            "4",
            "16",
        ]
        assert all("--model" in call and "Qwen/Qwen3-0.6B" in call for call in calls)


def test_production_disagg_recipe_uses_two_full_nodes_and_the_existing_workload():
    recipe = yaml.safe_load(PRODUCTION_DISAGG_RECIPE.read_text())

    assert recipe["resources"] == {
        "gpu_type": "mi355x",
        "gpus_per_node": 8,
        "prefill_nodes": 1,
        "decode_nodes": 1,
        "prefill_workers": 1,
        "decode_workers": 1,
        "gpus_per_prefill": 8,
        "gpus_per_decode": 8,
    }
    assert recipe["frontend"]["type"] == "sgl-router"
    for role in ("prefill", "decode"):
        config = recipe["backend"]["sglang_config"][role]
        assert config["tensor-parallel-size"] == 8
        assert config["disaggregation-transfer-backend"] == "mori"
        assert config["attention-backend"] == "aiter"
    command = recipe["benchmark"]["command"]
    assert "utils/bench_serving/benchmark_serving.py" in command
    assert "--random-input-len 8192" in command
    assert "--random-output-len 1024" in command
    assert "--max-concurrency \"${concurrency}\"" in command
