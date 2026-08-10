"""High-signal contract checks for the MI300X srt-slurm bring-up lane."""

import os
import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPE_PATH = (
    REPO_ROOT
    / "benchmarks/multi_node/srt-slurm-recipes/vllm/qwen3-0.6b/mi300x/agg-fixed-seq.yaml"
)
DISAGG_RECIPE_PATH = RECIPE_PATH.with_name("disagg-1p1d-fixed-seq.yaml")
CLUSTER_PATH = (
    REPO_ROOT
    / "benchmarks/multi_node/srt-slurm-recipes/cluster-configs/mi300x-amds.yaml"
)


def test_mi300x_cluster_uses_the_rocm_slurm_contract():
    cluster = yaml.safe_load(CLUSTER_PATH.read_text())

    assert cluster["accelerator_vendor"] == "amd"
    assert cluster["network_interface"] is None
    assert cluster["gpu_sbatch_directive"] == "gres"
    assert cluster["use_segment_sbatch_directive"] is False
    assert cluster["runtime_config_transport"] == "embedded"
    assert cluster["default_sbatch_directives"]["exclude"] == (
        "chi-mi300x-049,chi-mi300x-121"
    )
    assert cluster["default_mounts"]["/dev/kfd"] == "/dev/kfd"
    assert cluster["default_mounts"]["/dev/dri"] == "/dev/dri"
    image_path = cluster["containers"]["vllm-rocm-v0.26.0"]
    assert image_path.endswith("/vllm-openai-rocm-v0.26.0.sqsh")
    router_path = cluster["containers"]["vllm-router-20260809"]
    assert router_path.endswith("/vllm-router-nightly-20260809-d2ba586.sqsh")

    for recipe_path in (RECIPE_PATH, DISAGG_RECIPE_PATH):
        recipe = yaml.safe_load(recipe_path.read_text())
        assert recipe["model"]["container"] == "vllm-rocm-v0.26.0"
        assert recipe["identity"]["container"]["image"] == (
            "vllm/vllm-openai-rocm:v0.26.0"
        )


def test_disaggregated_recipe_uses_native_router_and_moriio():
    recipe = yaml.safe_load(DISAGG_RECIPE_PATH.read_text())

    assert recipe["frontend"] == {
        "type": "vllm-router",
        "enable_multiple_frontends": False,
        "container_image": "vllm-router-20260809",
        "args": {
            "policy": "consistent_hash",
            "prefill-policy": "consistent_hash",
            "decode-policy": "consistent_hash",
        },
    }
    assert recipe["backend"]["connector"] == "moriio"
    for role in ("prefill", "decode"):
        assert recipe["backend"][f"{role}_environment"][
            "VLLM_ROCM_USE_AITER"
        ] == "1"
        assert recipe["backend"]["vllm_config"][role][
            "attention-backend"
        ] == "ROCM_AITER_FA"
    assert "dynamo" not in recipe
    serialized = DISAGG_RECIPE_PATH.read_text().lower()
    assert "nixl" not in serialized
    assert "nats" not in serialized
    assert "etcd" not in serialized


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


def test_fixed_sequence_commands_keep_all_arguments_attached(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        "#!/bin/bash\n"
        'printf "%s\\n" "$@" >> "$FAKE_ARGS_LOG"\n'
        'printf "%s\\n" --CALL-END-- >> "$FAKE_ARGS_LOG"\n'
    )
    fake_python.chmod(0o755)

    for recipe_path in (RECIPE_PATH, DISAGG_RECIPE_PATH):
        command = yaml.safe_load(recipe_path.read_text())["benchmark"]["command"]
        result_dir = tmp_path / recipe_path.stem
        command = command.replace("/logs/fixed-seq", str(result_dir))
        args_log = tmp_path / f"{recipe_path.stem}.args"
        env = {
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "FAKE_ARGS_LOG": str(args_log),
            "SRT_FRONTEND_HOST": "127.0.0.1",
            "SRT_FRONTEND_PORT": "8000",
        }

        subprocess.run(["bash", "-n"], input=command, text=True, check=True)
        subprocess.run(["bash", "-c", command], env=env, check=True)

        calls = args_log.read_text().split("--CALL-END--\n")
        calls = [[arg for arg in call.splitlines() if arg] for call in calls if call]
        assert len(calls) == 2
        assert all("--model" in call and "Qwen/Qwen3-0.6B" in call for call in calls)
        assert [call[call.index("--num-prompts") + 1] for call in calls] == ["4", "16"]
