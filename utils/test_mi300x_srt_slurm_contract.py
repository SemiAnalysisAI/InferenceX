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
ATOM_RECIPE_PATH = (
    REPO_ROOT
    / "benchmarks/multi_node/srt-slurm-recipes/atom/qwen3-0.6b/mi300x/agg-2w-fixed-seq.yaml"
)
ATOM_DISAGG_RECIPE_PATH = ATOM_RECIPE_PATH.with_name("disagg-1p1d-fixed-seq.yaml")
CLUSTER_PATH = (
    REPO_ROOT
    / "benchmarks/multi_node/srt-slurm-recipes/cluster-configs/mi300x-amds.yaml"
)
MASTER_CONFIG_PATH = REPO_ROOT / "configs/amd-master.yaml"
SRT_LAUNCHER_PATH = REPO_ROOT / "runners/launch_mi300x-amds-srt.sh"


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


def test_official_matrix_routes_disagg_through_the_pinned_srt_launcher():
    config = yaml.safe_load(MASTER_CONFIG_PATH.read_text())[
        "qwen3-0.6b-fp16-mi300x-vllm-srt-disagg"
    ]
    search = config["scenarios"]["fixed-seq-len"][0]["search-space"][0]
    launcher = SRT_LAUNCHER_PATH.read_text()

    assert config["runner"] == "mi300x-disagg"
    assert config["router"] == {
        "name": "vllm-router",
        "version": "nightly-20260809-d2ba586",
    }
    assert config["kv-p2p-transfer"] == "moriio"
    assert search["prefill"]["additional-settings"] == [
        "CONFIG_FILE=recipes/vllm/qwen3-0.6b/mi300x/"
        "disagg-1p1d-fixed-seq.yaml"
    ]
    assert "141f035b5539fa8bbc1b4018ae4817283093092d" in launcher
    assert launcher.count("setup ARCH=x86_64") == 2
    assert "--no-preflight" in launcher
    assert 'ENROOT_RUNTIME_PATH="\\${TMPDIR:-/tmp}/enroot-runtime-\\${UID}"' in launcher
    assert "for attempt in 1 2 3" in launcher
    assert 'VLLM_IMAGE="vllm/vllm-openai-rocm:v0.26.0"' in launcher
    assert (
        'VLLM_ROUTER_IMAGE="vllm/vllm-router:nightly-20260809-d2ba586"'
        in launcher
    )
    assert 'enroot import -o "\\$tmp" "docker://\\${image}"' in launcher
    assert 'exec {lock_fd}>"\\${target}.lock"' in launcher
    assert 'flock -w 2400 "\\$lock_fd"' in launcher
    assert 'unsquashfs -s "\\$tmp"' in launcher
    assert 'mv "\\$tmp" "\\$target"' in launcher
    assert 'REMOTE_SRT_RUNTIME="${REMOTE_BASE}/runtime/srt-slurm-${SRT_SLURM_COMMIT}"' in launcher
    assert (
        'ensure_git_checkout "\\$srt_runtime" "${SRT_SLURM_REPOSITORY}" '
        '"${SRT_SLURM_COMMIT}"'
        in launcher
    )
    assert 'make -C "\\$srt_runtime" --no-print-directory setup ARCH=x86_64' in launcher
    assert 'export SRTCTL_RUNTIME_SOURCE_DIR="$REMOTE_SRT_RUNTIME"' in launcher
    assert '#SBATCH --nodes=7' in launcher
    assert 'ensure_git_checkout()' in launcher
    assert 'mv "\\$target" "\\$quarantine"' in launcher
    assert 'mv "\\$temporary" "\\$target"' in launcher
    assert "scancel" not in launcher


def test_official_matrix_routes_aggregate_through_the_pinned_srt_launcher():
    config = yaml.safe_load(MASTER_CONFIG_PATH.read_text())[
        "qwen3-0.6b-fp16-mi300x-vllm-srt-agg"
    ]
    search = config["scenarios"]["fixed-seq-len"][0]["search-space"]
    launcher = SRT_LAUNCHER_PATH.read_text()

    assert config["runner"] == "mi300x-disagg"
    assert config["multinode"] is True
    assert config["disagg"] is False
    assert search == [
        {
            "conc-list": [1],
            "prefill": {
                "num-worker": 1,
                "tp": 1,
                "ep": 1,
                "dp-attn": False,
                "additional-settings": [
                    "CONFIG_FILE=recipes/vllm/qwen3-0.6b/mi300x/"
                    "agg-fixed-seq.yaml"
                ],
            },
            "decode": {
                "num-worker": 0,
                "tp": 1,
                "ep": 1,
                "dp-attn": False,
            },
        }
    ]
    assert ': "${CONFIG_FILE:?CONFIG_FILE must name an srt-slurm recipe}"' in launcher
    assert 'JOB_BATCH_HOST=$(scontrol show job "$JOB_ID" -dd' in launcher
    assert '--nodelist="$JOB_BATCH_HOST"' in launcher
    assert "TOTAL_GPUS=$((PREFILL_NUM_WORKERS * PREFILL_TP" in launcher


def test_aggregate_recipe_uses_direct_vllm_without_dynamo_or_a_router():
    recipe = yaml.safe_load(RECIPE_PATH.read_text())

    assert recipe["resources"] == {
        "gpu_type": "mi300x",
        "gpus_per_node": 1,
        "agg_nodes": 1,
        "agg_workers": 1,
        "gpus_per_agg": 1,
    }
    assert recipe["frontend"] == {
        "type": "vllm",
        "enable_multiple_frontends": False,
    }
    assert recipe["backend"]["connector"] is None
    serialized = RECIPE_PATH.read_text().lower()
    assert "dynamo" not in serialized
    assert "nixl" not in serialized
    assert "moriio" not in serialized


def test_fixed_sequence_recipe_uses_inferencex_custom_benchmark():
    recipe = yaml.safe_load(RECIPE_PATH.read_text())
    benchmark = recipe["benchmark"]
    command = benchmark["command"]

    assert benchmark["type"] == "custom"
    assert "/infmax-workspace/utils/bench_serving/benchmark_serving.py" in command
    assert 'result_root="/results/${SLURM_JOB_ID}"' in command
    assert "--backend openai-chat" in command
    assert "--endpoint /v1/chat/completions" in command
    assert "--random-input-len 128" in command
    assert "--random-output-len 32" in command
    assert "--random-range-ratio 1.0" in command
    assert "best-of" not in command
    assert "sa-bench" not in command


def test_atom_recipes_use_infera_and_keep_worker_metrics_honest():
    cluster = yaml.safe_load(CLUSTER_PATH.read_text())
    aggregate = yaml.safe_load(ATOM_RECIPE_PATH.read_text())
    disaggregate = yaml.safe_load(ATOM_DISAGG_RECIPE_PATH.read_text())
    launcher = SRT_LAUNCHER_PATH.read_text()

    assert cluster["containers"]["infera-atom-v0.1.1"].endswith(
        "/infera-atom-v0.1.1.sqsh"
    )
    assert aggregate["resources"]["agg_workers"] == 2
    for recipe in (aggregate, disaggregate):
        assert recipe["model"]["container"] == "infera-atom-v0.1.1"
        assert recipe["identity"]["container"]["image"] == (
            "rocm/infera:atom-v0.1.1"
        )
        assert recipe["frontend"]["type"] == "infera"
        assert recipe["frontend"]["args"]["router-policy"] == "kv-aware"
        assert recipe["backend"]["type"] == "atom"
        assert recipe["backend"]["enable_kv_events"] is True
        command = recipe["benchmark"]["command"]
        assert "--backend openai" in command
        assert "--endpoint /v1/completions" in command
    assert disaggregate["backend"]["connector"] == "mooncake"
    assert disaggregate["backend"]["mooncake_protocol"] == "tcp"
    assert 'ATOM_IMAGE="rocm/infera:atom-v0.1.1"' in launcher
    assert 'INFERA_COMMIT="8ed8f1728c745d4e91ba9eaa09ed81159aa57e41"' in launcher
    assert 'REMOTE_INFERA_RUNTIME="${REMOTE_BASE}/runtime/infera-${INFERA_COMMIT}"' in launcher
    for recipe in (aggregate, disaggregate):
        assert recipe["frontend"]["env"]["PYTHONPATH"] == "/infera-source"
        role_environments = [
            value
            for key, value in recipe["backend"].items()
            if key.endswith("_environment")
        ]
        assert role_environments
        assert all(env["PYTHONPATH"] == "/infera-source" for env in role_environments)


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

    recipe_paths = (
        RECIPE_PATH,
        DISAGG_RECIPE_PATH,
        ATOM_RECIPE_PATH,
        ATOM_DISAGG_RECIPE_PATH,
    )
    for index, recipe_path in enumerate(recipe_paths):
        command = yaml.safe_load(recipe_path.read_text())["benchmark"]["command"]
        result_dir = tmp_path / f"recipe-{index}"
        command = command.replace(
            'result_root="/results/${SLURM_JOB_ID}"',
            f'result_root="{result_dir}"',
        )
        args_log = tmp_path / f"recipe-{index}.args"
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
        assert all("--model" in call and "Qwen/Qwen3-0.6B" in call for call in calls)
        assert [call[call.index("--num-prompts") + 1] for call in calls] == ["4", "16"]
