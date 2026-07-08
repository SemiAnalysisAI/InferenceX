"""Tests for cluster-owned runtime settings in configs/runners.yaml."""

from __future__ import annotations

from pathlib import Path
import re
import re

import yaml

from runner_config import load_config, resolve_model, resolve_paths, shell_export


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_CONFIG = load_config(REPO_ROOT / "configs" / "runners.yaml")


def test_framework_specific_mapping_wins_over_generic_mapping() -> None:
    trt = resolve_model(
        RUNNER_CONFIG,
        "gb300-nv",
        "glm5",
        "fp4",
        "dynamo-trt",
        "nvidia/GLM-5-NVFP4",
    )
    sglang = resolve_model(
        RUNNER_CONFIG,
        "gb300-nv",
        "glm5",
        "fp4",
        "dynamo-sglang",
        "nvidia/GLM-5-NVFP4",
    )

    assert trt["SRT_SLURM_MODEL_PREFIX"] == "nvidia/GLM-5-NVFP4"
    assert trt["SERVED_MODEL_NAME"] == "glm-5-nvfp4"
    assert sglang["SRT_SLURM_MODEL_PREFIX"] == "glm-5-fp4"


def test_wildcard_mapping_formats_public_model_values() -> None:
    values = resolve_model(
        RUNNER_CONFIG,
        "cluster:gb200-nv",
        "future-model",
        "bf16",
        "dynamo-sglang",
        "org/Future-Model",
    )

    assert values == {
        "MODEL_PATH": "org/Future-Model",
        "MODEL_PATH_LAYOUT": "direct",
        "SRT_SLURM_MODEL_PREFIX": "future-model",
        "SERVED_MODEL_NAME": "org/Future-Model",
    }


def test_every_registry_mapping_is_reachable() -> None:
    for cluster, cluster_config in RUNNER_CONFIG["clusters"].items():
        for entry in cluster_config["models"]:
            model_prefix = entry["model-prefix"].replace("*", "probe-model")
            precision = entry["precision"].replace("*", "probe-precision")
            framework = entry.get("framework", "probe-framework").replace(
                "*", "probe-framework"
            )
            values = resolve_model(
                RUNNER_CONFIG,
                cluster,
                model_prefix,
                precision,
                framework,
                "org/Probe-Model",
            )

            assert values["MODEL_PATH"]
            assert values["SRT_SLURM_MODEL_PREFIX"]
            assert values["SERVED_MODEL_NAME"]


def test_shell_output_quotes_untrusted_values() -> None:
    assert shell_export("MODEL_PATH", "/models/name with 'quotes'") == (
        "export MODEL_PATH='/models/name with '\"'\"'quotes'\"'\"''"
    )


def test_cluster_paths_expand_workspace_and_stable_names() -> None:
    values = resolve_paths(
        RUNNER_CONFIG,
        "mi355x-amds",
        "/workspace/name with spaces",
    )

    assert values["RUNNER_PATH_BENCHMARK_LOGS"] == (
        "/workspace/name with spaces/benchmark_logs"
    )
    assert values["RUNNER_PATH_MODEL_ROOT"] == "/it-share/data"

    b200_values = resolve_paths(
        RUNNER_CONFIG,
        "b200-dgxc",
        "/workspace",
    )
    assert b200_values["RUNNER_PATH_NODE_LOCAL_DSV4_MODEL"] == (
        "/raid/models/DeepSeek-V4-Pro-NVFP4"
    )


def test_srt_slurm_uses_one_immutable_main_snapshot() -> None:
    srt = RUNNER_CONFIG["srt-slurm"]

    assert srt["repository"] == "https://github.com/NVIDIA/srt-slurm.git"
    assert srt["branch"] == "main"
    assert len(srt["revision"]) == 40
    assert len(srt["legacy-recipes-revision"]) == 40
    assert srt["revision"] != srt["legacy-recipes-revision"]


def test_multinode_launchers_load_cluster_model_registry() -> None:
    launchers = (
        "launch_h100-dgxc-slurm.sh",
        "launch_h200-dgxc-slurm.sh",
        "launch_b200-dgxc.sh",
        "launch_b300-nv.sh",
        "launch_gb200-nv.sh",
        "launch_gb300-nv.sh",
        "launch_mi355x-amds.sh",
    )

    for launcher in launchers:
        contents = (REPO_ROOT / "runners" / launcher).read_text(encoding="utf-8")
        assert "load_runner_model" in contents, launcher

    mi_launcher = (
        REPO_ROOT / "runners" / "launch_mi355x-amds.sh"
    ).read_text(encoding="utf-8")
    assert '[[ "$MODEL_PATH_LAYOUT" != "root" ]]' in mi_launcher


def test_every_launcher_path_variable_is_declared_for_its_cluster() -> None:
    launchers = {
        "h100-dgxc": "launch_h100-dgxc-slurm.sh",
        "h200-dgxc": "launch_h200-dgxc-slurm.sh",
        "b200-dgxc": "launch_b200-dgxc.sh",
        "b300-nv": "launch_b300-nv.sh",
        "gb200-nv": "launch_gb200-nv.sh",
        "gb300-nv": "launch_gb300-nv.sh",
        "mi355x-amds": "launch_mi355x-amds.sh",
    }

    for cluster, launcher in launchers.items():
        contents = (REPO_ROOT / "runners" / launcher).read_text(encoding="utf-8")
        referenced = set(re.findall(r"\bRUNNER_PATH_[A-Z0-9_]+", contents))
        declared = set(resolve_paths(RUNNER_CONFIG, cluster, "/workspace"))

        assert not referenced - declared, (
            f"{launcher} references undeclared paths: "
            f"{sorted(referenced - declared)}"
        )


def test_b300_loads_cluster_paths_before_single_or_multinode_branch() -> None:
    launcher = (
        REPO_ROOT / "runners" / "launch_b300-nv.sh"
    ).read_text(encoding="utf-8")

    assert launcher.index("load_runner_paths b300-nv") < launcher.index(
        'if [[ "$IS_MULTINODE" == "true" ]]'
    )


def test_every_nvidia_multinode_config_has_a_cluster_model_mapping() -> None:
    with (REPO_ROOT / "configs" / "nvidia-master.yaml").open(
        encoding="utf-8"
    ) as config_file:
        master_config = yaml.safe_load(config_file)

    launcher_clusters = {
        "h100-dgxc-slurm": "h100-dgxc",
        "h200-dgxc-slurm": "h200-dgxc",
        "b200-dgxc": "b200-dgxc",
        "b300-nv": "b300-nv",
        "gb200-nv": "gb200-nv",
        "gb300-nv": "gb300-nv",
    }
    missing: list[str] = []

    for config_key, entry in master_config.items():
        if not entry.get("multinode"):
            continue
        runners = RUNNER_CONFIG["labels"].get(entry["runner"], [entry["runner"]])
        for launcher, cluster in launcher_clusters.items():
            if not any(runner.split("_", 1)[0] == launcher for runner in runners):
                continue
            try:
                resolve_model(
                    RUNNER_CONFIG,
                    cluster,
                    entry["model-prefix"],
                    entry["precision"],
                    entry["framework"],
                    entry["model"],
                )
            except ValueError as error:
                missing.append(f"{config_key} ({cluster}): {error}")

    assert not missing, "missing runner model mappings:\n" + "\n".join(missing)
