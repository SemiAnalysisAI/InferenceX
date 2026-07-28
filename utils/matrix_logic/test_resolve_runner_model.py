"""Tests for cluster-local model path resolution."""

from pathlib import Path

import pytest

from utils.matrix_logic.validation import load_config_files, load_runner_file
from utils.resolve_runner_model import resolve_runner_model, shell_exports

REPO_ROOT = Path(__file__).resolve().parents[2]


def runner_config(model_paths, allow_override=False):
    """Build a minimal runner model configuration."""
    return {
        "models": {
            "cluster:b200-dgxc": {
                "default": {
                    "dsv4": {
                        "fp4": {
                            "model-paths": model_paths,
                            "srt-slurm-model-prefix": "deepseek-v4-pro",
                            "allow-model-path-override": allow_override,
                        }
                    }
                }
            }
        }
    }


def test_resolve_uses_first_existing_configured_path():
    """The first existing configured path wins."""
    config = runner_config(["/models/preferred", "/models/fallback"])

    result = resolve_runner_model(
        config,
        "cluster:b200-dgxc",
        "dynamo-vllm",
        "dsv4",
        "fp4",
        path_is_dir=lambda path: path == "/models/fallback",
    )

    assert result == {
        "MODEL_PATH": "/models/fallback",
        "SRT_SLURM_MODEL_PREFIX": "deepseek-v4-pro",
    }


def test_resolve_falls_back_to_first_path_when_none_exist():
    """The first path remains the deterministic fallback."""
    config = runner_config(["/models/preferred", "/models/fallback"])

    result = resolve_runner_model(
        config,
        "cluster:b200-dgxc",
        "dynamo-vllm",
        "dsv4",
        "fp4",
        path_is_dir=lambda _path: False,
    )

    assert result["MODEL_PATH"] == "/models/preferred"


def test_resolve_honors_allowed_existing_override():
    """An allowed, existing MODEL_PATH override takes precedence."""
    config = runner_config(["/models/preferred"], allow_override=True)

    result = resolve_runner_model(
        config,
        "cluster:b200-dgxc",
        "dynamo-vllm",
        "dsv4",
        "fp4",
        current_model_path="/models/operator-override",
        path_is_dir=lambda path: path == "/models/operator-override",
    )

    assert result["MODEL_PATH"] == "/models/operator-override"


def test_resolve_prefers_framework_specific_mapping():
    """An exact framework mapping overrides the runner default."""
    config = runner_config(["/models/default"])
    config["models"]["cluster:b200-dgxc"]["dynamo-trt"] = {
        "dsv4": {
            "fp4": {
                "model-paths": ["/models/trt"],
                "srt-slurm-model-prefix": "deepseek-ai/DeepSeek-V4-Pro",
                "served-model-name": "deepseek-v4-pro",
            }
        }
    }

    result = resolve_runner_model(
        config,
        "cluster:b200-dgxc",
        "dynamo-trt",
        "dsv4",
        "fp4",
        path_is_dir=lambda _path: False,
    )

    assert result == {
        "MODEL_PATH": "/models/trt",
        "SRT_SLURM_MODEL_PREFIX": "deepseek-ai/DeepSeek-V4-Pro",
        "SERVED_MODEL_NAME": "deepseek-v4-pro",
    }


def test_resolve_supports_default_precision_mapping():
    """A default precision supports model mappings that ignore precision."""
    config = runner_config(["/models/default"])
    config["models"]["cluster:b200-dgxc"]["dynamo-trt"] = {
        "gptoss": {
            "default": {
                "model-paths": ["/models/gpt-oss"],
                "served-model-name": "gpt-oss-120b",
            }
        }
    }

    result = resolve_runner_model(
        config,
        "cluster:b200-dgxc",
        "dynamo-trt",
        "gptoss",
        "mxfp4",
        path_is_dir=lambda _path: False,
    )

    assert result == {
        "MODEL_PATH": "/models/gpt-oss",
        "SERVED_MODEL_NAME": "gpt-oss-120b",
    }


def test_resolve_supports_explicit_missing_mapping_fallback():
    """Callers may preserve a portable MODEL fallback for unmapped models."""
    config = runner_config(["/models/default"])

    result = resolve_runner_model(
        config,
        "cluster:b200-dgxc",
        "dynamo-sglang",
        "qwen",
        "bf16",
        fallback_model_path="Qwen/Qwen3",
        path_is_dir=lambda _path: False,
    )

    assert result == {"MODEL_PATH": "Qwen/Qwen3"}


def test_resolve_rejects_unsupported_combination():
    """Missing model combinations fail with the full lookup key."""
    config = runner_config(["/models/preferred"])

    with pytest.raises(ValueError) as exc_info:
        resolve_runner_model(
            config,
            "cluster:b200-dgxc",
            "dynamo-vllm",
            "dsr1",
            "fp8",
            path_is_dir=lambda _path: False,
        )

    assert (
        "cluster:b200-dgxc/dynamo-vllm/dsr1/fp8"
        in str(exc_info.value)
    )


def test_shell_exports_quote_values():
    """Shell output quotes values instead of interpolating them."""
    exports = shell_exports(
        {
            "MODEL_PATH": "/models/path with spaces",
            "SRT_SLURM_MODEL_PREFIX": "model;alias",
        }
    )

    assert exports == (
        "export MODEL_PATH='/models/path with spaces'\n"
        "export SRT_SLURM_MODEL_PREFIX='model;alias'"
    )


def test_resolve_every_configured_runner_model_mapping():
    """Every runners.yaml model entry is reachable through the resolver."""
    config = load_runner_file(str(REPO_ROOT / "configs" / "runners.yaml"))
    export_names = {
        "srt-slurm-model-prefix": "SRT_SLURM_MODEL_PREFIX",
        "served-model-name": "SERVED_MODEL_NAME",
        "model-name": "MODEL_NAME",
    }

    for runner_label, framework_mappings in config["models"].items():
        for framework_key, model_mappings in framework_mappings.items():
            framework = (
                "unconfigured-framework"
                if framework_key == "default"
                else framework_key
            )
            for model_prefix, precision_mappings in model_mappings.items():
                for precision_key, model_config in precision_mappings.items():
                    precision = (
                        "unconfigured-precision"
                        if precision_key == "default"
                        else precision_key
                    )
                    result = resolve_runner_model(
                        config,
                        runner_label,
                        framework,
                        model_prefix,
                        precision,
                        path_is_dir=lambda _path: False,
                    )

                    assert result["MODEL_PATH"] == model_config["model-paths"][0]
                    for config_name, export_name in export_names.items():
                        if config_name in model_config:
                            assert result[export_name] == model_config[config_name]


def test_every_nvidia_multinode_runner_model_resolves():
    """Every multinode config routed to a migrated launcher is supported."""
    runner_config = load_runner_file(
        str(REPO_ROOT / "configs" / "runners.yaml")
    )
    master_config = load_config_files(
        [str(REPO_ROOT / "configs" / "nvidia-master.yaml")]
    )
    cluster_by_runner = {
        "b200": "cluster:b200-dgxc",
        "b200-dsv4": "cluster:b200-dgxc",
        "b200-multinode": "cluster:b200-dgxc",
        "cluster:b200-dgxc": "cluster:b200-dgxc",
        "b300": "cluster:b300-nv",
        "b300-p1": "cluster:b300-nv",
        "cluster:b300-nv": "cluster:b300-nv",
        "gb200": "cluster:gb200-nv",
        "cluster:gb200-nv": "cluster:gb200-nv",
        "gb300": "cluster:gb300-nv",
        "gb300-nv": "cluster:gb300-nv",
        "cluster:gb300-nv": "cluster:gb300-nv",
        "h100-multinode": "cluster:h100-dgxc",
        "cluster:h100-dgxc": "cluster:h100-dgxc",
        "h200-multinode": "cluster:h200-dgxc",
        "cluster:h200-dgxc": "cluster:h200-dgxc",
    }
    strict_gb200_frameworks = {"llmd-vllm", "dynamo-trt", "dynamo-vllm"}
    unsupported = []

    for config_name, benchmark in master_config.items():
        if not benchmark.get("multinode", False):
            continue
        runner_label = cluster_by_runner.get(benchmark["runner"])
        if runner_label is None:
            continue
        fallback_model_path = None
        if (
            runner_label == "cluster:gb200-nv"
            and benchmark["framework"] not in strict_gb200_frameworks
        ):
            fallback_model_path = benchmark["model"]

        try:
            resolve_runner_model(
                runner_config,
                runner_label,
                benchmark["framework"],
                benchmark["model-prefix"],
                benchmark["precision"],
                fallback_model_path=fallback_model_path,
                path_is_dir=lambda _path: False,
            )
        except ValueError as error:
            unsupported.append(f"{config_name}: {error}")

    assert not unsupported, "\n".join(unsupported)
