"""Tests for cluster-local model path resolution."""

import pytest

from utils.resolve_runner_model import resolve_runner_model, shell_exports


def runner_config(model_paths, allow_override=False):
    """Build a minimal runner model configuration."""
    return {
        "models": {
            "cluster:b200-dgxc": {
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


def test_resolve_uses_first_existing_configured_path():
    """The first existing configured path wins."""
    config = runner_config(["/models/preferred", "/models/fallback"])

    result = resolve_runner_model(
        config,
        "cluster:b200-dgxc",
        "dsv4",
        "fp4",
        path_is_dir=lambda path: path == "/models/fallback",
    )

    assert result == ("/models/fallback", "deepseek-v4-pro")


def test_resolve_falls_back_to_first_path_when_none_exist():
    """The first path remains the deterministic fallback."""
    config = runner_config(["/models/preferred", "/models/fallback"])

    result = resolve_runner_model(
        config,
        "cluster:b200-dgxc",
        "dsv4",
        "fp4",
        path_is_dir=lambda _path: False,
    )

    assert result == ("/models/preferred", "deepseek-v4-pro")


def test_resolve_honors_allowed_existing_override():
    """An allowed, existing MODEL_PATH override takes precedence."""
    config = runner_config(["/models/preferred"], allow_override=True)

    result = resolve_runner_model(
        config,
        "cluster:b200-dgxc",
        "dsv4",
        "fp4",
        current_model_path="/models/operator-override",
        path_is_dir=lambda path: path == "/models/operator-override",
    )

    assert result == ("/models/operator-override", "deepseek-v4-pro")


def test_resolve_rejects_unsupported_combination():
    """Missing model combinations fail with the full lookup key."""
    config = runner_config(["/models/preferred"])

    with pytest.raises(ValueError) as exc_info:
        resolve_runner_model(
            config,
            "cluster:b200-dgxc",
            "dsr1",
            "fp8",
            path_is_dir=lambda _path: False,
        )

    assert "cluster:b200-dgxc/dsr1/fp8" in str(exc_info.value)


def test_shell_exports_quote_values():
    """Shell output quotes values instead of interpolating them."""
    exports = shell_exports("/models/path with spaces", "model;alias")

    assert exports == (
        "export MODEL_PATH='/models/path with spaces'\n"
        "export SRT_SLURM_MODEL_PREFIX='model;alias'"
    )
