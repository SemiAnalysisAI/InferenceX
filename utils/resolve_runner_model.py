#!/usr/bin/env python3
"""Resolve cluster-local model paths from configs/runners.yaml."""

import argparse
import os
import shlex
import sys
from typing import Callable, Optional

import yaml


def load_runner_config(runner_config_path: str) -> dict:
    """Load runner configuration for runtime model resolution."""
    try:
        with open(runner_config_path) as runner_config_file:
            runner_config = yaml.safe_load(runner_config_file)
    except FileNotFoundError as error:
        raise ValueError(
            f"Runner config file '{runner_config_path}' does not exist."
        ) from error
    if not isinstance(runner_config, dict):
        raise ValueError("Runner config must contain a mapping")
    return runner_config


def resolve_runner_model(
    runner_config: dict,
    runner_label: str,
    framework: str,
    model_prefix: str,
    precision: str,
    current_model_path: Optional[str] = None,
    fallback_model_path: Optional[str] = None,
    path_is_dir: Callable[[str], bool] = os.path.isdir,
) -> dict[str, str]:
    """Return model environment values for a runner/framework/model tuple."""
    models = runner_config.get("models", {})
    runner_models = models.get(runner_label, {})
    model_config = None
    for framework_key in (framework, "default"):
        framework_models = runner_models.get(framework_key, {})
        precision_models = framework_models.get(model_prefix, {})
        model_config = (
            precision_models.get(precision)
            or precision_models.get("default")
        )
        if model_config is not None:
            break

    if model_config is None and fallback_model_path is not None:
        return {"MODEL_PATH": fallback_model_path}
    if model_config is None:
        raise ValueError(
            "Unsupported runner/framework/model/precision combination: "
            f"{runner_label}/{framework}/{model_prefix}/{precision}"
        )

    configured_paths = model_config["model-paths"]
    if (
        model_config.get("allow-model-path-override", False)
        and current_model_path
        and path_is_dir(current_model_path)
    ):
        model_path = current_model_path
    else:
        model_path = next(
            (path for path in configured_paths if path_is_dir(path)),
            configured_paths[0],
        )

    environment = {"MODEL_PATH": model_path}
    optional_exports = {
        "SRT_SLURM_MODEL_PREFIX": "srt-slurm-model-prefix",
        "SERVED_MODEL_NAME": "served-model-name",
        "MODEL_NAME": "model-name",
    }
    for environment_name, config_name in optional_exports.items():
        value = model_config.get(config_name)
        if value is not None:
            environment[environment_name] = value
    return environment


def shell_exports(environment: dict[str, str]) -> str:
    """Render resolved values as shell-safe export statements."""
    return "\n".join(
        f"export {name}={shlex.quote(value)}"
        for name, value in environment.items()
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runner-config",
        default="configs/runners.yaml",
        help="Path to runners.yaml",
    )
    parser.add_argument("--runner-label", required=True)
    parser.add_argument("--framework", required=True)
    parser.add_argument("--model-prefix", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument(
        "--current-model-path",
        default=os.environ.get("MODEL_PATH"),
        help="Existing MODEL_PATH override, when the mapping allows it",
    )
    parser.add_argument(
        "--fallback-model-path",
        help="MODEL_PATH to export when no configured mapping exists",
    )
    return parser.parse_args()


def main() -> int:
    """Resolve and print shell exports for a runner model mapping."""
    args = parse_args()
    try:
        runner_config = load_runner_config(args.runner_config)
        environment = resolve_runner_model(
            runner_config,
            args.runner_label,
            args.framework,
            args.model_prefix,
            args.precision,
            args.current_model_path,
            args.fallback_model_path,
        )
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(shell_exports(environment))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
