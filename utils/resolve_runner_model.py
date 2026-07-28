#!/usr/bin/env python3
"""Resolve cluster-local model paths from configs/runners.yaml."""

import argparse
import os
from pathlib import Path
import shlex
import sys
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent / "matrix_logic"))

from validation import load_runner_file


def resolve_runner_model(
    runner_config: dict,
    runner_label: str,
    model_prefix: str,
    precision: str,
    current_model_path: Optional[str] = None,
    path_is_dir: Callable[[str], bool] = os.path.isdir,
) -> tuple[str, str]:
    """Return the model path and srt-slurm alias for a runner/model pair."""
    models = runner_config.get("models", {})
    try:
        model_config = models[runner_label][model_prefix][precision]
    except KeyError as error:
        raise ValueError(
            "Unsupported runner/model/precision combination: "
            f"{runner_label}/{model_prefix}/{precision}"
        ) from error

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

    return model_path, model_config["srt-slurm-model-prefix"]


def shell_exports(model_path: str, srt_slurm_model_prefix: str) -> str:
    """Render resolved values as shell-safe export statements."""
    return "\n".join(
        [
            f"export MODEL_PATH={shlex.quote(model_path)}",
            "export SRT_SLURM_MODEL_PREFIX="
            f"{shlex.quote(srt_slurm_model_prefix)}",
        ]
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
    parser.add_argument("--model-prefix", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument(
        "--current-model-path",
        default=os.environ.get("MODEL_PATH"),
        help="Existing MODEL_PATH override, when the mapping allows it",
    )
    return parser.parse_args()


def main() -> int:
    """Resolve and print shell exports for a runner model mapping."""
    args = parse_args()
    try:
        runner_config = load_runner_file(args.runner_config)
        model_path, srt_slurm_model_prefix = resolve_runner_model(
            runner_config,
            args.runner_label,
            args.model_prefix,
            args.precision,
            args.current_model_path,
        )
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(shell_exports(model_path, srt_slurm_model_prefix))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
