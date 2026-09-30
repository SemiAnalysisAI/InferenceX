"""Resolve recipe images with the job-local srtctl, before importing any containers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from infx.srt_slurm.synthetic_acceptance import selected_recipes


def resolve_images(config: str, expected_worker: str) -> list[str]:
    """Use native override expansion to validate and deduplicate one recipe's images."""
    path, _, selector = config.partition(":")
    try:
        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError("recipe must be a mapping")
        selected = selected_recipes(raw, selector or None)
        if len(selected) != 1:
            raise ValueError("image provisioning requires exactly one selected recipe")
        recipe = selected[0][1]
        worker = recipe["model"]["container"]
        if worker != expected_worker:
            raise ValueError("recipe model.container must match the matrix IMAGE")
        frontend_config = recipe.get("frontend", {})
        if not isinstance(frontend_config, dict):
            raise TypeError("frontend must be a mapping")
        frontend = frontend_config.get("container_image")
        images = [worker, *([frontend] if frontend is not None else [])]
        if any(
            not isinstance(image, str) or not image or any(c.isspace() for c in image)
            for image in images
        ):
            raise ValueError("container identities must be non-empty strings without whitespace")
        return list(dict.fromkeys(images))
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        raise ValueError(f"invalid recipe images for {config}: {error}") from error


def main() -> int:
    """Print a JSON image list; failed resolution must stop the launcher before import."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("expected_worker")
    args = parser.parse_args()
    try:
        images = resolve_images(args.config, args.expected_worker)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    print(json.dumps(images))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
