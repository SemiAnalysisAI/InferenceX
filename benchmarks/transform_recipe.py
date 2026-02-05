#!/usr/bin/env python3
"""
Transform srt-slurm recipes to use container aliases.

This script transforms SGLang recipes to use container/model path aliases that
match the srtslurm.yaml mappings. Once upstream recipes are updated to use
consistent aliases, this script can be removed.

Usage:
    python transform_recipe.py <input.yaml> <output.yaml>
    python transform_recipe.py <input.yaml>  # Prints to stdout
"""

import sys
import yaml

# Full container names -> alias mappings
CONTAINER_ALIASES = {
    "lmsysorg/sglang:v0.5.8-cu130": "dynamo-sglang",
    "lmsysorg/sglang:v0.5.8-cu130-runtime": "dynamo-sglang",
}

# Model path aliases -> MODEL_PREFIX mappings
# srtslurm.yaml uses "${MODEL_PREFIX}": "${MODEL_PATH}" so we translate to dsr1-fp4
MODEL_PATH_ALIASES = {
    "dsfp4": "dsr1-fp4",
}

# Frontend container aliases
FRONTEND_CONTAINER_ALIASES = {
    "nginx": "nginx-sqsh",
}


def transform_recipe(config: dict) -> dict:
    """Transform recipe to use container/model path aliases."""
    # Transform model.container
    if "model" in config and "container" in config["model"]:
        container = config["model"]["container"]
        if container in CONTAINER_ALIASES:
            config["model"]["container"] = CONTAINER_ALIASES[container]

    # Transform model.path (e.g., dsfp4 -> dsr1)
    if "model" in config and "path" in config["model"]:
        path = config["model"]["path"]
        if path in MODEL_PATH_ALIASES:
            config["model"]["path"] = MODEL_PATH_ALIASES[path]

    # Transform frontend.nginx_container (e.g., nginx -> nginx-sqsh)
    if "frontend" in config and "nginx_container" in config["frontend"]:
        nginx_container = config["frontend"]["nginx_container"]
        if nginx_container in FRONTEND_CONTAINER_ALIASES:
            config["frontend"]["nginx_container"] = FRONTEND_CONTAINER_ALIASES[nginx_container]

    return config


def main():
    if len(sys.argv) < 2:
        print("Usage: transform_recipe.py <input.yaml> [output.yaml]", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]

    with open(input_path) as f:
        config = yaml.safe_load(f)

    transformed = transform_recipe(config)

    if len(sys.argv) > 2:
        output_path = sys.argv[2]
        with open(output_path, "w") as f:
            yaml.dump(transformed, f, default_flow_style=False, sort_keys=False)
    else:
        yaml.dump(transformed, sys.stdout, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    main()
