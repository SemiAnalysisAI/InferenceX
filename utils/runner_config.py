#!/usr/bin/env python3
"""Resolve trusted runtime settings from configs/runners.yaml."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shlex
import sys
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a mapping")
    return config


def shell_export(name: str, value: str) -> str:
    return f"export {name}={shlex.quote(value)}"


def normalize_cluster(name: str) -> str:
    return name if name.startswith("cluster:") else f"cluster:{name}"


def resolve_model(
    config: dict[str, Any],
    cluster: str,
    model_prefix: str,
    precision: str,
    framework: str,
    model: str,
) -> dict[str, str]:
    cluster_key = normalize_cluster(cluster)
    cluster_config = config.get("clusters", {}).get(cluster_key)
    if not isinstance(cluster_config, dict):
        raise ValueError(f"runner config has no runtime entry for {cluster_key}")

    matches: list[tuple[int, dict[str, Any]]] = []
    for entry in cluster_config.get("models", []):
        selectors = (
            (entry.get("model-prefix"), model_prefix),
            (entry.get("precision"), precision),
            (entry.get("framework"), framework),
        )
        if all(expected in (None, "*", actual) for expected, actual in selectors):
            score = sum(expected not in (None, "*") for expected, _ in selectors)
            matches.append((score, entry))
    if not matches:
        raise ValueError(
            f"no model mapping for {cluster_key} "
            f"{model_prefix}/{precision}/{framework}"
        )

    best_score = max(score for score, _ in matches)
    best = [entry for score, entry in matches if score == best_score]
    if len(best) != 1:
        raise ValueError(
            f"ambiguous model mapping for {cluster_key} "
            f"{model_prefix}/{precision}/{framework}"
        )
    entry = best[0]
    format_values = {
        "model": model,
        "model_prefix": model_prefix,
        "precision": precision,
        "framework": framework,
    }
    values = {
        "MODEL_PATH": str(entry["path"]).format_map(format_values),
        "MODEL_PATH_LAYOUT": str(entry.get("layout", "direct")),
        "SRT_SLURM_MODEL_PREFIX": str(
            entry.get("srt-model-prefix", model_prefix)
        ).format_map(format_values),
        "SERVED_MODEL_NAME": str(
            entry.get("served-model-name", model)
        ).format_map(format_values),
    }
    return values


def resolve_paths(
    config: dict[str, Any],
    cluster: str,
    workspace: str,
) -> dict[str, str]:
    """Resolve named cluster paths to stable shell variable names."""
    cluster_key = normalize_cluster(cluster)
    cluster_config = config.get("clusters", {}).get(cluster_key)
    if not isinstance(cluster_config, dict):
        raise ValueError(f"runner config has no runtime entry for {cluster_key}")

    values: dict[str, str] = {}
    for key, raw_value in cluster_config.get("paths", {}).items():
        if not re.fullmatch(r"[a-z][a-z0-9-]*", key):
            raise ValueError(f"invalid path key for {cluster_key}: {key!r}")
        name = f"RUNNER_PATH_{key.replace('-', '_').upper()}"
        values[name] = str(raw_value).format(workspace=workspace)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/runners.yaml"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    srt_parser = subparsers.add_parser("srt")
    srt_parser.add_argument("--shell", action="store_true")

    model_parser = subparsers.add_parser("model")
    model_parser.add_argument("--cluster", required=True)
    model_parser.add_argument("--model-prefix", required=True)
    model_parser.add_argument("--precision", required=True)
    model_parser.add_argument("--framework", required=True)
    model_parser.add_argument("--model", required=True)
    model_parser.add_argument("--shell", action="store_true")

    paths_parser = subparsers.add_parser("paths")
    paths_parser.add_argument("--cluster", required=True)
    paths_parser.add_argument("--workspace", required=True)
    paths_parser.add_argument("--shell", action="store_true")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "srt":
        srt = config.get("srt-slurm")
        if not isinstance(srt, dict):
            raise ValueError("runner config does not define srt-slurm")
        values = {
            "SRT_SLURM_REPOSITORY": str(srt["repository"]),
            "SRT_SLURM_BRANCH": str(srt["branch"]),
            "SRT_SLURM_REVISION": str(srt["revision"]),
            "SRT_SLURM_LEGACY_RECIPES_REVISION": str(
                srt["legacy-recipes-revision"]
            ),
        }
    elif args.command == "model":
        values = resolve_model(
            config,
            args.cluster,
            args.model_prefix,
            args.precision,
            args.framework,
            args.model,
        )
    else:
        values = resolve_paths(
            config,
            args.cluster,
            args.workspace,
        )

    if not args.shell:
        raise ValueError("only --shell output is currently supported")
    for name, value in values.items():
        print(shell_export(name, value))


if __name__ == "__main__":
    try:
        main()
    except (KeyError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
