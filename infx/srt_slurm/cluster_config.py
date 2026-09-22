"""Render a runner's srt-slurm cluster profile with explicit job-local inputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from string import Template
from typing import Any

import yaml


def render_cluster_config(
    profile: dict[str, Any],
    variables: dict[str, str],
    overrides: dict[str, list[list[str]]],
) -> dict[str, Any]:
    """Substitute YAML scalars, then add caller-owned model/container/mount mappings."""

    def expand(value: Any) -> Any:
        if isinstance(value, str):
            return Template(value).substitute(variables)
        if isinstance(value, dict):
            return {expand(key): expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        return value

    config = expand(profile)
    for section, pairs in overrides.items():
        if pairs:
            config.setdefault(section, {}).update(pairs)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--var", nargs=2, action="append", default=[], metavar=("NAME", "VALUE"))
    parser.add_argument("--model", nargs=2, action="append", default=[], metavar=("ALIAS", "PATH"))
    parser.add_argument(
        "--container", nargs=2, action="append", default=[], metavar=("ALIAS", "PATH")
    )
    parser.add_argument(
        "--mount", nargs=2, action="append", default=[], metavar=("HOST", "CONTAINER")
    )
    args = parser.parse_args()
    try:
        profile = yaml.safe_load(args.profile.read_text())
        if not isinstance(profile, dict):
            raise ValueError("Cluster profile must be a YAML mapping")
        config = render_cluster_config(
            profile,
            dict(args.var),
            {"model_paths": args.model, "containers": args.container, "default_mounts": args.mount},
        )
        args.output.write_text(yaml.safe_dump(config, sort_keys=False))
    except (OSError, ValueError, KeyError, yaml.YAMLError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
