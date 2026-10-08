"""InferenceX command-line tools."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from infx.config import MASTER_CONFIGS, RUNNER_CONFIG, repository_root
from infx.srt_slurm.generate import generate_recipes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="infx")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser(
        "generate", help="Write the bound srt-slurm recipes of master-config points"
    )
    generate.add_argument(
        "--config-key", action="append", required=True, help="master-config key or glob; repeatable"
    )
    generate.add_argument(
        "--output-dir", type=Path, required=True, help="new or empty directory to write"
    )
    generate.add_argument(
        "--config-file", type=Path, action="append", help="master config; repeatable"
    )
    generate.add_argument("--runner-config", type=Path)
    args = parser.parse_args(argv)
    root = repository_root()
    try:
        manifest = generate_recipes(
            config_keys=args.config_key,
            config_files=args.config_file or [root / name for name in MASTER_CONFIGS],
            runner_file=args.runner_config or root / RUNNER_CONFIG,
            output=args.output_dir,
        )
    except ImportError as error:
        parser.error(f"{error}; run it with uv run --extra recipes")
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        parser.error(str(error))
    print(f"Wrote {len(manifest['recipes'])} recipes and manifest.json to {args.output_dir}")
    return 0
