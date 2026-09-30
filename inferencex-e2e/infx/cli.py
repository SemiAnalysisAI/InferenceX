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
        "generate", help="Generate native SRT recipes without submitting jobs"
    )
    generate.add_argument(
        "--config-key", action="append", required=True, help="Master key or wildcard; repeatable"
    )
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--project-root", type=Path, default=repository_root())
    generate.add_argument(
        "--config-file", type=Path, action="append", help="Master YAML; repeatable"
    )
    generate.add_argument("--runner-config", type=Path)
    generate.add_argument(
        "--refresh-exports",
        action="store_true",
        help="Refresh registered native recipe bundles after validating all outputs",
    )
    generate.add_argument("--gpu-monitor-interval", type=int, default=1)
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    try:
        manifest = generate_recipes(
            config_keys=args.config_key,
            config_files=args.config_file or [root / name for name in MASTER_CONFIGS],
            runner_file=args.runner_config or root / RUNNER_CONFIG,
            project=root,
            output=args.output_dir,
            gpu_monitor_interval=args.gpu_monitor_interval,
            refresh_exports=args.refresh_exports,
        )
    except ImportError as error:
        parser.error(
            f"Recipe generation dependency is unavailable ({error}); run uv sync --extra recipes"
        )
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        parser.error(str(error))
    print(f"Generated {len(manifest['recipes'])} recipes in {args.output_dir.resolve()}")
    return 0
