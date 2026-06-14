#!/usr/bin/env python3
"""Emit a clean external-MPI rank environment for the host launcher."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from trt_config import (
    CONTROLLED_ENVIRONMENT_VARIABLES,
    external_mpi_rank_environment,
    hardware_profile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--fixed-batch-arm-file", type=Path, required=True)
    parser.add_argument("--marker-file", type=Path, required=True)
    parser.add_argument("--cute-cache-dir", type=Path, required=True)
    parser.add_argument("--hardware-profile", required=True)
    parser.add_argument(
        "--format",
        choices=("json", "nul"),
        default="json",
    )
    return parser.parse_args()


def write_nul_record(*fields: str) -> None:
    for field in fields:
        encoded = field.encode("utf-8")
        if b"\0" in encoded:
            raise ValueError("Environment records cannot contain NUL bytes")
        sys.stdout.buffer.write(encoded + b"\0")


def main() -> int:
    args = parse_args()
    profile = hardware_profile(args.hardware_profile)
    environment = external_mpi_rank_environment(
        args.global_batch_size,
        args.fixed_batch_arm_file,
        args.marker_file,
        args.cute_cache_dir,
        profile,
    )
    if args.format == "json":
        json.dump(
            {
                "unset": sorted(CONTROLLED_ENVIRONMENT_VARIABLES),
                "export": environment,
            },
            sys.stdout,
            indent=2,
            sort_keys=True,
        )
        sys.stdout.write("\n")
        return 0

    for name in sorted(CONTROLLED_ENVIRONMENT_VARIABLES):
        write_nul_record("unset", name)
    for name, value in sorted(environment.items()):
        write_nul_record("export", name, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
