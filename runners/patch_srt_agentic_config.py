#!/usr/bin/env python3
"""Patch an srt-slurm recipe to run the InferenceX agentic replay as a custom benchmark."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


AGENTIC_ENV_VARS = (
    "RESULT_FILENAME",
    "RUNNER_TYPE",
    "IMAGE",
    "MODEL",
    "MODEL_PREFIX",
    "FRAMEWORK",
    "PRECISION",
    "SPEC_DECODING",
    "DISAGG",
    "USERS",
    "DURATION",
    "MAX_DELAY",
    "ADVANCE_MIN",
    "ADVANCE_MAX",
    "IGNORE_EOS",
    "HASH_BLOCK_MODE",
    "DEBUG_TRACE",
    "PREFILL_NUM_WORKERS",
    "PREFILL_TP",
    "PREFILL_EP",
    "PREFILL_DP_ATTN",
    "DECODE_NUM_WORKERS",
    "DECODE_TP",
    "DECODE_EP",
    "DECODE_DP_ATTN",
    "HF_TOKEN",
    "HF_HUB_CACHE",
)


def _custom_benchmark() -> dict[str, Any]:
    # TODO: If the srt recipe YAMLs move upstream into this repo, define the
    # custom benchmark block in those recipes directly and keep only whatever
    # dynamic env injection is still needed for per-run workflow values.
    env = {
        name: os.environ[name]
        for name in AGENTIC_ENV_VARS
        if os.environ.get(name) not in (None, "")
    }
    env.update({
        "INFMAX_CONTAINER_WORKSPACE": "/infmax-workspace",
        "RESULT_DIR": "/logs/agentic",
        "PORT": "8000",
        "IS_MULTINODE": "true",
    })
    return {
        "type": "custom",
        "command": "bash /infmax-workspace/benchmarks/multi_node/agentic_srt.sh",
        "env": env,
    }


def _patch_section(section: Any) -> None:
    if isinstance(section, dict):
        section["benchmark"] = _custom_benchmark()


def main() -> int:
    if os.environ.get("SCENARIO_TYPE") != "agentic-coding":
        return 0

    import yaml

    config_file = os.environ.get("CONFIG_FILE")
    if not config_file:
        print("ERROR: CONFIG_FILE must be set for multinode agentic srt-slurm runs", file=sys.stderr)
        return 1

    config_path = Path(config_file.split(":", 1)[0])
    if not config_path.exists():
        print(f"ERROR: CONFIG_FILE path not found: {config_path}", file=sys.stderr)
        return 1

    with config_path.open() as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        print(f"ERROR: {config_path} is not a YAML mapping", file=sys.stderr)
        return 1

    if isinstance(data.get("base"), dict):
        _patch_section(data["base"])
        for key, value in data.items():
            if key.startswith(("override_", "zip_override_")):
                _patch_section(value)
    else:
        _patch_section(data)

    with config_path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print(f"Patched {config_path} for agentic custom benchmark")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
