#!/usr/bin/env python3
"""Resolve one selected 8K/1K recipe and enable audited GPU telemetry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runners.inject_srt_power_concurrencies import _validate_concurrencies


def prepare_recipe(selector: str, concurrencies: list[int]) -> Path:
    # Use the installed, pinned runtime's selector semantics, including zipped
    # overrides. Mutating only base would let an override silently undo power.
    from srtctl.core.config import generate_override_configs
    from srtctl.core.schema import SrtConfig

    path_text, _, variant = selector.partition(":")
    source = Path(path_text)
    raw = yaml.safe_load(source.read_text())
    if not isinstance(raw, dict):
        raise ValueError("recipe must be a mapping")
    variants = generate_override_configs(raw, selector=variant or None) if "base" in raw else [("", raw)]
    if variant and "base" not in raw:
        raise ValueError("selector supplied for a recipe without overrides")
    if len(variants) != 1:
        raise ValueError("a matrix point must select exactly one recipe variant")
    recipe = variants[0][1]
    benchmark = recipe.get("benchmark", {})
    if (benchmark.get("type"), benchmark.get("isl"), benchmark.get("osl")) != ("sa-bench", 8192, 1024):
        raise ValueError("PowerX fixed-sequence policy requires an SA-Bench 8192/1024 recipe")
    benchmark["concurrencies"] = _validate_concurrencies(concurrencies)
    if benchmark.get("tokenizer_mode") == "deepseek_v4":
        benchmark.pop("tokenizer_mode")
        benchmark["custom_tokenizer"] = "sa_bench_tokenizers.vllm_deepseek_v4.VLLMDeepseekV4Tokenizer"
    recipe["telemetry"] = {
        "enabled": True,
        "provider": "dcgm-power",
        "default_frequency": 1.0,
        "storage_subdir": "power",
        "required": True,
        "startup_timeout_seconds": 120,
        "request_timeout_seconds": 2,
        "collector_join_timeout_seconds": 12,
        "dcgm_exporter": {"container_image": "dcgm-exporter", "port": 9401},
    }
    SrtConfig.Schema().load(recipe)
    target = source.with_name(f"{source.stem}.powerx.yaml")
    target.write_text(yaml.safe_dump(recipe, sort_keys=False))
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe")
    parser.add_argument("concurrency", type=int, nargs="+")
    args = parser.parse_args()
    print(prepare_recipe(args.recipe, args.concurrency))


if __name__ == "__main__":
    main()
