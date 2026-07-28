#!/usr/bin/env python3
"""Hydrate InferenceX srt-slurm custom benchmark recipes."""

from __future__ import annotations

import argparse
import hashlib
import shlex
from pathlib import Path
from typing import Any

import yaml

from srtctl.core.config import (
    load_cluster_config,
    resolve_config_with_defaults,
    resolve_override_yaml,
)
from srtctl.core.schema import SrtConfig


CUSTOM_BENCHMARK_ENTRYPOINT = (
    "/infmax-workspace/benchmarks/multi_node/"
    "srt_bench_serving_entrypoint.sh"
)
CUSTOM_BENCHMARK_COMMAND = f"bash {CUSTOM_BENCHMARK_ENTRYPOINT}"


def split_config_spec(config_spec: str) -> tuple[Path, str | None]:
    """Split srtctl's ``path.yaml:selector`` syntax."""
    marker = ".yaml:"
    if marker not in config_spec:
        return Path(config_spec), None
    path, selector = config_spec.split(marker, 1)
    return Path(f"{path}.yaml"), selector


def to_plain_data(value: Any) -> Any:
    """Remove ruamel container and scalar types before PyYAML serialization."""
    if isinstance(value, dict):
        return {key: to_plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain_data(item) for item in value]
    if isinstance(value, str):
        return str(value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return value


def load_selected_recipes(
    config_spec: str,
) -> tuple[Path, list[tuple[str | None, dict[str, Any]]], bool]:
    """Load a plain recipe or resolve the requested override variants."""
    config_path, selector = split_config_spec(config_spec)
    with config_path.open(encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)

    if not isinstance(raw_config, dict):
        raise ValueError(f"{config_path} is not a YAML mapping")

    if "base" not in raw_config:
        if selector is not None:
            raise ValueError(
                f"{config_path} is not an override recipe, but selector "
                f"{selector!r} was provided"
            )
        return config_path, [(None, raw_config)], False

    if selector is None:
        variants = [
            (suffix, to_plain_data(recipe))
            for suffix, recipe in resolve_override_yaml(config_path)
        ]
        return config_path, variants, True

    variants = resolve_override_yaml(config_path, selector)
    if not variants:
        raise ValueError(
            f"{config_spec} did not resolve to any variants"
        )
    return (
        config_path,
        [(None, to_plain_data(recipe)) for _, recipe in variants],
        len(variants) > 1,
    )


def stringify_concurrencies(concurrencies: list[int] | str | None) -> str:
    """Return the entrypoint's whitespace-separated concurrency format."""
    if isinstance(concurrencies, list):
        return " ".join(str(value) for value in concurrencies)
    if concurrencies is None:
        return ""
    return str(concurrencies).replace("x", " ")


def custom_benchmark_arguments(config: SrtConfig) -> list[str]:
    """Translate recipe settings to entrypoint arguments."""
    benchmark = config.benchmark
    resources = config.resources

    if benchmark.dataset_name not in (None, "random"):
        raise ValueError(
            "InferenceX benchmark_serving currently supports only the random "
            f"dataset, not {benchmark.dataset_name!r}"
        )
    if benchmark.reuse_http_connections:
        raise ValueError(
            "InferenceX benchmark_serving does not support "
            "benchmark.reuse_http_connections"
        )
    if (
        benchmark.slow_down_sleep_time is not None
        or benchmark.slow_down_wait_time is not None
    ):
        raise ValueError(
            "InferenceX benchmark_serving does not support SA-Bench slow_down"
        )

    concurrencies = stringify_concurrencies(benchmark.concurrencies)
    if not concurrencies:
        raise ValueError("benchmark.concurrencies is required")
    if benchmark.isl is None or benchmark.osl is None:
        raise ValueError("benchmark.isl and benchmark.osl are required")

    if resources.is_disaggregated:
        prefill_gpus = resources.prefill_gpus
        decode_gpus = resources.decode_gpus
        total_gpus = prefill_gpus + decode_gpus
    else:
        prefill_gpus = 0
        decode_gpus = 0
        total_gpus = (resources.agg_nodes or 1) * resources.gpus_per_node

    model_path = config.model.path
    tokenizer = (
        model_path.removeprefix("hf:")
        if model_path.startswith("hf:")
        else "/model"
    )
    custom_tokenizer = benchmark.custom_tokenizer or ""
    is_deepseek_v4 = "deepseek_v4" in custom_tokenizer.lower()

    return [
        config.served_model_name,
        tokenizer,
        str(benchmark.isl),
        str(benchmark.osl),
        concurrencies,
        str(benchmark.req_rate or "inf"),
        str(resources.is_disaggregated).lower(),
        str(prefill_gpus),
        str(decode_gpus),
        str(total_gpus),
        str(benchmark.random_range_ratio or 0.8),
        str(benchmark.num_prompts_mult or 10),
        str(
            benchmark.num_warmup_mult
            if benchmark.num_warmup_mult is not None
            else 2
        ),
        str(benchmark.use_chat_template).lower(),
        str(
            is_deepseek_v4 and benchmark.use_chat_template
        ).lower(),
    ]


def hydrate_recipe(recipe: dict[str, Any]) -> bool:
    """Hydrate an InferenceX throughput recipe; return whether it changed."""
    benchmark = recipe.get("benchmark")
    if not isinstance(benchmark, dict):
        return False

    benchmark_type = benchmark.get("type")
    if benchmark_type == "sa-bench":
        benchmark["type"] = "custom"
        benchmark["command"] = CUSTOM_BENCHMARK_COMMAND
    elif not (
        benchmark_type == "custom"
        and benchmark.get("command") == CUSTOM_BENCHMARK_COMMAND
    ):
        return False

    cluster_config = load_cluster_config()
    resolved = resolve_config_with_defaults(recipe, cluster_config)
    config = SrtConfig.Schema().load(resolved)
    if not isinstance(config, SrtConfig):
        raise TypeError("srt-slurm did not return a typed SrtConfig")

    benchmark["command"] = shlex.join(
        [
            "bash",
            CUSTOM_BENCHMARK_ENTRYPOINT,
            *custom_benchmark_arguments(config),
        ]
    )
    benchmark.pop("env", None)
    return True


def render_config(config_spec: str, output_dir: Path) -> str:
    """Render selected recipes and return their path for ``srtctl apply``."""
    config_path, variants, keep_override_format = load_selected_recipes(
        config_spec
    )
    changed = False
    for _, recipe in variants:
        changed = hydrate_recipe(recipe) or changed
    if not changed:
        return config_spec

    if keep_override_format:
        rendered: dict[str, Any] = {"base": {}}
        for index, (suffix, recipe) in enumerate(variants):
            variant_name = suffix or f"inferencex_{index}"
            rendered[f"override_{variant_name}"] = recipe
    else:
        if len(variants) != 1:
            raise ValueError(
                f"{config_spec} resolved to {len(variants)} variants but "
                "cannot preserve their selector"
            )
        rendered = variants[0][1]

    digest = hashlib.sha256(config_spec.encode()).hexdigest()[:12]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{config_path.stem}-{digest}.yaml"
    with output_path.open("w", encoding="utf-8") as output_file:
        yaml.safe_dump(rendered, output_file, sort_keys=False)
    return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_spec")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".inferencex-recipes"),
    )
    args = parser.parse_args()
    print(render_config(args.config_spec, args.output_dir))


if __name__ == "__main__":
    main()
