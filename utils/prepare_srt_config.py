#!/usr/bin/env python3
"""Resolve one srt-slurm recipe and use InferenceX's benchmark client."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import re
from typing import Any

import yaml


SELECTOR_RE = re.compile(
    r"^(?P<path>.+\.ya?ml):(?P<selector>"
    r"base|override_\S+|zip_override_[\w-]+\[\d+\])$"
)
WORKLOAD_CONTAINER_ALIAS = "inferencex-workload"
NGINX_CONTAINER_ALIAS = "inferencex-nginx"
MODEL_PATH_ALIAS = "inferencex-model"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge an srt-slurm override into a base recipe."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if value is None:
            result.pop(key, None)
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def zip_slice(value: Any, index: int) -> Any:
    """Resolve one zip_override index, including broadcast length-one lists."""
    if isinstance(value, dict):
        return {key: zip_slice(item, index) for key, item in value.items()}
    if isinstance(value, list):
        if not value:
            raise ValueError("zip_override contains an empty list")
        return copy.deepcopy(value[0 if len(value) == 1 else index])
    return copy.deepcopy(value)


def split_config_arg(config_arg: str) -> tuple[Path, str | None]:
    """Split srtctl's path:selector syntax without confusing path colons."""
    match = SELECTOR_RE.fullmatch(config_arg)
    if match:
        return Path(match.group("path")), match.group("selector")
    return Path(config_arg), None


def resolve_recipe(config_arg: str) -> tuple[Path, dict[str, Any]]:
    """Load a plain recipe or resolve one explicit override variant."""
    path, selector = split_config_arg(config_arg)
    with path.open(encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: recipe must be a mapping")

    if "base" not in raw:
        if selector is not None:
            raise ValueError(f"{path}: selector requires a base/override recipe")
        return path, raw

    base = raw["base"]
    if not isinstance(base, dict):
        raise ValueError(f"{path}: base must be a mapping")
    if selector in (None, "base"):
        return path, copy.deepcopy(base)

    zip_match = re.fullmatch(r"(zip_override_[\w-]+)\[(\d+)\]", selector)
    if zip_match:
        key, index = zip_match.group(1), int(zip_match.group(2))
        if key not in raw:
            raise ValueError(f"{path}: unknown selector {key}")
        sliced = zip_slice(raw[key], index)
        result = deep_merge(base, sliced)
        if not isinstance(sliced, dict) or "name" not in sliced:
            result["name"] = (
                f"{base.get('name', 'unnamed')}_"
                f"{key.removeprefix('zip_override_')}_{index}"
            )
        return path, result

    if selector not in raw or not selector.startswith("override_"):
        raise ValueError(f"{path}: unknown selector {selector}")
    result = deep_merge(base, raw[selector])
    if "name" not in raw[selector]:
        result["name"] = (
            f"{base.get('name', 'unnamed')}_{selector.removeprefix('override_')}"
        )
    return path, result


def find_served_model_name(
    recipe: dict[str, Any],
    default_served_model_name: str | None = None,
) -> str:
    """Find the public model name used by the OpenAI-compatible endpoint."""
    backend = recipe.get("backend", {})
    model_path = str(recipe.get("model", {}).get("path") or "")

    def walk(value: Any) -> str | None:
        if not isinstance(value, dict):
            return None
        for key, item in value.items():
            if key.replace("-", "_") == "served_model_name" and item:
                return str(item)
        for item in value.values():
            found = walk(item)
            if found:
                return found
        return None

    return (
        walk(backend)
        or str(recipe.get("served_model_name") or "")
        or str(default_served_model_name or "")
        # Mirror srt-slurm's SrtConfig.served_model_name fallback exactly.
        # Servers receive the basename, not the full Hugging Face repository
        # ID or cluster model alias.
        or Path(model_path.removeprefix("hf:")).name
    )


def stringify_concurrencies(value: Any) -> str:
    if isinstance(value, list):
        return "x".join(str(item) for item in value)
    return str(value or "")


def use_inferencex_benchmark(
    recipe: dict[str, Any],
    *,
    isl: int | None = None,
    osl: int | None = None,
    concurrencies: str | None = None,
    random_range_ratio: float | None = None,
    default_served_model_name: str | None = None,
) -> dict[str, Any]:
    """Convert sa-bench metadata to srt-slurm's custom benchmark contract."""
    model = recipe.get("model")
    if model is None:
        model = {}
    elif not isinstance(model, dict):
        raise ValueError("recipe model must be a mapping")
    source_model_path = str(model.get("path") or "")
    served_name_fallback = (
        None if source_model_path.startswith("hf:") else default_served_model_name
    )
    source_model_name = (
        find_served_model_name(recipe, served_name_fallback) if model else ""
    )
    if model.get("container"):
        # The master config's IMAGE is authoritative. Recipes come from several
        # historical branches and use incompatible image aliases (including
        # nvcr.io/ vs nvcr.io# spellings), so normalize the workload container
        # to one alias that every cluster launcher maps to its imported image.
        model["container"] = WORKLOAD_CONTAINER_ALIAS
    if source_model_path and not source_model_path.startswith("hf:"):
        model["path"] = MODEL_PATH_ALIAS

    frontend = recipe.setdefault("frontend", {})
    if not isinstance(frontend, dict):
        raise ValueError("recipe frontend must be a mapping")
    if frontend.get("enable_multiple_frontends", True) or frontend.get(
        "nginx_container"
    ):
        frontend["nginx_container"] = NGINX_CONTAINER_ALIAS

    benchmark = recipe.get("benchmark")
    if not isinstance(benchmark, dict):
        raise ValueError("recipe does not define a benchmark mapping")

    # Normalize fields carried by older recipe snapshots but removed from the
    # pinned main schema. tokenizer_mode=deepseek_v4 is represented by the
    # in-tree DSV4 encoder below. warmup_req_rate has no distinct equivalent in
    # InferenceX's client (warmups are concurrency-limited), and
    # aiperf_server_metrics becomes the compatibility overlay's explicit
    # custom-runner opt-in marker.
    legacy_tokenizer_mode = str(benchmark.pop("tokenizer_mode", "") or "")
    benchmark.pop("warmup_req_rate", None)
    if benchmark.pop("aiperf_server_metrics", False):
        benchmark_env = benchmark.setdefault("env", {})
        if not isinstance(benchmark_env, dict):
            raise ValueError("benchmark.env must be a mapping")
        benchmark_env["INFERENCEX_AIPERF_SERVER_METRICS"] = "true"
    if benchmark.get("type") != "sa-bench":
        return recipe

    required_values = {
        "isl": isl,
        "osl": osl,
        "concurrencies": concurrencies,
    }
    for field, override in required_values.items():
        if benchmark.get(field) is None and override is None:
            raise ValueError(f"sa-bench recipe is missing benchmark.{field}")

    resources = recipe.get("resources", {})
    gpus_per_node = int(resources.get("gpus_per_node", 1))

    def worker_gpus(role: str) -> int:
        workers = int(resources.get(f"{role}_workers", 0) or 0)
        nodes = int(resources.get(f"{role}_nodes", 0) or 0)
        explicit = resources.get(f"gpus_per_{role}")
        if explicit is not None:
            per_worker = int(explicit)
        elif nodes and workers:
            per_worker = nodes * gpus_per_node // workers
        elif role == "decode" and workers:
            prefill_workers = int(resources.get("prefill_workers", 0) or 0)
            prefill_nodes = int(resources.get("prefill_nodes", 0) or 0)
            per_worker = (
                prefill_nodes * gpus_per_node // prefill_workers
                if prefill_workers
                else gpus_per_node
            )
        else:
            per_worker = gpus_per_node
        return workers * per_worker

    prefill_gpus = worker_gpus("prefill")
    decode_gpus = worker_gpus("decode")
    if prefill_gpus or decode_gpus:
        total_gpus = prefill_gpus + decode_gpus
    else:
        # Match srt-slurm's sa-bench accounting for aggregated deployments:
        # report all provisioned GPUs, even when an aggregate worker uses a
        # partial node through an explicit gpus_per_agg value.
        total_gpus = int(resources.get("agg_nodes", 1) or 1) * gpus_per_node

    model_name = source_model_name
    if not model_name:
        raise ValueError("could not resolve served model name")

    use_chat_template = bool(benchmark.get("use_chat_template", True))
    custom_tokenizer = str(benchmark.get("custom_tokenizer") or "")
    model_path = source_model_path
    tokenizer_path = model_path.removeprefix("hf:") if model_path.startswith("hf:") else "/model"
    needs_dsv4_tokenizer = (
        legacy_tokenizer_mode == "deepseek_v4"
        or "deepseek_v4" in custom_tokenizer.lower()
        or "deepseek-v4" in model_path.lower()
    )
    use_dsv4_chat_template = use_chat_template and needs_dsv4_tokenizer

    effective_isl = isl if isl is not None else benchmark["isl"]
    effective_osl = osl if osl is not None else benchmark["osl"]
    effective_concurrencies = (
        concurrencies
        if concurrencies is not None
        else stringify_concurrencies(benchmark["concurrencies"])
    )
    # The workflow-level value is a compatibility default for recipes that do
    # not declare a distribution.  An explicit recipe value is benchmark
    # intent and must win; several tuned recipes intentionally use 1.0 rather
    # than InferenceX's usual 0.8.
    effective_random_range_ratio = benchmark.get("random_range_ratio")
    if effective_random_range_ratio is None:
        effective_random_range_ratio = (
            random_range_ratio if random_range_ratio is not None else 0.8
        )

    benchmark["type"] = "custom"
    benchmark["command"] = (
        "bash /infmax-workspace/benchmarks/multi_node/srt_benchmark.sh"
    )
    benchmark["env"] = {
        "INFMAX_CONTAINER_WORKSPACE": "/infmax-workspace",
        "RESULT_ROOT": "/logs",
        "PORT": "8000",
        "TOKENIZER_PATH": tokenizer_path,
        "TOKENIZER_MODE": "deepseek_v4" if needs_dsv4_tokenizer else "auto",
        "ISL": str(effective_isl),
        "OSL": str(effective_osl),
        "CONC_LIST": str(effective_concurrencies),
        "REQUEST_RATE": str(benchmark.get("req_rate", "inf")),
        "RANDOM_RANGE_RATIO": str(effective_random_range_ratio),
        "NUM_PROMPTS_MULTIPLIER": str(benchmark.get("num_prompts_mult", 10)),
        "NUM_WARMUP_MULTIPLIER": str(benchmark.get("num_warmup_mult", 2)),
        "MODEL_NAME": model_name,
        "TOTAL_GPUS": str(total_gpus),
        "PREFILL_GPUS": str(prefill_gpus),
        "DECODE_GPUS": str(decode_gpus),
        "USE_CHAT_TEMPLATE": str(use_chat_template).lower(),
        "DSV4_CHAT_TEMPLATE": str(use_dsv4_chat_template).lower(),
    }
    return recipe


def prepare_config(
    config_arg: str,
    output: Path | None = None,
    *,
    isl: int | None = None,
    osl: int | None = None,
    concurrencies: str | None = None,
    random_range_ratio: float | None = None,
    default_served_model_name: str | None = None,
) -> Path:
    """Resolve and transform a config, returning the generated config path."""
    source_path, recipe = resolve_recipe(config_arg)
    transformed = use_inferencex_benchmark(
        recipe,
        isl=isl,
        osl=osl,
        concurrencies=concurrencies,
        random_range_ratio=random_range_ratio,
        default_served_model_name=default_served_model_name,
    )
    output_path = output or source_path.with_name(
        f"{source_path.stem}.inferencex.yaml"
    )
    with output_path.open("w", encoding="utf-8") as output_file:
        yaml.safe_dump(transformed, output_file, sort_keys=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Recipe path, optionally with :selector")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--isl", type=int)
    parser.add_argument("--osl", type=int)
    parser.add_argument("--concurrencies")
    parser.add_argument("--random-range-ratio", type=float)
    parser.add_argument("--default-served-model-name")
    args = parser.parse_args()
    print(
        prepare_config(
            args.config,
            args.output,
            isl=args.isl,
            osl=args.osl,
            concurrencies=args.concurrencies,
            random_range_ratio=args.random_range_ratio,
            default_served_model_name=args.default_served_model_name,
        )
    )


if __name__ == "__main__":
    main()
