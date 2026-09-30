"""Generate standalone native SRT recipes from master-config matrix points."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from infx.matrix.generate import expand_config_keys, generate_config_matrix
from infx.matrix.validation import load_config_files, load_runner_file
from infx.srt_slurm.single_node import select_recipe
from infx.srt_slurm.synthetic_acceptance import build_overrides, selected_recipes
from infx.srt_slurm.workload import bind_workload


@contextmanager
def _project(root: Path) -> Iterator[None]:
    previous = os.environ.get("INFERENCEX_REPOSITORY_ROOT")
    os.environ["INFERENCEX_REPOSITORY_ROOT"] = str(root)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("INFERENCEX_REPOSITORY_ROOT", None)
        else:
            os.environ["INFERENCEX_REPOSITORY_ROOT"] = previous


def _upstream(project: Path) -> str:
    """Import validation from the checkout's pinned dependency, never an ambient installation."""
    checkout = project / "utils/srt-slurm"
    source = checkout / "src"
    if not (source / "srtctl/core/schema.py").is_file():
        raise ValueError(
            "Initialize the pinned SRT dependency: git submodule update --init inferencex-e2e/utils/srt-slurm"
        )
    source = source.resolve()
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    try:
        import srtctl
    except ImportError as error:
        raise ValueError(
            "Recipe generation dependencies are missing; run uv sync --extra recipes"
        ) from error
    if not Path(srtctl.__file__).resolve().is_relative_to(source):
        raise ValueError(
            "A different srtctl version is already imported; run infx generate in a fresh process"
        )
    return subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()


def _settings(point: Mapping[str, Any]) -> dict[str, str]:
    values: dict[str, str] = {}
    for role in ("prefill", "decode"):
        for setting in point.get(role, {}).get("additional-settings", []) or []:
            key, separator, value = setting.partition("=")
            if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError(f"Invalid additional-setting: {setting!r}")
            if key in values and values[key] != value:
                raise ValueError(f"Conflicting additional-setting: {key}")
            values[key] = value
    return values


def _environment(point: dict[str, Any], *, interval: int) -> dict[str, str]:
    multi = "prefill" in point
    environment = {
        "IMAGE": point["image"],
        "MODEL": point["model"],
        "MODEL_PREFIX": point["model-prefix"],
        "PRECISION": point["precision"],
        "FRAMEWORK": point["framework"],
        "SPEC_DECODING": point["spec-decoding"],
        "IS_AGENTIC": "0",
        "IS_MULTINODE": str(multi).lower(),
        "ISL": str(point["isl"]),
        "OSL": str(point["osl"]),
        "RUN_EVAL": "false",
        "EVAL_ONLY": "false",
        "RANDOM_RANGE_RATIO": "0.8",
        "GPU_MONITOR_INTERVAL": str(interval),
        "RESULT_DIR": "/logs",
        "MAX_MODEL_LEN": str(point["max-model-len"]),
    }
    if multi:
        environment["CONC_LIST"] = " ".join(map(str, point["conc"]))
        for role in ("prefill", "decode"):
            worker = point[role]
            environment[f"{role.upper()}_NUM_WORKERS"] = str(worker["num-worker"])
            environment[f"{role.upper()}_TP"] = str(worker["tp"])
    else:
        environment.update(
            {
                "CONC": str(point["conc"]),
                "TP": str(point["tp"]),
                "EP_SIZE": str(point["ep"]),
                "PP_SIZE": str(point["pp"]),
                "DCP_SIZE": str(point["dcp-size"]),
                "PCP_SIZE": str(point["pcp-size"]),
                "DP_ATTENTION": str(point["dp-attn"]).lower(),
                "GPU_COUNT": str(point["tp"] * point["pp"] * point["pcp-size"]),
            }
        )
    settings = _settings(point)
    if settings.get("BENCH_SCRIPT_OVERRIDE"):
        raise ValueError("infx generate supports native SRT recipes, not script workloads")
    # Additional settings may tune custom clients, but matrix identity remains authoritative.
    return {**settings, **environment}


def _source(point: dict[str, Any], environment: Mapping[str, str], project: Path) -> str:
    multi = "prefill" in point
    reference = environment.get("CONFIG_FILE") if multi else point.get("srt-recipe")
    if not reference:
        raise ValueError(
            "infx generate requires srt-recipe or CONFIG_FILE; legacy workloads are unsupported"
        )
    path, separator, selector = reference.partition(":")
    recipe_path = Path(path)
    if not recipe_path.is_absolute():
        if multi and path.startswith("recipes/"):
            recipe_path = (
                project / "benchmarks/multi_node/srt-slurm-recipes" / path.removeprefix("recipes/")
            )
        else:
            recipe_path = project / path
    return str(recipe_path) + (f":{selector}" if separator else "")


def _selected_variants(
    reference: str, environment: dict[str, str]
) -> list[tuple[str, dict[str, Any]]]:
    if environment["IS_MULTINODE"] == "false":
        return [select_recipe(reference, environment)]

    from infx.srt_slurm.common import load_recipe

    path, _, selector = reference.partition(":")
    variants = selected_recipes(load_recipe(Path(path), environment), selector or None)
    if not variants:
        raise ValueError(f"No multi-node SRT variants selected: {reference}")
    return [(f"{path}:{name}" if name else path, recipe) for name, recipe in variants]


def _materialize(
    selected: str, recipe: dict[str, Any], environment: dict[str, str]
) -> dict[str, Any]:
    from marshmallow import ValidationError
    from srtctl.core.config import resolve_config_with_defaults
    from srtctl.core.overrides import apply_overrides_to_recipe, parse_overrides
    from srtctl.core.schema import SrtConfig

    bound = bind_workload(recipe, environment)
    if bound["benchmark"].get("type") == "custom":
        client = bound["benchmark"].setdefault("env", {})
        for name in (
            "RUN_EVAL",
            "EVAL_ONLY",
            "FRAMEWORK",
            "RESULT_FILENAME",
            "RESULT_DIR",
            "GPU_MONITOR_INTERVAL",
        ):
            client[name] = environment[name]
        for name in ("PREFILL_NUM_WORKERS", "PREFILL_TP", "DECODE_NUM_WORKERS", "DECODE_TP"):
            if name in environment:
                client[name] = environment[name]
    # Fixed-sequence runs always verify speculative tokens; remove old simulation settings.
    overrides = build_overrides(bound, environment["FRAMEWORK"], environment)
    if overrides:
        pairs = list(zip(overrides[::2], overrides[1::2], strict=True))
        sets = [value for flag, value in pairs if flag == "--set"]
        unsets = [value for flag, value in pairs if flag == "--unset"]
        apply_overrides_to_recipe(bound, parse_overrides(sets, unsets))
    try:
        SrtConfig.Schema().load(resolve_config_with_defaults(deepcopy(bound), None))
    except ValidationError as error:
        raise ValueError(f"Invalid generated SRT recipe {selected}: {error}") from error
    return bound


def generate_recipes(
    *,
    config_keys: list[str],
    config_files: list[Path],
    runner_file: Path,
    project: Path,
    output: Path,
    gpu_monitor_interval: int = 1,
    refresh_exports: bool = False,
) -> dict[str, Any]:
    """Validate all selected points, then emit plain YAML recipes and a provenance manifest."""
    if gpu_monitor_interval <= 0:
        raise ValueError("gpu-monitor-interval must be positive")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output directory must be empty: {output}")
    revision = _upstream(project)
    master = load_config_files([str(path) for path in config_files])
    runners = load_runner_file(str(runner_file))
    artifacts: list[tuple[str, dict[str, Any]]] = []
    records: list[dict[str, Any]] = []
    exports: dict[Path, dict[str, Any]] = {}
    registry_path = project / "configs/srt-recipes/sources.yaml"
    registry = (
        yaml.safe_load(registry_path.read_text())
        if refresh_exports and registry_path.exists()
        else {}
    )
    with _project(project):
        for key in expand_config_keys(config_keys, master):
            config = master[key]
            if config["framework"] == "tilert":
                raise ValueError("infx generate does not support the TileRT fork")
            if config["scenarios"].get("agentic-coding"):
                raise ValueError("infx generate currently supports fixed-sequence scenarios only")
            points = generate_config_matrix([key], master, runners, eval_mode="none")
            if not points:
                raise ValueError(f"No fixed-sequence points selected for {key}")
            for point in points:
                environment = _environment(point, interval=gpu_monitor_interval)
                reference = _source(point, environment, project)
                for selected, recipe in _selected_variants(reference, environment):
                    variant = selected.partition(":")[2] or None
                    identity = {"matrix": point, "variant": variant}
                    digest = hashlib.sha256(
                        json.dumps(identity, sort_keys=True).encode()
                    ).hexdigest()[:12]
                    stem = f"{re.sub(r'[^A-Za-z0-9_.-]', '_', key)}-{digest}"
                    environment["RESULT_FILENAME"] = stem
                    bound = _materialize(selected, recipe, environment)
                    filename = f"{stem}.yaml"
                    artifacts.append((filename, bound))
                    records.append(
                        {
                            "file": filename,
                            "config-key": key,
                            "source": selected,
                            "variant": variant,
                            "matrix": point,
                        }
                    )
                if refresh_exports:
                    _prepare_export(reference, environment, project, registry, exports)
    manifest = {
        "schema": 1,
        "srt-slurm-revision": revision,
        "master-files": [str(path.resolve()) for path in config_files],
        "runner-file": str(runner_file.resolve()),
        "recipes": records,
    }
    output.mkdir(parents=True, exist_ok=True)
    for filename, recipe in artifacts:
        (output / filename).write_text(yaml.safe_dump(recipe, sort_keys=False))
    for path, recipe in exports.items():
        path.write_text(
            "# Generated by infx generate --refresh-exports; edit configs/srt-recipes sources.\n"
            + yaml.safe_dump(recipe, sort_keys=False)
        )
    manifest["refreshed-exports"] = [str(path) for path in exports]
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _prepare_export(
    reference: str,
    environment: Mapping[str, str],
    project: Path,
    registry: dict[str, Any],
    exports: dict[Path, dict[str, Any]],
) -> None:
    from marshmallow import ValidationError
    from srtctl.core.config import resolve_config_with_defaults
    from srtctl.core.schema import SrtConfig

    from infx.srt_slurm.common import load_recipe

    path = Path(reference.partition(":")[0]).resolve()
    try:
        relative = path.relative_to(project.resolve()).as_posix()
    except ValueError:
        return
    if relative not in registry:
        return
    raw = load_recipe(path, environment)
    if path in exports and exports[path] != raw:
        raise ValueError(f"Conflicting selected workloads for registered native export: {path}")
    for name, recipe in selected_recipes(raw, None):
        try:
            SrtConfig.Schema().load(resolve_config_with_defaults(recipe, None))
        except ValidationError as error:
            raise ValueError(f"Invalid native export {path}:{name}: {error}") from error
    exports[path] = raw
