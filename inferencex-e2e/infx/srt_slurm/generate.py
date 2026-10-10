"""Write the bound fixed-sequence srt-slurm recipes of master-config points for inspection."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from infx.matrix.generate import expand_config_keys, generate_config_matrix
from infx.matrix.validation import config_root, load_config_files, load_runner_file
from infx.srt_slurm.single_node import select_recipe
from infx.srt_slurm.synthetic_acceptance import build_overrides
from infx.srt_slurm.workload import bind_multinode

# What benchmark-tmpl.yml and benchmark-multinode-tmpl.yml export.
RANDOM_RANGE_RATIO = "0.8"


def _environment(point: Mapping[str, Any]) -> dict[str, str]:
    """The workflow environment the launcher reads for ``point``."""
    environment = {
        "IMAGE": point["image"],
        "MODEL": point["model"],
        "PRECISION": point["precision"],
        "FRAMEWORK": point["framework"],
        "SPEC_DECODING": point["spec-decoding"],
        "IS_AGENTIC": "0",
        "ISL": str(point["isl"]),
        "OSL": str(point["osl"]),
        "RANDOM_RANGE_RATIO": RANDOM_RANGE_RATIO,
        "EVAL_ONLY": "false",
    }
    if "prefill" in point:
        environment["CONC_LIST"] = " ".join(map(str, point["conc"]))
        return environment
    return {
        **environment,
        "CONC": str(point["conc"]),
        "TP": str(point["tp"]),
        "EP_SIZE": str(point["ep"]),
        "PP_SIZE": str(point["pp"]),
        "DCP_SIZE": str(point["dcp-size"]),
        "PCP_SIZE": str(point["pcp-size"]),
        "DP_ATTENTION": str(point["dp-attn"]).lower(),
        "GPU_COUNT": str(point["tp"] * point["pp"] * point["pcp-size"]),
    }


def _bound_variant(
    point: Mapping[str, Any], environment: Mapping[str, str], root: Path
) -> tuple[str | None, dict[str, Any]]:
    """The variant the launcher submits for ``point``, bound."""
    if "prefill" not in point:
        selected, recipe = select_recipe(str(root / point["srt-recipe"]), environment, root=root)
        return selected.partition(":")[2] or None, recipe
    return bind_multinode(str(root / point["srt-recipe"]), environment, root=root)


def _apply_acceptance_and_validate(
    recipe: dict[str, Any], environment: Mapping[str, str], source: str
) -> None:
    """Apply golden-acceptance cleanup, then load the result as srtctl would."""
    from marshmallow import ValidationError
    from srtctl.core.config import expand_engine_config_defaults, resolve_config_with_defaults
    from srtctl.core.overrides import apply_overrides_to_recipe, parse_overrides
    from srtctl.core.schema import SrtConfig

    arguments = build_overrides(recipe, environment["FRAMEWORK"], environment)
    pairs = list(zip(arguments[::2], arguments[1::2], strict=True))
    apply_overrides_to_recipe(
        recipe,
        parse_overrides(
            [value for flag, value in pairs if flag == "--set"],
            [value for flag, value in pairs if flag == "--unset"],
        ),
    )
    resolved = resolve_config_with_defaults(deepcopy(recipe), None)
    expand_engine_config_defaults(resolved)
    try:
        SrtConfig.Schema().load(resolved)
    except ValidationError as error:
        raise ValueError(f"{source}: srtctl rejects the bound recipe: {error}") from error


def generate_recipes(
    *, config_keys: list[str], config_files: list[Path], runner_file: Path, output: Path
) -> dict[str, Any]:
    """Bind every fixed-sequence point of ``config_keys``; write recipes and a manifest."""
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output directory must be empty: {output}")
    files = [str(path) for path in config_files]
    root = config_root(files)
    source = root / "utils/srt-slurm/src"
    if not (source / "srtctl").is_dir():
        raise ValueError(f"No srt-slurm checkout at {source}; initialize the submodule")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    master = load_config_files(files)
    runners = load_runner_file(str(runner_file))
    recipes: dict[str, dict[str, Any]] = {}
    records = []
    for key in expand_config_keys(config_keys, master):
        points = generate_config_matrix(
            [key], master, runners, scenario_types=["fixed-seq-len"], eval_mode="none", root=root
        )
        if not points:
            raise ValueError(f"{key} has no fixed-sequence points")
        for point in points:
            environment = _environment(point)
            variant, recipe = _bound_variant(point, environment, root)
            _apply_acceptance_and_validate(recipe, environment, point["srt-recipe"])
            identity = json.dumps({"point": point, "variant": variant}, sort_keys=True)
            digest = hashlib.sha256(identity.encode()).hexdigest()[:12]
            name = f"{re.sub(r'[^A-Za-z0-9_.-]', '_', key)}-{digest}.yaml"
            recipes[name] = recipe
            records.append({"file": name, "config-key": key, "variant": variant, "matrix": point})
    output.mkdir(parents=True, exist_ok=True)
    for name, recipe in recipes.items():
        (output / name).write_text(yaml.safe_dump(recipe, sort_keys=False))
    manifest = {"recipes": records}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
