"""Compose fixed-sequence recipe fragments and bind the master-owned workload into them.

A fragment holds only recipe-specific srt-slurm settings. Its lane's shared block is merged
under it, a variant is selected, and the binder then writes the matrix point's values.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from infx.config import repository_root
from infx.srt_slurm.synthetic_acceptance import selected_recipes, spec_parameters

SHARED_BLOCKS = {
    False: Path("configs/srt-recipes/fixed-sequence-single.yaml"),
    True: Path("configs/srt-recipes/fixed-sequence-multi.yaml"),
}
# The binder writes these, or the workflow exports them to the benchmark client.
BOUND_KEYS = (
    ("model", "path"),
    ("model", "container"),
    ("model", "precision"),
    ("identity", "model", "repo"),
    ("identity", "container", "image"),
    ("benchmark", "concurrencies"),
    *(
        ("benchmark", "env", name)
        for name in (
            "MODEL", "ISL", "OSL", "CONC", "CONC_LIST", "RANDOM_RANGE_RATIO", "USE_CHAT_TEMPLATE",
        )
    ),
)  # fmt: skip
# A single-node variant may name its point, pairing that concurrency with its tuning.
VARIANT_CONCURRENCY = ("benchmark", "env", "CONC")


def merge_blocks(shared: Mapping[str, Any], fragment: Mapping[str, Any]) -> dict[str, Any]:
    """Merge ``fragment`` over ``shared``: mappings merge, anything else replaces."""
    merged = {}
    for key, value in fragment.items():
        default = shared.get(key)
        merged[key] = (
            merge_blocks(default, value)
            if isinstance(default, Mapping) and isinstance(value, Mapping)
            else deepcopy(value)
        )
    for key, value in shared.items():
        merged.setdefault(key, deepcopy(value))
    return merged


def _lookup(block: Any, key: tuple[str, ...]) -> tuple[bool, Any]:
    for part in key:
        if not isinstance(block, Mapping) or part not in block:
            return False, None
        block = block[part]
    return True, block


def check_fragment(raw: Mapping[str, Any], source: Path, *, multinode: bool) -> None:
    """Reject a fragment that sets a bound key, even to the value the binder would write."""
    blocks = [(name, block) for name, block in raw.items() if name != "schema"]
    if "base" not in raw:
        blocks = [(None, raw)]
    found = []
    for name, block in blocks:
        variant = name not in (None, "base")
        for key in BOUND_KEYS:
            present, value = _lookup(block, key)
            if present and not (variant and not multinode and key == VARIANT_CONCURRENCY):
                found.append(f"{'.'.join(filter(None, (name, *key)))} (= {value!r})")
    if found:
        raise ValueError(
            f"{source}: remove {', '.join(found)} from the fragment; "
            "they are bound from the matrix point"
        )


def compose_recipe(path: Path, *, multinode: bool, root: Path) -> dict[str, Any]:
    """The fragment at ``path`` over its lane's shared block, under ``base`` for bundles."""
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: recipe must be a mapping")
    check_fragment(raw, path, multinode=multinode)
    shared = yaml.safe_load((root / SHARED_BLOCKS[multinode]).read_text())
    if "base" in raw:
        return {**raw, "base": merge_blocks(shared, raw["base"])}
    return merge_blocks(shared, raw)


def _required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name, "")
    if not value.strip():
        raise ValueError(f"Missing workload input: {name}")
    return value


def _positive(value: str, name: str) -> int:
    if not value.isascii() or not value.isdecimal() or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer: {value!r}")
    return int(value)


def bind_workload(
    recipe: Mapping[str, Any], environment: Mapping[str, str], *, multinode: bool
) -> dict[str, Any]:
    """Write the matrix point's model, image, precision, lengths and concurrency.

    Call after variant selection: binding a zip group would detach its concurrency
    from the tuning it pairs with.
    """
    image, model = _required(environment, "IMAGE"), _required(environment, "MODEL")
    lengths = {name: _positive(_required(environment, name), name) for name in ("ISL", "OSL")}
    bound = {key: deepcopy(value) for key, value in recipe.items() if key in ("schema", "name")}
    bound["model"] = {
        **recipe.get("model", {}),
        "path": f"hf:{model}",
        "container": image,
        "precision": _required(environment, "PRECISION"),
    }
    bound.update((key, deepcopy(value)) for key, value in recipe.items() if key not in bound)
    identity = bound.get("identity") or {}
    if isinstance(identity.get("model"), dict):
        identity["model"]["repo"] = model
    if isinstance(identity.get("container"), dict):
        identity["container"]["image"] = image
    benchmark = bound.setdefault("benchmark", {})
    workload = benchmark.setdefault("env", {})
    workload.update((name, str(value)) for name, value in lengths.items())
    if multinode:
        # The client reads CONC_LIST from the job environment the workflow exports.
        words = _required(environment, "CONC_LIST").split()
        concurrencies = [_positive(word, "CONC_LIST") for word in words]
    else:
        conc = _positive(_required(environment, "CONC"), "CONC")
        if "CONC" in workload and str(workload["CONC"]) != str(conc):
            raise ValueError(f"CONC: recipe {workload['CONC']} != point {conc}")
        engine = bound["engine"]
        speculates = spec_parameters(
            bound["roles"]["agg"], engine["type"] if isinstance(engine, Mapping) else engine
        )
        workload.update(
            MODEL=model,
            CONC=str(conc),
            RANDOM_RANGE_RATIO=_required(environment, "RANDOM_RANGE_RATIO"),
            USE_CHAT_TEMPLATE="true" if speculates else "false",
        )
        concurrencies = [conc]
    # srtctl derives power-telemetry windows from benchmark.concurrencies.
    if (bound.get("telemetry") or {}).get("enabled") is True:
        benchmark["concurrencies"] = concurrencies
    return bound


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Bind one multi-node fixed-sequence recipe")
    parser.add_argument("recipe", help="fragment[:selector]")
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    path, _, selector = args.recipe.partition(":")
    try:
        composed = compose_recipe(Path(path), multinode=True, root=repository_root())
        variants = selected_recipes(composed, selector or None)
        if len(variants) != 1:
            raise ValueError(f"{args.recipe} selects {len(variants)} variants, not one")
        bound = bind_workload(variants[0][1], os.environ, multinode=True)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        parser.error(str(error))
    args.output.write_text(yaml.safe_dump(bound, sort_keys=False))


if __name__ == "__main__":
    main()
