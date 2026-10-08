"""Compose srt-slurm recipe fragments and bind the master-owned workload into them.

A fragment holds only recipe-specific srt-slurm settings. Its lane's shared block, and the
DCGM telemetry block for a point that measures power, is merged under it, a variant is
selected, and the binder then writes the matrix point's and the launcher's values and
replaces ``'@fabric.<name>'`` values with the job's cluster facts.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from infx.config import repository_root
from infx.srt_slurm.synthetic_acceptance import selected_recipes, spec_parameters

# Keyed by (agentic, multinode).
SHARED_BLOCKS = {
    (False, False): Path("configs/srt-recipes/fixed-sequence-single.yaml"),
    (False, True): Path("configs/srt-recipes/fixed-sequence-multi.yaml"),
    (True, False): Path("configs/srt-recipes/agentic-single.yaml"),
    (True, True): Path("configs/srt-recipes/agentic-multi.yaml"),
}
TELEMETRY_BLOCK = Path("configs/srt-recipes/telemetry-dcgm.yaml")
# The binder writes these, the master power field owns telemetry, and the launcher hands
# the benchmark client its cache and result paths.
_POINT_KEYS = (
    ("model", "path"),
    ("model", "container"),
    ("model", "precision"),
    ("identity", "model", "repo"),
    ("identity", "container", "image"),
    ("benchmark", "concurrencies"),
    ("telemetry", "enabled"),
    ("telemetry", "dcgm_exporter", "port"),
)
_FIXED_ENV = ("MODEL", "ISL", "OSL", "CONC", "CONC_LIST", "RANDOM_RANGE_RATIO", "USE_CHAT_TEMPLATE")
_AGENTIC_ENV = ("CONC", "CONC_LIST", "RESULT_DIR", "AGENTIC_OUTPUT_DIR", "HF_HUB_CACHE",
                "HUGGINGFACE_HUB_CACHE")  # fmt: skip
BOUND_ENV = {
    (False, False): _FIXED_ENV,
    (False, True): _FIXED_ENV,
    (True, False): ("MODEL", *_AGENTIC_ENV),
    (True, True): (*_AGENTIC_ENV, "AIPERF_DATASET_MMAP_CACHE_DIR"),
}
# A single-node variant may name its point, pairing that concurrency with its tuning.
VARIANT_CONCURRENCY = ("benchmark", "env", "CONC")
FABRIC_REFERENCE = "@fabric."


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


def check_fragment(raw: Mapping[str, Any], source: Path, *, agentic: bool, multinode: bool) -> None:
    """Reject a fragment that sets a bound key, even to the value the binder would write."""
    keys = (*_POINT_KEYS, *(("benchmark", "env", name) for name in BOUND_ENV[agentic, multinode]))
    blocks = [(name, block) for name, block in raw.items() if name != "schema"]
    if "base" not in raw:
        blocks = [(None, raw)]
    found = []
    for name, block in blocks:
        variant = name not in (None, "base")
        for key in keys:
            present, value = _lookup(block, key)
            if present and not (variant and not multinode and key == VARIANT_CONCURRENCY):
                found.append(f"{'.'.join(filter(None, (name, *key)))} (= {value!r})")
    if found:
        raise ValueError(
            f"{source}: remove {', '.join(found)} from the fragment; the launcher binds them"
        )


def _block(path: Path) -> dict[str, Any]:
    block = yaml.safe_load(path.read_text())
    if not isinstance(block, dict):
        raise ValueError(f"{path}: shared block must be a mapping")
    return block


def compose_recipe(
    path: Path, *, agentic: bool, multinode: bool, root: Path, power_port: int | None = None
) -> dict[str, Any]:
    """The fragment at ``path`` over its lane's shared block, under ``base`` for bundles.

    ``power_port`` adds the DCGM telemetry block with its exporter on that port.
    """
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: recipe must be a mapping")
    check_fragment(raw, path, agentic=agentic, multinode=multinode)
    shared = _block(root / SHARED_BLOCKS[agentic, multinode])
    if power_port is not None:
        exporter = {"telemetry": {"dcgm_exporter": {"port": power_port}}}
        shared = merge_blocks(shared, merge_blocks(_block(root / TELEMETRY_BLOCK), exporter))
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
    recipe: Mapping[str, Any],
    environment: Mapping[str, str],
    *,
    agentic: bool,
    multinode: bool,
    client_env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Write the matrix point's model, image, precision, concurrency and, for fixed sequences,
    lengths; ``client_env`` holds the launcher's benchmark client paths.

    Call after variant selection: binding a zip group would detach its concurrency
    from the tuning it pairs with.
    """
    image, model = _required(environment, "IMAGE"), _required(environment, "MODEL")
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
        # Identity names the registry reference, not Pyxis's registry#path spelling.
        identity["container"]["image"] = image.replace("#", "/", 1)
    benchmark = bound.setdefault("benchmark", {})
    workload = benchmark.setdefault("env", {})
    if not agentic:
        workload.update(
            (name, str(_positive(_required(environment, name), name))) for name in ("ISL", "OSL")
        )
    if multinode:
        # The client reads CONC_LIST from the job environment the workflow exports.
        words = _required(environment, "CONC_LIST").split()
        concurrencies = [_positive(word, "CONC_LIST") for word in words]
    else:
        conc = _positive(_required(environment, "CONC"), "CONC")
        if "CONC" in workload and str(workload["CONC"]) != str(conc):
            raise ValueError(f"CONC: recipe {workload['CONC']} != point {conc}")
        workload.update(MODEL=model, CONC=str(conc))
        if not agentic:
            engine = bound["engine"]
            speculates = spec_parameters(
                bound["roles"]["agg"], engine["type"] if isinstance(engine, Mapping) else engine
            )
            workload.update(
                RANDOM_RANGE_RATIO=_required(environment, "RANDOM_RANGE_RATIO"),
                USE_CHAT_TEMPLATE="true" if speculates else "false",
            )
        concurrencies = [conc]
    # A fragment that sets HF_HOME keeps its own Hugging Face cache layout.
    workload.update(
        (name, value)
        for name, value in (client_env or {}).items()
        if not (name == "HF_HUB_CACHE" and "HF_HOME" in workload)
    )
    # srtctl derives power-telemetry windows from benchmark.concurrencies.
    if (bound.get("telemetry") or {}).get("enabled") is True:
        benchmark["concurrencies"] = concurrencies
    return bound


def resolve_fabric(node: Any, fabric: Mapping[str, str | None], where: str = "") -> Any:
    """``node`` with each ``'@fabric.<name>'`` value replaced by the cluster's rendering.

    ``fabric`` maps every field name to its rendering, None where the cluster sets none.
    """
    if isinstance(node, Mapping):
        return {
            key: resolve_fabric(value, fabric, f"{where}.{key}" if where else str(key))
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [
            resolve_fabric(item, fabric, f"{where}[{index}]") for index, item in enumerate(node)
        ]
    if not isinstance(node, str) or FABRIC_REFERENCE not in node:
        return node
    name = node.removeprefix(FABRIC_REFERENCE)
    if name == node or name not in fabric:
        raise ValueError(
            f"{where}: {node!r} is not a whole '@fabric.<name>' value naming one of: "
            + ", ".join(fabric)
        )
    if (value := fabric[name]) is None:
        raise ValueError(f"{where}: this cluster sets no srt-slurm.fabric.{name}")
    return value


def add_fabric_argument(parser: argparse.ArgumentParser) -> None:
    """``--fabric``: the job cluster's ``Fabric.rendered()`` as JSON."""
    parser.add_argument(
        "--fabric",
        type=json.loads,
        required=True,
        metavar="JSON",
        help="the cluster's fabric fields as recipes read them, null where unset",
    )


def _assignment(text: str) -> tuple[str, str]:
    name, separator, value = text.partition("=")
    if not separator or not name:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {text!r}")
    return name, value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Bind one multi-node recipe variant")
    parser.add_argument("recipe", help="fragment[:selector]")
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--power-port", type=int, help="merge the DCGM telemetry block, its exporter on this port"
    )
    parser.add_argument(
        "--client-env",
        type=_assignment,
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="a launcher-owned benchmark client setting",
    )
    add_fabric_argument(parser)
    args = parser.parse_args(argv)
    path, _, selector = args.recipe.partition(":")
    agentic = os.environ.get("IS_AGENTIC") == "1"
    try:
        composed = compose_recipe(
            Path(path),
            agentic=agentic,
            multinode=True,
            root=repository_root(),
            power_port=args.power_port,
        )
        variants = selected_recipes(composed, selector or None)
        if len(variants) != 1:
            raise ValueError(f"{args.recipe} selects {len(variants)} variants, not one")
        bound = bind_workload(
            variants[0][1],
            os.environ,
            agentic=agentic,
            multinode=True,
            client_env=dict(args.client_env),
        )
        bound = resolve_fabric(bound, args.fabric)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        parser.error(str(error))
    args.output.write_text(yaml.safe_dump(bound, sort_keys=False))


if __name__ == "__main__":
    main()
