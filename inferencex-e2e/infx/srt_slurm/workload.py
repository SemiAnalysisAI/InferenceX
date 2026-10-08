"""Compose srt-slurm recipe fragments and bind the master-owned workload into them.

A fragment holds only recipe-specific srt-slurm settings. Its lane's shared block, and the
DCGM telemetry block for a point that measures power, is merged under it, a variant is
selected, and the binder then writes the matrix point's and the launcher's values and
replaces ``'@dram.<name>'`` values with the point's host DRAM budget and
``'@fabric.<name>'`` values with the job's cluster facts.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from collections.abc import Iterator, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from infx.config import repository_root
from infx.srt_slurm.synthetic_acceptance import selected_recipes, spec_parameters

# Defined here: srtctl's venv runs this module on Python 3.10, so it cannot import infx.clusters.
FABRIC_REFERENCE = "@fabric."

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
_AGENTIC_ENV = ("CONC", "CONC_LIST", "KV_OFFLOADING", "TOTAL_CPU_DRAM_GB", "RESULT_DIR",
                "AGENTIC_OUTPUT_DIR", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE")  # fmt: skip
BOUND_ENV = {
    (False, False): _FIXED_ENV,
    (False, True): _FIXED_ENV,
    (True, False): ("MODEL", *_AGENTIC_ENV),
    (True, True): (*_AGENTIC_ENV, "AIPERF_DATASET_MMAP_CACHE_DIR"),
}
# A single-node variant may name its point, pairing that concurrency and KV offloading with
# its tuning.
VARIANT_POINT_KEYS = (("benchmark", "env", "CONC"), ("benchmark", "env", "KV_OFFLOADING"))

# '@dram.<name>' takes the point's TOTAL_CPU_DRAM_GB (decimal GB), in total or per GPU it
# covers on a node.
DRAM_REFERENCE = "@dram."
DRAM_NAMES = ("total-gb", "total-bytes", "per-gpu-gb", "per-gpu-bytes")
BYTES_PER_GB = 1_000_000_000
# Host DRAM sizes that only the budget sizes. HiCache, TRT-LLM host cache and Mooncake
# segment sizes may also be measured values.
DRAM_SIZES = frozenset({
    "cpu_bytes_to_use", "cpu_bytes_to_use_per_rank",  # vLLM SimpleCPUOffloadConnector
    "LMCACHE_MAX_LOCAL_CPU_SIZE", "lmcache.max_local_cpu_size", "--l1-size-gb",  # LMCache
})  # fmt: skip
# srt-slurm takes env values and argument list items as strings.
TEXT_MAPPINGS = frozenset({"env", "environment"})


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


def _is_dram(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(DRAM_REFERENCE)


def _dram_literals(node: Any, where: str) -> Iterator[str]:
    """Each DRAM_SIZES value in ``node``, a JSON object string included, that is not a
    ``'@dram.<name>'`` value; a size flag in an argument list sizes the next item."""
    if isinstance(node, str) and node.startswith("{"):
        with contextlib.suppress(ValueError):
            node = json.loads(node)
    if isinstance(node, Mapping):
        for key, value in node.items():
            path = f"{where}.{key}" if where else str(key)
            if key in DRAM_SIZES and not _is_dram(value):
                yield f"{path} (= {value!r})"
            else:
                yield from _dram_literals(value, path)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            flag = node[index - 1] if index else None
            if isinstance(flag, str) and flag in DRAM_SIZES and not _is_dram(item):
                yield f"{where}[{index}] (= {item!r})"
            else:
                yield from _dram_literals(item, f"{where}[{index}]")


def check_fragment(raw: Mapping[str, Any], source: Path, *, agentic: bool, multinode: bool) -> None:
    """Reject a fragment that sets a bound key, even to the value the binder would write, or
    sizes host DRAM with a literal instead of the point's budget."""
    keys = (*_POINT_KEYS, *(("benchmark", "env", name) for name in BOUND_ENV[agentic, multinode]))
    blocks = [(name, block) for name, block in raw.items() if name != "schema"]
    if "base" not in raw:
        blocks = [(None, raw)]
    found = []
    for name, block in blocks:
        variant = name not in (None, "base")
        for key in keys:
            present, value = _lookup(block, key)
            if present and not (variant and not multinode and key in VARIANT_POINT_KEYS):
                found.append(f"{'.'.join(filter(None, (name, *key)))} (= {value!r})")
    if found:
        raise ValueError(
            f"{source}: remove {', '.join(found)} from the fragment; the launcher binds them"
        )
    if sizes := [size for name, block in blocks for size in _dram_literals(block, name or "")]:
        raise ValueError(
            f"{source}: set {', '.join(sizes)} to a '@dram.<name>' value; the launcher binds the"
            " point's DRAM budget"
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
    """Write the matrix point's model, image, precision, concurrency, KV offloading and DRAM
    budget and, for fixed sequences, lengths; ``client_env`` holds the launcher's benchmark
    client paths.

    Call after variant selection: binding a zip group would detach its concurrency
    from the tuning it pairs with. ``resolve_dram`` then sizes the recipe's host DRAM.
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
    if agentic:
        offloading = _required(environment, "KV_OFFLOADING")
        if workload.get("KV_OFFLOADING", offloading) != offloading:
            raise ValueError(
                f"KV_OFFLOADING: recipe {workload['KV_OFFLOADING']} != point {offloading}"
            )
        workload["KV_OFFLOADING"] = offloading
        if offloading == "dram":
            total = _positive(_required(environment, "TOTAL_CPU_DRAM_GB"), "TOTAL_CPU_DRAM_GB")
            workload["TOTAL_CPU_DRAM_GB"] = str(total)
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


def dram_budget(
    environment: Mapping[str, str], *, multinode: bool, gpus_per_node: int | None = None
) -> dict[str, int] | None:
    """The point's ``'@dram.<name>'`` values; None unless it offloads KV to DRAM.

    As the matrix sizes it, the budget covers the GPUs a single-node point serves on, or
    those a multi-node point's prefill (or aggregated) worker uses on each of its nodes.
    """
    if environment.get("KV_OFFLOADING") != "dram":
        return None
    total = _positive(_required(environment, "TOTAL_CPU_DRAM_GB"), "TOTAL_CPU_DRAM_GB")
    if not multinode:
        gpus = _positive(_required(environment, "GPU_COUNT"), "GPU_COUNT")
    elif gpus_per_node is None:
        raise ValueError("A multi-node DRAM point needs its cluster's gpus-per-node")
    else:
        sizes = ("PREFILL_TP", "PREFILL_PP_SIZE", "PREFILL_PCP_SIZE")
        gpus = min(math.prod(_positive(_required(environment, n), n) for n in sizes), gpus_per_node)
    return {
        "total-gb": total,
        "total-bytes": total * BYTES_PER_GB,
        "per-gpu-gb": total // gpus,
        "per-gpu-bytes": total * BYTES_PER_GB // gpus,
    }


def resolve_dram(
    node: Any, budget: Mapping[str, int] | None, where: str = "", *, text: bool = False
) -> Any:
    """``node`` with each ``'@dram.<name>'`` value replaced by its ``budget`` value.

    A reference is a whole value, or a whole value in a JSON object string such as
    ``kv-transfer-config``. It becomes an integer, or its decimal string (``text``) as an env
    value or argument list item.
    """
    if isinstance(node, Mapping):
        return {
            key: resolve_dram(
                value,
                budget,
                f"{where}.{key}" if where else str(key),
                text=text or key in TEXT_MAPPINGS,
            )
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [
            resolve_dram(item, budget, f"{where}[{index}]", text=not isinstance(item, Mapping))
            for index, item in enumerate(node)
        ]
    if not isinstance(node, str) or DRAM_REFERENCE not in node:
        return node
    document = None
    if node.startswith("{"):
        with contextlib.suppress(ValueError):
            document = json.loads(node)
    if isinstance(document, Mapping):
        return json.dumps(resolve_dram(document, budget, where), separators=(",", ":"))
    name = node.removeprefix(DRAM_REFERENCE)
    if name == node or name not in DRAM_NAMES:
        raise ValueError(
            f"{where}: {node!r} is not a whole '@dram.<name>' value naming one of: "
            + ", ".join(DRAM_NAMES)
        )
    if budget is None:
        raise ValueError(f"{where}: {node!r} sizes host DRAM on a point without a DRAM budget")
    return str(budget[name]) if text else budget[name]


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
    parser.add_argument(
        "--gpus-per-node", type=int, help="the cluster's GPUs per node, which a DRAM budget covers"
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
        budget = dram_budget(os.environ, multinode=True, gpus_per_node=args.gpus_per_node)
        bound = resolve_dram(bound, budget)
        bound = resolve_fabric(bound, args.fabric)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        parser.error(str(error))
    args.output.write_text(yaml.safe_dump(bound, sort_keys=False))


if __name__ == "__main__":
    main()
