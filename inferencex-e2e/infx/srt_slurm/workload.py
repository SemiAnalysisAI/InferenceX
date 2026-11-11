"""Compose srt-slurm recipe fragments and bind the master-owned workload into them.

A fragment holds only recipe-specific srt-slurm settings. Its lane's shared block, and the
GPU telemetry block for a point that measures power, is merged under it, a variant is
selected, and the binder then writes the matrix point's and the launcher's values and
replaces ``'@fabric.<name>'`` values with the job's cluster facts.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import re
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any, Literal, get_args

import yaml

from infx import yamlio
from infx.config import repository_root
from infx.fastcopy import deepcopy
from infx.results.metadata import parse_component_metadata
from infx.srt_slurm.spec_identity import speculation_facts
from infx.srt_slurm.synthetic_acceptance import (
    ENGINES,
    generation_role,
    selected_recipes,
    spec_parameters,
)
from infx.srt_slurm.variants import expand_variants

SHARED_BLOCKS = {
    (False, False): Path("configs/srt-recipes/fixed-sequence-single.yaml"),
    (False, True): Path("configs/srt-recipes/fixed-sequence-multi.yaml"),
    (True, False): Path("configs/srt-recipes/agentic.yaml"),
    (True, True): Path("configs/srt-recipes/agentic.yaml"),
}
BOUND_RECIPE = "recipe.yaml"
TELEMETRY_BLOCK = Path("configs/srt-recipes/telemetry-dcgm.yaml")
# The CPU power exporter sources a cluster declares (srt-slurm.cpu-power-exporter).
CpuPowerSource = Literal["acpi", "rapl"]
NATIVE_TELEMETRY_BLOCK = Path("configs/srt-recipes/telemetry-native.yaml")
VIDEO_BLOCK = Path("configs/srt-recipes/video.yaml")
# How a recipe measures GPU power: the DCGM telemetry block with its exporter on this port, or
# NATIVE, the cluster's own GPU exporter.
NATIVE = "native"
GpuPower = int | Literal["native"]
_POINT_KEYS = (
    ("model", "path"),
    ("model", "container"),
    ("model", "precision"),
    ("identity", "model", "repo"),
    ("identity", "container", "image"),
    ("benchmark", "concurrencies"),
    ("telemetry", "enabled"),
    ("telemetry", "dcgm_exporter", "port"),
    ("telemetry", "cpu_power_exporter"),
)
_FIXED_ENV = ("MODEL", "ISL", "OSL", "CONC", "CONC_LIST", "RANDOM_RANGE_RATIO", "USE_CHAT_TEMPLATE")
_AGENTIC_ENV = ("CONC", "CONC_LIST", "RESULT_DIR", "AGENTIC_OUTPUT_DIR", "HF_HUB_CACHE",
                "HUGGINGFACE_HUB_CACHE")  # fmt: skip
_WORKFLOW_ENV = (
    "IMAGE", "MODEL_PREFIX", "FRAMEWORK", "PRECISION", "DISAGG", "DURATION", "MAX_MODEL_LEN",
    "KV_OFFLOAD_BACKEND", "KV_OFFLOAD_BACKEND_METADATA", "ROUTER_METADATA", "KV_P2P_TRANSFER",
    "POWER", "REQUIRE_POWER", "IS_AGENTIC", "SCENARIO_TYPE", "RUN_EVAL", "EVAL_ONLY",
    "SPEC_ALGORITHM", "SPEC_DRAFT_MODEL",
)  # fmt: skip
_MULTINODE_ENV = (
    "SPEC_DECODING", "KV_OFFLOADING", "TOTAL_CPU_DRAM_GB", "DECODE_CPU_DRAM_GB", "PREFILL_DWDP_SIZE",
    *(f"{role}_{field}" for role in ("PREFILL", "DECODE")
      for field in ("NUM_WORKERS", "TP", "PP_SIZE", "DCP_SIZE", "PCP_SIZE", "EP", "DP_ATTN")),
)  # fmt: skip
_SINGLE_NODE_ENV = (
    "TP", "EP_SIZE", "PP_SIZE", "DCP_SIZE", "PCP_SIZE", "DP_ATTENTION", "GPU_COUNT", "SPEC_DECODING",
)  # fmt: skip
BOUND_ENV = {
    (False, False): (*_FIXED_ENV, *_WORKFLOW_ENV, *_SINGLE_NODE_ENV),
    (False, True): (*_FIXED_ENV, *_WORKFLOW_ENV, *_MULTINODE_ENV),
    (True, False): (
        "MODEL",
        *_AGENTIC_ENV,
        *_WORKFLOW_ENV,
        *_SINGLE_NODE_ENV,
        "AIPERF_DATASET_MMAP_CACHE_DIR",
        "ENABLE_AGENTX_POWER",
    ),
    (True, True): (
        "MODEL",
        *_AGENTIC_ENV,
        *_WORKFLOW_ENV,
        *_MULTINODE_ENV,
        "AIPERF_DATASET_MMAP_CACHE_DIR",
        "IS_MULTINODE",
        "ENABLE_AGENTX_POWER",
    ),
}
FABRIC_REFERENCE = "@fabric."
INSTALLERS = {
    "tokenspeed-nightly.sh": ("router", "smg"),
    "glm5.3-tilert-rocm.sh": ("router", "tilert-pd-router"),
    "vllm-mooncake.sh": ("kv-offload-backend", "mooncake"),
    "vllm-router.sh": ("router", "vllm-router"),
}
VERSION_ENV = {"router": "ROUTER_VERSION", "kv-offload-backend": "KV_OFFLOAD_BACKEND_VERSION"}
_METADATA_ENV = {"router": "ROUTER_METADATA", "kv-offload-backend": "KV_OFFLOAD_BACKEND_METADATA"}
SETUP_SCRIPT_DIRS = (
    Path("benchmarks/multi_node/srt-slurm-recipes/configs"),
    Path("utils/srt-slurm/configs"),
)


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
    """Follow ``key`` into nested mappings and return whether it is present and its value."""
    for part in key:
        if not isinstance(block, Mapping) or part not in block:
            return False, None
        block = block[part]
    return True, block


def _fragment_versions(block: Any, path: str) -> Iterator[str]:
    """Where ``block`` sets a bound component version, in an env mapping at any depth."""
    if isinstance(block, list):
        for index, item in enumerate(block):
            yield from _fragment_versions(item, f"{path}[{index}]")
    elif isinstance(block, Mapping):
        for key, value in block.items():
            where = f"{path}.{key}" if path else str(key)
            if key in ("env", "environment") and isinstance(value, Mapping):
                for name in VERSION_ENV.values():
                    if name in value:
                        yield f"{where}.{name} (= {value[name]!r})"
            else:
                yield from _fragment_versions(value, where)


def _bound_keys(raw: Mapping[str, Any], *, agentic: bool, multinode: bool) -> list[str]:
    """Where a fragment sets a bound key, even to the value the binder would write."""
    keys = (
        *_POINT_KEYS,
        *(("benchmark", "env", name) for name in BOUND_ENV[agentic, multinode]),
        *(() if multinode else (("telemetry", "required"),)),
    )
    if "base" in raw:
        blocks = [(name, block) for name, block in raw.items() if name != "schema"]
    else:
        blocks = [(None, raw)]
    found = []
    for name, block in blocks:
        conc_allowed = not multinode and name not in (None, "base")
        for key in keys:
            present, value = _lookup(block, key)
            if present and not (conc_allowed and key == ("benchmark", "env", "CONC")):
                found.append(f"{'.'.join(filter(None, (name, *key)))} (= {value!r})")
        found.extend(_fragment_versions(block, name or ""))
    return found


@functools.lru_cache(maxsize=512)
def _text_bound_keys(text: str, agentic: bool, multinode: bool) -> tuple[str, ...]:
    """``_bound_keys`` of the fragment with this YAML ``text``, computed once per process."""
    return tuple(_bound_keys(yamlio.safe_load(text), agentic=agentic, multinode=multinode))


def _block(path: Path) -> dict[str, Any]:
    """Load the YAML mapping at ``path``, or raise ValueError if it is not a mapping."""
    block = yamlio.safe_load(path.read_text())
    if not isinstance(block, dict):
        raise ValueError(f"{path}: shared block must be a mapping")
    return block


def check_setup_script(recipe: Mapping[str, Any], source: Path, root: Path) -> None:
    """Fail where srtctl would only warn: a bound recipe's setup_script in none of the
    configs/ it stages. Launch and ``infx generate`` only: planner snapshots lack the
    srt-slurm submodule."""
    script = recipe.get("setup_script")
    if isinstance(script, str) and not any(
        (root / directory / sub / script).is_file()
        for directory in SETUP_SCRIPT_DIRS
        for sub in ("", "patches")
    ):
        raise ValueError(
            f"{source}: setup_script {script} is in none of "
            f"{', '.join(map(str, SETUP_SCRIPT_DIRS))} or their patches/"
        )


def compose_recipe(
    path: Path,
    *,
    agentic: bool,
    multinode: bool,
    root: Path,
    gpu_power: GpuPower | None = None,
    cpu_power_source: CpuPowerSource | None = None,
) -> dict[str, Any]:
    """The fragment at ``path`` over its lane's shared block, under ``base`` for bundles.

    ``gpu_power`` adds its telemetry block, and ``cpu_power_source`` the cluster's CPU power
    exporter to the telemetry block.
    """
    text = path.read_text()
    raw = yamlio.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: recipe must be a mapping")
    if found := _text_bound_keys(text, agentic, multinode):
        raise ValueError(
            f"{path}: remove {', '.join(found)} from the fragment; "
            "the launcher or the workflow sets them"
        )
    shared = _block(root / SHARED_BLOCKS[agentic, multinode])
    if gpu_power == NATIVE:
        shared = merge_blocks(shared, _block(root / NATIVE_TELEMETRY_BLOCK))
    dcgm = gpu_power if isinstance(gpu_power, int) else None
    exporters: dict[str, Any] = {}
    if dcgm is not None:
        exporters["dcgm_exporter"] = {"port": dcgm}
    if cpu_power_source is not None:
        exporters["cpu_power_exporter"] = {"source": cpu_power_source}
    if exporters:
        block = _block(root / TELEMETRY_BLOCK) if dcgm is not None else {}
        shared = merge_blocks(shared, merge_blocks(block, {"telemetry": exporters}))
    if "base" in raw:
        return {**raw, "base": merge_blocks(shared, raw["base"])}
    return merge_blocks(shared, raw)


POWER_FLAGS = frozenset({"1", "true", "TRUE", "yes", "YES"})


def _required(environment: Mapping[str, str], name: str) -> str:
    """Return the input ``name`` from ``environment``, or raise ValueError if it is blank."""
    value = environment.get(name, "")
    if not value.strip():
        raise ValueError(f"Missing workload input: {name}")
    return value


def _positive(value: str, name: str) -> int:
    """Return ``value`` as a positive int, or raise ValueError naming ``name``."""
    if not value.isascii() or not value.isdecimal() or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer: {value!r}")
    return int(value)


def point_concurrency(environment: Mapping[str, str]) -> int:
    """Return the point's CONC, or raise ValueError if it is blank or not a positive integer."""
    return _positive(_required(environment, "CONC"), "CONC")


def parse_concurrencies(conc_list: str) -> list[int]:
    """Parse a whitespace-separated CONC_LIST of unique, canonical positive integers."""
    values = []
    for word in conc_list.split():
        if not word.isascii() or not word.isdecimal() or str(int(word)) != word or int(word) <= 0:
            raise ValueError(f"CONC_LIST entries must be canonical positive integers: {word!r}")
        values.append(int(word))
    if not values or len(set(values)) != len(values):
        raise ValueError("concurrencies must be positive unique integers")
    return values


def _master_version(
    environment: Mapping[str, str], component: tuple[str, str], installer: str
) -> str:
    """The master version of the component ``installer`` installs; the point must declare it."""
    field, name = component
    variable = _METADATA_ENV[field]
    metadata = parse_component_metadata(environment.get(variable), variable, version_optional=True)
    if metadata is None or metadata["name"] != name or "version" not in metadata:
        raise ValueError(
            f"{installer} installs {name}, so the master config must declare "
            f"{field} {{name: {name}, version: ...}}; the point has {metadata or 'none'}"
        )
    return metadata["version"]


def _bind_components(bound: dict[str, Any], environment: Mapping[str, str]) -> None:
    """Write the master version of each component a repo script installs where the script
    runs: the top-level environment for setup_script, a service's env for its preamble
    (services do not inherit environment). A SETUP_PIP_PACKAGES vllm-router pin must be the
    master router's."""
    if (script := bound.get("setup_script")) in INSTALLERS:
        version = _master_version(environment, INSTALLERS[script], script)
        bound.setdefault("environment", {})[VERSION_ENV[INSTALLERS[script][0]]] = version
    for service in bound.get("services") or []:
        for script, component in INSTALLERS.items():
            if f"/configs/{script}" in str(service.get("preamble") or ""):
                version = _master_version(environment, component, script)
                service.setdefault("env", {})[VERSION_ENV[component[0]]] = version
    envs = [bound.get("environment"), (bound.get("frontend") or {}).get("env")]
    envs += [(role or {}).get("env") for role in (bound.get("roles") or {}).values()]
    for env in envs:
        for spec in str((env or {}).get("SETUP_PIP_PACKAGES", "")).split():
            package = re.split(r"[^\w.-]", spec, maxsplit=1)[0]
            if re.sub(r"[-_.]+", "-", package).lower() != "vllm-router":
                continue
            pin = f"SETUP_PIP_PACKAGES {spec}"
            version = _master_version(environment, INSTALLERS["vllm-router.sh"], pin)
            if spec != f"{package}=={version}":
                raise ValueError(f"{pin} is not the master router {version}")


def bind_workload(
    recipe: Mapping[str, Any],
    environment: Mapping[str, str],
    *,
    agentic: bool,
    multinode: bool,
    client_env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Write the matrix point's values, the master versions of the components repo scripts
    install, and the launcher's client paths (``client_env``) into a selected variant.

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
        identity["container"]["image"] = image.replace("#", "/", 1)
    benchmark = bound.setdefault("benchmark", {})
    workload = benchmark.setdefault("env", {})
    if not agentic:
        workload.update(
            (name, str(_positive(_required(environment, name), name))) for name in ("ISL", "OSL")
        )
    if multinode:
        concurrencies = parse_concurrencies(_required(environment, "CONC_LIST"))
    else:
        conc = point_concurrency(environment)
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
    workload.update(
        (name, value)
        for name, value in (client_env or {}).items()
        if not (name == "HF_HUB_CACHE" and "HF_HOME" in workload)
    )
    if (bound.get("telemetry") or {}).get("enabled") is True:
        benchmark["concurrencies"] = concurrencies
        required = environment.get("REQUIRE_POWER", "0") in POWER_FLAGS
        # A multi-node point measures power by default; only its power field makes srtctl skip
        # the benchmark when telemetry fails.
        required = required or (multinode and environment.get("POWER") == "1")
        bound["telemetry"]["required"] = required
        if not multinode:
            workload["REQUIRE_POWER"] = "1" if required else "0"
    if agentic and not multinode:
        workload["ENABLE_AGENTX_POWER"] = (
            "1" if (bound.get("telemetry") or {}).get("enabled") is True else "0"
        )
    _bind_components(bound, environment)
    return bound


def bind_multinode(
    recipe: str,
    environment: Mapping[str, str],
    *,
    root: Path,
    expand: Callable[..., list[tuple[str | None, dict[str, Any]]]] = selected_recipes,
    gpu_power: GpuPower | None = None,
    cpu_power_source: CpuPowerSource | None = None,
    client_env: Mapping[str, str] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """Bind the one variant ``recipe`` (``fragment[:selector]``) selects; return its name too."""
    agentic = environment["IS_AGENTIC"] == "1"
    name, selected = select_multinode(
        recipe,
        agentic=agentic,
        root=root,
        expand=expand,
        gpu_power=gpu_power,
        cpu_power_source=cpu_power_source,
    )
    bound = bind_workload(
        selected, environment, agentic=agentic, multinode=True, client_env=client_env
    )
    bind_speculation(bound, environment)
    return name, bound


def select_multinode(
    recipe: str,
    *,
    agentic: bool,
    root: Path,
    expand: Callable[..., list[tuple[str | None, dict[str, Any]]]] = selected_recipes,
    gpu_power: GpuPower | None = None,
    cpu_power_source: CpuPowerSource | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """The one composed, unbound variant ``recipe`` (``fragment[:selector]``) selects."""
    path, _, selector = recipe.partition(":")
    composed = compose_recipe(
        Path(path),
        agentic=agentic,
        multinode=True,
        root=root,
        gpu_power=gpu_power,
        cpu_power_source=cpu_power_source,
    )
    variants = expand(composed, selector or None)
    if len(variants) != 1:
        raise ValueError(f"{recipe} selects {len(variants)} variants, not one")
    return variants[0]


def discovery_on_dedicated_node(recipe: Mapping[str, Any]) -> bool:
    """Whether ``recipe`` runs etcd and NATS on a node of their own.

    srtctl then refuses GPU power telemetry: its samples would leave the benchmark's clock.
    """
    discovery = [
        service
        for service in recipe.get("services") or []
        if service.get("enabled", True) and service.get("type") in ("etcd", "nats")
    ]
    if not discovery:
        return bool((recipe.get("infra") or {}).get("etcd_nats_dedicated_node", False))
    placements = {
        (service.get("placement") or {}).get("node") == "dedicated" for service in discovery
    }
    if len(placements) != 1:
        raise ValueError("Recipe etcd and nats must agree on dedicated placement")
    return placements.pop()


def selects_dedicated_discovery(recipe: str, *, agentic: bool, root: Path) -> bool:
    """Whether the variant ``recipe`` (``fragment[:selector]``) selects runs etcd and NATS on a
    node of their own.

    Expands variants without srtctl, which the launcher's environment does not install.
    """
    _, selected = select_multinode(recipe, agentic=agentic, root=root, expand=expand_variants)
    return discovery_on_dedicated_node(selected)


def bind_speculation(recipe: dict[str, Any], environment: Mapping[str, str]) -> None:
    """Fail unless the matrix's SPEC_DECODING says whether the generation worker drafts, and
    give the client the algorithm and draft model it drafts with."""
    engine = recipe.get("engine")
    engine = ENGINES.get(environment["FRAMEWORK"]) or (
        engine.get("type") if isinstance(engine, Mapping) else engine
    )
    spec = spec_parameters(generation_role(recipe), engine)
    label = environment["SPEC_DECODING"]
    if bool(spec) != (label != "none"):
        raise ValueError(
            f"SPEC_DECODING: matrix {label!r}, but the generation worker "
            f"{'drafts' if spec else 'does not draft'}"
        )
    if spec:
        algorithm, draft_model = speculation_facts(_required(environment, "MODEL_PREFIX"), spec)
        workload = recipe.setdefault("benchmark", {}).setdefault("env", {})
        workload.update(SPEC_ALGORITHM=algorithm, SPEC_DRAFT_MODEL=draft_model)


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


def add_cpu_power_source_argument(parser: argparse.ArgumentParser) -> None:
    """``--cpu-power-source``: the cluster's CPU power exporter source, for a cpu-power point."""
    parser.add_argument(
        "--cpu-power-source",
        choices=get_args(CpuPowerSource),
        help="bind the cluster's CPU power exporter with this source",
    )


def _gpu_power(text: str) -> GpuPower:
    """Parse ``--gpu-power``: a DCGM exporter port, or NATIVE."""
    return NATIVE if text == NATIVE else int(text)


def _assignment(text: str) -> tuple[str, str]:
    """Split a ``NAME=VALUE`` argument, or raise ArgumentTypeError if malformed."""
    name, separator, value = text.partition("=")
    if not separator or not name:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {text!r}")
    return name, value


def main(argv: list[str] | None = None) -> None:
    """Bind the multi-node recipe variant ``argv`` names and write it to the output path."""
    parser = argparse.ArgumentParser(description="Bind one multi-node recipe variant")
    parser.add_argument("recipe", help="fragment[:selector]")
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--gpu-power",
        type=_gpu_power,
        help=f"merge a GPU telemetry block: the DCGM one with its exporter on this port, or "
        f"{NATIVE!r} for the cluster's own exporter",
    )
    add_cpu_power_source_argument(parser)
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
    try:
        root = repository_root()
        _, bound = bind_multinode(
            args.recipe,
            os.environ,
            root=root,
            gpu_power=args.gpu_power,
            cpu_power_source=args.cpu_power_source,
            client_env=dict(args.client_env),
        )
        bound = resolve_fabric(bound, args.fabric)
        check_setup_script(bound, Path(args.recipe.partition(":")[0]), root)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        parser.error(str(error))
    args.output.write_text(yaml.safe_dump(bound, sort_keys=False))


if __name__ == "__main__":
    main()
