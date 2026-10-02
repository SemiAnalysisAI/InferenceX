"""Bind every planned srt-slurm point to the recipe variants it would launch, before dispatch.

Sweep entry points pipe their matrix through this check after ``infx.workflows.benchmark_schema``.
A single-node point must select exactly one variant through the runtime's ``select_recipe``,
which compares the variant's ``model.container`` with the point's image. A multi-node point must
set ``CONFIG_FILE``, unless it is eval-only and sets ``EVAL_CONFIG_FILE``, the recipe the launcher
then selects. Every multi-node recipe a point can launch must select at least one variant, and
each variant's worker containers must resolve to an image the launcher stages for the job: the
point's image, or ``PREFILL_IMAGE`` for a TileRT prefill role. A TileRT frontend or client must
run one of those images, and any other benchmark client that names another tag of the point's
image is stale, because srtctl would pull that literal instead. The matrix is echoed unchanged on
success; any problem exits non-zero so no benchmark job is dispatched.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml

from infx.clusters import CLUSTER_LABEL_PREFIX, RunnerInventory, load_inventory
from infx.clusters.slurm import slurm_settings
from infx.launch.drivers.srt.config import pyxis_spelling
from infx.launch.drivers.srt.recipe import RECIPES_MIRROR, recipe_mirror_path
from infx.srt_slurm.single_node import select_recipe
from infx.srt_slurm.synthetic_acceptance import selected_recipes

E2E_ROOT = Path(__file__).resolve().parents[2]
SINGLE_NODE_RECIPES = Path("benchmarks/single_node/srt-slurm-recipes")
RECIPE_SETTINGS = ("CONFIG_FILE", "EVAL_CONFIG_FILE")


def matrix_points(matrix: Any) -> Iterator[Mapping[str, Any]]:
    """Every benchmark point in a sweep plan or a flat generated matrix."""
    if isinstance(matrix, Mapping):
        if "image" in matrix and ("srt-recipe" in matrix or "prefill" in matrix):
            yield matrix
            return
        for value in matrix.values():
            yield from matrix_points(value)
    elif isinstance(matrix, list):
        for item in matrix:
            yield from matrix_points(item)


def workflow_text(value: Any) -> str:
    """A matrix value as a GitHub expression renders it into an environment variable."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def single_node_environment(point: Mapping[str, Any]) -> dict[str, str]:
    """The recipe-binding inputs ``benchmark-tmpl.yml`` exports for a single-node point."""
    agentic = point.get("scenario-type") == "agentic-coding"
    tp, pp, pcp = (int(point.get(name, 1)) for name in ("tp", "pp", "pcp-size"))
    return {
        "FRAMEWORK": str(point["framework"]),
        "MODEL": str(point["model"]),
        "IMAGE": str(point["image"]),
        "PRECISION": str(point["precision"]),
        "TP": str(tp),
        "PP_SIZE": str(pp),
        "DCP_SIZE": workflow_text(point.get("dcp-size", 1)),
        "PCP_SIZE": str(pcp),
        "EP_SIZE": workflow_text(point["ep"]),
        "DP_ATTENTION": workflow_text(point.get("dp-attn", False)),
        "CONC": workflow_text(point["conc"]),
        "SPEC_DECODING": str(point["spec-decoding"]),
        "GPU_COUNT": str(tp * pp * pcp),
        "IS_AGENTIC": "1" if agentic else "0",
        "KV_OFFLOADING": workflow_text(point.get("kv-offloading")) if agentic else "",
        "TOTAL_CPU_DRAM_GB": workflow_text(point.get("total-cpu-dram-gb")) if agentic else "0",
        "ISL": "0" if agentic else workflow_text(point["isl"]),
        "OSL": "0" if agentic else workflow_text(point["osl"]),
        "RANDOM_RANGE_RATIO": "0.8",
    }


def docker_spelling(image: str) -> str:
    """``registry/path`` for a pyxis ``registry#path`` image, else ``image``."""
    host, separator, path = image.partition("#")
    return f"{host}/{path}" if separator else image


def repository(image: str) -> str:
    """``image`` without its tag or digest, in ``registry/path`` spelling."""
    head, _, last = docker_spelling(image).partition("@")[0].rpartition("/")
    name = last.partition(":")[0]
    return f"{head}/{name}" if head else name


def point_label(point: Mapping[str, Any]) -> str:
    """The point as its benchmark job is named, with the eval suffix the job title carries."""
    suffix = (
        " | eval-only" if point.get("eval-only") else " | eval" if point.get("run-eval") else ""
    )
    return f"{point.get('exp-name', '<unnamed>')} on {point.get('runner', '<no runner>')}{suffix}"


def container_fields(config: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """The ``(field, value)`` pairs naming the images a variant's workers, frontend and client run."""
    fields: list[tuple[str, Any]] = [
        ("model.container", (config.get("model") or {}).get("container"))
    ]
    for role, spec in (config.get("roles") or {}).items():
        if isinstance(spec, Mapping) and "container" in spec:
            fields.append((f"roles.{role}.container", spec["container"]))
    for block in ("frontend", "benchmark"):
        section = config.get(block)
        if isinstance(section, Mapping) and "container_image" in section:
            fields.append((f"{block}.container_image", section["container_image"]))
    return fields


def container_problems(
    config: Mapping[str, Any], image: str, aliases: frozenset[str], prefill: str | None
) -> list[str]:
    """Containers of one variant that would not run the images the launcher stages for it.

    Workers run the point's image (directly or through a cluster alias), except a TileRT prefill
    role, which must name ``PREFILL_IMAGE``. A TileRT frontend and client run one of those images.
    Other frontends may pin an image of their own, but a benchmark client that names another tag
    of the point's image is stale.
    """
    resolved = {image, pyxis_spelling(image), *aliases}
    fields = container_fields(config)
    problems = []
    if prefill is not None and all(field != "roles.prefill.container" for field, _ in fields):
        # srtctl runs a role without its own container on model.container.
        problems.append(f"roles.prefill.container is not set, so prefill would not run {prefill}")
    for field, value in fields:
        if value is None:
            if field == "model.container":
                problems.append(f"{field} is not set")
        elif field == "roles.prefill.container" and prefill is not None:
            if value != prefill:
                problems.append(f"{field} {value!r} is not PREFILL_IMAGE {prefill}")
        elif field == "model.container" or field.startswith("roles."):
            if value not in resolved:
                known = ", ".join(sorted(aliases)) or "none"
                problems.append(
                    f"{field} {value!r} does not resolve to {image} (container aliases: {known})"
                )
        elif prefill is not None:
            if value not in resolved and value != prefill:
                problems.append(f"{field} {value!r} is neither {image} nor PREFILL_IMAGE {prefill}")
        elif (
            field == "benchmark.container_image"
            and value not in resolved
            and repository(str(value)) == repository(image)
        ):
            problems.append(
                f"{field} {value!r} is another tag of {image}; "
                "srtctl would pull it instead of the staged image"
            )
    return problems


def check_single_node(point: Mapping[str, Any], root: Path) -> list[str]:
    reference = str(point["srt-recipe"])
    path, _, selector = reference.partition(":")
    recipe = (root / path).resolve()
    if not recipe.is_relative_to((root / SINGLE_NODE_RECIPES).resolve()):
        return [f"srt-recipe={reference}: not inside {SINGLE_NODE_RECIPES}"]
    config = str(recipe) + (f":{selector}" if selector else "")
    try:
        _, variant = select_recipe(config, single_node_environment(point))
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        return [f"srt-recipe={reference}: {exc}"]
    problems = container_problems(variant, str(point["image"]), frozenset(), None)
    return [f"srt-recipe={reference}: {problem}" for problem in problems]


def point_settings(point: Mapping[str, Any]) -> dict[str, str]:
    """NAME=value additional-settings in the order ``benchmark-multinode-tmpl.yml`` exports them."""
    settings: dict[str, str] = {}
    for role in ("prefill", "decode"):
        for setting in (point.get(role) or {}).get("additional-settings") or []:
            name, _, value = str(setting).partition("=")
            settings[name] = value
    return settings


def container_aliases(inventory: RunnerInventory, runner: str) -> frozenset[str]:
    """Aliases the srt-slurm config maps to the job image on every cluster ``runner`` reaches."""
    if runner.startswith(CLUSTER_LABEL_PREFIX):
        cluster_ids = {runner.removeprefix(CLUSTER_LABEL_PREFIX)}
    elif runner in inventory.labels:
        cluster_ids = {inventory.cluster_for(name).id for name in inventory.labels[runner]}
    else:
        # Some master entries schedule on a bare cluster id rather than a runners.yaml label.
        cluster_ids = {runner}
    aliases: frozenset[str] | None = None
    for cluster_id in sorted(cluster_ids):
        cluster = inventory.clusters.get(cluster_id)
        srt = slurm_settings(cluster).srt_slurm if cluster is not None else None
        names = frozenset(srt.container_aliases) if srt is not None else frozenset()
        aliases = names if aliases is None else aliases & names
    return aliases or frozenset()


def check_multi_node(point: Mapping[str, Any], root: Path, inventory: RunnerInventory) -> list[str]:
    image = str(point["image"])
    aliases = container_aliases(inventory, str(point.get("runner", "")))
    settings = point_settings(point)
    prefill = None
    if point.get("framework") == "tilert":
        prefill = settings.get("PREFILL_IMAGE") or None
        if prefill is None:
            return ["TileRT needs a PREFILL_IMAGE setting for its prefill role"]
    problems = []
    if not settings.get("CONFIG_FILE") and not (
        point.get("eval-only") and settings.get("EVAL_CONFIG_FILE")
    ):
        problems.append(
            "CONFIG_FILE is not set; only an eval-only point may launch its EVAL_CONFIG_FILE instead"
        )
    mirror = (root / RECIPES_MIRROR).resolve()
    for name in RECIPE_SETTINGS:
        reference = settings.get(name)
        if not reference:
            continue
        path = recipe_mirror_path(root, reference).resolve()
        if not reference.startswith("recipes/") or not path.is_relative_to(mirror):
            problems.append(f"{name}={reference}: not a recipes/ path inside {RECIPES_MIRROR}")
            continue
        selector = reference.partition(":")[2] or None
        try:
            recipe = yaml.safe_load(path.read_text())
            if not isinstance(recipe, Mapping):
                problems.append(f"{name}={reference}: recipe is not a mapping")
                continue
            variants = selected_recipes(recipe, selector)
        except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            problems.append(f"{name}={reference}: {exc}")
            continue
        if not variants:
            problems.append(
                f"{name}={reference}: selects no variant, so srtctl would submit nothing"
            )
        for variant, config in variants:
            label = f"{name}={reference}" + ("" if variant in (None, selector) else f" ({variant})")
            found = container_problems(config, image, aliases, prefill)
            declared = ((config.get("identity") or {}).get("container") or {}).get("image")
            if declared is not None and docker_spelling(str(declared)) != docker_spelling(image):
                found.append(f"identity.container.image {declared!r} is not {image}")
            problems.extend(f"{label}: {problem}" for problem in found)
    return problems


def check_matrix(
    matrix: Any, root: Path, inventory: RunnerInventory | None
) -> dict[str, list[str]]:
    """Each problem found, mapped to the points it affects."""
    problems: dict[str, list[str]] = {}
    for point in matrix_points(matrix):
        if "prefill" in point:
            found = (
                check_multi_node(point, root, inventory)
                if inventory is not None
                else ["multi-node points need a runner inventory"]
            )
        elif point.get("srt-recipe"):
            found = check_single_node(point, root)
        else:
            continue
        label = point_label(point)
        for problem in found:
            points = problems.setdefault(problem, [])
            if label not in points:
                points.append(label)
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=E2E_ROOT,
        help="inferencex-e2e tree whose recipes the matrix names",
    )
    parser.add_argument(
        "--runner-config", type=Path, help="runner inventory (default: ROOT/configs/runners.yaml)"
    )
    args = parser.parse_args()
    raw = sys.stdin.read()
    matrix = json.loads(raw)
    inventory = None
    if any("prefill" in point for point in matrix_points(matrix)):
        runner_config = args.runner_config or args.root / "configs/runners.yaml"
        try:
            inventory = load_inventory(runner_config)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            print(
                f"srt-slurm recipe preflight cannot read {runner_config} "
                f"with this tooling's runner schema: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)
    problems = check_matrix(matrix, args.root, inventory)
    if problems:
        affected = len({label for points in problems.values() for label in points})
        print(
            f"srt-slurm recipe preflight found {len(problems)} problem(s) affecting {affected} point(s):",
            file=sys.stderr,
        )
        for problem, points in problems.items():
            print(f"  {problem}", file=sys.stderr)
            for label in points:
                print(f"    - {label}", file=sys.stderr)
        sys.exit(1)
    sys.stdout.write(raw)


if __name__ == "__main__":
    main()
