"""Bind master-config points to the recipes the launcher submits; ``infx generate`` writes them."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from infx.clusters import load_inventory
from infx.clusters.slurm import Fabric, slurm_settings
from infx.matrix.generate import expand_config_keys, generate_config_matrix
from infx.matrix.validation import config_root, load_config_files, load_runner_file
from infx.srt_slurm.single_node import select_recipe
from infx.srt_slurm.synthetic_acceptance import build_overrides, selected_recipes
from infx.srt_slurm.workload import bind_multinode, dram_budget, resolve_dram, resolve_fabric

if TYPE_CHECKING:
    from infx.clusters import Cluster, RunnerInventory

# What benchmark-tmpl.yml and benchmark-multinode-tmpl.yml export.
RANDOM_RANGE_RATIO = "0.8"
THINKING_MODE = "thinking_on"


def point_environment(point: Mapping[str, Any]) -> dict[str, str]:
    """The workflow environment the launcher reads for ``point``."""
    agentic = point.get("scenario-type") == "agentic-coding"
    environment = {
        "IMAGE": point["image"],
        "MODEL": point["model"],
        "MODEL_PREFIX": point["model-prefix"],
        "PRECISION": point["precision"],
        "FRAMEWORK": point["framework"],
        "SPEC_DECODING": point["spec-decoding"],
        "IS_AGENTIC": "1" if agentic else "0",
        "ISL": "0" if agentic else str(point["isl"]),
        "OSL": "0" if agentic else str(point["osl"]),
        "RANDOM_RANGE_RATIO": RANDOM_RANGE_RATIO,
        "THINKING_MODE": THINKING_MODE,
        "RUN_EVAL": "false",
        "EVAL_ONLY": "false",
    }
    if "prefill" in point:
        prefill = point["prefill"]
        environment.update(
            IS_MULTINODE="true",
            SRT_RECIPE=point["srt-recipe"],
            POWER="1" if point.get("power") else "0",
            CONC_LIST=" ".join(map(str, point["conc"])),
            PREFILL_TP=str(prefill["tp"]),
            PREFILL_PP_SIZE=str(prefill["pp"]),
            PREFILL_PCP_SIZE=str(prefill["pcp-size"]),
            KV_OFFLOADING=str(point["kv-offloading"]) if agentic else "",
            TOTAL_CPU_DRAM_GB=str(point["total-cpu-dram-gb"]) if agentic else "",
        )
        if agentic:
            # A multi-node AgentX job serves its one concurrency.
            environment["CONC"] = str(point["conc"][0])
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
        "KV_OFFLOADING": str(point["kv-offloading"]) if agentic else "",
        "TOTAL_CPU_DRAM_GB": str(point["total-cpu-dram-gb"]) if agentic else "0",
    }


def _placement(inventory: RunnerInventory, label: str) -> tuple[str, Cluster] | None:
    """A runner of ``label`` and its cluster, when every runner of ``label`` is on one cluster."""
    runners = inventory.labels.get(label, [])
    if len({inventory.cluster_for(runner).id for runner in runners}) != 1:
        return None
    return runners[0], inventory.cluster_for(runners[0])


def _binder_inputs(
    point: Mapping[str, Any],
    environment: Mapping[str, str],
    placement: tuple[str, Cluster] | None,
    root: Path,
) -> tuple[int | None, dict[str, str]]:
    """The DCGM exporter port and AgentX client paths the launcher binds a multi-node point with.

    Raises ``ValueError`` when they depend on a cluster the runner label does not single out,
    or the cluster's lanes refuse the point.
    """
    agentic = environment["IS_AGENTIC"] == "1"
    if "prefill" not in point or not (agentic or point.get("power")):
        return None, {}
    if placement is None:
        raise ValueError(
            f"runner {point['runner']!r} must schedule on one cluster to bind the cluster's "
            "power telemetry and AgentX client paths"
        )
    from infx.launch.context import LaunchError
    from infx.launch.drivers.srt import config, lanes, power
    from infx.launch.policy import launch_path
    from infx.launch.request import MultiNodeRequest

    runner, cluster = placement
    srt = slurm_settings(cluster).srt_slurm
    if srt is None:
        raise ValueError(f"cluster {cluster.id!r} has no slurm.srt-slurm settings")
    # The job reads these; the binder inputs depend on none of them.
    runtime = {
        "RUNNER_NAME": runner,
        "GITHUB_WORKSPACE": str(root),
        "RESULT_FILENAME": point["exp-name"],
    }
    request = MultiNodeRequest.from_env({**environment, **runtime})
    try:
        path = launch_path(cluster.id, request)
        lane = lanes.srt_lane(cluster.id, path)
        decision = power.decide_power(cluster.id, path, request)
        client_env = config.agentic_client_env(cluster, srt, lane, request) if agentic else {}
    except LaunchError as error:
        raise ValueError(f"cluster {cluster.id!r}: {error}") from error
    return (srt.power_exporter_port if decision.dcgm else None), client_env


def bound_variant(
    point: Mapping[str, Any],
    environment: Mapping[str, str],
    root: Path,
    *,
    expand: Callable[..., list[tuple[str | None, dict[str, Any]]]] = selected_recipes,
    power_port: int | None = None,
    client_env: Mapping[str, str] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """The variant the launcher submits for ``point``, composed and bound.

    Multi-node variants get the DCGM telemetry block on ``power_port`` and the client paths
    ``client_env``, the binder inputs the launcher computes. ``'@fabric.<name>'`` values stay
    unresolved: recipe fingerprints hash them as written.
    """
    if "prefill" not in point:
        selected, recipe = select_recipe(
            str(root / point["srt-recipe"]), environment, root=root, expand=expand
        )
        return selected.partition(":")[2] or None, recipe
    return bind_multinode(
        str(root / point["srt-recipe"]),
        environment,
        root=root,
        expand=expand,
        power_port=power_port,
        client_env=client_env,
    )


def _cluster_fabric(cluster: Cluster) -> dict[str, str | None]:
    """``cluster``'s fabric as recipes read it."""
    srt = slurm_settings(cluster).srt_slurm
    return (srt.fabric if srt is not None else Fabric()).rendered()


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
    """Bind every srt-slurm point of ``config_keys``; write recipes and a manifest.

    A multi-node point is bound with its cluster's DCGM exporter port and AgentX client
    paths, a DRAM point's ``'@dram.<name>'`` values take its budget over the GPUs it covers
    there, and ``'@fabric.<name>'`` values take its cluster's facts; the manifest names that
    cluster, the one its runner label schedules on. A label on several clusters leaves fabric
    references as written and the manifest's ``cluster`` null.
    """
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
    inventory = load_inventory(runner_file)
    recipes: dict[str, dict[str, Any]] = {}
    records = []
    for key in expand_config_keys(config_keys, master):
        points = generate_config_matrix([key], master, runners, eval_mode="none", root=root)
        points = [point for point in points if point.get("srt-recipe")]
        if not points:
            raise ValueError(f"{key} has no srt-slurm points")
        for point in points:
            environment = point_environment(point)
            placement = _placement(inventory, point["runner"])
            power_port, client_env = _binder_inputs(point, environment, placement, root)
            fabric = _cluster_fabric(placement[1]) if placement else None
            variant, recipe = bound_variant(
                point, environment, root, power_port=power_port, client_env=client_env
            )
            budget = dram_budget(
                environment,
                multinode="prefill" in point,
                gpus_per_node=placement[1].gpus_per_node if placement else None,
            )
            recipe = resolve_dram(recipe, budget)
            if fabric is not None:
                recipe = resolve_fabric(recipe, fabric)
            _apply_acceptance_and_validate(recipe, environment, point["srt-recipe"])
            identity = json.dumps({"point": point, "variant": variant}, sort_keys=True)
            digest = hashlib.sha256(identity.encode()).hexdigest()[:12]
            name = f"{re.sub(r'[^A-Za-z0-9_.-]', '_', key)}-{digest}.yaml"
            recipes[name] = recipe
            records.append({
                "file": name, "config-key": key, "variant": variant,
                "cluster": placement[1].id if placement else None, "matrix": point,
            })  # fmt: skip
    output.mkdir(parents=True, exist_ok=True)
    for name, recipe in recipes.items():
        (output / name).write_text(yaml.safe_dump(recipe, sort_keys=False))
    manifest = {"recipes": records}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
