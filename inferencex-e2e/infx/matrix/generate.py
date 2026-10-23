import argparse
import fnmatch
import json
import math
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from infx import yamlio
from infx.clusters import CLUSTER_LABEL_PREFIX
from infx.clusters.slurm import Fabric
from infx.srt_slurm.variants import deep_merge, expand_variants
from infx.srt_slurm.workload import resolve_fabric

from .validation import (
    DEFAULT_AGENTIC_DURATION_SECONDS,
    Fields,
    config_root,
    load_config_files,
    load_runner_file,
    srt_recipe_path,
    validate_agentic_matrix_entry,
    validate_matrix_entry,
)

seq_len_stoi = {"1k1k": (1024, 1024), "8k1k": (8192, 1024)}

MIN_EVAL_CONC = 16
DEFAULT_EVAL_FRAMEWORK = "lm-eval"
AUTOMATIC_AGENTIC_VENDOR_EVALS = {
    "kimik3": ("kimi-vendor", "kimi_tool_call_schema_full"),
    "minimaxm3": ("minimax-vendor", "minimax_m3_full"),
}
BYTES_PER_MIB = 1024 * 1024
BYTES_PER_GB = 1_000_000_000
MAX_AGENTIC_AVAILABLE_CPU_DRAM_MIB = 2_861_022

seq_len_itos = {v: k for k, v in seq_len_stoi.items()}


def seq_len_to_str(isl: int, osl: int) -> str:
    """Return the short name of an ISL/OSL pair, such as 1k1k, or ``{isl}_{osl}``."""
    return seq_len_itos.get((isl, osl), f"{isl}_{osl}")


def freeze_config_value(value: Any) -> Any:
    """Return ``value`` with dicts and lists turned into sorted tuples so it can be hashed."""
    if isinstance(value, dict):
        return tuple(sorted((key, freeze_config_value(item)) for key, item in value.items()))
    if isinstance(value, list):
        return tuple(freeze_config_value(item) for item in value)
    return value


def trim_conc(entries: list[dict]) -> list[dict]:
    """Keep only the lowest-concurrency row of each otherwise identical group of rows.

    Multinode rows shrink to their lowest conc, and the kept row inherits the group's eval flags.
    The recipe fingerprint is ignored since single-node bundles bind one variant per conc.
    """
    ignored_fields = {
        "conc",
        "exp-name",
        Fields.RECIPE_FINGERPRINT.value,
        "run-eval",
        "eval-only",
        "eval-conc",
        "eval-all-concs",
        Fields.EVAL_FRAMEWORK.value,
        Fields.EVAL_SUITE.value,
    }
    groups: dict[tuple, list[int]] = {}
    out: list[dict] = []

    def minimum_concurrency(entry: dict) -> int:
        """Return the lowest concurrency of a row whose conc is an int or a list."""
        conc = entry["conc"]
        return min(conc) if isinstance(conc, list) else conc

    for source_entry in entries:
        entry = source_entry
        conc = entry.get("conc")
        if entry.get("prefill") is not None and isinstance(conc, list) and conc:
            minimum_conc = min(conc)
            if len(conc) > 1 or entry.get("eval-conc") != minimum_conc:
                entry = {**entry, "conc": [minimum_conc]}
                if "eval-conc" in entry:
                    entry["eval-conc"] = minimum_conc

        key = tuple(
            sorted(
                (key, freeze_config_value(value))
                for key, value in entry.items()
                if key not in ignored_fields
            )
        )
        if entry.get("eval-only"):
            key += (
                (
                    "eval-only",
                    entry.get(Fields.EVAL_FRAMEWORK.value),
                    entry.get(Fields.EVAL_SUITE.value),
                ),
            )
        groups.setdefault(key, []).append(len(out))
        out.append(entry)

    drop: set[int] = set()
    for indices in groups.values():
        keep = min(indices, key=lambda index: minimum_concurrency(out[index]))
        kept_entry = out[keep]
        eval_rows = [out[index] for index in indices if out[index].get("run-eval") is True]
        if eval_rows:
            kept_entry = {**kept_entry, "run-eval": True}
            if not kept_entry.get(Fields.EVAL_FRAMEWORK.value):
                for field in (Fields.EVAL_FRAMEWORK.value, Fields.EVAL_SUITE.value):
                    if field in eval_rows[0]:
                        kept_entry[field] = eval_rows[0][field]
            if kept_entry.get("prefill") is not None:
                kept_entry["eval-conc"] = minimum_concurrency(kept_entry)
            if any(out[index].get("eval-all-concs") is True for index in indices):
                kept_entry["eval-all-concs"] = True
            out[keep] = kept_entry
        drop.update(index for index in indices if index != keep)
    return [entry for index, entry in enumerate(out) if index not in drop]


def smoke_entries(entries: list[dict]) -> list[dict]:
    """Return trimmed benchmark rows with evals off, followed by trimmed eval-only rows."""
    benchmarks = trim_conc(
        [{**row, "run-eval": False} for row in entries if not row.get(Fields.EVAL_ONLY.value)]
    )
    evals = []
    for row in entries:
        if not row.get("run-eval"):
            continue
        row = {**row, "eval-only": True}
        row.pop(Fields.REQUIRE_POWER.value, None)
        if row.get("prefill") is not None:
            row["conc"] = [row["eval-conc"]]
            row.pop("eval-all-concs", None)
        evals.append(row)
    return benchmarks + trim_conc(evals)


def runner_labels(runner_data: dict) -> dict:
    """Return the runner config's map of runner labels to node names."""
    return runner_data["labels"]


def runner_hardware(runner_data: dict) -> dict:
    """Return GPU and CPU DRAM facts keyed by cluster label, for clusters that declare them."""
    node_fields = (Fields.GPUS_PER_NODE.value, Fields.AVAILABLE_CPU_DRAM_MIB.value)
    return {
        f"{CLUSTER_LABEL_PREFIX}{cluster_id}": {
            field: cluster[field] for field in node_fields if field in cluster
        }
        for cluster_id, cluster in runner_data.get("clusters", {}).items()
    }


def runner_nodes_for_label(runner: str, runner_data: dict) -> list[str]:
    """Return the node names of runner label ``runner``, or an empty list."""
    return runner_labels(runner_data).get(runner, [])


def runner_hardware_int(runner: str, runner_data: dict, field: str) -> int:
    """Return the integer hardware ``field`` of ``runner``, or raise ValueError if it is unset."""
    hardware = runner_hardware(runner_data).get(runner, {})
    value = hardware.get(field)
    if value is None:
        available = ", ".join(sorted(runner_hardware(runner_data).keys()))
        raise ValueError(
            f"Runner '{runner}' requires '{field}' "
            f"in runner hardware metadata. Available hardware keys: {available}"
        )
    return value


def runner_available_cpu_dram_mib(runner: str, runner_data: dict) -> int:
    """Return the CPU DRAM in MiB available to ``runner``."""
    return runner_hardware_int(runner, runner_data, Fields.AVAILABLE_CPU_DRAM_MIB.value)


def runner_gpus_per_node(runner: str, runner_data: dict) -> int:
    """Return the GPUs per node of ``runner``."""
    return runner_hardware_int(runner, runner_data, Fields.GPUS_PER_NODE.value)


def _hardware_family(label: str) -> str:
    """Return the hardware family of a label, such as b200 for cluster:b200-dgxc."""
    return label.removeprefix("cluster:").split("-", 1)[0]


def scheduling_gpus_per_node(label: str, runner_data: dict) -> int:
    """Return the GPUs per node of ``label``, or of its hardware family when that is unambiguous.

    Raise ValueError when no cluster or more than one GPU count matches.
    """
    hardware = runner_hardware(runner_data)
    exact = hardware.get(label)
    if exact is not None:
        return exact[Fields.GPUS_PER_NODE.value]

    family = _hardware_family(label)
    matches = {
        facts[Fields.GPUS_PER_NODE.value]
        for hardware_label, facts in hardware.items()
        if _hardware_family(hardware_label) == family
    }
    if len(matches) == 1:
        return matches.pop()
    if not matches:
        raise ValueError(f"Cannot resolve {Fields.GPUS_PER_NODE.value} for '{label}'")
    raise ValueError(f"Ambiguous {Fields.GPUS_PER_NODE.value} for '{label}': {sorted(matches)}")


def _worker_node_override(worker: dict, setting_name: str) -> int | None:
    """Return the positive ``setting_name=N`` value in a worker's additional settings, or None.

    Raise ValueError when the settings disagree or are not positive.
    """
    pattern = re.compile(rf"^{re.escape(setting_name)}=(\d+)$")
    values = []
    for setting in worker.get(Fields.ADDITIONAL_SETTINGS.value, []) or []:
        match = pattern.match(setting)
        if match:
            values.append(int(match.group(1)))
    if not values:
        return None
    if len(set(values)) != 1 or values[0] <= 0:
        raise ValueError(f"Conflicting or invalid {setting_name} settings: {values}")
    return values[0]


def recipe_auxiliary_node_count(recipe: dict) -> int:
    """Return the nodes a recipe needs beyond its workers, for service pools and dedicated roles."""
    services = [service for service in recipe.get("services") or [] if service.get("enabled", True)]
    pool_nodes = sum(service.get("nodes") or 0 for service in services)
    infra_dedicated = (recipe.get("infra") or {}).get("etcd_nats_dedicated_node", False)
    discovery = [service for service in services if service.get("type") in ("etcd", "nats")]
    if discovery:
        placements = {
            (service.get("placement") or {}).get("node") == "dedicated" for service in discovery
        }
        if len(placements) != 1:
            raise ValueError("Recipe etcd and nats must agree on dedicated placement")
        infra_dedicated = placements.pop()
    dedicated_roles = int(infra_dedicated)
    for section, legacy_key in (
        ("frontend", "dedicated_node"),
        ("benchmark", "client_dedicated_node"),
    ):
        settings = recipe.get(section) or {}
        dedicated_roles += int(
            (settings.get("placement") or {}).get("node") == "dedicated"
            or settings.get(legacy_key, False)
        )
    if dedicated_roles and (recipe.get("benchmark") or {}).get("colocate_with_frontend", True):
        dedicated_roles = 1
    return pool_nodes + dedicated_roles


def recipe_node_count(root: Path, srt_recipe: str) -> int | None:
    """Return the total node count of a schema 2 srt recipe, or None for an unknown variant.

    Raise ValueError when the recipe is not schema 2 or its roles lack node counts.
    """
    path, _, selector = srt_recipe.partition(":")
    recipe_path = root / path
    recipe = yamlio.safe_load(recipe_path.read_text())
    if isinstance(recipe, dict) and "base" in recipe:
        if selector.startswith("zip_override_"):
            [(_, recipe)] = expand_variants(recipe, selector)
        elif selector == "base" or selector.startswith("override_"):
            schema = recipe.get("schema")
            recipe = deep_merge(recipe["base"], recipe[selector] or {})
            recipe.setdefault("schema", schema)
        else:
            return None
        if recipe.get("schema") != 2:
            return None
    if not isinstance(recipe, dict) or recipe.get("schema") != 2:
        raise ValueError(f"srt-slurm recipes must declare schema: 2: {recipe_path}")
    roles = recipe.get("roles")
    if roles:
        for name, role in roles.items():
            if not isinstance(role, dict) or role.get("nodes") is None:
                raise ValueError(f"Recipe role {name!r} must specify nodes: {recipe_path}")
        worker_nodes = sum(
            0 if role["nodes"] == "colocate" else int(role["nodes"]) for role in roles.values()
        )
        return worker_nodes + recipe_auxiliary_node_count(recipe)
    raise ValueError(f"Recipe has no worker roles: {recipe_path}")


def _label_fabrics(label: str, runner_data: dict) -> dict[str, Fabric]:
    """Return the srt-slurm fabric of each cluster that runs ``label``, keyed by cluster ID."""
    labels = runner_labels(runner_data)
    runners = set(labels.get(label, [label]))
    fabrics = {}
    for cluster_label, members in labels.items():
        if cluster_label.startswith(CLUSTER_LABEL_PREFIX) and runners & set(members):
            cluster_id = cluster_label.removeprefix(CLUSTER_LABEL_PREFIX)
            record = runner_data["clusters"][cluster_id]
            srt = (record.get("slurm") or {}).get("srt-slurm") or {}
            fabrics[cluster_id] = Fabric.model_validate(srt.get("fabric") or {})
    return fabrics


def check_fabric_references(root: Path, srt_recipe: str, label: str, runner_data: dict) -> None:
    """Raise ValueError if a recipe variant references a fabric field ``label``'s clusters lack."""
    path, _, selector = srt_recipe.partition(":")
    variants = expand_variants(yamlio.safe_load((root / path).read_text()), selector or None)
    fabrics = {
        f"cluster {cluster_id!r}": fabric.rendered()
        for cluster_id, fabric in _label_fabrics(label, runner_data).items()
    }
    fabrics = fabrics or {f"runner {label!r}": dict.fromkeys(Fabric().rendered(), "")}
    for reach, fabric in fabrics.items():
        for _, recipe in variants:
            try:
                resolve_fabric(recipe, fabric)
            except ValueError as error:
                raise ValueError(f"{srt_recipe}: {error} ({reach})") from None


def worker_node_count(
    worker: dict,
    role: str,
    runner: str,
    runner_data: dict,
) -> int:
    """Return the nodes a worker role needs, from a ``<ROLE>_NODES`` setting or its GPU count."""
    override = _worker_node_override(worker, f"{role.upper()}_NODES")
    if override is not None:
        return override

    hardware_label = worker.get(Fields.HARDWARE.value) or runner
    gpus_per_node = scheduling_gpus_per_node(hardware_label, runner_data)
    total_gpus = (
        worker[Fields.NUM_WORKER.value]
        * worker[Fields.TP.value]
        * worker.get(Fields.PP.value, 1)
        * worker.get(Fields.PCP_SIZE.value, 1)
        * worker.get(Fields.DWDP_SIZE.value, 1)
    )
    return math.ceil(total_gpus / gpus_per_node)


def multinode_node_count(
    prefill: dict,
    decode: dict,
    runner: str,
    runner_data: dict,
    srt_recipe: str | None,
    root: Path | None,
) -> int:
    """Return a multinode entry's node count from its srt recipe, else from its workers."""
    if srt_recipe is not None and root is None:
        raise ValueError(f"Reading {srt_recipe} needs the master config's project root")
    recipe_count = recipe_node_count(root, srt_recipe) if srt_recipe is not None else None
    if recipe_count is not None:
        return recipe_count
    return worker_node_count(prefill, "prefill", runner, runner_data) + worker_node_count(
        decode, "decode", runner, runner_data
    )


def add_multinode_node_count(
    entry: dict,
    runner_data: dict,
    num_nodes: int | None,
    root: Path | None,
) -> dict:
    """Set the entry's node count from ``num_nodes`` or its workers, and return the entry.

    Raise ValueError when ``num_nodes`` is given for a disaggregated entry.
    """
    if not entry[Fields.DISAGG.value] and num_nodes is not None:
        entry[Fields.NODE_COUNT.value] = num_nodes
    elif num_nodes is not None:
        raise ValueError(f"{Fields.NUM_NODES.value} is not valid for disaggregated entries")
    else:
        entry[Fields.NODE_COUNT.value] = multinode_node_count(
            entry[Fields.PREFILL.value],
            entry[Fields.DECODE.value],
            entry[Fields.RUNNER.value],
            runner_data,
            entry.get(Fields.SRT_RECIPE.value),
            root,
        )
    return entry


def effective_gpu_count(benchmark: dict) -> int:
    """Return the GPUs one benchmark worker uses, TP times PP times PCP."""
    return (
        benchmark[Fields.TP.value]
        * benchmark.get(Fields.PP.value, 1)
        * benchmark.get(Fields.PCP_SIZE.value, 1)
    )


def with_worker_parallelism_defaults(worker: dict) -> dict:
    """Return a copy of the worker with PP, DCP, and PCP defaulting to 1."""
    return {
        **worker,
        Fields.PP.value: worker.get(Fields.PP.value, 1),
        Fields.DCP_SIZE.value: worker.get(Fields.DCP_SIZE.value, 1),
        Fields.PCP_SIZE.value: worker.get(Fields.PCP_SIZE.value, 1),
    }


def multinode_worker_pair(benchmark: dict, disagg: bool) -> tuple[dict, dict]:
    """Return the prefill and decode workers, splitting an aggregated worker into a pair."""
    if disagg:
        return (
            with_worker_parallelism_defaults(benchmark[Fields.PREFILL.value]),
            with_worker_parallelism_defaults(benchmark[Fields.DECODE.value]),
        )

    worker = with_worker_parallelism_defaults(benchmark[Fields.WORKER.value])
    prefill = {Fields.NUM_WORKER.value: 1, **worker}
    decode = {
        Fields.NUM_WORKER.value: 0,
        **{
            key: value
            for key, value in worker.items()
            if key
            not in (
                Fields.NUM_WORKER.value,
                Fields.ADDITIONAL_SETTINGS.value,
            )
        },
    }
    return prefill, decode


def worker_gpus_per_node(worker: dict, gpus_per_node: int, role: str = "worker") -> int:
    """Return how many of a worker's GPUs land on each node.

    Raise ValueError when its replica size and ``gpus_per_node`` do not divide one another.
    """
    gpus_per_replica = (
        worker[Fields.TP.value]
        * worker.get(Fields.PP.value, 1)
        * worker.get(Fields.PCP_SIZE.value, 1)
        * worker.get(Fields.DWDP_SIZE.value, 1)
    )
    factors = f"{Fields.TP.value}*{Fields.PP.value}*{Fields.PCP_SIZE.value}"
    if worker.get(Fields.DWDP_SIZE.value, 1) > 1:
        factors += f"*{Fields.DWDP_SIZE.value}"
    if gpus_per_replica > gpus_per_node:
        if gpus_per_replica % gpus_per_node != 0:
            raise ValueError(
                f"{role} {factors}"
                f"={gpus_per_replica} spans multiple nodes but is not a multiple "
                f"of {Fields.GPUS_PER_NODE.value}={gpus_per_node}"
            )
        return gpus_per_node
    if gpus_per_node % gpus_per_replica != 0:
        raise ValueError(
            f"{role} {factors}"
            f"={gpus_per_replica} does not divide "
            f"{Fields.GPUS_PER_NODE.value}={gpus_per_node} evenly"
        )
    return gpus_per_replica


def agentic_dram_offload_gb(
    agentic_config: dict, benchmark: dict, runner: str, runner_data: dict
) -> tuple[int, int]:
    """Return the CPU DRAM offload in GB for the prefill and decode sides of an agentic point.

    Both are 0 without DRAM offload. Raise ValueError when the workers do not fit the nodes.
    """
    kv_offloading = benchmark.get(Fields.KV_OFFLOADING.value, "none")
    if kv_offloading != "dram":
        return 0, 0

    available_mib = min(
        runner_available_cpu_dram_mib(runner, runner_data),
        MAX_AGENTIC_AVAILABLE_CPU_DRAM_MIB,
    )
    utilization = Decimal(str(agentic_config[Fields.DRAM_UTILIZATION.value]))
    gpus_per_node = runner_gpus_per_node(runner, runner_data)

    def share_gb(gpu_count: int) -> int:
        """Return the GB of offload DRAM that ``gpu_count`` GPUs get as their share of a node."""
        proportional_bytes = (
            Decimal(available_mib) * BYTES_PER_MIB * utilization * gpu_count / gpus_per_node
        )
        return int(proportional_bytes / BYTES_PER_GB)

    if Fields.WORKER.value in benchmark:
        return share_gb(worker_gpus_per_node(benchmark[Fields.WORKER.value], gpus_per_node)), 0
    if Fields.PREFILL.value in benchmark:
        return (
            share_gb(
                worker_gpus_per_node(benchmark[Fields.PREFILL.value], gpus_per_node, "prefill")
            ),
            share_gb(worker_gpus_per_node(benchmark[Fields.DECODE.value], gpus_per_node, "decode")),
        )
    gpu_count = effective_gpu_count(benchmark)
    if gpu_count > gpus_per_node:
        raise ValueError(
            f"tp={benchmark[Fields.TP.value]} with "
            f"{Fields.PP.value}={benchmark.get(Fields.PP.value, 1)} and "
            f"{Fields.PCP_SIZE.value}={benchmark.get(Fields.PCP_SIZE.value, 1)} "
            f"requires {gpu_count} GPUs and exceeds "
            f"{Fields.GPUS_PER_NODE.value}={gpus_per_node} for runner '{runner}'"
        )
    return share_gb(gpu_count), 0


def agentic_kv_offload_suffix(
    kv_offloading: str,
    kv_offload_backend: dict | None,
) -> str:
    """Return the experiment-name suffix for a KV offload mode and its backend."""
    if kv_offloading == "none":
        return "kvnone"
    return f"kv{kv_offloading}-{kv_offload_backend['name']}"


def multinode_agentic_exp_name(
    model_code: str,
    prefill: dict,
    decode: dict,
    conc_batch: list[int],
    offload_suffix: str,
) -> str:
    """Return a multinode agentic experiment name from the model, worker shapes, and concs."""

    def _worker_tag(worker: dict, role_prefix: str) -> str:
        """Return a short tag of a worker's count, TP, EP, DP attention, and DWDP size."""
        ep = worker.get(Fields.EP.value, 1)
        dpa = worker.get(Fields.DP_ATTN.value, False)
        tag = f"{role_prefix}{worker[Fields.NUM_WORKER.value]}x{worker[Fields.TP.value]}"
        if ep != 1:
            tag += f"ep{ep}"
        if dpa:
            tag += "dpa"
        dwdp = worker.get(Fields.DWDP_SIZE.value, 1)
        if dwdp > 1:
            tag += f"dwdp{dwdp}"
        return tag

    return (
        f"{model_code}_{_worker_tag(prefill, 'p')}_{_worker_tag(decode, 'd')}"
        f"_conc{'x'.join(str(c) for c in conc_batch)}"
        f"{offload_suffix}"
    )


def component_metadata(benchmark: dict, config: dict) -> dict:
    """Return the router and KV transfer settings of a benchmark, falling back to its config."""
    metadata = {}
    for field in (Fields.ROUTER, Fields.KV_P2P_TRANSFER):
        value = benchmark.get(field.value, config.get(field.value))
        if value is not None:
            metadata[field.value] = value
    return metadata


def srt_recipe_fields(config: dict, benchmark: dict) -> dict:
    """Return the benchmark's srt recipe fields resolved to paths under the config's recipe dir."""
    return {
        field.value: srt_recipe_path(
            config.get(Fields.MULTINODE.value, False),
            config[Fields.SRT_RECIPE_DIR.value],
            benchmark[field.value],
        )
        for field in (Fields.SRT_RECIPE, Fields.EVAL_SRT_RECIPE)
        if benchmark.get(field.value) is not None
    }


def _multinode_parallelism_key(entry: dict) -> tuple:
    """Return a hashable identity of a multinode row, ignoring conc, eval, and naming fields."""
    ignored_fields = {
        Fields.CONC.value,
        Fields.RUN_EVAL.value,
        Fields.EVAL_ONLY.value,
        Fields.EVAL_CONC.value,
        Fields.EVAL_ALL_CONCS.value,
        Fields.EXP_NAME.value,
        Fields.EVAL_FRAMEWORK.value,
        Fields.EVAL_SUITE.value,
    }
    return tuple(
        sorted(
            (key, freeze_config_value(value))
            for key, value in entry.items()
            if key not in ignored_fields
        )
    )


def automatic_agentic_vendor_eval(entry: dict) -> tuple[str, str] | None:
    """Return the vendor eval framework and suite run automatically for an agentic row, or None."""
    if entry.get(Fields.SCENARIO_TYPE.value) != "agentic-coding":
        return None
    return AUTOMATIC_AGENTIC_VENDOR_EVALS.get(entry.get(Fields.MODEL_PREFIX.value))


def app_eval_identity(entry: dict, conc: int) -> tuple:
    """Return the identity of the app eval a row runs at ``conc``, to detect duplicate evals."""
    if Fields.PREFILL.value in entry:
        topology = tuple(
            tuple(
                entry[role].get(field.value)
                for field in (Fields.TP, Fields.EP, Fields.DP_ATTN, Fields.NUM_WORKER)
            )
            for role in (Fields.PREFILL.value, Fields.DECODE.value)
        )
    else:
        topology = tuple(entry.get(field.value) for field in (Fields.TP, Fields.EP, Fields.DP_ATTN))
    return (
        *(
            entry.get(field.value)
            for field in (
                Fields.MODEL,
                Fields.MODEL_PREFIX,
                Fields.RUNNER,
                Fields.FRAMEWORK,
                Fields.PRECISION,
            )
        ),
        entry.get(Fields.SPEC_DECODING.value, "none"),
        entry.get(Fields.DISAGG.value, False),
        topology,
        entry.get(Fields.EVAL_FRAMEWORK.value),
        entry.get(Fields.EVAL_SUITE.value),
        entry.get(Fields.ISL.value),
        entry.get(Fields.OSL.value),
        conc,
    )


def _eval_concs(entry: dict) -> list[int]:
    """Return the concurrencies a row would evaluate at."""
    conc = entry[Fields.CONC.value]
    conc_values = conc if isinstance(conc, list) else [conc]
    if Fields.PREFILL.value in entry and not entry.get(Fields.EVAL_ALL_CONCS.value):
        return [entry[Fields.EVAL_CONC.value]]
    return sorted(set(conc_values))


def drop_app_colliding_evals(rows: list[dict]) -> list[dict]:
    """Remove eval concurrencies already claimed by another row and return the rows.

    Rows without KV offload claim first. A row left with none stops running evals.
    """
    claimed: set[tuple] = set()
    candidates = sorted(
        (row for row in rows if row.get(Fields.RUN_EVAL.value)),
        key=lambda row: row.get(Fields.KV_OFFLOADING.value, "none") != "none",
    )
    for row in candidates:
        concs = _eval_concs(row)
        free = [conc for conc in concs if app_eval_identity(row, conc) not in claimed]
        claimed.update(app_eval_identity(row, conc) for conc in free)
        if free == concs:
            continue
        if free:
            row[Fields.CONC.value] = free
            continue
        row[Fields.RUN_EVAL.value] = False
        for field in (
            Fields.EVAL_FRAMEWORK,
            Fields.EVAL_SUITE,
            Fields.EVAL_CONC,
            Fields.EVAL_ALL_CONCS,
        ):
            row.pop(field.value, None)
    return rows


def mark_eval_entries(matrix_values: list[dict]) -> list[dict]:
    """Choose which rows run evals and return the rows, plus extra gsm8k eval-only rows.

    Pick the top and median 8k1k concs per single-node group, the top eligible conc per multinode
    shape, one gsm8k point per agentic deployment, and every automatic vendor eval.
    """
    from collections import defaultdict

    target_isl, target_osl = seq_len_stoi["8k1k"]
    eval_indices = set()
    mn_eval_conc = {}

    def _eligible_eval_concs(entry: dict) -> list[int]:
        """Return a row's concurrencies of at least MIN_EVAL_CONC, sorted."""
        conc = entry[Fields.CONC.value]
        conc_values = conc if isinstance(conc, list) else [conc]
        return sorted(c for c in conc_values if c >= MIN_EVAL_CONC)

    automatic_eval_specs: dict[int, tuple[str, str]] = {}
    for i, entry in enumerate(matrix_values):
        eval_spec = automatic_agentic_vendor_eval(entry)
        if eval_spec is None:
            continue
        automatic_eval_specs[i] = eval_spec
        eval_indices.add(i)
        if Fields.PREFILL.value in entry:
            conc = entry[Fields.CONC.value]
            conc_values = conc if isinstance(conc, list) else [conc]
            mn_eval_conc[i] = max(conc_values)

    sn_groups = defaultdict(list)
    for i, entry in enumerate(matrix_values):
        if Fields.TP.value not in entry:
            continue
        if entry.get(Fields.ISL.value) != target_isl or entry.get(Fields.OSL.value) != target_osl:
            continue
        if not _eligible_eval_concs(entry):
            continue
        key = (
            entry[Fields.MODEL.value],
            entry[Fields.RUNNER.value],
            entry[Fields.FRAMEWORK.value],
            entry[Fields.PRECISION.value],
            entry[Fields.ISL.value],
            entry[Fields.OSL.value],
            entry[Fields.SPEC_DECODING.value],
            entry[Fields.DP_ATTN.value],
        )
        sn_groups[key].append((i, entry))

    for entries in sn_groups.values():
        conc_values = sorted({e[Fields.CONC.value] for _, e in entries})
        median_conc = conc_values[len(conc_values) // 2]
        target_concs = {conc_values[-1], median_conc}
        for i, e in entries:
            if e[Fields.CONC.value] in target_concs:
                eval_indices.add(i)

    mn_groups = defaultdict(list)
    for i, entry in enumerate(matrix_values):
        if Fields.TP.value in entry:
            continue
        if Fields.PREFILL.value not in entry:
            continue
        if entry.get(Fields.ISL.value) != target_isl or entry.get(Fields.OSL.value) != target_osl:
            continue
        eval_concs = _eligible_eval_concs(entry)
        if not eval_concs:
            continue
        mn_groups[_multinode_parallelism_key(entry)].append((i, eval_concs[-1]))

    for entries in mn_groups.values():
        best_idx, best_eval_conc = max(entries, key=lambda item: item[1])
        eval_indices.add(best_idx)
        mn_eval_conc[best_idx] = best_eval_conc

    ag_sn_groups = defaultdict(list)
    ag_mn_groups = defaultdict(list)
    ag_mn_fallback: dict[tuple, tuple[int, int]] = {}
    for i, entry in enumerate(matrix_values):
        if entry.get(Fields.SCENARIO_TYPE.value) != "agentic-coding":
            continue
        if Fields.PREFILL.value in entry:
            conc = entry[Fields.CONC.value]
            top_conc = max(conc) if isinstance(conc, list) else conc
            deployment = (
                entry[Fields.MODEL.value],
                entry[Fields.RUNNER.value],
                entry[Fields.FRAMEWORK.value],
                entry[Fields.PRECISION.value],
            )
            if top_conc > ag_mn_fallback.get(deployment, (-1, -1))[1]:
                ag_mn_fallback[deployment] = (i, top_conc)
            eval_concs = _eligible_eval_concs(entry)
            if eval_concs:
                ag_mn_groups[_multinode_parallelism_key(entry)].append((i, eval_concs[-1]))
            continue
        conc = entry[Fields.CONC.value]
        conc_val = max(conc) if isinstance(conc, list) else conc
        key = (
            entry[Fields.MODEL.value],
            entry[Fields.RUNNER.value],
            entry[Fields.FRAMEWORK.value],
            entry[Fields.PRECISION.value],
            entry[Fields.SPEC_DECODING.value],
            entry[Fields.DP_ATTN.value],
            entry[Fields.IMAGE.value],
        )
        ag_sn_groups[key].append((i, conc_val))
    gsm8k_picks: dict[int, int | None] = {}
    for entries in ag_sn_groups.values():
        gsm8k_picks[max(entries, key=lambda item: item[1])[0]] = None
    for entries in ag_mn_groups.values():
        best_idx, best_eval_conc = max(entries, key=lambda item: item[1])
        gsm8k_picks[best_idx] = best_eval_conc
    covered = {
        (
            matrix_values[i][Fields.MODEL.value],
            matrix_values[i][Fields.RUNNER.value],
            matrix_values[i][Fields.FRAMEWORK.value],
            matrix_values[i][Fields.PRECISION.value],
        )
        for i, eval_conc in gsm8k_picks.items()
        if eval_conc is not None
    }
    for deployment, (i, top_conc) in ag_mn_fallback.items():
        if deployment not in covered:
            gsm8k_picks[i] = top_conc
    extra_gsm8k: dict[int, int | None] = {}
    for i, eval_conc in gsm8k_picks.items():
        if i in automatic_eval_specs:
            extra_gsm8k[i] = eval_conc
            continue
        eval_indices.add(i)
        if eval_conc is not None:
            mn_eval_conc[i] = eval_conc

    for i, entry in enumerate(matrix_values):
        run_eval = i in eval_indices
        entry[Fields.RUN_EVAL.value] = run_eval
        if run_eval:
            eval_framework, eval_suite = automatic_eval_specs.get(i, (DEFAULT_EVAL_FRAMEWORK, ""))
            entry[Fields.EVAL_FRAMEWORK.value] = eval_framework
            entry[Fields.EVAL_SUITE.value] = eval_suite
        if i in mn_eval_conc:
            entry[Fields.EVAL_CONC.value] = mn_eval_conc[i]

    for i, eval_conc in extra_gsm8k.items():
        matrix_values.append(_gsm8k_eval_only_row(matrix_values[i], eval_conc))

    return [
        row
        for row in drop_app_colliding_evals(matrix_values)
        if row.get(Fields.RUN_EVAL.value) or not row.get(Fields.EVAL_ONLY.value)
    ]


def _gsm8k_eval_only_row(entry: dict, eval_conc: int | None = None) -> dict:
    """Return a gsm8k eval-only copy of ``entry``, at ``eval_conc`` when given."""
    row = {
        **entry,
        Fields.RUN_EVAL.value: True,
        Fields.EVAL_ONLY.value: True,
        Fields.EVAL_FRAMEWORK.value: DEFAULT_EVAL_FRAMEWORK,
        Fields.EVAL_SUITE.value: "",
    }
    row.pop(Fields.REQUIRE_POWER.value, None)
    if eval_conc is not None:
        row[Fields.EVAL_CONC.value] = eval_conc
    return row


def mark_all_eval_entries(matrix_values: list[dict]) -> list[dict]:
    """Mark every eligible row to run evals and return the rows, batching multinode concs.

    Agentic vendor evals become separate eval-only rows.
    """
    expanded_entries: list[dict] = []
    multinode_indices: dict[tuple, int] = {}
    multinode_agentic_indices: dict[tuple, int] = {}

    target_isl, target_osl = seq_len_stoi["8k1k"]

    for entry in matrix_values:
        if entry.get(Fields.EVAL_ONLY.value):
            continue
        automatic_eval = automatic_agentic_vendor_eval(entry)
        if automatic_eval is not None:
            eval_framework, eval_suite = automatic_eval
            eval_entry = {
                **entry,
                Fields.RUN_EVAL.value: True,
                Fields.EVAL_ONLY.value: True,
                Fields.EVAL_FRAMEWORK.value: eval_framework,
                Fields.EVAL_SUITE.value: eval_suite,
            }
            if Fields.PREFILL.value in entry:
                conc = entry[Fields.CONC.value]
                conc_values = conc if isinstance(conc, list) else [conc]
                eval_entry[Fields.CONC.value] = sorted(set(conc_values))
                eval_entry[Fields.EVAL_CONC.value] = max(conc_values)
            expanded_entries.append(eval_entry)
            entry = {
                key: value
                for key, value in entry.items()
                if key
                not in (
                    Fields.EVAL_FRAMEWORK.value,
                    Fields.EVAL_SUITE.value,
                    Fields.EVAL_CONC.value,
                )
            }

        if entry.get(Fields.SCENARIO_TYPE.value) == "agentic-coding":
            if Fields.PREFILL.value not in entry:
                entry[Fields.RUN_EVAL.value] = True
                expanded_entries.append(entry)
                continue

            conc = entry[Fields.CONC.value]
            conc_values = conc if isinstance(conc, list) else [conc]
            parallelism_key = _multinode_parallelism_key(entry)
            if parallelism_key in multinode_agentic_indices:
                existing = expanded_entries[multinode_agentic_indices[parallelism_key]]
                merged_conc = sorted(set(existing[Fields.CONC.value] + conc_values))
                existing[Fields.CONC.value] = merged_conc
                existing[Fields.EVAL_CONC.value] = max(merged_conc)
                continue

            eval_entry = {
                **entry,
                Fields.CONC.value: sorted(set(conc_values)),
                Fields.RUN_EVAL.value: True,
                Fields.EVAL_CONC.value: max(conc_values),
            }
            multinode_agentic_indices[parallelism_key] = len(expanded_entries)
            expanded_entries.append(eval_entry)
            continue

        if entry.get(Fields.ISL.value) != target_isl or entry.get(Fields.OSL.value) != target_osl:
            expanded_entries.append(entry)
            continue

        if Fields.PREFILL.value in entry:
            conc = entry[Fields.CONC.value]
            conc_values = conc if isinstance(conc, list) else [conc]
            parallelism_key = _multinode_parallelism_key(entry)
            if parallelism_key in multinode_indices:
                existing = expanded_entries[multinode_indices[parallelism_key]]
                existing[Fields.CONC.value] = sorted(set(existing[Fields.CONC.value] + conc_values))
                continue

            batched_entry = {
                **entry,
                Fields.CONC.value: sorted(set(conc_values)),
                Fields.RUN_EVAL.value: True,
                Fields.EVAL_ALL_CONCS.value: True,
            }
            batched_entry.pop(Fields.EVAL_CONC.value, None)
            multinode_indices[parallelism_key] = len(expanded_entries)
            expanded_entries.append(batched_entry)
            continue

        entry[Fields.RUN_EVAL.value] = True
        expanded_entries.append(entry)

    for entry in expanded_entries:
        if not entry.get(Fields.RUN_EVAL.value):
            continue
        if not entry.get(Fields.EVAL_FRAMEWORK.value):
            entry[Fields.EVAL_FRAMEWORK.value] = DEFAULT_EVAL_FRAMEWORK
        if entry.get(Fields.EVAL_SUITE.value) is None:
            entry[Fields.EVAL_SUITE.value] = ""

    return drop_app_colliding_evals(expanded_entries)


def _concurrency_range(start: int, end: int, step: int) -> list[int]:
    """Return ``start``, ``start * step``, and so on, capped so the last value is ``end``."""
    values = []
    while start <= end:
        values.append(start)
        if start == end:
            break
        start = min(start * step, end)
    return values


def _fixed_sequence_entries(
    config: dict,
    benchmark: dict,
    sequence: dict,
    conc_values: list[int],
    runners: list[str],
    runner_data: dict,
    root: Path | None,
) -> list[dict]:
    """Return validated fixed-sequence rows for one benchmark, per runner and concurrency.

    Multinode rows carry all concs in one row. Raise ValueError for an invalid power request.
    """
    is_multinode = config.get(Fields.MULTINODE.value, False)
    disagg = config.get(Fields.DISAGG.value, False)
    isl, osl = sequence[Fields.ISL.value], sequence[Fields.OSL.value]
    require_power = sequence.get(Fields.REQUIRE_POWER.value, sequence.get("require_power", False))
    if require_power and (isl, osl) != (8192, 1024):
        raise ValueError("require-power rollout supports only fixed-sequence 8192/1024")
    model_code = config[Fields.MODEL_PREFIX.value]
    spec_decoding = benchmark.get(Fields.SPEC_DECODING.value, "none")
    if is_multinode:
        prefill, decode = multinode_worker_pair(benchmark, disagg)
    else:
        ep = benchmark.get(Fields.EP.value)
        dp_attn = benchmark.get(Fields.DP_ATTN.value)

    entries = []
    for conc in [conc_values] if is_multinode else conc_values:
        for runner in runners:
            entry = {
                Fields.IMAGE.value: config[Fields.IMAGE.value],
                Fields.MODEL.value: config[Fields.MODEL.value],
                Fields.MODEL_PREFIX.value: model_code,
                Fields.PRECISION.value: config[Fields.PRECISION.value],
                Fields.FRAMEWORK.value: config[Fields.FRAMEWORK.value],
                Fields.RUNNER.value: runner,
                Fields.ISL.value: isl,
                Fields.OSL.value: osl,
            }
            if require_power:
                entry[Fields.REQUIRE_POWER.value] = True
            if is_multinode:
                entry.update(
                    {
                        Fields.SPEC_DECODING.value: spec_decoding,
                        Fields.PREFILL.value: prefill,
                        Fields.DECODE.value: decode,
                        Fields.CONC.value: conc,
                        Fields.MAX_MODEL_LEN.value: isl + osl + 256,
                    }
                )
            else:
                entry.update(
                    {
                        Fields.TP.value: benchmark[Fields.TP.value],
                        Fields.PP.value: benchmark.get(Fields.PP.value, 1),
                        Fields.DCP_SIZE.value: benchmark.get(Fields.DCP_SIZE.value, 1),
                        Fields.PCP_SIZE.value: benchmark.get(Fields.PCP_SIZE.value, 1),
                        Fields.CONC.value: conc,
                        Fields.MAX_MODEL_LEN.value: isl + osl + 256,
                        Fields.EP.value: ep if ep is not None else 1,
                        Fields.DP_ATTN.value: dp_attn if dp_attn is not None else False,
                        Fields.SPEC_DECODING.value: spec_decoding,
                    }
                )
            entry.update(srt_recipe_fields(config, benchmark))
            if benchmark.get(Fields.POWER.value):
                entry[Fields.POWER.value] = True
            entry.update(
                {
                    Fields.EXP_NAME.value: f"{model_code}_{seq_len_to_str(isl, osl)}",
                    Fields.DISAGG.value: disagg,
                    Fields.RUN_EVAL.value: False,
                }
            )
            entry.update(component_metadata(benchmark, config))
            if is_multinode:
                add_multinode_node_count(
                    entry, runner_data, benchmark.get(Fields.NUM_NODES.value), root
                )
            entries.append(validate_matrix_entry(entry, is_multinode))
    return entries


def _agentic_entries(
    config: dict,
    benchmark: dict,
    scenario: dict,
    runners: list[str],
    runner_data: dict,
    *,
    step_size: int = 2,
    min_conc: int | None = None,
    max_conc: int | None = None,
    conc_filter: list[int] | None = None,
    root: Path | None = None,
) -> list[dict]:
    """Return validated agentic rows for one benchmark at each runner and allowed concurrency."""
    is_multinode = config.get(Fields.MULTINODE.value, False)
    disagg = config.get(Fields.DISAGG.value, False)
    model_code = config[Fields.MODEL_PREFIX.value]
    if is_multinode:
        prefill, decode = multinode_worker_pair(benchmark, disagg)
        kv_offloading = benchmark.get(Fields.KV_OFFLOADING.value, "none")
    else:
        tp = benchmark[Fields.TP.value]
        pp = benchmark.get(Fields.PP.value, 1)
        dcp_size = benchmark.get(Fields.DCP_SIZE.value, 1)
        pcp_size = benchmark.get(Fields.PCP_SIZE.value, 1)
        ep = benchmark.get(Fields.EP.value)
        dp_attn = benchmark.get(Fields.DP_ATTN.value)
        kv_offloading = benchmark[Fields.KV_OFFLOADING.value]
    spec_decoding = benchmark.get(Fields.SPEC_DECODING.value, "none")
    kv_offload_backend = benchmark.get(Fields.KV_OFFLOAD_BACKEND.value)
    total_cpu_dram_gb, decode_cpu_dram_gb = agentic_dram_offload_gb(
        scenario, benchmark, config[Fields.RUNNER.value], runner_data
    )

    conc_values = benchmark.get(Fields.CONC_LIST.value)
    if not conc_values:
        conc_values = _concurrency_range(
            benchmark[Fields.CONC_START.value],
            benchmark[Fields.CONC_END.value],
            step_size,
        )
    if min_conc is not None:
        conc_values = [c for c in conc_values if c >= min_conc]
    if max_conc is not None:
        conc_values = [c for c in conc_values if c <= max_conc]
    if conc_filter:
        conc_values = [c for c in conc_values if c in conc_filter]
    if not conc_values:
        return []

    if is_multinode:
        offload_suffix = (
            f"_{agentic_kv_offload_suffix(kv_offloading, kv_offload_backend)}"
            if kv_offloading != "none"
            else ""
        )
        points = ((runner, [conc]) for runner in runners for conc in conc_values)
    else:
        points = ((runner, conc) for conc in conc_values for runner in runners)

    entries = []
    for runner, conc in points:
        entry = {
            Fields.IMAGE.value: config[Fields.IMAGE.value],
            Fields.MODEL.value: config[Fields.MODEL.value],
            Fields.MODEL_PREFIX.value: model_code,
            Fields.PRECISION.value: config[Fields.PRECISION.value],
            Fields.FRAMEWORK.value: config[Fields.FRAMEWORK.value],
            Fields.RUNNER.value: runner,
        }
        if is_multinode:
            entry.update(
                {
                    Fields.SPEC_DECODING.value: spec_decoding,
                    Fields.PREFILL.value: prefill,
                    Fields.DECODE.value: decode,
                    Fields.CONC.value: conc,
                }
            )

            exp_name = multinode_agentic_exp_name(model_code, prefill, decode, conc, offload_suffix)
        else:
            entry.update(
                {
                    Fields.TP.value: tp,
                    Fields.PP.value: pp,
                    Fields.DCP_SIZE.value: dcp_size,
                    Fields.PCP_SIZE.value: pcp_size,
                    Fields.EP.value: ep if ep is not None else 1,
                    Fields.DP_ATTN.value: dp_attn if dp_attn is not None else False,
                    Fields.SPEC_DECODING.value: spec_decoding,
                    Fields.CONC.value: conc,
                }
            )
            exp_name = (
                f"{model_code}_tp{tp}_conc{conc}_"
                f"{agentic_kv_offload_suffix(kv_offloading, kv_offload_backend)}"
                + (f"_spec-{spec_decoding}" if spec_decoding != "none" else "")
            )
        entry.update(srt_recipe_fields(config, benchmark))
        if benchmark.get(Fields.POWER.value):
            entry[Fields.POWER.value] = True
        entry.update(
            {
                Fields.KV_OFFLOADING.value: kv_offloading,
                Fields.TOTAL_CPU_DRAM_GB.value: total_cpu_dram_gb,
                Fields.DURATION.value: DEFAULT_AGENTIC_DURATION_SECONDS,
                Fields.EXP_NAME.value: exp_name,
            }
        )
        if is_multinode:
            if decode_cpu_dram_gb:
                entry[Fields.DECODE_CPU_DRAM_GB.value] = decode_cpu_dram_gb
            entry[Fields.DISAGG.value] = disagg
        entry[Fields.SCENARIO_TYPE.value] = "agentic-coding"
        if kv_offload_backend is not None:
            entry[Fields.KV_OFFLOAD_BACKEND.value] = kv_offload_backend
        entry.update(component_metadata(benchmark, config))
        if is_multinode:
            add_multinode_node_count(
                entry, runner_data, benchmark.get(Fields.NUM_NODES.value), root
            )
        entries.append(validate_agentic_matrix_entry(entry))
    return entries


@dataclass(frozen=True)
class FullSweepOptions:
    model_prefix: list[str] | None = None
    precision: list[str] | None = None
    framework: list[str] | None = None
    runner_type: list[str] | None = None
    seq_lens: list[str] | None = None
    scenario_types: list[str] | None = None
    runner_node_filter: str | None = None
    single_node: bool = True
    multi_node: bool = True
    step_size: int = 2
    min_conc: int | None = None
    max_conc: int | None = None
    max_tp: int | None = None
    max_ep: int | None = None


def generate_full_sweep(
    args: argparse.Namespace,
    all_config_data: dict,
    runner_data: dict,
    root: Path | None = None,
) -> list[dict]:
    """Return the full-sweep rows selected by the parsed command-line ``args``."""
    return expand_full_sweep(
        all_config_data,
        runner_data,
        options=FullSweepOptions(
            model_prefix=args.model_prefix,
            precision=args.precision,
            framework=args.framework,
            runner_type=args.runner_type,
            seq_lens=args.seq_lens,
            scenario_types=getattr(args, "scenario_type", None),
            runner_node_filter=args.runner_node_filter,
            single_node=args.single_node,
            multi_node=args.multi_node,
            step_size=args.step_size,
            min_conc=args.min_conc,
            max_conc=args.max_conc,
            max_tp=args.max_tp,
            max_ep=args.max_ep,
        ),
        root=root,
    )


def expand_full_sweep(
    master_config: dict,
    runner_data: dict,
    *,
    options: FullSweepOptions = FullSweepOptions(),
    root: Path | None = None,
) -> list[dict]:
    """Return rows for every master config that passes the full-sweep filters in ``options``.

    Raise ValueError for invalid options or filters that match no config.
    """
    if options.step_size <= 1:
        raise ValueError("step_size must be greater than 1")
    if (
        options.min_conc is not None
        and options.max_conc is not None
        and options.min_conc > options.max_conc
    ):
        raise ValueError("min_conc must be less than or equal to max_conc")
    if options.runner_type:
        valid_runner_types = set(runner_labels(runner_data))
        invalid_runners = set(options.runner_type) - valid_runner_types
        if invalid_runners:
            raise ValueError(
                f"Invalid runner type(s): {invalid_runners}. "
                f"Valid runner types are: {', '.join(sorted(valid_runner_types))}"
            )
    precisions = {config[Fields.PRECISION.value] for config in master_config.values()}
    frameworks = {config[Fields.FRAMEWORK.value] for config in master_config.values()}
    for label, requested, exists in (
        (
            "model prefix",
            options.model_prefix,
            lambda p: any(k.startswith(f"{p}-") for k in master_config),
        ),
        ("precision", options.precision, precisions.__contains__),
        ("framework", options.framework, frameworks.__contains__),
    ):
        if missing := [value for value in requested or () if not exists(value)]:
            raise ValueError(f"No config has {label}: {', '.join(missing)}")

    seq_filter = {seq_len_stoi[sl] for sl in options.seq_lens} if options.seq_lens else None
    configs = (
        config
        for key, config in master_config.items()
        if (not options.model_prefix or any(key.startswith(f"{p}-") for p in options.model_prefix))
        and (not options.precision or config[Fields.PRECISION.value] in options.precision)
        and (not options.framework or config[Fields.FRAMEWORK.value] in options.framework)
        and (not options.runner_type or config[Fields.RUNNER.value] in options.runner_type)
    )
    return _expand_configs(
        configs,
        runner_data,
        seq_filter=seq_filter,
        scenario_types=options.scenario_types,
        runner_node_filter=options.runner_node_filter,
        full_sweep=options,
        root=root,
    )


def _fixed_sequence_concurrencies(
    benchmark: dict,
    *,
    multinode: bool,
    full_sweep: FullSweepOptions | None,
    concurrencies: list[int] | None,
) -> list[int] | None:
    """Return a fixed-sequence benchmark's concurrencies after the sweep or ``--conc`` filters.

    Return None when the filters leave nothing to run.
    """
    if full_sweep is None:
        values = (
            benchmark[Fields.CONC_LIST.value]
            if Fields.CONC_LIST.value in benchmark
            else _concurrency_range(
                benchmark[Fields.CONC_START.value], benchmark[Fields.CONC_END.value], 2
            )
        )
        return ([c for c in values if c in concurrencies] or None) if concurrencies else values

    minimum, maximum = full_sweep.min_conc, full_sweep.max_conc
    values = benchmark.get(Fields.CONC_LIST.value)
    if not values:
        start, end = (
            benchmark[Fields.CONC_START.value],
            benchmark[Fields.CONC_END.value],
        )
        if not multinode:
            if minimum is not None:
                if minimum <= 0 or end < minimum:
                    return None
                start = max(start, minimum)
            if maximum is not None:
                if maximum <= 0:
                    return None
                start, end = min(start, maximum), min(end, maximum)
            return _concurrency_range(start, end, full_sweep.step_size)
        values = _concurrency_range(start, end, full_sweep.step_size)

    if minimum is not None:
        if minimum <= 0:
            return None
        values = [c for c in values if c >= minimum]
        if not values:
            return None
    if maximum is not None:
        if maximum <= 0:
            return None
        values = [c for c in values if c <= maximum] or [maximum]
    return values


def _runner_values_for_filter(
    runner: str,
    runner_data: dict,
    runner_node_filter: str | None,
    *,
    include_label: bool = True,
) -> list[str]:
    """Return the runner label, or its nodes and itself, that match ``runner_node_filter``."""
    if not runner_node_filter:
        return [runner]
    candidates = runner_nodes_for_label(runner, runner_data)
    if not include_label:
        return [node for node in candidates if runner_node_filter in node]
    if runner_node_filter in runner:
        candidates = [runner, *candidates]
    return list(dict.fromkeys(node for node in candidates if runner_node_filter in node))


def generate_test_config_sweep(
    args: argparse.Namespace,
    all_config_data: dict,
    runner_data: dict | None = None,
    root: Path | None = None,
) -> list[dict]:
    """Return the rows of the config keys named on the ``test-config`` command line."""
    return _expand_selected_configs(
        args.config_keys,
        all_config_data,
        runner_data,
        runner_node_filter=getattr(args, "runner_node_filter", None),
        seq_lens=getattr(args, "seq_lens", None),
        scenario_types=getattr(args, "scenario_type", None),
        concurrencies=getattr(args, "conc", None),
        root=root,
    )


def _expand_selected_configs(
    config_keys: list[str],
    all_config_data: dict,
    runner_data: dict | None,
    *,
    runner_node_filter: str | None = None,
    seq_lens: list[str] | None = None,
    scenario_types: tuple[str, ...] | list[str] | None = None,
    concurrencies: list[int] | None = None,
    root: Path | None = None,
) -> list[dict]:
    """Return rows for the selected config keys, with the given filters applied.

    Raise ValueError when a requested concurrency is generated by no row.
    """
    resolved_keys = expand_config_keys(config_keys, all_config_data.keys())
    rows = _expand_configs(
        (all_config_data[key] for key in resolved_keys),
        runner_data or {},
        runner_node_filter=runner_node_filter,
        seq_lens=seq_lens,
        scenario_types=scenario_types,
        concurrencies=concurrencies,
        root=root,
    )
    if concurrencies:
        generated = set()
        for row in rows:
            conc = row[Fields.CONC.value]
            generated.update(conc if isinstance(conc, list) else [conc])
        if missing := [conc for conc in concurrencies if conc not in generated]:
            raise ValueError(
                "--conc value(s) not generated by the selected configs: "
                + ", ".join(map(str, missing))
            )
    return rows


def _expand_configs(
    configs: Iterable[dict],
    runner_data: dict,
    *,
    runner_node_filter: str | None = None,
    seq_lens: list[str] | None = None,
    seq_filter: set[tuple[int, int]] | None = None,
    scenario_types: tuple[str, ...] | list[str] | None = None,
    concurrencies: list[int] | None = None,
    full_sweep: FullSweepOptions | None = None,
    root: Path | None = None,
) -> list[dict]:
    """Return the fixed-sequence and agentic rows of each config that pass the filters.

    With ``root``, raise ValueError when a selected recipe references an unknown fabric field.
    """
    rows = []
    for config in configs:
        multinode = config.get(Fields.MULTINODE.value, False)
        if full_sweep is not None:
            scenarios = config[Fields.SCENARIOS.value]
            fixed = (
                scenarios.get(Fields.FIXED_SEQ_LEN.value, [])
                if not scenario_types or "fixed-seq-len" in scenario_types
                else []
            )
        runners = _runner_values_for_filter(
            config[Fields.RUNNER.value],
            runner_data,
            runner_node_filter,
            include_label=full_sweep is None,
        )
        if not runners:
            continue
        if seq_lens:
            seq_filter = {seq_len_stoi[sl] for sl in seq_lens}
        if full_sweep is None:
            fixed = (
                config[Fields.SCENARIOS.value].get(Fields.FIXED_SEQ_LEN.value, [])
                if not scenario_types or "fixed-seq-len" in scenario_types
                else []
            )
        node_allowed = full_sweep is None or (
            full_sweep.multi_node if multinode else full_sweep.single_node
        )

        for sequence in fixed:
            isl, osl = sequence[Fields.ISL.value], sequence[Fields.OSL.value]
            if seq_filter and (isl, osl) not in seq_filter:
                continue
            for benchmark in sequence[Fields.SEARCH_SPACE.value]:
                if not node_allowed:
                    continue
                if full_sweep is not None and not multinode:
                    tp, ep = benchmark[Fields.TP.value], benchmark.get(Fields.EP.value)
                    if full_sweep.max_tp is not None and (
                        full_sweep.max_tp <= 0 or tp > full_sweep.max_tp
                    ):
                        continue
                    if full_sweep.max_ep is not None:
                        if full_sweep.max_ep <= 0:
                            continue
                        if ep is not None:
                            ep = min(ep, full_sweep.max_ep)
                    benchmark = {**benchmark, Fields.EP.value: ep}
                values = _fixed_sequence_concurrencies(
                    benchmark,
                    multinode=multinode,
                    full_sweep=full_sweep,
                    concurrencies=concurrencies,
                )
                if values is None:
                    continue
                rows.extend(
                    _fixed_sequence_entries(
                        config, benchmark, sequence, values, runners, runner_data, root
                    )
                )

        agentic = (
            config[Fields.SCENARIOS.value].get(Fields.AGENTIC_CODING.value, [])
            if not scenario_types or "agentic-coding" in scenario_types
            else []
        )
        if not node_allowed:
            continue
        for scenario in agentic:
            for benchmark in scenario[Fields.SEARCH_SPACE.value]:
                rows.extend(
                    _agentic_entries(
                        config,
                        benchmark,
                        scenario,
                        runners,
                        runner_data,
                        step_size=full_sweep.step_size if full_sweep is not None else 2,
                        min_conc=full_sweep.min_conc if full_sweep is not None else None,
                        max_conc=full_sweep.max_conc if full_sweep is not None else None,
                        conc_filter=concurrencies,
                        root=root,
                    )
                )
    if root is not None:
        selected = {
            (row[field.value], row[Fields.RUNNER.value])
            for row in rows
            for field in (Fields.SRT_RECIPE, Fields.EVAL_SRT_RECIPE)
            if row.get(field.value)
        }
        for srt_recipe, label in sorted(selected):
            check_fabric_references(root, srt_recipe, label, runner_data)
    return rows


def expand_config_keys(config_keys: Iterable[str], available_keys: Iterable[str]) -> list[str]:
    """Return the config keys, with glob patterns expanded, in order and without duplicates.

    Raise ValueError for a key or pattern that matches nothing.
    """
    available = list(available_keys)
    seen = {}
    for key in config_keys:
        if "*" in key or "?" in key:
            matches = fnmatch.filter(available, key)
            if not matches:
                raise ValueError(
                    f"Pattern '{key}' matched no config keys.\n"
                    f"Available keys: {', '.join(sorted(available))}"
                )
            for m in matches:
                seen.setdefault(m, None)
        else:
            if key not in available:
                raise ValueError(
                    f"Config key(s) not found: {key}.\n"
                    f"Available keys: {', '.join(sorted(available))}"
                )
            seen.setdefault(key, None)
    return list(seen)


def filter_exp_names(entries: list[dict], exp_names: list[str]) -> list[dict]:
    """Return the rows named in ``exp_names``, raising ValueError unless each matches one row."""
    requested = set(exp_names)
    if len(requested) != len(exp_names):
        raise ValueError("--exp-names contains duplicate values")

    matches: dict[str, int] = dict.fromkeys(exp_names, 0)
    for entry in entries:
        exp_name = entry.get(Fields.EXP_NAME.value)
        if exp_name in matches:
            matches[exp_name] += 1

    missing = sorted(name for name, count in matches.items() if count == 0)
    ambiguous = sorted(name for name, count in matches.items() if count > 1)
    if missing:
        raise ValueError("Experiment name(s) not found: " + ", ".join(missing))
    if ambiguous:
        raise ValueError("Experiment name(s) matched multiple rows: " + ", ".join(ambiguous))
    return [entry for entry in entries if entry.get(Fields.EXP_NAME.value) in requested]


def apply_node_type_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Enable both node types in ``args`` when neither is set, and return ``args``."""
    if (
        hasattr(args, "single_node")
        and hasattr(args, "multi_node")
        and not args.single_node
        and not args.multi_node
    ):
        args.single_node = True
        args.multi_node = True
    return args


EvalMode = Literal["default", "none", "subset", "all", "smoke"]


def select_matrix_evals(
    rows: list[dict],
    *,
    mode: EvalMode = "default",
    trim: bool = False,
) -> list[dict]:
    """Apply eval selection ``mode`` to rows, optionally trimming to minimum concurrency.

    Raise ValueError for an unknown mode or smoke combined with trimming.
    """
    if mode not in ("default", "none", "subset", "all", "smoke"):
        raise ValueError(f"Unknown eval mode: {mode!r}")
    if mode == "smoke" and trim:
        raise ValueError("smoke cannot be combined with trimming")
    if mode != "none":
        rows = mark_eval_entries(rows)
        if mode == "all":
            rows = mark_all_eval_entries(rows)
    if mode == "smoke":
        return smoke_entries(rows)
    if trim:
        rows = [
            row
            for row in drop_app_colliding_evals(trim_conc(rows))
            if row.get(Fields.RUN_EVAL.value) or not row.get(Fields.EVAL_ONLY.value)
        ]
    if mode in ("subset", "all"):
        rows = [row for row in rows if row.get(Fields.RUN_EVAL.value, False)]
        for row in rows:
            row[Fields.EVAL_ONLY.value] = True
            row.pop(Fields.REQUIRE_POWER.value, None)
    return rows


def generate_config_matrix(
    config_keys: list[str],
    master_config: dict,
    runner_data: dict,
    *,
    scenario_types: tuple[str, ...] | list[str] | None = None,
    eval_mode: EvalMode = "default",
    root: Path | None = None,
) -> list[dict]:
    """Return the JSON-ready rows of the config keys with ``eval_mode`` eval selection applied."""
    rows = _expand_selected_configs(
        config_keys,
        master_config,
        runner_data,
        scenario_types=scenario_types,
        root=root,
    )
    rows = select_matrix_evals(rows, mode=eval_mode)
    return json.loads(json.dumps(rows))


def with_fingerprints(rows: list[dict], root: Path) -> list[dict]:
    """``rows`` with the ``recipe-fingerprint`` changelog plans give the same throughput rows.

    Eval selection copies it to the rows it derives. A row whose recipe cannot be bound gets
    none and a warning, and its job fails when it binds the recipe. The fingerprint module is
    imported here because it imports this module through srt_slurm.
    """
    from infx.matrix.fingerprint import recipe_fingerprint

    fingerprinted = []
    for row in rows:
        try:
            fingerprint = recipe_fingerprint(row, root)
        except ValueError as error:
            print(f"WARNING: no recipe fingerprint: {error}", file=sys.stderr)
            fingerprinted.append(row)
        else:
            fingerprinted.append({**row, Fields.RECIPE_FINGERPRINT.value: fingerprint})
    return fingerprinted


def main() -> list[dict]:
    """Parse the command line, print the generated matrix rows as JSON, and return them."""
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--config-files",
        nargs="+",
        required=True,
        help="One or more configuration files (YAML format)",
    )
    parent_parser.add_argument(
        "--runner-config",
        default="configs/runners.yaml",
        help="Configuration file holding runner information (YAML format, defaults to configs/runners.yaml)",
    )
    eval_group = parent_parser.add_mutually_exclusive_group()
    eval_group.add_argument(
        "--no-evals",
        action="store_true",
        help="When specified, skip evals (throughput benchmarks only).",
    )
    eval_group.add_argument(
        "--evals-only",
        action="store_true",
        help="When specified, run ONLY the eval subset (excludes non-eval configs).",
    )
    parent_parser.add_argument(
        "--all-evals",
        action="store_true",
        help=(
            "Expand eval selection to every eligible 8k1k fixed-sequence and agentic config "
            "and emit only those eval-only jobs, with or without --evals-only."
        ),
    )
    parent_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Minimum-concurrency throughput plus canonical representative evals.",
    )
    parent_parser.add_argument(
        "--trim-conc",
        action="store_true",
        help=(
            "Trim each generated deployment shape to its minimum concurrency "
            "after applying eval selection."
        ),
    )
    parent_parser.add_argument(
        "--runner-node-filter",
        required=False,
        help='Filter runner nodes by substring match (e.g., "amd" to only include nodes containing that string). Expands each config to individual matching nodes.',
    )
    parent_parser.add_argument(
        "--scenario-type",
        nargs="+",
        choices=["fixed-seq-len", "agentic-coding"],
        required=False,
        help="Scenario type(s) to include. If not specified, all scenario types are generated.",
    )

    parser = argparse.ArgumentParser(
        description="Generate benchmark configurations from YAML config files"
    )

    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    full_sweep_parser = subparsers.add_parser(
        "full-sweep",
        parents=[parent_parser],
        add_help=False,
        help="Generate full sweep configurations with optional filtering by model, precision, framework, runner type, and sequence lengths",
    )
    full_sweep_parser.add_argument(
        "--model-prefix",
        nargs="+",
        required=False,
        help="Model prefix(es) to filter configurations (optional, can specify multiple)",
    )
    full_sweep_parser.add_argument(
        "--precision",
        nargs="+",
        required=False,
        help="Precision(s) to filter by (e.g., fp4, fp8) (optional, can specify multiple)",
    )
    full_sweep_parser.add_argument(
        "--framework",
        nargs="+",
        required=False,
        help="Framework(s) to filter by (e.g., vllm, trt, sglang) (optional, can specify multiple)",
    )
    full_sweep_parser.add_argument(
        "--runner-type",
        nargs="+",
        required=False,
        help="Runner type(s) to filter by (e.g., h200, h100) (optional, can specify multiple)",
    )
    full_sweep_parser.add_argument(
        "--seq-lens",
        nargs="+",
        choices=list(seq_len_stoi.keys()),
        required=False,
        help=f"Sequence length configurations to include: {', '.join(seq_len_stoi.keys())}. If not specified, all sequence lengths are included.",
    )
    full_sweep_parser.add_argument(
        "--step-size",
        type=int,
        default=2,
        help="Step size for concurrency values (default: 2)",
    )
    full_sweep_parser.add_argument(
        "--min-conc",
        type=int,
        required=False,
        help=(
            "Drop concurrencies below this value. A single-node fixed-sequence "
            "conc-start/conc-end range that begins lower starts at this value instead."
        ),
    )
    full_sweep_parser.add_argument(
        "--max-conc",
        type=int,
        required=False,
        help=(
            "Drop concurrencies above this value. A single-node fixed-sequence range that "
            "ends higher ends at it, and a fixed-sequence config with nothing left runs at it."
        ),
    )
    full_sweep_parser.add_argument(
        "--max-tp",
        type=int,
        required=False,
        help="Drop single-node fixed-sequence configs whose TP exceeds this value",
    )
    full_sweep_parser.add_argument(
        "--max-ep",
        type=int,
        required=False,
        help="Cap the EP of single-node fixed-sequence configs at this value",
    )
    full_sweep_parser.add_argument(
        "--single-node",
        action="store_true",
        help="Only generate single-node configurations. If neither --single-node nor --multi-node is specified, both types are generated.",
    )
    full_sweep_parser.add_argument(
        "--multi-node",
        action="store_true",
        help="Only generate multi-node configurations. If neither --single-node nor --multi-node is specified, both types are generated.",
    )
    full_sweep_parser.add_argument(
        "-h", "--help", action="help", help="Show this help message and exit"
    )

    test_config_keys_parser = subparsers.add_parser(
        "test-config",
        parents=[parent_parser],
        add_help=False,
        help="Generate full sweep for specific config keys. Validates that all specified keys exist before generating.",
    )
    test_config_keys_parser.add_argument(
        "--config-keys",
        nargs="+",
        required=True,
        help="One or more config keys to generate sweep for (e.g., dsr1-fp4-b200-sglang dsr1-fp8-h200-trt)",
    )
    test_config_keys_parser.add_argument(
        "--conc",
        nargs="+",
        type=int,
        required=False,
        help="Only include these concurrency values. Values must exist in the config conc-range/list.",
    )
    test_config_keys_parser.add_argument(
        "--exp-names",
        nargs="+",
        required=False,
        help=(
            "Only include exact generated experiment names. Each name must "
            "match exactly one row after config and concurrency filtering."
        ),
    )
    test_config_keys_parser.add_argument(
        "--seq-lens",
        nargs="+",
        choices=list(seq_len_stoi.keys()),
        required=False,
        help="Only include these sequence length configurations (e.g., 1k1k 8k1k)",
    )
    test_config_keys_parser.add_argument(
        "-h", "--help", action="help", help="Show this help message and exit"
    )

    args = parser.parse_args()
    apply_node_type_defaults(args)
    if args.command == "full-sweep" and args.step_size <= 1:
        parser.error("--step-size must be greater than 1")
    if (
        args.command == "full-sweep"
        and args.min_conc is not None
        and args.max_conc is not None
        and args.min_conc > args.max_conc
    ):
        parser.error("--min-conc must be less than or equal to --max-conc")
    if args.no_evals and args.all_evals:
        parser.error("--all-evals cannot be combined with --no-evals")

    all_config_data = load_config_files(args.config_files)
    runner_data = load_runner_file(args.runner_config)
    root = config_root(args.config_files)

    if args.command == "full-sweep":
        matrix_values = generate_full_sweep(args, all_config_data, runner_data, root)
    elif args.command == "test-config":
        matrix_values = generate_test_config_sweep(args, all_config_data, runner_data, root)
    else:
        parser.error(f"Unknown command: {args.command}")
    matrix_values = with_fingerprints(matrix_values, root)

    if args.command == "test-config" and args.exp_names:
        try:
            matrix_values = filter_exp_names(matrix_values, args.exp_names)
        except ValueError as error:
            parser.error(str(error))

    if args.smoke and (args.trim_conc or args.no_evals or args.evals_only or args.all_evals):
        parser.error("--smoke cannot be combined with trimming or eval overrides")

    matrix_values = select_matrix_evals(
        matrix_values,
        mode=(
            "smoke"
            if args.smoke
            else "none"
            if args.no_evals
            else "all"
            if args.all_evals
            else "subset"
            if args.evals_only
            else "default"
        ),
        trim=args.trim_conc,
    )

    print(json.dumps(matrix_values))
    return matrix_values


if __name__ == "__main__":
    main()
