import argparse
import fnmatch
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal

import yaml

from .validation import (
    DEFAULT_AGENTIC_DURATION_SECONDS,
    Fields,
    load_config_files,
    load_runner_file,
    validate_agentic_matrix_entry,
    validate_matrix_entry,
)

seq_len_stoi = {
    "1k1k": (1024, 1024),
    "8k1k": (8192, 1024)
}

MIN_EVAL_CONC = 16
DEFAULT_EVAL_FRAMEWORK = "lm-eval"
AUTOMATIC_AGENTIC_VENDOR_EVALS = {
    "kimik3": ("kimi-vendor", "kimi_tool_call_schema"),
    "minimaxm3": ("minimax-vendor", "minimax_m3_smoke"),
}
# Bound how many multinode agentic conc points share one server allocation.
# 1 = one task/SLURM allocation per concurrency (matches single-node agentic).
MAX_MULTINODE_AGENTIC_CONCURRENCIES_PER_ALLOCATION = 1
BYTES_PER_MIB = 1024 * 1024
BYTES_PER_GB = 1_000_000_000
# 3 TB decimal DRAM cap, expressed in MiB, before utilization scaling.
MAX_AGENTIC_AVAILABLE_CPU_DRAM_MIB = 2_861_022

# Reverse mapping for exp-name generation
seq_len_itos = {v: k for k, v in seq_len_stoi.items()}


def seq_len_to_str(isl: int, osl: int) -> str:
    """Convert sequence lengths to short string representation.

    Returns the short name (e.g., '1k1k') if it exists in the mapping,
    otherwise returns 'isl_osl' format.
    """
    return seq_len_itos.get((isl, osl), f"{isl}_{osl}")

def freeze_config_value(value):
    """Convert JSON-shaped config values into deterministic hashable values."""
    if isinstance(value, dict):
        return tuple(
            sorted((key, freeze_config_value(item)) for key, item in value.items())
        )
    if isinstance(value, list):
        return tuple(freeze_config_value(item) for item in value)
    return value


def trim_conc(entries: list[dict]) -> list[dict]:
    """Retain the lowest concurrency for each generated deployment shape.

    Entries are grouped by every non-eval field except ``conc`` and the
    generated ``exp-name``. Multi-node rows may encode concurrency either as a
    list within one row or as one-row list chunks; both representations collapse
    to one row whose ``conc`` and dispatch-facing ``eval-conc`` use the minimum.
    """
    ignored_fields = {
        "conc",
        "exp-name",
        "run-eval",
        "eval-only",
        "eval-conc",
        "eval-all-concs",
        Fields.EVAL_FRAMEWORK.value,
        Fields.EVAL_SUITE.value,
    }
    groups: dict[tuple, list[int]] = {}
    out: list[dict] = []

    def minimum_concurrency(entry: dict):
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
        groups.setdefault(key, []).append(len(out))
        out.append(entry)

    drop: set[int] = set()
    for indices in groups.values():
        keep = min(indices, key=lambda index: minimum_concurrency(out[index]))
        kept_entry = out[keep]
        if any(out[index].get("run-eval") is True for index in indices):
            kept_entry = {**kept_entry, "run-eval": True}
            if kept_entry.get("prefill") is not None:
                kept_entry["eval-conc"] = minimum_concurrency(kept_entry)
            if any(
                out[index].get("eval-all-concs") is True for index in indices
            ):
                kept_entry["eval-all-concs"] = True
            out[keep] = kept_entry
        drop.update(index for index in indices if index != keep)
    return [entry for index, entry in enumerate(out) if index not in drop]


def smoke_entries(entries: list[dict]) -> list[dict]:
    """Minimum-concurrency throughput plus one canonical eval per deployment shape.

    Eval concurrency comes from the existing default selection, never from the
    throughput minimum. Keep separate eval-only rows so neither client is run twice.
    """
    benchmarks = trim_conc([{**row, 'run-eval': False} for row in entries])
    evals = []
    for row in entries:
        if not row.get('run-eval'):
            continue
        row = {**row, 'eval-only': True}
        if row.get('prefill') is not None:
            row['conc'] = [row['eval-conc']]
            row['eval-all-concs'] = False
        evals.append(row)
    return benchmarks + trim_conc(evals)


def runner_labels(runner_data: dict) -> dict:
    """Return runner scheduling labels."""
    return runner_data["labels"]


def runner_hardware(runner_data: dict) -> dict:
    """Return runner hardware metadata, if present."""
    return runner_data.get("hardware", {})


def runner_nodes_for_label(runner: str, runner_data: dict) -> list[str]:
    """Return concrete runner names for a scheduling label."""
    return runner_labels(runner_data).get(runner, [])


def runner_hardware_int(runner: str, runner_data: dict, field: str) -> int:
    """Return an integer hardware field for a runner label."""
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
    """Return available CPU DRAM for a runner label."""
    return runner_hardware_int(runner, runner_data, Fields.AVAILABLE_CPU_DRAM_MIB.value)


def runner_gpus_per_node(runner: str, runner_data: dict) -> int:
    """Return GPUs per node for a runner label."""
    return runner_hardware_int(runner, runner_data, Fields.GPUS_PER_NODE.value)


def _hardware_family(label: str) -> str:
    """Return the GPU family encoded in a runner or cluster label."""
    return label.removeprefix("cluster:").split("-", 1)[0]


def scheduling_gpus_per_node(label: str, runner_data: dict) -> int:
    """Resolve GPUs per node for an abstract runner or worker hardware label.

    Legacy scheduling labels may not duplicate the hardware facts stored under
    their canonical ``cluster:`` label. Fall back to the
    GPU family when the exact label has no hardware record, while rejecting
    ambiguous families that disagree about node shape.
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
        raise ValueError(
            f"Cannot resolve {Fields.GPUS_PER_NODE.value} for '{label}'"
        )
    raise ValueError(
        f"Ambiguous {Fields.GPUS_PER_NODE.value} for '{label}': {sorted(matches)}"
    )


def _worker_node_override(worker: dict, setting_name: str) -> int | None:
    """Read an explicit role node count from additional settings."""
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


def recipe_node_count(prefill: dict, decode: dict) -> int | None:
    """Read the authoritative node count from a checked-in srt-slurm recipe."""
    config_files = {
        setting.split("=", 1)[1]
        for worker in (prefill, decode)
        for setting in (worker.get(Fields.ADDITIONAL_SETTINGS.value, []) or [])
        if setting.startswith("CONFIG_FILE=")
    }
    if not config_files:
        return None
    if len(config_files) != 1:
        raise ValueError(f"Conflicting CONFIG_FILE settings: {sorted(config_files)}")

    config_file = config_files.pop()
    repo_root = Path(__file__).resolve().parents[2]
    recipe_root = repo_root / "benchmarks" / "multi_node" / "srt-slurm-recipes"
    if config_file.startswith("benchmarks/multi_node/srt-slurm-recipes/"):
        recipe_path = repo_root / config_file
    else:
        recipe_path = recipe_root / config_file.removeprefix("recipes/")
    if not recipe_path.exists():
        # Some srt-slurm recipes live only in the runtime image. Their master
        # config topology remains the best available scheduling estimate.
        return None

    resources = yaml.safe_load(recipe_path.read_text())["resources"]
    if "agg_nodes" in resources:
        return int(resources["agg_nodes"])
    if "prefill_nodes" in resources and "decode_nodes" in resources:
        return int(resources["prefill_nodes"]) + int(resources["decode_nodes"])
    raise ValueError(f"Recipe has no supported node resource fields: {recipe_path}")


def worker_node_count(
    worker: dict,
    role: str,
    runner: str,
    runner_data: dict,
) -> int:
    """Return physical nodes consumed by one prefill or decode role."""
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
    )
    return math.ceil(total_gpus / gpus_per_node)


def multinode_node_count(
    prefill: dict,
    decode: dict,
    runner: str,
    runner_data: dict,
) -> int:
    """Return the total Slurm node request represented by a matrix row."""
    recipe_count = recipe_node_count(prefill, decode)
    if recipe_count is not None:
        return recipe_count
    return (
        worker_node_count(prefill, "prefill", runner, runner_data)
        + worker_node_count(decode, "decode", runner, runner_data)
    )


def add_multinode_node_count(
    entry: dict,
    runner_data: dict,
    num_nodes: int | None,
) -> dict:
    """Annotate a multi-node row with its scheduling node count."""
    if not entry[Fields.DISAGG.value] and num_nodes is not None:
        entry[Fields.NODE_COUNT.value] = num_nodes
    elif num_nodes is not None:
        raise ValueError(
            f"{Fields.NUM_NODES.value} is not valid for disaggregated entries"
        )
    else:
        entry[Fields.NODE_COUNT.value] = multinode_node_count(
            entry[Fields.PREFILL.value],
            entry[Fields.DECODE.value],
            entry[Fields.RUNNER.value],
            runner_data,
        )
    return entry


def effective_gpu_count(benchmark: dict) -> int:
    """Return GPUs used by a single-node TP/PP/PCP topology."""
    return (
        benchmark[Fields.TP.value]
        * benchmark.get(Fields.PP.value, 1)
        * benchmark.get(Fields.PCP_SIZE.value, 1)
    )

def with_worker_parallelism_defaults(worker: dict) -> dict:
    """Return a worker config with explicit parallelism defaults."""
    return {
        **worker,
        Fields.PP.value: worker.get(Fields.PP.value, 1),
        Fields.DCP_SIZE.value: worker.get(Fields.DCP_SIZE.value, 1),
        Fields.PCP_SIZE.value: worker.get(Fields.PCP_SIZE.value, 1),
    }


def multinode_worker_pair(benchmark: dict, disagg: bool) -> tuple[dict, dict]:
    """Return the legacy prefill/decode matrix pair for a master entry."""
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
            if key not in (
                Fields.NUM_WORKER.value,
                Fields.ADDITIONAL_SETTINGS.value,
            )
        },
    }
    return prefill, decode


def worker_gpus_per_node(worker: dict, gpus_per_node: int) -> int:
    """Return GPUs a single worker replica occupies on one node.

    The DRAM offload budget is sized per server process (per replica), matching
    the single-node path: a replica claims the node-DRAM fraction that mirrors
    its GPU footprint on the node. Topologies that do not tile a node cleanly
    are rejected rather than silently truncated, keeping parity with the
    single-node "must fit the node" rule:

    * A replica larger than one node (tp*pp*pcp > gpus-per-node) must fill whole
      nodes, i.e. be an exact multiple of gpus-per-node; each of its nodes is
      then fully occupied (fraction 1).
    * A replica within one node must divide it evenly so co-located replicas of
      the same role tile the node without overlap.
    """
    gpus_per_replica = (
        worker[Fields.TP.value]
        * worker.get(Fields.PP.value, 1)
        * worker.get(Fields.PCP_SIZE.value, 1)
    )
    if gpus_per_replica > gpus_per_node:
        if gpus_per_replica % gpus_per_node != 0:
            raise ValueError(
                f"worker {Fields.TP.value}*{Fields.PP.value}*{Fields.PCP_SIZE.value}"
                f"={gpus_per_replica} spans multiple nodes but is not a multiple "
                f"of {Fields.GPUS_PER_NODE.value}={gpus_per_node}"
            )
        return gpus_per_node
    if gpus_per_node % gpus_per_replica != 0:
        raise ValueError(
            f"worker {Fields.TP.value}*{Fields.PP.value}*{Fields.PCP_SIZE.value}"
            f"={gpus_per_replica} does not divide "
            f"{Fields.GPUS_PER_NODE.value}={gpus_per_node} evenly"
        )
    return gpus_per_replica


def agentic_dram_offload_gb(
    agentic_config: dict, benchmark: dict, runner: str, runner_data: dict
) -> int:
    """Return the DRAM offload budget (GB) for one agentic server node.

    The budget scales the node's (MAX-capped) available CPU DRAM by the
    utilization and by the fraction of the node's GPUs in use:

    * Single-node entries use the TP/PP/PCP topology, which must fit one node.
    * Disaggregated multinode entries use the prefill worker's per-node GPU
      footprint, since only prefill offloads KV to CPU DRAM today (decode can be
      budgeted separately if it ever gains its own pool).
    """
    kv_offloading = benchmark.get(Fields.KV_OFFLOADING.value, "none")
    if kv_offloading != "dram":
        return 0

    available_mib = min(
        runner_available_cpu_dram_mib(runner, runner_data),
        MAX_AGENTIC_AVAILABLE_CPU_DRAM_MIB,
    )
    utilization = Decimal(str(agentic_config[Fields.DRAM_UTILIZATION.value]))
    gpus_per_node = runner_gpus_per_node(runner, runner_data)

    if Fields.WORKER.value in benchmark:
        gpu_count = worker_gpus_per_node(
            benchmark[Fields.WORKER.value], gpus_per_node)
    elif Fields.PREFILL.value in benchmark:
        gpu_count = worker_gpus_per_node(
            benchmark[Fields.PREFILL.value], gpus_per_node)
    else:
        gpu_count = effective_gpu_count(benchmark)
        if gpu_count > gpus_per_node:
            raise ValueError(
                f"tp={benchmark[Fields.TP.value]} with "
                f"{Fields.PP.value}={benchmark.get(Fields.PP.value, 1)} and "
                f"{Fields.PCP_SIZE.value}={benchmark.get(Fields.PCP_SIZE.value, 1)} "
                f"requires {gpu_count} GPUs and exceeds "
                f"{Fields.GPUS_PER_NODE.value}={gpus_per_node} for runner '{runner}'"
            )
    proportional_bytes = (
        Decimal(available_mib) * BYTES_PER_MIB * utilization
        * gpu_count / gpus_per_node
    )
    return int(proportional_bytes / BYTES_PER_GB)


def agentic_kv_offload_suffix(
    kv_offloading: str,
    kv_offload_backend: dict | None,
) -> str:
    """Return a compact exp-name suffix for agentic KV offload settings."""
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
    """Build a multinode agentic exp-name that encodes topology, not just TP."""

    def _worker_tag(worker: dict, role_prefix: str) -> str:
        ep = worker.get(Fields.EP.value, 1)
        dpa = worker.get(Fields.DP_ATTN.value, False)
        tag = (
            f"{role_prefix}{worker[Fields.NUM_WORKER.value]}"
            f"x{worker[Fields.TP.value]}"
        )
        if ep != 1:
            tag += f"ep{ep}"
        if dpa:
            tag += "dpa"
        return tag

    return (
        f"{model_code}_{_worker_tag(prefill, 'p')}_{_worker_tag(decode, 'd')}"
        f"_conc{'x'.join(str(c) for c in conc_batch)}"
        f"{offload_suffix}"
    )


def component_metadata(benchmark: dict, config: dict) -> dict:
    """Resolve optional component metadata from its validated scope."""
    metadata = {}
    for field in (Fields.ROUTER, Fields.KV_P2P_TRANSFER):
        value = benchmark.get(field.value, config.get(field.value))
        if value is not None:
            metadata[field.value] = value
    return metadata


def chunk_multinode_agentic_concurrencies(conc_values: list[int]) -> list[list[int]]:
    """Bound sequential agentic profiles sharing one server allocation."""
    size = MAX_MULTINODE_AGENTIC_CONCURRENCIES_PER_ALLOCATION
    return [conc_values[index:index + size] for index in range(0, len(conc_values), size)]


def _multinode_parallelism_key(entry: dict) -> tuple:
    """Identify a multi-node config independently of eval/concurrency fields.

    exp-name is derived from (and ignored alongside) conc: fixed-seq-len
    exp-names never embed conc, but agentic exp-names do (each concurrency
    gets its own single-conc allocation, per
    MAX_MULTINODE_AGENTIC_CONCURRENCIES_PER_ALLOCATION), so entries for the
    same topology at different concurrencies would otherwise falsely land in
    different groups.
    """
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
    return tuple(sorted(
        (key, freeze_config_value(value))
        for key, value in entry.items()
        if key not in ignored_fields
    ))


def automatic_agentic_vendor_eval(entry: dict) -> tuple[str, str] | None:
    """Return the default vendor evaluator for supported agentic models."""
    if entry.get(Fields.SCENARIO_TYPE.value) != "agentic-coding":
        return None
    return AUTOMATIC_AGENTIC_VENDOR_EVALS.get(
        entry.get(Fields.MODEL_PREFIX.value)
    )


def mark_eval_entries(matrix_values: list[dict], include_agentic: bool = False) -> list[dict]:
    """Apply the default eval selection policy.

    Kimi K3 and MiniMax M3 agentic rows use their vendor validators at every
    generated concurrency. Other agentic rows remain opt-in and use GSM8K at
    the highest concurrency in each deployment group.

    Fixed-sequence selection is unchanged: single-node 8k1k rows use the
    highest and median concurrency per model/runtime group, while multi-node
    8k1k rows use the highest eligible concurrency per parallelism topology.
    """
    from collections import defaultdict

    target_isl, target_osl = seq_len_stoi["8k1k"]
    eval_indices = set()
    mn_eval_conc = {}  # index -> chosen eval concurrency for multinode entries

    def _eligible_eval_concs(entry):
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

    # Single-node: group by (model, runner, framework, precision, isl, osl, spec-decoding, dp-attn).
    # Only 8k1k entries with a top-level TP (single-node schema).
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
        conc_values = sorted(set(e[Fields.CONC.value] for _, e in entries))
        median_conc = conc_values[len(conc_values) // 2]
        target_concs = {conc_values[-1], median_conc}
        for i, e in entries:
            if e[Fields.CONC.value] in target_concs:
                eval_indices.add(i)

    # Multi-node: group rows that differ only in concurrency, then evaluate each
    # distinct parallelism configuration at its highest configured concurrency.
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

    # Default sweeps preserve every agentic throughput result.
    if include_agentic:
        ag_sn_groups = defaultdict(list)
        # Multi-node agentic: same "highest eligible conc per distinct
        # parallelism config" policy as the fixed-seq-len mn_groups above.
        # The selected eval subset uses exactly one conc per group.
        ag_mn_groups = defaultdict(list)
        for i, entry in enumerate(matrix_values):
            if i in automatic_eval_specs:
                continue
            if entry.get(Fields.SCENARIO_TYPE.value) != 'agentic-coding':
                continue
            if Fields.PREFILL.value in entry:
                eval_concs = _eligible_eval_concs(entry)
                if not eval_concs:
                    continue
                ag_mn_groups[_multinode_parallelism_key(entry)].append((i, eval_concs[-1]))
                continue
            conc = entry[Fields.CONC.value]
            conc_val = max(conc) if isinstance(conc, list) else conc
            key = (
                entry[Fields.MODEL.value],
                entry[Fields.RUNNER.value],
                entry[Fields.FRAMEWORK.value],
                entry[Fields.PRECISION.value],
            )
            ag_sn_groups[key].append((i, conc_val))
        for entries in ag_sn_groups.values():
            eval_indices.add(max(entries, key=lambda item: item[1])[0])
        for entries in ag_mn_groups.values():
            best_idx, best_eval_conc = max(entries, key=lambda item: item[1])
            eval_indices.add(best_idx)
            mn_eval_conc[best_idx] = best_eval_conc

    for i, entry in enumerate(matrix_values):
        run_eval = i in eval_indices
        entry[Fields.RUN_EVAL.value] = run_eval
        if run_eval:
            eval_framework, eval_suite = automatic_eval_specs.get(
                i, (DEFAULT_EVAL_FRAMEWORK, "")
            )
            entry[Fields.EVAL_FRAMEWORK.value] = eval_framework
            entry[Fields.EVAL_SUITE.value] = eval_suite
        if i in mn_eval_conc:
            entry[Fields.EVAL_CONC.value] = mn_eval_conc[i]

    return matrix_values


def mark_all_eval_entries(matrix_values: list[dict]) -> list[dict]:
    """Expand eval selection across all eligible entries.

    Kimi K3 and MiniMax M3 agentic rows remain one eval job per generated
    concurrency, using the model's vendor validator. Other agentic entries use
    GSM8K through lm-eval. Their multi-node rows are merged by topology and
    select the highest resulting concurrency.

    Fixed-sequence evals only run at 8k1k. Multi-node rows with the same engine
    topology are merged into one eval row that runs every concurrency
    sequentially against the live engine.
    """
    expanded_entries: list[dict] = []
    multinode_indices: dict[tuple, int] = {}
    multinode_agentic_indices: dict[tuple, int] = {}

    target_isl, target_osl = seq_len_stoi["8k1k"]

    for entry in matrix_values:
        automatic_eval = automatic_agentic_vendor_eval(entry)
        if automatic_eval is not None:
            eval_framework, eval_suite = automatic_eval
            eval_entry = {
                **entry,
                Fields.RUN_EVAL.value: True,
                Fields.EVAL_FRAMEWORK.value: eval_framework,
                Fields.EVAL_SUITE.value: eval_suite,
            }
            if Fields.PREFILL.value in entry:
                conc = entry[Fields.CONC.value]
                conc_values = conc if isinstance(conc, list) else [conc]
                eval_entry[Fields.CONC.value] = sorted(set(conc_values))
                eval_entry[Fields.EVAL_CONC.value] = max(conc_values)
            expanded_entries.append(eval_entry)
            continue

        if entry.get(Fields.SCENARIO_TYPE.value) == 'agentic-coding':
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

        # Only 8k1k is eligible for evals; leave other sequence lengths as-is
        # (their RUN_EVAL stays False, so the evals-only filter drops them).
        if (
            entry.get(Fields.ISL.value) != target_isl
            or entry.get(Fields.OSL.value) != target_osl
        ):
            expanded_entries.append(entry)
            continue

        if Fields.PREFILL.value in entry:
            conc = entry[Fields.CONC.value]
            conc_values = conc if isinstance(conc, list) else [conc]
            parallelism_key = _multinode_parallelism_key(entry)
            if parallelism_key in multinode_indices:
                existing = expanded_entries[multinode_indices[parallelism_key]]
                existing[Fields.CONC.value] = sorted(set(
                    existing[Fields.CONC.value] + conc_values
                ))
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

    return expanded_entries


def _concurrency_range(start: int, end: int, step: int) -> list[int]:
    """Expand a validated positive range, including its end even after overshoot."""
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
) -> list[dict]:
    """Build fixed-sequence rows after the command has selected runners and points.

    Callers retain filtering and overrides; this owns row defaults, derived
    identity, topology and validation for both full-sweep and test-config.
    """
    is_multinode = config.get(Fields.MULTINODE.value, False)
    disagg = config.get(Fields.DISAGG.value, False)
    isl, osl = sequence[Fields.ISL.value], sequence[Fields.OSL.value]
    model_code = config[Fields.MODEL_PREFIX.value]
    spec_decoding = benchmark.get(Fields.SPEC_DECODING.value, "none")
    if is_multinode:
        prefill, decode = multinode_worker_pair(benchmark, disagg)
    else:
        ep = benchmark.get(Fields.EP.value)
        dp_attn = benchmark.get(Fields.DP_ATTN.value)

    entries = []
    # Multi-node rows carry the whole concurrency list; single-node rows carry
    # one point. Keep point-major, runner-minor order for existing consumers.
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
            if is_multinode:
                entry.update({
                    Fields.SPEC_DECODING.value: spec_decoding,
                    Fields.PREFILL.value: prefill,
                    Fields.DECODE.value: decode,
                    Fields.CONC.value: conc,
                    Fields.MAX_MODEL_LEN.value: isl + osl + 256,
                })
            else:
                entry.update({
                    Fields.TP.value: benchmark[Fields.TP.value],
                    Fields.PP.value: benchmark.get(Fields.PP.value, 1),
                    Fields.DCP_SIZE.value: benchmark.get(Fields.DCP_SIZE.value, 1),
                    Fields.PCP_SIZE.value: benchmark.get(Fields.PCP_SIZE.value, 1),
                    Fields.CONC.value: conc,
                    Fields.MAX_MODEL_LEN.value: isl + osl + 256,
                    Fields.EP.value: ep if ep is not None else 1,
                    Fields.DP_ATTN.value: dp_attn if dp_attn is not None else False,
                    Fields.SPEC_DECODING.value: spec_decoding,
                })
            entry.update({
                Fields.EXP_NAME.value: f"{model_code}_{seq_len_to_str(isl, osl)}",
                Fields.DISAGG.value: disagg,
                Fields.RUN_EVAL.value: False,
            })
            entry.update(component_metadata(benchmark, config))
            if is_multinode:
                add_multinode_node_count(
                    entry, runner_data, benchmark.get(Fields.NUM_NODES.value))
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
) -> list[dict]:
    """Expand one AgentX deployment for either generator command.

    Resolve topology and the offload budget before filtering concurrency so an
    invalid deployment still fails when its points are filtered out. Agentic
    bounds filter existing points; they never introduce a capped point.
    """
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
    total_cpu_dram_gb = agentic_dram_offload_gb(
        scenario, benchmark, config[Fields.RUNNER.value], runner_data)

    conc_values = benchmark.get(Fields.CONC_LIST.value)
    if not conc_values:
        conc_values = _concurrency_range(
            benchmark[Fields.CONC_START.value], benchmark[Fields.CONC_END.value], step_size)
    if min_conc is not None:
        conc_values = [c for c in conc_values if c >= min_conc]
    if max_conc is not None:
        conc_values = [c for c in conc_values if c <= max_conc]
    if conc_filter:
        conc_values = [c for c in conc_values if c in conc_filter]
    if not conc_values:
        return []

    # Multi-node batches are runner-major; single-node points are conc-major.
    if is_multinode:
        offload_suffix = (
            f"_{agentic_kv_offload_suffix(kv_offloading, kv_offload_backend)}"
            if kv_offloading != "none" else ""
        )
        points = (
            (runner, batch) for runner in runners
            for batch in chunk_multinode_agentic_concurrencies(conc_values)
        )
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
            entry.update({
                Fields.SPEC_DECODING.value: spec_decoding,
                Fields.PREFILL.value: prefill,
                Fields.DECODE.value: decode,
                Fields.CONC.value: conc,
            })
            exp_name = multinode_agentic_exp_name(
                model_code, prefill, decode, conc, offload_suffix)
        else:
            entry.update({
                Fields.TP.value: tp,
                Fields.PP.value: pp,
                Fields.DCP_SIZE.value: dcp_size,
                Fields.PCP_SIZE.value: pcp_size,
                Fields.EP.value: ep if ep is not None else 1,
                Fields.DP_ATTN.value: dp_attn if dp_attn is not None else False,
                Fields.SPEC_DECODING.value: spec_decoding,
                Fields.CONC.value: conc,
            })
            exp_name = (
                f"{model_code}_tp{tp}_conc{conc}_"
                f"{agentic_kv_offload_suffix(kv_offloading, kv_offload_backend)}"
                + (f"_spec-{spec_decoding}" if spec_decoding != "none" else "")
            )
        entry.update({
            Fields.KV_OFFLOADING.value: kv_offloading,
            Fields.TOTAL_CPU_DRAM_GB.value: total_cpu_dram_gb,
            Fields.DURATION.value: DEFAULT_AGENTIC_DURATION_SECONDS,
            Fields.EXP_NAME.value: exp_name,
        })
        if is_multinode:
            entry[Fields.DISAGG.value] = disagg
        entry[Fields.SCENARIO_TYPE.value] = "agentic-coding"
        if kv_offload_backend is not None:
            entry[Fields.KV_OFFLOAD_BACKEND.value] = kv_offload_backend
        entry.update(component_metadata(benchmark, config))
        if is_multinode:
            add_multinode_node_count(
                entry, runner_data, benchmark.get(Fields.NUM_NODES.value))
        entries.append(validate_agentic_matrix_entry(entry))
    return entries


@dataclass(frozen=True)
class FullSweepOptions:
    """Full-sweep selection and overrides, independent of the CLI parser."""

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
    args: argparse.Namespace, all_config_data: dict, runner_data: dict,
) -> list[dict]:
    """Compatibility adapter for callers passing the full-sweep CLI namespace."""
    return expand_full_sweep(all_config_data, runner_data, options=FullSweepOptions(
        model_prefix=args.model_prefix, precision=args.precision,
        framework=args.framework, runner_type=args.runner_type,
        seq_lens=args.seq_lens, scenario_types=getattr(args, "scenario_type", None),
        runner_node_filter=args.runner_node_filter,
        single_node=args.single_node, multi_node=args.multi_node,
        step_size=args.step_size, min_conc=args.min_conc, max_conc=args.max_conc,
        max_tp=args.max_tp, max_ep=args.max_ep,
    ))


def expand_full_sweep(
    master_config: dict, runner_data: dict, *,
    options: FullSweepOptions = FullSweepOptions(),
) -> list[dict]:
    """Expand validated configs in declaration order, before eval selection."""
    if options.step_size <= 1:
        raise ValueError("step_size must be greater than 1")
    if (
        options.min_conc is not None and options.max_conc is not None
        and options.min_conc > options.max_conc
    ):
        raise ValueError("min_conc must be less than or equal to max_conc")
    if options.runner_type:
        valid_runner_types = set(runner_labels(runner_data))
        invalid_runners = set(options.runner_type) - valid_runner_types
        if invalid_runners:
            raise ValueError(
                f"Invalid runner type(s): {invalid_runners}. "
                f"Valid runner types are: {', '.join(sorted(valid_runner_types))}")

    # Full-sweep validates sequence names even when no config matches.
    seq_filter = {seq_len_stoi[sl] for sl in options.seq_lens} if options.seq_lens else None
    configs = (
        config for key, config in master_config.items()
        if (not options.model_prefix or any(key.startswith(p) for p in options.model_prefix))
        and (not options.precision or config[Fields.PRECISION.value] in options.precision)
        and (not options.framework or config[Fields.FRAMEWORK.value] in options.framework)
        and (not options.runner_type or config[Fields.RUNNER.value] in options.runner_type)
    )
    return _expand_configs(
        configs, runner_data, seq_filter=seq_filter,
        scenario_types=options.scenario_types,
        runner_node_filter=options.runner_node_filter, full_sweep=options,
    )


def _fixed_sequence_concurrencies(
    benchmark: dict, *, multinode: bool, full_sweep: FullSweepOptions | None,
    concurrencies: list[int] | None,
) -> list[int] | None:
    """Keep full-sweep capping separate from selected-key intersection.

    Single-node ranges are clipped before expansion; multi-node ranges and
    explicit lists are expanded first. AgentX only filters its existing points.
    None skips a deployment; an empty list retains the legacy empty batch.
    """
    if full_sweep is None:
        values = (
            benchmark[Fields.CONC_LIST.value] if Fields.CONC_LIST.value in benchmark
            else _concurrency_range(
                benchmark[Fields.CONC_START.value], benchmark[Fields.CONC_END.value], 2)
        )
        return ([c for c in values if c in concurrencies] or None) if concurrencies else values

    minimum, maximum = full_sweep.min_conc, full_sweep.max_conc
    values = benchmark.get(Fields.CONC_LIST.value)
    if not values:
        start, end = benchmark[Fields.CONC_START.value], benchmark[Fields.CONC_END.value]
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
    runner: str, runner_data: dict, runner_node_filter: str | None, *,
    include_label: bool = True,
) -> list[str]:
    if not runner_node_filter:
        return [runner]
    candidates = runner_nodes_for_label(runner, runner_data)
    if not include_label:
        # Full-sweep expands concrete nodes in inventory order, including repeats.
        return [node for node in candidates if runner_node_filter in node]
    if runner_node_filter in runner:
        candidates = [runner, *candidates]
    return list(dict.fromkeys(node for node in candidates if runner_node_filter in node))


def generate_test_config_sweep(
    args: argparse.Namespace, all_config_data: dict, runner_data: dict | None = None,
) -> list[dict]:
    """Compatibility API for selected-key expansion without eval selection."""
    return _expand_selected_configs(
        args.config_keys, all_config_data, runner_data,
        runner_node_filter=getattr(args, "runner_node_filter", None),
        seq_lens=getattr(args, "seq_lens", None),
        scenario_types=getattr(args, "scenario_type", None),
        concurrencies=getattr(args, "conc", None),
    )


def _expand_selected_configs(
    config_keys: list[str], all_config_data: dict, runner_data: dict | None,
    *, runner_node_filter: str | None = None, seq_lens: list[str] | None = None,
    scenario_types: tuple[str, ...] | list[str] | None = None,
    concurrencies: list[int] | None = None,
) -> list[dict]:
    resolved_keys = expand_config_keys(config_keys, all_config_data.keys())
    return _expand_configs(
        (all_config_data[key] for key in resolved_keys), runner_data or {},
        runner_node_filter=runner_node_filter, seq_lens=seq_lens,
        scenario_types=scenario_types, concurrencies=concurrencies,
    )


def _expand_configs(
    configs: Iterable[dict], runner_data: dict, *,
    runner_node_filter: str | None = None, seq_lens: list[str] | None = None,
    seq_filter: set[tuple[int, int]] | None = None,
    scenario_types: tuple[str, ...] | list[str] | None = None,
    concurrencies: list[int] | None = None,
    full_sweep: FullSweepOptions | None = None,
) -> list[dict]:
    """Traverse configs and scenarios once for both generation commands.

    Selection retains each command's ordering and overrides. Row builders own
    identity, topology and validation; eval policy is applied by the caller.
    """
    rows = []
    for config in configs:
        multinode = config.get(Fields.MULTINODE.value, False)
        # Full-sweep reads scenarios before runner filtering; selected keys defer it.
        if full_sweep is not None:
            scenarios = config[Fields.SCENARIOS.value]
            fixed = (
                scenarios.get(Fields.FIXED_SEQ_LEN.value, [])
                if not scenario_types or "fixed-seq-len" in scenario_types else []
            )
        runners = _runner_values_for_filter(
            config[Fields.RUNNER.value], runner_data, runner_node_filter,
            include_label=full_sweep is None,
        )
        if not runners:
            continue
        if seq_lens:
            seq_filter = {seq_len_stoi[sl] for sl in seq_lens}
        if full_sweep is None:
            fixed = (
                config[Fields.SCENARIOS.value].get(Fields.FIXED_SEQ_LEN.value, [])
                if not scenario_types or "fixed-seq-len" in scenario_types else []
            )
        node_allowed = full_sweep is None or (
            full_sweep.multi_node if multinode else full_sweep.single_node)

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
                    benchmark, multinode=multinode, full_sweep=full_sweep,
                    concurrencies=concurrencies,
                )
                if values is None:
                    continue
                rows.extend(_fixed_sequence_entries(
                    config, benchmark, sequence, values, runners, runner_data))

        agentic = (
            config[Fields.SCENARIOS.value].get(Fields.AGENTIC_CODING.value, [])
            if not scenario_types or "agentic-coding" in scenario_types else []
        )
        if not node_allowed:
            continue
        for scenario in agentic:
            for benchmark in scenario[Fields.SEARCH_SPACE.value]:
                rows.extend(_agentic_entries(
                    config, benchmark, scenario, runners, runner_data,
                    step_size=full_sweep.step_size if full_sweep is not None else 2,
                    min_conc=full_sweep.min_conc if full_sweep is not None else None,
                    max_conc=full_sweep.max_conc if full_sweep is not None else None,
                    conc_filter=concurrencies,
                ))
    return rows


def expand_config_keys(config_keys, available_keys):
    """Expand config key patterns (glob wildcards) against available keys.

    Keys containing '*' or '?' are treated as glob patterns and expanded via
    fnmatch.filter(). Plain keys are validated for existence. Results are
    deduplicated while preserving order.

    Raises ValueError if a pattern matches nothing or an exact key is missing.
    """
    available = list(available_keys)
    seen = {}  # use dict to preserve insertion order
    for key in config_keys:
        if '*' in key or '?' in key:
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
    """Select exact generated experiment identities and reject ambiguity."""
    requested = set(exp_names)
    if len(requested) != len(exp_names):
        raise ValueError("--exp-names contains duplicate values")

    matches: dict[str, int] = {name: 0 for name in exp_names}
    for entry in entries:
        exp_name = entry.get(Fields.EXP_NAME.value)
        if exp_name in matches:
            matches[exp_name] += 1

    missing = sorted(name for name, count in matches.items() if count == 0)
    ambiguous = sorted(name for name, count in matches.items() if count > 1)
    if missing:
        raise ValueError("Experiment name(s) not found: " + ", ".join(missing))
    if ambiguous:
        raise ValueError(
            "Experiment name(s) matched multiple rows: " + ", ".join(ambiguous)
        )
    return [
        entry
        for entry in entries
        if entry.get(Fields.EXP_NAME.value) in requested
    ]


def apply_node_type_defaults(args):
    """Default both single_node and multi_node to True when neither is specified."""
    if hasattr(args, 'single_node') and hasattr(args, 'multi_node'):
        if not args.single_node and not args.multi_node:
            args.single_node = True
            args.multi_node = True
    return args


EvalMode = Literal["default", "none", "subset", "all", "smoke"]


def select_matrix_evals(
    rows: list[dict], *, mode: EvalMode = "default", trim: bool = False,
) -> list[dict]:
    """Apply eval policy and optional trimming to freshly generated rows."""
    if mode not in ("default", "none", "subset", "all", "smoke"):
        raise ValueError(f"Unknown eval mode: {mode!r}")
    if mode == "smoke" and trim:
        raise ValueError("smoke cannot be combined with trimming")
    if mode != "none":
        rows = mark_eval_entries(rows, include_agentic=mode in ("subset", "all", "smoke"))
        if mode == "all":
            rows = mark_all_eval_entries(rows)
    if mode == "smoke":
        return smoke_entries(rows)
    if trim:
        rows = trim_conc(rows)
    if mode in ("subset", "all"):
        rows = [row for row in rows if row.get(Fields.RUN_EVAL.value, False)]
        for row in rows:
            row[Fields.EVAL_ONLY.value] = True
    return rows


def generate_config_matrix(
    config_keys: list[str], master_config: dict, runner_data: dict,
    *, scenario_types: tuple[str, ...] | list[str] | None = None,
    eval_mode: EvalMode = "default",
) -> list[dict]:
    """Build selected configs and evals without a generator subprocess.

    Every call builds independent rows. The caller loads master/runner inputs;
    node-count resolution still reads checked-in recipes. Default,
    throughput-only, subset-only, all-eval, and smoke modes use the same policy as the CLI.
    """
    rows = _expand_selected_configs(
        config_keys, master_config, runner_data, scenario_types=scenario_types,
    )
    rows = select_matrix_evals(rows, mode=eval_mode)
    # Retain the former JSON boundary: values and nested objects cannot leak
    # between generation passes, and unsupported values still reject.
    return json.loads(json.dumps(rows))


def main():
    # Create parent parser with common arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        '--config-files',
        nargs='+',
        required=True,
        help='One or more configuration files (YAML format)'
    )
    parent_parser.add_argument(
        '--runner-config',
        default='configs/runners.yaml',
        help='Configuration file holding runner information (YAML format, defaults to configs/runners.yaml)'
    )
    eval_group = parent_parser.add_mutually_exclusive_group()
    eval_group.add_argument(
        '--no-evals',
        action='store_true',
        help='When specified, skip evals (throughput benchmarks only).'
    )
    eval_group.add_argument(
        '--evals-only',
        action='store_true',
        help='When specified, run ONLY the eval subset (excludes non-eval configs).'
    )
    parent_parser.add_argument(
        '--all-evals',
        action='store_true',
        help=(
            'Expand eval selection to every generated fixed-sequence config. '
            'Can be combined with --evals-only; used alone, it also emits eval-only jobs.'
        )
    )
    parent_parser.add_argument(
        '--smoke',
        action='store_true',
        help='Minimum-concurrency throughput plus canonical representative evals.'
    )
    parent_parser.add_argument(
        '--trim-conc',
        action='store_true',
        help=(
            'Trim each generated deployment shape to its minimum concurrency '
            'after applying eval selection.'
        )
    )
    parent_parser.add_argument(
        '--runner-node-filter',
        required=False,
        help='Filter runner nodes by substring match (e.g., "amd" to only include nodes containing that string). Expands each config to individual matching nodes.'
    )
    parent_parser.add_argument(
        '--scenario-type',
        nargs='+',
        choices=['fixed-seq-len', 'agentic-coding'],
        required=False,
        help='Scenario type(s) to include. If not specified, all scenario types are generated.'
    )

    # Create main parser
    parser = argparse.ArgumentParser(
        description='Generate benchmark configurations from YAML config files'
    )

    # Create subparsers for subcommands
    subparsers = parser.add_subparsers(
        dest='command',
        required=True,
        help='Available commands'
    )

    # Subcommand: full-sweep
    full_sweep_parser = subparsers.add_parser(
        'full-sweep',
        parents=[parent_parser],
        add_help=False,
        help='Generate full sweep configurations with optional filtering by model, precision, framework, runner type, and sequence lengths'
    )
    full_sweep_parser.add_argument(
        '--model-prefix',
        nargs='+',
        required=False,
        help='Model prefix(es) to filter configurations (optional, can specify multiple)'
    )
    full_sweep_parser.add_argument(
        '--precision',
        nargs='+',
        required=False,
        help='Precision(s) to filter by (e.g., fp4, fp8) (optional, can specify multiple)'
    )
    full_sweep_parser.add_argument(
        '--framework',
        nargs='+',
        required=False,
        help='Framework(s) to filter by (e.g., vllm, trt, sglang) (optional, can specify multiple)'
    )
    full_sweep_parser.add_argument(
        '--runner-type',
        nargs='+',
        required=False,
        help='Runner type(s) to filter by (e.g., h200, h100) (optional, can specify multiple)'
    )
    full_sweep_parser.add_argument(
        '--seq-lens',
        nargs='+',
        choices=list(seq_len_stoi.keys()),
        required=False,
        help=f"Sequence length configurations to include: {', '.join(seq_len_stoi.keys())}. If not specified, all sequence lengths are included."
    )
    full_sweep_parser.add_argument(
        '--step-size',
        type=int,
        default=2,
        help='Step size for concurrency values (default: 2)'
    )
    full_sweep_parser.add_argument(
        '--min-conc',
        type=int,
        required=False,
        help='Minimum concurrency value to include (filters out lower concurrency values)'
    )
    full_sweep_parser.add_argument(
        '--max-conc',
        type=int,
        required=False,
        help='Maximum concurrency value to include (filters out higher concurrency values)'
    )
    full_sweep_parser.add_argument(
        '--max-tp',
        type=int,
        required=False,
        help='Maximum tensor parallelism value to include (single-node only)'
    )
    full_sweep_parser.add_argument(
        '--max-ep',
        type=int,
        required=False,
        help='Maximum expert parallelism value to include (single-node only)'
    )
    full_sweep_parser.add_argument(
        '--single-node',
        action='store_true',
        help='Only generate single-node configurations. If neither --single-node nor --multi-node is specified, both types are generated.'
    )
    full_sweep_parser.add_argument(
        '--multi-node',
        action='store_true',
        help='Only generate multi-node configurations. If neither --single-node nor --multi-node is specified, both types are generated.'
    )
    full_sweep_parser.add_argument(
        '-h', '--help',
        action='help',
        help='Show this help message and exit'
    )

    # Subcommand: test-config
    test_config_keys_parser = subparsers.add_parser(
        'test-config',
        parents=[parent_parser],
        add_help=False,
        help='Generate full sweep for specific config keys. Validates that all specified keys exist before generating.'
    )
    test_config_keys_parser.add_argument(
        '--config-keys',
        nargs='+',
        required=True,
        help='One or more config keys to generate sweep for (e.g., dsr1-fp4-b200-sglang dsr1-fp8-h200-trt)'
    )
    test_config_keys_parser.add_argument(
        '--conc',
        nargs='+',
        type=int,
        required=False,
        help='Only include these concurrency values. Values must exist in the config conc-range/list.'
    )
    test_config_keys_parser.add_argument(
        '--exp-names',
        nargs='+',
        required=False,
        help=(
            'Only include exact generated experiment names. Each name must '
            'match exactly one row after config and concurrency filtering.'
        )
    )
    test_config_keys_parser.add_argument(
        '--seq-lens',
        nargs='+',
        choices=list(seq_len_stoi.keys()),
        required=False,
        help='Only include these sequence length configurations (e.g., 1k1k 8k1k)'
    )
    test_config_keys_parser.add_argument(
        '-h', '--help',
        action='help',
        help='Show this help message and exit'
    )

    args = parser.parse_args()
    apply_node_type_defaults(args)
    if args.command == 'full-sweep' and args.step_size <= 1:
        parser.error("--step-size must be greater than 1")
    if (
        args.command == 'full-sweep'
        and args.min_conc is not None
        and args.max_conc is not None
        and args.min_conc > args.max_conc
    ):
        parser.error("--min-conc must be less than or equal to --max-conc")
    if args.no_evals and args.all_evals:
        parser.error("--all-evals cannot be combined with --no-evals")

    # Load and validate configuration files (validation happens by default in load functions)
    all_config_data = load_config_files(args.config_files)
    runner_data = load_runner_file(args.runner_config)

    # Route to appropriate function based on subcommand
    if args.command == 'full-sweep':
        matrix_values = generate_full_sweep(args, all_config_data, runner_data)
    elif args.command == 'test-config':
        matrix_values = generate_test_config_sweep(args, all_config_data, runner_data)
    else:
        parser.error(f"Unknown command: {args.command}")

    if args.command == 'test-config' and args.exp_names:
        try:
            matrix_values = filter_exp_names(matrix_values, args.exp_names)
        except ValueError as error:
            parser.error(str(error))
        

    if args.smoke and (args.trim_conc or args.no_evals or args.evals_only or args.all_evals):
        parser.error('--smoke cannot be combined with trimming or eval overrides')

    matrix_values = select_matrix_evals(
        matrix_values,
        mode=("smoke" if args.smoke else "none" if args.no_evals else "all" if args.all_evals
              else "subset" if args.evals_only else "default"),
        trim=args.trim_conc,
    )

    print(json.dumps(matrix_values))
    return matrix_values


if __name__ == "__main__":
    main()
