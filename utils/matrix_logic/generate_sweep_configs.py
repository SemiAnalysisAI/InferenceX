import fnmatch
import json
import argparse
import sys
from decimal import Decimal
from pathlib import Path

# Ensure sibling modules are importable regardless of how script is invoked
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validation import (
    validate_matrix_entry,
    validate_agentic_matrix_entry,
    load_config_files,
    load_runner_file,
    Fields,
    DEFAULT_AGENTIC_DURATION_SECONDS,
)

seq_len_stoi = {
    "1k1k": (1024, 1024),
    "8k1k": (8192, 1024)
}

MIN_EVAL_CONC = 16
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

    if Fields.PREFILL.value in benchmark:
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


def _freeze_matrix_value(value):
    """Convert nested matrix values into hashable equivalents."""
    if isinstance(value, dict):
        return tuple(sorted(
            (key, _freeze_matrix_value(item))
            for key, item in value.items()
        ))
    if isinstance(value, list):
        return tuple(_freeze_matrix_value(item) for item in value)
    return value


def _multinode_parallelism_key(entry: dict) -> tuple:
    """Identify a multi-node config independently of eval/concurrency fields."""
    ignored_fields = {
        Fields.CONC.value,
        Fields.RUN_EVAL.value,
        Fields.EVAL_ONLY.value,
        Fields.EVAL_CONC.value,
        Fields.EVAL_ALL_CONCS.value,
    }
    return tuple(sorted(
        (key, _freeze_matrix_value(value))
        for key, value in entry.items()
        if key not in ignored_fields
    ))


def mark_eval_entries(matrix_values: list[dict], include_agentic: bool = False) -> list[dict]:
    """Eval selection policy:
    - Single-node: only consider 8k1k (isl=8192, osl=1024).
      For each unique (model, runner, framework, precision, isl, osl, spec-decoding, dp-attn):
        - Ignore entries with conc < MIN_EVAL_CONC
        - Mark all entries at the highest CONC (all TPs)
        - Mark all entries at the median CONC (all TPs)
    - Multi-node: only consider 8k1k entries. For every distinct parallelism
      configuration:
        - Ignore entries with all conc values < MIN_EVAL_CONC
        - Mark the entry containing its highest eligible concurrency
        - Set eval-conc to that highest eligible concurrency
    - Agentic evals are opt-in to preserve default throughput coverage.
    """
    from collections import defaultdict

    target_isl, target_osl = seq_len_stoi["8k1k"]
    eval_indices = set()
    mn_eval_conc = {}  # index -> chosen eval concurrency for multinode entries

    def _eligible_eval_concs(entry):
        conc = entry[Fields.CONC.value]
        conc_values = conc if isinstance(conc, list) else [conc]
        return sorted(c for c in conc_values if c >= MIN_EVAL_CONC)

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
        for i, entry in enumerate(matrix_values):
            if entry.get(Fields.SCENARIO_TYPE.value) != 'agentic-coding':
                continue
            # Multi-node agentic eval is unsupported.
            if Fields.PREFILL.value in entry:
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

    for i, entry in enumerate(matrix_values):
        entry[Fields.RUN_EVAL.value] = i in eval_indices
        if i in mn_eval_conc:
            entry[Fields.EVAL_CONC.value] = mn_eval_conc[i]

    return matrix_values


def mark_all_eval_entries(matrix_values: list[dict]) -> list[dict]:
    """Expand eval selection to every 8k1k fixed-sequence entry.

    Evals only run at 8k1k (matching mark_eval_entries), so entries at other
    sequence lengths (e.g. 1k1k) are passed through untouched rather than
    expanded into eval rows.
    Single-node agentic entries use SWE-bench; multi-node eval is unsupported.
    Multi-node rows with the same engine topology are merged into one eval row
    whose full concurrency list is run sequentially against the same engine.
    """
    expanded_entries: list[dict] = []
    multinode_indices: dict[tuple, int] = {}

    target_isl, target_osl = seq_len_stoi["8k1k"]

    for entry in matrix_values:
        if entry.get(Fields.SCENARIO_TYPE.value) == 'agentic-coding':
            if Fields.PREFILL.value not in entry:
                entry[Fields.RUN_EVAL.value] = True
            expanded_entries.append(entry)
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

    return expanded_entries


def generate_full_sweep(args, all_config_data, runner_data):
    """Generate full sweep configurations with optional filtering.

    Supports filtering by model prefix, precision, framework, runner type, sequence lengths,
    and max concurrency.

    All filters are optional - can generate sweeps for all configs or filter by specific criteria.

    Assumes all_config_data has been validated by validate_master_config().
    """
    if args.step_size <= 1:
        raise ValueError("step_size must be greater than 1")
    if (
        args.min_conc is not None
        and args.max_conc is not None
        and args.min_conc > args.max_conc
    ):
        raise ValueError("min_conc must be less than or equal to max_conc")

    # Validate runner types if specified
    if args.runner_type:
        valid_runner_types = set(runner_labels(runner_data).keys())
        invalid_runners = set(args.runner_type) - valid_runner_types
        if invalid_runners:
            raise ValueError(
                f"Invalid runner type(s): {invalid_runners}. "
                f"Valid runner types are: {', '.join(sorted(valid_runner_types))}")

    matrix_values = []

    # Convert seq-lens to set of (isl, osl) tuples for filtering
    seq_lens_filter = None
    if args.seq_lens:
        seq_lens_filter = {seq_len_stoi[sl] for sl in args.seq_lens}

    # Iterate through all configurations and apply filters as specified (this is just "selecting" 
    # configs from all of the master configs subject to some pattern matching)
    for key, val in all_config_data.items():
        # Filter by model prefix if specified
        if args.model_prefix:
            if not any(key.startswith(prefix) for prefix in args.model_prefix):
                continue

        # Filter by precision if specified
        if args.precision and val[Fields.PRECISION.value] not in args.precision:
            continue

        # Filter by framework if specified
        if args.framework and val[Fields.FRAMEWORK.value] not in args.framework:
            continue

        # Filter by runner type if specified
        if args.runner_type and val[Fields.RUNNER.value] not in args.runner_type:
            continue

        # Check if this is a multinode config
        is_multinode = val.get(Fields.MULTINODE.value, False)
        # Get disagg value, defaulting to False if not specified
        disagg = val.get(Fields.DISAGG.value, False)

        scenarios = val[Fields.SCENARIOS.value]
        scenario_filter = set(args.scenario_type) if getattr(args, 'scenario_type', None) else None
        seq_len_configs = scenarios.get(Fields.FIXED_SEQ_LEN.value, []) if (scenario_filter is None or 'fixed-seq-len' in scenario_filter) else []
        image = val[Fields.IMAGE.value]
        model = val[Fields.MODEL.value]
        precision = val[Fields.PRECISION.value]
        framework = val[Fields.FRAMEWORK.value]
        runner = val[Fields.RUNNER.value]
        model_code = val[Fields.MODEL_PREFIX.value]

        # Compute filtered runner nodes for this config if filter is specified
        runner_nodes_to_use = None
        if args.runner_node_filter:
            runner_nodes = runner_nodes_for_label(runner, runner_data)
            runner_nodes_to_use = [
                node for node in runner_nodes if args.runner_node_filter in node]
            if not runner_nodes_to_use:
                # No matching nodes for this config's runner type, skip this config
                continue

        for seq_config in seq_len_configs:
            isl = seq_config[Fields.ISL.value]
            osl = seq_config[Fields.OSL.value]

            # Filter by sequence lengths if specified
            if seq_lens_filter and (isl, osl) not in seq_lens_filter:
                continue

            bmk_space = seq_config[Fields.SEARCH_SPACE.value]

            for bmk in bmk_space:
                # Skip configs that don't match the requested node type
                if is_multinode and not args.multi_node:
                    continue
                if not is_multinode and not args.single_node:
                    continue

                if is_multinode:
                    # Multinode configuration
                    # spec_decoding defaults to "none" if not specified
                    spec_decoding = bmk.get(Fields.SPEC_DECODING.value, "none")

                    prefill = with_worker_parallelism_defaults(
                        bmk[Fields.PREFILL.value])
                    decode = with_worker_parallelism_defaults(
                        bmk[Fields.DECODE.value])

                    # Get concurrency values (can be list or range)
                    conc_list = bmk.get(Fields.CONC_LIST.value)
                    # If it's a list
                    if conc_list:
                        conc_values = conc_list
                    # If it's a range
                    else:
                        conc_start = bmk[Fields.CONC_START.value]
                        conc_end = bmk[Fields.CONC_END.value]
                        conc_values = []
                        conc = conc_start
                        while conc <= conc_end:
                            conc_values.append(conc)
                            if conc == conc_end:
                                break
                            conc *= args.step_size
                            if conc > conc_end:
                                conc = conc_end

                    # Apply min-conc filter if specified
                    if args.min_conc is not None:
                        if args.min_conc <= 0:
                            continue  # Skip if min_conc is not positive
                        conc_values = [c for c in conc_values if c >= args.min_conc]
                        if not conc_values:
                            continue  # Skip if no values meet the min_conc requirement

                    # Apply max-conc filter if specified
                    # If max_conc is less than all values, use max_conc directly (if valid)
                    if args.max_conc is not None:
                        filtered_conc = [c for c in conc_values if c <= args.max_conc]
                        if not filtered_conc:
                            # No existing values <= max_conc, so use max_conc directly if valid
                            if args.max_conc > 0:
                                conc_values = [args.max_conc]
                            else:
                                continue  # Skip if max_conc is not positive
                        else:
                            conc_values = filtered_conc

                    seq_len_str = seq_len_to_str(isl, osl)
                    runners_for_entry = runner_nodes_to_use if runner_nodes_to_use else [runner]

                    for runner_value in runners_for_entry:
                        entry = {
                            Fields.IMAGE.value: image,
                            Fields.MODEL.value: model,
                            Fields.MODEL_PREFIX.value: model_code,
                            Fields.PRECISION.value: precision,
                            Fields.FRAMEWORK.value: framework,
                            Fields.RUNNER.value: runner_value,
                            Fields.ISL.value: isl,
                            Fields.OSL.value: osl,
                            Fields.SPEC_DECODING.value: spec_decoding,
                            Fields.PREFILL.value: prefill,
                            Fields.DECODE.value: decode,
                            Fields.CONC.value: conc_values,  # Pass the entire list for multinode
                            Fields.MAX_MODEL_LEN.value: isl + osl + 256,
                            Fields.EXP_NAME.value: f"{model_code}_{seq_len_str}",
                            Fields.DISAGG.value: disagg,
                            Fields.RUN_EVAL.value: False,  # Default, may be overridden by mark_eval_entries
                        }
                        entry.update(component_metadata(bmk, val))

                        validate_matrix_entry(entry, is_multinode)
                        matrix_values.append(entry)
                else:
                    # Single-node configuration
                    tp = bmk[Fields.TP.value]
                    pp = bmk.get(Fields.PP.value, 1)
                    dcp_size = bmk.get(Fields.DCP_SIZE.value, 1)
                    pcp_size = bmk.get(Fields.PCP_SIZE.value, 1)
                    ep = bmk.get(Fields.EP.value)
                    dp_attn = bmk.get(Fields.DP_ATTN.value)
                    spec_decoding = bmk.get(Fields.SPEC_DECODING.value, "none")

                    # Apply max-tp filter if specified
                    if args.max_tp is not None:
                        if args.max_tp <= 0:
                            continue  # Skip if max_tp is not positive
                        if tp > args.max_tp:
                            continue

                    # Apply max-ep filter if specified
                    # If ep > max_ep, use max_ep instead of skipping (if valid)
                    if args.max_ep is not None:
                        if args.max_ep <= 0:
                            continue  # Skip if max_ep is not positive
                        if ep is not None and ep > args.max_ep:
                            ep = args.max_ep

                    conc_list = bmk.get(Fields.CONC_LIST.value)
                    if conc_list:
                        conc_values = list(conc_list)

                        if args.min_conc is not None:
                            if args.min_conc <= 0:
                                continue
                            conc_values = [
                                conc for conc in conc_values
                                if conc >= args.min_conc
                            ]
                            if not conc_values:
                                continue

                        if args.max_conc is not None:
                            if args.max_conc <= 0:
                                continue
                            filtered_conc = [
                                conc for conc in conc_values
                                if conc <= args.max_conc
                            ]
                            conc_values = (
                                filtered_conc
                                if filtered_conc
                                else [args.max_conc]
                            )
                    else:
                        conc_start = bmk[Fields.CONC_START.value]
                        conc_end = bmk[Fields.CONC_END.value]

                        # If conc_end < min_conc, skip this config entirely.
                        if args.min_conc is not None:
                            if args.min_conc <= 0:
                                continue
                            if conc_end < args.min_conc:
                                continue
                            conc_start = max(conc_start, args.min_conc)

                        # If conc_start > max_conc, use max_conc directly.
                        if args.max_conc is not None:
                            if args.max_conc <= 0:
                                continue
                            if conc_start > args.max_conc:
                                conc_start = args.max_conc
                                conc_end = args.max_conc
                            else:
                                conc_end = min(conc_end, args.max_conc)

                        conc_values = []
                        conc = conc_start
                        while conc <= conc_end:
                            conc_values.append(conc)
                            if conc == conc_end:
                                break
                            conc *= args.step_size
                            if conc > conc_end:
                                conc = conc_end

                    seq_len_str = seq_len_to_str(isl, osl)
                    runners_for_entry = runner_nodes_to_use if runner_nodes_to_use else [runner]

                    for conc in conc_values:
                        for runner_value in runners_for_entry:
                            entry = {
                                Fields.IMAGE.value: image,
                                Fields.MODEL.value: model,
                                Fields.MODEL_PREFIX.value: model_code,
                                Fields.PRECISION.value: precision,
                                Fields.FRAMEWORK.value: framework,
                                Fields.RUNNER.value: runner_value,
                                Fields.ISL.value: isl,
                                Fields.OSL.value: osl,
                                Fields.TP.value: tp,
                                Fields.PP.value: pp,
                                Fields.DCP_SIZE.value: dcp_size,
                                Fields.PCP_SIZE.value: pcp_size,
                                Fields.CONC.value: conc,
                                Fields.MAX_MODEL_LEN.value: isl + osl + 256,
                                Fields.EP.value: 1,  # Default
                                Fields.DP_ATTN.value: False,  # Default
                                Fields.SPEC_DECODING.value: spec_decoding,
                                Fields.EXP_NAME.value: f"{model_code}_{seq_len_str}",
                                Fields.DISAGG.value: disagg,
                                Fields.RUN_EVAL.value: False,  # Default, may be overridden by mark_eval_entries
                            }

                            if ep is not None:
                                entry[Fields.EP.value] = ep
                            if dp_attn is not None:
                                entry[Fields.DP_ATTN.value] = dp_attn

                            entry.update(component_metadata(bmk, val))
                            validate_matrix_entry(entry, is_multinode)
                            matrix_values.append(entry)

        # ---- Agentic-coding scenarios ----
        agentic_configs = scenarios.get(Fields.AGENTIC_CODING.value, []) if (scenario_filter is None or 'agentic-coding' in scenario_filter) else []
        if is_multinode and not args.multi_node:
            continue
        if not is_multinode and not args.single_node:
            continue

        for agentic_config in agentic_configs:
            bmk_space = agentic_config[Fields.SEARCH_SPACE.value]
            duration = DEFAULT_AGENTIC_DURATION_SECONDS

            for bmk in bmk_space:
                if is_multinode:
                    prefill = with_worker_parallelism_defaults(
                        bmk[Fields.PREFILL.value])
                    decode = with_worker_parallelism_defaults(
                        bmk[Fields.DECODE.value])
                    spec_decoding = bmk.get(Fields.SPEC_DECODING.value, "none")
                    kv_offloading = bmk.get(Fields.KV_OFFLOADING.value, "none")
                    kv_offload_backend = bmk.get(Fields.KV_OFFLOAD_BACKEND.value)
                else:
                    tp = bmk[Fields.TP.value]
                    pp = bmk.get(Fields.PP.value, 1)
                    dcp_size = bmk.get(Fields.DCP_SIZE.value, 1)
                    pcp_size = bmk.get(Fields.PCP_SIZE.value, 1)
                    ep = bmk.get(Fields.EP.value)
                    dp_attn = bmk.get(Fields.DP_ATTN.value)
                    spec_decoding = bmk.get(Fields.SPEC_DECODING.value, "none")
                    kv_offloading = bmk[Fields.KV_OFFLOADING.value]
                    kv_offload_backend = bmk.get(Fields.KV_OFFLOAD_BACKEND.value)
                total_cpu_dram_gb = agentic_dram_offload_gb(
                    agentic_config, bmk, runner, runner_data)

                # Get concurrency values
                conc_list = bmk.get(Fields.CONC_LIST.value)
                if conc_list:
                    conc_values = conc_list
                else:
                    conc_start = bmk[Fields.CONC_START.value]
                    conc_end = bmk[Fields.CONC_END.value]
                    conc_values = []
                    conc = conc_start
                    while conc <= conc_end:
                        conc_values.append(conc)
                        if conc == conc_end:
                            break
                        conc *= args.step_size
                        if conc > conc_end:
                            conc = conc_end

                # Apply conc filters
                if args.min_conc is not None:
                    conc_values = [c for c in conc_values if c >= args.min_conc]
                if args.max_conc is not None:
                    conc_values = [c for c in conc_values if c <= args.max_conc]
                if not conc_values:
                    continue

                runners_for_entry = runner_nodes_to_use if runner_nodes_to_use else [runner]

                if is_multinode:
                    # Preserve historical exp-names for the default (no offload)
                    # case; only append a suffix when KV offloading is active.
                    offload_suffix = (
                        f"_{agentic_kv_offload_suffix(kv_offloading, kv_offload_backend)}"
                        if kv_offloading != "none"
                        else ""
                    )
                    for runner_value in runners_for_entry:
                        for conc_batch in chunk_multinode_agentic_concurrencies(conc_values):
                            entry = {
                                Fields.IMAGE.value: image,
                                Fields.MODEL.value: model,
                                Fields.MODEL_PREFIX.value: model_code,
                                Fields.PRECISION.value: precision,
                                Fields.FRAMEWORK.value: framework,
                                Fields.RUNNER.value: runner_value,
                                Fields.SPEC_DECODING.value: spec_decoding,
                                Fields.PREFILL.value: prefill,
                                Fields.DECODE.value: decode,
                                Fields.CONC.value: conc_batch,
                                Fields.KV_OFFLOADING.value: kv_offloading,
                                Fields.TOTAL_CPU_DRAM_GB.value: total_cpu_dram_gb,
                                Fields.DURATION.value: duration,
                                Fields.EXP_NAME.value: (
                                    f"{model_code}_p{prefill[Fields.NUM_WORKER.value]}x{prefill[Fields.TP.value]}"
                                    f"_d{decode[Fields.NUM_WORKER.value]}x{decode[Fields.TP.value]}"
                                    f"_conc{'x'.join(str(c) for c in conc_batch)}"
                                    f"{offload_suffix}"
                                ),
                                Fields.DISAGG.value: disagg,
                                Fields.SCENARIO_TYPE.value: "agentic-coding",
                            }
                            if kv_offload_backend is not None:
                                entry[Fields.KV_OFFLOAD_BACKEND.value] = kv_offload_backend
                            entry.update(component_metadata(bmk, val))
                            validate_agentic_matrix_entry(entry)
                            matrix_values.append(entry)
                else:
                    for conc in conc_values:
                        for runner_value in runners_for_entry:
                            entry = {
                                Fields.IMAGE.value: image,
                                Fields.MODEL.value: model,
                                Fields.MODEL_PREFIX.value: model_code,
                                Fields.PRECISION.value: precision,
                                Fields.FRAMEWORK.value: framework,
                                Fields.RUNNER.value: runner_value,
                                Fields.TP.value: tp,
                                Fields.PP.value: pp,
                                Fields.DCP_SIZE.value: dcp_size,
                                Fields.PCP_SIZE.value: pcp_size,
                                Fields.EP.value: ep if ep is not None else 1,
                                Fields.DP_ATTN.value: dp_attn if dp_attn is not None else False,
                                Fields.SPEC_DECODING.value: spec_decoding,
                                Fields.CONC.value: conc,
                                Fields.KV_OFFLOADING.value: kv_offloading,
                                Fields.TOTAL_CPU_DRAM_GB.value: total_cpu_dram_gb,
                                Fields.DURATION.value: duration,
                                Fields.EXP_NAME.value: (
                                    f"{model_code}_tp{tp}_conc{conc}_"
                                    f"{agentic_kv_offload_suffix(kv_offloading, kv_offload_backend)}"
                                    + (f"_spec-{spec_decoding}" if spec_decoding != "none" else "")
                                ),
                                Fields.SCENARIO_TYPE.value: "agentic-coding",
                            }
                            if kv_offload_backend is not None:
                                entry[Fields.KV_OFFLOAD_BACKEND.value] = kv_offload_backend
                            entry.update(component_metadata(bmk, val))
                            validate_agentic_matrix_entry(entry)
                            matrix_values.append(entry)

    return matrix_values


def _runner_values_for_filter(runner: str, runner_data: dict, runner_node_filter: str | None) -> list[str]:
    if not runner_node_filter:
        return [runner]

    candidates = runner_nodes_for_label(runner, runner_data)
    if runner_node_filter in runner:
        candidates = [runner, *candidates]

    matches = []
    seen = set()
    for node in candidates:
        if runner_node_filter in node and node not in seen:
            matches.append(node)
            seen.add(node)
    return matches


def generate_test_config_sweep(args, all_config_data, runner_data=None):
    """Generate full sweep for specific config keys.

    Validates that all specified config keys exist before generating.
    Expands all configs fully without any filtering.
    """
    resolved_keys = expand_config_keys(args.config_keys, all_config_data.keys())

    matrix_values = []

    runner_data = runner_data or {}

    for key in resolved_keys:
        val = all_config_data[key]
        is_multinode = val.get(Fields.MULTINODE.value, False)

        image = val[Fields.IMAGE.value]
        model = val[Fields.MODEL.value]
        model_code = val[Fields.MODEL_PREFIX.value]
        precision = val[Fields.PRECISION.value]
        framework = val[Fields.FRAMEWORK.value]
        runner = val[Fields.RUNNER.value]
        runners_for_entry = _runner_values_for_filter(
            runner, runner_data, getattr(args, 'runner_node_filter', None))
        if not runners_for_entry:
            continue
        disagg = val.get(Fields.DISAGG.value, False)

        # Build seq-len filter if --seq-lens was provided
        seq_lens_filter = None
        if getattr(args, 'seq_lens', None):
            seq_lens_filter = {seq_len_stoi[s] for s in args.seq_lens}

        scenario_filter = set(args.scenario_type) if getattr(args, 'scenario_type', None) else None
        fixed_configs = val[Fields.SCENARIOS.value].get(Fields.FIXED_SEQ_LEN.value, []) if (scenario_filter is None or 'fixed-seq-len' in scenario_filter) else []
        for seq_len_config in fixed_configs:
            isl = seq_len_config[Fields.ISL.value]
            osl = seq_len_config[Fields.OSL.value]

            if seq_lens_filter and (isl, osl) not in seq_lens_filter:
                continue

            seq_len_str = seq_len_to_str(isl, osl)

            for bmk in seq_len_config[Fields.SEARCH_SPACE.value]:
                if is_multinode:
                    # Multinode config
                    spec_decoding = bmk.get(Fields.SPEC_DECODING.value, "none")
                    prefill = with_worker_parallelism_defaults(
                        bmk[Fields.PREFILL.value])
                    decode = with_worker_parallelism_defaults(
                        bmk[Fields.DECODE.value])

                    # Get concurrency values
                    if Fields.CONC_LIST.value in bmk:
                        conc_values = bmk[Fields.CONC_LIST.value]
                    else:
                        conc_start = bmk[Fields.CONC_START.value]
                        conc_end = bmk[Fields.CONC_END.value]
                        conc_values = []
                        conc = conc_start
                        while conc <= conc_end:
                            conc_values.append(conc)
                            if conc == conc_end:
                                break
                            conc *= 2
                            if conc > conc_end:
                                conc = conc_end

                    # Apply --conc filter if provided (only for test-config)
                    if getattr(args, 'conc', None):
                        conc_values = [c for c in conc_values if c in args.conc]
                        if not conc_values:
                            # No intersection with requested conc values; skip
                            continue

                    for runner_value in runners_for_entry:
                        entry = {
                            Fields.IMAGE.value: image,
                            Fields.MODEL.value: model,
                            Fields.MODEL_PREFIX.value: model_code,
                            Fields.PRECISION.value: precision,
                            Fields.FRAMEWORK.value: framework,
                            Fields.RUNNER.value: runner_value,
                            Fields.ISL.value: isl,
                            Fields.OSL.value: osl,
                            Fields.SPEC_DECODING.value: spec_decoding,
                            Fields.PREFILL.value: prefill,
                            Fields.DECODE.value: decode,
                            Fields.CONC.value: conc_values,
                            Fields.MAX_MODEL_LEN.value: isl + osl + 256,
                            Fields.EXP_NAME.value: f"{model_code}_{seq_len_str}",
                            Fields.DISAGG.value: disagg,
                            Fields.RUN_EVAL.value: False,
                        }
                        entry.update(component_metadata(bmk, val))
                        matrix_values.append(validate_matrix_entry(entry, is_multinode=True))
                else:
                    # Single-node config
                    tp = bmk[Fields.TP.value]
                    pp = bmk.get(Fields.PP.value, 1)
                    dcp_size = bmk.get(Fields.DCP_SIZE.value, 1)
                    pcp_size = bmk.get(Fields.PCP_SIZE.value, 1)
                    ep = bmk.get(Fields.EP.value)
                    dp_attn = bmk.get(Fields.DP_ATTN.value)
                    spec_decoding = bmk.get(Fields.SPEC_DECODING.value, "none")

                    # Get concurrency values
                    if Fields.CONC_LIST.value in bmk:
                        conc_values = bmk[Fields.CONC_LIST.value]
                    else:
                        conc_start = bmk[Fields.CONC_START.value]
                        conc_end = bmk[Fields.CONC_END.value]
                        conc_values = []
                        conc = conc_start
                        while conc <= conc_end:
                            conc_values.append(conc)
                            if conc == conc_end:
                                break
                            conc *= 2
                            if conc > conc_end:
                                conc = conc_end

                    # Apply --conc filter if provided (only for test-config)
                    if getattr(args, 'conc', None):
                        conc_values = [c for c in conc_values if c in args.conc]
                        if not conc_values:
                            # No intersection with requested conc values; skip
                            continue

                    for conc in conc_values:
                        for runner_value in runners_for_entry:
                            entry = {
                                Fields.IMAGE.value: image,
                                Fields.MODEL.value: model,
                                Fields.MODEL_PREFIX.value: model_code,
                                Fields.PRECISION.value: precision,
                                Fields.FRAMEWORK.value: framework,
                                Fields.RUNNER.value: runner_value,
                                Fields.ISL.value: isl,
                                Fields.OSL.value: osl,
                                Fields.TP.value: tp,
                                Fields.PP.value: pp,
                                Fields.DCP_SIZE.value: dcp_size,
                                Fields.PCP_SIZE.value: pcp_size,
                                Fields.CONC.value: conc,
                                Fields.MAX_MODEL_LEN.value: isl + osl + 256,
                                Fields.EP.value: ep if ep is not None else 1,
                                Fields.DP_ATTN.value: dp_attn if dp_attn is not None else False,
                                Fields.SPEC_DECODING.value: spec_decoding,
                                Fields.EXP_NAME.value: f"{model_code}_{seq_len_str}",
                                Fields.DISAGG.value: disagg,
                                Fields.RUN_EVAL.value: False,
                            }
                            entry.update(component_metadata(bmk, val))
                            matrix_values.append(validate_matrix_entry(entry, is_multinode=False))

        # ---- Agentic-coding scenarios ----
        agentic_configs = val[Fields.SCENARIOS.value].get(Fields.AGENTIC_CODING.value, []) if (scenario_filter is None or 'agentic-coding' in scenario_filter) else []
        for agentic_config in agentic_configs:
            duration = DEFAULT_AGENTIC_DURATION_SECONDS
            bmk_space = agentic_config[Fields.SEARCH_SPACE.value]

            for bmk in bmk_space:
                if is_multinode:
                    prefill = with_worker_parallelism_defaults(
                        bmk[Fields.PREFILL.value])
                    decode = with_worker_parallelism_defaults(
                        bmk[Fields.DECODE.value])
                    spec_decoding = bmk.get(Fields.SPEC_DECODING.value, "none")
                    kv_offloading = bmk.get(Fields.KV_OFFLOADING.value, "none")
                    kv_offload_backend = bmk.get(Fields.KV_OFFLOAD_BACKEND.value)
                else:
                    tp = bmk[Fields.TP.value]
                    pp = bmk.get(Fields.PP.value, 1)
                    dcp_size = bmk.get(Fields.DCP_SIZE.value, 1)
                    pcp_size = bmk.get(Fields.PCP_SIZE.value, 1)
                    ep = bmk.get(Fields.EP.value)
                    dp_attn = bmk.get(Fields.DP_ATTN.value)
                    spec_decoding = bmk.get(Fields.SPEC_DECODING.value, "none")
                    kv_offloading = bmk[Fields.KV_OFFLOADING.value]
                    kv_offload_backend = bmk.get(Fields.KV_OFFLOAD_BACKEND.value)
                total_cpu_dram_gb = agentic_dram_offload_gb(
                    agentic_config, bmk, runner, runner_data)

                conc_list = bmk.get(Fields.CONC_LIST.value)
                if conc_list:
                    conc_values = conc_list
                else:
                    conc_start = bmk[Fields.CONC_START.value]
                    conc_end = bmk[Fields.CONC_END.value]
                    conc_values = []
                    conc = conc_start
                    while conc <= conc_end:
                        conc_values.append(conc)
                        if conc == conc_end:
                            break
                        conc *= 2
                        if conc > conc_end:
                            conc = conc_end

                if getattr(args, 'conc', None):
                    conc_values = [c for c in conc_values if c in args.conc]
                if not conc_values:
                    continue

                if is_multinode:
                    # Preserve historical exp-names for the default (no offload)
                    # case; only append a suffix when KV offloading is active.
                    offload_suffix = (
                        f"_{agentic_kv_offload_suffix(kv_offloading, kv_offload_backend)}"
                        if kv_offloading != "none"
                        else ""
                    )
                    for runner_value in runners_for_entry:
                        for conc_batch in chunk_multinode_agentic_concurrencies(conc_values):
                            entry = {
                                Fields.IMAGE.value: image,
                                Fields.MODEL.value: model,
                                Fields.MODEL_PREFIX.value: model_code,
                                Fields.PRECISION.value: precision,
                                Fields.FRAMEWORK.value: framework,
                                Fields.RUNNER.value: runner_value,
                                Fields.SPEC_DECODING.value: spec_decoding,
                                Fields.PREFILL.value: prefill,
                                Fields.DECODE.value: decode,
                                Fields.CONC.value: conc_batch,
                                Fields.KV_OFFLOADING.value: kv_offloading,
                                Fields.TOTAL_CPU_DRAM_GB.value: total_cpu_dram_gb,
                                Fields.DURATION.value: duration,
                                Fields.EXP_NAME.value: (
                                    f"{model_code}_p{prefill[Fields.NUM_WORKER.value]}x{prefill[Fields.TP.value]}"
                                    f"_d{decode[Fields.NUM_WORKER.value]}x{decode[Fields.TP.value]}"
                                    f"_conc{'x'.join(str(c) for c in conc_batch)}"
                                    f"{offload_suffix}"
                                ),
                                Fields.DISAGG.value: disagg,
                                Fields.SCENARIO_TYPE.value: "agentic-coding",
                            }
                            if kv_offload_backend is not None:
                                entry[Fields.KV_OFFLOAD_BACKEND.value] = kv_offload_backend
                            entry.update(component_metadata(bmk, val))
                            matrix_values.append(validate_agentic_matrix_entry(entry))
                else:
                    for conc in conc_values:
                        for runner_value in runners_for_entry:
                            entry = {
                                Fields.IMAGE.value: image,
                                Fields.MODEL.value: model,
                                Fields.MODEL_PREFIX.value: model_code,
                                Fields.PRECISION.value: precision,
                                Fields.FRAMEWORK.value: framework,
                                Fields.RUNNER.value: runner_value,
                                Fields.TP.value: tp,
                                Fields.PP.value: pp,
                                Fields.DCP_SIZE.value: dcp_size,
                                Fields.PCP_SIZE.value: pcp_size,
                                Fields.EP.value: ep if ep is not None else 1,
                                Fields.DP_ATTN.value: dp_attn if dp_attn is not None else False,
                                Fields.SPEC_DECODING.value: spec_decoding,
                                Fields.CONC.value: conc,
                                Fields.KV_OFFLOADING.value: kv_offloading,
                                Fields.TOTAL_CPU_DRAM_GB.value: total_cpu_dram_gb,
                                Fields.DURATION.value: duration,
                                Fields.EXP_NAME.value: (
                                    f"{model_code}_tp{tp}_conc{conc}_"
                                    f"{agentic_kv_offload_suffix(kv_offloading, kv_offload_backend)}"
                                    + (f"_spec-{spec_decoding}" if spec_decoding != "none" else "")
                                ),
                                Fields.SCENARIO_TYPE.value: "agentic-coding",
                            }
                            if kv_offload_backend is not None:
                                entry[Fields.KV_OFFLOAD_BACKEND.value] = kv_offload_backend
                            entry.update(component_metadata(bmk, val))
                            matrix_values.append(validate_agentic_matrix_entry(entry))

    return matrix_values


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


def apply_node_type_defaults(args):
    """Default both single_node and multi_node to True when neither is specified."""
    if hasattr(args, 'single_node') and hasattr(args, 'multi_node'):
        if not args.single_node and not args.multi_node:
            args.single_node = True
            args.multi_node = True
    return args


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
        
    # Apply the existing eval policy first, then expand it when requested.
    if not args.no_evals:
        matrix_values = mark_eval_entries(matrix_values, include_agentic=args.evals_only or args.all_evals)
        if args.all_evals:
            matrix_values = mark_all_eval_entries(matrix_values)

    if args.evals_only or args.all_evals:
        matrix_values = [e for e in matrix_values if e.get(Fields.RUN_EVAL.value, False)]
        for entry in matrix_values:
            entry[Fields.EVAL_ONLY.value] = True

    print(json.dumps(matrix_values))
    return matrix_values


if __name__ == "__main__":
    main()
