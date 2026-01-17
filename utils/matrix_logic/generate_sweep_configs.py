import json
import argparse
import sys
from pathlib import Path

# Ensure sibling modules are importable regardless of how script is invoked
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validation import (
    validate_matrix_entry,
    load_config_files,
    load_runner_file,
    Fields
)

seq_len_stoi = {
    "1k1k": (1024, 1024),
    "1k8k": (1024, 8192),
    "8k1k": (8192, 1024)
}

# Reverse mapping for exp-name generation
seq_len_itos = {v: k for k, v in seq_len_stoi.items()}


def seq_len_to_str(isl: int, osl: int) -> str:
    """Convert sequence lengths to short string representation.

    Returns the short name (e.g., '1k1k') if it exists in the mapping,
    otherwise returns 'isl_osl' format.
    """
    return seq_len_itos.get((isl, osl), f"{isl}_{osl}")


def generate_full_sweep(args, all_config_data, runner_data):
    """Generate full sweep configurations with optional filtering.

    Supports filtering by model prefix, precision, framework, runner type, sequence lengths,
    and max concurrency.

    All filters are optional - can generate sweeps for all configs or filter by specific criteria.

    Assumes all_config_data has been validated by validate_master_config().
    """
    # Validate runner types if specified
    if args.runner_type:
        valid_runner_types = set(runner_data.keys())
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

        seq_len_configs = val[Fields.SEQ_LEN_CONFIGS.value]
        image = val[Fields.IMAGE.value]
        model = val[Fields.MODEL.value]
        precision = val[Fields.PRECISION.value]
        framework = val[Fields.FRAMEWORK.value]
        runner = val[Fields.RUNNER.value]
        model_code = val[Fields.MODEL_PREFIX.value]

        for seq_config in seq_len_configs:
            isl = seq_config[Fields.ISL.value]
            osl = seq_config[Fields.OSL.value]

            # Filter by sequence lengths if specified
            if seq_lens_filter and (isl, osl) not in seq_lens_filter:
                continue

            bmk_space = seq_config[Fields.SEARCH_SPACE.value]

            for bmk in bmk_space:
                if is_multinode:
                    # Skip multinode configs when --single-node is specified
                    if not args.multi_node:
                        continue

                    # Multinode configuration
                    # spec_decoding defaults to "none" if not specified
                    spec_decoding = bmk.get(Fields.SPEC_DECODING.value, "none")

                    prefill = bmk[Fields.PREFILL.value]
                    decode = bmk[Fields.DECODE.value]

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

                    # For multinode, create a single entry with conc as a list
                    seq_len_str = seq_len_to_str(isl, osl)
                    entry = {
                        Fields.IMAGE.value: image,
                        Fields.MODEL.value: model,
                        Fields.MODEL_PREFIX.value: model_code,
                        Fields.PRECISION.value: precision,
                        Fields.FRAMEWORK.value: framework,
                        Fields.RUNNER.value: runner,
                        Fields.ISL.value: isl,
                        Fields.OSL.value: osl,
                        Fields.SPEC_DECODING.value: spec_decoding,
                        Fields.PREFILL.value: prefill,
                        Fields.DECODE.value: decode,
                        Fields.CONC.value: conc_values,  # Pass the entire list for multinode
                        Fields.MAX_MODEL_LEN.value: isl + osl + 200,
                        Fields.EXP_NAME.value: f"{model_code}_{seq_len_str}",
                        Fields.DISAGG.value: disagg,
                    }

                    validate_matrix_entry(entry, is_multinode)
                    matrix_values.append(entry)
                elif args.single_node:
                    # Single-node configuration
                    tp = bmk[Fields.TP.value]
                    conc_start = bmk[Fields.CONC_START.value]
                    conc_end = bmk[Fields.CONC_END.value]
                    ep = bmk.get(Fields.EP.value)
                    dp_attn = bmk.get(Fields.DP_ATTN.value)
                    spec_decoding = bmk.get(Fields.SPEC_DECODING.value, "none")

                    # Apply max-tp filter if specified
                    # If tp > max_tp, use max_tp instead of skipping (if valid)
                    if args.max_tp is not None:
                        if args.max_tp <= 0:
                            continue  # Skip if max_tp is not positive
                        if tp > args.max_tp:
                            tp = args.max_tp

                    # Apply max-ep filter if specified
                    # If ep > max_ep, use max_ep instead of skipping (if valid)
                    if args.max_ep is not None:
                        if args.max_ep <= 0:
                            continue  # Skip if max_ep is not positive
                        if ep is not None and ep > args.max_ep:
                            ep = args.max_ep

                    # Apply max-conc filter if specified
                    # If conc_start > max_conc, use max_conc as both start and end (if valid)
                    if args.max_conc is not None:
                        if args.max_conc <= 0:
                            continue  # Skip if max_conc is not positive
                        if conc_start > args.max_conc:
                            conc_start = args.max_conc
                            conc_end = args.max_conc
                        else:
                            conc_end = min(conc_end, args.max_conc)

                    conc = conc_start
                    while conc <= conc_end:
                        seq_len_str = seq_len_to_str(isl, osl)
                        entry = {
                            Fields.IMAGE.value: image,
                            Fields.MODEL.value: model,
                            Fields.MODEL_PREFIX.value: model_code,
                            Fields.PRECISION.value: precision,
                            Fields.FRAMEWORK.value: framework,
                            Fields.RUNNER.value: runner,
                            Fields.ISL.value: isl,
                            Fields.OSL.value: osl,
                            Fields.TP.value: tp,
                            Fields.CONC.value: conc,
                            Fields.MAX_MODEL_LEN.value: isl + osl + 200,
                            Fields.EP.value: 1,  # Default
                            Fields.DP_ATTN.value: False,  # Default
                            Fields.SPEC_DECODING.value: spec_decoding,
                            Fields.EXP_NAME.value: f"{model_code}_{seq_len_str}",
                            Fields.DISAGG.value: disagg,
                        }

                        if ep is not None:
                            entry[Fields.EP.value] = ep
                        if dp_attn is not None:
                            entry[Fields.DP_ATTN.value] = dp_attn

                        validate_matrix_entry(entry, is_multinode)
                        matrix_values.append(entry)

                        if conc == conc_end:
                            break
                        conc *= args.step_size
                        if conc > conc_end:
                            conc = conc_end

    return matrix_values


def generate_runner_model_sweep_config(args, all_config_data, runner_data):
    """Generate runner-model sweep configurations.

    Assumes all_config_data has been validated by validate_config_structure().
    Supports both single-node and multinode configurations.
    """
    runner_nodes = runner_data.get(args.runner_type)

    if not runner_nodes:
        raise ValueError(
            f"Runner '{args.runner_type}' does not exist in runner config '{args.runner_config}'. Must choose from existing runner types: '{', '.join(runner_data.keys())}'.")

    # Filter runner nodes if filter is specified
    if args.runner_node_filter:
        runner_nodes = [
            node for node in runner_nodes if args.runner_node_filter in node]
        if not runner_nodes:
            raise ValueError(
                f"No runner nodes found matching filter '{args.runner_node_filter}' for runner type '{args.runner_type}'.")

    matrix_values = []
    for key, val in all_config_data.items():
        # Only consider configs with specified runner
        if val[Fields.RUNNER.value] != args.runner_type:
            continue

        is_multinode = val.get(Fields.MULTINODE.value, False)

        # Skip configs that don't match the requested node type
        if args.single_node and is_multinode:
            continue
        if args.multi_node and not is_multinode:
            continue

        # Get model code for exp_name
        model_code = val[Fields.MODEL_PREFIX.value]
        # Get disagg value, defaulting to False if not specified
        disagg = val.get(Fields.DISAGG.value, False)

        # Find 1k1k config
        target_config = None
        for config in val[Fields.SEQ_LEN_CONFIGS.value]:
            if config[Fields.ISL.value] == 1024 and config[Fields.OSL.value] == 1024:
                target_config = config
                break

        if target_config is None:
            continue

        if is_multinode:
            # For multinode, find the search space entry with the lowest concurrency
            def get_lowest_conc(search_space_entry):
                conc_list = search_space_entry.get(Fields.CONC_LIST.value, [])
                return min(conc_list) if conc_list else float('inf')

            lowest_conc_entry = min(
                target_config[Fields.SEARCH_SPACE.value], key=get_lowest_conc)

            conc_list = lowest_conc_entry.get(Fields.CONC_LIST.value, [])
            lowest_conc = min(conc_list) if conc_list else 1

            spec_decoding = lowest_conc_entry.get(
                Fields.SPEC_DECODING.value, "none")
            prefill_config = lowest_conc_entry[Fields.PREFILL.value]
            decode_config = lowest_conc_entry[Fields.DECODE.value]

            for node in runner_nodes:
                entry = {
                    Fields.IMAGE.value: val[Fields.IMAGE.value],
                    Fields.MODEL.value: val[Fields.MODEL.value],
                    Fields.MODEL_PREFIX.value: model_code,
                    Fields.PRECISION.value: val[Fields.PRECISION.value],
                    Fields.FRAMEWORK.value: val[Fields.FRAMEWORK.value],
                    Fields.RUNNER.value: node,
                    Fields.ISL.value: 1024,
                    Fields.OSL.value: 1024,
                    Fields.SPEC_DECODING.value: spec_decoding,
                    Fields.PREFILL.value: {
                        Fields.NUM_WORKER.value: prefill_config[Fields.NUM_WORKER.value],
                        Fields.TP.value: prefill_config[Fields.TP.value],
                        Fields.EP.value: prefill_config[Fields.EP.value],
                        Fields.DP_ATTN.value: prefill_config[Fields.DP_ATTN.value],
                        Fields.ADDITIONAL_SETTINGS.value: prefill_config.get(Fields.ADDITIONAL_SETTINGS.value, []),
                    },
                    Fields.DECODE.value: {
                        Fields.NUM_WORKER.value: decode_config[Fields.NUM_WORKER.value],
                        Fields.TP.value: decode_config[Fields.TP.value],
                        Fields.EP.value: decode_config[Fields.EP.value],
                        Fields.DP_ATTN.value: decode_config[Fields.DP_ATTN.value],
                        Fields.ADDITIONAL_SETTINGS.value: decode_config.get(Fields.ADDITIONAL_SETTINGS.value, []),
                    },
                    Fields.CONC.value: [lowest_conc],
                    Fields.MAX_MODEL_LEN.value: 2048,
                    Fields.EXP_NAME.value: f"{model_code}_test",
                    Fields.DISAGG.value: disagg,
                }
                matrix_values.append(validate_matrix_entry(entry, is_multinode=True))
        else:
            # Single-node: pick highest TP config with lowest concurrency
            highest_tp_bmk = max(
                target_config[Fields.SEARCH_SPACE.value], key=lambda x: x[Fields.TP.value])
            highest_tp = highest_tp_bmk[Fields.TP.value]
            lowest_conc = highest_tp_bmk[Fields.CONC_START.value]

            ep = highest_tp_bmk.get(Fields.EP.value)
            dp_attn = highest_tp_bmk.get(Fields.DP_ATTN.value)
            spec_decoding = highest_tp_bmk.get(Fields.SPEC_DECODING.value, "none")

            for node in runner_nodes:
                entry = {
                    Fields.IMAGE.value: val[Fields.IMAGE.value],
                    Fields.MODEL.value: val[Fields.MODEL.value],
                    Fields.MODEL_PREFIX.value: model_code,
                    Fields.PRECISION.value: val[Fields.PRECISION.value],
                    Fields.FRAMEWORK.value: val[Fields.FRAMEWORK.value],
                    Fields.RUNNER.value: node,
                    Fields.ISL.value: 1024,
                    Fields.OSL.value: 1024,
                    Fields.TP.value: highest_tp,
                    Fields.EP.value: ep if ep is not None else 1,
                    Fields.DP_ATTN.value: dp_attn if dp_attn is not None else False,
                    Fields.SPEC_DECODING.value: spec_decoding,
                    Fields.CONC.value: lowest_conc,
                    Fields.MAX_MODEL_LEN.value: 2048,
                    Fields.EXP_NAME.value: f"{model_code}_test",
                    Fields.DISAGG.value: disagg,
                }
                matrix_values.append(validate_matrix_entry(entry, is_multinode=False))

    return matrix_values


def generate_test_config_sweep(args, all_config_data):
    """Generate full sweep for specific config keys.

    Validates that all specified config keys exist before generating.
    Expands all configs fully without any filtering.
    """
    # Validate all config keys exist
    missing_keys = [key for key in args.config_keys if key not in all_config_data]
    if missing_keys:
        available_keys = sorted(all_config_data.keys())
        raise ValueError(
            f"Config key(s) not found: {', '.join(missing_keys)}.\n"
            f"Available keys: {', '.join(available_keys)}"
        )

    matrix_values = []

    for key in args.config_keys:
        val = all_config_data[key]
        is_multinode = val.get(Fields.MULTINODE.value, False)

        image = val[Fields.IMAGE.value]
        model = val[Fields.MODEL.value]
        model_code = val[Fields.MODEL_PREFIX.value]
        precision = val[Fields.PRECISION.value]
        framework = val[Fields.FRAMEWORK.value]
        runner = val[Fields.RUNNER.value]
        disagg = val.get(Fields.DISAGG.value, False)

        for seq_len_config in val[Fields.SEQ_LEN_CONFIGS.value]:
            isl = seq_len_config[Fields.ISL.value]
            osl = seq_len_config[Fields.OSL.value]
            seq_len_str = seq_len_to_str(isl, osl)

            for bmk in seq_len_config[Fields.SEARCH_SPACE.value]:
                if is_multinode:
                    # Multinode config
                    spec_decoding = bmk.get(Fields.SPEC_DECODING.value, "none")
                    prefill = bmk[Fields.PREFILL.value]
                    decode = bmk[Fields.DECODE.value]

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

                    entry = {
                        Fields.IMAGE.value: image,
                        Fields.MODEL.value: model,
                        Fields.MODEL_PREFIX.value: model_code,
                        Fields.PRECISION.value: precision,
                        Fields.FRAMEWORK.value: framework,
                        Fields.RUNNER.value: runner,
                        Fields.ISL.value: isl,
                        Fields.OSL.value: osl,
                        Fields.SPEC_DECODING.value: spec_decoding,
                        Fields.PREFILL.value: prefill,
                        Fields.DECODE.value: decode,
                        Fields.CONC.value: conc_values,
                        Fields.MAX_MODEL_LEN.value: isl + osl + 200,
                        Fields.EXP_NAME.value: f"{model_code}_{seq_len_str}",
                        Fields.DISAGG.value: disagg,
                    }
                    matrix_values.append(validate_matrix_entry(entry, is_multinode=True))
                else:
                    # Single-node config
                    tp = bmk[Fields.TP.value]
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

                    for conc in conc_values:
                        entry = {
                            Fields.IMAGE.value: image,
                            Fields.MODEL.value: model,
                            Fields.MODEL_PREFIX.value: model_code,
                            Fields.PRECISION.value: precision,
                            Fields.FRAMEWORK.value: framework,
                            Fields.RUNNER.value: runner,
                            Fields.ISL.value: isl,
                            Fields.OSL.value: osl,
                            Fields.TP.value: tp,
                            Fields.CONC.value: conc,
                            Fields.MAX_MODEL_LEN.value: isl + osl + 200,
                            Fields.EP.value: ep if ep is not None else 1,
                            Fields.DP_ATTN.value: dp_attn if dp_attn is not None else False,
                            Fields.SPEC_DECODING.value: spec_decoding,
                            Fields.EXP_NAME.value: f"{model_code}_{seq_len_str}",
                            Fields.DISAGG.value: disagg,
                        }
                        matrix_values.append(validate_matrix_entry(entry, is_multinode=False))

    return matrix_values


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
        default='.github/configs/runners.yaml',
        help='Configuration file holding runner information (YAML format, defaults to .github/configs/runners.yaml)'
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
    node_type_group = full_sweep_parser.add_mutually_exclusive_group(required=True)
    node_type_group.add_argument(
        '--single-node',
        action='store_true',
        help='Only generate single-node configurations'
    )
    node_type_group.add_argument(
        '--multi-node',
        action='store_true',
        help='Only generate multi-node configurations'
    )
    full_sweep_parser.add_argument(
        '-h', '--help',
        action='help',
        help='Show this help message and exit'
    )

    # Subcommand: runner-model-sweep
    test_config_parser = subparsers.add_parser(
        'runner-model-sweep',
        parents=[parent_parser],
        add_help=False,
        help='Given a runner type, find all configurations matching the type, and run that configuration on all individual runner nodes for the specified runner type. This is meant to validate that all runner nodes work on all configurations for a runner type. For instance, to validate that all configs that specify an h200 runner successfully run across all h200 runner nodes.'
    )
    test_config_parser.add_argument(
        '--runner-type',
        required=True,
        help='Runner type (e.g., b200-trt, h100)'
    )
    test_config_parser.add_argument(
        '--runner-node-filter',
        required=False,
        help='Filter runner nodes by substring match (e.g., "mi300x-amd" to only include nodes containing that string)'
    )
    test_node_group = test_config_parser.add_mutually_exclusive_group(
        required=True)
    test_node_group.add_argument(
        '--single-node',
        action='store_true',
        help='Generate single-node configurations only'
    )
    test_node_group.add_argument(
        '--multi-node',
        action='store_true',
        help='Generate multi-node configurations only'
    )
    test_config_parser.add_argument(
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
        '-h', '--help',
        action='help',
        help='Show this help message and exit'
    )

    args = parser.parse_args()

    # Load and validate configuration files (validation happens by default in load functions)
    all_config_data = load_config_files(args.config_files)
    runner_data = load_runner_file(args.runner_config)

    # Route to appropriate function based on subcommand
    if args.command == 'full-sweep':
        matrix_values = generate_full_sweep(args, all_config_data, runner_data)
    elif args.command == 'runner-model-sweep':
        matrix_values = generate_runner_model_sweep_config(
            args, all_config_data, runner_data)
    elif args.command == 'test-config':
        matrix_values = generate_test_config_sweep(args, all_config_data)
    else:
        parser.error(f"Unknown command: {args.command}")

    print(json.dumps(matrix_values))
    return matrix_values


if __name__ == "__main__":
    main()
