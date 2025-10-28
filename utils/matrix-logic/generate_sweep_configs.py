import json
import yaml
import argparse

seq_len_stoi = {
    "1k1k": (1024, 1024),
    "1k8k": (1024, 8192),
    "8k1k": (8192, 1024)
}


def validate_master_configs_structure(all_config_data):
    """Validate the structure of all master config entries.

    This validates that all required fields are present, have correct types,
    and no extra fields exist. Should be called once after loading config files.
    """
    for key, val in all_config_data.items():
        # Check for required top-level fields and their types
        required_fields = {
            'image': str,
            'model': str,
            'precision': str,
            'framework': str,
            'runner': str,
            'seq-len-configs': list
        }

        for field, expected_type in required_fields.items():
            if field not in val or val[field] is None:
                raise ValueError(
                    f"Missing required field '{field}' for key '{key}'")
            if not isinstance(val[field], expected_type):
                raise ValueError(
                    f"Field '{field}' must be {expected_type.__name__} for key '{key}', got {type(val[field]).__name__}")

        seq_len_configs = val['seq-len-configs']
        if len(seq_len_configs) == 0:
            raise ValueError(
                f"'seq-len-configs' must be a non-empty list for key '{key}'")

        # Validate each seq-len-config
        for i, seq_config in enumerate(seq_len_configs):
            # Check isl
            if 'isl' not in seq_config or seq_config['isl'] is None:
                raise ValueError(
                    f"Missing 'isl' in seq-len-config[{i}] for key '{key}'")
            if not isinstance(seq_config['isl'], int):
                raise ValueError(
                    f"'isl' must be int in seq-len-config[{i}] for key '{key}'")

            # Check osl
            if 'osl' not in seq_config or seq_config['osl'] is None:
                raise ValueError(
                    f"Missing 'osl' in seq-len-config[{i}] for key '{key}'")
            if not isinstance(seq_config['osl'], int):
                raise ValueError(
                    f"'osl' must be int in seq-len-config[{i}] for key '{key}'")

            bmk_space = seq_config.get('bmk-space')
            if not bmk_space or not isinstance(bmk_space, list) or len(bmk_space) == 0:
                raise ValueError(
                    f"Missing or invalid 'bmk-space' in seq-len-config[{i}] for key '{key}'")

            # Validate each benchmark in bmk-space
            for j, bmk in enumerate(bmk_space):
                # Define allowed fields
                allowed_fields = {'tp', 'conc-start',
                                  'conc-end', 'ep', 'dp-attn'}
                required_bmk_fields = {'tp': int,
                                       'conc-start': int, 'conc-end': int}
                optional_bmk_fields = {'ep': int, 'dp-attn': bool}

                # Check for extra fields
                extra_fields = set(bmk.keys()) - allowed_fields
                if extra_fields:
                    raise ValueError(
                        f"Extra fields {extra_fields} in bmk-space[{j}] of seq-len-config[{i}] for key '{key}'")

                # Validate required fields
                for field, expected_type in required_bmk_fields.items():
                    if field not in bmk or bmk[field] is None:
                        raise ValueError(
                            f"Missing '{field}' in bmk-space[{j}] of seq-len-config[{i}] for key '{key}'")
                    if not isinstance(bmk[field], expected_type):
                        raise ValueError(
                            f"'{field}' must be {expected_type.__name__} in bmk-space[{j}] of seq-len-config[{i}] for key '{key}'")

                # Validate optional fields if they exist
                for field, expected_type in optional_bmk_fields.items():
                    if field in bmk and bmk[field] is not None:
                        if not isinstance(bmk[field], expected_type):
                            raise ValueError(
                                f"'{field}' must be {expected_type.__name__} in bmk-space[{j}] of seq-len-config[{i}] for key '{key}'")


def generate_full_sweep(args, all_config_data):
    """Generate full sweep configurations based on model prefix and sequence lengths.

    Assumes all_config_data has been validated by validate_config_structure().
    """
    isl, osl = seq_len_stoi[args.seq_lens]

    matrix_values = []
    for key, val in all_config_data.items():
        # Filter by model prefix
        if not key.startswith(args.model_prefix):
            continue

        seq_len_configs = val['seq-len-configs']
        image = val['image']
        model = val['model']
        precision = val['precision']
        framework = val['framework']
        runner = val['runner']
        # I.e., for 70b-fp4-... the model_code is 70b which is necessary for exp_name
        # so that it can be bubbled down to bash script benchmarks... this is probably a FIXME
        model_code = key.split('-')[0]

        # Check if this config has matching sequence lengths
        matching_seq_config = None
        for slq in seq_len_configs:
            if slq['isl'] == isl and slq['osl'] == osl:
                matching_seq_config = slq
                break

        if not matching_seq_config:
            continue  # Skip this config if no matching sequence length

        bmk_space = matching_seq_config['bmk-space']

        for bmk in bmk_space:
            tp = bmk['tp']
            conc_start = bmk['conc-start']
            conc_end = bmk['conc-end']
            ep = bmk.get('ep')
            dp_attn = bmk.get('dp-attn')

            # Generate entries for each concurrency value in the range
            conc = conc_start
            while conc <= conc_end:
                entry = {
                    'image': image,
                    'model': model,
                    'precision': precision,
                    'framework': framework,
                    'runner': runner,
                    'isl': isl,
                    'osl': osl,
                    'tp': tp,
                    'conc': conc,
                    'model-code': model_code,
                    'max-model-len': isl + osl,
                }

                # Add optional fields if they exist
                if ep is not None:
                    entry['ep'] = ep
                if dp_attn is not None:
                    entry['dp-attn'] = dp_attn

                matrix_values.append(entry)

                if conc == conc_end:
                    break
                conc *= args.step_size
                if conc > conc_end:
                    conc = conc_end

    return matrix_values


def generate_test_config(args, all_config_data):
    """Generate test configurations for a specific key.

    Assumes all_config_data has been validated by validate_config_structure().
    """
    # Extract model code from config key
    model_code = args.key.split('-')[0]

    val = all_config_data[args.key]

    seq_len_configs = val['seq-len-configs']
    image = val['image']
    model = val['model']
    precision = val['precision']
    framework = val['framework']
    # Use default runner or specific runner node if input by user
    runner = val['runner'] if not args.runner_node else args.runner_node

    # Convert seq-lens to set of (isl, osl) tuples for filtering
    seq_lens_filter = None
    if args.seq_lens:
        seq_lens_filter = {seq_len_stoi[sl] for sl in args.seq_lens}

    matrix_values = []

    # Process each sequence length configuration
    for seq_config in seq_len_configs:
        isl = seq_config['isl']
        osl = seq_config['osl']

        # Filter by sequence lengths if specified
        if seq_lens_filter and (isl, osl) not in seq_lens_filter:
            continue

        bmk_space = seq_config['bmk-space']

        for bmk in bmk_space:
            tp = bmk['tp']
            conc_start = bmk['conc-start']
            conc_end = bmk['conc-end']
            ep = bmk.get('ep')
            dp_attn = bmk.get('dp-attn')

            # In test mode, only use the lowest concurrency (conc_start)
            if args.test_mode:
                entry = {
                    'image': image,
                    'model': model,
                    'model-code': model_code,
                    'precision': precision,
                    'framework': framework,
                    'runner': runner,
                    'isl': isl,
                    'osl': osl,
                    'tp': tp,
                    'conc': conc_start,
                    'max-model-len': isl + osl,
                }

                # Add optional fields if they exist
                if ep is not None:
                    entry['ep'] = ep
                if dp_attn is not None:
                    entry['dp-attn'] = dp_attn

                matrix_values.append(entry)
            else:
                # Generate entries for each concurrency value in the range
                conc = conc_start
                while conc <= conc_end:
                    entry = {
                        'image': image,
                        'model': model,
                        'model-code': model_code,
                        'precision': precision,
                        'framework': framework,
                        'runner': runner,
                        'isl': isl,
                        'osl': osl,
                        'tp': tp,
                        'conc': conc,
                        'max-model-len': isl + osl,
                    }

                    # Add optional fields if they exist
                    if ep is not None:
                        entry['ep'] = ep
                    if dp_attn is not None:
                        entry['dp-attn'] = dp_attn

                    matrix_values.append(entry)

                    if conc == conc_end:
                        break
                    conc *= args.step_size
                    if conc > conc_end:
                        conc = conc_end

    return matrix_values


def generate_runner_model_sweep_config(args, all_config_data):
    """Generate runner-model sweep configurations.

    Assumes all_config_data has been validated by validate_config_structure().
    """
    with open(args.runner_config, 'r') as f:
        runner_config = yaml.safe_load(f)

    runner_nodes = runner_config.get(args.runner_type)

    if not runner_nodes:
        raise ValueError(
            f"Runner '{args.runner_type}' does not exist in runner config '{args.runner_config}'. Must choose from existing runner types: '{', '.join(runner_config.keys())}'.")

    matrix_values = []
    for key, val in all_config_data.items():
        # Only consider configs with specified runner
        if val['runner'] != args.runner_type:
            continue
        
        # I.e., for 70b-fp4-... the model_code is 70b which is necessary for exp_name
        # so that it can be bubbled down to bash script benchmarks... this is probably a FIXME
        model_code = key.split('-')[0]

        # Find 1k1k config
        target_config = None
        for config in val['seq-len-configs']:
            if config['isl'] == 1024 and config['osl'] == 1024:
                target_config = config
                break

        highest_tp_bmk = max(target_config['bmk-space'], key=lambda x: x['tp'])
        # Since we are just testing, pick the highest TP for this config and just test
        # on that TP with the lowest concurrency available
        highest_tp = highest_tp_bmk['tp']
        lowest_conc = highest_tp_bmk['conc-start']

        ep = highest_tp_bmk.get('ep')
        dp_attn = highest_tp_bmk.get('dp-attn')

        for node in runner_nodes:
            entry = {
                'image': val['image'],
                'model': val['model'],
                'precision': val['precision'],
                'framework': val['framework'],
                # Add one entry for each node under specified runner type
                'runner': node,
                # Again, just use 1k1k since this is just meant to smoke test all runners
                'isl': 1024,
                'osl': 1024,
                'tp': highest_tp,
                'conc': lowest_conc,
                'model-code': model_code,
                'max-model-len': 2048,
            }

            # Add optional fields if they exist
            if ep is not None:
                entry['ep'] = ep
            if dp_attn is not None:
                entry['dp-attn'] = dp_attn

            matrix_values.append(entry)

    return matrix_values


def load_config_files(config_files):
    """Load and merge configuration files."""
    all_config_data = {}
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
                assert isinstance(
                    config_data, dict), f"Config file '{config_file}' must contain a dictionary"

                # Check for duplicate keys, this is only in place to prevent against the very unlikely
                # case where an entry in one config accidentally/purposefully tries to override an entry in another config
                duplicate_keys = set(all_config_data.keys()) & set(
                    config_data.keys())
                if duplicate_keys:
                    raise ValueError(
                        f"Duplicate configuration keys found in '{config_file}': {', '.join(sorted(duplicate_keys))}"
                    )

                all_config_data.update(config_data)
        except FileNotFoundError:
            raise ValueError(f"Input file '{config_file}' does not exist.")

    return all_config_data


def main():
    # Create parent parser with common arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        '--config-files',
        nargs='+',
        required=True,
        help='One or more configuration files (YAML format)'
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
        help='Generate full sweep configurations based on model prefix'
    )
    full_sweep_parser.add_argument(
        '--seq-lens',
        choices=list(seq_len_stoi.keys()),
        required=True,
        help=f"Sequence length configuration: {', '.join(seq_len_stoi.keys())}"
    )
    full_sweep_parser.add_argument(
        '--model-prefix',
        required=True,
        help='Model prefix to filter configurations'
    )
    full_sweep_parser.add_argument(
        '--step-size',
        type=int,
        default=2,
        help='Step size for concurrency values (default: 2)'
    )
    full_sweep_parser.add_argument(
        '-h', '--help',
        action='help',
        help='Show this help message and exit'
    )

    # Subcommand: test-config
    test_config_parser = subparsers.add_parser(
        'test-config',
        parents=[parent_parser],
        add_help=False,
        help='Generate test configurations for a specific key'
    )
    test_config_parser.add_argument(
        '--key',
        required=True,
        help='Configuration key to use'
    )
    test_config_parser.add_argument(
        '--runner-node',
        required=False,
        help='Specific runner node to use'
    )
    test_config_parser.add_argument(
        '--seq-lens',
        nargs='+',
        choices=list(seq_len_stoi.keys()),
        required=False,
        help=f"Sequence length configurations to include: {', '.join(seq_len_stoi.keys())}. If not specified, all sequence lengths are included."
    )
    test_config_parser.add_argument(
        '--step-size',
        type=int,
        default=2,
        help='Step size for concurrency values (default: 2)'
    )
    test_config_parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Generate only the lowest concurrency value for each TP level'
    )
    test_config_parser.add_argument(
        '-h', '--help',
        action='help',
        help='Show this help message and exit'
    )

    # Subcommand: runner-model-sweep
    test_config_parser = subparsers.add_parser(
        'runner-model-sweep',
        parents=[parent_parser],
        add_help=False,
        help='Sweep across all runner nodes and all compatible models for a given runner'
    )
    test_config_parser.add_argument(
        '--runner-type',
        required=True,
        help='Runner type (e.g., h200-trt, h100)'
    )
    test_config_parser.add_argument(
        '--runner-config',
        required=True,
        help='Configuration file holding runner information'
    )
    test_config_parser.add_argument(
        '-h', '--help',
        action='help',
        help='Show this help message and exit'
    )

    args = parser.parse_args()

    # Load and validate configuration files
    all_config_data = load_config_files(args.config_files)
    validate_master_configs_structure(all_config_data)

    # Route to appropriate function based on subcommand
    if args.command == 'full-sweep':
        matrix_values = generate_full_sweep(args, all_config_data)
    elif args.command == 'test-config':
        matrix_values = generate_test_config(args, all_config_data)
    elif args.command == 'runner-model-sweep':
        matrix_values = generate_runner_model_sweep_config(
            args, all_config_data)
    else:
        parser.error(f"Unknown command: {args.command}")

    print(json.dumps(matrix_values))
    return matrix_values


if __name__ == "__main__":
    main()
