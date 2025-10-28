import json
import yaml
import argparse

seq_len_stoi = {
    "1k1k": (1024, 1024),
    "1k8k": (1024, 8192),
    "8k1k": (8192, 1024)
}

def generate_full_sweep(args, all_config_data):
    """Generate full sweep configurations based on model prefix and sequence lengths."""
    isl, osl = seq_len_stoi[args.seq_lens]

    matrix_values = []
    for key, val in all_config_data.items():
        # Filter by model prefix
        if not key.startswith(args.model_prefix):
            continue

        seq_len_configs = val.get('seq-len-configs')
        assert seq_len_configs, f"Missing 'seq-len-configs' for key '{key}'"

        image = val.get('image')
        model = val.get('model')
        precision = val.get('precision')
        framework = val.get('framework')
        runner = val.get('runner')

        assert None not in (image, model, precision, framework, runner), \
            f"Missing required fields for key '{key}'"

        # Check if this config has matching sequence lengths
        matching_seq_config = None
        for slq in seq_len_configs:
            if slq.get('isl') == isl and slq.get('osl') == osl:
                matching_seq_config = slq
                break

        if not matching_seq_config:
            continue  # Skip this config if no matching sequence length

        bmk_space = matching_seq_config.get('bmk-space')
        assert bmk_space, f"Missing 'bmk-space' in matching seq-len-config for key '{key}'"

        for bmk in bmk_space:
            tp = bmk.get('tp')
            conc_start = bmk.get('conc-start')
            conc_end = bmk.get('conc-end')
            ep = bmk.get('ep')
            dp_attn = bmk.get('dp-attn')

            assert None not in (tp, conc_start, conc_end), \
                f"Missing 'tp', 'conc-start', or 'conc-end' in bmk-space for key '{key}'"

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
                    'conc': conc
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
    """Generate test configurations for a specific key."""
    # Check if the key exists
    if args.key not in all_config_data:
        available_keys = ', '.join(sorted(all_config_data.keys()))
        raise ValueError(
            f"Key '{args.key}' not found in configuration files. "
            f"Available keys: {available_keys}"
        )

    # Extract model code from config key
    model_code = args.key.split('-')[0]
    # Extract GPU from config key
    config_gpu = args.key.split('-')[2]
    runner_gpu = args.runner_node.split('-')[0] if args.runner_node else None
    
    # If user enters a runner not compatible with input GPU sku, error
    if runner_gpu and config_gpu != runner_gpu:
        raise ValueError(f"GPU '{config_gpu}' used in selected config '{args.key}' cannot run on selected runner node '{args.runner_node}'.")
    
    val = all_config_data[args.key]

    # Validate required fields
    seq_len_configs = val.get('seq-len-configs')
    assert seq_len_configs, f"Missing 'seq-len-configs' for key '{args.key}'"

    image = val.get('image')
    model = val.get('model')
    precision = val.get('precision')
    framework = val.get('framework')
    # Use default runner or specific runner node if input by user
    runner = val.get('runner') if not args.runner_node else args.runner_node

    assert None not in (image, model, precision, framework, runner), \
        f"Missing required fields (image, model, precision, framework, runner) for key '{args.key}'"

    # Convert seq-lens to set of (isl, osl) tuples for filtering
    seq_lens_filter = None
    if args.seq_lens:
        seq_lens_filter = {seq_len_stoi[sl] for sl in args.seq_lens}

    matrix_values = []

    # Process each sequence length configuration
    for seq_config in seq_len_configs:
        isl = seq_config.get('isl')
        osl = seq_config.get('osl')

        assert None not in (isl, osl), \
            f"Missing 'isl' or 'osl' in seq-len-config for key '{args.key}'"

        # Filter by sequence lengths if specified
        if seq_lens_filter and (isl, osl) not in seq_lens_filter:
            continue

        bmk_space = seq_config.get('bmk-space')
        assert bmk_space, f"Missing 'bmk-space' in seq-len-config for key '{args.key}'"

        for bmk in bmk_space:
            tp = bmk.get('tp')
            conc_start = bmk.get('conc-start')
            conc_end = bmk.get('conc-end')
            ep = bmk.get('ep')
            dp_attn = bmk.get('dp-attn')

            assert None not in (tp, conc_start, conc_end), \
                f"Missing 'tp', 'conc-start', or 'conc-end' in bmk-space for key '{args.key}'"

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

def load_config_files(config_files):
    """Load and merge configuration files."""
    all_config_data = {}
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
                assert isinstance(config_data, dict), f"Config file '{config_file}' must contain a dictionary"

                # Check for duplicate keys
                duplicate_keys = set(all_config_data.keys()) & set(config_data.keys())
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

    args = parser.parse_args()

    # Load configuration files
    all_config_data = load_config_files(args.config_files)

    # Route to appropriate function based on subcommand
    if args.command == 'full-sweep':
        matrix_values = generate_full_sweep(args, all_config_data)
    elif args.command == 'test-config':
        matrix_values = generate_test_config(args, all_config_data)
    else:
        parser.error(f"Unknown command: {args.command}")

    print(json.dumps(matrix_values))
    return matrix_values

if __name__ == "__main__":
    main()
