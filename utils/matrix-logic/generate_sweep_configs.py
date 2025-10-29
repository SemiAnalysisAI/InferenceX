import json
import yaml
import argparse
from pydantic import BaseModel, Field, ValidationError, ConfigDict
from typing import List

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


class MatrixEntry(BaseModel):
    """Pydantic model for validating matrix entry structure."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    image: str
    model: str
    precision: str
    framework: str
    runner: str
    isl: int
    osl: int
    tp: int
    ep: int
    dp_attn: bool = Field(alias='dp-attn')
    conc: int
    max_model_len: int = Field(alias='max-model-len')
    exp_name: str = Field(alias='exp-name')


def validate_matrix_output(matrix_values: List[dict]) -> List[dict]:
    """Validate that matrix_values entries match the expected structure.

    Raises ValueError if any entry fails validation.
    Returns the original list if all entries are valid.
    """
    for i, entry in enumerate(matrix_values):
        try:
            MatrixEntry(**entry)
        except ValidationError as e:
            raise ValueError(f"Matrix entry at index {i} failed validation:\n{e}")
    return matrix_values


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

            bmk_space = seq_config.get('search-space')
            if not bmk_space or not isinstance(bmk_space, list) or len(bmk_space) == 0:
                raise ValueError(
                    f"Missing or invalid 'search-space' in seq-len-config[{i}] for key '{key}'")

            # Validate each benchmark in search-space
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
                        f"Extra fields {extra_fields} in search-space[{j}] of seq-len-config[{i}] for key '{key}'")

                # Validate required fields
                for field, expected_type in required_bmk_fields.items():
                    if field not in bmk or bmk[field] is None:
                        raise ValueError(
                            f"Missing '{field}' in search-space[{j}] of seq-len-config[{i}] for key '{key}'")
                    if not isinstance(bmk[field], expected_type):
                        raise ValueError(
                            f"'{field}' must be {expected_type.__name__} in search-space[{j}] of seq-len-config[{i}] for key '{key}'")

                # Validate optional fields if they exist
                for field, expected_type in optional_bmk_fields.items():
                    if field in bmk and bmk[field] is not None:
                        if not isinstance(bmk[field], expected_type):
                            raise ValueError(
                                f"'{field}' must be {expected_type.__name__} in search-space[{j}] of seq-len-config[{i}] for key '{key}'")


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

        bmk_space = matching_seq_config['search-space']

        for bmk in bmk_space:
            tp = bmk['tp']
            conc_start = bmk['conc-start']
            conc_end = bmk['conc-end']
            ep = bmk.get('ep')
            dp_attn = bmk.get('dp-attn')

            # Generate entries for each concurrency value in the range
            conc = conc_start
            while conc <= conc_end:
                seq_len_str = seq_len_to_str(isl, osl)
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
                    'max-model-len': isl + osl + 200,
                    'ep': 1,  # Default
                    'dp-attn': False,  # Default
                    'exp-name': f"{model_code}_{seq_len_str}",
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


def generate_filtered_sweep(args, all_config_data):
    """Generate sweep configurations with filtering options.

    Allows filtering by model prefix, precision, framework, runner type, and sequence lengths.
    Supports test mode to only run highest TP with lowest concurrency.

    Assumes all_config_data has been validated by validate_config_structure().
    """
    # Validate runner types if specified
    if args.runner_type:
        if not args.runner_config:
            raise ValueError(
                "--runner-config is required when --runner-type is specified")

        try:
            with open(args.runner_config, 'r') as f:
                runner_config = yaml.safe_load(f)
        except FileNotFoundError:
            raise ValueError(
                f"Runner config file '{args.runner_config}' does not exist.")

        valid_runner_types = set(runner_config.keys())
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

    for key, val in all_config_data.items():
        # Filter by model prefix if specified
        if args.model_prefix:
            if not any(key.startswith(prefix) for prefix in args.model_prefix):
                continue

        # Filter by precision if specified
        if args.precision and val['precision'] not in args.precision:
            continue

        # Filter by framework if specified
        if args.framework and val['framework'] not in args.framework:
            continue

        # Filter by runner type if specified
        if args.runner_type and val['runner'] not in args.runner_type:
            continue

        seq_len_configs = val['seq-len-configs']
        image = val['image']
        model = val['model']
        precision = val['precision']
        framework = val['framework']
        runner = val['runner']
        model_code = key.split('-')[0]

        for seq_config in seq_len_configs:
            isl = seq_config['isl']
            osl = seq_config['osl']

            # Filter by sequence lengths if specified
            if seq_lens_filter and (isl, osl) not in seq_lens_filter:
                continue

            bmk_space = seq_config['search-space']

            if args.test_mode:
                # In test mode, use highest TP with lowest concurrency
                highest_tp_bmk = max(bmk_space, key=lambda x: x['tp'])
                tp = highest_tp_bmk['tp']
                conc = highest_tp_bmk['conc-start']
                ep = highest_tp_bmk.get('ep')
                dp_attn = highest_tp_bmk.get('dp-attn')

                seq_len_str = seq_len_to_str(isl, osl)
                entry = {
                    'image': image,
                    'model': model,
                    'precision': precision,
                    'framework': framework,
                    'runner': runner,
                    'isl': isl,
                    'osl': osl,
                    'tp': tp,
                    'ep': 1,  # Default
                    'dp-attn': False,  # Default
                    'conc': conc,
                    'max-model-len': isl + osl + 200,
                    'exp-name': f"{model_code}_{seq_len_str}",
                }

                if ep is not None:
                    entry['ep'] = ep
                if dp_attn is not None:
                    entry['dp-attn'] = dp_attn

                matrix_values.append(entry)
            else:
                # Full sweep mode
                for bmk in bmk_space:
                    tp = bmk['tp']
                    conc_start = bmk['conc-start']
                    conc_end = bmk['conc-end']
                    ep = bmk.get('ep')
                    dp_attn = bmk.get('dp-attn')

                    conc = conc_start
                    while conc <= conc_end:
                        seq_len_str = seq_len_to_str(isl, osl)
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
                            'max-model-len': isl + osl + 200,
                            'ep': 1,  # Default
                            'dp-attn': False,  # Default
                            'exp-name': f"{model_code}_{seq_len_str}",
                        }

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

    if len(matrix_values) == 0:
        error_msg = "No configs found matching filters:"
        if args.model_prefix:
            error_msg += f" model-prefix={args.model_prefix}"
        if args.precision:
            error_msg += f" precision={args.precision}"
        if args.framework:
            error_msg += f" framework={args.framework}"
        if args.runner_type:
            error_msg += f" runner-type={args.runner_type}"
        if seq_lens_filter:
            error_msg += f" seq-lens={args.seq_lens}"
        raise ValueError(error_msg)

    return matrix_values


def generate_test_config(args, all_config_data):
    """Generate test configurations for a specific key.

    Assumes all_config_data has been validated by validate_config_structure().
    """
    try:
        with open(args.runner_config, 'r') as f:
            runner_config = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise ValueError(
            f"Runner config file '{args.runner_config}' does not exist.")

    # Extract model code from config key
    model_code = args.key.split('-')[0]

    val = all_config_data.get(args.key)

    if not val:
        raise ValueError(
            f"Specified key '{args.key}' does not exist in config files.")

    runner_nodes = runner_config.get(val['runner'], [])
    if args.runner_node not in runner_nodes:
        raise ValueError(
            f"Runner node '{args.runner_node}' is not compatible with config '{args.key}' which runs on runner type '{val['runner']}'. Available runner nodes for this config are '{', '.join(runner_nodes)}'.")

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

        bmk_space = seq_config['search-space']

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
                    'precision': precision,
                    'framework': framework,
                    'runner': runner,
                    'isl': isl,
                    'osl': osl,
                    'tp': tp,
                    'ep': 1, # Default,
                    'dp-attn': False, # Default
                    'conc': conc_start,
                    'max-model-len': isl + osl,
                    'exp-name': f"{model_code}_test",
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
                        'precision': precision,
                        'framework': framework,
                        'runner': runner,
                        'isl': isl,
                        'osl': osl,
                        'tp': tp,
                        'ep': 1, # Default,
                        'dp-attn': False, # Default
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
    try:
        with open(args.runner_config, 'r') as f:
            runner_config = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise ValueError(
            f"Runner config file '{args.runner_config}' does not exist.")

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

        highest_tp_bmk = max(target_config['search-space'], key=lambda x: x['tp'])
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
                'ep': 1, # Default,
                'dp-attn': False, # Default
                'conc': lowest_conc,
                'max-model-len': 2048,
                'exp-name': f"{model_code}_test",
            }

            # Add optional fields if they exist
            if ep is not None:
                entry['ep'] = ep
            if dp_attn is not None:
                entry['dp-attn'] = dp_attn

            matrix_values.append(entry)

    return matrix_values


def generate_custom_test(args):
    """Generate single 1k1k job for custom inputs.
    """
    try:
        with open(args.runner_config, 'r') as f:
            runner_config = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise ValueError(
            f"Runner config file '{args.runner_config}' does not exist.")
    
    found_runner_label = False
    for runner_type, runner_nodes in runner_config.items():
        if args.runner_label == runner_type or args.runner_label in runner_nodes:
            found_runner_label = True
    
    if not found_runner_label:
        raise ValueError(f"Unable to find specified runner label '{args.runner_label}'.")

    if not runner_nodes:
        raise ValueError(
            f"Runner '{args.runner_type}' does not exist in runner config '{args.runner_config}'. Must choose from existing runner types: '{', '.join(runner_config.keys())}'.")

    return [
        {
            'image': args.image,
            'model': args.model,
            'precision': args.precision,
            'framework': args.framework,
            'runner': args.runner_label,
            # Again, just use 1k1k since this is just meant to smoke test all runners
            'isl': 1024,
            'osl': 1024,
            'tp': 8,
            'ep': 1,
            'dp-attn': False,
            'conc': 4,
            'exp-name': args.exp_name,
            'max-model-len': 2048,
        }
    ]


def generate_runner_sweep_config(args, all_config_data):
    """Generate runner sweep configurations.

    Assumes all_config_data has been validated by validate_config_structure().
    """
    try:
        with open(args.runner_config, 'r') as f:
            runner_config = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise ValueError(
            f"Runner config file '{args.runner_config}' does not exist.")

    matrix_values = []
    for key, val in all_config_data.items():
        # Only consider configs with specified runner
        if not key.startswith(args.model_prefix):
            continue

        # Optionally filter by precision and framework
        if (args.precision and val['precision'] != args.precision) or (args.framework and val['framework'] != args.framework):
            continue

        # I.e., for 70b-fp4-... the model_code is 70b which is necessary for exp_name
        # so that it can be bubbled down to bash script benchmarks... this is probably a FIXME
        model_code = key.split('-')[0]

        runner_nodes = runner_config.get(val['runner'])
        if not runner_nodes:
            raise ValueError(
                f"Runner '{val['runner']}' does not exist in runner config '{args.runner_config}'. Must choose from existing runner types: '{', '.join(runner_config.keys())}'.")

        # Find 1k1k config
        target_config = None
        for config in val['seq-len-configs']:
            if config['isl'] == 1024 and config['osl'] == 1024:
                target_config = config
                break

        highest_tp_bmk = max(target_config['search-space'], key=lambda x: x['tp'])
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
                'ep': 1, # Default,
                'dp-attn': False, # Default
                'conc': lowest_conc,
                'exp-name': f"{model_code}_test",
                'max-model-len': 2048,
            }

            # Add optional fields if they exist
            if ep is not None:
                entry['ep'] = ep
            if dp_attn is not None:
                entry['dp-attn'] = dp_attn

            matrix_values.append(entry)

    if len(matrix_values) == 0:
        error_msg = f"No configs found matching model prefix '{args.model_prefix}'"
        if args.precision:
            error_msg += f", precision '{args.precision}'"
        if args.framework:
            error_msg += f", framework '{args.framework}'"
        raise ValueError(error_msg + ".")

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

    # Subcommand: filtered-sweep
    filtered_sweep_parser = subparsers.add_parser(
        'filtered-sweep',
        parents=[parent_parser],
        add_help=False,
        help='Generate sweep configurations with optional filtering by model, precision, framework, runner type, and sequence lengths'
    )
    filtered_sweep_parser.add_argument(
        '--model-prefix',
        nargs='+',
        required=False,
        help='Model prefix(es) to filter configurations (optional, can specify multiple)'
    )
    filtered_sweep_parser.add_argument(
        '--precision',
        nargs='+',
        required=False,
        help='Precision(s) to filter by (e.g., fp4, fp8) (optional, can specify multiple)'
    )
    filtered_sweep_parser.add_argument(
        '--framework',
        nargs='+',
        required=False,
        help='Framework(s) to filter by (e.g., vllm, trt, sglang) (optional, can specify multiple)'
    )
    filtered_sweep_parser.add_argument(
        '--runner-type',
        nargs='+',
        required=False,
        help='Runner type(s) to filter by (e.g., h200, h100) (optional, can specify multiple)'
    )
    filtered_sweep_parser.add_argument(
        '--runner-config',
        required=False,
        help='Configuration file holding runner information (required if --runner-type is specified)'
    )
    filtered_sweep_parser.add_argument(
        '--seq-lens',
        nargs='+',
        choices=list(seq_len_stoi.keys()),
        required=False,
        help=f"Sequence length configurations to include: {', '.join(seq_len_stoi.keys())}. If not specified, all sequence lengths are included."
    )
    filtered_sweep_parser.add_argument(
        '--step-size',
        type=int,
        default=2,
        help='Step size for concurrency values (default: 2)'
    )
    filtered_sweep_parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Test mode: only run highest TP with lowest concurrency for each matching config'
    )
    filtered_sweep_parser.add_argument(
        '-h', '--help',
        action='help',
        help='Show this help message and exit'
    )

    # Subcommand: test-config
    test_config_parser = subparsers.add_parser(
        'test-config',
        parents=[parent_parser],
        add_help=False,
        help='Given a config key, run that configuration as specified. Optionally specify --test-mode to only run one parallelism-concurrency pair for the config.'
    )
    test_config_parser.add_argument(
        '--runner-config',
        required=True,
        help='Configuration file holding runner information'
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
        help='Given a runner type, find all configurations matching the type, and run that configuration on all individual runner nodes for the specified runner type. This is meant to validate that all runner nodes work on all configurations for a runner type. For instance, to validate that all configs that specify an h200 runner successfully run across all h200 runner nodes.'
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

    # Subcommand: runner-sweep
    test_config_parser = subparsers.add_parser(
        'runner-sweep',
        parents=[parent_parser],
        add_help=False,
        help='Given a model (and optionally a precision and framework), find all configurations matching the inputs, and run those configurations across all compatible runner nodes. This is meant to validate all runner nodes that should run a particular model can. For instance, this should be used to validate that all runners nodes that should run gptoss-120b actually do so successfully.'
    )
    test_config_parser.add_argument(
        '--model-prefix',
        required=True,
        help='Model prefix (e.g., 70b)'
    )
    test_config_parser.add_argument(
        '--precision',
        required=False,
        help='Precision to filter by (e.g., fp4) (optional)'
    )
    test_config_parser.add_argument(
        '--framework',
        required=False,
        help='Framework to filter by (e.g., trt) (optional)'
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

    # Subcommand: custom
    test_config_parser = subparsers.add_parser(
        'custom',
        parents=[parent_parser],
        add_help=False,
        help='Enter custom values'
    )
    test_config_parser.add_argument(
        '--runner-label',
        required=True,
        help='Label associated with runner on which to launch the corresponding job (e.g., h200, h200-nv_1, etc.)'
    )
    test_config_parser.add_argument(
        '--image',
        required=True,
        help='Image to run the benchmark (e.g., openai/gpt-oss-120b)'
    )
    test_config_parser.add_argument(
        '--model',
        required=True,
        help='Model to run (e.g., vllm/vllm-openai:latest)'
    )
    test_config_parser.add_argument(
        '--framework',
        required=True,
        help='Framework to run on (e.g., vllm, trt, sglang)'
    )
    test_config_parser.add_argument(
        '--precision',
        required=True,
        help='Precision to run (e.g., fp4, fp8)'
    )
    test_config_parser.add_argument(
        '--exp-name',
        required=True,
        help='Experiment name (e.g., 70b_test)'
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
    elif args.command == 'filtered-sweep':
        matrix_values = generate_filtered_sweep(args, all_config_data)
    elif args.command == 'test-config':
        matrix_values = generate_test_config(args, all_config_data)
    elif args.command == 'runner-model-sweep':
        matrix_values = generate_runner_model_sweep_config(
            args, all_config_data)
    elif args.command == 'runner-sweep':
        matrix_values = generate_runner_sweep_config(
            args, all_config_data)
    elif args.command == 'custom':
        matrix_values = generate_custom_test(args)
    else:
        parser.error(f"Unknown command: {args.command}")

    # Validate output before printing
    validate_matrix_output(matrix_values)

    print(json.dumps(matrix_values))
    return matrix_values


if __name__ == "__main__":
    main()
