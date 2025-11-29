import json
import yaml
import argparse
from pydantic import BaseModel, Field, ValidationError, ConfigDict
from typing import List

# Field name constants
# Top-level config fields
FIELD_IMAGE = 'image'
FIELD_MODEL = 'model'
FIELD_MODEL_PREFIX = 'model-prefix'
FIELD_PRECISION = 'precision'
FIELD_FRAMEWORK = 'framework'
FIELD_RUNNER = 'runner'
FIELD_SEQ_LEN_CONFIGS = 'seq-len-configs'

# Seq-len-config fields
FIELD_ISL = 'isl'
FIELD_OSL = 'osl'
FIELD_SEARCH_SPACE = 'search-space'

# Search-space/benchmark fields
FIELD_TP = 'tp'
FIELD_CONC_START = 'conc-start'
FIELD_CONC_END = 'conc-end'
FIELD_EP = 'ep'
FIELD_DP_ATTN = 'dp-attn'

# Matrix entry fields
FIELD_CONC = 'conc'
FIELD_MAX_MODEL_LEN = 'max-model-len'
FIELD_EXP_NAME = 'exp-name'

# Eval
FIELD_RUN_EVAL = 'run-eval'

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
    run_eval: bool = Field(alias='run-eval', default=False)


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

def mark_eval_entries(matrix_values: List[dict]) -> List[dict]:
    """Mark entries that should run evaluation.
    
    For each unique (model, runner, isl, osl) combination:
    - Mark highest TP with highest conc
    - Mark lowest TP with highest conc
    """
    from collections import defaultdict
    
    # Group entries by (model, runner, isl, osl)
    groups = defaultdict(list)
    for i, entry in enumerate(matrix_values):
        key = (entry[FIELD_MODEL], entry[FIELD_RUNNER], entry[FIELD_ISL], entry[FIELD_OSL])
        groups[key].append((i, entry))
    
    # For each group, find highest TP/highest conc and lowest TP/highest conc
    eval_indices = set()
    for key, entries in groups.items():
        if not entries:
            continue
        
        # Find min and max TP values
        min_tp = min(e[FIELD_TP] for _, e in entries)
        max_tp = max(e[FIELD_TP] for _, e in entries)
        
        # Find highest conc for highest TP
        highest_tp_entries = [(i, e) for i, e in entries if e[FIELD_TP] == max_tp]
        if highest_tp_entries:
            max_conc_highest_tp = max(e[FIELD_CONC] for _, e in highest_tp_entries)
            for i, e in highest_tp_entries:
                if e[FIELD_CONC] == max_conc_highest_tp:
                    eval_indices.add(i)
        
        # Find highest conc for lowest TP (only if different from max_tp)
        if min_tp != max_tp:
            lowest_tp_entries = [(i, e) for i, e in entries if e[FIELD_TP] == min_tp]
            if lowest_tp_entries:
                max_conc_lowest_tp = max(e[FIELD_CONC] for _, e in lowest_tp_entries)
                for i, e in lowest_tp_entries:
                    if e[FIELD_CONC] == max_conc_lowest_tp:
                        eval_indices.add(i)
    
    # Mark the selected entries
    for i, entry in enumerate(matrix_values):
        entry[FIELD_RUN_EVAL] = i in eval_indices
    
    return matrix_values

def validate_master_configs_structure(all_config_data):
    """Validate the structure of all master config entries.

    This validates that all required fields are present, have correct types,
    and no extra fields exist. Should be called once after loading config files.
    """
    for key, val in all_config_data.items():
        # Check for required top-level fields and their types
        required_fields = {
            FIELD_IMAGE: str,
            FIELD_MODEL: str,
            FIELD_MODEL_PREFIX: str,
            FIELD_PRECISION: str,
            FIELD_FRAMEWORK: str,
            FIELD_RUNNER: str,
            FIELD_SEQ_LEN_CONFIGS: list
        }

        for field, expected_type in required_fields.items():
            if field not in val or val[field] is None:
                raise ValueError(
                    f"Missing required field '{field}' for key '{key}'")
            if not isinstance(val[field], expected_type):
                raise ValueError(
                    f"Field '{field}' must be {expected_type.__name__} for key '{key}', got {type(val[field]).__name__}")

        seq_len_configs = val[FIELD_SEQ_LEN_CONFIGS]
        if len(seq_len_configs) == 0:
            raise ValueError(
                f"'{FIELD_SEQ_LEN_CONFIGS}' must be a non-empty list for key '{key}'")

        # Validate each seq-len-config
        for i, seq_config in enumerate(seq_len_configs):
            # Check isl
            if FIELD_ISL not in seq_config or seq_config[FIELD_ISL] is None:
                raise ValueError(
                    f"Missing '{FIELD_ISL}' in seq-len-config[{i}] for key '{key}'")
            if not isinstance(seq_config[FIELD_ISL], int):
                raise ValueError(
                    f"'{FIELD_ISL}' must be int in seq-len-config[{i}] for key '{key}'")

            # Check osl
            if FIELD_OSL not in seq_config or seq_config[FIELD_OSL] is None:
                raise ValueError(
                    f"Missing '{FIELD_OSL}' in seq-len-config[{i}] for key '{key}'")
            if not isinstance(seq_config[FIELD_OSL], int):
                raise ValueError(
                    f"'{FIELD_OSL}' must be int in seq-len-config[{i}] for key '{key}'")

            bmk_space = seq_config.get(FIELD_SEARCH_SPACE)
            if not bmk_space or not isinstance(bmk_space, list) or len(bmk_space) == 0:
                raise ValueError(
                    f"Missing or invalid '{FIELD_SEARCH_SPACE}' in seq-len-config[{i}] for key '{key}'")

            # Validate each benchmark in search-space
            for j, bmk in enumerate(bmk_space):
                # Define allowed fields
                allowed_fields = {FIELD_TP, FIELD_CONC_START,
                                  FIELD_CONC_END, FIELD_EP, FIELD_DP_ATTN}
                required_bmk_fields = {FIELD_TP: int,
                                       FIELD_CONC_START: int, FIELD_CONC_END: int}
                optional_bmk_fields = {FIELD_EP: int, FIELD_DP_ATTN: bool}

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
    """Generate full sweep configurations with optional filtering.

    Supports filtering by model prefix, precision, framework, runner type, and sequence lengths.
    Supports test mode to only run highest TP with lowest concurrency.

    All filters are optional - can generate sweeps for all configs or filter by specific criteria.

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
        if args.precision and val[FIELD_PRECISION] not in args.precision:
            continue

        # Filter by framework if specified
        if args.framework and val[FIELD_FRAMEWORK] not in args.framework:
            continue

        # Filter by runner type if specified
        if args.runner_type and val[FIELD_RUNNER] not in args.runner_type:
            continue

        seq_len_configs = val[FIELD_SEQ_LEN_CONFIGS]
        image = val[FIELD_IMAGE]
        model = val[FIELD_MODEL]
        precision = val[FIELD_PRECISION]
        framework = val[FIELD_FRAMEWORK]
        runner = val[FIELD_RUNNER]
        model_code = val[FIELD_MODEL_PREFIX]

        for seq_config in seq_len_configs:
            isl = seq_config[FIELD_ISL]
            osl = seq_config[FIELD_OSL]

            # Filter by sequence lengths if specified
            if seq_lens_filter and (isl, osl) not in seq_lens_filter:
                continue

            bmk_space = seq_config[FIELD_SEARCH_SPACE]

            if args.test_mode:
                # In test mode, use highest TP with lowest concurrency
                highest_tp_bmk = max(bmk_space, key=lambda x: x[FIELD_TP])
                tp = highest_tp_bmk[FIELD_TP]
                conc = highest_tp_bmk[FIELD_CONC_START]
                ep = highest_tp_bmk.get(FIELD_EP)
                dp_attn = highest_tp_bmk.get(FIELD_DP_ATTN)

                seq_len_str = seq_len_to_str(isl, osl)
                entry = {
                    FIELD_IMAGE: image,
                    FIELD_MODEL: model,
                    FIELD_PRECISION: precision,
                    FIELD_FRAMEWORK: framework,
                    FIELD_RUNNER: runner,
                    FIELD_ISL: isl,
                    FIELD_OSL: osl,
                    FIELD_TP: tp,
                    FIELD_EP: 1,  # Default
                    FIELD_DP_ATTN: False,  # Default
                    FIELD_CONC: conc,
                    FIELD_MAX_MODEL_LEN: isl + osl + 200,
                    FIELD_EXP_NAME: f"{model_code}_{seq_len_str}",
                }

                if ep is not None:
                    entry[FIELD_EP] = ep
                if dp_attn is not None:
                    entry[FIELD_DP_ATTN] = dp_attn

                matrix_values.append(entry)
            else:
                # Full sweep mode
                for bmk in bmk_space:
                    tp = bmk[FIELD_TP]
                    conc_start = bmk[FIELD_CONC_START]
                    conc_end = bmk[FIELD_CONC_END]
                    ep = bmk.get(FIELD_EP)
                    dp_attn = bmk.get(FIELD_DP_ATTN)

                    conc = conc_start
                    while conc <= conc_end:
                        seq_len_str = seq_len_to_str(isl, osl)
                        entry = {
                            FIELD_IMAGE: image,
                            FIELD_MODEL: model,
                            FIELD_PRECISION: precision,
                            FIELD_FRAMEWORK: framework,
                            FIELD_RUNNER: runner,
                            FIELD_ISL: isl,
                            FIELD_OSL: osl,
                            FIELD_TP: tp,
                            FIELD_CONC: conc,
                            FIELD_MAX_MODEL_LEN: isl + osl + 200,
                            FIELD_EP: 1,  # Default
                            FIELD_DP_ATTN: False,  # Default
                            FIELD_EXP_NAME: f"{model_code}_{seq_len_str}",
                        }

                        if ep is not None:
                            entry[FIELD_EP] = ep
                        if dp_attn is not None:
                            entry[FIELD_DP_ATTN] = dp_attn

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

    val = all_config_data.get(args.key)

    if not val:
        raise ValueError(
            f"Specified key '{args.key}' does not exist in config files.")

    # Extract model code from config
    model_code = val[FIELD_MODEL_PREFIX]

    runner_nodes = runner_config.get(val[FIELD_RUNNER], [])
    if args.runner_node and args.runner_node not in runner_nodes:
        raise ValueError(
            f"Runner node '{args.runner_node}' is not compatible with config '{args.key}' which runs on runner type '{val[FIELD_RUNNER]}'. Available runner nodes for this config are '{', '.join(runner_nodes)}'.")

    seq_len_configs = val[FIELD_SEQ_LEN_CONFIGS]
    image = val[FIELD_IMAGE]
    model = val[FIELD_MODEL]
    precision = val[FIELD_PRECISION]
    framework = val[FIELD_FRAMEWORK]
    # Use default runner or specific runner node if input by user
    runner = val[FIELD_RUNNER] if not args.runner_node else args.runner_node

    # Convert seq-lens to set of (isl, osl) tuples for filtering
    seq_lens_filter = None
    if args.seq_lens:
        seq_lens_filter = {seq_len_stoi[sl] for sl in args.seq_lens}

    matrix_values = []

    # Process each sequence length configuration
    for seq_config in seq_len_configs:
        isl = seq_config[FIELD_ISL]
        osl = seq_config[FIELD_OSL]

        # Filter by sequence lengths if specified
        if seq_lens_filter and (isl, osl) not in seq_lens_filter:
            continue

        bmk_space = seq_config[FIELD_SEARCH_SPACE]

        for bmk in bmk_space:
            tp = bmk[FIELD_TP]
            conc_start = bmk[FIELD_CONC_START]
            conc_end = bmk[FIELD_CONC_END]
            ep = bmk.get(FIELD_EP)
            dp_attn = bmk.get(FIELD_DP_ATTN)

            # In test mode, only use the lowest concurrency (conc_start)
            if args.test_mode:
                entry = {
                    FIELD_IMAGE: image,
                    FIELD_MODEL: model,
                    FIELD_PRECISION: precision,
                    FIELD_FRAMEWORK: framework,
                    FIELD_RUNNER: runner,
                    FIELD_ISL: isl,
                    FIELD_OSL: osl,
                    FIELD_TP: tp,
                    FIELD_EP: 1, # Default,
                    FIELD_DP_ATTN: False, # Default
                    FIELD_CONC: conc_start,
                    FIELD_MAX_MODEL_LEN: isl + osl,
                    FIELD_EXP_NAME: f"{model_code}_test",
                }

                # Add optional fields if they exist
                if ep is not None:
                    entry[FIELD_EP] = ep
                if dp_attn is not None:
                    entry[FIELD_DP_ATTN] = dp_attn

                matrix_values.append(entry)
            else:
                # Generate entries for each concurrency value in the range
                conc = conc_start
                while conc <= conc_end:
                    seq_len_str = seq_len_to_str(isl, osl)
                    entry = {
                        FIELD_IMAGE: image,
                        FIELD_MODEL: model,
                        FIELD_PRECISION: precision,
                        FIELD_FRAMEWORK: framework,
                        FIELD_RUNNER: runner,
                        FIELD_ISL: isl,
                        FIELD_OSL: osl,
                        FIELD_TP: tp,
                        FIELD_EP: 1, # Default,
                        FIELD_DP_ATTN: False, # Default
                        FIELD_CONC: conc,
                        FIELD_MAX_MODEL_LEN: isl + osl,
                        FIELD_EXP_NAME: f"{model_code}_{seq_len_str}",
                    }

                    # Add optional fields if they exist
                    if ep is not None:
                        entry[FIELD_EP] = ep
                    if dp_attn is not None:
                        entry[FIELD_DP_ATTN] = dp_attn

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

    # Filter runner nodes if filter is specified
    if args.runner_node_filter:
        runner_nodes = [node for node in runner_nodes if args.runner_node_filter in node]
        if not runner_nodes:
            raise ValueError(
                f"No runner nodes found matching filter '{args.runner_node_filter}' for runner type '{args.runner_type}'.")

    matrix_values = []
    for key, val in all_config_data.items():
        # Only consider configs with specified runner
        if val[FIELD_RUNNER] != args.runner_type:
            continue

        # Get model code for exp_name
        model_code = val[FIELD_MODEL_PREFIX]

        # Find 1k1k config
        target_config = None
        for config in val[FIELD_SEQ_LEN_CONFIGS]:
            if config[FIELD_ISL] == 1024 and config[FIELD_OSL] == 1024:
                target_config = config
                break

        highest_tp_bmk = max(target_config[FIELD_SEARCH_SPACE], key=lambda x: x[FIELD_TP])
        # Since we are just testing, pick the highest TP for this config and just test
        # on that TP with the lowest concurrency available
        highest_tp = highest_tp_bmk[FIELD_TP]
        lowest_conc = highest_tp_bmk[FIELD_CONC_START]

        ep = highest_tp_bmk.get(FIELD_EP)
        dp_attn = highest_tp_bmk.get(FIELD_DP_ATTN)

        for node in runner_nodes:
            entry = {
                FIELD_IMAGE: val[FIELD_IMAGE],
                FIELD_MODEL: val[FIELD_MODEL],
                FIELD_PRECISION: val[FIELD_PRECISION],
                FIELD_FRAMEWORK: val[FIELD_FRAMEWORK],
                # Add one entry for each node under specified runner type
                FIELD_RUNNER: node,
                # Again, just use 1k1k since this is just meant to smoke test all runners
                FIELD_ISL: 1024,
                FIELD_OSL: 1024,
                FIELD_TP: highest_tp,
                FIELD_EP: 1, # Default,
                FIELD_DP_ATTN: False, # Default
                FIELD_CONC: lowest_conc,
                FIELD_MAX_MODEL_LEN: 2048,
                FIELD_EXP_NAME: f"{model_code}_test",
            }

            # Add optional fields if they exist
            if ep is not None:
                entry[FIELD_EP] = ep
            if dp_attn is not None:
                entry[FIELD_DP_ATTN] = dp_attn

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
            FIELD_IMAGE: args.image,
            FIELD_MODEL: args.model,
            FIELD_PRECISION: args.precision,
            FIELD_FRAMEWORK: args.framework,
            FIELD_RUNNER: args.runner_label,
            # Again, just use 1k1k since this is just meant to smoke test all runners
            FIELD_ISL: 1024,
            FIELD_OSL: 1024,
            FIELD_TP: 8,
            FIELD_EP: 1,
            FIELD_DP_ATTN: False,
            FIELD_CONC: 4,
            FIELD_EXP_NAME: args.exp_name,
            FIELD_MAX_MODEL_LEN: 2048,
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

    if not runner_config.get(args.runner_type):
        raise ValueError(
            f"Runner '{args.runner_type}' does not exist in runner config '{args.runner_config}'. Must choose from existing runner types: '{', '.join(runner_config.keys())}'.")


    matrix_values = []
    for key, val in all_config_data.items():
        # Only consider configs with specified runner
        if not key.startswith(args.model_prefix):
            continue

        if not val[FIELD_RUNNER] == args.runner_type:
            continue

        # Optionally filter by precision and framework
        if (args.precision and val[FIELD_PRECISION] != args.precision) or (args.framework and val[FIELD_FRAMEWORK] != args.framework):
            continue

        # Get model code for exp_name
        model_code = val[FIELD_MODEL_PREFIX]

        runner_nodes = runner_config.get(val[FIELD_RUNNER])
        if not runner_nodes:
            raise ValueError(
                f"Runner '{val[FIELD_RUNNER]}' does not exist in runner config '{args.runner_config}'. Must choose from existing runner types: '{', '.join(runner_config.keys())}'.")

        # Find 1k1k config
        target_config = None
        for config in val[FIELD_SEQ_LEN_CONFIGS]:
            if config[FIELD_ISL] == 1024 and config[FIELD_OSL] == 1024:
                target_config = config
                break

        highest_tp_bmk = max(target_config[FIELD_SEARCH_SPACE], key=lambda x: x[FIELD_TP])
        # Since we are just testing, pick the highest TP for this config and just test
        # on that TP with the lowest concurrency available
        highest_tp = highest_tp_bmk[FIELD_TP]
        lowest_conc = highest_tp_bmk[FIELD_CONC_START]

        ep = highest_tp_bmk.get(FIELD_EP)
        dp_attn = highest_tp_bmk.get(FIELD_DP_ATTN)

        for node in runner_nodes:
            entry = {
                FIELD_IMAGE: val[FIELD_IMAGE],
                FIELD_MODEL: val[FIELD_MODEL],
                FIELD_PRECISION: val[FIELD_PRECISION],
                FIELD_FRAMEWORK: val[FIELD_FRAMEWORK],
                # Add one entry for each node under specified runner type
                FIELD_RUNNER: node,
                # Again, just use 1k1k since this is just meant to smoke test all runners
                FIELD_ISL: 1024,
                FIELD_OSL: 1024,
                FIELD_TP: highest_tp,
                FIELD_EP: 1, # Default,
                FIELD_DP_ATTN: False, # Default
                FIELD_CONC: lowest_conc,
                FIELD_EXP_NAME: f"{model_code}_test",
                FIELD_MAX_MODEL_LEN: 2048,
            }

            # Add optional fields if they exist
            if ep is not None:
                entry[FIELD_EP] = ep
            if dp_attn is not None:
                entry[FIELD_DP_ATTN] = dp_attn

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
        '--runner-config',
        required=False,
        help='Configuration file holding runner information (required if --runner-type is specified)'
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
        '--test-mode',
        action='store_true',
        help='Test mode: only run highest TP with lowest concurrency for each matching config'
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
        help='Runner type (e.g., b200-trt, h100)'
    )
    test_config_parser.add_argument(
        '--runner-config',
        required=True,
        help='Configuration file holding runner information'
    )
    test_config_parser.add_argument(
        '--runner-node-filter',
        required=False,
        help='Filter runner nodes by substring match (e.g., "mi300x-amd" to only include nodes containing that string)'
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
        '--runner-type',
        required=True,
        help='Runner type (e.g., b200-trt, h100)'
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

    # Choose eval
    matrix_values = mark_eval_entries(matrix_values)

    # Validate output before printing
    validate_matrix_output(matrix_values)

    print(json.dumps(matrix_values))
    return matrix_values


if __name__ == "__main__":
    main()
