from pydantic import BaseModel, Field, ValidationError, ConfigDict, model_validator
from typing import List, Optional, Union, Literal
from enum import Enum

import pprint
import yaml

"""
    The below class defines the field names expected to be present in the JSON entries
    for both single-node and multi-node configurations.
"""


class Fields(Enum):
    # Field name constants
    # Top-level config fields
    IMAGE = 'image'
    MODEL = 'model'
    MODEL_PREFIX = 'model-prefix'
    PRECISION = 'precision'
    FRAMEWORK = 'framework'
    RUNNER = 'runner'
    SEQ_LEN_CONFIGS = 'seq-len-configs'
    MULTINODE = 'multinode'

    # Seq-len-config fields
    ISL = 'isl'
    OSL = 'osl'
    SEARCH_SPACE = 'search-space'

    # Search-space/benchmark fields
    TP = 'tp'
    CONC_START = 'conc-start'
    CONC_END = 'conc-end'
    CONC_LIST = 'conc-list'
    EP = 'ep'
    DP_ATTN = 'dp-attn'

    # Multinode-specific fields (when MULTINODE = true)
    SPEC_DECODING = 'spec-decoding'
    PREFILL = 'prefill'
    DECODE = 'decode'
    NUM_WORKER = 'num-worker'
    BATCH_SIZE = 'batch-size'
    MAX_NUM_TOKENS = 'max-num-tokens'
    ADDITIONAL_SETTINGS = 'additional-settings'

    # Matrix entry fields
    CONC = 'conc'
    MAX_MODEL_LEN = 'max-model-len'
    EXP_NAME = 'exp-name'
    DISAGG = 'disagg'

    # Eval
    RUN_EVAL = 'run-eval'


"""
    Below is the validation logic for the OUTPUT of utils/matrix_logic/generate_sweep_configs.py, i.e., 
    the input to the actual workflow files. The validation enforces a strict set of rules on the structure
    of the generated matrix entries to ensure correctness before proceeding with benchmarking. This ensures
    that no validation has to happen in the workflow itself, i.e., at runtime, it is assumed that all inputs
    are valid. Threfore, there should not be any default values set in these Pydantic models. Any missing value
    should raise a validation error.
"""


class SingleNodeMatrixEntry(BaseModel):
    """Pydantic model for validating single node matrix entry structure.
    This validates the input that should be expected to .github/workflows/benchmark-tmpl.yml"""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    image: str
    model: str
    model_prefix: str = Field(alias=Fields.MODEL_PREFIX.value)
    precision: str
    framework: str
    spec_decoding: Literal["mtp", "draft_model", "none"] = Field(
        alias=Fields.SPEC_DECODING.value
    )
    runner: str
    isl: int
    osl: int
    tp: int
    ep: int
    dp_attn: bool = Field(alias=Fields.DP_ATTN.value)
    conc: Union[int, List[int]]
    max_model_len: int = Field(alias=Fields.MAX_MODEL_LEN.value)
    exp_name: str = Field(alias=Fields.EXP_NAME.value)
    disagg: bool
    run_eval: bool = Field(alias=Fields.RUN_EVAL.value)


class WorkerConfig(BaseModel):
    """Pydantic model for validating worker configuration in multinode entries."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    num_worker: int = Field(alias=Fields.NUM_WORKER.value)
    tp: int
    ep: int
    dp_attn: bool = Field(alias=Fields.DP_ATTN.value)
    additional_settings: Optional[List[str]] = Field(
        default=[], alias=Fields.ADDITIONAL_SETTINGS.value)


class MultiNodeMatrixEntry(BaseModel):
    """Pydantic model for validating multinode matrix entry structure.
    This validates the input that should be expected to .github/workflows/benchmark-multinode-tmpl.yml"""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    image: str
    model: str
    model_prefix: str = Field(alias=Fields.MODEL_PREFIX.value)
    precision: str
    framework: str
    spec_decoding: Literal["mtp", "draft_model", "none"] = Field(
        alias=Fields.SPEC_DECODING.value
    )
    runner: str
    isl: int
    osl: int
    prefill: WorkerConfig
    decode: WorkerConfig
    conc: List[int]
    max_model_len: int = Field(alias=Fields.MAX_MODEL_LEN.value)
    exp_name: str = Field(alias=Fields.EXP_NAME.value)
    disagg: bool
    run_eval: bool = Field(alias=Fields.RUN_EVAL.value)


def validate_matrix_entry(entry: dict, is_multinode: bool) -> dict:
    """Validate that matrix_values entries match the expected structure.

    Raises ValueError if any entry fails validation.
    Returns the original list if all entries are valid.
    """
    try:
        if is_multinode:
            MultiNodeMatrixEntry(**entry)
        else:
            SingleNodeMatrixEntry(**entry)
    except ValidationError as e:
        raise ValueError(
            f"The following parsed matrix entry failed validation:\n{pprint.pformat(entry)}\n{e}")
    return entry


"""
    Below is the validation logic for the INPUT to utils/matrix_logic/generate_sweep_configs.py, i.e., 
    the master configuration files found in .github/configs. The validation enforces a strict set of 
    rules on the structure of the master configuration files to ensure correctness before proceeding 
    with matrix generation.
"""


def _validate_conc_fields(self):
    """Ensure either (conc_start AND conc_end) OR conc_list is provided, but not both."""
    has_range = self.conc_start is not None and self.conc_end is not None
    has_list = self.conc_list is not None and len(self.conc_list) > 0

    if has_range and has_list:
        raise ValueError(
            f"Cannot specify both '{Fields.CONC_LIST.value}' list and "
            f"'{Fields.CONC_START.value}'/'{Fields.CONC_END.value}'. "
            "Use either a list or a range, not both."
        )

    if not has_range and not has_list:
        raise ValueError(
            f"Must specify either '{Fields.CONC_LIST.value}' list or both "
            f"'{Fields.CONC_START.value}' and '{Fields.CONC_END.value}'."
        )

    if has_range:
        if self.conc_start is None or self.conc_end is None:
            raise ValueError(
                f"Both '{Fields.CONC_START.value}' and '{Fields.CONC_END.value}' "
                "must be provided together."
            )

        if self.conc_start > self.conc_end:
            raise ValueError(
                f"'{Fields.CONC_START.value}' ({self.conc_start}) must be <= "
                f"'{Fields.CONC_END.value}' ({self.conc_end})."
            )

    if has_list:
        if not all(x > 0 for x in self.conc_list):
            raise ValueError(
                f"Input '{Fields.CONC_LIST.value}' entries must be greater than 0."
            )

    return self


class SingleNodeSearchSpaceEntry(BaseModel):
    """Single node search space configuration."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    tp: int
    ep: Optional[int] = None
    spec_decoding: Literal["mtp", "draft_model", "none"] = Field(
        default="none", alias=Fields.SPEC_DECODING.value)
    dp_attn: Optional[bool] = Field(
        default=None, alias=Fields.DP_ATTN.value)
    conc_start: Optional[int] = Field(
        default=None, alias=Fields.CONC_START.value)
    conc_end: Optional[int] = Field(
        default=None, alias=Fields.CONC_END.value)
    conc_list: Optional[List[int]] = Field(
        default=None, alias=Fields.CONC_LIST.value)

    @model_validator(mode='after')
    def validate_conc_fields(self):
        return _validate_conc_fields(self)


class MultiNodeSearchSpaceEntry(BaseModel):
    """Multinode search space configuration."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    spec_decoding: Literal["mtp", "draft_model", "none"] = Field(
        default="none", alias=Fields.SPEC_DECODING.value)
    prefill: WorkerConfig
    decode: WorkerConfig
    conc_start: Optional[int] = Field(
        default=None, alias=Fields.CONC_START.value)
    conc_end: Optional[int] = Field(
        default=None, alias=Fields.CONC_END.value)
    conc_list: Optional[List[int]] = Field(
        default=None, alias=Fields.CONC_LIST.value)

    @model_validator(mode='after')
    def validate_conc_fields(self):
        return _validate_conc_fields(self)


class SingleNodeSeqLenConfig(BaseModel):
    """Single node sequence length configuration."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    isl: int
    osl: int
    search_space: List[SingleNodeSearchSpaceEntry] = Field(
        alias=Fields.SEARCH_SPACE.value)


class MultiNodeSeqLenConfig(BaseModel):
    """Multinode sequence length configuration."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    isl: int
    osl: int
    search_space: List[MultiNodeSearchSpaceEntry] = Field(
        alias=Fields.SEARCH_SPACE.value)


class SingleNodeMasterConfigEntry(BaseModel):
    """Top-level single node master configuration entry."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    image: str
    model: str
    model_prefix: str = Field(alias=Fields.MODEL_PREFIX.value)
    precision: str
    framework: str
    runner: str
    multinode: Literal[False]
    disagg: bool = Field(default=False)
    seq_len_configs: List[SingleNodeSeqLenConfig] = Field(
        alias=Fields.SEQ_LEN_CONFIGS.value)


class MultiNodeMasterConfigEntry(BaseModel):
    """Top-level multinode master configuration entry."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    image: str
    model: str
    model_prefix: str = Field(alias=Fields.MODEL_PREFIX.value)
    precision: str
    framework: str
    runner: str
    multinode: Literal[True]
    disagg: bool = Field(default=False)
    seq_len_configs: List[MultiNodeSeqLenConfig] = Field(
        alias=Fields.SEQ_LEN_CONFIGS.value)


def validate_master_config(master_configs: dict) -> List[dict]:
    """Validate input master configuration structure."""
    for key, entry in master_configs.items():
        is_multinode = entry.get('multinode', False)

        try:
            if is_multinode:
                MultiNodeMasterConfigEntry(**entry)
            else:
                SingleNodeMasterConfigEntry(**entry)
        except ValidationError as e:
            raise ValueError(
                f"Master config entry '{key}' failed validation:\n{e}")
    return master_configs

# Runner Config Validation


def validate_runner_config(runner_configs: dict) -> List[dict]:
    """Validate input master configuration structure."""
    for key, value in runner_configs.items():
        if not isinstance(value, list):
            raise ValueError(
                f"Runner config entry '{key}' must be a list, got {type(value).__name__}")

        if not all(isinstance(item, str) for item in value):
            raise ValueError(
                f"Runner config entry '{key}' must contain only strings")

        if not value:
            raise ValueError(
                f"Runner config entry '{key}' cannot be an empty list")

    return runner_configs


"""
    Below is the validation logic for the changelog entries found in perf-changelog.yaml.
    This ensures that the changelog entries conform to the expected structure before
    proceeding with processing.
"""


class ChangelogEntry(BaseModel):
    """Pydantic model for validating changelog entry structure."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    config_keys: list[str] = Field(alias="config-keys", min_length=1)
    description: list[str] = Field(min_length=1)
    pr_link: str = Field(alias="pr-link")
    evals_only: bool = Field(alias="evals-only", default=False)


class ChangelogMetadata(BaseModel):
    """Pydantic model for validating changelog metadata structure."""
    model_config = ConfigDict(extra="forbid")

    base_ref: str
    head_ref: str
    entries: list[ChangelogEntry]


class ChangelogMatrixEntry(BaseModel):
    """Pydantic model for validating final changelog matrix entry structure.
    This imposes a strict contract on the output of process_changelog.py, dictated by
    the expected input to the run-sweep.yml workflow file.
    """
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    single_node: dict[str, list[SingleNodeMatrixEntry]
                      ] = Field(default_factory=dict)
    multi_node: dict[str, list[MultiNodeMatrixEntry]
                     ] = Field(default_factory=dict)
    changelog_metadata: ChangelogMetadata


# =============================================================================
# File Loading Functions
# =============================================================================


def load_config_files(config_files: List[str], validate: bool = True) -> dict:
    """Load and merge configuration files.

    Args:
        config_files: List of paths to YAML configuration files.
        validate: If True, run validate_master_config on loaded data. Defaults to True.

    Returns:
        Merged configuration dictionary.

    Raises:
        ValueError: If file doesn't exist, isn't a dict, or has duplicate keys.
    """
    all_config_data = {}
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
                assert isinstance(
                    config_data, dict), f"Config file '{config_file}' must contain a dictionary"

                # Don't allow '*' wildcard in master config keys as we need to reserve these
                # for expansion in process_changelog.py
                for key in config_data.keys():
                    if "*" in key:
                        raise ValueError(
                            f" Wildcard '*' is not allowed in master config keys: '{key}'")

                # Check for duplicate keys
                duplicate_keys = set(all_config_data.keys()) & set(
                    config_data.keys())
                if duplicate_keys:
                    raise ValueError(
                        f"Duplicate configuration keys found in '{config_file}': {', '.join(sorted(duplicate_keys))}"
                    )

                all_config_data.update(config_data)
        except FileNotFoundError:
            raise ValueError(f"Input file '{config_file}' does not exist.")

    if validate:
        validate_master_config(all_config_data)

    return all_config_data


def load_runner_file(runner_file: str, validate: bool = True) -> dict:
    """Load runner configuration file.

    Args:
        runner_file: Path to the runner YAML configuration file.
        validate: If True, run validate_runner_config on loaded data. Defaults to True.

    Returns:
        Runner configuration dictionary.

    Raises:
        ValueError: If file doesn't exist or fails validation.
    """
    try:
        with open(runner_file, 'r') as f:
            runner_config = yaml.safe_load(f)
    except FileNotFoundError:
        raise ValueError(
            f"Runner config file '{runner_file}' does not exist.")

    if validate:
        validate_runner_config(runner_config)

    return runner_config
