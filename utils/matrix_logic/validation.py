from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from typing import List, Optional, Union, Literal, Dict
from enum import Enum

import pprint
import yaml

CLUSTER_LABEL_PREFIX = "cluster:"
DEFAULT_AGENTIC_DURATION_SECONDS = 3600

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
    HARDWARE = 'hardware'
    SCENARIOS = 'scenarios'
    MULTINODE = 'multinode'

    # Scenario type keys
    FIXED_SEQ_LEN = 'fixed-seq-len'
    AGENTIC_CODING = 'agentic-coding'

    # Seq-len-config fields
    ISL = 'isl'
    OSL = 'osl'
    SEARCH_SPACE = 'search-space'

    # Search-space/benchmark fields
    TP = 'tp'
    PP = 'pp'
    DCP_SIZE = 'dcp-size'
    PCP_SIZE = 'pcp-size'
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

    # Agentic coding fields
    KV_OFFLOADING = 'kv-offloading'
    KV_OFFLOAD_BACKEND = 'kv-offload-backend'
    ROUTER = 'router'
    KV_P2P_TRANSFER = 'kv-p2p-transfer'
    TOTAL_CPU_DRAM_GB = 'total-cpu-dram-gb'
    AVAILABLE_CPU_DRAM_MIB = 'available-cpu-dram-mib'
    DRAM_UTILIZATION = 'dram-utilization'
    GPUS_PER_NODE = 'gpus-per-node'
    DURATION = 'duration'

    # Matrix entry fields
    CONC = 'conc'
    MAX_MODEL_LEN = 'max-model-len'
    EXP_NAME = 'exp-name'
    DISAGG = 'disagg'
    SCENARIO_TYPE = 'scenario-type'

    # Eval
    RUN_EVAL = 'run-eval'
    EVAL_ONLY = 'eval-only'
    EVAL_CONC = 'eval-conc'
    EVAL_ALL_CONCS = 'eval-all-concs'


"""
    Below is the validation logic for the OUTPUT of utils/matrix_logic/generate_sweep_configs.py, i.e., 
    the input to the actual workflow files. The validation enforces a strict set of rules on the structure
    of the generated matrix entries to ensure correctness before proceeding with benchmarking. This ensures
    that no validation has to happen in the workflow itself, i.e., at runtime, it is assumed that all inputs
    are valid. Threfore, there should not be any default values set in these Pydantic models. Any missing value
    should raise a validation error.
"""


class ComponentMetadata(BaseModel):
    """Strict name and version metadata for an optional runtime component."""
    model_config = ConfigDict(extra='forbid')

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)

    @field_validator('version')
    @classmethod
    def validate_component_version(cls, version: str) -> str:
        """Require the component's own version rather than its image provenance."""
        if version.startswith('image:'):
            raise ValueError(
                "component version must be a release, package version, or "
                "source commit, not an image reference"
            )
        return version


class KVOffloadBackendMetadata(BaseModel):
    """KV offload backend metadata with an optional independent version."""
    model_config = ConfigDict(extra='forbid')

    name: str = Field(min_length=1)
    version: Optional[str] = Field(default=None, min_length=1)

    @field_validator('version')
    @classmethod
    def validate_component_version(cls, version: Optional[str]) -> Optional[str]:
        """Reject image provenance when an independent version is available."""
        if version is not None and version.startswith('image:'):
            raise ValueError(
                "component version must be a release, package version, or "
                "source commit, not an image reference"
            )
        return version


def _validate_tp_context_topology(self):
    """Validate TP/DCP topology shared by single-node and worker schemas."""
    if self.tp % self.dcp_size != 0:
        raise ValueError(
            f"'{Fields.TP.value}' ({self.tp}) must be divisible by "
            f"'{Fields.DCP_SIZE.value}' ({self.dcp_size})"
        )
    return self


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
    pp: int = Field(gt=0, strict=True)
    dcp_size: int = Field(alias=Fields.DCP_SIZE.value, gt=0, strict=True)
    pcp_size: int = Field(alias=Fields.PCP_SIZE.value, gt=0, strict=True)
    ep: int
    dp_attn: bool = Field(alias=Fields.DP_ATTN.value)
    conc: Union[int, List[int]]
    max_model_len: int = Field(alias=Fields.MAX_MODEL_LEN.value)
    exp_name: str = Field(alias=Fields.EXP_NAME.value)
    disagg: Literal[False]
    run_eval: bool = Field(alias=Fields.RUN_EVAL.value)
    eval_only: bool = Field(alias=Fields.EVAL_ONLY.value, default=False)
    router: Optional[ComponentMetadata] = None

    @model_validator(mode='after')
    def validate_single_node_topology(self):
        return _validate_tp_context_topology(self)


class WorkerConfig(BaseModel):
    """Pydantic model for validating worker configuration in multinode entries."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    num_worker: int = Field(alias=Fields.NUM_WORKER.value)
    tp: int
    pp: int = Field(default=1, gt=0, strict=True)
    dcp_size: int = Field(
        default=1, alias=Fields.DCP_SIZE.value, gt=0, strict=True)
    pcp_size: int = Field(
        default=1, alias=Fields.PCP_SIZE.value, gt=0, strict=True)
    ep: int
    dp_attn: bool = Field(alias=Fields.DP_ATTN.value)
    hardware: Optional[str] = Field(default=None, min_length=1)
    additional_settings: Optional[List[str]] = Field(
        default=[], alias=Fields.ADDITIONAL_SETTINGS.value)

    @model_validator(mode='after')
    def validate_worker_topology(self):
        return _validate_tp_context_topology(self)


def _validate_worker_hardware_pair(self):
    """Require prefill and decode workers to declare hardware together."""
    if bool(self.prefill.hardware) != bool(self.decode.hardware):
        raise ValueError(
            f"'{Fields.HARDWARE.value}' must be specified for both "
            f"'{Fields.PREFILL.value}' and '{Fields.DECODE.value}', or neither"
        )
    return self


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
    eval_only: bool = Field(alias=Fields.EVAL_ONLY.value, default=False)
    eval_conc: Optional[int] = Field(default=None, alias=Fields.EVAL_CONC.value)
    eval_all_concs: bool = Field(
        default=False, alias=Fields.EVAL_ALL_CONCS.value
    )
    router: Optional[ComponentMetadata] = None
    kv_p2p_transfer: Optional[str] = Field(
        default=None, alias=Fields.KV_P2P_TRANSFER.value, min_length=1
    )

    @model_validator(mode='after')
    def validate_worker_hardware_pair(self):
        return _validate_worker_hardware_pair(self)

    @model_validator(mode='after')
    def validate_disagg_transfer(self):
        if self.disagg and self.kv_p2p_transfer is None:
            raise ValueError(
                f"{Fields.DISAGG.value}=true requires "
                f"{Fields.KV_P2P_TRANSFER.value}"
            )
        return self


class SingleNodeAgenticMatrixEntry(BaseModel):
    """Pydantic model for validating single-node agentic coding matrix entries."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    image: str
    model: str
    model_prefix: str = Field(alias=Fields.MODEL_PREFIX.value)
    precision: str
    framework: str
    runner: str
    tp: int
    pp: int = Field(gt=0, strict=True)
    dcp_size: int = Field(alias=Fields.DCP_SIZE.value, gt=0, strict=True)
    pcp_size: int = Field(alias=Fields.PCP_SIZE.value, gt=0, strict=True)
    ep: int
    dp_attn: bool = Field(alias=Fields.DP_ATTN.value)
    spec_decoding: Literal["mtp", "draft_model", "none"] = Field(
        default="none", alias=Fields.SPEC_DECODING.value
    )
    conc: int
    kv_offloading: Literal["none", "dram"] = Field(
        alias=Fields.KV_OFFLOADING.value
    )
    kv_offload_backend: Optional[KVOffloadBackendMetadata] = Field(
        default=None, alias=Fields.KV_OFFLOAD_BACKEND.value
    )
    router: Optional[ComponentMetadata] = None
    total_cpu_dram_gb: int = Field(alias=Fields.TOTAL_CPU_DRAM_GB.value, ge=0)
    duration: int = Field(alias=Fields.DURATION.value)
    exp_name: str = Field(alias=Fields.EXP_NAME.value)
    scenario_type: str = Field(alias=Fields.SCENARIO_TYPE.value)
    # Present on agentic eval rows (SWE-bench selection via mark_eval_entries);
    # benchmark rows are generated with --no-evals and omit both fields.
    run_eval: bool = Field(alias=Fields.RUN_EVAL.value, default=False)
    eval_only: bool = Field(alias=Fields.EVAL_ONLY.value, default=False)

    @model_validator(mode='after')
    def validate_kv_offload_fields(self):
        return _validate_kv_offload_fields(self)

    @model_validator(mode='after')
    def validate_single_node_topology(self):
        return _validate_tp_context_topology(self)


class MultiNodeAgenticMatrixEntry(BaseModel):
    """Pydantic model for validating multinode agentic coding matrix entries."""
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
    prefill: WorkerConfig
    decode: WorkerConfig
    conc: list[int]
    kv_offloading: Literal["none", "dram"] = Field(alias=Fields.KV_OFFLOADING.value)
    kv_offload_backend: Optional[KVOffloadBackendMetadata] = Field(
        default=None, alias=Fields.KV_OFFLOAD_BACKEND.value
    )
    router: Optional[ComponentMetadata] = None
    kv_p2p_transfer: Optional[str] = Field(
        default=None, alias=Fields.KV_P2P_TRANSFER.value, min_length=1
    )
    duration: int = Field(alias=Fields.DURATION.value)
    exp_name: str = Field(alias=Fields.EXP_NAME.value)
    disagg: bool
    scenario_type: str = Field(alias=Fields.SCENARIO_TYPE.value)

    @model_validator(mode='after')
    def validate_worker_hardware_pair(self):
        return _validate_worker_hardware_pair(self)

    @model_validator(mode='after')
    def validate_kv_offload_fields(self):
        return _validate_kv_offload_fields(self)

    @model_validator(mode='after')
    def validate_disagg_transfer(self):
        if self.disagg and self.kv_p2p_transfer is None:
            raise ValueError(
                f"{Fields.DISAGG.value}=true requires "
                f"{Fields.KV_P2P_TRANSFER.value}"
            )
        return self


AgenticMatrixEntry = Union[SingleNodeAgenticMatrixEntry, MultiNodeAgenticMatrixEntry]


def validate_agentic_matrix_entry(entry: dict) -> dict:
    """Validate that an agentic matrix entry matches the expected structure."""
    try:
        if Fields.PREFILL.value in entry:
            MultiNodeAgenticMatrixEntry(**entry)
        else:
            SingleNodeAgenticMatrixEntry(**entry)
    except ValidationError as e:
        raise ValueError(
            f"The following parsed agentic matrix entry failed validation:\n{pprint.pformat(entry)}\n{e}")
    return entry


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
    the master configuration files found in configs. The validation enforces a strict set of
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

        if self.conc_start <= 0 or self.conc_end <= 0:
            raise ValueError(
                f"Input '{Fields.CONC_START.value}' and "
                f"'{Fields.CONC_END.value}' must be greater than 0."
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


def _validate_agentic_runner_is_cluster(runner: str, scenarios) -> None:
    if scenarios.agentic_coding and not runner.startswith(CLUSTER_LABEL_PREFIX):
        raise ValueError(
            f"Agentic master configs must use a '{CLUSTER_LABEL_PREFIX}<name>' runner "
            "so every point runs on one exact hardware fleet."
        )


def _validate_kv_offload_fields(self):
    backend = getattr(self, "kv_offload_backend", None)
    if self.kv_offloading is None:
        if backend is not None:
            raise ValueError(
                f"{Fields.KV_OFFLOAD_BACKEND.value} requires "
                f"{Fields.KV_OFFLOADING.value}"
            )
        return self
    if self.kv_offloading == "none":
        if backend is not None:
            raise ValueError(
                f"{Fields.KV_OFFLOAD_BACKEND.value} can only be set when "
                f"{Fields.KV_OFFLOADING.value} is not 'none'"
            )
        return self
    if backend is None:
        raise ValueError(
            f"{Fields.KV_OFFLOAD_BACKEND.value} is required when "
            f"{Fields.KV_OFFLOADING.value} is '{self.kv_offloading}'"
        )
    return self


class SingleNodeSearchSpaceEntry(BaseModel):
    """Single node search space configuration."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    tp: int
    pp: int = Field(default=1, gt=0, strict=True)
    dcp_size: int = Field(
        default=1, alias=Fields.DCP_SIZE.value, gt=0, strict=True)
    pcp_size: int = Field(
        default=1, alias=Fields.PCP_SIZE.value, gt=0, strict=True)
    ep: Optional[int] = None
    spec_decoding: Literal["mtp", "draft_model", "none"] = Field(
        default="none", alias=Fields.SPEC_DECODING.value)
    dp_attn: Optional[bool] = Field(
        default=None, alias=Fields.DP_ATTN.value)
    router: Optional[ComponentMetadata] = None
    conc_start: Optional[int] = Field(
        default=None, alias=Fields.CONC_START.value)
    conc_end: Optional[int] = Field(
        default=None, alias=Fields.CONC_END.value)
    conc_list: Optional[List[int]] = Field(
        default=None, alias=Fields.CONC_LIST.value)

    @model_validator(mode='after')
    def validate_conc_fields(self):
        return _validate_conc_fields(self)

    @model_validator(mode='after')
    def validate_single_node_topology(self):
        return _validate_tp_context_topology(self)


class MultiNodeSearchSpaceEntry(BaseModel):
    """Multinode search space configuration."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    spec_decoding: Literal["mtp", "draft_model", "none"] = Field(
        default="none", alias=Fields.SPEC_DECODING.value)
    prefill: WorkerConfig
    decode: WorkerConfig
    router: Optional[ComponentMetadata] = None
    kv_p2p_transfer: Optional[str] = Field(
        default=None, alias=Fields.KV_P2P_TRANSFER.value, min_length=1
    )
    conc_start: Optional[int] = Field(
        default=None, alias=Fields.CONC_START.value)
    conc_end: Optional[int] = Field(
        default=None, alias=Fields.CONC_END.value)
    conc_list: Optional[List[int]] = Field(
        default=None, alias=Fields.CONC_LIST.value)

    @model_validator(mode='after')
    def validate_conc_fields(self):
        return _validate_conc_fields(self)

    @model_validator(mode='after')
    def validate_worker_hardware_pair(self):
        return _validate_worker_hardware_pair(self)


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


class AgenticCodingSearchSpaceEntry(BaseModel):
    """Agentic coding search space configuration."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    tp: Optional[int] = None
    pp: int = Field(default=1, gt=0, strict=True)
    dcp_size: int = Field(
        default=1, alias=Fields.DCP_SIZE.value, gt=0, strict=True)
    pcp_size: int = Field(
        default=1, alias=Fields.PCP_SIZE.value, gt=0, strict=True)
    ep: Optional[int] = None
    dp_attn: Optional[bool] = Field(default=None, alias=Fields.DP_ATTN.value)
    spec_decoding: Literal["mtp", "draft_model", "none"] = Field(
        default="none", alias=Fields.SPEC_DECODING.value)
    prefill: Optional[WorkerConfig] = None
    decode: Optional[WorkerConfig] = None
    kv_offloading: Optional[Literal["none", "dram"]] = Field(
        default=None, alias=Fields.KV_OFFLOADING.value
    )
    kv_offload_backend: Optional[KVOffloadBackendMetadata] = Field(
        default=None, alias=Fields.KV_OFFLOAD_BACKEND.value
    )
    router: Optional[ComponentMetadata] = None
    kv_p2p_transfer: Optional[str] = Field(
        default=None, alias=Fields.KV_P2P_TRANSFER.value, min_length=1
    )
    conc_start: Optional[int] = Field(default=None, alias=Fields.CONC_START.value)
    conc_end: Optional[int] = Field(default=None, alias=Fields.CONC_END.value)
    conc_list: Optional[List[int]] = Field(default=None, alias=Fields.CONC_LIST.value)

    @model_validator(mode='after')
    def validate_conc_fields(self):
        return _validate_conc_fields(self)

    @model_validator(mode='after')
    def validate_kv_offload_fields(self):
        return _validate_kv_offload_fields(self)

    @model_validator(mode='after')
    def validate_topology_fields(self):
        has_single_node = self.tp is not None
        has_any_multinode_field = self.prefill is not None or self.decode is not None
        has_complete_multinode = self.prefill is not None and self.decode is not None
        if has_single_node:
            valid = not has_any_multinode_field
        else:
            valid = has_complete_multinode
        if not valid:
            raise ValueError("Agentic search-space entries must specify either tp or both prefill and decode")
        if has_single_node:
            if self.kv_offloading is None:
                raise ValueError(
                    f"Single-node agentic search-space entries must specify "
                    f"{Fields.KV_OFFLOADING.value}"
                )
            _validate_tp_context_topology(self)
        if has_complete_multinode:
            explicitly_single_node_fields = {
                "pp",
                "dcp_size",
                "pcp_size",
            } & self.model_fields_set
            if explicitly_single_node_fields:
                field_names = ", ".join(
                    f"'{name}'"
                    for name in (
                        Fields.PP.value,
                        Fields.DCP_SIZE.value,
                        Fields.PCP_SIZE.value,
                    )
                )
                raise ValueError(
                    "Multinode agentic search-space entries cannot specify "
                    f"{field_names}"
                )
            _validate_worker_hardware_pair(self)
        return self

class AgenticCodingConfig(BaseModel):
    """Agentic coding scenario configuration for trace replay benchmarks."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    search_space: List[AgenticCodingSearchSpaceEntry] = Field(alias=Fields.SEARCH_SPACE.value)
    dram_utilization: Optional[float] = Field(
        default=None, alias=Fields.DRAM_UTILIZATION.value, gt=0, le=1
    )

    @model_validator(mode='after')
    def validate_dram_offload_capacity(self):
        for entry in self.search_space:
            if entry.kv_offloading != "dram":
                continue
            if self.dram_utilization is None:
                raise ValueError(
                    f"{Fields.KV_OFFLOADING.value}='dram' requires "
                    f"{Fields.DRAM_UTILIZATION.value} with runner hardware metadata"
                )
        return self


class SingleNodeScenarios(BaseModel):
    """Scenarios wrapper for single-node configs."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    fixed_seq_len: Optional[List[SingleNodeSeqLenConfig]] = Field(
        default=None, alias=Fields.FIXED_SEQ_LEN.value)
    agentic_coding: Optional[List[AgenticCodingConfig]] = Field(
        default=None, alias=Fields.AGENTIC_CODING.value)

    @model_validator(mode='after')
    def at_least_one_scenario(self):
        if not self.fixed_seq_len and not self.agentic_coding:
            raise ValueError("At least one scenario type must be specified")
        return self


class MultiNodeScenarios(BaseModel):
    """Scenarios wrapper for multinode configs."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    fixed_seq_len: Optional[List[MultiNodeSeqLenConfig]] = Field(
        default=None, alias=Fields.FIXED_SEQ_LEN.value)
    agentic_coding: Optional[List[AgenticCodingConfig]] = Field(
        default=None, alias=Fields.AGENTIC_CODING.value)

    @model_validator(mode='after')
    def at_least_one_scenario(self):
        if not self.fixed_seq_len and not self.agentic_coding:
            raise ValueError("At least one scenario type must be specified")
        return self


def _validate_component_metadata_scope(self: BaseModel) -> BaseModel:
    """Require unambiguous component metadata across a master config."""
    search_space_entries = [
        entry
        for scenario_configs in (
            self.scenarios.fixed_seq_len,
            self.scenarios.agentic_coding,
        )
        for scenario_config in scenario_configs or []
        for entry in scenario_config.search_space
    ]

    for field in (Fields.ROUTER, Fields.KV_P2P_TRANSFER):
        attribute = field.value.replace("-", "_")
        top_level_value = getattr(self, attribute, None)
        has_search_space_value = any(
            getattr(entry, attribute, None) is not None
            for entry in search_space_entries
        )
        if top_level_value is not None and has_search_space_value:
            raise ValueError(
                f"{field.value} must be declared either at the top level or "
                "in search-space entries, not both"
            )

    has_search_space_transfer = any(
        getattr(entry, "kv_p2p_transfer", None) is not None
        for entry in search_space_entries
    )
    if not self.multinode and has_search_space_transfer:
        raise ValueError(
            f"{Fields.KV_P2P_TRANSFER.value} is only valid when "
            f"{Fields.MULTINODE.value}=true"
        )

    top_level_transfer = getattr(self, "kv_p2p_transfer", None)
    if self.disagg and top_level_transfer is None:
        if not search_space_entries or any(
            entry.kv_p2p_transfer is None for entry in search_space_entries
        ):
            raise ValueError(
                f"{Fields.DISAGG.value}=true requires "
                f"{Fields.KV_P2P_TRANSFER.value} at the top level or in every "
                "search-space entry"
            )

    return self


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
    disagg: Literal[False] = Field(default=False)
    router: Optional[ComponentMetadata] = None
    scenarios: SingleNodeScenarios

    @model_validator(mode='after')
    def validate_agentic_runner(self):
        _validate_agentic_runner_is_cluster(self.runner, self.scenarios)
        return self

    @model_validator(mode='after')
    def validate_component_metadata_scope(self):
        return _validate_component_metadata_scope(self)


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
    router: Optional[ComponentMetadata] = None
    kv_p2p_transfer: Optional[str] = Field(
        default=None, alias=Fields.KV_P2P_TRANSFER.value, min_length=1
    )
    scenarios: MultiNodeScenarios

    @model_validator(mode='after')
    def validate_agentic_runner(self):
        _validate_agentic_runner_is_cluster(self.runner, self.scenarios)
        return self

    @model_validator(mode='after')
    def validate_component_metadata_scope(self):
        return _validate_component_metadata_scope(self)


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


def _validate_runner_labels(labels: dict) -> None:
    for key, value in labels.items():
        if not isinstance(value, list):
            raise ValueError(
                f"Runner config entry '{key}' must be a list, got {type(value).__name__}")

        if not all(isinstance(item, str) for item in value):
            raise ValueError(
                f"Runner config entry '{key}' must contain only strings")

        if not value:
            raise ValueError(
                f"Runner config entry '{key}' cannot be an empty list")


class RunnerHardwareConfig(BaseModel):
    """Per-hardware runner facts used when generating benchmark matrices."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    available_cpu_dram_mib: int = Field(
        alias=Fields.AVAILABLE_CPU_DRAM_MIB.value, gt=0
    )
    gpus_per_node: int = Field(
        alias=Fields.GPUS_PER_NODE.value, gt=0
    )


class RunnerConfig(BaseModel):
    """Top-level runner configuration file."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    labels: Dict[str, List[str]]
    hardware: Dict[str, RunnerHardwareConfig] = Field(default_factory=dict)


def validate_runner_config(runner_configs: dict) -> dict:
    """Validate runner labels and hardware metadata."""
    labels = runner_configs.get("labels")
    if not isinstance(labels, dict):
        raise ValueError("Runner config must define a labels mapping")
    _validate_runner_labels(labels)
    try:
        RunnerConfig(**runner_configs)
    except ValidationError as e:
        raise ValueError(f"Runner config failed validation:\n{e}")
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
    all_evals: bool = Field(alias="all-evals", default=False)
    scenario_type: Optional[List[Literal["fixed-seq-len", "agentic-coding"]]] = Field(
        alias="scenario-type", default=None, min_length=1,
        description="Restrict to specific scenario types (e.g., ['fixed-seq-len', 'agentic-coding'])"
    )


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

    single_node: dict[str, list[Union[SingleNodeMatrixEntry, SingleNodeAgenticMatrixEntry]]
                      ] = Field(default_factory=dict)
    multi_node: dict[str, list[Union[MultiNodeMatrixEntry, MultiNodeAgenticMatrixEntry]]
                     ] = Field(default_factory=dict)
    # Fixed-seq-len evals plus single-node agentic (SWE-bench) eval rows,
    # selected by mark_eval_entries since the SWE-bench Lite eval (PR #1947).
    evals: list[Union[SingleNodeMatrixEntry, SingleNodeAgenticMatrixEntry]] = Field(
        default_factory=list)
    multinode_evals: list[MultiNodeMatrixEntry] = Field(default_factory=list)
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
