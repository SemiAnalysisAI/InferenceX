from pydantic import BaseModel, Field, ValidationError, ConfigDict, model_validator
from typing import List, Optional, Union, Literal
from enum import Enum

import json
import pprint
import re
import warnings
import yaml
from pathlib import Path

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
    EVAL_ONLY = 'eval-only'

    # ISB1 replay fields
    BENCHMARK_TYPE = 'benchmark-type'
    EXPORT_FILE = 'export-file'
    RUNTIME_STACK_ID = 'runtime-stack-id'
    HARDWARE_PROFILE_ID = 'hardware-profile-id'
    CANONICAL_MODEL_ID = 'canonical-model-id'
    REQUEST_MODE = 'request-mode'
    MAX_CONCURRENCY = 'max-concurrency'
    SUPPORT_STATUS = 'support-status'
    MAX_SESSIONS = 'max-sessions'
    MAX_TURNS_PER_SESSION = 'max-turns-per-session'
    MAX_OUTPUT_LEN = 'max-output-len'
    NUM_WARMUP_SESSIONS = 'num-warmup-sessions'
    IGNORE_WAITS = 'ignore-waits'
    IGNORE_EOS = 'ignore-eos'
    REPLAY_CONFIGS = 'replay-configs'
    KV_STRESS_CONFIGS = 'kv-stress-configs'
    OFFLOAD_MODE = 'offload-mode'
    OFFLOAD_MODES = 'offload-modes'
    KV_CACHE_DTYPE = 'kv-cache-dtype'
    DISABLE_PREFIX_CACHING = 'disable-prefix-caching'
    USERS = 'users'
    DURATION_S = 'duration-s'
    WORKLOAD_TYPE = 'workload-type'


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
    eval_only: bool = Field(alias=Fields.EVAL_ONLY.value, default=False)


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


class ISB1ReplayMatrixEntry(BaseModel):
    """Pydantic model for validating ISB1 replay matrix entry structure."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    image: str
    model: str
    model_prefix: str = Field(alias=Fields.MODEL_PREFIX.value)
    precision: str
    framework: str
    runner: str
    benchmark_type: Literal["isb1_replay"] = Field(
        alias=Fields.BENCHMARK_TYPE.value
    )
    export_file: str = Field(alias=Fields.EXPORT_FILE.value)
    runtime_stack_id: str = Field(alias=Fields.RUNTIME_STACK_ID.value)
    hardware_profile_id: str = Field(alias=Fields.HARDWARE_PROFILE_ID.value)
    canonical_model_id: str = Field(alias=Fields.CANONICAL_MODEL_ID.value)
    support_status: Optional[
        Literal["supported", "reviewed_preview", "gated", "artifact_only", "unsupported"]
    ] = Field(default=None, alias=Fields.SUPPORT_STATUS.value)
    request_mode: str = Field(alias=Fields.REQUEST_MODE.value)
    max_concurrency: int = Field(alias=Fields.MAX_CONCURRENCY.value, gt=0)
    max_sessions: Optional[int] = Field(
        default=None, alias=Fields.MAX_SESSIONS.value, gt=0
    )
    max_turns_per_session: Optional[int] = Field(
        default=None, alias=Fields.MAX_TURNS_PER_SESSION.value, gt=0
    )
    max_output_len: Optional[int] = Field(
        default=None, alias=Fields.MAX_OUTPUT_LEN.value, gt=0
    )
    num_warmup_sessions: int = Field(
        default=0, alias=Fields.NUM_WARMUP_SESSIONS.value, ge=0
    )
    ignore_waits: bool = Field(default=False, alias=Fields.IGNORE_WAITS.value)
    ignore_eos: bool = Field(default=False, alias=Fields.IGNORE_EOS.value)
    max_model_len: Optional[int] = Field(
        default=None, alias=Fields.MAX_MODEL_LEN.value, gt=0
    )
    offload_mode: Optional[Literal["on", "off", "noprefix", "legacy"]] = Field(
        default=None, alias=Fields.OFFLOAD_MODE.value
    )
    kv_cache_dtype: Optional[Literal["auto", "fp8"]] = Field(
        default=None, alias=Fields.KV_CACHE_DTYPE.value
    )
    disable_prefix_caching: Optional[bool] = Field(
        default=None, alias=Fields.DISABLE_PREFIX_CACHING.value
    )
    benchmark_duration_s: Optional[int] = Field(
        default=None, alias='benchmark-duration-s', gt=0
    )
    exp_name: str = Field(alias=Fields.EXP_NAME.value)


def validate_isb1_matrix_entry(entry: dict) -> dict:
    """Validate that ISB1 replay matrix entries match the expected structure."""
    try:
        ISB1ReplayMatrixEntry(**entry)
    except ValidationError as e:
        raise ValueError(
            f"The following ISB1 matrix entry failed validation:\n{pprint.pformat(entry)}\n{e}"
        )
    return entry


class ISB1KVStressMatrixEntry(BaseModel):
    """Pydantic model for validating ISB1 KV stress matrix entry structure."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    image: str
    model: str
    model_prefix: str = Field(alias=Fields.MODEL_PREFIX.value)
    precision: str
    framework: str
    runner: str
    benchmark_type: Literal["isb1_kv_stress"] = Field(
        alias=Fields.BENCHMARK_TYPE.value
    )
    export_file: str = Field(alias=Fields.EXPORT_FILE.value)
    runtime_stack_id: str = Field(alias=Fields.RUNTIME_STACK_ID.value)
    hardware_profile_id: str = Field(alias=Fields.HARDWARE_PROFILE_ID.value)
    canonical_model_id: str = Field(alias=Fields.CANONICAL_MODEL_ID.value)
    support_status: Optional[
        Literal["supported", "reviewed_preview", "gated", "artifact_only", "unsupported"]
    ] = Field(default=None, alias=Fields.SUPPORT_STATUS.value)
    request_mode: str = Field(alias=Fields.REQUEST_MODE.value)
    max_concurrency: int = Field(alias=Fields.MAX_CONCURRENCY.value, gt=0)
    offload_mode: Literal["on", "off", "noprefix", "legacy"] = Field(
        alias=Fields.OFFLOAD_MODE.value
    )
    kv_cache_dtype: Literal["auto", "fp8"] = Field(alias=Fields.KV_CACHE_DTYPE.value)
    disable_prefix_caching: bool = Field(alias=Fields.DISABLE_PREFIX_CACHING.value)
    benchmark_duration_s: int = Field(alias='benchmark-duration-s', gt=0)
    workload_type: Literal["chat", "code"] = Field(alias=Fields.WORKLOAD_TYPE.value)
    tp: Optional[int] = Field(default=None, alias=Fields.TP.value, gt=0)
    ep: Optional[int] = Field(default=None, alias=Fields.EP.value, gt=0)
    max_model_len: Optional[int] = Field(
        default=None, alias=Fields.MAX_MODEL_LEN.value, gt=0
    )
    exp_name: str = Field(alias=Fields.EXP_NAME.value)


def validate_isb1_kv_stress_matrix_entry(entry: dict) -> dict:
    """Validate that ISB1 KV stress matrix entries match the expected structure."""
    try:
        ISB1KVStressMatrixEntry(**entry)
    except ValidationError as e:
        raise ValueError(
            f"The following ISB1 KV stress matrix entry failed validation:\n{pprint.pformat(entry)}\n{e}"
        )
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


class ISB1ReplaySearchSpaceEntry(BaseModel):
    """ISB1 replay search space configuration."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    max_concurrency: int = Field(alias=Fields.MAX_CONCURRENCY.value, gt=0)
    max_sessions: Optional[int] = Field(
        default=None, alias=Fields.MAX_SESSIONS.value, gt=0
    )
    max_turns_per_session: Optional[int] = Field(
        default=None, alias=Fields.MAX_TURNS_PER_SESSION.value, gt=0
    )
    max_output_len: Optional[int] = Field(
        default=None, alias=Fields.MAX_OUTPUT_LEN.value, gt=0
    )
    num_warmup_sessions: int = Field(
        default=0, alias=Fields.NUM_WARMUP_SESSIONS.value, ge=0
    )
    ignore_waits: bool = Field(default=False, alias=Fields.IGNORE_WAITS.value)
    ignore_eos: bool = Field(default=False, alias=Fields.IGNORE_EOS.value)
    benchmark_duration_s: Optional[int] = Field(
        default=None, alias='benchmark-duration-s', gt=0
    )


class ISB1ReplayConfigEntry(BaseModel):
    """Per-export replay configuration for ISB1."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    export_file: str = Field(alias=Fields.EXPORT_FILE.value)
    request_mode: str = Field(alias=Fields.REQUEST_MODE.value)
    support_status: Optional[
        Literal["supported", "reviewed_preview", "gated", "artifact_only", "unsupported"]
    ] = Field(default=None, alias=Fields.SUPPORT_STATUS.value)
    search_space: List[ISB1ReplaySearchSpaceEntry] = Field(
        alias=Fields.SEARCH_SPACE.value, min_length=1
    )


class ISB1KVStressSearchSpaceEntry(BaseModel):
    """ISB1 KV stress search space configuration."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    users: List[int] = Field(alias=Fields.USERS.value, min_length=1)
    offload_modes: List[Literal["on", "off", "noprefix", "legacy"]] = Field(
        alias=Fields.OFFLOAD_MODES.value,
        min_length=1,
    )
    duration_s: int = Field(alias=Fields.DURATION_S.value, gt=0)


class ISB1KVStressTPConfig(BaseModel):
    """Per-TP KV stress configuration for ISB1 parity sweeps."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    tp: int = Field(gt=0)
    ep: int = Field(default=1, gt=0)
    users: List[int] = Field(alias=Fields.USERS.value, min_length=1)
    offload_modes: List[Literal["on", "off", "noprefix", "legacy"]] = Field(
        alias=Fields.OFFLOAD_MODES.value,
        min_length=1,
    )
    duration_s: int = Field(alias=Fields.DURATION_S.value, gt=0)


class ISB1KVStressConfigEntry(BaseModel):
    """Per-export KV stress configuration for ISB1."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    export_file: str = Field(alias=Fields.EXPORT_FILE.value)
    request_mode: str = Field(alias=Fields.REQUEST_MODE.value)
    support_status: Optional[
        Literal["supported", "reviewed_preview", "gated", "artifact_only", "unsupported"]
    ] = Field(default=None, alias=Fields.SUPPORT_STATUS.value)
    workload_type: Literal["chat", "code"] = Field(alias=Fields.WORKLOAD_TYPE.value)
    search_space: List[ISB1KVStressSearchSpaceEntry] = Field(
        alias=Fields.SEARCH_SPACE.value, min_length=1
    )
    tp_configs: Optional[List[ISB1KVStressTPConfig]] = Field(
        default=None,
        alias='tp-configs',
    )


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


class ISB1MasterConfigEntry(BaseModel):
    """Top-level ISB1 replay master configuration entry."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    image: str
    model: str
    model_prefix: str = Field(alias=Fields.MODEL_PREFIX.value)
    precision: str
    framework: str
    runner: str
    benchmark_type: Literal["isb1_replay"] = Field(
        alias=Fields.BENCHMARK_TYPE.value
    )
    runtime_stack_id: str = Field(alias=Fields.RUNTIME_STACK_ID.value)
    hardware_profile_id: str = Field(alias=Fields.HARDWARE_PROFILE_ID.value)
    canonical_model_id: str = Field(alias=Fields.CANONICAL_MODEL_ID.value)
    max_model_len: Optional[int] = Field(
        default=None, alias=Fields.MAX_MODEL_LEN.value, gt=0
    )
    offload_mode: Optional[Literal["on", "off", "noprefix", "legacy"]] = Field(
        default=None, alias=Fields.OFFLOAD_MODE.value
    )
    kv_cache_dtype: Optional[Literal["auto", "fp8"]] = Field(
        default=None, alias=Fields.KV_CACHE_DTYPE.value
    )
    disable_prefix_caching: Optional[bool] = Field(
        default=None, alias=Fields.DISABLE_PREFIX_CACHING.value
    )
    replay_configs: List[ISB1ReplayConfigEntry] = Field(
        alias=Fields.REPLAY_CONFIGS.value, min_length=1
    )


class ISB1KVStressMasterConfigEntry(BaseModel):
    """Top-level ISB1 KV stress master configuration entry."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    image: str
    model: str
    model_prefix: str = Field(alias=Fields.MODEL_PREFIX.value)
    precision: str
    framework: str
    runner: str
    benchmark_type: Literal["isb1_kv_stress"] = Field(
        alias=Fields.BENCHMARK_TYPE.value
    )
    runtime_stack_id: str = Field(alias=Fields.RUNTIME_STACK_ID.value)
    hardware_profile_id: str = Field(alias=Fields.HARDWARE_PROFILE_ID.value)
    canonical_model_id: str = Field(alias=Fields.CANONICAL_MODEL_ID.value)
    max_model_len: Optional[int] = Field(
        default=None, alias=Fields.MAX_MODEL_LEN.value, gt=0
    )
    kv_cache_dtype: Literal["auto", "fp8"] = Field(alias=Fields.KV_CACHE_DTYPE.value)
    kv_stress_configs: List[ISB1KVStressConfigEntry] = Field(
        alias=Fields.KV_STRESS_CONFIGS.value,
        min_length=1,
    )


ISB1_SHAPE_STEM_RE = re.compile(r"(?P<isl>\d+)k(?P<osl>\d+)k")
ISB1_RUNNABLE_CERTIFICATION_STATUSES = ["dataset_replay_verified"]


def _candidate_config_roots(config_file: str) -> list[Path]:
    """Return candidate repo roots for resolving relative export-file paths."""
    config_path = Path(config_file).resolve()
    parent_candidates = [config_path.parents[i] for i in range(min(3, len(config_path.parents)))]
    candidates = [
        config_path.parent,
        *parent_candidates,
        Path.cwd().resolve(),
    ]

    unique_candidates: list[Path] = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


def _resolve_export_path(config_file: str, export_file: str) -> Path:
    """Resolve an export file relative to the config file or current repo root."""
    export_path = Path(export_file)
    if export_path.is_absolute():
        return export_path

    candidate_roots = _candidate_config_roots(config_file)
    for candidate_root in candidate_roots:
        candidate = candidate_root / export_path
        if candidate.exists():
            return candidate

    return candidate_roots[0] / export_path


def _load_export_payload(export_path: Path) -> dict:
    """Load an ISB1 export payload from disk."""
    try:
        with export_path.open("r") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(f"Referenced ISB1 export file does not exist: '{export_path}'.") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Referenced ISB1 export file is not valid JSON: '{export_path}'.") from exc

    exports = payload.get("exports")
    if not isinstance(exports, list) or not exports:
        raise ValueError(
            f"Referenced ISB1 export file must contain a non-empty 'exports' list: '{export_path}'."
        )
    return payload


def _identity_cells(payload: dict, entry: dict) -> list[dict]:
    """Return export cells matching the configured runtime/hardware/model identity."""
    return [
        cell
        for cell in payload["exports"]
        if cell.get("runtime_stack_id") == entry[Fields.RUNTIME_STACK_ID.value]
        and cell.get("hardware_profile_id") == entry[Fields.HARDWARE_PROFILE_ID.value]
        and cell.get("canonical_model_id") == entry[Fields.CANONICAL_MODEL_ID.value]
    ]


def _warn_manifest_max_model_len_mismatch(
    *,
    export_path: Path,
    export_file: str,
    max_model_len: Optional[int],
    key: str,
) -> None:
    """Emit advisory warning if sibling manifest max_model_len disagrees with config."""
    if max_model_len is None:
        return

    for manifest_path in sorted(export_path.parent.glob("manifest*.json")):
        try:
            manifest_payload = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        manifest_exports = manifest_payload.get("exports")
        if isinstance(manifest_exports, list):
            export_files = {
                item.get("export_file")
                for item in manifest_exports
                if isinstance(item, dict) and isinstance(item.get("export_file"), str)
            }
            if export_files and export_file not in export_files:
                continue

        manifest_max_model_len = manifest_payload.get("max_model_len")
        if manifest_max_model_len is None:
            continue

        try:
            manifest_max_model_len = int(manifest_max_model_len)
        except (TypeError, ValueError):
            continue

        if manifest_max_model_len != max_model_len:
            warnings.warn(
                f"ISB1 master config entry '{key}' sets '{Fields.MAX_MODEL_LEN.value}'="
                f"{max_model_len} for export '{export_file}', but sibling manifest "
                f"'{manifest_path}' declares max_model_len={manifest_max_model_len}.",
                stacklevel=2,
            )


def certify_isb1_replay_contract(master_configs: dict, config_file: str) -> dict:
    """Validate that every replay-config resolves to a real, runnable export selection."""
    for key, entry in master_configs.items():
        max_model_len = entry.get(Fields.MAX_MODEL_LEN.value)

        for replay_config in entry[Fields.REPLAY_CONFIGS.value]:
            export_file = replay_config[Fields.EXPORT_FILE.value]
            support_status = replay_config.get(Fields.SUPPORT_STATUS.value)
            export_path = _resolve_export_path(config_file, export_file)
            payload = _load_export_payload(export_path)
            _warn_manifest_max_model_len_mismatch(
                export_path=export_path,
                export_file=export_file,
                max_model_len=max_model_len,
                key=key,
            )

            if not ISB1_SHAPE_STEM_RE.search(export_path.stem) and max_model_len is None:
                raise ValueError(
                    f"ISB1 master config entry '{key}' references mixed-shape export "
                    f"'{export_file}' without '{Fields.MAX_MODEL_LEN.value}'."
                )

            identity_cells = _identity_cells(payload, entry)
            identity_statuses = sorted(
                {
                    cell.get("support_status")
                    for cell in identity_cells
                    if cell.get("support_status") is not None
                }
            )
            matching_cells = [
                cell
                for cell in identity_cells
                if support_status is None or cell.get("support_status") == support_status
            ]

            if support_status is None and len(identity_statuses) > 1:
                raise ValueError(
                    f"ISB1 master config entry '{key}' must pin "
                    f"'{Fields.SUPPORT_STATUS.value}' for export '{export_file}'. "
                    f"Matching cells span multiple tiers: {identity_statuses}."
                )

            if not matching_cells:
                available_statuses = identity_statuses or ["<none>"]
                raise ValueError(
                    f"ISB1 master config entry '{key}' requests export '{export_file}' "
                    f"with support-status '{support_status}', but no export cell matches "
                    f"runtime_stack_id='{entry[Fields.RUNTIME_STACK_ID.value]}', "
                    f"hardware_profile_id='{entry[Fields.HARDWARE_PROFILE_ID.value]}', "
                    f"canonical_model_id='{entry[Fields.CANONICAL_MODEL_ID.value]}'. "
                    f"Available support tiers for that identity: {available_statuses}."
                )

            certification_statuses = sorted(
                {
                    cell.get("benchmark_certification_status")
                    for cell in matching_cells
                    if cell.get("benchmark_certification_status") is not None
                }
            )
            if not certification_statuses:
                raise ValueError(
                    f"ISB1 master config entry '{key}' requests export '{export_file}' "
                    "but the selected export cells do not declare "
                    "'benchmark_certification_status'."
                )
            if certification_statuses != ISB1_RUNNABLE_CERTIFICATION_STATUSES:
                raise ValueError(
                    f"ISB1 master config entry '{key}' requests export '{export_file}' "
                    "with runnable support tier selection, but the selected export cells "
                    f"have benchmark_certification_status values {certification_statuses}. "
                    "Current InferenceX consumer lanes only accept "
                    f"{ISB1_RUNNABLE_CERTIFICATION_STATUSES}."
                )

    return master_configs


def certify_isb1_kv_stress_contract(master_configs: dict, config_file: str) -> dict:
    """Validate that every kv-stress-config resolves to a real, runnable export selection."""
    for key, entry in master_configs.items():
        max_model_len = entry.get(Fields.MAX_MODEL_LEN.value)

        for kv_stress_config in entry[Fields.KV_STRESS_CONFIGS.value]:
            export_file = kv_stress_config[Fields.EXPORT_FILE.value]
            support_status = kv_stress_config.get(Fields.SUPPORT_STATUS.value)
            export_path = _resolve_export_path(config_file, export_file)
            payload = _load_export_payload(export_path)
            _warn_manifest_max_model_len_mismatch(
                export_path=export_path,
                export_file=export_file,
                max_model_len=max_model_len,
                key=key,
            )

            if not ISB1_SHAPE_STEM_RE.search(export_path.stem) and max_model_len is None:
                raise ValueError(
                    f"ISB1 KV stress config entry '{key}' references mixed-shape export "
                    f"'{export_file}' without '{Fields.MAX_MODEL_LEN.value}'."
                )

            identity_cells = _identity_cells(payload, entry)
            identity_statuses = sorted(
                {
                    cell.get("support_status")
                    for cell in identity_cells
                    if cell.get("support_status") is not None
                }
            )
            matching_cells = [
                cell
                for cell in identity_cells
                if support_status is None or cell.get("support_status") == support_status
            ]

            if support_status is None and len(identity_statuses) > 1:
                raise ValueError(
                    f"ISB1 KV stress config entry '{key}' must pin "
                    f"'{Fields.SUPPORT_STATUS.value}' for export '{export_file}'. "
                    f"Matching cells span multiple tiers: {identity_statuses}."
                )

            if not matching_cells:
                available_statuses = identity_statuses or ["<none>"]
                raise ValueError(
                    f"ISB1 KV stress config entry '{key}' requests export '{export_file}' "
                    f"with support-status '{support_status}', but no export cell matches "
                    f"runtime_stack_id='{entry[Fields.RUNTIME_STACK_ID.value]}', "
                    f"hardware_profile_id='{entry[Fields.HARDWARE_PROFILE_ID.value]}', "
                    f"canonical_model_id='{entry[Fields.CANONICAL_MODEL_ID.value]}'. "
                    f"Available support tiers for that identity: {available_statuses}."
                )

            certification_statuses = sorted(
                {
                    cell.get("benchmark_certification_status")
                    for cell in matching_cells
                    if cell.get("benchmark_certification_status") is not None
                }
            )
            if not certification_statuses:
                raise ValueError(
                    f"ISB1 KV stress config entry '{key}' requests export '{export_file}' "
                    "but the selected export cells do not declare "
                    "'benchmark_certification_status'."
                )
            if certification_statuses != ISB1_RUNNABLE_CERTIFICATION_STATUSES:
                raise ValueError(
                    f"ISB1 KV stress config entry '{key}' requests export '{export_file}' "
                    "with runnable support tier selection, but the selected export cells "
                    f"have benchmark_certification_status values {certification_statuses}. "
                    "Current InferenceX consumer lanes only accept "
                    f"{ISB1_RUNNABLE_CERTIFICATION_STATUSES}."
                )

    return master_configs


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


def validate_isb1_master_config(master_configs: dict) -> List[dict]:
    """Validate ISB1 replay master configuration structure."""
    for key, entry in master_configs.items():
        try:
            ISB1MasterConfigEntry(**entry)
        except ValidationError as e:
            raise ValueError(
                f"ISB1 master config entry '{key}' failed validation:\n{e}"
            )
    return master_configs


def validate_isb1_kv_stress_master_config(master_configs: dict) -> List[dict]:
    """Validate ISB1 KV stress master configuration structure."""
    for key, entry in master_configs.items():
        try:
            ISB1KVStressMasterConfigEntry(**entry)
        except ValidationError as e:
            raise ValueError(
                f"ISB1 KV stress master config entry '{key}' failed validation:\n{e}"
            )
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
    evals: list[SingleNodeMatrixEntry] = Field(default_factory=list)
    changelog_metadata: ChangelogMetadata


# =============================================================================
# File Loading Functions
# =============================================================================


def _load_and_merge_yaml_files(config_files: List[str]) -> dict:
    """Load and merge YAML configuration files."""
    all_config_data = {}
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
                if not isinstance(config_data, dict):
                    raise ValueError(
                        f"Config file '{config_file}' must contain a dictionary"
                    )

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

    return all_config_data


def load_config_files(config_files: List[str], validate: bool = True) -> dict:
    """Load and merge throughput configuration files.

    Args:
        config_files: List of paths to YAML configuration files.
        validate: If True, run validate_master_config on loaded data. Defaults to True.

    Returns:
        Merged configuration dictionary.

    Raises:
        ValueError: If file doesn't exist, isn't a dict, or has duplicate keys.
    """
    all_config_data = _load_and_merge_yaml_files(config_files)

    if validate:
        validate_master_config(all_config_data)

    return all_config_data


def load_isb1_config_files(config_files: List[str], validate: bool = True) -> dict:
    """Load and merge ISB1 replay configuration files."""
    all_config_data = _load_and_merge_yaml_files(config_files)

    if validate:
        validate_isb1_master_config(all_config_data)
        for config_file in config_files:
            certify_isb1_replay_contract(
                _load_and_merge_yaml_files([config_file]),
                config_file=config_file,
            )

    return all_config_data


def load_isb1_kv_stress_config_files(config_files: List[str], validate: bool = True) -> dict:
    """Load and merge ISB1 KV stress configuration files."""
    all_config_data = _load_and_merge_yaml_files(config_files)

    if validate:
        validate_isb1_kv_stress_master_config(all_config_data)
        for config_file in config_files:
            certify_isb1_kv_stress_contract(
                _load_and_merge_yaml_files([config_file]),
                config_file=config_file,
            )

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
