import pprint
import re
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from infx import yamlio
from infx.clusters import CLUSTER_LABEL_PREFIX, RunnerInventory

DEFAULT_AGENTIC_DURATION_SECONDS = 3600
SINGLE_NODE_RECIPES = "benchmarks/single_node/srt-slurm-recipes"
MULTI_NODE_RECIPES = "benchmarks/multi_node/srt-slurm-recipes"
RECIPE_SELECTOR = re.compile(r"base|override_[\w-]+|zip_override_[\w-]+\[\d+\]")


class Fields(Enum):
    IMAGE = "image"
    MODEL = "model"
    MODEL_PREFIX = "model-prefix"
    PRECISION = "precision"
    FRAMEWORK = "framework"
    RUNNER = "runner"
    HARDWARE = "hardware"
    SCENARIOS = "scenarios"
    MULTINODE = "multinode"

    FIXED_SEQ_LEN = "fixed-seq-len"
    AGENTIC_CODING = "agentic-coding"

    ISL = "isl"
    OSL = "osl"
    SEARCH_SPACE = "search-space"

    TP = "tp"
    SRT_RECIPE = "srt-recipe"
    SRT_RECIPE_DIR = "srt-recipe-dir"
    EVAL_SRT_RECIPE = "eval-srt-recipe"
    PP = "pp"
    DCP_SIZE = "dcp-size"
    PCP_SIZE = "pcp-size"
    DWDP_SIZE = "dwdp-size"
    CONC_START = "conc-start"
    CONC_END = "conc-end"
    CONC_LIST = "conc-list"
    EP = "ep"
    DP_ATTN = "dp-attn"

    SPEC_DECODING = "spec-decoding"
    WORKER = "worker"
    PREFILL = "prefill"
    DECODE = "decode"
    NUM_WORKER = "num-worker"
    BATCH_SIZE = "batch-size"
    MAX_NUM_TOKENS = "max-num-tokens"
    ADDITIONAL_SETTINGS = "additional-settings"

    KV_OFFLOADING = "kv-offloading"
    KV_OFFLOAD_BACKEND = "kv-offload-backend"
    ROUTER = "router"
    KV_P2P_TRANSFER = "kv-p2p-transfer"
    TOTAL_CPU_DRAM_GB = "total-cpu-dram-gb"
    DECODE_CPU_DRAM_GB = "decode-cpu-dram-gb"
    AVAILABLE_CPU_DRAM_MIB = "available-cpu-dram-mib"
    DRAM_UTILIZATION = "dram-utilization"
    GPUS_PER_NODE = "gpus-per-node"
    NUM_NODES = "num-nodes"
    NODE_COUNT = "node-count"
    DURATION = "duration"
    REQUIRE_POWER = "require-power"
    POWER = "power"

    CONC = "conc"
    MAX_MODEL_LEN = "max-model-len"
    EXP_NAME = "exp-name"
    RECIPE_FINGERPRINT = "recipe-fingerprint"
    DISAGG = "disagg"
    SCENARIO_TYPE = "scenario-type"

    RUN_EVAL = "run-eval"
    EVAL_ONLY = "eval-only"
    EVAL_CONC = "eval-conc"
    EVAL_ALL_CONCS = "eval-all-concs"
    EVAL_FRAMEWORK = "eval-framework"
    EVAL_SUITE = "eval-suite"


class ComponentMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)

    @field_validator("version")
    @classmethod
    def validate_component_version(cls, version: str) -> str:
        """Return ``version``, rejecting an ``image:`` reference with ValueError."""
        if version.startswith("image:"):
            raise ValueError(
                "component version must be a release, package version, or "
                "source commit, not an image reference"
            )
        return version


class KVOffloadBackendMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    version: str | None = Field(default=None, min_length=1)

    @field_validator("version")
    @classmethod
    def validate_component_version(cls, version: str | None) -> str | None:
        """Return ``version``, rejecting an ``image:`` reference with ValueError."""
        if version is not None and version.startswith("image:"):
            raise ValueError(
                "component version must be a release, package version, or "
                "source commit, not an image reference"
            )
        return version


def _validate_tp_context_topology(self: Any) -> Any:
    """Return the model, raising ValueError unless its TP is divisible by its DCP size."""
    if self.tp % self.dcp_size != 0:
        raise ValueError(
            f"'{Fields.TP.value}' ({self.tp}) must be divisible by "
            f"'{Fields.DCP_SIZE.value}' ({self.dcp_size})"
        )
    return self


class SingleNodeMatrixEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    image: str
    srt_recipe: str | None = Field(default=None, alias=Fields.SRT_RECIPE.value, min_length=1)
    model: str
    model_prefix: str = Field(alias=Fields.MODEL_PREFIX.value)
    precision: str
    framework: str
    spec_decoding: Literal["mtp", "draft_model", "none"] = Field(alias=Fields.SPEC_DECODING.value)
    runner: str
    isl: int
    osl: int
    tp: int
    pp: int = Field(gt=0, strict=True)
    dcp_size: int = Field(alias=Fields.DCP_SIZE.value, gt=0, strict=True)
    pcp_size: int = Field(alias=Fields.PCP_SIZE.value, gt=0, strict=True)
    ep: int
    dp_attn: bool = Field(alias=Fields.DP_ATTN.value)
    conc: int | list[int]
    max_model_len: int = Field(alias=Fields.MAX_MODEL_LEN.value)
    exp_name: str = Field(alias=Fields.EXP_NAME.value)
    disagg: Literal[False]
    run_eval: bool = Field(alias=Fields.RUN_EVAL.value)
    eval_only: bool = Field(alias=Fields.EVAL_ONLY.value, default=False)
    eval_framework: str | None = Field(default=None, alias=Fields.EVAL_FRAMEWORK.value)
    eval_suite: str | None = Field(default=None, alias=Fields.EVAL_SUITE.value)
    router: ComponentMetadata | None = None
    recipe_fingerprint: str | None = Field(
        default=None,
        alias=Fields.RECIPE_FINGERPRINT.value,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_single_node_topology(self) -> Self:
        """Return the entry, raising ValueError unless TP is divisible by DCP size."""
        return _validate_tp_context_topology(self)


class WorkerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    num_worker: int = Field(alias=Fields.NUM_WORKER.value)
    tp: int = Field(gt=0)
    pp: int = Field(default=1, gt=0, strict=True)
    dcp_size: int = Field(default=1, alias=Fields.DCP_SIZE.value, gt=0, strict=True)
    pcp_size: int = Field(default=1, alias=Fields.PCP_SIZE.value, gt=0, strict=True)
    dwdp_size: int = Field(default=1, alias=Fields.DWDP_SIZE.value, gt=0, strict=True)
    ep: int
    dp_attn: bool = Field(alias=Fields.DP_ATTN.value)
    hardware: str | None = Field(default=None, min_length=1)
    additional_settings: list[str] | None = Field(
        default_factory=list, alias=Fields.ADDITIONAL_SETTINGS.value
    )

    @model_validator(mode="after")
    def validate_worker_topology(self) -> Self:
        """Return the worker, raising ValueError for an invalid DWDP or TP and DCP topology."""
        if self.dwdp_size > 1 and (
            self.num_worker <= 0
            or (self.tp, self.ep, self.pp, self.dcp_size, self.pcp_size) != (1, 1, 1, 1, 1)
            or self.dp_attn
        ):
            raise ValueError(
                "DWDP requires active workers with TP=EP=PP=DCP=PCP=1 and dp-attn=false"
            )
        return _validate_tp_context_topology(self)


class AggregateWorkerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    num_worker: int = Field(default=1, alias=Fields.NUM_WORKER.value, gt=0, strict=True)
    tp: int = Field(gt=0)
    pp: int = Field(default=1, gt=0, strict=True)
    dcp_size: int = Field(default=1, alias=Fields.DCP_SIZE.value, gt=0, strict=True)
    pcp_size: int = Field(default=1, alias=Fields.PCP_SIZE.value, gt=0, strict=True)
    ep: int
    dp_attn: bool = Field(alias=Fields.DP_ATTN.value)
    hardware: str | None = Field(default=None, min_length=1)
    additional_settings: list[str] | None = Field(
        default_factory=list, alias=Fields.ADDITIONAL_SETTINGS.value
    )

    @model_validator(mode="after")
    def validate_worker_topology(self) -> Self:
        """Return the worker, raising ValueError unless TP is divisible by DCP size."""
        return _validate_tp_context_topology(self)


def _validate_worker_hardware_pair(self: Any) -> Any:
    """Return the model, raising ValueError for DWDP on decode or hardware set on one side only."""
    if self.decode.dwdp_size > 1:
        raise ValueError("DWDP is supported only on prefill workers")
    if bool(self.prefill.hardware) != bool(self.decode.hardware):
        raise ValueError(
            f"'{Fields.HARDWARE.value}' must be specified for both "
            f"'{Fields.PREFILL.value}' and '{Fields.DECODE.value}', or neither"
        )
    return self


class MultiNodeMatrixEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    image: str
    model: str
    model_prefix: str = Field(alias=Fields.MODEL_PREFIX.value)
    precision: str
    framework: str
    spec_decoding: Literal["mtp", "draft_model", "none"] = Field(alias=Fields.SPEC_DECODING.value)
    runner: str
    srt_recipe: str | None = Field(default=None, alias=Fields.SRT_RECIPE.value, min_length=1)
    eval_srt_recipe: str | None = Field(
        default=None, alias=Fields.EVAL_SRT_RECIPE.value, min_length=1
    )
    node_count: int = Field(alias=Fields.NODE_COUNT.value, gt=0, strict=True)
    isl: int
    osl: int
    require_power: bool = Field(default=False, alias=Fields.REQUIRE_POWER.value, strict=True)
    power: bool = Field(default=False, strict=True)
    prefill: WorkerConfig
    decode: WorkerConfig
    conc: list[int]
    max_model_len: int = Field(alias=Fields.MAX_MODEL_LEN.value)
    exp_name: str = Field(alias=Fields.EXP_NAME.value)
    disagg: bool
    run_eval: bool = Field(alias=Fields.RUN_EVAL.value)
    eval_only: bool = Field(alias=Fields.EVAL_ONLY.value, default=False)
    eval_conc: int | None = Field(default=None, alias=Fields.EVAL_CONC.value)
    eval_all_concs: bool = Field(default=False, alias=Fields.EVAL_ALL_CONCS.value)
    eval_framework: str | None = Field(default=None, alias=Fields.EVAL_FRAMEWORK.value)
    eval_suite: str | None = Field(default=None, alias=Fields.EVAL_SUITE.value)
    router: ComponentMetadata | None = None
    kv_p2p_transfer: str | None = Field(
        default=None, alias=Fields.KV_P2P_TRANSFER.value, min_length=1
    )
    recipe_fingerprint: str | None = Field(
        default=None,
        alias=Fields.RECIPE_FINGERPRINT.value,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_worker_hardware_pair(self) -> Self:
        """Return the entry, raising ValueError unless its workers' hardware fields agree."""
        return _validate_worker_hardware_pair(self)

    @model_validator(mode="after")
    def validate_disagg_transfer(self) -> Self:
        """Return the entry, raising ValueError if it is disaggregated without a KV transfer."""
        if self.disagg and self.kv_p2p_transfer is None:
            raise ValueError(f"{Fields.DISAGG.value}=true requires {Fields.KV_P2P_TRANSFER.value}")
        return self


class SingleNodeAgenticMatrixEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    image: str
    srt_recipe: str | None = Field(default=None, alias=Fields.SRT_RECIPE.value, min_length=1)
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
    kv_offloading: Literal["none", "dram"] = Field(alias=Fields.KV_OFFLOADING.value)
    kv_offload_backend: KVOffloadBackendMetadata | None = Field(
        default=None, alias=Fields.KV_OFFLOAD_BACKEND.value
    )
    router: ComponentMetadata | None = None
    total_cpu_dram_gb: int = Field(alias=Fields.TOTAL_CPU_DRAM_GB.value, ge=0)
    duration: int = Field(alias=Fields.DURATION.value)
    exp_name: str = Field(alias=Fields.EXP_NAME.value)
    scenario_type: str = Field(alias=Fields.SCENARIO_TYPE.value)
    run_eval: bool | None = Field(default=None, alias=Fields.RUN_EVAL.value)
    eval_only: bool | None = Field(default=None, alias=Fields.EVAL_ONLY.value)
    eval_framework: str | None = Field(default=None, alias=Fields.EVAL_FRAMEWORK.value)
    eval_suite: str | None = Field(default=None, alias=Fields.EVAL_SUITE.value)
    recipe_fingerprint: str | None = Field(
        default=None,
        alias=Fields.RECIPE_FINGERPRINT.value,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_kv_offload_fields(self) -> Self:
        """Return the entry, raising ValueError for inconsistent KV offload fields."""
        return _validate_kv_offload_fields(self)

    @model_validator(mode="after")
    def validate_single_node_topology(self) -> Self:
        """Return the entry, raising ValueError unless TP is divisible by DCP size."""
        return _validate_tp_context_topology(self)


class MultiNodeAgenticMatrixEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    image: str
    model: str
    model_prefix: str = Field(alias=Fields.MODEL_PREFIX.value)
    precision: str
    framework: str
    spec_decoding: Literal["mtp", "draft_model", "none"] = Field(alias=Fields.SPEC_DECODING.value)
    runner: str
    srt_recipe: str | None = Field(default=None, alias=Fields.SRT_RECIPE.value, min_length=1)
    eval_srt_recipe: str | None = Field(
        default=None, alias=Fields.EVAL_SRT_RECIPE.value, min_length=1
    )
    node_count: int = Field(alias=Fields.NODE_COUNT.value, gt=0, strict=True)
    power: bool = Field(default=False, strict=True)
    prefill: WorkerConfig
    decode: WorkerConfig
    conc: list[int]
    kv_offloading: Literal["none", "dram"] = Field(alias=Fields.KV_OFFLOADING.value)
    kv_offload_backend: KVOffloadBackendMetadata | None = Field(
        default=None, alias=Fields.KV_OFFLOAD_BACKEND.value
    )
    router: ComponentMetadata | None = None
    kv_p2p_transfer: str | None = Field(
        default=None, alias=Fields.KV_P2P_TRANSFER.value, min_length=1
    )
    total_cpu_dram_gb: int = Field(alias=Fields.TOTAL_CPU_DRAM_GB.value, ge=0)
    decode_cpu_dram_gb: int = Field(default=0, alias=Fields.DECODE_CPU_DRAM_GB.value, ge=0)
    duration: int = Field(alias=Fields.DURATION.value)
    exp_name: str = Field(alias=Fields.EXP_NAME.value)
    disagg: bool
    scenario_type: str = Field(alias=Fields.SCENARIO_TYPE.value)
    run_eval: bool | None = Field(default=None, alias=Fields.RUN_EVAL.value)
    eval_only: bool | None = Field(default=None, alias=Fields.EVAL_ONLY.value)
    eval_conc: int | None = Field(default=None, alias=Fields.EVAL_CONC.value)
    eval_framework: str | None = Field(default=None, alias=Fields.EVAL_FRAMEWORK.value)
    eval_suite: str | None = Field(default=None, alias=Fields.EVAL_SUITE.value)
    recipe_fingerprint: str | None = Field(
        default=None,
        alias=Fields.RECIPE_FINGERPRINT.value,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_worker_hardware_pair(self) -> Self:
        """Return the entry, raising ValueError unless its workers' hardware fields agree."""
        return _validate_worker_hardware_pair(self)

    @model_validator(mode="after")
    def validate_kv_offload_fields(self) -> Self:
        """Return the entry, raising ValueError for inconsistent KV offload fields."""
        return _validate_kv_offload_fields(self)

    @model_validator(mode="after")
    def validate_disagg_transfer(self) -> Self:
        """Return the entry, raising ValueError if it is disaggregated without a KV transfer."""
        if self.disagg and self.kv_p2p_transfer is None:
            raise ValueError(f"{Fields.DISAGG.value}=true requires {Fields.KV_P2P_TRANSFER.value}")
        return self


AgenticMatrixEntry = SingleNodeAgenticMatrixEntry | MultiNodeAgenticMatrixEntry


def validate_agentic_matrix_entry(entry: dict) -> dict:
    """Return an agentic matrix row unchanged, raising ValueError with the row if it is invalid."""
    try:
        if Fields.PREFILL.value in entry:
            MultiNodeAgenticMatrixEntry(**entry)
        else:
            SingleNodeAgenticMatrixEntry(**entry)
    except ValidationError as e:
        raise ValueError(
            f"The following parsed agentic matrix entry failed validation:\n{pprint.pformat(entry)}\n{e}"
        ) from e
    return entry


def validate_matrix_entry(entry: dict, is_multinode: bool) -> dict:
    """Return a fixed-sequence matrix row unchanged, raising ValueError with the row if invalid."""
    try:
        if is_multinode:
            MultiNodeMatrixEntry(**entry)
        else:
            SingleNodeMatrixEntry(**entry)
    except ValidationError as e:
        raise ValueError(
            f"The following parsed matrix entry failed validation:\n{pprint.pformat(entry)}\n{e}"
        ) from e
    return entry


def _validate_conc_fields(self: Any) -> Any:
    """Return the model, raising ValueError unless it has exactly one valid conc range or list."""
    has_range = self.conc_start is not None and self.conc_end is not None
    has_list = self.conc_list is not None and len(self.conc_list) > 0

    if has_list and (self.conc_start is not None or self.conc_end is not None):
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

    if has_list and not all(x > 0 for x in self.conc_list):
        raise ValueError(f"Input '{Fields.CONC_LIST.value}' entries must be greater than 0.")

    if has_list and len(set(self.conc_list)) != len(self.conc_list):
        raise ValueError(f"Input '{Fields.CONC_LIST.value}' entries must not repeat.")

    return self


def _validate_agentic_runner_is_cluster(runner: str, scenarios: Any) -> None:
    """Raise ValueError if an agentic config's ``runner`` is not a cluster label."""
    if scenarios.agentic_coding and not runner.startswith(CLUSTER_LABEL_PREFIX):
        raise ValueError(
            f"Agentic master configs must use a '{CLUSTER_LABEL_PREFIX}<name>' runner "
            "so every point runs on one exact hardware fleet."
        )


def _validate_kv_offload_fields(self: Any) -> Any:
    """Return the model, raising ValueError unless a KV offload backend comes with offloading."""
    backend = getattr(self, "kv_offload_backend", None)
    if self.kv_offloading is None:
        if backend is not None:
            raise ValueError(
                f"{Fields.KV_OFFLOAD_BACKEND.value} requires {Fields.KV_OFFLOADING.value}"
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


def _recipe_path(value: str) -> str:
    """Return ``value``, raising ValueError unless it is a normalized relative POSIX path."""
    path = PurePosixPath(value)
    if (
        not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or ":" in value
        or path.as_posix() != value
    ):
        raise ValueError(f"{value!r} must be a normalized relative path")
    return value


def _recipe_reference(value: str) -> str:
    """Return a ``path[:selector]`` recipe reference, raising ValueError for a bad part."""
    path, separator, selector = value.partition(":")
    _recipe_path(path)
    if separator and not RECIPE_SELECTOR.fullmatch(selector):
        raise ValueError(
            f"{value!r} must select base, override_<name> or zip_override_<name>[<index>]"
        )
    return value


RecipeDir = Annotated[str, AfterValidator(_recipe_path)]
RecipeReference = Annotated[str, AfterValidator(_recipe_reference)]


class SingleNodeSearchSpaceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    tp: int = Field(gt=0)
    srt_recipe: RecipeReference | None = Field(default=None, alias=Fields.SRT_RECIPE.value)
    pp: int = Field(default=1, gt=0, strict=True)
    dcp_size: int = Field(default=1, alias=Fields.DCP_SIZE.value, gt=0, strict=True)
    pcp_size: int = Field(default=1, alias=Fields.PCP_SIZE.value, gt=0, strict=True)
    ep: int | None = None
    spec_decoding: Literal["mtp", "draft_model", "none"] = Field(
        default="none", alias=Fields.SPEC_DECODING.value
    )
    dp_attn: bool | None = Field(default=None, alias=Fields.DP_ATTN.value)
    router: ComponentMetadata | None = None
    conc_start: int | None = Field(default=None, alias=Fields.CONC_START.value)
    conc_end: int | None = Field(default=None, alias=Fields.CONC_END.value)
    conc_list: list[int] | None = Field(default=None, alias=Fields.CONC_LIST.value)

    @model_validator(mode="after")
    def validate_conc_fields(self) -> Self:
        """Return the entry, raising ValueError for an invalid conc range or list."""
        return _validate_conc_fields(self)

    @model_validator(mode="after")
    def validate_single_node_topology(self) -> Self:
        """Return the entry, raising ValueError unless TP is divisible by DCP size."""
        return _validate_tp_context_topology(self)


class MultiNodeSearchSpaceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    spec_decoding: Literal["mtp", "draft_model", "none"] = Field(
        default="none", alias=Fields.SPEC_DECODING.value
    )
    worker: AggregateWorkerConfig | None = None
    prefill: WorkerConfig | None = None
    decode: WorkerConfig | None = None
    srt_recipe: RecipeReference | None = Field(default=None, alias=Fields.SRT_RECIPE.value)
    eval_srt_recipe: RecipeReference | None = Field(
        default=None, alias=Fields.EVAL_SRT_RECIPE.value
    )
    power: bool = Field(default=False, strict=True)
    num_nodes: int | None = Field(default=None, alias=Fields.NUM_NODES.value, gt=0, strict=True)
    router: ComponentMetadata | None = None
    kv_p2p_transfer: str | None = Field(
        default=None, alias=Fields.KV_P2P_TRANSFER.value, min_length=1
    )
    conc_start: int | None = Field(default=None, alias=Fields.CONC_START.value)
    conc_end: int | None = Field(default=None, alias=Fields.CONC_END.value)
    conc_list: list[int] | None = Field(default=None, alias=Fields.CONC_LIST.value)

    @model_validator(mode="after")
    def validate_conc_fields(self) -> Self:
        """Return the entry, raising ValueError for an invalid conc range or list."""
        return _validate_conc_fields(self)

    @model_validator(mode="after")
    def validate_worker_hardware_pair(self) -> Self:
        """Return the entry, raising ValueError unless it has one worker or a matching pair."""
        has_worker = self.worker is not None
        has_any_disagg_worker = self.prefill is not None or self.decode is not None
        has_complete_disagg_workers = self.prefill is not None and self.decode is not None
        if has_worker == has_any_disagg_worker or (
            has_any_disagg_worker and not has_complete_disagg_workers
        ):
            raise ValueError(
                "Multinode search-space entries must specify either worker "
                "or both prefill and decode"
            )
        if has_complete_disagg_workers:
            _validate_worker_hardware_pair(self)
        return self


class SingleNodeSeqLenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    isl: int
    osl: int
    search_space: list[SingleNodeSearchSpaceEntry] = Field(alias=Fields.SEARCH_SPACE.value)


class MultiNodeSeqLenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    isl: int
    osl: int
    require_power: bool = Field(default=False, alias=Fields.REQUIRE_POWER.value, strict=True)
    search_space: list[MultiNodeSearchSpaceEntry] = Field(alias=Fields.SEARCH_SPACE.value)


class AgenticCodingSearchSpaceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    tp: int | None = Field(default=None, gt=0)
    srt_recipe: RecipeReference | None = Field(default=None, alias=Fields.SRT_RECIPE.value)
    eval_srt_recipe: RecipeReference | None = Field(
        default=None, alias=Fields.EVAL_SRT_RECIPE.value
    )
    power: bool = Field(default=False, strict=True)
    pp: int = Field(default=1, gt=0, strict=True)
    dcp_size: int = Field(default=1, alias=Fields.DCP_SIZE.value, gt=0, strict=True)
    pcp_size: int = Field(default=1, alias=Fields.PCP_SIZE.value, gt=0, strict=True)
    ep: int | None = None
    dp_attn: bool | None = Field(default=None, alias=Fields.DP_ATTN.value)
    spec_decoding: Literal["mtp", "draft_model", "none"] = Field(
        default="none", alias=Fields.SPEC_DECODING.value
    )
    worker: AggregateWorkerConfig | None = None
    prefill: WorkerConfig | None = None
    decode: WorkerConfig | None = None
    num_nodes: int | None = Field(default=None, alias=Fields.NUM_NODES.value, gt=0, strict=True)
    kv_offloading: Literal["none", "dram"] | None = Field(
        default=None, alias=Fields.KV_OFFLOADING.value
    )
    kv_offload_backend: KVOffloadBackendMetadata | None = Field(
        default=None, alias=Fields.KV_OFFLOAD_BACKEND.value
    )
    router: ComponentMetadata | None = None
    kv_p2p_transfer: str | None = Field(
        default=None, alias=Fields.KV_P2P_TRANSFER.value, min_length=1
    )
    conc_start: int | None = Field(default=None, alias=Fields.CONC_START.value)
    conc_end: int | None = Field(default=None, alias=Fields.CONC_END.value)
    conc_list: list[int] | None = Field(default=None, alias=Fields.CONC_LIST.value)

    @model_validator(mode="after")
    def validate_conc_fields(self) -> Self:
        """Return the entry, raising ValueError for an invalid conc range or list."""
        return _validate_conc_fields(self)

    @model_validator(mode="after")
    def validate_kv_offload_fields(self) -> Self:
        """Return the entry, raising ValueError for inconsistent KV offload fields."""
        return _validate_kv_offload_fields(self)

    @model_validator(mode="after")
    def validate_topology_fields(self) -> Self:
        """Return the entry, raising ValueError unless it sets exactly one valid topology."""
        has_single_node = self.tp is not None
        has_aggregate_worker = self.worker is not None
        has_any_multinode_field = self.prefill is not None or self.decode is not None
        has_complete_multinode = self.prefill is not None and self.decode is not None
        topology_count = sum(
            (
                has_single_node,
                has_aggregate_worker,
                has_complete_multinode,
            )
        )
        if topology_count != 1 or (has_any_multinode_field and not has_complete_multinode):
            raise ValueError(
                "Agentic search-space entries must specify exactly one of tp, "
                "worker, or both prefill and decode"
            )
        if has_single_node:
            if self.kv_offloading is None:
                raise ValueError(
                    f"Single-node agentic search-space entries must specify "
                    f"{Fields.KV_OFFLOADING.value}"
                )
            if self.power:
                raise ValueError(
                    f"Only multinode entries measure DCGM power; remove {Fields.POWER.value}"
                )
            _validate_tp_context_topology(self)
        if has_aggregate_worker or has_complete_multinode:
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
                        Fields.DWDP_SIZE.value,
                    )
                )
                raise ValueError(
                    f"Multinode agentic search-space entries cannot specify {field_names}"
                )
            if has_complete_multinode:
                _validate_worker_hardware_pair(self)
        return self


class AgenticCodingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    search_space: list[AgenticCodingSearchSpaceEntry] = Field(alias=Fields.SEARCH_SPACE.value)
    dram_utilization: float | None = Field(
        default=None, alias=Fields.DRAM_UTILIZATION.value, gt=0, le=1
    )

    @model_validator(mode="after")
    def validate_dram_offload_capacity(self) -> Self:
        """Return the config, raising ValueError for DRAM offload it cannot size."""
        for entry in self.search_space:
            if entry.kv_offloading != "dram":
                continue
            if self.dram_utilization is None:
                raise ValueError(
                    f"{Fields.KV_OFFLOADING.value}='dram' requires "
                    f"{Fields.DRAM_UTILIZATION.value} with runner hardware metadata"
                )
            if entry.prefill is not None and entry.prefill.hardware is not None:
                raise ValueError(
                    f"{Fields.KV_OFFLOADING.value}='dram' does not support per-worker "
                    f"'{Fields.HARDWARE.value}'"
                )
        return self


class SingleNodeScenarios(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    fixed_seq_len: list[SingleNodeSeqLenConfig] | None = Field(
        default=None, alias=Fields.FIXED_SEQ_LEN.value
    )
    agentic_coding: list[AgenticCodingConfig] | None = Field(
        default=None, alias=Fields.AGENTIC_CODING.value
    )

    @model_validator(mode="after")
    def at_least_one_scenario(self) -> Self:
        """Return the scenarios, raising ValueError when none is set."""
        if not self.fixed_seq_len and not self.agentic_coding:
            raise ValueError("At least one scenario type must be specified")
        return self


class MultiNodeScenarios(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    fixed_seq_len: list[MultiNodeSeqLenConfig] | None = Field(
        default=None, alias=Fields.FIXED_SEQ_LEN.value
    )
    agentic_coding: list[AgenticCodingConfig] | None = Field(
        default=None, alias=Fields.AGENTIC_CODING.value
    )

    @model_validator(mode="after")
    def at_least_one_scenario(self) -> Self:
        """Return the scenarios, raising ValueError when none is set."""
        if not self.fixed_seq_len and not self.agentic_coding:
            raise ValueError("At least one scenario type must be specified")
        return self


def _master_search_space_entries(self: BaseModel) -> list[BaseModel]:
    """Return every search-space entry across a master config's scenarios."""
    return [
        entry
        for scenario_configs in (
            self.scenarios.fixed_seq_len,
            self.scenarios.agentic_coding,
        )
        for scenario_config in scenario_configs or []
        for entry in scenario_config.search_space
    ]


def _validate_component_metadata_scope(self: BaseModel) -> BaseModel:
    """Return the config, raising ValueError if router or KV transfer settings are misplaced."""
    search_space_entries = _master_search_space_entries(self)

    for field in (Fields.ROUTER, Fields.KV_P2P_TRANSFER):
        attribute = field.value.replace("-", "_")
        top_level_value = getattr(self, attribute, None)
        has_search_space_value = any(
            getattr(entry, attribute, None) is not None for entry in search_space_entries
        )
        if top_level_value is not None and has_search_space_value:
            raise ValueError(
                f"{field.value} must be declared either at the top level or "
                "in search-space entries, not both"
            )

    has_search_space_transfer = any(
        getattr(entry, "kv_p2p_transfer", None) is not None for entry in search_space_entries
    )
    if not self.multinode and has_search_space_transfer:
        raise ValueError(
            f"{Fields.KV_P2P_TRANSFER.value} is only valid when {Fields.MULTINODE.value}=true"
        )

    top_level_transfer = getattr(self, "kv_p2p_transfer", None)
    if (
        self.disagg
        and top_level_transfer is None
        and (
            not search_space_entries
            or any(entry.kv_p2p_transfer is None for entry in search_space_entries)
        )
    ):
        raise ValueError(
            f"{Fields.DISAGG.value}=true requires "
            f"{Fields.KV_P2P_TRANSFER.value} at the top level or in every "
            "search-space entry"
        )

    return self


def _validate_multinode_entry_scope(self: BaseModel) -> BaseModel:
    """Return the config, raising ValueError if an entry's workers do not fit its node type."""
    search_space_entries = _master_search_space_entries(self)
    for entry in search_space_entries:
        worker = getattr(entry, "worker", None)
        prefill = getattr(entry, "prefill", None)
        decode = getattr(entry, "decode", None)
        num_nodes = getattr(entry, "num_nodes", None)

        if not self.multinode:
            if (
                worker is not None
                or prefill is not None
                or decode is not None
                or num_nodes is not None
            ):
                raise ValueError(
                    "Single-node search-space entries must specify tp topology "
                    "and cannot declare worker, prefill, decode, or num-nodes"
                )
            continue

        if self.disagg:
            if worker is not None or num_nodes is not None:
                raise ValueError(
                    f"{Fields.DISAGG.value}=true requires prefill and decode "
                    f"and rejects {Fields.WORKER.value} and "
                    f"{Fields.NUM_NODES.value}"
                )
            if prefill is None or decode is None:
                raise ValueError(f"{Fields.DISAGG.value}=true requires prefill and decode")
            continue

        if worker is None or prefill is not None or decode is not None:
            raise ValueError(
                f"{Fields.DISAGG.value}=false requires one "
                f"{Fields.WORKER.value} and rejects prefill and decode"
            )
        if num_nodes is None:
            raise ValueError(
                f"{Fields.DISAGG.value}=false requires "
                f"{Fields.NUM_NODES.value} in every search-space entry"
            )
    return self


_RECIPE_SETTINGS = {"CONFIG_FILE": Fields.SRT_RECIPE, "EVAL_CONFIG_FILE": Fields.EVAL_SRT_RECIPE}


def _validate_srt_recipes(self: Any) -> Any:
    """Return the config, raising ValueError unless every entry names a valid srt recipe."""
    entries = _master_search_space_entries(self)
    if self.srt_recipe_dir is None:
        raise ValueError(f"srt-slurm configs require {Fields.SRT_RECIPE_DIR.value}")
    for entry in entries:
        if entry.srt_recipe is None:
            raise ValueError(f"Every search-space entry requires {Fields.SRT_RECIPE.value}")
        if not self.multinode and getattr(entry, "eval_srt_recipe", None) is not None:
            raise ValueError(f"{Fields.EVAL_SRT_RECIPE.value} is only valid for multinode configs")
        workers = (getattr(entry, role, None) for role in ("worker", "prefill", "decode"))
        for worker in filter(None, workers):
            for setting in worker.additional_settings or []:
                name = setting.partition("=")[0]
                if name in _RECIPE_SETTINGS:
                    raise ValueError(
                        f"{name} is not an additional-setting; use {_RECIPE_SETTINGS[name].value}"
                    )
    return self


class SingleNodeMasterConfigEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    image: str
    model: str
    model_prefix: str = Field(alias=Fields.MODEL_PREFIX.value)
    precision: str
    framework: str
    runner: str
    multinode: Literal[False]
    disagg: Literal[False] = Field(default=False)
    router: ComponentMetadata | None = None
    srt_recipe_dir: RecipeDir = Field(alias=Fields.SRT_RECIPE_DIR.value)
    scenarios: SingleNodeScenarios

    @model_validator(mode="after")
    def validate_agentic_runner(self) -> Self:
        """Return the config, raising ValueError if agentic scenarios use a non-cluster runner."""
        _validate_agentic_runner_is_cluster(self.runner, self.scenarios)
        return self

    @model_validator(mode="after")
    def validate_component_metadata_scope(self) -> Self:
        """Return the config, raising ValueError if router or KV transfer settings are misplaced."""
        return _validate_component_metadata_scope(self)

    @model_validator(mode="after")
    def validate_multinode_entry_scope(self) -> Self:
        """Return the config, raising ValueError if an entry's workers do not fit its node type."""
        return _validate_multinode_entry_scope(self)

    @model_validator(mode="after")
    def validate_srt_recipes(self) -> Self:
        """Return the config, raising ValueError unless every entry names a valid srt recipe."""
        return _validate_srt_recipes(self)


class MultiNodeMasterConfigEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    image: str
    model: str
    model_prefix: str = Field(alias=Fields.MODEL_PREFIX.value)
    precision: str
    framework: str
    runner: str
    multinode: Literal[True]
    disagg: bool = Field(default=False)
    router: ComponentMetadata | None = None
    kv_p2p_transfer: str | None = Field(
        default=None, alias=Fields.KV_P2P_TRANSFER.value, min_length=1
    )
    srt_recipe_dir: RecipeDir | None = Field(default=None, alias=Fields.SRT_RECIPE_DIR.value)
    scenarios: MultiNodeScenarios

    @model_validator(mode="after")
    def validate_agentic_runner(self) -> Self:
        """Return the config, raising ValueError if agentic scenarios use a non-cluster runner."""
        _validate_agentic_runner_is_cluster(self.runner, self.scenarios)
        return self

    @model_validator(mode="after")
    def validate_component_metadata_scope(self) -> Self:
        """Return the config, raising ValueError if router or KV transfer settings are misplaced."""
        return _validate_component_metadata_scope(self)

    @model_validator(mode="after")
    def validate_multinode_entry_scope(self) -> Self:
        """Return the config, raising ValueError if an entry's workers do not fit its node type."""
        return _validate_multinode_entry_scope(self)

    @model_validator(mode="after")
    def validate_srt_recipes(self) -> Self:
        """Return the config, raising ValueError unless every entry names a valid srt recipe."""
        return _validate_srt_recipes(self)


def srt_recipe_path(multinode: bool, directory: str, reference: str) -> str:
    """Return the repository path of a recipe ``reference`` in its single- or multinode dir."""
    return f"{MULTI_NODE_RECIPES if multinode else SINGLE_NODE_RECIPES}/{directory}/{reference}"


def _zip_length(group: Any) -> int:
    """Return the variant count of a zip override group, its longest list."""
    if isinstance(group, list):
        return len(group)
    if isinstance(group, dict):
        return max(map(_zip_length, group.values()), default=0)
    return 0


def _check_srt_recipe(root: Path, reference: str, recipes: dict[str, Any]) -> None:
    """Raise ValueError unless the recipe file exists, parses, and has the selected variant.

    Parsed recipes are cached in ``recipes`` by path.
    """
    path, _, selector = reference.partition(":")
    if path not in recipes:
        if not (root / path).is_file():
            raise ValueError(f"{reference} does not exist")
        try:
            recipes[path] = yamlio.safe_load((root / path).read_text())
        except yaml.YAMLError as error:
            raise ValueError(f"{path} is not valid YAML") from error
    recipe = recipes[path]
    if not selector:
        return
    if not isinstance(recipe, dict) or "base" not in recipe:
        raise ValueError(f"{reference}: only an override bundle with a base takes a selector")
    group, _, index = selector.removesuffix("]").partition("[")
    if group not in recipe or (index and int(index) >= _zip_length(recipe[group])):
        raise ValueError(f"{reference}: {path} has no variant {selector}")


def config_root(config_files: list[str]) -> Path:
    """Return the project directory shared by the config files, or raise ValueError."""
    roots = {Path(config_file).resolve().parent.parent for config_file in config_files}
    if len(roots) != 1:
        raise ValueError(f"Master configs must share one project directory: {sorted(roots)}")
    return roots.pop()


def srt_recipe_references(master_configs: dict) -> dict[str, list[str]]:
    """Return the srt recipe paths each master config key references.

    Raise ValueError when a master config entry fails validation.
    """
    references = {}
    for key, entry in master_configs.items():
        model = (
            MultiNodeMasterConfigEntry
            if entry.get("multinode", False)
            else SingleNodeMasterConfigEntry
        )
        try:
            config = model(**entry)
        except ValidationError as e:
            raise ValueError(f"Master config entry '{key}' failed validation:\n{e}") from e
        references[key] = [
            srt_recipe_path(config.multinode, config.srt_recipe_dir, reference)
            for row in _master_search_space_entries(config)
            for reference in (row.srt_recipe, getattr(row, "eval_srt_recipe", None))
            if reference is not None
        ]
    return references


def validate_master_config(master_configs: dict, root: Path) -> dict:
    """Return the master configs, raising ValueError if an entry or its recipes are invalid."""
    recipes: dict[str, Any] = {}
    for key, references in srt_recipe_references(master_configs).items():
        for reference in references:
            try:
                _check_srt_recipe(root, reference, recipes)
            except ValueError as e:
                raise ValueError(f"Master config entry '{key}' failed validation:\n{e}") from e
    return master_configs


def _validate_runner_labels(labels: dict) -> None:
    """Raise ValueError unless every runner label maps to a non-empty list of strings."""
    for key, value in labels.items():
        if not isinstance(value, list):
            raise ValueError(
                f"Runner config entry '{key}' must be a list, got {type(value).__name__}"
            )

        if not all(isinstance(item, str) for item in value):
            raise ValueError(f"Runner config entry '{key}' must contain only strings")

        if not value:
            raise ValueError(f"Runner config entry '{key}' cannot be an empty list")


def validate_runner_config(runner_configs: dict) -> dict:
    """Return the runner config, raising ValueError if its labels or clusters are invalid."""
    labels = runner_configs.get("labels")
    if not isinstance(labels, dict):
        raise ValueError("Runner config must define a labels mapping")
    _validate_runner_labels(labels)
    try:
        RunnerInventory.model_validate(runner_configs)
    except ValidationError as e:
        raise ValueError(f"Runner config failed validation:\n{e}") from e
    return runner_configs


class ChangelogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    config_keys: list[str] = Field(alias="config-keys", min_length=1)
    description: list[str] = Field(min_length=1)
    pr_link: str = Field(alias="pr-link")
    evals_only: bool = Field(alias="evals-only", default=False)
    all_evals: bool = Field(alias="all-evals", default=False)
    no_evals: bool = Field(alias="no-evals", default=False)
    append_only: bool = Field(
        alias="append-only",
        default=False,
        description=(
            "Run only generated points or recipe variants added while preserving "
            "every existing generated point, then append them to the latest curve"
        ),
    )
    eval_min_prefill_ep: int | None = Field(
        alias="eval-min-prefill-ep",
        default=None,
        ge=1,
        description=(
            "When set, multinode eval rows whose prefill.ep is below this "
            "threshold are dropped after eval selection."
        ),
    )
    scenario_type: list[Literal["fixed-seq-len", "agentic-coding"]] | None = Field(
        alias="scenario-type",
        default=None,
        min_length=1,
        description="Restrict to specific scenario types (e.g., ['fixed-seq-len', 'agentic-coding'])",
    )

    @model_validator(mode="after")
    def validate_append_only_mode(self) -> Self:
        """Return the entry, raising ValueError if no-evals or append-only selects evals."""
        if self.no_evals and (
            self.evals_only or self.all_evals or self.eval_min_prefill_ep is not None
        ):
            raise ValueError("no-evals cannot be combined with eval selection fields")
        if self.append_only and (
            self.evals_only or self.all_evals or self.eval_min_prefill_ep is not None
        ):
            raise ValueError("append-only cannot be combined with eval selection fields")
        return self


class ChangelogMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_ref: str
    head_ref: str
    entries: list[ChangelogEntry]


class ChangelogMatrixEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    single_node: dict[str, list[SingleNodeMatrixEntry | SingleNodeAgenticMatrixEntry]] = Field(
        default_factory=dict
    )
    multi_node: dict[str, list[MultiNodeMatrixEntry | MultiNodeAgenticMatrixEntry]] = Field(
        default_factory=dict
    )
    evals: list[SingleNodeMatrixEntry] = Field(default_factory=list)
    agentic_evals: list[SingleNodeAgenticMatrixEntry] = Field(default_factory=list)
    multinode_evals: list[MultiNodeMatrixEntry] = Field(default_factory=list)
    multinode_agentic_evals: list[MultiNodeAgenticMatrixEntry] = Field(default_factory=list)
    changelog_metadata: ChangelogMetadata


def load_config_files(config_files: list[str], validate: bool = True) -> dict:
    """Load and merge the master config files, validating them unless ``validate`` is False.

    Raise ValueError for a missing file, bad or duplicate keys, or an invalid config.
    """
    all_config_data = {}
    for config_file in config_files:
        try:
            with open(config_file) as f:
                config_data = yamlio.load(f, yamlio.UniqueKeySafeLoader)
                if not isinstance(config_data, dict):
                    raise ValueError(f"Config file '{config_file}' must contain a dictionary")

                for key in config_data:
                    if not isinstance(key, str):
                        raise ValueError(
                            f"Configuration key {key!r} in '{config_file}' must be a string"
                        )
                    if "*" in key:
                        raise ValueError(
                            f" Wildcard '*' is not allowed in master config keys: '{key}'"
                        )

                duplicate_keys = all_config_data.keys() & config_data.keys()
                if duplicate_keys:
                    raise ValueError(
                        f"Duplicate configuration keys found in '{config_file}': {', '.join(sorted(duplicate_keys))}"
                    )

                all_config_data.update(config_data)
        except FileNotFoundError as e:
            raise ValueError(f"Input file '{config_file}' does not exist.") from e

    if validate:
        validate_master_config(all_config_data, config_root(config_files))

    return all_config_data


def load_runner_file(runner_file: str, validate: bool = True) -> dict:
    """Load the runner config, validating it unless ``validate`` is False.

    Raise ValueError when the file is missing or invalid.
    """
    try:
        with open(runner_file) as f:
            runner_config = yamlio.safe_load(f)
    except FileNotFoundError as e:
        raise ValueError(f"Runner config file '{runner_file}' does not exist.") from e

    if validate:
        validate_runner_config(runner_config)

    return runner_config
