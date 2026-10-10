"""Pydantic models of the aggregate rows that collectors publish."""

import re
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    JsonValue,
    Tag,
    TypeAdapter,
    model_validator,
)

from infx.results.power import ALL_POWER_METRIC_KEYS, POWER_METRIC_SCHEMA_VERSION

from . import RESULT_SCHEMA_VERSION


def _exactly(value: int) -> Any:
    # Literal[1] would also accept True and 1.0.
    return Annotated[int, Field(ge=value, le=value, json_schema_extra={"const": value})]


SchemaVersion = _exactly(RESULT_SCHEMA_VERSION)
PowerSchemaVersion = _exactly(POWER_METRIC_SCHEMA_VERSION)
PowerValid = Annotated[int, Field(ge=0, le=1)]
NonEmpty = Annotated[str, Field(min_length=1)]
Positive = Annotated[int, Field(gt=0)]
NonNegative = Annotated[int, Field(ge=0)]
DpAttention = Annotated[
    Literal["true", "false"],
    Field(description="Workflow flag copied as a string, not a JSON boolean."),
]
RecipeFingerprint = Annotated[
    str,
    Field(pattern=r"^(?:[0-9a-f]{64})?$", description="Empty when the dispatched config has none."),
]
StdErr = Annotated[
    float | Literal["N/A"],
    Field(description="lm-eval writes 'N/A' when it cannot estimate a standard error."),
]

_POWER_METRICS = "|".join(re.escape(key) for key in ALL_POWER_METRIC_KEYS)
_LATENCY_METRICS = r"(?:mean|median|std|p[0-9]+(?:\.[0-9]+)?)_(?:ttft|tpot|itl|e2el|intvty)"
FIXED_SEQUENCE_METRIC = rf"^(?:{_LATENCY_METRICS}|{_POWER_METRICS})$"
AGENTX_METRIC = rf"^(?:{_POWER_METRICS})$"


def _metric_families(pattern: str) -> ConfigDict:
    def document(schema: dict[str, Any]) -> None:
        schema["patternProperties"] = {pattern: {"type": "number"}}
        schema["additionalProperties"] = False

    return ConfigDict(strict=True, extra="allow", allow_inf_nan=False, json_schema_extra=document)


class _Contract(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)


class ComponentMetadata(_Contract):
    name: NonEmpty
    version: NonEmpty


class OffloadBackend(_Contract):
    name: NonEmpty
    version: NonEmpty = None


class BenchmarkOutcome(_Contract):
    status: Literal["passed", "failed"]
    requested: Positive
    completed: NonNegative
    failed: NonNegative
    max_failure_rate: float


class RequestAccounting(_Contract):
    records_total: NonNegative
    records_profiled: NonNegative
    records_dropped_total: NonNegative
    records_warmup_dropped: NonNegative
    records_error_dropped: NonNegative
    error_categories: dict[str, NonNegative]


class _FixedSequenceRow(BaseModel):
    model_config = _metric_families(FIXED_SEQUENCE_METRIC)
    __pydantic_extra__: dict[Annotated[str, Field(pattern=FIXED_SEQUENCE_METRIC)], float] = Field(
        init=False
    )

    result_schema_version: SchemaVersion
    hw: NonEmpty
    conc: Positive
    image: NonEmpty
    model: NonEmpty
    infmax_model_prefix: NonEmpty
    framework: NonEmpty
    precision: NonEmpty
    spec_decoding: NonEmpty
    disagg: bool
    recipe_fingerprint: RecipeFingerprint
    isl: Positive
    osl: Positive
    benchmark_outcome: BenchmarkOutcome = None
    router: ComponentMetadata = None
    kv_p2p_transfer: NonEmpty = None
    tput_per_gpu: float
    output_tput_per_gpu: float
    input_tput_per_gpu: float
    power_metric_schema_version: PowerSchemaVersion = None
    power_valid: PowerValid = None
    power_invalid_reasons: list[str] = None
    power_audit: dict[str, JsonValue] = None


class FixedSequenceSingleNodeRow(_FixedSequenceRow):
    """Fixed-sequence point served by one single-node deployment."""

    is_multinode: Literal[False]
    tp: Positive
    pp: Positive
    dcp_size: Positive
    pcp_size: Positive
    ep: Positive
    dp_attention: DpAttention


class FixedSequenceMultinodeRow(_FixedSequenceRow):
    """Fixed-sequence point; decode TP/EP are 0 when the deployment has no decode GPUs."""

    is_multinode: Literal[True]
    prefill_tp: Positive
    prefill_pp: Positive
    prefill_dcp_size: Positive
    prefill_pcp_size: Positive
    prefill_ep: Positive
    prefill_dp_attention: DpAttention
    prefill_num_workers: NonNegative
    decode_tp: NonNegative
    decode_pp: Positive
    decode_dcp_size: Positive
    decode_pcp_size: Positive
    decode_ep: NonNegative
    decode_dp_attention: DpAttention
    decode_num_workers: NonNegative
    num_prefill_gpu: Positive
    num_decode_gpu: NonNegative
    num_aggregate_gpu: Positive = None
    prefill_hw: NonEmpty = None
    decode_hw: NonEmpty = None


class _AgentXRow(BaseModel):
    model_config = _metric_families(AGENTX_METRIC)
    __pydantic_extra__: dict[Annotated[str, Field(pattern=AGENTX_METRIC)], float] = Field(
        init=False
    )

    result_schema_version: SchemaVersion
    hw: NonEmpty
    conc: Positive
    image: NonEmpty
    recipe_fingerprint: RecipeFingerprint
    model: NonEmpty
    infmax_model_prefix: NonEmpty
    framework: NonEmpty
    precision: NonEmpty
    spec_decoding: NonEmpty
    disagg: bool
    scenario_type: Literal["agentic-coding"]
    num_gpus: Positive
    tp: Positive
    ep: Positive
    dp_attention: DpAttention
    kv_offloading: NonEmpty
    kv_offload_backend: OffloadBackend | None
    allocated_cpu_dram_gb: NonNegative
    num_requests_total: NonNegative
    num_requests_successful: NonNegative
    request_accounting: RequestAccounting
    router: ComponentMetadata = None
    kv_p2p_transfer: NonEmpty = None
    dataset: dict[str, JsonValue] = None
    request_metrics: dict[str, JsonValue]
    server_metrics: dict[str, JsonValue]
    kv_cache_pool_tokens: NonNegative | None
    warnings: list[str] = None
    power_metric_schema_version: PowerSchemaVersion = None
    power_valid: PowerValid = None


class AgentXSingleNodeRow(_AgentXRow):
    """AgentX point served by one single-node deployment."""

    is_multinode: Literal[False]
    pp: Positive
    dcp_size: Positive
    pcp_size: Positive


class AgentXMultinodeRow(_AgentXRow):
    """AgentX point; tp sums both roles, ep is the larger role's, dp_attention is either role's."""

    is_multinode: Literal[True]
    prefill_num_workers: NonNegative
    prefill_tp: Positive
    prefill_pp: Positive
    prefill_dcp_size: Positive
    prefill_pcp_size: Positive
    prefill_ep: Positive
    prefill_dp_attention: DpAttention
    num_prefill_gpu: NonNegative
    decode_num_workers: NonNegative
    decode_tp: NonNegative
    decode_pp: Positive
    decode_dcp_size: Positive
    decode_pcp_size: Positive
    decode_ep: NonNegative
    decode_dp_attention: DpAttention
    num_decode_gpu: NonNegative
    prefill_hw: NonEmpty = None
    decode_hw: NonEmpty = None


class EvalRow(_Contract):
    """One task score; string fields keep build_row defaults such as 'unknown'."""

    result_schema_version: SchemaVersion
    is_multinode: bool
    model_prefix: str
    model: str
    hw: Annotated[str, Field(description="Upper-cased runner type.")]
    framework: str
    precision: str
    spec_decoding: str
    isl: Annotated[NonNegative, Field(description="0 when the eval has no fixed lengths.")]
    osl: Annotated[NonNegative, Field(description="0 when the eval has no fixed lengths.")]
    tp: NonNegative
    ep: NonNegative
    prefill_tp: NonNegative
    prefill_ep: NonNegative
    prefill_num_workers: NonNegative
    decode_tp: NonNegative
    decode_ep: NonNegative
    decode_num_workers: NonNegative
    conc: Positive
    dp_attention: Annotated[
        str,
        Field(description="'true', 'false', 'none' or 'prefill=<flag>,decode=<flag>'."),
    ]
    prefill_dp_attention: str
    decode_dp_attention: str
    task: NonEmpty
    em_strict: float | None
    em_strict_se: StdErr | None
    em_flexible: float | None
    em_flexible_se: StdErr | None
    n_eff: Annotated[float, Field(ge=0)] | None
    source: NonEmpty
    infrastructure_success: bool
    integration_error: dict[str, JsonValue] | None
    eval_suite: NonEmpty = None
    score: Annotated[float, Field(ge=0, le=1)] | None
    score_name: Literal["em_strict", "accuracy", "em_flexible"] | None
    score_se: StdErr | None


class RunStatsRow(_Contract):
    """Job counts for one hardware key of the run-stats object."""

    result_schema_version: SchemaVersion
    n_success: NonNegative
    total: NonNegative

    @model_validator(mode="after")
    def _within_total(self) -> Self:
        if self.n_success > self.total:
            raise ValueError("n_success exceeds total")
        return self


def _topology(row: Any) -> str | None:
    flag = row.get("is_multinode") if isinstance(row, dict) else None
    if not isinstance(flag, bool):
        return None
    return "multinode" if flag else "single-node"


def _by_topology(single_node: type[BaseModel], multinode: type[BaseModel]) -> Any:
    return Annotated[
        Annotated[single_node, Tag("single-node")] | Annotated[multinode, Tag("multinode")],
        Discriminator(
            _topology,
            custom_error_type="topology",
            custom_error_message="Row must be an object with a boolean is_multinode",
        ),
    ]


FixedSequenceRow = _by_topology(FixedSequenceSingleNodeRow, FixedSequenceMultinodeRow)
AgentXRow = _by_topology(AgentXSingleNodeRow, AgentXMultinodeRow)

FIXED_SEQUENCE_ROW: TypeAdapter[Any] = TypeAdapter(FixedSequenceRow)
AGENTX_ROW: TypeAdapter[Any] = TypeAdapter(AgentXRow)
EVAL_ROW: TypeAdapter[Any] = TypeAdapter(EvalRow)
RUN_STATS_ROW: TypeAdapter[Any] = TypeAdapter(RunStatsRow)


def benchmark_row(row: object) -> TypeAdapter[Any]:
    """Only AgentX rows carry scenario_type; results_bmk mixes both kinds."""
    return AGENTX_ROW if isinstance(row, dict) and "scenario_type" in row else FIXED_SEQUENCE_ROW
