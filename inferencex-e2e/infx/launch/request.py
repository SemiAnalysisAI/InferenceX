"""Typed views of the workflow environment a launch reads.

An empty variable counts as unset. Each launch path parses the environment again with its
own model, whose required fields are the inputs that path and its job scripts need, so a
launch fails before any work naming every missing or invalid one. Children still get the
raw environment (``LaunchRequest.env``).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, model_validator


class RequestError(ValueError):
    """The workflow environment lacks an input the launch path needs, or has an invalid one."""

    @classmethod
    def missing(cls, *names: str) -> RequestError:
        return cls(f"required environment variables are not set: {', '.join(names)}")


def _equals(*truthy: str) -> BeforeValidator:
    return BeforeValidator(lambda value: value if isinstance(value, bool) else value in truthy)


def _int_list(value: Any) -> Any:
    return value.split() if isinstance(value, str) else value


OneFlag = Annotated[bool, _equals("1")]
TrueFlag = Annotated[bool, _equals("true")]
PowerFlag = Annotated[bool, _equals("1", "true", "TRUE", "yes", "YES")]
IntList = Annotated[list[int], BeforeValidator(_int_list)]


def _describe(error: ValidationError) -> str:
    missing: list[str] = []
    invalid: list[str] = []
    for detail in error.errors(include_url=False, include_input=False):
        name = ".".join(str(part) for part in detail["loc"])
        if detail["type"] == "missing":
            missing.append(name)
        elif not name and detail["type"] == "value_error":
            invalid.append(str(detail["ctx"]["error"]))
        else:
            invalid.append(f"{name}: {detail['msg']}")
    return "; ".join([*([RequestError.missing(*missing).args[0]] if missing else []), *invalid])


BATCH_REENTRY_ENV = "INFX_BATCH_REENTRY"


class LaunchRequest(BaseModel):
    """What routing, the shared policy and the backends read; each path's model adds its own."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, frozen=True)

    runner_name: str = Field(alias="RUNNER_NAME")
    github_workspace: Path | None = Field(None, alias="GITHUB_WORKSPACE")
    model: str | None = Field(None, alias="MODEL")
    model_prefix: str | None = Field(None, alias="MODEL_PREFIX")
    image: str | None = Field(None, alias="IMAGE")
    framework: str | None = Field(None, alias="FRAMEWORK")
    precision: str | None = Field(None, alias="PRECISION")
    spec_decoding: str | None = Field(None, alias="SPEC_DECODING")
    is_multinode: TrueFlag = Field(False, alias="IS_MULTINODE")
    is_agentic: OneFlag = Field(False, alias="IS_AGENTIC")
    config_file: str | None = Field(None, alias="CONFIG_FILE")
    eval_config_file: str | None = Field(None, alias="EVAL_CONFIG_FILE")
    bench_script_override: str | None = Field(None, alias="BENCH_SCRIPT_OVERRIDE")
    batch_reentry: OneFlag = Field(False, alias=BATCH_REENTRY_ENV)
    conc: int | None = Field(None, alias="CONC")
    run_eval: TrueFlag = Field(False, alias="RUN_EVAL")
    eval_only: TrueFlag = Field(False, alias="EVAL_ONLY")
    salloc_time_limit: int | None = Field(None, alias="SALLOC_TIME_LIMIT")
    enroot_import_time_limit: int | None = Field(None, alias="ENROOT_IMPORT_TIME_LIMIT")

    env: dict[str, str] = Field(default_factory=dict, exclude=True, repr=False)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Self:
        """Parse ``env`` (default ``os.environ``); raise ``RequestError`` naming bad inputs."""
        source = dict(os.environ if env is None else env)
        aliases = {field.alias for field in cls.model_fields.values() if field.alias}
        values = {key: value for key, value in source.items() if key in aliases and value}
        try:
            return cls.model_validate({**values, "env": source})
        except ValidationError as error:
            raise RequestError(_describe(error)) from None

    @property
    def workspace(self) -> Path:
        if self.github_workspace is None:
            raise RequestError.missing("GITHUB_WORKSPACE")
        return self.github_workspace


class SrtRequest(LaunchRequest):
    """What every srt-slurm submission reads; a multi-node srt-slurm job needs exactly this."""

    github_workspace: Path = Field(alias="GITHUB_WORKSPACE")
    image: str = Field(alias="IMAGE")
    framework: str = Field(alias="FRAMEWORK")
    model_prefix: str = Field(alias="MODEL_PREFIX")
    precision: str = Field(alias="PRECISION")
    spec_decoding: str = Field(alias="SPEC_DECODING")
    result_filename: str = Field(alias="RESULT_FILENAME")
    is_agentic: OneFlag = Field(alias="IS_AGENTIC")
    run_eval: TrueFlag = Field(alias="RUN_EVAL")
    eval_only: TrueFlag = Field(alias="EVAL_ONLY")
    thinking_mode: str | None = Field(None, alias="THINKING_MODE")
    conc_list: IntList = Field(default_factory=list, alias="CONC_LIST")
    require_power: PowerFlag = Field(False, alias="REQUIRE_POWER")
    inferencex_results_python: str | None = Field(None, alias="INFERENCEX_RESULTS_PYTHON")
    eval_conc: str | None = Field(None, alias="EVAL_CONC")

    @model_validator(mode="after")
    def _golden_curve_key(self) -> Self:
        """Speculative AgentX throughput runs select their golden curve by THINKING_MODE."""
        speculative = self.spec_decoding != "none"
        if self.is_agentic and speculative and not self.eval_only and not self.thinking_mode:
            raise ValueError("THINKING_MODE is required for speculative AgentX throughput runs")
        return self


class SingleNodeRequest(SrtRequest):
    """One native single-node srt-slurm point, bound to one recipe variant."""

    srt_recipe: str = Field(alias="SRT_RECIPE")
    model: str = Field(alias="MODEL")
    tp: int = Field(alias="TP")
    pp_size: int = Field(alias="PP_SIZE")
    dcp_size: int = Field(alias="DCP_SIZE")
    pcp_size: int = Field(alias="PCP_SIZE")
    ep_size: int = Field(alias="EP_SIZE")
    dp_attention: str = Field(alias="DP_ATTENTION")
    gpu_count: int = Field(alias="GPU_COUNT")
    conc: int = Field(alias="CONC")
    isl: int = Field(alias="ISL")
    osl: int = Field(alias="OSL")
    random_range_ratio: str = Field(alias="RANDOM_RANGE_RATIO")
    gpu_monitor_interval: str = Field(alias="GPU_MONITOR_INTERVAL")
    hf_hub_cache: str = Field(alias="HF_HUB_CACHE")
    salloc_time_limit: int = Field(alias="SALLOC_TIME_LIMIT")


class ScriptRequest(LaunchRequest):
    """One explicit script (``BENCH_SCRIPT_OVERRIDE``, e.g. a SPEED-Bench collector)."""

    github_workspace: Path = Field(alias="GITHUB_WORKSPACE")
    bench_script_override: str = Field(alias="BENCH_SCRIPT_OVERRIDE")
    image: str = Field(alias="IMAGE")
    model: str = Field(alias="MODEL")
    gpu_count: int = Field(alias="GPU_COUNT")
    salloc_time_limit: int = Field(alias="SALLOC_TIME_LIMIT")


class LegacyRequest(LaunchRequest):
    """A pre-srt-slurm lane; its script path is built from these inputs."""

    github_workspace: Path = Field(alias="GITHUB_WORKSPACE")
    exp_name: str = Field(alias="EXP_NAME")
    precision: str = Field(alias="PRECISION")
    framework: str = Field(alias="FRAMEWORK")
    scenario_subdir: str | None = Field(None, alias="SCENARIO_SUBDIR")


class AmdUtilsRequest(LegacyRequest):
    """An MI355X AgentX job submitted through amd_utils, which serves ``MODEL``'s basename."""

    model: str = Field(alias="MODEL")
    user: str | None = Field(None, alias="USER")
    keep_logs: OneFlag = Field(False, alias="KEEP_LOGS")


class LlmdRequest(SrtRequest):
    """An llm-d vLLM multinode job submitted through benchmarks/multi_node/llm-d."""

    model: str = Field(alias="MODEL")
    disagg: TrueFlag = Field(alias="DISAGG")
    prefill_nodes: int = Field(alias="PREFILL_NODES")
    decode_nodes: int = Field(alias="DECODE_NODES")
    prefill_num_workers: int = Field(1, alias="PREFILL_NUM_WORKERS")
    decode_num_workers: int = Field(1, alias="DECODE_NUM_WORKERS")
    isl: int = Field(alias="ISL")
    osl: int = Field(alias="OSL")
    random_range_ratio: str = Field(alias="RANDOM_RANGE_RATIO")
