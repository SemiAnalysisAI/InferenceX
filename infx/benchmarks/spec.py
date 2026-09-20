"""Versioned, explicit contracts for the first aggregate client lane."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PositiveInt = Annotated[int, Field(gt=0)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def secret_environment_key(key: str) -> bool:
    upper = key.upper()
    return (
        any(
            part in upper
            for part in (
                "SECRET",
                "PASSWORD",
                "CREDENTIAL",
                "PRIVATE_KEY",
                "API_KEY",
                "AUTHORIZATION",
            )
        )
        or upper.endswith("_TOKEN")
        or upper in {"SSH_AUTH_SOCK", "SSH_AGENT_PID"}
    )


def validate_environment(env: dict[str, str], unset: list[str]) -> None:
    if set(env) & set(unset):
        raise ValueError("environment set/unset keys must not overlap")
    if any(not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", key) for key in [*env, *unset]):
        raise ValueError("invalid environment variable name")
    if any(secret_environment_key(key) for key in env):
        raise ValueError("offline prepared environments may not contain secret-bearing variables")
    if any("\0" in value for value in env.values()):
        raise ValueError("environment values may not contain NUL")
    if env.get("HF_HUB_OFFLINE") != "1" or env.get("HF_DATASETS_OFFLINE") != "1":
        raise ValueError("prepared clients require offline Hugging Face operation")
    for key in ("HF_HUB_CACHE", "HF_DATASETS_CACHE"):
        if not env.get(key) or not Path(env[key]).is_absolute():
            raise ValueError("explicit absolute Hugging Face cache paths are required")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class PreparedFile(StrictModel):
    path: str
    sha256: Digest

    @field_validator("path")
    @classmethod
    def absolute_path(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("prepared paths must be absolute")
        return value


class RuntimeSpec(StrictModel):
    """An already installed child environment; execution never resolves packages."""

    python: str
    identity: PreparedFile
    distributions: Annotated[list[str], Field(min_length=1)]
    env: dict[str, str]
    env_unset: list[str]
    assets: Annotated[list[PreparedFile], Field(min_length=1)]
    timeout_seconds: PositiveInt
    terminate_grace_seconds: PositiveInt

    @field_validator("python")
    @classmethod
    def absolute_python(cls, value: str) -> str:
        return PreparedFile.absolute_path(value)

    @model_validator(mode="after")
    def environment_contract(self) -> Self:
        validate_environment(self.env, self.env_unset)
        return self


class ResultMetadata(StrictModel):
    hw: str
    model: str
    model_prefix: str
    image: str
    framework: Literal["vllm"]
    precision: Literal["fp4"]
    spec_decoding: Literal["mtp"]
    tp: PositiveInt
    pp: Literal[1]
    dcp_size: Literal[1]
    pcp_size: Literal[1]
    ep: Literal[1]
    dp_attention: Literal[False]
    total_cpu_dram_gb: Annotated[int, Field(ge=0)]
    recipe_fingerprint: Digest

    def normalizer_env(self, concurrency: int) -> dict[str, str]:
        return {
            "RUNNER_TYPE": self.hw,
            "MODEL": self.model,
            "MODEL_PREFIX": self.model_prefix,
            "IMAGE": self.image,
            "FRAMEWORK": self.framework,
            "PRECISION": self.precision,
            "SPEC_DECODING": self.spec_decoding,
            "RECIPE_FINGERPRINT": self.recipe_fingerprint,
            "TP": str(self.tp),
            "PP_SIZE": str(self.pp),
            "DCP_SIZE": str(self.dcp_size),
            "PCP_SIZE": str(self.pcp_size),
            "EP_SIZE": str(self.ep),
            "DP_ATTENTION": "false",
            "CONC": str(concurrency),
            "IS_MULTINODE": "false",
            "DISAGG": "false",
            "KV_OFFLOADING": "none",
            "TOTAL_CPU_DRAM_GB": str(self.total_cpu_dram_gb),
        }


class AgentXSpec(StrictModel):
    schema_version: Literal[1]
    runtime: RuntimeSpec
    metadata: ResultMetadata
    concurrency: PositiveInt
    result_filename: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]
    tokenizer: str
    dataset_revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    dataset_loader: Literal["semianalysis_cc_traces_weka_062126"]
    dataset_repository: Literal["semianalysisai/cc-traces-weka-062126"]
    dataset_entries: Literal[393]
    duration_seconds: Literal[3600]
    warmup_requests_per_lane: Literal[10]
    warmup_grace_seconds: PositiveInt
    trace_idle_gap_cap_seconds: PositiveInt
    live_failed_request_threshold: Annotated[float, Field(ge=0, le=1)]
    failed_request_threshold: Literal[0.1]
    random_seed: Literal[42]
    required_server_metric_prefix: Literal["vllm:"]

    @model_validator(mode="after")
    def client_contract(self) -> Self:
        if self.tokenizer != self.metadata.model:
            raise ValueError("the pilot tokenizer must be the model identity")
        if "aiperf" not in self.runtime.distributions:
            raise ValueError("the prepared runtime must identify aiperf")
        cache = self.runtime.env.get("AIPERF_DATASET_MMAP_CACHE_DIR")
        if not cache or not Path(cache).is_absolute():
            raise ValueError("the prepared runtime requires an explicit absolute mmap cache base")
        return self


class EvalSpec(StrictModel):
    schema_version: Literal[1]
    runtime: RuntimeSpec
    metadata: ResultMetadata
    concurrency: PositiveInt
    task: PreparedFile
    task_name: Literal["gsm8k"]
    expected_documents: Literal[1319]
    max_length: Literal[16384]
    max_tokens: Literal[12288]
    minimum_score: Annotated[float, Field(ge=0, le=1)]
    # Independent prepared identities bind the full test split, not self-reported n-samples.
    # JSON object maps string doc_id to the harness doc_hash.
    document_identities: PreparedFile

    @model_validator(mode="after")
    def client_contract(self) -> Self:
        if "lm-eval" not in self.runtime.distributions:
            raise ValueError("the prepared runtime must identify lm-eval")
        return self
