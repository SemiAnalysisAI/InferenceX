"""Strict workflow-to-runtime boundary for the first aggregate migration lane."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from infx.benchmarks.common import decode_json
from infx.srt_slurm.contracts import digest, resolve_reference
from infx.workflows.benchmark_schema import AgenticConfig


class SchedulingEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)

    priority: str
    queue_token: str = Field(alias="queue-token", min_length=1)
    node_count: Literal[1] = Field(alias="node-count")

    @field_validator("node_count", mode="before")
    @classmethod
    def integer_node_count(cls, value: Any) -> int:
        if type(value) is not int:
            raise ValueError("node-count must be an integer, not a boolean or coerced value")
        return value

    @field_validator("priority")
    @classmethod
    def finite_priority(cls, value: str) -> str:
        from decimal import Decimal, InvalidOperation

        try:
            priority = Decimal(value)
        except InvalidOperation as error:
            raise ValueError("priority must be a finite decimal") from error
        if not priority.is_finite():
            raise ValueError("priority must be a finite decimal")
        return value


class JobSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1]
    row: AgenticConfig
    scheduling: SchedulingEnvelope

    @model_validator(mode="after")
    def pilot_boundary(self) -> JobSpec:
        row = self.row
        if row.execution is None:
            raise ValueError("native execution reference is required")
        if (
            row.model_prefix != "dsv41flash"
            or row.model != "deepseek-ai/DeepSeek-V4.1-Flash"
            or row.framework != "vllm"
            or row.precision != "fp4"
            or row.runner != "cluster:h100-dgxc"
            or (row.tp, row.pp, row.dcp_size, row.pcp_size, row.ep) != (8, 1, 1, 1, 1)
            or row.dp_attn
            or row.kv_offloading != "none"
            or row.spec_decoding != "mtp"
            or row.duration != 3600
            or row.conc not in (1, 2, 4, 8, 16, 20, 24, 28)
            or row.router is not None
            or row.eval_suite not in (None, "")
        ):
            raise ValueError("execution is restricted to the Phase 1 H100 aggregate pilot")
        if row.run_eval and not row.eval_only:
            raise ValueError("throughput and real eval must be separate jobs")
        if row.eval_only and (row.eval_framework != "lm-eval" or row.conc != 28):
            raise ValueError("the pilot requires its representative c28 lm-eval job")
        return self

    @property
    def mode(self) -> str:
        return "eval" if self.row.eval_only else "throughput"

    def semantic_inputs(self) -> dict[str, Any]:
        row = self.row.model_dump(by_alias=True, exclude_none=True)
        for name in ("exp-name", "recipe-fingerprint", "run-eval", "eval-only"):
            row.pop(name, None)
        return {"schema_version": self.schema_version, "mode": self.mode, "row": row}

    @property
    def point_id(self) -> str:
        return digest(self.semantic_inputs())


def parse_job(raw: dict[str, Any], root: Path, scheduling: dict[str, Any] | None = None) -> JobSpec:
    """Validate separately supplied or priority-annotated workflow inputs without defaults."""
    row = dict(raw)
    annotation = {
        key: row.pop(key) for key in ("priority", "queue-token", "node-count") if key in row
    }
    if scheduling is not None:
        if any(key in annotation and annotation[key] != value for key, value in scheduling.items()):
            raise ValueError("conflicting workflow scheduling envelopes")
        annotation.update(scheduling)
    if "execution" not in row:
        raise ValueError("missing explicit native execution reference")
    row["execution"] = resolve_reference(row["execution"], root)
    return JobSpec.model_validate(
        {"schema_version": 1, "row": row, "scheduling": annotation}, strict=True
    )


def intent_id(repository: str, run_id: str, attempt: str, point_id: str) -> str:
    """A repeated workflow launch recovers one intent instead of allocating again."""
    if not repository or not run_id.isdecimal() or not attempt.isdecimal():
        raise ValueError("explicit GitHub execution identity is required")
    return digest({"repository": repository, "run": run_id, "attempt": attempt, "point": point_id})


def read_json(path: Path) -> dict[str, Any]:
    value = decode_json(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
