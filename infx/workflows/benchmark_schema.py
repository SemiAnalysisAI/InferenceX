"""Validate single-node workflow inputs before fan-out, without rewriting them.

The matrix models own recipe fields and cross-field rules. This boundary adds
scalar concurrency and permits older generators to omit parallelism fields.
Defaults are used only for validation; the original JSON reaches the workflow.
"""

import argparse
import json
import sys
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from infx.matrix.validation import SingleNodeAgenticMatrixEntry, SingleNodeMatrixEntry


class _WorkflowFields(BaseModel):
    conc: int = Field(gt=0)
    pp: int = Field(default=1, gt=0)
    dcp_size: int = Field(default=1, alias="dcp-size", gt=0)
    pcp_size: int = Field(default=1, alias="pcp-size", gt=0)


class SingleNodeConfig(_WorkflowFields, SingleNodeMatrixEntry):
    """Fixed-sequence input to benchmark-tmpl.yml, before priority annotation."""


class AgenticConfig(_WorkflowFields, SingleNodeAgenticMatrixEntry):
    """AgentX input to benchmark-tmpl.yml, before priority annotation."""

    scenario_type: Literal["agentic-coding"] = Field(alias="scenario-type")


def _validate_rows(
    rows: object, *, path: str, mixed: bool = False, agentic: bool | None = None,
) -> None:
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a list of matrix rows")
    for index, row in enumerate(rows):
        location = f"{path}[{index}]"
        if not isinstance(row, dict):
            raise ValueError(f"{location}: expected a matrix object")
        # Manual generation includes multinode rows, routed to another template.
        if mixed and "prefill" in row:
            continue
        is_agentic = row.get("scenario-type") == "agentic-coding" if agentic is None else agentic
        schema = AgenticConfig if is_agentic else SingleNodeConfig
        try:
            schema.model_validate(row, strict=True, by_alias=True, by_name=False)
        except ValidationError as error:
            raise ValueError(f"{location}: {error}") from error


def validate_matrix(matrix: object, *, plan: bool = False) -> None:
    """Check only rows sent to the single-node template; preserve the input."""
    if not plan:
        _validate_rows(matrix, path="matrix", mixed=True)
        return
    if not isinstance(matrix, dict) or not isinstance(matrix.get("single_node", {}), dict):
        raise ValueError("plan: expected an object with single_node groups")
    for group, rows in matrix.get("single_node", {}).items():
        _validate_rows(rows, path=f"single_node.{group}", agentic=group == "agentic-coding")
    _validate_rows(matrix.get("evals", []), path="evals", agentic=False)
    _validate_rows(matrix.get("agentic_evals", []), path="agentic_evals", agentic=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="Read a changelog plan instead of a flat matrix")
    args = parser.parse_args()
    raw = sys.stdin.read()
    try:
        validate_matrix(json.loads(raw), plan=args.plan)
    except ValueError as error:
        parser.error(str(error))
    sys.stdout.write(raw)


if __name__ == "__main__":
    main()
