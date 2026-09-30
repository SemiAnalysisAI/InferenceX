"""Validate benchmark workflow inputs before fan-out, without rewriting them.

The matrix models own recipe fields and cross-field rules. This boundary adds
positive concurrency and permits older generators to omit parallelism fields.
Defaults are used only for validation; the original JSON reaches the workflow.
"""

import argparse
import json
import sys
from typing import Annotated, Literal

from pydantic import BaseModel, Field, ValidationError

from infx.matrix.validation import (
    MultiNodeAgenticMatrixEntry,
    MultiNodeMatrixEntry,
    SingleNodeAgenticMatrixEntry,
    SingleNodeMatrixEntry,
)


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


class _BatchFields(BaseModel):
    conc: list[Annotated[int, Field(gt=0)]] = Field(min_length=1)


class MultiNodeConfig(_BatchFields, MultiNodeMatrixEntry):
    """Fixed-sequence input to benchmark-multinode-tmpl.yml."""


class MultiNodeAgenticConfig(_BatchFields, MultiNodeAgenticMatrixEntry):
    """AgentX input to benchmark-multinode-tmpl.yml."""

    scenario_type: Literal["agentic-coding"] = Field(alias="scenario-type")


def _validate_rows(
    rows: object,
    *,
    path: str,
    multinode: bool | None = None,
    agentic: bool | None = None,
) -> None:
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a list of matrix rows")
    for index, row in enumerate(rows):
        location = f"{path}[{index}]"
        if not isinstance(row, dict):
            raise ValueError(f"{location}: expected a matrix object")
        is_multinode = "prefill" in row if multinode is None else multinode
        is_agentic = row.get("scenario-type") == "agentic-coding" if agentic is None else agentic
        if is_multinode:
            schema = MultiNodeAgenticConfig if is_agentic else MultiNodeConfig
        else:
            schema = AgenticConfig if is_agentic else SingleNodeConfig
        try:
            schema.model_validate(row, strict=True, by_alias=True, by_name=False)
        except ValidationError as error:
            raise ValueError(f"{location}: {error}") from error


def validate_matrix(matrix: object, *, plan: bool = False) -> None:
    """Check single-node and multinode workflow rows; preserve the input."""
    if not plan:
        _validate_rows(matrix, path="matrix")
        return
    if not isinstance(matrix, dict):
        raise ValueError("plan: expected an object")
    for family, multinode in (("single_node", False), ("multi_node", True)):
        groups = matrix.get(family, {})
        if not isinstance(groups, dict):
            raise ValueError(f"{family}: expected scenario groups")
        for group, rows in groups.items():
            _validate_rows(
                rows,
                path=f"{family}.{group}",
                multinode=multinode,
                agentic=group == "agentic",
            )
        prefix = "multinode_" if multinode else ""
        for suffix, agentic in (("evals", False), ("agentic_evals", True)):
            bucket = prefix + suffix
            _validate_rows(
                matrix.get(bucket, []),
                path=bucket,
                multinode=multinode,
                agentic=agentic,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Read a changelog plan instead of a flat matrix",
    )
    args = parser.parse_args()
    raw = sys.stdin.read()
    try:
        validate_matrix(json.loads(raw), plan=args.plan)
    except ValueError as error:
        parser.error(str(error))
    sys.stdout.write(raw)


if __name__ == "__main__":
    main()
