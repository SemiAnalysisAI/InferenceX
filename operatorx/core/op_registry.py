from __future__ import annotations

from typing import Sequence

from operatorx.core import parallel
from operatorx.core.op import Op, OpSpec

_REGISTRY: dict[str, OpSpec] = {}


def register(spec: OpSpec) -> None:
    if spec.type in _REGISTRY:
        raise ValueError(f"op type {spec.type!r} already registered")
    _REGISTRY[spec.type] = spec


def get(type: str) -> OpSpec:
    if type not in _REGISTRY:
        raise KeyError(f"unknown op type: {type!r}")
    return _REGISTRY[type]


def all_ops() -> Sequence[OpSpec]:
    return tuple(_REGISTRY.values())


def validate(op: Op) -> Op:
    """Check args against the schema; "parallel" is checked against the op type's axes."""
    spec = get(op.type)
    args = dict(op.args)
    parallel.check(args.pop("parallel", None), spec.parallel_axes)
    try:
        spec.arg_schema(**args)
    except TypeError as e:
        raise ValueError(f"op {op.type!r} args don't match schema: {e}") from e
    return op
