from __future__ import annotations

import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True, eq=False)
class Op:
    type: str
    args: Mapping[str, Any]
    backend: str
    # Optional preset name for shapes that match a real model
    # (e.g. "dsv3", "llama3-8b"). Equality/hash ignore this — same
    # (type, args, backend) is the same op regardless of label.
    name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", MappingProxyType(dict(self.args)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Op):
            return NotImplemented
        return (
            self.type == other.type
            and dict(self.args) == dict(other.args)
            and self.backend == other.backend
        )

    def __hash__(self) -> int:
        # args may nest (e.g. gemm operand descriptors); canonical JSON is hashable.
        return hash((self.type, json.dumps(dict(self.args), sort_keys=True), self.backend))


@dataclass(frozen=True)
class OpSpec:
    type: str
    arg_schema: type
    description: str = ""
