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
    # Where the case comes from: `sources` are the checkpoint ids whose layers run this
    # op (several when models share a shape; empty for a shape from no model), `name`
    # its roles in them (e.g. "q_proj", "down_proj", "mlp"; one shape can be a q_proj in
    # one model and an o_proj in another). Equality/hash ignore both - the same
    # (type, args, backend) is the same op whichever model it came from.
    name: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", MappingProxyType(dict(self.args)))
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "name", tuple(self.name))

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
