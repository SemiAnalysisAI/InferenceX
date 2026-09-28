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
    # Where the case comes from: one "<checkpoint>/<role>" per layer that runs this op,
    # e.g. "deepseek-ai/DeepSeek-V4-Pro/q_a_proj" (a checkpoint id is "org/model", so the
    # role is what follows the last "/"). A shape shared by several models, or by several
    # roles in one, lists every pair; a shape from no model lists none. Equality/hash
    # ignore it - the same (type, args, backend) is the same op whichever model it came from.
    sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", MappingProxyType(dict(self.args)))
        object.__setattr__(self, "sources", tuple(self.sources))

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
