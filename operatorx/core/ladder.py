"""A case's layer: its op and args without the dimension its ladder sweeps.

Planning keeps a layer's ladder in one process; the harness memoizes a layer whose build fails.
"""
from __future__ import annotations

import json

# the case dimension a ladder sweeps; attention sweeps its batch (requests, query, context)
LADDER_ARG = {"gemm": "m", "moe": "tokens"}


def layer_key(op_type: str, args: dict) -> str:
    rest = dict(args)
    rest.pop(LADDER_ARG.get(op_type, "batch"), None)
    return json.dumps([op_type, rest], sort_keys=True)
