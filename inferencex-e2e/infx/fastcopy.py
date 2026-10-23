"""``copy.deepcopy`` without its per-object dispatch for the dicts and lists YAML and JSON build."""

from __future__ import annotations

import copy
from typing import Any


def deepcopy(value: Any) -> Any:
    """Return what ``copy.deepcopy(value)`` returns, shared references included."""
    return _copy(value, {})


def _copy(value: Any, memo: dict[int, Any]) -> Any:
    """Return a deep copy of ``value``, reusing copies already recorded in ``memo`` by object id."""
    cls = type(value)
    if cls is str or cls is int or cls is float or cls is bool or value is None:
        return value
    if (found := memo.get(id(value))) is not None:
        return found
    if cls is dict:
        result: Any = {}
        memo[id(value)] = result
        for key, item in value.items():
            result[_copy(key, memo)] = _copy(item, memo)
        return result
    if cls is list:
        result = []
        memo[id(value)] = result
        for item in value:
            result.append(_copy(item, memo))
        return result
    return copy.deepcopy(value, memo)
