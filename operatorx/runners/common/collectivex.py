"""CollectiveX modules OperatorX reuses as-is (the sibling collectivex/ tree)."""
from __future__ import annotations

import functools
import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3] / "collectivex"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # its dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


@functools.cache
def harness():
    return _load("collectivex_ep_harness", "bench/ep_harness.py")

