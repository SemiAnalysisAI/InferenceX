"""Shared test doubles. Import after the caller has put collectivex/bench on sys.path."""
from __future__ import annotations

import importlib
import sys
import types
from unittest import mock

import ep_backend


def stub_torch(**attrs):
    """Just enough torch for an adapter to import: dtype names as strings, `torch.compile` passing
    the function through (as a decorator or a call), and a WORLD group. `attrs` add or override."""
    torch = types.ModuleType("torch")
    for name in ("bfloat16", "float32", "int32", "int64", "float8_e4m3fn", "float8_e4m3fnuz"):
        setattr(torch, name, name)
    torch.compile = lambda fn=None, **_: fn if fn is not None else (lambda inner: inner)
    torch._dynamo = types.SimpleNamespace(config=types.SimpleNamespace(cache_size_limit=8))
    dist = types.ModuleType("torch.distributed")
    dist.group = types.SimpleNamespace(WORLD="world")
    torch.distributed = dist
    for name, value in attrs.items():
        setattr(torch, name, value)
    return {"torch": torch, "torch.distributed": dist}


def fresh_import(name, modules, also_pop=()):
    """A fresh import of `name` against the fake `modules`, which are withdrawn afterwards: a fake
    torch left in sys.modules makes genuinely torch-dependent tests error instead of skipping."""
    with mock.patch.dict(sys.modules, modules):
        for module in (*also_pop, name):
            sys.modules.pop(module, None)
        return importlib.import_module(name)


# Per-operation device cost in the stub clock (ms). Distinct primes so that any window
# reports a sum unique to the operations it actually brackets.
DISPATCH_MS = 3.0
STAGE_MS = 7.0
COMBINE_MS = 5.0


class _Clock:
    """Stub device clock; only the fake backend's operations advance it."""

    def __init__(self):
        self.now_ms = 0.0

    def advance(self, ms):
        self.now_ms += ms


class _Combined:
    """Just enough combined-output tensor for the chain's final-output capture."""

    def __init__(self, value):
        self.value = value
        self.cloned = False

    def fill_(self, value):
        self.value = value
        return self

    def clone(self):
        detached = _Combined(self.value)
        detached.cloned = True
        return detached


class _ChainBackend(ep_backend.EPBackend):
    """Records the call order and charges each operation a fixed slice of the stub clock."""

    name = "chain-stub"

    def __init__(self, stage_device_work=True, fp8_consume="native", precision="fp8",
                 dispatch_schedule=None):
        self.calls: list[str] = []
        self.consumed: list = []
        self.clock = _Clock()
        self.stage_device_work = stage_device_work
        self.fp8_consume = fp8_consume
        self.precision = precision
        self.device = "cpu"
        # Per-dispatch cost overrides, consumed in order; the constant cost applies after.
        self._dispatch_schedule = list(dispatch_schedule or [])

    def create_buffer(self, spec):  # pragma: no cover - unused
        raise NotImplementedError

    def dispatch(self, problem):
        self.calls.append("dispatch")
        cost = self._dispatch_schedule.pop(0) if self._dispatch_schedule else DISPATCH_MS
        self.clock.advance(cost)
        return types.SimpleNamespace(combine_input=None)

    def stage(self, problem, handle):
        self.calls.append("stage")
        self.clock.advance(STAGE_MS)
        handle.combine_input = "staged-by-stage"

    def combine(self, problem, handle):
        self.calls.append("combine")
        self.consumed.append(handle.combine_input)
        self.clock.advance(COMBINE_MS)
        return _Combined(handle.combine_input)

    def recv_tokens(self, handle):
        return 0

    def inspect_dispatch(self, problem, handle):  # pragma: no cover - unused
        return {}

    def combine_transformed(self, problem, handle, transformed):  # pragma: no cover
        return transformed
