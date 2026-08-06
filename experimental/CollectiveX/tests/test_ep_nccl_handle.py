#!/usr/bin/env python3
"""Torch-free tests for the nccl-ep single-handle contract: one handle per group, rebound per
problem shape. Two handles from the same group config resolve to the same LL parity signal slots
while advancing parity independently, which corrupts signalling (NVIDIA/nccl#2303).
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def _stub_modules():
    """Fake torch / nccl modules so `import ep_nccl` succeeds without the benchmark image."""
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    torch.int32 = "int32"
    torch.empty = lambda *a, **k: types.SimpleNamespace(shape=a[0] if a else ())
    torch.zeros = lambda *a, **k: types.SimpleNamespace(item=lambda: 7)
    torch.cuda = types.SimpleNamespace(synchronize=lambda: None)
    dist = types.ModuleType("torch.distributed")
    torch.distributed = dist

    ep = types.ModuleType("nccl.ep")
    for name in (
        "Algorithm", "CombineConfig", "CombineInputs", "CombineOutputs", "DispatchConfig",
        "DispatchInputs", "DispatchOutputs", "GroupConfig", "HandleConfig", "Layout",
        "LayoutInfo", "Tensor",
    ):
        setattr(ep, name, type(name, (), {"__init__": lambda self, *a, **k: None}))
    ep.Algorithm = types.SimpleNamespace(LOW_LATENCY="LL", HIGH_THROUGHPUT="HT")
    ep.Layout = types.SimpleNamespace(EXPERT_MAJOR="EM", FLAT="FLAT")
    core = types.ModuleType("nccl.core")
    pkg = types.ModuleType("nccl")
    pkg.ep, pkg.core = ep, core
    return {
        "torch": torch, "torch.distributed": dist,
        "nccl": pkg, "nccl.ep": ep, "nccl.core": core,
    }


sys.path[:0] = [str(ROOT), str(ROOT / "bench")]

# Import ep_nccl against the stubs, then withdraw them: a fake torch left in sys.modules makes
# genuinely torch-dependent modules (test_runtime, test_ll_oracle) error instead of skipping.
with mock.patch.dict(sys.modules, _stub_modules()):
    import ep_nccl  # noqa: E402

    sys.modules.pop("ep_nccl", None)


class FakeHandle:
    """Records every rebind so the tests can assert on the collective call pattern."""

    def __init__(self):
        self.updates = []
        self.destroyed = False

    def update(self, topk_idx, *, layout_info=None, stream=None):
        self.updates.append((topk_idx, layout_info))

    def destroy(self):
        self.destroyed = True


class FakeGroup:
    def __init__(self):
        self.created = 0
        self.handle = FakeHandle()

    def create_handle(self, layout, topk_idx, *, layout_info=None, config=None, stream=None):
        self.created += 1
        return self.handle


def backend(ll=True):
    """An NCCLEPBackend with just the fields _ensure_handle touches (no __init__, no GPU)."""
    b = object.__new__(ep_nccl.NCCLEPBackend)
    b._ll = ll
    b._layout = "EM" if ll else "FLAT"
    b._handle = None
    b._bound = None
    b._ep_group = FakeGroup()
    b.device = "cuda:0"
    b.num_local_experts = 4
    b.args = types.SimpleNamespace(hidden=16)
    b._t = lambda x: x
    b._stream = lambda: 0
    # create_buffer always runs before the first _ensure_handle, so the HT receive plane exists
    # by then; a list stands in for the tensor because `_t` is identity here.
    b._recv_x = list(range(64))
    return b


def problem(T):
    return types.SimpleNamespace(
        T=T, dispatch_x=f"x{T}", topk_idx=f"idx{T}", topk_weights=f"w{T}"
    )


class TestSingleHandle(unittest.TestCase):
    def test_one_handle_across_many_shapes(self):
        """Nine ladder rungs must still produce exactly one create_handle."""
        b = backend()
        for T in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            b._ensure_handle(problem(T))
        self.assertEqual(b._ep_group.created, 1)

    def test_shape_change_rebinds_and_repeat_does_not(self):
        """update() on a shape switch; no collective when the bound shape is re-entered."""
        b = backend()
        pa, pb = problem(1), problem(2)
        b._ensure_handle(pa)
        self.assertEqual(len(b._ep_group.handle.updates), 0)  # first bind is the create

        b._ensure_handle(pb)
        self.assertEqual(len(b._ep_group.handle.updates), 1)

        # Re-entering the bound problem must not enter a collective, or every iteration of the
        # timed loop gains a rank-synchronising step.
        for _ in range(8):
            b._ensure_handle(pb)
        self.assertEqual(len(b._ep_group.handle.updates), 1)

        # Returning to an earlier shape rebinds again (its cached namespace is reused).
        b._ensure_handle(pa)
        self.assertEqual(len(b._ep_group.handle.updates), 2)

    def test_every_problem_shares_the_one_handle(self):
        b = backend()
        handles = {id(b._ensure_handle(problem(T)).handle) for T in (1, 2, 4)}
        self.assertEqual(len(handles), 1)
        self.assertIs(b._ensure_handle(problem(1)).handle, b._handle)

    def test_ll_never_passes_layout_info_on_rebind(self):
        """The API forbids layout_info on create/update in LL mode."""
        b = backend(ll=True)
        b._ensure_handle(problem(1))
        b._ensure_handle(problem(2))
        self.assertEqual([info for _, info in b._ep_group.handle.updates], [None])

    def test_ll_gate_wrapper_is_built_once_per_handle(self):
        """LL applies the gate in combine, so its weights wrapper must be cached: building one
        per timed combine puts a torch resolve and an np.asarray inside `time_us`."""
        ll = backend(ll=True)
        pa = problem(1)
        h = ll._ensure_handle(pa)
        self.assertTrue(hasattr(h, "combine_weights_t"))
        self.assertEqual(h.combine_weights_t, "w1")
        # Re-entering the same problem reuses the handle and therefore the wrapper.
        self.assertIs(ll._ensure_handle(pa).combine_weights_t, h.combine_weights_t)

        ht = backend(ll=False)
        self.assertFalse(hasattr(ht._ensure_handle(problem(1)), "combine_weights_t"))

    def test_ht_combine_input_is_sliced_to_the_received_count(self):
        """HT combine's staging copy is sized by the tensor it is handed: the whole ladder-max
        receive plane put a rung-independent floor under it. LL keeps the full padded plane."""
        b = backend(ll=False)
        h = b._ensure_handle(problem(1))
        # 7 is what the stubbed `torch.zeros(...).item()` reports as the received count.
        self.assertEqual(h.count, 7)
        self.assertEqual(h.combine_in_t, list(range(7)))
        self.assertLess(len(h.combine_in_t), len(b._recv_x))

        ll = backend(ll=True)
        ll_h = ll._ensure_handle(problem(1))
        self.assertFalse(hasattr(ll_h, "combine_in_t"))

    def test_ht_rebind_carries_that_problems_counters(self):
        """HT re-runs the metadata exchange into the rebound problem's own counter tensors."""
        b = backend(ll=False)
        ha = b._ensure_handle(problem(1))
        hb = b._ensure_handle(problem(2))
        self.assertEqual(len(b._ep_group.handle.updates), 1)
        self.assertIs(b._ep_group.handle.updates[0][1], hb.layout_info)
        self.assertIsNot(ha.layout_info, hb.layout_info)
        self.assertEqual(hb.count, 7)  # re-read after the exchange

    def test_destroy_releases_the_handle_once(self):
        b = backend()
        b._ensure_handle(problem(1))
        handle = b._handle
        b._destroy_handles()
        self.assertTrue(handle.destroyed)
        self.assertIsNone(b._handle)
        self.assertIsNone(b._bound)
        b._destroy_handles()  # idempotent


if __name__ == "__main__":
    unittest.main()
