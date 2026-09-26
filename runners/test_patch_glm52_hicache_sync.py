"""CPU ordering checks for the exact runtime code inserted by the candidate.

The fake device exposes an explicit readiness gate so no GPU or timing-based
claim is needed. These tests do not establish that this caused the saved hang.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import logging
import tempfile
import threading
import unittest
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "hicache_patch", Path(__file__).with_name("patch_glm52_hicache_sync.py")
)
patch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(patch)


class DeviceEvent:
    def __init__(self, gate=None):
        self.gate = gate
        self.recorded = False

    def record(self):
        self.recorded = True

    def query(self):
        return not self.recorded or self.gate is None or self.gate.is_set()

    def synchronize(self):
        if self.gate is not None and not self.gate.wait(2):
            raise TimeoutError("test device readiness not released")


class Device:
    def __init__(self):
        self.local = threading.local()
        self.local.index = 3
        self.local.stream = "scheduler"

    def current_device(self):
        return self.local.index

    def set_device(self, index):
        self.local.index = index

    @contextlib.contextmanager
    def stream(self, stream):
        previous = getattr(self.local, "stream", None)
        self.local.stream = stream
        try:
            yield
        finally:
            self.local.stream = previous


def runtime_types(device):
    """Execute the same method bodies that the installer writes to SGLang."""
    scope = {
        "threading": threading,
        "Queue": Queue,
        "device_module": device,
        "logger": logging.getLogger("hicache-patch-test"),
        "CacheOperation": SimpleNamespace(merge_ops=lambda ops: tuple(ops)),
    }
    exec(  # noqa: S102 - execute the trusted runtime code that the patch inserts.
        "class LoadingEvent:\n"
        "    def __init__(self):\n"
        + patch.EVENT_INIT
        + patch.EVENT_METHODS
        + "class Counter:\n"
        + patch.PRODUCER_METHOD
        + patch.CONSUMER_METHOD
        + "class Controller:\n"
        + patch.DISPATCH_METHODS
        + patch.START_LOADING,
        scope,
    )
    return scope


class DirectLoadOrderingTests(unittest.TestCase):
    def controller(self, *, direct=True):
        device = Device()
        runtime = runtime_types(device)
        gate = threading.Event()
        counter = runtime["Counter"]()
        counter.events = [runtime["LoadingEvent"]() for _ in range(3)]
        for event in counter.events:
            event.start_event = DeviceEvent(gate)
            event.finish_event = DeviceEvent()
        counter.num_counters = 3
        counter.producer_index = -1
        counter.consumer_index = -1
        controller = runtime["Controller"]()
        controller.io_backend = "direct" if direct else "kernel"
        controller.layer_done_counter = counter
        controller.load_stream = "load"
        controller.load_queue = []
        controller.acks = []
        controller.submissions = []

        def enqueue(op, producer_id):
            controller.submissions.append(
                (op, device.current_device(), device.local.stream)
            )
            counter.events[producer_id].finish_event.record()
            controller.acks.append(op)

        controller._enqueue_load = enqueue
        controller._init_load_dispatch()
        self.addCleanup(gate.set)
        return controller, counter, gate, device

    def test_direct_load_returns_before_device_ready_and_consumer_waits(self):
        controller, counter, gate, _ = self.controller()
        controller.load_queue = ["kv", "indexer"]
        producer = controller.start_loading()
        self.assertEqual(producer, 0)
        self.assertEqual(controller.submissions, [])
        self.assertFalse(counter.events[0]._submission_ready.is_set())

        entered = threading.Event()
        finished = threading.Event()

        def consume():
            entered.set()
            counter.set_consumer(producer)
            finished.set()

        thread = threading.Thread(target=consume)
        thread.start()
        self.assertTrue(entered.wait(1))
        self.assertFalse(finished.wait(0.02))
        # Represents the scheduler reaching the next collective while the
        # helper waits for earlier device work. The inline baseline cannot.
        gate.set()
        self.assertTrue(finished.wait(1))
        thread.join(1)
        self.assertEqual(controller.submissions, [(("kv", "indexer"), 3, "load")])
        self.assertEqual(controller.acks, [("kv", "indexer")])
        self.assertTrue(counter.events[0].finish_event.recorded)

    def test_slot_reuse_waits_for_queued_burst_not_old_cuda_event(self):
        controller, counter, gate, _ = self.controller()
        for value in range(3):
            controller.load_queue = [value]
            self.assertEqual(controller.start_loading(), value)
        self.assertTrue(counter.events[0].finish_event.query())
        finished = threading.Event()

        def reuse():
            controller.load_queue = [3]
            controller.start_loading()
            finished.set()

        thread = threading.Thread(target=reuse)
        thread.start()
        self.assertFalse(finished.wait(0.02))
        gate.set()
        self.assertTrue(finished.wait(1))
        thread.join(1)
        controller._drain_load_dispatch()
        self.assertEqual(controller.acks, [(0,), (1,), (2,), (3,)])

    def test_submission_error_unblocks_consumer_and_prevents_stale_success(self):
        controller, counter, gate, _ = self.controller()
        failure = RuntimeError("copy enqueue failed")
        controller._enqueue_load = mock.Mock(side_effect=failure)
        controller.load_queue = ["kv"]
        producer = controller.start_loading()
        gate.set()
        self.assertTrue(counter.events[producer]._submission_ready.wait(1))
        with self.assertRaisesRegex(RuntimeError, "HiCache load submission"):
            counter.set_consumer(producer)
        self.assertEqual(controller.acks, [])
        with self.assertRaisesRegex(RuntimeError, "HiCache load submission"):
            controller.start_loading()
        self.assertIs(controller._load_dispatch_error, failure)

    def test_device_binding_error_also_unblocks_consumer(self):
        controller, counter, gate, device = self.controller()
        device.set_device = mock.Mock(side_effect=RuntimeError("wrong device"))
        controller.load_queue = ["kv"]
        producer = controller.start_loading()
        gate.set()
        self.assertTrue(counter.events[producer]._submission_ready.wait(1))
        with self.assertRaisesRegex(RuntimeError, "HiCache load submission"):
            counter.set_consumer(producer)
        self.assertEqual(controller.submissions, [])

    def test_kernel_load_remains_inline(self):
        controller, counter, _, _ = self.controller(direct=False)
        controller.load_queue = ["kv"]
        self.assertEqual(controller.start_loading(), 0)
        self.assertIsNone(controller._load_dispatch_queue)
        self.assertEqual(controller.submissions, [(("kv",), 3, "scheduler")])
        counter.set_consumer(0)
        counter.set_consumer(-1)
        self.assertEqual(counter.consumer_index, -1)

    def test_transformed_kernel_body_records_readiness_after_index_copy(self):
        # The kernel backend's index normalization issues an async H2D copy.
        # A record-before-normalize edit would let load_stream read stale indices.
        source = """class Hybrid(Controller):
    def start_loading(self) -> int:
        if not self.load_queue:
            return -1
        producer_id = self.layer_done_counter.update_producer()
        op = CacheOperation.merge_ops(self.load_queue)
        host_indices, device_indices = self.move_hybrid_indices(op)
        self.load_queue.clear()
        producer_event = self.layer_done_counter.events[producer_id]
        producer_event.start_event.record()
        self.copy_and_ack(host_indices, device_indices)
        return producer_id

    def next_method(self):
        pass
"""
        controller, counter, _, _ = self.controller(direct=False)
        scope = {
            "Controller": type(controller),
            "CacheOperation": SimpleNamespace(merge_ops=lambda ops: tuple(ops)),
        }
        exec(patch.transform_source(source, hybrid=True), scope)  # noqa: S102 - trusted fixture.
        controller.__class__ = scope["Hybrid"]
        del controller._enqueue_load
        order = []
        controller.move_hybrid_indices = lambda op: (order.append("indices"), op)
        counter.events[0].start_event.record = lambda: order.append("ready")
        controller.copy_and_ack = lambda *_: order.append("copies")
        controller.load_queue = ["kv"]
        controller.start_loading()
        self.assertEqual(order, ["indices", "ready", "copies"])

    def test_reset_drain_prevents_late_ack_after_clear(self):
        controller, _, gate, _ = self.controller()
        controller.load_queue = ["kv"]
        controller.start_loading()
        finished = threading.Event()

        def reset():
            controller._drain_load_dispatch()
            controller.acks.clear()
            finished.set()

        thread = threading.Thread(target=reset)
        thread.start()
        self.assertFalse(finished.wait(0.02))
        gate.set()
        self.assertTrue(finished.wait(1))
        thread.join(1)
        self.assertEqual(controller.acks, [])


class SourceGuardTests(unittest.TestCase):
    def test_unknown_source_is_rejected_without_changing_either_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in patch.SOURCE_HASHES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("unknown source\n")
            with self.assertRaisesRegex(RuntimeError, "unsupported SGLang source hash"):
                patch.patch_package(root)
            self.assertTrue(
                all(
                    (root / name).read_text() == "unknown source\n"
                    for name in patch.SOURCE_HASHES
                )
            )

    def test_already_applied_hash_pair_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hashes = {}
            for name in patch.SOURCE_HASHES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("already patched\n")
                hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            with mock.patch.object(patch, "PATCHED_HASHES", hashes):
                self.assertFalse(patch.patch_package(root))


if __name__ == "__main__":
    unittest.main()
