#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Candidate direct HiCache load-enqueue repair for SGLang 211ee642 only.

Keeps CPU index readiness off the scheduler thread while retaining the existing
copy operations. Adapted from the mechanism proposed in sgl-project/sglang#34515;
this backport has not yet been validated on the campaign's GPU runtime.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

SOURCE_HASHES = {
    "srt/managers/cache_controller.py": "e8d0755d9c2cbc97a5d78a300a6abbe537664c9a6b91366efc903de20c5b5919",
    "srt/mem_cache/hybrid_cache/hybrid_cache_controller.py": "cba2b0db471bf4b23e69a8dd04629a8aed779072cbd198dbe3a7bca3ff01f7a3",
}
PATCHED_HASHES = {
    "srt/managers/cache_controller.py": "95fb2f90a6314c42f6b4705c8a3c4699fc12ea66306ffa13506811519c938028",
    "srt/mem_cache/hybrid_cache/hybrid_cache_controller.py": "368e935976d44956e74bec168571d07b5166f3a7d674adc9a2c09a377ea59c67",
}

EVENT_INIT = """        # A CUDA wait on an event that has never been recorded is a no-op.
        # Publish submission before a consumer may enqueue its device wait.
        self._submission_ready = threading.Event()
        self._submission_ready.set()
        self._submission_error = None
"""

EVENT_METHODS = """    def reset_submission(self):
        self._submission_error = None
        self._submission_ready.clear()

    def mark_submitted(self, error=None):
        self._submission_error = error
        self._submission_ready.set()

    def wait_until_submitted(self):
        self._submission_ready.wait()
        if self._submission_error is not None:
            raise RuntimeError("HiCache load submission failed") from self._submission_error

"""

PRODUCER_METHOD = """    def update_producer(self):
        producer_index = (self.producer_index + 1) % self.num_counters
        producer_event = self.events[producer_index]
        # A queued burst may still expose the preceding rotation's CUDA event.
        producer_event.wait_until_submitted()
        assert producer_event.finish_event.query(), (
            "Producer finish event should be ready before being reused."
        )
        producer_event.reset_submission()
        self.producer_index = producer_index
        return producer_index

"""

CONSUMER_METHOD = """    def set_consumer(self, index: int):
        if index >= 0:
            self.events[index].wait_until_submitted()
        self.consumer_index = index

"""

DISPATCH_METHODS = """    def _init_load_dispatch(self):
        self._load_dispatch_error = None
        self._load_dispatch_queue = None
        if self.io_backend == "direct":
            # Device selection is thread-local. Capture this rank's device now.
            self._load_dispatch_device = device_module.current_device()
            self._load_dispatch_queue = Queue()
            self._load_dispatch_thread = threading.Thread(
                target=self._load_dispatch_loop,
                name="hicache-direct-load",
                daemon=True,
            )
            self._load_dispatch_thread.start()
            logger.info("HiCache direct load enqueue repair: enabled (211ee642)")

    def _check_load_dispatch_error(self):
        if self._load_dispatch_error is not None:
            raise RuntimeError("HiCache load submission failed") from self._load_dispatch_error

    def _submit_load(self, op, producer_id):
        producer_event = self.layer_done_counter.events[producer_id]
        try:
            self._check_load_dispatch_error()
            self._enqueue_load(op, producer_id)
        except BaseException as error:
            self._load_dispatch_error = error
            producer_event.mark_submitted(error)
            raise
        else:
            producer_event.mark_submitted()

    def _load_dispatch_loop(self):
        while True:
            op, producer_id = self._load_dispatch_queue.get()
            producer_event = self.layer_done_counter.events[producer_id]
            try:
                self._check_load_dispatch_error()
                device_module.set_device(self._load_dispatch_device)
                # Allocator/index merge work was issued on the scheduler stream.
                # Wait here, not on the scheduler that must reach DP collectives.
                producer_event.start_event.synchronize()
                with device_module.stream(self.load_stream):
                    self._submit_load(op, producer_id)
            except BaseException as error:
                self._load_dispatch_error = error
                producer_event.mark_submitted(error)
                logger.exception("HiCache direct load submission failed")
            finally:
                self._load_dispatch_queue.task_done()

    def _drain_load_dispatch(self):
        if self._load_dispatch_queue is not None:
            self._load_dispatch_queue.join()
        self._check_load_dispatch_error()

"""

START_LOADING = """    def start_loading(self) -> int:
        self._check_load_dispatch_error()
        if not self.load_queue:
            return -1
        producer_id = self.layer_done_counter.update_producer()
        op = CacheOperation.merge_ops(self.load_queue)
        self.load_queue.clear()
        producer_event = self.layer_done_counter.events[producer_id]
        if self._load_dispatch_queue is not None:
            producer_event.start_event.record()
            self._load_dispatch_queue.put((op, producer_id))
        else:
            self._submit_load(op, producer_id)
        return producer_id

"""


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError("unsupported SGLang source: patch anchor is not unique")
    return source.replace(old, new, 1)


def replace_method(source: str, name: str, replacement: str) -> str:
    start = source.index(f"    def {name}(")
    end = source.index("\n    def ", start + 1) + 1
    return source[:start] + replacement + source[end:]


def transform_source(source: str, *, hybrid: bool) -> str:
    """Transform already-identified source; package-level hash checks run first."""
    start = source.index("    def start_loading(")
    end = source.index("\n    def ", start + 1) + 1
    original = source[start:end]
    move = original.index("        host_indices, device_indices")
    body = original[move:]
    body = replace_once(body, "        self.load_queue.clear()\n", "")
    body = replace_once(
        body,
        "        producer_event.start_event.record()\n",
        "        if self._load_dispatch_queue is None:\n"
        "            producer_event.start_event.record()\n",
    )
    body = replace_once(body, "        return producer_id\n", "")
    enqueue = "    def _enqueue_load(self, op, producer_id):\n" + body
    source = source[:start] + START_LOADING + enqueue + source[end:]
    if hybrid:
        return source
    source = replace_once(
        source,
        "        self.start_event = device_module.Event()  # start event on controller stream\n",
        "        self.start_event = device_module.Event()  # start event on controller stream\n"
        + EVENT_INIT,
    )
    source = replace_once(
        source,
        "    def complete(self, layer_index: int):\n",
        EVENT_METHODS + "    def complete(self, layer_index: int):\n",
    )
    source = replace_method(source, "update_producer", PRODUCER_METHOD)
    source = replace_method(source, "set_consumer", CONSUMER_METHOD)
    source = replace_once(
        source,
        "        self.load_stream = device_module.Stream()\n",
        "        self.load_stream = device_module.Stream()\n"
        "        self._init_load_dispatch()\n",
    )
    source = replace_once(
        source,
        "    def reset(self):\n        self.storage_stop_event.set()",
        DISPATCH_METHODS + "    def reset(self):\n"
        "        self._drain_load_dispatch()\n"
        "        self.storage_stop_event.set()",
    )
    return source


def patch_package(package_root: Path) -> bool:
    sources = {name: (package_root / name).read_text() for name in SOURCE_HASHES}
    digests = {
        name: hashlib.sha256(source.encode()).hexdigest()
        for name, source in sources.items()
    }
    if digests == PATCHED_HASHES:
        return False
    # Verify both files before changing either. Unknown or partially patched
    # runtimes must fail before engine startup, not silently receive half a fix.
    for name, source in sources.items():
        digest = digests[name]
        if digest != SOURCE_HASHES[name]:
            raise RuntimeError(f"unsupported SGLang source hash for {name}: {digest}")
    patched = {
        name: transform_source(source, hybrid="hybrid_cache" in name)
        for name, source in sources.items()
    }
    for name, source in patched.items():
        compile(source, str(package_root / name), "exec")
        if hashlib.sha256(source.encode()).hexdigest() != PATCHED_HASHES[name]:
            raise RuntimeError(f"candidate patch output changed for {name}")
    for name, source in patched.items():
        (package_root / name).write_text(source)
    return True


def main(argv: list[str]) -> int:
    if len(argv) > 2:
        print(f"Usage: {argv[0]} [SGLANG_PACKAGE_ROOT]", file=sys.stderr)
        return 2
    try:
        if len(argv) == 2:
            root = Path(argv[1])
        else:
            spec = importlib.util.find_spec("sglang")
            if spec is None or not spec.submodule_search_locations:
                raise RuntimeError("sglang is not installed")
            root = Path(next(iter(spec.submodule_search_locations)))
        changed = patch_package(root)
    except (OSError, RuntimeError, ValueError, SyntaxError) as error:
        print(f"ERROR: HiCache candidate patch rejected: {error}", file=sys.stderr)
        return 1
    state = "Patched" if changed else "Already patched"
    print(f"{state} SGLang 211ee642 direct HiCache load enqueue candidate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
