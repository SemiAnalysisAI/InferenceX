"""Execute patch output with fake NIXL/CUDA dependencies; no GPU proof implied."""

from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "nixl_patch", Path(__file__).with_name("patch_glm52_nixl_sync.py")
)
patch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(patch)

# A small package input for the installer. NIXL/CUDA calls are the observable seam;
# assertions below execute the transformed methods rather than inspect their text.
SOURCE = """class Manager:
    def __init__(self, backend, disaggregation_mode, backend_params):
        num_threads = 8 if disaggregation_mode == DisaggregationMode.PREFILL else 0
        agent_config = nixl_agent_config(
            backends=[],
            num_threads=num_threads,
            sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_STRICT,
        )
        self.agent = nixl_agent("test", agent_config)
        if num_threads > 0 and backend == "UCX":
            backend_params.setdefault("num_threads", str(num_threads))
        self.agent.create_backend(backend, backend_params)
        self.exceptions = {}

    def transfer_worker(self, queue: FastQueue, staging_buffer=None):
        while True:
            room = queue.get()
            try:
                if self.check_status(room) == KVPoll.Failed:
                    continue
                return self.agent.check_xfer_state(room)
            except Exception as error:
                self.exceptions[room] = error
                return "FAILED"

    def check_status(self, room):
        return "WAITING"
"""


def runtime(source, events):
    class Agent:
        def __init__(self, name, config):
            self.config = config

        def create_backend(self, backend, params):
            self.backend = backend
            self.params = params

        def check_xfer_state(self, handle):
            events.append(("progress", handle))
            return "DONE"

    def config(*, enable_prog_thread=True, **kwargs):
        return {"enable_prog_thread": enable_prog_thread, **kwargs}

    scope = {
        "DisaggregationMode": SimpleNamespace(PREFILL="prefill"),
        "nixl_agent_config": config,
        "nixl_agent": Agent,
        "nixl_thread_sync_t": SimpleNamespace(NIXL_THREAD_SYNC_STRICT="strict"),
        "FastQueue": object,
        "KVPoll": SimpleNamespace(Failed="FAILED"),
    }
    exec(source, scope)  # noqa: S102 - trusted controlled installer input/output.
    return scope["Manager"]


class NixlConfigurationTests(unittest.TestCase):
    def test_prefill_disables_both_thread_sources_and_retains_backend_options(self):
        manager = runtime(patch.transform_source(SOURCE), [])(
            "UCX", "prefill", {"num_threads": "8", "device_list": "mlx5_0"}
        )
        self.assertEqual(
            manager.agent.config,
            {
                "backends": [],
                "num_threads": 0,
                "enable_prog_thread": False,
                "sync_mode": "strict",
            },
        )
        self.assertEqual(
            manager.agent.params, {"num_threads": "0", "device_list": "mlx5_0"}
        )

    def test_decode_and_non_ucx_preserve_agent_configuration(self):
        for backend, role, count in [("UCX", "decode", 0), ("LIBFABRIC", "prefill", 8)]:
            with self.subTest(backend=backend, role=role):
                manager = runtime(patch.transform_source(SOURCE), [])(
                    backend, role, {"custom": "retained"}
                )
                self.assertEqual(
                    manager.agent.config,
                    {
                        "backends": [],
                        "num_threads": count,
                        "enable_prog_thread": True,
                        "sync_mode": "strict",
                    },
                )
                self.assertEqual(manager.agent.params, {"custom": "retained"})

    def test_device_is_bound_before_caller_driven_progress(self):
        events = []
        manager = runtime(patch.transform_source(SOURCE), events)("UCX", "prefill", {})
        manager.kv_args = SimpleNamespace(gpu_id=3)
        torch = SimpleNamespace(
            cuda=SimpleNamespace(
                set_device=lambda index: events.append(("device", index))
            )
        )
        with mock.patch.dict("sys.modules", {"torch": torch}):
            result = manager.transfer_worker(SimpleNamespace(get=lambda: "transfer"))
        self.assertEqual(result, "DONE")
        self.assertEqual(events, [("device", 3), ("progress", "transfer")])

    def test_device_binding_failure_prevents_progress(self):
        events = []
        manager = runtime(patch.transform_source(SOURCE), events)("UCX", "prefill", {})
        manager.kv_args = SimpleNamespace(gpu_id=3)
        torch = SimpleNamespace(
            cuda=SimpleNamespace(
                set_device=mock.Mock(side_effect=RuntimeError("device unavailable"))
            )
        )
        with mock.patch.dict("sys.modules", {"torch": torch}):
            result = manager.transfer_worker(SimpleNamespace(get=lambda: "transfer"))
        self.assertEqual(result, "FAILED")
        self.assertEqual(str(manager.exceptions["transfer"]), "device unavailable")
        self.assertEqual(events, [])


class SourceGuardTests(unittest.TestCase):
    def test_patch_writes_executable_output_and_repeated_application_is_noop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / patch.SOURCE_PATH
            path.parent.mkdir(parents=True)
            path.write_text(SOURCE)
            # Guard identities are inputs; use a small executable package fixture.
            with (
                mock.patch.object(
                    patch, "SOURCE_HASH", hashlib.sha256(SOURCE.encode()).hexdigest()
                ),
                mock.patch.object(
                    patch,
                    "PATCHED_HASH",
                    hashlib.sha256(patch.transform_source(SOURCE).encode()).hexdigest(),
                ),
            ):
                self.assertTrue(patch.patch_package(root))
                written = path.read_bytes()
                manager = runtime(written.decode(), [])("UCX", "prefill", {})
                self.assertFalse(manager.agent.config["enable_prog_thread"])
                self.assertEqual(manager.agent.params, {"num_threads": "0"})
                self.assertFalse(patch.patch_package(root))
                self.assertEqual(path.read_bytes(), written)

    def test_unknown_or_partial_source_is_rejected_without_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / patch.SOURCE_PATH
            path.parent.mkdir(parents=True)
            path.write_text("# unknown or partially patched runtime\n")
            original = path.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "unsupported SGLang source hash"):
                patch.patch_package(root)
            self.assertEqual(path.read_bytes(), original)

    def test_unexpected_output_is_rejected_before_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / patch.SOURCE_PATH
            path.parent.mkdir(parents=True)
            path.write_text(SOURCE)
            with (
                mock.patch.object(
                    patch, "SOURCE_HASH", hashlib.sha256(SOURCE.encode()).hexdigest()
                ),
                self.assertRaisesRegex(RuntimeError, "candidate patch output changed"),
            ):
                patch.patch_package(root)
            self.assertEqual(path.read_text(), SOURCE)


if __name__ == "__main__":
    unittest.main()
