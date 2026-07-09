#!/usr/bin/env python3
"""Focused tests for the standalone runtime helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


RUNTIME = Path(__file__).resolve().parents[1] / "runtime"
sys.path.insert(0, str(RUNTIME))

import probe  # noqa: E402
import config  # noqa: E402
import stage  # noqa: E402


class ProbeTests(unittest.TestCase):
    def test_default_route_interface(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            route = Path(directory) / "route"
            route.write_text(
                "Iface Destination Gateway Flags RefCnt Use Metric Mask MTU Window IRTT\n"
                "eth9 00000000 00000000 0003 0 0 0 00000000 0 0 0\n"
            )
            self.assertEqual(probe.default_route_interface(route), "eth9")

    def test_prepare_cache_is_private_and_reusable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(probe.prepare_cache(directory))
            second = Path(probe.prepare_cache(directory))
            self.assertEqual(first, second)
            self.assertEqual(first.stat().st_mode & 0o777, 0o700)


class ConfigTests(unittest.TestCase):
    def test_operator_config_emits_allowlisted_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "operator.json"
            path.write_text(json.dumps({
                "schema_version": 1,
                "runners": {
                    "h100-dgxc": {
                        "partition": "gpu",
                        "account": "bench",
                        "squash_dir": directory,
                    }
                },
            }))
            path.chmod(0o600)
            read_fd, write_fd = os.pipe()
            stdout = sys.stdout
            try:
                sys.stdout = os.fdopen(write_fd, "w")
                config.operator_config(str(path), "h100-dgxc")
                sys.stdout.flush()
            finally:
                sys.stdout.close()
                sys.stdout = stdout
            payload = os.read(read_fd, 4096)
            os.close(read_fd)
            self.assertIn(b"CX_PARTITION\0gpu\0", payload)
            self.assertIn(b"CX_SQUASH_DIR\0" + directory.encode() + b"\0", payload)

    def test_canonical_policy_rejects_wrong_gpu_count(self) -> None:
        with self.assertRaises(SystemExit):
            config.canonical_policy("gb200", 2, 8, "nvidia:image", "amd:image", "commit")

    def test_network_mode_defaults_to_normal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shard.json"
            path.write_text('{"cases":[{}]}')
            config.network_mode(str(path))


class StageTests(unittest.TestCase):
    def test_create_copy_and_validate_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            target = root / "stage"
            (source / "runtime").mkdir(parents=True)
            (source / "runtime" / "common.sh").write_text("test")
            (source / "goal.md").write_text("private")
            args = type("Args", (), {"stage": str(target), "tag": "test-run"})
            stage.create_stage(args)
            copy_args = type(
                "Args", (), {"source": str(source), "target": str(target / "experimental" / "CollectiveX")}
            )
            stage.copy_repository(copy_args)
            self.assertTrue((target / "experimental" / "CollectiveX" / "runtime" / "common.sh").is_file())
            self.assertFalse((target / "experimental" / "CollectiveX" / "goal.md").exists())
            cleanup_args = type(
                "Args", (), {"root": str(target), "tag": "test-run"}
            )
            stage.validate_cleanup(cleanup_args)

    def test_common_network_mode_resolves_shard_from_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory) / "shard.json"
            shard.write_text('{"cases":[{"mode":"normal"}]}')
            common = RUNTIME / "common.sh"
            command = (
                "set -euo pipefail; "
                f"source {common!s}; "
                "CX_SHARD_FILE=shard.json; "
                f"cx_load_network_control_mode {directory!s}; "
                'test "$CX_MODE" = normal'
            )
            subprocess.run(["bash", "-c", command], check=True)


if __name__ == "__main__":
    unittest.main()
