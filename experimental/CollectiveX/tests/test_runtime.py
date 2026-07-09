#!/usr/bin/env python3
"""Focused tests for the standalone runtime helpers."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import re
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


# The per-node probe (runtime/probe.py) and the launcher gate
# (runtime/common.sh: cx_validate_network_profile_on_job) share an implicit string contract:
# the probe prints these markers, the launcher greps them back out to derive CX_SOCKET_IFNAME
# and CX_RDMA_LINK_LAYER. The patterns are duplicated here on purpose — the test fails if
# either side drifts, which is exactly the failure that slipped through when 5506c623 moved the
# probe into Python but left the emit statements behind, silently zeroing the marker count for
# every non-MNNVL multi-node leg.
SOCKET_MARKER = r"^\[collectivex-private\] socket-interface-selected=([A-Za-z][A-Za-z0-9_.-]{0,31})$"
LINK_MARKER = r"^\[collectivex-private\] rdma-link-layer=(roce|infiniband)$"
FAILURE_MARKER = (
    r"(socket-interface|rdma-(device|port))-[0-9]+="
    r"(missing|down|inactive|default-route-missing|gid-missing|gid-empty|"
    r"link-layer-missing|link-layer-invalid|link-layer-mixed)"
)


class NetworkProfileContract(unittest.TestCase):
    def _fabric(self, root: Path, *, state: str = "4: ACTIVE",
                link_layer: str = "Ethernet", gid: str = "fe80::1") -> None:
        net = root / "class" / "net" / "eth0"
        net.mkdir(parents=True)
        (net / "operstate").write_text("up\n")
        port = root / "class" / "infiniband" / "mlx5_0" / "ports" / "1"
        (port / "gids").mkdir(parents=True)
        (port / "state").write_text(state + "\n")
        (port / "link_layer").write_text(link_layer + "\n")
        (port / "gids" / "3").write_text(gid + "\n")

    def _run(self, root: Path, route: Path, socket_names: str = "eth0"):
        buffer = io.StringIO()
        rc = 0
        try:
            with contextlib.redirect_stdout(buffer):
                probe.validate_network_profile(socket_names, "mlx5_0:1", "3",
                                                sys_root=root, route_path=route)
        except SystemExit:
            rc = 1
        return rc, buffer.getvalue().splitlines()

    @staticmethod
    def _captures(pattern: str, lines: list) -> list:
        return [match.group(1) for line in lines
                for match in [re.match(pattern, line)] if match]

    def test_launcher_still_declares_the_marker_patterns(self) -> None:
        common = (RUNTIME / "common.sh").read_text()
        self.assertIn(SOCKET_MARKER, common)
        self.assertIn(LINK_MARKER, common)
        self.assertIn(FAILURE_MARKER, common)

    def test_healthy_fabric_emits_the_success_markers_the_launcher_extracts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fabric(root)
            rc, lines = self._run(root, root / "route")
            self.assertEqual(rc, 0)
            self.assertEqual(self._captures(SOCKET_MARKER, lines), ["eth0"])
            self.assertEqual(self._captures(LINK_MARKER, lines), ["roce"])

    def test_infiniband_link_layer_maps_to_the_launcher_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fabric(root, link_layer="InfiniBand")
            rc, lines = self._run(root, root / "route")
            self.assertEqual(rc, 0)
            self.assertEqual(self._captures(LINK_MARKER, lines), ["infiniband"])

    def test_socket_interface_resolves_from_default_route(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fabric(root)
            route = root / "route"
            route.write_text(
                "Iface Destination Gateway Flags RefCnt Use Metric Mask MTU Window IRTT\n"
                "eth0 00000000 00000000 0003 0 0 0 00000000 0 0 0\n"
            )
            rc, lines = self._run(root, route, socket_names="")
            self.assertEqual(rc, 0)
            self.assertEqual(self._captures(SOCKET_MARKER, lines), ["eth0"])

    def test_inactive_port_emits_a_launcher_recognized_failure_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fabric(root, state="1: DOWN")
            rc, lines = self._run(root, root / "route")
            self.assertEqual(rc, 1)
            failures = [line for line in lines if re.search(FAILURE_MARKER, line)]
            self.assertTrue(any("rdma-port-1=inactive" in line for line in failures), failures)

    def test_all_zero_gid_emits_gid_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fabric(root, gid="0000:0000:0000:0000:0000:0000:0000:0000")
            rc, lines = self._run(root, root / "route")
            self.assertEqual(rc, 1)
            self.assertTrue(any("rdma-port-1=gid-empty" in line for line in lines), lines)


if __name__ == "__main__":
    unittest.main()
