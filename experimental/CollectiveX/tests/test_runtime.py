#!/usr/bin/env python3
"""Focused tests for the standalone runtime helpers."""

from __future__ import annotations

import argparse
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
BENCH = Path(__file__).resolve().parents[1] / "bench"
sys.path.insert(0, str(RUNTIME))
sys.path.insert(0, str(BENCH))

import probe  # noqa: E402
import config  # noqa: E402
import stage  # noqa: E402
import ep_harness  # noqa: E402  (stdlib-only at module top)


# configs/platform_config.json is shared by matrix scheduling, operator/network
# loading, and backend builds.
class PlatformRegistryTests(unittest.TestCase):
    REGISTRY = RUNTIME.parent / "configs" / "platform_config.json"
    NETWORK_FIELDS = {
        "socket_ifname", "rdma_devices", "ib_gid_index",
        "rdma_service_level", "rdma_traffic_class", "rail_isolated",
    }

    def test_every_platform_entry_is_complete_and_typed(self) -> None:
        platforms = json.loads(self.REGISTRY.read_text())["platforms"]
        self.assertTrue(platforms)
        for name, entry in platforms.items():
            with self.subTest(sku=name):
                for field in (
                    "arch", "product", "image", "image_platform",
                    "scale_up_transport", "launcher",
                ):
                    self.assertIsInstance(entry[field], str)
                    self.assertTrue(entry[field])
                for field in ("gpus_per_node", "scale_up_domain"):
                    self.assertIsInstance(entry[field], int)
                    self.assertGreater(entry[field], 0)
                self.assertTrue(entry["backends"])
                for degrees in entry["backends"].values():
                    self.assertTrue(degrees)
                    self.assertLessEqual(set(degrees), {8, 16})
                self.assertLessEqual(
                    set(entry.get("network", {})), self.NETWORK_FIELDS
                )
                # Fabric provenance: each cluster records its scale-out NIC and
                # switch so same-GPU clusters on different fabrics stay distinct.
                fabric = entry["fabric"]
                self.assertEqual(set(fabric), {"nic", "switch"})
                for value in fabric.values():
                    self.assertIsInstance(value, str)
                    self.assertTrue(value)
                self.assertRegex(entry["arch"], r"^(sm|gfx)\d+$")
                self.assertRegex(entry["image"], r"^[A-Za-z0-9._/-]+:[A-Za-z0-9._-]+$")
                self.assertIn(entry["image_platform"], {"linux/amd64", "linux/arm64"})


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
            self.assertIn(b"COLLX_PARTITION\0gpu\0", payload)
            self.assertIn(b"COLLX_SQUASH_DIR\0" + directory.encode() + b"\0", payload)
            self.assertIn(b"COLLX_IMAGE\0lmsysorg/sglang:v0.5.11-cu130\0", payload)
            self.assertIn(b"COLLX_IMAGE_PLATFORM\0linux/amd64\0", payload)

    def _emit_registry_only(self, runner: str) -> bytes:
        read_fd, write_fd = os.pipe()
        stdout = sys.stdout
        try:
            sys.stdout = os.fdopen(write_fd, "w")
            config.operator_config("-", runner)
            sys.stdout.flush()
        finally:
            sys.stdout.close()
            sys.stdout = stdout
        payload = os.read(read_fd, 4096)
        os.close(read_fd)
        return payload

    def test_operator_config_registry_only_emits_tracked_baseline(self) -> None:
        # "-" = no operator document: the registry's per-SKU operator block is
        # the tracked baseline (plus its network overlay where present).
        payload = self._emit_registry_only("h200-dgxc")
        self.assertIn(b"COLLX_PARTITION\0main\0", payload)
        self.assertIn(b"COLLX_SQUASH_DIR\0/home/sa-shared/containers\0", payload)
        self.assertIn(b"COLLX_RDMA_DEVICES\0", payload)

    def test_operator_config_registry_only_emits_image_for_secret_fed_sku(self) -> None:
        # A SKU without tracked operator settings still gets its public image
        # configuration; private scheduler values can arrive through the overlay.
        payload = self._emit_registry_only("mi325x")
        self.assertIn(b"COLLX_IMAGE\0rocm/sgl-dev:sglang-0.5.14-rocm720-mi35x-mori-0701\0", payload)
        self.assertIn(b"COLLX_IMAGE_PLATFORM\0linux/amd64\0", payload)


class StageTests(unittest.TestCase):
    def test_create_copy_and_validate_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            target = root / "stage"
            (source / "runtime").mkdir(parents=True)
            (source / "runtime" / "common.sh").write_text("test")
            (source / "goal.md").write_text("private")
            (source / ".shards").mkdir()
            (source / ".shards" / "leg.json").write_text("{}")
            args = type("Args", (), {"stage": str(target)})
            stage.create_stage(args)
            copy_args = type(
                "Args", (), {"source": str(source), "target": str(target / "experimental" / "CollectiveX")}
            )
            stage.copy_repository(copy_args)
            staged = target / "experimental" / "CollectiveX"
            self.assertTrue((staged / "runtime" / "common.sh").is_file())
            self.assertFalse((staged / ".shards").exists())
            self.assertFalse((staged / "goal.md").exists())
            cleanup_args = type("Args", (), {"root": str(target)})
            stage.validate_cleanup(cleanup_args)

# The per-node probe (runtime/probe.py) and the launcher gate
# (runtime/common.sh: collx_validate_network_profile_on_job) share an implicit string contract:
# the probe prints these markers, the launcher greps them back out to derive COLLX_SOCKET_IFNAME
# and COLLX_RDMA_LINK_LAYER. The patterns are duplicated here on purpose — the test fails if
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


class StageContract(unittest.TestCase):
    # runtime/common.sh drives runtime/stage.py purely by literal subcommand name and positional
    # argv — there are no optional flags. That argv shape is a string contract: a subcommand or
    # flag the launcher passes but stage.py does not declare fails with "unrecognized arguments"
    # and aborts the leg at repository-stage. This extracts every stage.py call out of common.sh
    # and proves stage.py's parser accepts it — the guard that would have caught the --allow-*
    # flags surviving on the callers after they were dropped from stage.py's argparse.
    @staticmethod
    def _invocations(text: str) -> list:
        calls = []
        for line in text.splitlines():
            if "stage.py" not in line or line.lstrip().startswith("#"):
                continue
            subcommand, flags = None, []
            for raw in line.split("stage.py", 1)[1].split():
                token = raw.strip('"').strip("'")
                if token.startswith("--"):
                    flags.append(token.split("=", 1)[0])
                elif subcommand is None and token and not token.startswith(("$", "${")):
                    subcommand = token
            if subcommand:
                calls.append((subcommand, flags))
        return calls

    def test_launcher_only_invokes_declared_subcommands_and_flags(self) -> None:
        invocations = self._invocations((RUNTIME / "common.sh").read_text())
        self.assertGreaterEqual(len(invocations), len(stage.SPECS), invocations)
        parser = stage.build_parser()
        for subcommand, flags in invocations:
            self.assertIn(subcommand, stage.SPECS, subcommand)
            argv = [subcommand] + ["x"] * len(stage.SPECS[subcommand]) + flags
            with contextlib.redirect_stderr(io.StringIO()):
                try:
                    parser.parse_args(argv)
                except SystemExit:
                    self.fail(f"common.sh invokes stage.py with an argv shape it rejects: {argv}")

    def test_contract_test_has_teeth(self) -> None:
        # A flag common.sh must never pass has to be rejected by the parser — this is the exact
        # failure (unrecognized arguments: --allow-parent-owner) the reconcile removed.
        parser = stage.build_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["validate-stage-path", "x", "x", "x", "--allow-parent-owner"])


# config.py case-args is the single case→invocation codec: collx_run_shard decodes one
# null-delimited argv per case and hands it verbatim to bench/run_ep.py. Parse the
# emitted argv with the same parser shape run_ep builds so the two sides cannot
# drift — a flag the codec emits but run_ep does not declare (or vice versa) fails
# here instead of on a GPU allocation.
class CaseArgvContract(unittest.TestCase):
    CASE = {
        "backend": "deepep-v2", "mode": "normal", "precision": "bf16",
        "phase": "decode",
        "routing": "uniform", "ep": 16, "nodes": 2, "gpus_per_node": 8,
        "scale_up_domain": 8, "scope": "scale-out",
        "scale_up_transport": "nvlink", "scale_out_transport": "rdma",
        "transport": "nvlink-rdma", "topology_class": "h200-nvlink-rdma",
        "hidden": 7168, "topk": 8, "experts": 256, "seed": 67,
        "ladder": "1 2 4", "timing": "8:256:32",
        "case_id": "h200-dgxc-deepep-v2-deepseek-v3-normal-decode-ep16-uniform-bf16",
        "suite": "ep-core", "workload": "deepseek-v3",
    }

    @staticmethod
    def _run_ep_parser() -> argparse.ArgumentParser:
        # Mirror of the parser bench/run_ep.py builds in main().
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--backend", required=True, choices=["deepep-v2", "mori"]
        )
        ep_harness.add_common_args(parser)
        return parser

    def _decode(self, stdout: bytes) -> list:
        parts = stdout.split(b"\0")
        self.assertEqual(parts[-1], b"")
        return [part.decode() for part in parts[:-1]]

    def _case_argv(self, placement: list) -> list:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shard.json"
            path.write_text(json.dumps({"version": 1, "cases": [self.CASE]}))
            result = subprocess.run(
                [sys.executable, str(RUNTIME / "config.py"), "case-args",
                 str(path), "0", "h200-dgxc", "TS", *placement],
                capture_output=True, check=True,
            )
        return self._decode(result.stdout)

    def test_case_args_round_trips_through_the_run_ep_parser(self) -> None:
        argv = self._case_argv(["16", "2", "8", "8"])
        args = self._run_ep_parser().parse_args(argv)
        self.assertEqual(
            (args.backend, args.mode, args.phase, args.routing, args.scope),
            ("deepep-v2", "normal", "decode", "uniform", "scale-out"),
        )
        self.assertEqual(args.precision, "bf16")
        self.assertEqual((args.hidden, args.topk, args.experts), (7168, 8, 256))
        self.assertEqual((args.gpus_per_node, args.scale_up_domain), (8, 8))
        self.assertEqual(args.tokens_ladder, "1 2 4")
        self.assertEqual(args.scale_out_transport, "rdma")
        self.assertEqual(args.case_id, self.CASE["case_id"])
        self.assertEqual(args.version, 1)
        self.assertEqual(args.seed, self.CASE["seed"])
        self.assertEqual((args.iters, args.trials, args.warmup), (8, 256, 32))
        self.assertEqual(args.out, "results/h200-dgxc_deepep-v2_bf16_decode_TS-c000.json")

    def test_case_args_fails_closed_on_placement_mismatch(self) -> None:
        with self.assertRaises(subprocess.CalledProcessError):
            self._case_argv(["8", "1", "8", "8"])


# logical_byte_provenance is where FP8 changes MEASUREMENT semantics (asymmetric
# per-direction byte counts), so its arithmetic and guards are pinned here on CPU.
class LogicalByteProvenanceTests(unittest.TestCase):
    def test_bf16_default_is_two_bytes_per_value_no_scales(self) -> None:
        got = ep_harness.logical_byte_provenance(logical_copies=10, hidden=7168)
        self.assertEqual(got["activation_data_bytes"], 10 * 7168 * 2)
        self.assertEqual(got["scale_bytes"], 0)
        self.assertEqual(got["total_logical_bytes"], 10 * 7168 * 2)

    def test_fp8_blockwise_dispatch_is_one_byte_plus_per_copy_scales(self) -> None:
        # DeepEP FP8 dispatch: 1 byte/value + ceil(hidden/128)*4 FP32 scale bytes/copy.
        scale_per_copy = ((7168 + 127) // 128) * 4  # 224
        got = ep_harness.logical_byte_provenance(
            logical_copies=10, hidden=7168, value_bytes=1,
            scale_bytes_per_copy=scale_per_copy,
        )
        self.assertEqual(got["activation_data_bytes"], 10 * 7168)
        self.assertEqual(got["scale_bytes"], 10 * scale_per_copy)
        self.assertEqual(got["total_logical_bytes"], 10 * 7168 + 10 * scale_per_copy)

    def test_fp8_direct_cast_dispatch_is_one_byte_no_scales(self) -> None:
        # MoRI's scale-free e4m3 cast: 1 byte/value, no scale payload.
        got = ep_harness.logical_byte_provenance(
            logical_copies=10, hidden=7168, value_bytes=1, scale_bytes_per_copy=0,
        )
        self.assertEqual(got["activation_data_bytes"], 10 * 7168)
        self.assertEqual(got["scale_bytes"], 0)

    def test_roundtrip_is_the_per_field_sum_of_dispatch_and_combine(self) -> None:
        # run_sweep assembles the roundtrip as the per-field sum of an FP8 dispatch and a
        # BF16 combine; the direction bytes differ, so it is not 2x a single direction.
        dispatch = ep_harness.logical_byte_provenance(
            logical_copies=10, hidden=7168, value_bytes=1, scale_bytes_per_copy=224,
        )
        combine = ep_harness.logical_byte_provenance(logical_copies=10, hidden=7168)
        roundtrip = {field: dispatch[field] + combine[field] for field in dispatch}
        self.assertEqual(roundtrip["activation_data_bytes"], 10 * 7168 * (1 + 2))
        self.assertEqual(roundtrip["scale_bytes"], 10 * 224)
        self.assertNotEqual(roundtrip["total_logical_bytes"], 2 * combine["total_logical_bytes"])

    def test_guards_fail_closed(self) -> None:
        for kwargs in (
            {"logical_copies": -1, "hidden": 8},
            {"logical_copies": 1, "hidden": -1},
            {"logical_copies": 1, "hidden": 8, "value_bytes": 0},
            {"logical_copies": 1, "hidden": 8, "value_bytes": -1},
            {"logical_copies": 1, "hidden": 8, "scale_bytes_per_copy": -1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ep_harness.logical_byte_provenance(**kwargs)


if __name__ == "__main__":
    unittest.main()
