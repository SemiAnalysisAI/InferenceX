#!/usr/bin/env python3
"""Focused tests for the standalone runtime helpers."""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import types
import unittest


RUNTIME = Path(__file__).resolve().parents[1] / "runtime"
BENCH = Path(__file__).resolve().parents[1] / "bench"
sys.path.insert(0, str(RUNTIME))
sys.path.insert(0, str(BENCH))

import probe  # noqa: E402
import config  # noqa: E402
import stage  # noqa: E402
import ep_harness  # noqa: E402  (stdlib-only at module top)
import ep_backend  # noqa: E402  (torch is imported lazily inside its methods)


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

    def test_prepare_cache_bootstraps_a_missing_squash_dir(self) -> None:
        # The probe runs before the first container import, so on a fresh pool squash_dir does
        # not exist yet; a bare mkdir killed every b200-nscale leg of that pool's first sweep.
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "sqsh"
            self.assertFalse(parent.exists())
            cache = Path(probe.prepare_cache(str(parent)))
            self.assertTrue(cache.is_dir())
            self.assertEqual(cache.parent, parent.resolve())
            self.assertEqual(cache.stat().st_mode & 0o777, 0o700)


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
        payload = self._emit_registry_only("mi325x-tw")
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

    # config.py accepts `exclude_nodes` for every SKU, but only the launcher named by that SKU's
    # `launcher` field can turn it into salloc --exclude. When a launcher ignores it the key is
    # accepted, exported, and silently dropped -- so a tray quarantined in the registry keeps
    # getting scheduled and the config reads like a fix that never fired. That is exactly what
    # launch_gb-nv.sh did: it built no allocation array at all.
    def test_every_skus_denylist_reaches_its_launcher(self) -> None:
        registry = json.loads(
            (RUNTIME.parent / "configs" / "platform_config.json").read_text())["platforms"]
        launchers = RUNTIME.parent / "launchers"
        checked = 0
        for sku, platform in registry.items():
            if not platform.get("operator", {}).get("exclude_nodes"):
                continue
            script = launchers / f"launch_{platform['launcher']}.sh"
            self.assertTrue(script.exists(), f"{sku} names a launcher that does not exist")
            self.assertIn(
                "COLLX_EXCLUDE_NODES", script.read_text(),
                f"{sku} declares exclude_nodes but {script.name} never reads it, "
                "so the denylist is silently discarded",
            )
            checked += 1
        self.assertGreater(checked, 0, "no SKU declares exclude_nodes -- test proves nothing")

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
        "ladder": "1 2 4",
        # The current producer shape: an object naming every knob (sweep_matrix emits this).
        # The colon-string fixtures below are legacy shards, exercising _migrate_timing.
        "timing": {
            "iters_per_trial": 8, "trials_per_point": 256, "warmup_iters_per_trial": 32,
            "chain_iters_per_trial": 128, "chain_trials_per_point": 4, "chain_drop": 16,
        },
        "case_id": "h200-dgxc-deepep-v2-deepseek-v3-normal-decode-ep16-uniform-bf16",
        "suite": "ep-core", "workload": "deepseek-v3",
    }

    @staticmethod
    def _run_ep_parser() -> argparse.ArgumentParser:
        # Mirror of the parser bench/run_ep.py builds in main().
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--backend", required=True,
            choices=["deepep-v2", "mori", "uccl-ep", "nccl-ep", "flashinfer-ep"],
        )
        ep_harness.add_common_args(parser)
        return parser

    def _decode(self, stdout: bytes) -> list:
        parts = stdout.split(b"\0")
        self.assertEqual(parts[-1], b"")
        return [part.decode() for part in parts[:-1]]

    def _case_argv(self, placement: list, case: dict | None = None) -> list:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shard.json"
            path.write_text(json.dumps({"version": 1, "cases": [case or self.CASE]}))
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
        self.assertEqual(args.out, f"results/{self.CASE['case_id']}_TS-c000.json")

    def test_chain_knobs_round_trip_through_the_run_ep_parser(self) -> None:
        # The chain budget is case identity like iters/trials/warmup, so it rides the same
        # timing block; hardcoded in the harness, two chain lengths would look identical.
        argv = self._case_argv(["16", "2", "8", "8"])
        args = self._run_ep_parser().parse_args(argv)
        self.assertEqual((args.iters, args.trials, args.warmup), (8, 256, 32))
        self.assertEqual((args.chain_iters, args.chain_trials, args.chain_drop), (128, 4, 16))

    def test_a_legacy_colon_string_profile_still_decodes(self) -> None:
        # Sweep `version` does not bump for the codec change, so a shard staged before it --
        # or one built by hand -- must still produce a runnable argv. Both legacy arities:
        # six positional fields, and the pre-chain three whose chain knobs then come from
        # run_ep's own defaults rather than being duplicated in the codec.
        for profile, chain in (("8:256:32:128:4:16", (128, 4, 16)), ("8:256:32", None)):
            with self.subTest(timing=profile):
                args = self._run_ep_parser().parse_args(self._case_argv(
                    ["16", "2", "8", "8"], case={**self.CASE, "timing": profile},
                ))
                self.assertEqual((args.iters, args.trials, args.warmup), (8, 256, 32))
                if chain:
                    self.assertEqual(
                        (args.chain_iters, args.chain_trials, args.chain_drop), chain
                    )
                    continue
                for flag in ("chain_iters", "chain_trials", "chain_drop"):
                    self.assertIsInstance(getattr(args, flag), int)
                # A chain that drops everything it measured has no samples left to reduce.
                self.assertGreater(args.chain_iters, args.chain_drop)
                self.assertGreater(args.chain_trials, 0)

    def test_a_timing_object_with_unknown_keys_fails_closed(self) -> None:
        # The object path must be as strict as the arity check on the string path, or a
        # renamed knob would silently fall back to run_ep's default.
        for timing in ({"iters_per_trial": 8}, {**self.CASE["timing"], "extra": 1}, {}):
            with self.subTest(timing=timing):
                with self.assertRaises(subprocess.CalledProcessError):
                    self._case_argv(["16", "2", "8", "8"], case={**self.CASE, "timing": timing})

    def test_a_timing_profile_of_the_wrong_arity_fails_closed(self) -> None:
        # Three fields is the pre-chain profile and six is the chain profile; any other length
        # is a shard built against a codec that no longer exists, so the codec refuses.
        for profile in ("8:256", "8:256:32:128", "8:256:32:128:4:16:2", ""):
            with self.subTest(timing=profile):
                with self.assertRaises(subprocess.CalledProcessError):
                    self._case_argv(["16", "2", "8", "8"], case={**self.CASE, "timing": profile})

    def test_a_non_numeric_timing_field_cannot_reach_a_run(self) -> None:
        # The codec checks arity, not types -- as it always has for iters/trials/warmup -- so
        # the property held here is that the argv it emits still cannot parse into a run.
        argv = self._case_argv(
            ["16", "2", "8", "8"], case={**self.CASE, "timing": "8:256:32:128:4:x"},
        )
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self._run_ep_parser().parse_args(argv)

    def test_case_args_fails_closed_on_placement_mismatch(self) -> None:
        with self.assertRaises(subprocess.CalledProcessError):
            self._case_argv(["8", "1", "8", "8"])

    def test_low_latency_case_round_trips_through_the_run_ep_parser(self) -> None:
        # A low-latency decode EP8 case flows through the same codec; run_ep's --mode
        # choices must accept "low-latency" or the leg dies before allocation.
        ll_case = {
            **self.CASE,
            "mode": "low-latency", "phase": "decode",
            "ep": 8, "nodes": 1, "gpus_per_node": 8, "scale_up_domain": 8,
            "scope": "scale-up", "scale_up_transport": "nvlink",
            "scale_out_transport": "", "transport": "nvlink",
            "topology_class": "h200-nvlink-island", "ladder": "1 2 4 8",
            "case_id": "h200-dgxc-deepep-v2-deepseek-v3-low-latency-decode-ep8-uniform-bf16",
        }
        argv = self._case_argv(["8", "1", "8", "8"], case=ll_case)
        args = self._run_ep_parser().parse_args(argv)
        self.assertEqual((args.mode, args.phase, args.scope), ("low-latency", "decode", "scale-up"))
        self.assertEqual(args.case_id, ll_case["case_id"])
        # Without the mode in the filename this case and its normal-mode sibling produce
        # byte-identical paths, so driving both under one timestamp overwrites one artifact.
        self.assertEqual(
            args.out, f"results/{ll_case['case_id']}_TS-c000.json"
        )

    def test_uccl_ep_case_round_trips_through_the_run_ep_parser(self) -> None:
        # A uccl-ep case flows through the same generic codec; run_ep's --backend choices
        # must accept "uccl-ep" and the result filename must carry the backend token so a
        # uccl-ep leg never collides with the deepep-v2/mori legs of the same cell.
        uccl_case = {
            **self.CASE,
            "backend": "uccl-ep",
            "case_id": "h200-dgxc-uccl-ep-deepseek-v3-normal-decode-ep16-uniform-bf16",
        }
        argv = self._case_argv(["16", "2", "8", "8"], case=uccl_case)
        args = self._run_ep_parser().parse_args(argv)
        self.assertEqual(args.backend, "uccl-ep")
        self.assertEqual(args.case_id, uccl_case["case_id"])
        self.assertEqual(args.out, f"results/{uccl_case['case_id']}_TS-c000.json")

    def test_nccl_ep_case_round_trips_through_the_run_ep_parser(self) -> None:
        # A nccl-ep case flows through the same generic codec; run_ep's --backend choices must
        # accept "nccl-ep" and the result filename must carry the backend token so a nccl-ep leg
        # never collides with the deepep-v2/uccl-ep legs of the same cell. BF16 only.
        nccl_case = {
            **self.CASE,
            "backend": "nccl-ep",
            "case_id": "h200-dgxc-nccl-ep-deepseek-v3-normal-decode-ep16-uniform-bf16",
        }
        argv = self._case_argv(["16", "2", "8", "8"], case=nccl_case)
        args = self._run_ep_parser().parse_args(argv)
        self.assertEqual(args.backend, "nccl-ep")
        self.assertEqual(args.case_id, nccl_case["case_id"])
        self.assertEqual(args.out, f"results/{nccl_case['case_id']}_TS-c000.json")

    def test_flashinfer_ep_case_round_trips_through_the_run_ep_parser(self) -> None:
        # A flashinfer-ep case flows through the same generic codec; run_ep's --backend
        # choices must accept "flashinfer-ep" and the result filename must carry the backend
        # token so it never collides with the deepep-v2/nccl-ep legs of the same cell.
        # The codec is SKU-agnostic, so this reuses the shared h200 fixture like its
        # siblings; that flashinfer-ep is GB-only is a registry fact, pinned separately by
        # test_matrix.test_flashinfer_ep_rollout_shape.
        flashinfer_case = {
            **self.CASE,
            "backend": "flashinfer-ep",
            "case_id": "h200-dgxc-flashinfer-ep-deepseek-v3-normal-decode-ep16-uniform-bf16",
        }
        argv = self._case_argv(["16", "2", "8", "8"], case=flashinfer_case)
        args = self._run_ep_parser().parse_args(argv)
        self.assertEqual(args.backend, "flashinfer-ep")
        self.assertEqual(args.case_id, flashinfer_case["case_id"])
        self.assertEqual(
            args.out, f"results/{flashinfer_case['case_id']}_TS-c000.json"
        )

    def test_mirrored_backend_choices_match_run_ep(self) -> None:
        """The mirror is only worth having if it cannot drift from the real parser.

        Kept in sync by convention it has now drifted twice — a backend was added to
        run_ep.py's choices while this fixture kept the old list, so a case_id that the real
        CLI accepts raised SystemExit here. Read the real list out of the source (AST, not
        import: importing run_ep pulls in torch and the vendor EP libraries) and compare.
        """
        tree = ast.parse((BENCH / "run_ep.py").read_text())
        real = [
            [element.value for element in keyword.value.elts]
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for argument in node.args
            if isinstance(argument, ast.Constant) and argument.value == "--backend"
            for keyword in node.keywords
            if keyword.arg == "choices"
        ]
        self.assertEqual(len(real), 1, "expected exactly one --backend choices list")
        mirrored = next(
            action.choices
            for action in self._run_ep_parser()._actions
            if action.dest == "backend"
        )
        self.assertEqual(sorted(real[0]), sorted(mirrored))


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


class ModeSemanticsContract(unittest.TestCase):
    # The combine contract is a backend fact, not a pure function of mode: DeepEP's
    # low-latency combine is weighted-kernel-sum while MoRI's IntraNodeLL is
    # unweighted-rank-sum, so low-latency must admit both. Normal stays unweighted-only.
    def test_mode_allowed_semantics(self) -> None:
        self.assertEqual(
            ep_harness.MODE_ALLOWED_SEMANTICS["normal"], {"unweighted-rank-sum"}
        )
        self.assertEqual(
            ep_harness.MODE_ALLOWED_SEMANTICS["low-latency"],
            {"weighted-kernel-sum", "unweighted-rank-sum"},
        )

    def test_oracle_modeled_contract_pairs(self) -> None:
        # receive_layout and combine_weight_semantics are independent declarations;
        # only these two pairings have an expected-combine model, and run_sweep fails
        # closed on the other two rather than verify against the wrong oracle.
        self.assertEqual(
            ep_harness.ORACLE_MODELED_CONTRACTS,
            {
                ("token-rank", "unweighted-rank-sum"),
                ("token-expert", "weighted-kernel-sum"),
            },
        )


try:
    import torch as _torch
except Exception:  # torch is absent in the CPU test image; these checks run on GPU CI
    _torch = None


@unittest.skipUnless(_torch is not None, "combine-oracle math checks require torch")
class WeightedCombineSemanticsTests(unittest.TestCase):
    """Pin the semantic distinction between the two combine contracts, independent of any
    GPU backend. Normal mode folds the gate weight INTO the staged transform (kernel
    sums); low-latency stages the UNWEIGHTED transform and the kernel applies the gate."""

    def _problem(self, weight_scale: float = 1.0):
        torch = _torch
        x = torch.randn(4, 64, dtype=torch.bfloat16)
        idx = torch.tensor([[0, 3], [1, 2], [2, 0], [3, 1]], dtype=torch.int64)
        weights = (torch.rand(4, 2, dtype=torch.float32) + 0.1) * weight_scale
        return types.SimpleNamespace(x=x, topk_idx=idx, topk_weights=weights)

    def test_transform_drops_the_gate_under_weighted_kernel_sum(self):
        torch = _torch
        payload = torch.randn(3, 64, dtype=torch.bfloat16)
        ids = torch.tensor([[2, -1], [5, -1], [7, -1]], dtype=torch.int64)
        low = ep_harness._expert_transform(
            torch, payload, ids, torch.full((3, 2), 0.2), "weighted-kernel-sum"
        )
        high = ep_harness._expert_transform(
            torch, payload, ids, torch.full((3, 2), 0.9), "weighted-kernel-sum"
        )
        # Unit coefficient: the staged value cannot depend on the gate magnitude.
        self.assertTrue(torch.equal(low, high))

    def test_transform_folds_the_gate_under_unweighted_rank_sum(self):
        torch = _torch
        payload = torch.randn(3, 64, dtype=torch.bfloat16)
        ids = torch.tensor([[2, -1], [5, -1], [7, -1]], dtype=torch.int64)
        low = ep_harness._expert_transform(
            torch, payload, ids, torch.full((3, 2), 0.2), "unweighted-rank-sum"
        )
        high = ep_harness._expert_transform(
            torch, payload, ids, torch.full((3, 2), 0.9), "unweighted-rank-sum"
        )
        # The gate IS in the transform here, so a larger weight changes the staged value.
        self.assertFalse(torch.equal(low, high))

    def test_expected_combine_is_linear_in_the_gate_under_weighted_kernel_sum(self):
        torch = _torch
        p = self._problem(1.0)
        p2 = types.SimpleNamespace(
            x=p.x, topk_idx=p.topk_idx, topk_weights=p.topk_weights * 2
        )
        base = ep_harness._expected_transformed_combine(
            torch, p, 4, 8, "weighted-kernel-sum"
        )
        doubled = ep_harness._expected_transformed_combine(
            torch, p2, 4, 8, "weighted-kernel-sum"
        )
        # Same routing/activations, gate x2 -> expected x2 (the kernel applies the gate).
        self.assertTrue(torch.allclose(doubled, base * 2, atol=1e-3, rtol=1e-3))

    def test_unknown_semantics_fail_closed(self):
        torch = _torch
        with self.assertRaises(ValueError):
            ep_harness._expected_transformed_combine(
                torch, self._problem(), 4, 8, "made-up"
            )




@unittest.skipUnless(_torch is not None, "combine-oracle math checks require torch")
class TopkSlotTreeReductionTests(unittest.TestCase):
    """Pin the payload-dtype reduction against a value measured on the real kernel.

    Eight contributions of 1.0 and 7 x 2^-9 reduce to three different answers depending on
    the model, which is what makes this case worth pinning: FP32-then-narrow gives
    1.015625, a sequential BF16 sum gives 1.0, and the pairwise BF16 tree gives 1.0078125.
    gb200 returns 1.0078125.
    """

    def _tree(self, values):
        torch = _torch
        slots = [torch.full((1, 1), v, dtype=torch.float32) for v in values]
        destination = torch.arange(len(values)).unsqueeze(0)
        messages = torch.stack(slots)
        return ep_harness._topk_slot_tree_combine(
            torch, destination, torch.ones_like(destination, dtype=torch.bool),
            messages, torch.bfloat16,
        ).item()

    def test_matches_the_value_the_kernel_returns(self):
        self.assertEqual(self._tree([1.0] + [2.0**-9] * 7), 1.0078125)

    def test_differs_from_both_rejected_models(self):
        values = [1.0] + [2.0**-9] * 7
        self.assertNotEqual(self._tree(values), 1.015625)  # FP32 accumulate, narrow once
        self.assertNotEqual(self._tree(values), 1.0)       # sequential BF16 accumulate

    def test_a_rank_claimed_by_an_earlier_slot_contributes_once(self):
        torch = _torch
        # Both top-k slots route to rank 0; the kernel blanks the later slot in place.
        destination = torch.zeros((1, 2), dtype=torch.int64)
        messages = torch.full((1, 1, 1), 0.5)
        combined = ep_harness._topk_slot_tree_combine(
            torch, destination, torch.ones_like(destination, dtype=torch.bool),
            messages, torch.bfloat16,
        )
        self.assertEqual(combined.item(), 0.5)

    def test_the_first_slot_claiming_a_rank_is_the_one_that_survives(self):
        torch = _torch
        # These cancel only under keep-first: blanking the later slot reduces (256+1)->256 and
        # (-256+0)->-256 to 0.0, blanking the earlier gives 1.0 (257->256 is BF16 rounding).
        destination = torch.tensor([[0, 1, 2, 0]])
        messages = torch.tensor([[[256.0]], [[1.0]], [[-256.0]]])
        combined = ep_harness._topk_slot_tree_combine(
            torch, destination, torch.ones_like(destination, dtype=torch.bool),
            messages, torch.bfloat16,
        )
        self.assertEqual(combined.item(), 0.0)



@unittest.skipUnless(_torch is not None, "quantize-identity checks require torch")
class FusedQuantizeGate(unittest.TestCase):
    """The oracle's payload gate compares the sender's [T, hidden] quantize against the oracle's
    [receive_count, hidden] one, so a fused callable must be bit-identical and per-row invariant."""

    @staticmethod
    def _fuse(mode, eager):
        # Both methods read only `self.mode`, so call them unbound rather than build a backend.
        return ep_backend.EPBackend.fused_quantize(types.SimpleNamespace(mode=mode), eager)

    @staticmethod
    def _check(eager, fused, x):
        return ep_backend.EPBackend.assert_quantize_identity(
            types.SimpleNamespace(mode="normal"), eager, fused, x
        )

    def test_low_latency_keeps_the_eager_helper(self):
        # Its dispatch kernel quantises internally and the oracle gate is pinned to those bits,
        # so a compiled callable would red every LL fp8 cell without touching timing.
        def eager(x):
            return x, x
        self.assertIs(self._fuse("low-latency", eager), eager)

    def test_normal_mode_wraps_the_helper(self):
        def eager(x):
            return x, x
        self.assertIsNot(self._fuse("normal", eager), eager)

    def test_identity_check_accepts_an_equivalent_callable(self):
        torch = _torch
        x = torch.randn(8, 256, dtype=torch.bfloat16)

        def eager(t):
            return t.to(torch.float8_e4m3fn), t.float().abs().amax(dim=1)

        self._check(eager, lambda t: eager(t), x)  # must not raise

    def test_identity_check_rejects_a_divergent_callable(self):
        torch = _torch
        x = torch.randn(8, 256, dtype=torch.bfloat16)

        def eager(t):
            return t.to(torch.float8_e4m3fn), t.float().abs().amax(dim=1)

        def divergent(t):
            values, scales = eager(t)
            return values, scales + 1          # one differing scale is enough to red a cell
        with self.assertRaises(RuntimeError):
            self._check(eager, divergent, x)

    def test_identity_check_rejects_a_shape_dependent_callable(self):
        # A callable that is deterministic but NOT per-row invariant still breaks the gate,
        # because the oracle quantises a different row count than the sender did.
        torch = _torch
        x = torch.randn(8, 256, dtype=torch.bfloat16)

        def eager(t):
            return t.to(torch.float8_e4m3fn), t.float().abs().amax(dim=1)

        def shape_dependent(t):
            values, scales = eager(t)
            return values, scales * float(t.shape[0])
        with self.assertRaises(RuntimeError):
            self._check(eager, shape_dependent, x)

class GpuHealthProbe(unittest.TestCase):
    """Reject an allocation holding a throttled GPU before it burns the wall-clock guard: one
    clamped device paces every rank (a B200 at 120 MHz against 1965 MHz ran a case 17x slower)."""

    HEALTHY = "\n".join(f"{i}, Not Active, Not Active, 3{i} " for i in range(8))

    def _swap(self, line_in: str, line_out: str) -> str:
        self.assertIn(line_in, self.HEALTHY)  # guard the fixture against silent drift
        return self.HEALTHY.replace(line_in, line_out)

    def test_healthy_allocation_passes(self):
        self.assertEqual(probe.gpu_health_faults(self.HEALTHY), [])

    def test_a_thermally_throttled_gpu_is_rejected(self):
        output = self._swap("7, Not Active, Not Active, 37 ", "7, Active, Active, 93 ")
        faults = probe.gpu_health_faults(output)
        self.assertEqual(len(faults), 1)
        self.assertIn("gpu 7", faults[0])

    def test_either_throttle_flag_alone_is_enough(self):
        for cells in ("7, Active, Not Active, 88 ", "7, Not Active, Active, 88 "):
            with self.subTest(cells=cells):
                output = self._swap("7, Not Active, Not Active, 37 ", cells)
                self.assertEqual(len(probe.gpu_health_faults(output)), 1)

    def test_not_active_is_not_read_as_active(self):
        # A substring search for "Active" also matches "Not Active".
        self.assertEqual(probe.gpu_health_faults(self.HEALTHY), [])

    def test_temperature_is_an_independent_signal(self):
        # The flag can clear between samples while the fault persists, so heat alone rejects.
        output = self._swap("3, Not Active, Not Active, 33 ", "3, Not Active, Not Active, 95 ")
        faults = probe.gpu_health_faults(output)
        self.assertEqual(len(faults), 1)
        self.assertIn("gpu 3", faults[0])

    def test_unreadable_output_fails_open(self):
        # Blocking legs when the hardware cannot be read is worse than the fault being looked for.
        for output in ("", "nonsense\n", "1, Not Active\n", self.HEALTHY.replace("32 ", "[N/A] ")):
            with self.subTest(output=output[:20]):
                self.assertEqual(probe.gpu_health_faults(output), [])

    def _run_validate(self, csv: str, has_smi: bool = True):
        """Drive validate_gpu_health with a stubbed nvidia-smi; returns (exit_code, stdout)."""
        import shutil
        real_which = shutil.which
        shutil.which = (lambda name: "/usr/bin/nvidia-smi") if has_smi else (lambda name: None)

        class FakeSubprocess:
            SubprocessError = subprocess.SubprocessError

            @staticmethod
            def run(*args, **kwargs):
                return types.SimpleNamespace(stdout=csv)

        sys.modules["subprocess"] = FakeSubprocess
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured):
                probe.validate_gpu_health()
            code = 0
        except SystemExit as exit_:
            code = exit_.code
        finally:
            sys.modules["subprocess"] = subprocess
            shutil.which = real_which
        return code, captured.getvalue()

    def test_a_healthy_check_records_how_many_gpus_it_saw(self):
        # Without this marker a gate that went blind -- no visible devices, or a driver spelling
        # the fields clocks_throttle_reasons.* -- looks like one that saw eight healthy GPUs.
        code, out = self._run_validate(self.HEALTHY)
        self.assertEqual(code, 0)
        self.assertIn("gpu-health-checked gpus=8", out)

    def test_a_fault_exits_nonzero_and_names_the_gpu(self):
        code, out = self._run_validate(
            self._swap("7, Not Active, Not Active, 37 ", "7, Active, Active, 93 ")
        )
        self.assertEqual(code, 1)
        self.assertIn("gpu-health-fault gpu 7", out)
        self.assertNotIn("gpu-health-checked", out)

    def test_a_missing_nvidia_smi_is_silent_and_passes(self):
        code, out = self._run_validate(self.HEALTHY, has_smi=False)
        self.assertEqual(code, 0)
        self.assertEqual(out, "")
    def test_the_temperature_spread_is_reported_for_the_signal_no_gate_can_see(self):
        # An H100 engages software thermal slowdown at ~86-87 C, so a clamped one never crosses
        # the 90 C limit; the measured fault showed only as an idle outlier (55 C vs ~30 C).
        sick = self._swap("3, Not Active, Not Active, 33 ", "3, Not Active, Not Active, 55 ")
        self.assertEqual(probe.gpu_temperature_spread(sick), (55, 35, 20))
        hottest, median, spread = probe.gpu_temperature_spread(self.HEALTHY)
        self.assertEqual((hottest, median), (37, 34))  # 8 temps -> median is index 4
        self.assertLess(spread, 10)

    def test_the_spread_appears_in_the_healthy_marker(self):
        sick = self._swap("3, Not Active, Not Active, 33 ", "3, Not Active, Not Active, 55 ")
        code, out = self._run_validate(sick)
        self.assertEqual(code, 0)  # reported, deliberately not gated on
        self.assertIn("spread=20C", out)

    def test_the_spread_is_none_when_unreadable(self):
        for output in ("", "nonsense\n", self.HEALTHY.replace("33 ", "[N/A] ")):
            with self.subTest(output=output[:16]):
                result = probe.gpu_temperature_spread(output)
                self.assertTrue(result is None or result[2] < 10)


class LowLatencyCapDecoupling(unittest.TestCase):
    """The LL receive size and the measured ladder must stay two numbers -- sizing the receive from
    `max(ladder)` would shift every rung. Over the source: importing the adapter needs deep_ep."""

    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse((BENCH / "ep_deepep_v2.py").read_text())
        cls.consts = {
            t.id: node.value.value
            for node in cls.tree.body
            if isinstance(node, ast.Assign)
            for t in node.targets
            if isinstance(t, ast.Name) and isinstance(node.value, ast.Constant)
        }

    def _func(self, name):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        self.fail(f"{name} not found in ep_deepep_v2.py")

    def test_both_caps_exist_and_the_ladder_fits_inside_the_buffer(self):
        buf, ladder = self.consts.get("_LL_BUFFER_CAP"), self.consts.get("_LL_LADDER_CAP")
        self.assertIsInstance(buf, int)
        self.assertIsInstance(ladder, int)
        # Equality is expected while the ladder runs to the full receive; a ladder larger than
        # the buffer would dispatch past the allocated slots.
        self.assertLessEqual(ladder, buf)
        # NVSHMEM_QP_DEPTH=1024 asserts nvshmem_qp_depth >= (cap + 1) * 2 at construction.
        self.assertLessEqual(buf, 511)

    def test_buffer_cap_clamps_the_ladder_by_the_constant_not_a_literal(self):
        returns = [
            n.value for n in ast.walk(self._func("buffer_cap"))
            if isinstance(n, ast.Return) and n.value is not None
        ]
        names = {n.id for n in returns if isinstance(n, ast.Name)}
        self.assertIn("_LL_LADDER_CAP", names)
        # A bare literal here would drift out of step with the constants above.
        self.assertEqual([n for n in returns if isinstance(n, ast.Constant) and n.value is not None], [])

    def test_the_low_latency_receive_is_sized_from_the_cap_not_the_ladder(self):
        # Guards the regression that would silently re-baseline every LL row.
        assigned = [
            node.value for node in ast.walk(self._func("create_buffer"))
            if isinstance(node, ast.Assign)
            for t in node.targets
            if isinstance(t, ast.Attribute) and t.attr == "max_tokens"
        ]
        self.assertTrue(
            any(isinstance(v, ast.Name) and v.id == "_LL_BUFFER_CAP" for v in assigned),
            "create_buffer must set self.max_tokens = _LL_BUFFER_CAP on the low-latency path",
        )

    def test_a_clamped_ladder_is_recorded_in_the_artifact_not_only_on_stdout(self):
        # A clamped ladder was previously visible only as a rank-0 stdout NOTE, so an artifact
        # from a backend that measured 8 rungs looked identical to one that measured 9.
        harness = ast.parse((BENCH / "ep_harness.py").read_text())
        keys = {
            k.value for node in ast.walk(harness) if isinstance(node, ast.Dict)
            for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)
        }
        for required in ("ladder_measured", "ladder_dropped", "ladder_cap"):
            self.assertIn(required, keys, f"the emitted record must include {required}")


class ContainerImportRetry(unittest.TestCase):
    """A failed container import is retried, because the failure is usually the storage blinking.

    The import writes tens of GB to operator-supplied squash storage, which on some clusters is a
    SOFT-mounted network filesystem -- one that returns an error instead of blocking when its
    transport drops. gb300's /data is NFSv3 over RDMA, and a transport gap there surfaces from
    `mkdir` as "Protocol family not supported", which reads like a missing mount but is not: the
    same node writes it fine minutes later. Run 31089556516 lost its gb300 shards that way, ~25
    minutes into each leg, so the import must not treat one such failure as terminal.
    """

    SOURCE = (RUNTIME / "common.sh").read_text()

    def _function(self) -> str:
        body = self.SOURCE.split("collx_ensure_squash_on_job() {", 1)
        self.assertEqual(len(body), 2, "collx_ensure_squash_on_job moved or was renamed")
        return body[1].split("\n}\n", 1)[0]

    def test_the_import_is_attempted_more_than_once(self):
        fn = self._function()
        self.assertIn("for attempt in", fn, "the import must loop rather than run once")
        self.assertIn("COLLX_IMPORT_ATTEMPTS", fn, "attempt count must be overridable")

    def test_an_architecture_mismatch_is_not_retried(self):
        # The remote block exits 13 when the image platform does not match the allocated
        # machine. That is a property of the case, not the moment: retrying only delays the
        # real message by two backoffs.
        fn = self._function()
        self.assertRegex(fn, r'"\$rc"\s*=\s*13', "rc 13 must be handled explicitly")
        thirteen = fn.split('= 13', 1)[1]
        self.assertIn("return 1", thirteen.split("fi", 1)[0],
                      "the architecture guard must return, not fall through to another attempt")

    def test_each_attempt_gets_its_own_log(self):
        # collx_private_log_path truncates, so reusing one path across attempts would erase the
        # evidence of the failure that caused the retry.
        fn = self._function()
        self.assertIn("-r${attempt}", fn, "retries must not clobber the first attempt's log")

    def test_success_still_prints_only_the_squash_path(self):
        # Callers capture stdout: SQUASH_FILE="$(collx_ensure_squash_on_job ...)". Any extra
        # stdout from the retry bookkeeping would corrupt the path. The heredoc body is excluded
        # because it executes on the compute node under srun, whose own stdout is redirected to
        # the attempt log -- what it echoes never reaches this function's caller.
        fn = self._function()
        local, inside = [], False
        for line in fn.splitlines():
            if "<<'BASH'" in line:
                inside = True
                continue
            if inside:
                inside = line.strip() != "BASH"
                continue
            local.append(line.strip())
        emits = [line for line in local
                 if ("printf" in line or "echo" in line) and ">&2" not in line]
        self.assertTrue(emits, "the function must still emit the squash path")
        for line in emits:
            self.assertIn("$sq", line, f"unexpected stdout from the import helper: {line!r}")


if __name__ == "__main__":
    unittest.main()
