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


# configs/platform_config.json is shared by matrix scheduling, operator/network
# loading, and backend builds.
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
# and COLLX_RDMA_LINK_LAYER. These are the launcher's patterns; the tests below check the
# probe's output against them.
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
        self.assertEqual(args.out, "results/h200-dgxc_deepep-v2_bf16_decode_TS-c000.json")

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

    def test_other_backends_round_trip_with_their_own_result_filename(self) -> None:
        # Every backend flows through the same generic codec; run_ep's --backend choices
        # must accept it and the result filename must carry the backend token so legs of
        # the same cell never collide on the same output path.
        for backend in ("uccl-ep", "nccl-ep", "flashinfer-ep"):
            case = {
                **self.CASE,
                "backend": backend,
                "case_id": f"h200-dgxc-{backend}-deepseek-v3-normal-decode-ep16-uniform-bf16",
            }
            with self.subTest(backend=backend):
                argv = self._case_argv(["16", "2", "8", "8"], case=case)
                args = self._run_ep_parser().parse_args(argv)
                self.assertEqual(args.backend, backend)
                self.assertEqual(args.case_id, case["case_id"])
                self.assertEqual(
                    args.out, f"results/h200-dgxc_{backend}_bf16_decode_TS-c000.json"
                )

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

if __name__ == "__main__":
    unittest.main()
