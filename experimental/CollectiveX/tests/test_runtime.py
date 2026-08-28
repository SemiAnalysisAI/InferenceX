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
from unittest import mock


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

    def test_inactive_port_emits_a_launcher_recognized_failure_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fabric(root, state="1: DOWN")
            rc, lines = self._run(root, root / "route")
            self.assertEqual(rc, 1)
            failures = [line for line in lines if re.search(FAILURE_MARKER, line)]
            self.assertTrue(any("rdma-port-1=inactive" in line for line in failures), failures)

# config.py case-args is the single case→invocation codec: collx_run_shard decodes one
# null-delimited argv per case and hands it verbatim to bench/run_ep.py. Parse the
# emitted argv with the same parser shape run_ep builds so the two sides cannot
# drift — a flag the codec emits but run_ep does not declare (or vice versa) fails
# here instead of on a GPU allocation.
# logical_byte_provenance is where FP8 changes MEASUREMENT semantics (asymmetric
# per-direction byte counts), so its arithmetic and guards are pinned here on CPU.
try:
    import torch as _torch
except Exception:  # torch is absent in the CPU test image; these checks run on GPU CI
    _torch = None


class ContainerImportRetry(unittest.TestCase):
    """A failed container import is retried, because the failure is usually the storage blinking.

    The import writes tens of GB to operator-supplied squash storage, which on some clusters is a
    SOFT-mounted network filesystem -- one that returns an error instead of blocking when its
    transport drops. gb300's /data is NFSv3 over RDMA, and a transport gap there surfaces from
    `mkdir` as "Protocol family not supported", which reads like a missing mount but is not: the
    same node writes it fine minutes later. Run 31089556516 lost its gb300 shards that way, ~25
    minutes into each leg, so the import must not treat one such failure as terminal.
    """

    HARNESS = """
set -u
export COLLX_IMAGE_PLATFORM=linux/amd64
export COLLX_JOB_ROOT="$ROOT/job"
mkdir -p "$COLLX_JOB_ROOT"
mkdir -p "$ROOT/bin" "$ROOT/sqsh"
# Fake srun: appends one line per invocation and replays a scripted exit-code sequence.
cat > "$ROOT/bin/srun" <<'FAKE'
#!/bin/bash
echo call >> "$ROOT/calls"
n=$(wc -l < "$ROOT/calls" | tr -d ' ')
codes=($RC_SEQUENCE)
idx=$(( n - 1 )); [ $idx -ge ${#codes[@]} ] && idx=$(( ${#codes[@]} - 1 ))
exit ${codes[$idx]}
FAKE
chmod +x "$ROOT/bin/srun"
export PATH="$ROOT/bin:$PATH"
source "$COMMON"
sleep() { :; }              # collapse the backoff
unsquashfs() { return 0; }  # a present squash short-circuits the import
out="$(collx_ensure_squash_on_job 12345 "$ROOT/sqsh" some/image:tag)"; rc=$?
echo "RC=$rc"
echo "OUT=$out"
echo "CALLS=$(wc -l < "$ROOT/calls" 2>/dev/null | tr -d ' ' || echo 0)"
"""

    def _run(self, rc_sequence: str):
        with tempfile.TemporaryDirectory() as root:
            proc = subprocess.run(
                ["bash", "-c", self.HARNESS],
                env={
                    **os.environ, "ROOT": root, "COMMON": str(RUNTIME / "common.sh"),
                    "RC_SEQUENCE": rc_sequence, "COLLX_IMPORT_ATTEMPTS": "3",
                },
                capture_output=True, text=True,
            )
        fields = dict(
            line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line
            and line.split("=", 1)[0] in ("RC", "OUT", "CALLS")
        )
        return fields, proc

    def test_a_transient_failure_is_retried_and_then_succeeds(self):
        # Also the fixture's own control: if the job-root shape were wrong the function would
        # fail before ever reaching srun, CALLS would be 0, and every assertion here would pass
        # vacuously. Asserting the invocation count is what makes that impossible.
        fields, proc = self._run("1 0")
        self.assertEqual(fields.get("CALLS"), "2", proc.stdout + proc.stderr)
        self.assertEqual(fields.get("RC"), "0", proc.stdout + proc.stderr)
        # Callers capture stdout as the squash path, so nothing else may reach it.
        self.assertTrue(fields.get("OUT", "").endswith(".sqsh"), fields)

    def test_an_architecture_mismatch_is_not_retried(self):
        # rc 13 is the remote platform mismatch: a property of the case, not the moment, so
        # retrying only delays the real message by two backoffs.
        fields, proc = self._run("13 13 13")
        self.assertEqual(fields.get("CALLS"), "1", proc.stdout + proc.stderr)
        self.assertEqual(fields.get("RC"), "1", proc.stdout + proc.stderr)


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

    # CX_FP8_CONSUME is read at class-body evaluation and fails closed on an unrecognised
    # value, so the workflow must never hand it one. A blank dispatch input is the trap: it
    # sets the variable to "" rather than leaving it unset, os.environ.get's default never
    # applies, and every leg dies at import before any measurement.
    def test_the_workflow_never_passes_an_invalid_fp8_consume(self) -> None:
        workflow = (
            RUNTIME.parents[2] / ".github" / "workflows" / "collectivex-sweep.yml"
        )
        if not workflow.exists():  # pragma: no cover - repo layout guard
            self.skipTest("workflow not present in this checkout")
        text = workflow.read_text()
        if "CX_FP8_CONSUME" not in text:
            self.skipTest("workflow does not set CX_FP8_CONSUME")
        line = next(l for l in text.splitlines() if l.strip().startswith("CX_FP8_CONSUME:"))
        self.assertIn(
            "|| 'native'", line,
            "a blank fp8_consume input must fall back to 'native'; passing it through empty "
            "sets the variable to \"\" and the harness fails closed at import",
        )
        for value in re.findall(r"'([a-z]*)'", line):
            if value:
                self.assertIn(value, ("native", "dequant"), value)

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

    def test_a_malformed_timing_profile_cannot_reach_a_run(self) -> None:
        # Every rejection path in one place. Objects: an unknown, renamed or missing key must be
        # as fatal as a bad string arity, or a renamed knob silently falls back to run_ep's
        # default. Strings: three fields is the pre-chain profile and six the chain profile; any
        # other length is a shard built against a codec that no longer exists.
        for timing in (
            {"iters_per_trial": 8}, {**self.CASE["timing"], "extra": 1}, {},
            "8:256", "8:256:32:128", "8:256:32:128:4:16:2", "",
        ):
            with self.subTest(timing=timing):
                with self.assertRaises(subprocess.CalledProcessError):
                    self._case_argv(["16", "2", "8", "8"], case={**self.CASE, "timing": timing})
        # Types are NOT checked by the codec -- as has always been true for iters/trials/warmup
        # -- so the property is that the argv it emits still cannot parse into a run.
        argv = self._case_argv(
            ["16", "2", "8", "8"], case={**self.CASE, "timing": "8:256:32:128:4:x"},
        )
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self._run_ep_parser().parse_args(argv)

    def test_case_args_fails_closed_on_placement_mismatch(self) -> None:
        with self.assertRaises(subprocess.CalledProcessError):
            self._case_argv(["8", "1", "8", "8"])

    def test_each_backend_round_trips_through_the_run_ep_parser(self) -> None:
        # The codec is backend-agnostic, so one loop replaces three near-identical tests:
        # run_ep's --backend choices must accept each name, and the filename must carry the
        # backend token or two legs of one cell collide in results/. That flashinfer-ep is
        # GB-only is a registry fact, pinned in test_matrix.
        for backend in ("uccl-ep", "nccl-ep", "flashinfer-ep"):
            with self.subTest(backend=backend):
                case = {
                    **self.CASE, "backend": backend,
                    "case_id": f"h200-dgxc-{backend}-deepseek-v3-normal-decode-ep16-uniform-bf16",
                }
                args = self._run_ep_parser().parse_args(
                    self._case_argv(["16", "2", "8", "8"], case=case)
                )
                self.assertEqual(args.backend, backend)
                self.assertEqual(args.case_id, case["case_id"])
                self.assertEqual(args.out, f"results/{case['case_id']}_TS-c000.json")

# logical_byte_provenance is where FP8 changes MEASUREMENT semantics (asymmetric
# per-direction byte counts), so its arithmetic and guards are pinned here on CPU.
class LogicalByteProvenanceTests(unittest.TestCase):
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

class GpuHealthProbe(unittest.TestCase):
    """Reject an allocation holding a throttled GPU before it burns the wall-clock guard: one
    clamped device paces every rank (a B200 at 120 MHz against 1965 MHz ran a case 17x slower)."""

    HEALTHY = "\n".join(f"{i}, Not Active, Not Active, 3{i} " for i in range(8))

    def _swap(self, line_in: str, line_out: str) -> str:
        self.assertIn(line_in, self.HEALTHY)  # guard the fixture against silent drift
        return self.HEALTHY.replace(line_in, line_out)

    def test_a_clamped_gpu_is_rejected_by_either_signal(self):
        # Throttle flags and temperature are INDEPENDENT signals: the flag can clear between
        # samples while the fault persists, so heat alone must reject, and either flag alone is
        # enough. The healthy fixture is the negative control -- a substring search for "Active"
        # also matches "Not Active", which is the bug this shape guards.
        self.assertEqual(probe.gpu_health_faults(self.HEALTHY), [])
        for gpu, cells in (
            (7, "7, Active, Active, 93 "), (7, "7, Active, Not Active, 88 "),
            (7, "7, Not Active, Active, 88 "), (3, "3, Not Active, Not Active, 95 "),
        ):
            with self.subTest(cells=cells):
                faults = probe.gpu_health_faults(
                    self._swap(f"{gpu}, Not Active, Not Active, 3{gpu} ", cells)
                )
                self.assertEqual(len(faults), 1)
                self.assertIn(f"gpu {gpu}", faults[0])

    def test_unreadable_output_fails_open(self):
        # Blocking legs when the hardware cannot be read is worse than the fault being sought.
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

    def test_a_fault_exits_nonzero_and_names_the_gpu(self):
        code, out = self._run_validate(
            self._swap("7, Not Active, Not Active, 37 ", "7, Active, Active, 93 ")
        )
        self.assertEqual(code, 1)
        self.assertIn("gpu-health-fault gpu 7", out)
        self.assertNotIn("gpu-health-checked", out)

    def test_the_temperature_spread_is_reported_but_never_gated(self):
        # The signal no gate can see: an H100 engages software thermal slowdown at ~86-87 C, so
        # a clamped one never crosses the 90 C limit and the measured fault showed only as an
        # idle outlier (55 C against ~30 C). Reported so a human can act, deliberately not gated,
        # and absent rather than wrong when the output cannot be read.
        sick = self._swap("3, Not Active, Not Active, 33 ", "3, Not Active, Not Active, 55 ")
        self.assertEqual(probe.gpu_temperature_spread(sick), (55, 35, 20))
        hottest, median, spread = probe.gpu_temperature_spread(self.HEALTHY)
        self.assertEqual((hottest, median), (37, 34))  # 8 temps -> median is index 4
        self.assertLess(spread, 10)
        code, out = self._run_validate(sick)
        self.assertEqual(code, 0)
        self.assertIn("spread=20C", out)
        for output in ("", "nonsense\n", self.HEALTHY.replace("33 ", "[N/A] ")):
            with self.subTest(output=output[:16]):
                result = probe.gpu_temperature_spread(output)
                self.assertTrue(result is None or result[2] < 10)


class LowLatencyCapDecoupling(unittest.TestCase):
    """The LL receive size and the measured ladder must stay two numbers -- sizing the receive
    from `max(ladder)` would shift every rung. Driven through the adapter with deep_ep stubbed,
    so the constants are exercised rather than read out of the syntax tree."""

    @staticmethod
    def _adapter():
        """Import ep_deepep_v2 with its vendor dependency stubbed out."""
        torch_stub = types.ModuleType("torch")
        torch_stub.bfloat16 = torch_stub.float32 = torch_stub.int64 = "dtype"
        torch_stub.distributed = types.SimpleNamespace(group=types.SimpleNamespace(WORLD=None))
        # The module decorates helpers at import time; pass them through untouched.
        torch_stub.compile = lambda *a, **k: (a[0] if a else (lambda fn: fn))
        torch_stub._dynamo = types.SimpleNamespace(config=types.SimpleNamespace())
        deep_ep = types.ModuleType("deep_ep")
        deep_ep.Buffer = type("Buffer", (), {})
        # The adapter imports ElasticBuffer by name and fails closed without it.
        deep_ep.ElasticBuffer = type("ElasticBuffer", (), {})
        stubs = {
            "torch": torch_stub, "torch.distributed": torch_stub.distributed,
            "deep_ep": deep_ep,
        }
        with mock.patch.dict(sys.modules, stubs):
            import importlib
            import ep_deepep_v2
            return importlib.reload(ep_deepep_v2)

    def test_buffer_cap_tracks_the_ladder_constant(self):
        # Patching the constant and watching the return move proves it reads the constant, which
        # a literal that merely happens to equal it today would not.
        module = self._adapter()
        backend = module.DeepEPV2Backend.__new__(module.DeepEPV2Backend)
        backend.mode = "low-latency"
        with mock.patch.object(module, "_LL_LADDER_CAP", 64):
            self.assertEqual(backend.buffer_cap(None), 64)
        backend.mode = "normal"
        self.assertIsNone(backend.buffer_cap(None))

    def test_the_receive_is_sized_from_the_buffer_cap_not_the_ladder(self):
        # The regression this guards would silently re-baseline every LL row: clamping the
        # measured ladder must not shrink the receive the remaining rungs are measured against.
        module = self._adapter()
        source = (BENCH / "ep_deepep_v2.py").read_text()
        sized = source.split("def create_buffer", 1)[1].split("def ", 1)[0]
        self.assertIn("_LL_BUFFER_CAP", sized)
        self.assertNotIn("_LL_LADDER_CAP", sized)
        self.assertNotEqual(module._LL_BUFFER_CAP, None)

    def test_a_clamped_ladder_is_recorded_in_the_artifact_not_only_on_stdout(self):
        # The clamp must reach the artifact: a rank-0 stdout NOTE alone leaves a document that
        # measured 8 rungs indistinguishable from one that measured 9. Asserted on a real
        # emitted document rather than on the presence of key literals in the source.
        sys.path.insert(0, str(RUNTIME.parent / "tests"))
        import test_chain
        workload = test_chain.drive().doc["workload"]
        for required in ("ladder_measured", "ladder_dropped", "ladder_cap"):
            self.assertIn(required, workload, f"the emitted record must include {required}")
        self.assertEqual(workload["ladder_measured"], list(test_chain.LADDER))


if __name__ == "__main__":
    unittest.main()
