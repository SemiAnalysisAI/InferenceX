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
from unittest import mock
import urllib.error


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
        "single_node_rdma_devices",
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
                    for degree in degrees:
                        self.assertIs(type(degree), int)
                        self.assertGreater(degree, 0)
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


class _FakeRegistry:
    """Registry-v2 anonymous token dance: 401 challenge, token grant, digest HEAD."""

    class _Response:
        def __init__(self, headers=None, body=b""):
            self.headers, self._body = headers or {}, body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def __init__(self, digest: str):
        self.digest, self.urls = digest, []

    def open(self, request, timeout=None):
        url = request if isinstance(request, str) else request.full_url
        self.urls.append(url)
        if "scope=" in url:
            return self._Response(body=json.dumps({"token": "anonymous"}).encode())
        if not request.get_header("Authorization"):
            raise urllib.error.HTTPError(url, 401, "unauthorized", {
                "WWW-Authenticate":
                    'Bearer realm="https://auth.example/token",service="registry.example"',
            }, None)
        return self._Response(headers={"Docker-Content-Digest": self.digest})


# The digest only stamps the staged squash's sidecar, but a wrong or accepted-malformed
# digest silently turns stage-once-reuse-many back into per-launch imports, so the
# resolution path is pinned without touching the network.
class ImageDigestResolutionTests(unittest.TestCase):
    DIGEST = "sha256:" + "0123456789abcdef" * 4

    def test_references_resolve_with_docker_hub_rules(self) -> None:
        for image, expected in (
            ("rocm/sgl-dev:v1", ("registry-1.docker.io", "rocm/sgl-dev", "v1")),
            ("python:3.12", ("registry-1.docker.io", "library/python", "3.12")),
            ("ghcr.io/org/image:tag", ("ghcr.io", "org/image", "tag")),
            ("nvcr.io/nvidia/pytorch:25.06-py3", ("nvcr.io", "nvidia/pytorch", "25.06-py3")),
        ):
            self.assertEqual(probe.registry_reference(image), expected, image)

    def test_resolution_follows_the_anonymous_token_dance(self) -> None:
        registry = _FakeRegistry(self.DIGEST)
        self.assertEqual(
            probe.resolve_image_digest("ghcr.io/org/image:tag", opener=registry),
            self.DIGEST,
        )
        self.assertEqual(registry.urls[0], "https://ghcr.io/v2/org/image/manifests/tag")
        self.assertIn("scope=repository%3Aorg%2Fimage%3Apull", registry.urls[1])

    def test_a_malformed_digest_or_registry_failure_yields_empty(self) -> None:
        down = types.SimpleNamespace(open=mock.Mock(side_effect=OSError("no route")))
        for opener in (_FakeRegistry("sha256:nothex"), down):
            self.assertEqual(
                probe.resolve_image_digest("ghcr.io/org/image:tag", opener=opener), "")


# runtime/common.sh collx_squash_path/collx_squash_verdict are the stage-once seam:
# keying the squash by GITHUB_RUN_ID is exactly the defect that re-imported 30-65GB
# per run per cluster, and keying it by digest made a transient registry blip miss
# the staged file -- so the path is a pure function of platform + image reference,
# and freshness is decided by the verdict against the digest sidecar.
class SquashCacheKeyTests(unittest.TestCase):
    IMAGE = "rocm/sgl-dev:sglang-v1"
    DIGEST = "sha256:" + "ab" * 32

    def _bash(self, script: str, env: dict, args: list) -> str:
        result = subprocess.run(
            ["bash", "-c", f'source "{RUNTIME / "common.sh"}" && {script}', "collx", *args],
            capture_output=True, text=True, check=True,
            env={"PATH": os.environ["PATH"], "COLLX_IMAGE_PLATFORM": "linux/amd64", **env},
        )
        return result.stdout

    def test_the_path_is_invariant_across_runs_and_digests(self) -> None:
        paths = {
            self._bash('collx_squash_path /squash "$1"', env, [self.IMAGE])
            for env in (
                {"GITHUB_RUN_ID": "1111", "GITHUB_RUN_ATTEMPT": "1"},
                {"GITHUB_RUN_ID": "2222", "COLLECTIVEX_EXECUTION_ID": "2222_2_c007"},
                {"COLLX_IMAGE_DIGEST": self.DIGEST},
                {},
            )
        }
        self.assertEqual(paths, {"/squash/_rocm_sgl-dev_sglang-v1.sqsh"})
        arm = self._bash('collx_squash_path /squash "$1"',
                         {"COLLX_IMAGE_PLATFORM": "linux/arm64"}, [self.IMAGE])
        self.assertEqual(arm, "/squash/_linux_arm64_rocm_sgl-dev_sglang-v1.sqsh")

    def test_the_verdict_orders_refresh_stamp_and_reuse_correctly(self) -> None:
        verdict = lambda sq, digest="", epoch="": self._bash(  # noqa: E731
            'collx_squash_verdict "$1" "$2" "$3"', {}, [str(sq), digest, epoch])
        with tempfile.TemporaryDirectory() as directory:
            sq = Path(directory) / "img.sqsh"
            self.assertEqual(verdict(sq), "absent")
            sq.write_bytes(b"x")
            Path(f"{sq}.digest").write_text(self.DIGEST + "\n")
            # The registry-blip regression: an unresolved digest must reuse the
            # stamped file, never re-import tens of GB.
            self.assertEqual(verdict(sq), "reuse")
            self.assertEqual(verdict(sq, digest=self.DIGEST), "reuse")
            self.assertEqual(verdict(sq, digest="sha256:" + "cd" * 32), "digest-moved")
            mtime = int(sq.stat().st_mtime)
            self.assertEqual(verdict(sq, epoch=str(mtime + 10)), "refresh-requested")
            # A file imported during this launch (mtime >= epoch) is kept, so
            # concurrent legs of a refreshing run still import exactly once.
            self.assertEqual(verdict(sq, epoch=str(mtime - 10)), "reuse")


class ConfigTests(unittest.TestCase):
    @staticmethod
    def _emit_operator(path: str) -> bytes:
        platform = {
            "image": "example/engine:test", "image_platform": "linux/arm64",
            "operator": {"partition": "baseline", "account": "shared"},
            "network": {"rdma_devices": "mlx5_1:1"},
        }
        output = io.BytesIO()
        with mock.patch.object(config, "_platforms", return_value={"test-sku": platform}), \
                mock.patch.object(sys, "stdout", types.SimpleNamespace(buffer=output)):
            config.operator_config(path, "test-sku")
        return output.getvalue()

    def test_operator_config_overrides_baseline_and_preserves_platform_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "operator.json"
            path.write_text(json.dumps({
                "runners": {
                    "test-sku": {
                        "partition": "gpu",
                        "account": "bench",
                        "squash_dir": directory,
                    }
                },
            }))
            path.chmod(0o600)
            payload = self._emit_operator(str(path))
            self.assertIn(b"COLLX_PARTITION\0gpu\0", payload)
            self.assertIn(b"COLLX_ACCOUNT\0bench\0", payload)
            self.assertIn(b"COLLX_SQUASH_DIR\0" + directory.encode() + b"\0", payload)
            self.assertIn(b"COLLX_IMAGE\0example/engine:test\0", payload)
            self.assertIn(b"COLLX_IMAGE_PLATFORM\0linux/arm64\0", payload)
            self.assertIn(b"COLLX_RDMA_DEVICES\0mlx5_1:1\0", payload)

    def test_operator_config_registry_only_emits_tracked_baseline(self) -> None:
        # "-" = no operator document: the registry's per-SKU operator block is
        # the tracked baseline (plus its network overlay where present).
        payload = self._emit_operator("-")
        self.assertIn(b"COLLX_PARTITION\0baseline\0", payload)
        self.assertIn(b"COLLX_ACCOUNT\0shared\0", payload)
        self.assertIn(b"COLLX_RDMA_DEVICES\0mlx5_1:1\0", payload)

class SingleNodeHcaOverrideTests(unittest.TestCase):
    # collx_apply_network_profile's single-node early return must still honor a
    # SKU's pinned single-node HCA list (b300: DeepEP's legacy LL Buffer
    # self-enables IBGDA even single-node, and only the storage-IB rails accept
    # AH/DCT creation), while scale-out runs keep resolving NVSHMEM_HCA_LIST
    # from the ordinary scale-out selector.
    @staticmethod
    def _profile_env(script: str) -> str:
        completed = subprocess.run(
            ["bash", "-c", script], cwd=RUNTIME.parent,
            capture_output=True, text=True, check=True,
        )
        return completed.stdout.strip().splitlines()[-1]

    def test_single_node_override_exports_the_pinned_hca_list(self) -> None:
        line = self._profile_env(
            "source runtime/common.sh 2>/dev/null;"
            " export COLLX_SINGLE_NODE_RDMA_DEVICES='mlx5_12:1,mlx5_13:1';"
            " collx_apply_network_profile 1 nvlink;"
            " echo \"${NVSHMEM_HCA_LIST:-unset}\""
        )
        self.assertEqual(line, "mlx5_12:1,mlx5_13:1")

    def test_single_node_without_override_exports_nothing(self) -> None:
        line = self._profile_env(
            "source runtime/common.sh 2>/dev/null;"
            " collx_apply_network_profile 1 nvlink;"
            " echo \"${NVSHMEM_HCA_LIST:-unset}\""
        )
        self.assertEqual(line, "unset")

    def test_scale_out_ignores_the_single_node_selector(self) -> None:
        line = self._profile_env(
            "source runtime/common.sh 2>/dev/null;"
            " export COLLX_SINGLE_NODE_RDMA_DEVICES='mlx5_12:1';"
            " export COLLX_RDMA_DEVICES='mlx5_0:1,mlx5_1:1';"
            " collx_apply_network_profile 2 nvlink-rdma;"
            " echo \"${NVSHMEM_HCA_LIST:-unset}\""
        )
        self.assertEqual(line, "mlx5_0:1,mlx5_1:1")


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

# Probe output is consumed by the launcher to select an interface and link layer.
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

    def test_healthy_fabric_reports_selected_interface_and_link_layer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fabric(root)
            rc, lines = self._run(root, root / "route")
            self.assertEqual(rc, 0)
            self.assertEqual(self._captures(SOCKET_MARKER, lines), ["eth0"])
            self.assertEqual(self._captures(LINK_MARKER, lines), ["roce"])

    def test_inactive_port_reports_failure(self) -> None:
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


# config.py case-args is the single case→invocation codec: collx_run_shard decodes one
# null-delimited argv per case and hands it verbatim to bench/run_ep.py. Parse the
# emitted argv with the actual parser run_ep builds so the two sides cannot
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

    def _run_ep_parser(self) -> argparse.ArgumentParser:
        import run_ep

        # Capture the real entrypoint's parser before it initializes any GPU runtime.
        with mock.patch.object(argparse.ArgumentParser, "parse_args", autospec=True,
                               side_effect=SystemExit) as parse:
            with self.assertRaises(SystemExit):
                run_ep.main()
        return parse.call_args.args[0]

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
        # backend token or two legs of one cell collide in results/.
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
    def test_fp8_dispatch_and_bf16_combine_have_different_byte_counts(self) -> None:
        dispatch = ep_harness.logical_byte_provenance(
            logical_copies=10, hidden=128, value_bytes=1, scale_bytes_per_copy=8,
        )
        combine = ep_harness.logical_byte_provenance(logical_copies=10, hidden=128)
        self.assertEqual(dispatch, {
            "activation_data_bytes": 1280, "scale_bytes": 80, "total_logical_bytes": 1360,
        })
        self.assertEqual(combine, {
            "activation_data_bytes": 2560, "scale_bytes": 0, "total_logical_bytes": 2560,
        })

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

    def test_ladder_cap_drops_only_oversized_measurement_points(self):
        module = self._adapter()
        backend = module.DeepEPV2Backend.__new__(module.DeepEPV2Backend)
        backend.mode = "low-latency"
        backend.world_size = 8
        backend._build_rank_inputs = mock.Mock(return_value=None)
        args = types.SimpleNamespace(experts=256, tokens_ladder="32 64 128")
        with mock.patch.object(module, "_LL_LADDER_CAP", 64):
            spec = backend.make_inputs(args)
        self.assertEqual(spec.ladder, [32, 64])
        self.assertEqual(spec.dropped, [128])

    def test_the_receive_is_sized_from_the_buffer_cap_not_the_ladder(self):
        module = self._adapter()
        backend = module.DeepEPV2Backend.__new__(module.DeepEPV2Backend)
        backend.mode, backend.world_size, backend.group = "low-latency", 8, object()
        backend.args = types.SimpleNamespace(experts=256, hidden=16)
        vendor_buffer = mock.Mock()
        vendor_buffer.get_low_latency_rdma_size_hint.return_value = 4096
        with mock.patch.object(module.deep_ep, "Buffer", vendor_buffer), \
                mock.patch.object(module, "_LL_BUFFER_CAP", 128), \
                mock.patch.object(module, "_LL_LADDER_CAP", 64):
            for ladder_max in (16, 64):
                backend.create_buffer(types.SimpleNamespace(max_tokens_per_rank=ladder_max))
                vendor_buffer.get_low_latency_rdma_size_hint.assert_called_with(128, 16, 8, 256)
                self.assertEqual(vendor_buffer.call_args.kwargs["num_rdma_bytes"], 4096)
            with self.assertRaisesRegex(RuntimeError, "exceeds"):
                backend.create_buffer(types.SimpleNamespace(max_tokens_per_rank=129))

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
