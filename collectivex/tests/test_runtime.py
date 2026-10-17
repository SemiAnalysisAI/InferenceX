#!/usr/bin/env python3
"""Focused tests for the standalone runtime helpers."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
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
import ep_oracle  # noqa: E402


def run_common(script: str, *args: str, **kwargs) -> subprocess.CompletedProcess:
    """Run `script` in bash after sourcing runtime/common.sh; `args` arrive as $1, $2, ..."""
    return subprocess.run(
        ["bash", "-c", f'source "{RUNTIME / "common.sh"}" && {script}', "collx", *args],
        capture_output=True, text=True, **kwargs,
    )


class ProbeTests(unittest.TestCase):
    def test_prepare_cache_bootstraps_a_missing_squash_dir_privately_and_reuses_it(self) -> None:
        # The probe runs before the first container import, so on a fresh pool squash_dir does
        # not exist yet.
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "sqsh"
            cache = Path(probe.prepare_cache(str(parent)))
            self.assertEqual(Path(probe.prepare_cache(str(parent))), cache)
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


# A wrong or accepted-malformed digest silently turns stage-once-reuse-many back into
# per-launch imports.
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


# The staged squash path is a pure function of platform + image reference (keying it by run
# re-imports tens of GB per run; by digest, a registry blip misses the staged file), and
# freshness is the verdict against the digest sidecar.
class SquashCacheKeyTests(unittest.TestCase):
    IMAGE = "rocm/sgl-dev:sglang-v1"
    DIGEST = "sha256:" + "ab" * 32

    def _bash(self, script: str, env: dict, args: list) -> str:
        return run_common(
            script, *args, check=True,
            env={"PATH": os.environ["PATH"], "COLLX_IMAGE_PLATFORM": "linux/amd64", **env},
        ).stdout

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
    # The single-node early return must still honor a SKU's pinned single-node HCA list (DeepEP's
    # legacy LL Buffer self-enables IBGDA even single-node), while scale-out keeps its selector.
    def test_nvshmem_hca_list_per_placement(self) -> None:
        for name, setup, placement, expected in (
            ("single-node override", "export COLLX_SINGLE_NODE_RDMA_DEVICES='mlx5_12:1,mlx5_13:1'",
             "1 nvlink", "mlx5_12:1,mlx5_13:1"),
            ("single-node without override", ":", "1 nvlink", "unset"),
            ("scale-out ignores the single-node selector",
             "export COLLX_SINGLE_NODE_RDMA_DEVICES='mlx5_12:1'"
             " COLLX_RDMA_DEVICES='mlx5_0:1,mlx5_1:1'",
             "2 nvlink-rdma", "mlx5_0:1,mlx5_1:1"),
        ):
            with self.subTest(name):
                stdout = run_common(
                    f'{setup}; collx_apply_network_profile {placement};'
                    ' echo "${NVSHMEM_HCA_LIST:-unset}"',
                    check=True,
                ).stdout
                self.assertEqual(stdout.strip().splitlines()[-1], expected)


class RelaxedOrderingProfileTests(unittest.TestCase):
    def test_only_the_registry_flag_relaxes_the_deepep_window(self) -> None:
        for flag, expected in (("1", "1"), ("0", "unset"), ("", "unset")):
            with self.subTest(flag=flag):
                stdout = run_common(
                    "export COLLX_RDMA_DEVICES=mlx5_0:1 EP_WIN_RELAXED_ORDERING=stale"
                    f" COLLX_RDMA_RELAXED_ORDERING='{flag}';"
                    " collx_apply_network_profile 2 nvlink-rdma;"
                    ' echo "${EP_WIN_RELAXED_ORDERING:-unset}"',
                    check=True,
                ).stdout
                self.assertEqual(stdout.strip().splitlines()[-1], expected)


class EfaPluginProfileTests(unittest.TestCase):
    def test_ranks_load_the_prepared_plugin_and_proxy_gin(self) -> None:
        lib = "/cx-cache/ofi/lib"
        for lib_dir, expected in (
            (lib, f"{lib}/libnccl-net-ofi.so {lib}/libnccl-ofi-tuner.so 2"),
            ("", "ofi unset 2"),
        ):
            with self.subTest(lib_dir=lib_dir):
                stdout = run_common(
                    "export COLLX_RDMA_FABRIC=efa COLLX_RDMA_DEVICES=rdmap86s0"
                    f" NCCL_TUNER_PLUGIN=stale COLLX_OFI_NCCL_LIB_DIR='{lib_dir}';"
                    " collx_apply_network_profile 2 nvlink-rdma;"
                    ' echo "$NCCL_NET_PLUGIN ${NCCL_TUNER_PLUGIN:-unset} $NCCL_GIN_TYPE"',
                    check=True,
                ).stdout
                self.assertEqual(stdout.strip().splitlines()[-1], expected)

    @staticmethod
    def _loader_path(root: str, start: str) -> str:
        """LD_LIBRARY_PATH after the rank wrapper's profile, re-applied over what backend
        preparation persisted (`start`), with the host EFA user space bound under `root`."""
        return run_common(
            'COLLX_EFA_HOST_ROOT="$1" LD_LIBRARY_PATH="$2";'
            " export COLLX_RDMA_FABRIC=efa COLLX_RDMA_DEVICES=rdmap86s0;"
            " collx_apply_network_profile 2 nvlink-rdma; collx_apply_network_profile 2 nvlink-rdma;"
            ' echo "$LD_LIBRARY_PATH"',
            root, start, check=True,
        ).stdout.strip().splitlines()[-1]

    def test_the_host_efa_user_space_leads_the_loader_path_once(self) -> None:
        # Persisted rank environments already carry the prefix behind the backend's own NCCL, so
        # it is moved to the front rather than repeated, and an empty start never leaves an empty
        # (current-directory) entry behind.
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "lib").mkdir()
            host = [f"{root}/lib", f"{root}/libdir/libibverbs",
                    "/opt/amazon/efa/lib", "/opt/amazon/ofi-nccl/lib"]
            for start, rest in (
                ("", []),
                ("/usr/local/cuda/lib64", ["/usr/local/cuda/lib64"]),
                (f"/venv/nccl/lib:{root}/lib:/usr/local/cuda/lib64:/opt/amazon/efa/lib",
                 ["/venv/nccl/lib", "/usr/local/cuda/lib64"]),
            ):
                with self.subTest(start=start):
                    self.assertEqual(self._loader_path(root, start), ":".join(host + rest))

    def test_outside_a_container_the_loader_path_is_untouched(self) -> None:
        # The launcher applies the same profile on the login node, which has no binds.
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(self._loader_path(root, "/usr/local/cuda/lib64"),
                             "/usr/local/cuda/lib64")


class EfaHostMountTests(unittest.TestCase):
    LIBDIR = "/usr/lib/x86_64-linux-gnu"

    @staticmethod
    def _mounts(fabric: str, nodes: int, platform: str = "linux/amd64", root: str = ""):
        return run_common(
            '[ -z "$1" ] || COLLX_EFA_HOST_ROOT="$1";'
            f' COLLX_RDMA_FABRIC="{fabric}" NODES={nodes} COLLX_TRANSPORT=nvlink-rdma'
            f" COLLX_IMAGE_PLATFORM={platform} CONTAINER_MOUNTS=/stage:/ix;"
            ' collx_add_efa_host_mounts && echo "$CONTAINER_MOUNTS"',
            root,
        )

    def test_multi_node_efa_shards_bind_the_host_user_space_read_only(self) -> None:
        mounts = self._mounts("efa", 2).stdout.strip().split(",")
        self.assertEqual(mounts[0], "/stage:/ix")
        expected = [
            "/opt/amazon/efa:/opt/amazon/efa:ro",
            "/opt/amazon/ofi-nccl:/opt/amazon/ofi-nccl:ro",
            f"{self.LIBDIR}:/opt/collectivex/host-efa/libdir:ro",
        ]
        # NCCL dlopens the linker name before the soname; both must reach the host copy.
        for name in ("libibverbs", "libefa", "libmlx5"):
            expected += [f"{self.LIBDIR}/{name}.so.1:/opt/collectivex/host-efa/lib/{target}:ro"
                         for target in (f"{name}.so.1", f"{name}.so")]
        self.assertEqual(mounts[1:], expected)

    def test_single_node_and_non_efa_shards_keep_their_mounts(self) -> None:
        for fabric, nodes in (("efa", 1), ("", 2)):
            with self.subTest(fabric=fabric, nodes=nodes):
                result = self._mounts(fabric, nodes)
                self.assertEqual(result.returncode, 0)
                self.assertEqual(result.stdout.strip(), "/stage:/ix")

    def test_an_unknown_image_platform_fails_closed(self) -> None:
        result = self._mounts("efa", 2, platform="linux/riscv64")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no host EFA user space layout", result.stderr)

    def test_the_profile_loader_path_is_what_the_mounts_provide(self) -> None:
        # Every directory the EFA profile puts first on the loader path is a bound directory, a
        # subdirectory of one, or the directory the soname binds land in; the host's whole libdir
        # is bound only so its providers resolve, and never reaches the loader path itself.
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "lib").mkdir()
            mounts = self._mounts("efa", 2, root=root).stdout.strip().split(",")[1:]
            targets = [mount.split(":")[1] for mount in mounts]
            prefix = EfaPluginProfileTests._loader_path(root, "").split(":")
            for entry in prefix:
                with self.subTest(entry=entry):
                    self.assertTrue(any(
                        entry == target or entry.startswith(target + "/")
                        or os.path.dirname(target) == entry for target in targets))
            self.assertNotIn(f"{root}/libdir", prefix)


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
                "Args", (), {"source": str(source), "target": str(target / "collectivex")}
            )
            stage.copy_repository(copy_args)
            staged = target / "collectivex"
            self.assertTrue((staged / "runtime" / "common.sh").is_file())
            self.assertFalse((staged / ".shards").exists())
            self.assertFalse((staged / "goal.md").exists())


class NetworkProfileContract(unittest.TestCase):
    """The marker lines the launcher reads to select an interface and link layer."""

    @staticmethod
    def _fabric(root: Path, *, interface: str = "eth0", device: str = "mlx5_0",
                state: str = "4: ACTIVE", link_layer: str = "Ethernet",
                gid: str | None = "fe80::1") -> None:
        net = root / "class" / "net" / interface
        net.mkdir(parents=True)
        (net / "operstate").write_text("up\n")
        port = root / "class" / "infiniband" / device / "ports" / "1"
        (port / "gids").mkdir(parents=True)
        (port / "state").write_text(state + "\n")
        (port / "link_layer").write_text(link_layer + "\n")
        if gid is not None:
            (port / "gids" / "3").write_text(gid + "\n")

    @staticmethod
    def _run(root: Path, *profile: str):
        """validate_network_profile over `root` (default: eth0 + mlx5_0:1 at GID index 3)."""
        buffer = io.StringIO()
        rc = 0
        try:
            with contextlib.redirect_stdout(buffer):
                probe.validate_network_profile(*(profile or ("eth0", "mlx5_0:1", "3")),
                                                sys_root=root, route_path=root / "route")
        except SystemExit:
            rc = 1
        return rc, buffer.getvalue().splitlines()

    def test_healthy_fabric_reports_selected_interface_and_link_layer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fabric(root)
            rc, lines = self._run(root)
        self.assertEqual(rc, 0)
        self.assertEqual(lines, ["[collectivex-private] socket-interface-selected=eth0",
                                 "[collectivex-private] rdma-link-layer=roce"])

    def test_inactive_port_reports_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fabric(root, state="1: DOWN")
            rc, lines = self._run(root)
        self.assertEqual(rc, 1)
        self.assertIn("[collectivex-private] rdma-port-1=inactive", lines)

    def _efa_fabric(self, root: Path) -> None:
        # An EFA node as sysfs shows it: default-route interface up, verbs device whose port is
        # ACTIVE but carries link_layer Unspecified and no usable GID table (rdma-core -> rdmap*).
        self._fabric(root, interface="enp71s0", device="rdmap86s0",
                     link_layer="Unspecified", gid=None)

    def _run_efa(self, root: Path, fabric: str):
        return self._run(root, "enp71s0", "rdmap86s0", "", fabric)

    def test_declared_efa_fabric_accepts_the_unspecified_link_layer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._efa_fabric(root)
            rc, lines = self._run_efa(root, "efa")
        self.assertEqual(rc, 0, lines)
        self.assertEqual(lines, ["[collectivex-private] socket-interface-selected=enp71s0",
                                 "[collectivex-private] rdma-link-layer=efa"])

    def test_undeclared_fabric_still_rejects_the_unspecified_link_layer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._efa_fabric(root)
            rc, lines = self._run_efa(root, "")
        self.assertEqual(rc, 1)
        self.assertIn("[collectivex-private] rdma-port-1=link-layer-invalid", lines)


class ContainerImportRetry(unittest.TestCase):
    """A failed container import is retried: on soft-mounted squash storage (gb300's NFS over
    RDMA) one failure is usually the transport blinking, not a missing mount."""

    HARNESS = """
set -u
export COLLX_IMAGE_PLATFORM=linux/amd64
export COLLX_LOG_DIR="$ROOT/logs"
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
sleep() { :; }              # collapse the backoff
unsquashfs() { return 0; }  # a present squash short-circuits the import
out="$(collx_ensure_squash_on_job 12345 "$ROOT/sqsh" some/image:tag)"; rc=$?
echo "RC=$rc"
echo "OUT=$out"
echo "CALLS=$(wc -l < "$ROOT/calls" 2>/dev/null | tr -d ' ' || echo 0)"
"""

    def _run(self, rc_sequence: str):
        with tempfile.TemporaryDirectory() as root:
            proc = run_common(self.HARNESS, env={
                **os.environ, "ROOT": root,
                "RC_SEQUENCE": rc_sequence, "COLLX_IMPORT_ATTEMPTS": "3",
            })
        fields = dict(
            line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line
            and line.split("=", 1)[0] in ("RC", "OUT", "CALLS")
        )
        return fields, proc

    def test_a_transient_failure_is_retried_and_then_succeeds(self):
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


# config.py case-args encodes each case as a null-delimited argv for the entrypoint its leading
# --entrypoint pair names; parsing it with that entrypoint's real parser catches a flag one side
# drops before a GPU allocation does.
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

    def _decode(self, stdout: bytes, entrypoint: str = "run_ep") -> list:
        parts = stdout.split(b"\0")
        self.assertEqual(parts[-1], b"")
        argv = [part.decode() for part in parts[:-1]]
        self.assertEqual(argv[:2], ["--entrypoint", entrypoint])
        return argv[2:]

    def _case_argv(self, placement: list, case: dict | None = None,
                   entrypoint: str = "run_ep") -> list:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shard.json"
            path.write_text(json.dumps({"version": 1, "cases": [case or self.CASE]}))
            result = subprocess.run(
                [sys.executable, str(RUNTIME / "config.py"), "case-args",
                 str(path), "0", "h200-dgxc", *placement],
                capture_output=True, check=True,
            )
        return self._decode(result.stdout, entrypoint)

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
        self.assertEqual(args.out, f"results/h200-dgxc/{self.CASE['case_id']}.json")

    def test_a_legacy_colon_string_profile_still_decodes(self) -> None:
        # A shard staged before the timing object must still produce a runnable argv: six
        # positional fields, or the pre-chain three, whose chain knobs run_ep's defaults supply.
        for profile, chain in (("8:256:32:128:4:16", (128, 4, 16)), ("8:256:32", None)):
            with self.subTest(timing=profile):
                argv = self._case_argv(["16", "2", "8", "8"], case={**self.CASE, "timing": profile})
                args = self._run_ep_parser().parse_args(argv)
                self.assertEqual((args.iters, args.trials, args.warmup), (8, 256, 32))
                if chain:
                    self.assertEqual(
                        (args.chain_iters, args.chain_trials, args.chain_drop), chain
                    )
                else:
                    self.assertEqual([flag for flag in argv if flag.startswith("--chain")], [])

    def test_a_malformed_timing_profile_cannot_reach_a_run(self) -> None:
        # An unknown, renamed or missing key must be as fatal as a bad string arity, or a
        # renamed knob silently falls back to run_ep's default.
        for timing in (
            {"iters_per_trial": 8}, {**self.CASE["timing"], "extra": 1}, {},
            "8:256", "8:256:32:128", "8:256:32:128:4:16:2", "",
        ):
            with self.subTest(timing=timing):
                with self.assertRaises(subprocess.CalledProcessError):
                    self._case_argv(["16", "2", "8", "8"], case={**self.CASE, "timing": timing})
        # The codec does not check types, so a non-integer must instead fail run_ep's parser.
        argv = self._case_argv(
            ["16", "2", "8", "8"], case={**self.CASE, "timing": "8:256:32:128:4:x"},
        )
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self._run_ep_parser().parse_args(argv)

    def test_case_args_fails_closed_on_placement_mismatch(self) -> None:
        with self.assertRaises(subprocess.CalledProcessError):
            self._case_argv(["8", "1", "8", "8"])

    def test_a_swap_blocks_case_round_trips_through_its_own_parser(self) -> None:
        import run_swap_blocks
        import sweep_matrix

        platform = {"h200-dgxc": {"arch": "sm90", "launcher": "single-slurm"}}
        with mock.patch.object(sweep_matrix, "PLATFORMS", platform):
            shard, = sweep_matrix.resolve_matrix(suites="swap-blocks")["include"]
        case = shard["cases"][1]
        argv = self._case_argv(["1", "1", "1", "1"], case=case, entrypoint="run_swap_blocks")
        with mock.patch.object(argparse.ArgumentParser, "parse_args", autospec=True,
                               side_effect=SystemExit) as parse:
            with self.assertRaises(SystemExit):
                run_swap_blocks.main(argv)
        args = parse.call_args.args[0].parse_args(argv)
        self.assertEqual(args.directions, case["directions"].split())
        self.assertEqual(args.block_bytes, [int(v) for v in case["block_bytes"].split()])
        self.assertEqual(args.num_blocks, [int(v) for v in case["num_blocks"].split()])
        self.assertEqual((args.layout, args.warmup, args.iterations),
                         (case["layout"], case["warmup"], case["iterations"]))
        self.assertEqual(args.max_payload_bytes, case["max_payload_bytes"])
        self.assertEqual((args.case_id, args.profile, args.sku),
                         (case["case_id"], case["profile"], "h200-dgxc"))
        self.assertEqual(str(args.output), f"results/h200-dgxc/{case['case_id']}.json")
        with self.assertRaises(subprocess.CalledProcessError):
            self._case_argv(["8", "1", "8", "8"], case=case, entrypoint="run_swap_blocks")

    def test_a_case_from_an_unknown_suite_cannot_reach_a_run(self) -> None:
        with self.assertRaises(subprocess.CalledProcessError):
            self._case_argv(["16", "2", "8", "8"], case={**self.CASE, "suite": "turbo"})

    def test_the_rank_wrapper_execs_only_a_named_entrypoint(self) -> None:
        # The wrapper's own guards run before it sources anything, so feed it valid Slurm
        # identity and stop at the entrypoint gate with a python3 shim that records its argv.
        with tempfile.TemporaryDirectory() as directory:
            shim = Path(directory) / "python3"
            shim.write_text('#!/bin/sh\necho "ARGV $*"\n')
            shim.chmod(0o755)
            wrapper = run_common("collx_slurm_rank_wrapper").stdout.replace(
                ". /ix/collectivex/runtime/common.sh || exit 68", ":")
            env = {**os.environ, "PATH": f"{directory}:{os.environ['PATH']}",
                   "SLURM_PROCID": "0", "SLURM_NTASKS": "1", "SLURM_LOCALID": "0",
                   "SLURM_NODEID": "0", "COLLX_NGPUS": "1", "COLLX_GPUS_PER_NODE": "1"}
            for argv, rc, out in (
                (["--entrypoint", "run_swap_blocks", "--layout", "random"], 0,
                 "ARGV bench/run_swap_blocks.py --layout random"),
                (["--entrypoint", "run_ep", "--backend", "mori"], 0,
                 "ARGV bench/run_ep.py --backend mori"),
                (["--entrypoint", "../../bin/sh"], 67, ""),
                (["--backend", "mori"], 67, ""),
            ):
                with self.subTest(argv=argv):
                    result = subprocess.run(["bash", "-c", wrapper, "_", *argv], env=env,
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, rc, result.stderr)
                    self.assertEqual(result.stdout.strip(), out)


# logical_byte_provenance is where FP8 changes measurement semantics (asymmetric
# per-direction byte counts).
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


class WeightedCombineSemanticsTests(unittest.TestCase):
    """Normal mode folds the gate weight INTO the staged transform (kernel sums); low-latency
    stages the UNWEIGHTED transform and the kernel applies the gate."""

    @unittest.skipUnless(_torch is not None, "combine-oracle math checks require torch")
    def test_transform_folds_the_gate_under_unweighted_rank_sum(self):
        torch = _torch
        payload = torch.randn(3, 64, dtype=torch.bfloat16)
        ids = torch.tensor([[2, -1], [5, -1], [7, -1]], dtype=torch.int64)
        model = ep_oracle.combine_model("unweighted-rank-sum")
        low = model.transform(torch, payload, ids, torch.full((3, 2), 0.2))
        high = model.transform(torch, payload, ids, torch.full((3, 2), 0.9))
        # The gate IS in the transform here, so a larger weight changes the staged value.
        self.assertFalse(torch.equal(low, high))

    def test_unknown_semantics_fail_closed(self):
        with self.assertRaises(ValueError):
            ep_oracle.combine_model("made-up")
        with self.assertRaises(ValueError):
            ep_oracle.combine_model("unweighted-rank-sum", "made-up")


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
        return ep_oracle.topk_slot_tree_combine(
            torch, destination, torch.ones_like(destination, dtype=torch.bool),
            messages, torch.bfloat16,
        ).item()

    def test_matches_the_value_the_kernel_returns(self):
        self.assertEqual(self._tree([1.0] + [2.0**-9] * 7), 1.0078125)


@unittest.skipUnless(_torch is not None, "combine-oracle math checks require torch")
class RankFp32ReductionTests(unittest.TestCase):
    """Rank-major combine keeps its unique-rank accumulator in FP32."""

    def test_accumulates_unique_ranks_in_topk_order(self):
        torch = _torch
        values = [1.0] + [2.0**-9] * 7
        messages = torch.tensor(values, dtype=torch.float32).reshape(8, 1, 1)
        destination = torch.arange(8).unsqueeze(0)
        valid = torch.ones_like(destination, dtype=torch.bool)
        result = ep_oracle.topk_rank_fp32_combine(
            torch, destination, valid, messages
        )
        self.assertEqual(result.item(), 1.0 + 7 * 2.0**-9)

        duplicate = torch.tensor([[0, 0, 1]])
        result = ep_oracle.topk_rank_fp32_combine(
            torch, duplicate, torch.ones_like(duplicate, dtype=torch.bool), messages
        )
        self.assertEqual(result.item(), 1.0 + 2.0**-9)


class FusedQuantizeGate(unittest.TestCase):
    """The oracle's payload gate compares the sender's [T, hidden] quantize against the oracle's
    [receive_count, hidden] one, so a fused callable must be bit-identical and per-row invariant.
    Both methods read only `self.mode` and the LL flag, so they are called unbound."""

    def test_low_latency_keeps_the_eager_helper(self):
        # Its dispatch kernel quantises internally and the oracle gate is pinned to those bits.
        def eager(x):
            return x, x
        backend = types.SimpleNamespace(mode="low-latency", ll_quantizes_in_kernel=True)
        self.assertIs(ep_backend.EPBackend.fused_quantize(backend, eager), eager)

    @unittest.skipUnless(_torch is not None, "quantize-identity checks require torch")
    def test_identity_check_rejects_a_divergent_callable(self):
        torch = _torch
        x = torch.randn(8, 256, dtype=torch.bfloat16)

        def eager(t):
            return t.to(torch.float8_e4m3fn), t.float().abs().amax(dim=1)

        def divergent(t):
            values, scales = eager(t)
            return values, scales + 1          # one differing scale is enough to red a cell
        with self.assertRaises(RuntimeError):
            ep_backend.EPBackend.assert_quantize_identity(
                types.SimpleNamespace(mode="normal"), eager, divergent, x)


class GpuHealthProbe(unittest.TestCase):
    """Reject an allocation holding a throttled GPU before it burns the wall-clock guard: one
    clamped device paces every rank (a B200 at 120 MHz against 1965 MHz ran a case 17x slower)."""

    HEALTHY = "\n".join(f"{i}, Not Active, Not Active, 3{i} " for i in range(8))

    def _swap(self, line_in: str, line_out: str) -> str:
        return self.HEALTHY.replace(line_in, line_out)

    def test_a_clamped_gpu_is_rejected_by_either_signal(self):
        # The flag can clear between samples while the fault persists, so heat alone rejects, as
        # does either flag alone. "Not Active" must not read as "Active".
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

    @staticmethod
    def _run_validate(csv: str):
        """Drive validate_gpu_health with a stubbed nvidia-smi; returns (exit_code, stdout)."""
        captured = io.StringIO()
        with mock.patch("shutil.which", return_value="/usr/bin/nvidia-smi"), \
                mock.patch("subprocess.run", return_value=types.SimpleNamespace(stdout=csv)), \
                contextlib.redirect_stdout(captured):
            try:
                probe.validate_gpu_health()
                code = 0
            except SystemExit as exit_:
                code = exit_.code
        return code, captured.getvalue()

    def test_a_fault_exits_nonzero_and_names_the_gpu(self):
        code, out = self._run_validate(
            self._swap("7, Not Active, Not Active, 37 ", "7, Active, Active, 93 ")
        )
        self.assertEqual(code, 1)
        self.assertIn("gpu-health-fault gpu 7", out)
        self.assertNotIn("gpu-health-checked", out)

    def test_the_temperature_spread_is_reported_but_never_gated(self):
        # A software-throttled H100 never crosses the 90 C limit and shows only as an idle
        # outlier; the spread is reported for a human, never gated, and absent when unreadable.
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


if __name__ == "__main__":
    unittest.main()
