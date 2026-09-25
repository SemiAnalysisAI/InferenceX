"""Compute-host transport, image cache, and per-rank bootstrap behavior."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import build, node, storage
from runtime.scheduler import SlurmAllocation


class NodeTests(unittest.TestCase):
    def test_rank_cli_executes_the_benchmark_with_its_unmodified_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(
                Path(node.__file__).parent,
                root / "runtime",
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            binary = root / "bin"
            binary.mkdir()
            interpreter = binary / "python3"
            interpreter.write_text(
                f"#!{sys.executable}\nimport json, os, sys\n"
                "print(json.dumps({'args': sys.argv[1:], 'rank': os.environ['RANK']}))\n"
            )
            interpreter.chmod(0o755)
            build.write_rank_environment(root, "0", "mori", {})
            result = subprocess.run(
                [
                    sys.executable,
                    str(root / "runtime/node.py"),
                    "rank",
                    "--",
                    "--backend",
                    "mori",
                    "--mode",
                    "normal",
                ],
                env={
                    **os.environ,
                    "PATH": f"{binary}:{os.environ['PATH']}",
                    "SLURM_NODEID": "0",
                    "SLURM_PROCID": "2",
                    "SLURM_LOCALID": "2",
                    "SLURM_NTASKS": "8",
                    "COLLX_NGPUS": "8",
                    "COLLX_GPUS_PER_NODE": "8",
                    "COLLX_NODES": "1",
                },
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertEqual(
                json.loads(result.stdout),
                {
                    "args": ["bench/run_ep.py", "--backend", "mori", "--mode", "normal"],
                    "rank": "2",
                },
            )

    def test_remote_zipapp_runs_without_reading_the_staged_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "bin"
            binary.mkdir()
            for name, body in {
                "srun": "args = sys.argv[1:]\nwhile args[0].startswith('--'): args.pop(0)\nos.execvp(args[0], args)\n",
                "hostname": "print('compute-zero')\n",
            }.items():
                path = binary / name
                path.write_text(f"#!{sys.executable}\nimport os, sys\n" + body)
                path.chmod(0o755)
            allocation = SlurmAllocation(
                root, {**os.environ, "PATH": f"{binary}:{os.environ['PATH']}"}
            )
            allocation.job_id = "321"
            allocation.host(1, ["address", ""], root / "probe.log")
            self.assertEqual((root / "probe.log").read_text(), "compute-zero\n")

    def test_rank_identity_comes_from_slurm_and_only_backend_fields_are_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build.write_rank_environment(
                root,
                "1",
                "deepep-v2",
                {
                    "PATH": "/venv/bin:/bin",
                    "EP_REUSE_NCCL_COMM": "1",
                    "SECRET": "not-forwarded",
                },
            )
            incoming = {
                "SLURM_NODEID": "1",
                "SLURM_PROCID": "5",
                "SLURM_LOCALID": "1",
                "SLURM_NTASKS": "8",
                "COLLX_NGPUS": "8",
                "COLLX_GPUS_PER_NODE": "4",
                "COLLX_NODES": "2",
                "COLLX_TRANSPORT": "mnnvl",
                "EP_SUPPRESS_NCCL_CHECK": "1",
                "RANK": "wrong",
                "PATH": "/bin",
            }
            env = node.rank_environment(root, incoming)
            self.assertEqual(
                (env["RANK"], env["LOCAL_RANK"], env["WORLD_SIZE"], env["LOCAL_WORLD_SIZE"]),
                ("5", "1", "8", "4"),
            )
            self.assertEqual(env["PATH"], "/venv/bin:/bin")
            self.assertEqual(env["EP_REUSE_NCCL_COMM"], "1")
            self.assertNotIn("EP_SUPPRESS_NCCL_CHECK", env)
            self.assertNotIn("SECRET", env)
            with self.assertRaises(SystemExit) as caught:
                node.rank_environment(root, {**incoming, "SLURM_LOCALID": "4"})
            self.assertEqual(caught.exception.code, 67)

    def test_bad_rank_environment_fails_before_starting_a_gpu_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = root / ".collx_backend/env/node-0.json"
            record.parent.mkdir(parents=True)
            record.write_text(json.dumps({"set": {"ARBITRARY": "value"}, "unset": []}))
            with self.assertRaises(SystemExit) as caught:
                node.rank_environment(root, {"SLURM_NODEID": "0"})
            self.assertEqual(caught.exception.code, 66)


class ImageCacheTests(unittest.TestCase):
    def test_import_reuses_a_digest_and_rebuilds_only_when_the_tag_moves(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "bin"
            binary.mkdir()
            programs = {
                "enroot": """
if sys.argv[1] == 'import':
    pathlib.Path(sys.argv[sys.argv.index('-o') + 1]).write_text('image')
    with open(os.environ['IMPORTS'], 'a') as stream: stream.write('imported\\n')
""",
                "unsquashfs": "sys.exit(0 if pathlib.Path(sys.argv[-1]).is_file() else 1)\n",
            }
            for name, body in programs.items():
                path = binary / name
                path.write_text(f"#!{sys.executable}\nimport os, pathlib, sys\n" + body)
                path.chmod(0o755)
            image = storage.squash_path(root, "some/image:tag", "linux/amd64")
            options = {
                "path": str(image),
                "lock": str(root / "image.lock"),
                "platform": "linux/amd64",
                "mode": "local",
                "image": "some/image:tag",
                "digest": "first",
                "local_scratch": True,
            }
            env = {
                **os.environ,
                "PATH": f"{binary}:{os.environ['PATH']}",
                "IMPORTS": str(root / "imports"),
            }
            with mock.patch("runtime.storage.platform.machine", return_value="x86_64"):
                storage.import_image(options, env)
                storage.import_image(options, env)
                storage.import_image({**options, "digest": "second"}, env)
            self.assertEqual((root / "imports").read_text(), "imported\nimported\n")
            self.assertEqual(Path(f"{image}.digest").read_text(), "second\n")
            self.assertEqual(image.name, "_some_image_tag.sqsh")
            self.assertEqual(image.stat().st_mode & 0o444, 0o444)

    def test_architecture_failure_and_incomplete_digest_keep_their_original_semantics(self):
        with mock.patch("runtime.storage.platform.machine", return_value="arm64"):
            with self.assertRaises(storage.ImportFailure) as caught:
                storage.import_image({"platform": "linux/amd64"}, {})
        self.assertEqual(caught.exception.status, 13)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.sqsh"
            path.touch()
            Path(f"{path}.digest").write_text("incomplete")
            self.assertEqual(storage.squash_verdict(path, "different", None), "reuse")
            os.utime(path, (100, 100))
            self.assertEqual(storage.squash_verdict(path, "different", 101), "refresh-requested")

    def test_staged_results_remain_collectable_on_a_failed_case(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, checkout = root / "stage", root / "checkout"
            result = source / "experimental/CollectiveX/results/case.json"
            result.parent.mkdir(parents=True)
            result.write_text('{"outcome":{"status":"invalid"}}\n')
            storage.collect_results(source, checkout)
            copied = checkout / "experimental/CollectiveX/results/case.json"
            self.assertEqual(copied.read_text(), '{"outcome":{"status":"invalid"}}\n')
            storage.cleanup_stage(source, checkout)
            self.assertFalse(source.exists())
            self.assertTrue(copied.is_file())
