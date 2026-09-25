"""Exercise the actual simple-slurm adapter against small executable scheduler doubles."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.network import network_environment, validated_selectors
from runtime.scheduler import SlurmAllocation


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.env = {**os.environ, "PATH": f"{self.bin}:{os.environ['PATH']}",
                    "RECORD": str(self.root / "record.json"), "PAYLOAD": "literal $value"}
        self.allocation = SlurmAllocation(self.root, self.env)

    def executable(self, name, body):
        path = self.bin / name
        path.write_text(f"#!{sys.executable}\nimport json, os, pathlib, sys\n" + body)
        path.chmod(0o755)

    def test_library_step_preserves_bare_flags_and_literal_arguments(self):
        self.executable("srun", '''
pathlib.Path(os.environ['RECORD']).write_text(json.dumps({
    'args': sys.argv[1:], 'payload': os.environ['PAYLOAD'], 'stdin': sys.stdin.read(),
}))
print('step output')
''')
        self.allocation.job_id = "321"
        path = self.root / "step.log"
        payload = "spaces; $(touch should-not-exist) 'quoted'"
        self.allocation.step(
            {"nodes": 2, "ntasks_per_node": 4, "container_writable": True,
             "container_mounts": "/some path:/ix", "no_container_entrypoint": True},
            ["python3", "bench.py", payload], path, stdin=b"node input",
        )
        record = json.loads((self.root / "record.json").read_text())
        self.assertEqual(record["args"][-3:], ["python3", "bench.py", payload])
        self.assertEqual(set(record["args"][:-3]), {
            "--jobid=321", "--nodes=2", "--ntasks-per-node=4", "--container-writable",
            "--container-mounts=/some path:/ix", "--no-container-entrypoint",
        })
        self.assertEqual((record["payload"], record["stdin"]), ("literal $value", "node input"))
        self.assertEqual(path.read_text(), "step output\n")
        self.assertFalse((Path.cwd() / "should-not-exist").exists())

    def test_step_failure_retains_exit_code_and_private_log(self):
        self.executable("srun", "print('native failure', file=sys.stderr)\nsys.exit(7)\n")
        self.allocation.job_id = "321"
        path = self.root / "failure.log"
        original = dict(os.environ)
        with self.assertRaises(subprocess.CalledProcessError) as caught:
            self.allocation.step({"nodes": 1}, ["python3", "bench.py"], path)
        self.assertEqual(caught.exception.returncode, 7)
        self.assertEqual(path.read_text(), "native failure\n")
        self.assertEqual(dict(os.environ), original)

    def test_allocation_recovery_waits_for_its_own_job_to_disappear(self):
        self.executable("salloc", "print('salloc: Granted job allocation 321', file=sys.stderr)\n")
        self.executable("scancel", "pathlib.Path(os.environ['RECORD']).write_text(json.dumps(sys.argv[1:]))\n")
        self.executable("squeue", '''
counter = pathlib.Path(os.environ['RECORD'] + '.queries')
if not counter.exists():
    counter.write_text('queried')
    print('321')
''')
        self.assertEqual(self.allocation.allocate(["--nodes=1"]), "321")
        self.assertEqual((self.root / "jobid").read_text(), "321\n")
        # A new cleanup process knows only the durable job-id record.
        recovery = SlurmAllocation(self.root, self.env)
        with mock.patch("runtime.scheduler.time.sleep"):
            recovery.release()
        self.assertEqual(json.loads((self.root / "record.json").read_text()), ["321"])
        self.assertFalse((self.root / "jobid").exists())


class NetworkEnvironmentTests(unittest.TestCase):
    def test_scale_out_preserves_exact_port_selectors_and_roce_gid(self):
        env = network_environment({
            "COLLX_RDMA_DEVICES": "mlx5_1:2,mlx5_10", "COLLX_SOCKET_IFNAME": "eth0",
            "COLLX_IB_GID_INDEX": "3", "COLLX_RDMA_LINK_LAYER": "roce",
            "COLLX_RAIL_ISOLATED": "1", "NCCL_NET_PLUGIN": "stale",
        }, 2, "nvlink-rdma")
        self.assertEqual(env["NCCL_IB_HCA"], "=mlx5_1:2,mlx5_10")
        self.assertEqual(env["UCCL_IB_HCA"], "=mlx5_1:2,mlx5_10")
        self.assertEqual(env["MORI_RDMA_DEVICES"], "mlx5_1,mlx5_10")
        self.assertEqual(env["UCCL_IB_GID_INDEX"], "3")
        self.assertEqual(env["NCCL_CROSS_NIC"], "0")
        self.assertNotIn("NCCL_NET_PLUGIN", env)

    def test_efa_does_not_retain_verbs_selectors(self):
        env = network_environment({
            "COLLX_RDMA_DEVICES": "rdmap1s0,rdmap2s0", "COLLX_RDMA_FABRIC": "efa",
            "NCCL_IB_HCA": "wrong", "NVSHMEM_IB_ENABLE_IBGDA": "1",
        }, 2, "nvlink-rdma")
        self.assertEqual(env["NCCL_NET_PLUGIN"], "ofi")
        self.assertEqual(env["NVSHMEM_LIBFABRIC_PROVIDER"], "efa")
        self.assertNotIn("NCCL_IB_HCA", env)
        self.assertNotIn("NVSHMEM_IB_ENABLE_IBGDA", env)

    def test_different_node_interfaces_are_resolved_on_each_rank(self):
        output = "\n".join([
            "[collectivex-private] socket-interface-selected=eth0",
            "[collectivex-private] socket-interface-selected=eth1",
            "[collectivex-private] rdma-link-layer=infiniband",
            "[collectivex-private] rdma-link-layer=infiniband",
        ])
        env = validated_selectors(output, 2, {"COLLX_SOCKET_IFNAME": "stale"})
        self.assertNotIn("COLLX_SOCKET_IFNAME", env)
        self.assertEqual(env["COLLX_RDMA_LINK_LAYER"], "infiniband")
        with self.assertRaisesRegex(RuntimeError, "disagree"):
            validated_selectors(output.replace("infiniband", "roce", 1), 2, {})
