"""Execute the real Python CLI across allocation, retry, case failure, and cancellation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.probe import network_environment, validated_selectors
from runtime.scheduler import SlurmAllocation

# External scheduler/container double. The implementation under test still creates the
# allocation, resolves the profile, stages files, drives simple-slurm, and handles cleanup.
SCHEDULER = r"""
import json, os, pathlib, signal, sys, time
tool = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
state = pathlib.Path(os.environ['SCHEDULER_STATE'])
with (state / 'calls').open('a') as stream:
    stream.write(json.dumps({'tool': tool, 'args': args}) + '\n')
def option(name):
    return next((arg.split('=', 1)[1] for arg in args if arg.startswith('--' + name + '=')), '')
if tool == 'timeout':
    start = args.index('srun')
    os.execvp(args[start], args[start:])
if tool == 'salloc':
    count = state / 'allocations'
    value = int(count.read_text()) + 1 if count.exists() else 321
    count.write_text(str(value))
    print('salloc: Granted job allocation ' + str(value), file=sys.stderr)
elif tool == 'scancel':
    job = args[0]
    pid = state / ('worker-' + job)
    if pid.exists():
        try: os.kill(int(pid.read_text()), signal.SIGTERM)
        except ProcessLookupError: pass
    (state / ('cancelled-' + job)).touch()
elif tool == 'squeue':
    job = args[args.index('-j') + 1]
    if args[-1] == '%N': print('node' + job)
    elif not (state / ('cancelled-' + job)).exists(): print(job)
elif tool == 'scontrol':
    print(args[-1])
elif tool == 'srun':
    job = option('jobid')
    command = args[next(i for i, value in enumerate(args) if not value.startswith('--')):]
    if command[0] == 'enroot': sys.exit(0)
    if command[1] == '-c':
        operation = command[3]
        if operation == 'address':
            print('srun: informational scheduler diagnostic', file=sys.stderr)
            print('[collectivex-private] rendezvous=compute-zero')
        elif operation == 'gpu-health' and os.environ.get('FAIL_FIRST_HEALTH') == '1' and job == '321':
            print('[collectivex-private] gpu-health-fault gpu 0: throttled')
            sys.exit(1)
        elif operation == 'network-profile':
            for _ in range(int(option('nodes'))):
                print('[collectivex-private] socket-interface-selected=eth0')
                print('[collectivex-private] rdma-link-layer=roce')
        elif operation == 'import-image':
            options = json.loads(command[4])
            image = pathlib.Path(options['path'])
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_text('image')
        sys.exit(0)
    if command[2] == 'prepare':
        print('prepared')
        sys.exit(42 if os.environ.get('FAIL_PREPARE') == '1' else 0)
    source = pathlib.Path(option('container-mounts').split(':/ix')[0])
    benchmark = command[4:]
    case = benchmark[benchmark.index('--case-id') + 1]
    if case == os.environ.get('SLEEP_CASE'):
        (state / ('worker-' + job)).write_text(str(os.getpid()))
        (state / 'running').touch()
        time.sleep(60)
    failed = case == os.environ.get('FAIL_CASE')
    output = source / 'experimental/CollectiveX' / benchmark[benchmark.index('--out') + 1]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'record_type': 'case-attempt', 'case': case,
                                 'outcome': {'status': 'invalid' if failed else 'success'}}))
    sys.exit(3 if failed else 0)
elif tool == 'getent':
    print(args[1] + ':x:44:runner')
elif tool == 'docker':
    if args[0] != 'run': sys.exit(0)
    mount = args[args.index('-v') + 1]
    source, target = mount.rsplit(':', 1)
    feature = pathlib.Path(source) if target == '/cx' else pathlib.Path(source) / 'experimental/CollectiveX'
    output = feature / args[args.index('--output' if '--output' in args else '--out') + 1]
    if '--case-id' in args:
        case = args[args.index('--case-id') + 1]
        retried = state / 'docker-retried'
        if case == 'case-0' and os.environ.get('DOCKER_FAIL_FIRST') == '1' and not retried.exists():
            retried.touch()
            sys.exit(1)
        record = {'case': case, 'outcome': {'status': 'success'}}
    else:
        record = {'layout': args[args.index('--layout') + 1]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record))
"""


class PythonExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.job = Path(tempfile.mkdtemp(prefix="inferencex-collectivex-123-1-test-", dir="/tmp"))
        self.addCleanup(lambda: shutil.rmtree(self.job, ignore_errors=True))
        self.repo = self.base / "repo"
        self.feature = self.repo / "experimental/CollectiveX"
        self.feature.mkdir(parents=True)
        shutil.copytree(
            ROOT / "runtime", self.feature / "runtime", ignore=shutil.ignore_patterns("__pycache__")
        )
        for name in ("ci.py", "summarize.py", "bandwidth.py"):
            shutil.copyfile(ROOT / name, self.feature / name)
        (self.feature / "configs").mkdir()
        operator = {
            "partition": "fixture",
            "squash_dir": str(self.base / "images"),
            "stage_dir": str(self.base / "stage"),
        }
        (self.base / "stage").mkdir()
        (self.feature / "configs/platform_config.json").write_text(
            json.dumps(
                {
                    "platforms": {
                        "h200-dgxc": {
                            "arch": "sm90",
                            "image": "some/image:tag",
                            "image_platform": "linux/amd64",
                            "operator": operator,
                        },
                    }
                }
            )
        )
        cases = [
            {
                "case_id": f"case-{index}",
                "backend": "nccl-ep",
                "mode": "normal",
                "precision": "bf16",
                "phase": "decode",
                "routing": "uniform",
                "ep": 8,
                "nodes": 1,
                "gpus_per_node": 8,
                "scale_up_domain": 8,
                "scope": "scale-up",
                "scale_up_transport": "nvlink",
                "scale_out_transport": None,
                "transport": "nvlink",
                "ladder": "1 2",
                "hidden": 64,
                "topk": 2,
                "experts": 32,
                "seed": 67,
                "suite": "ep-core",
                "workload": "fixture",
                "topology_class": "fixture-nvlink",
                "timing": "1:1:1",
            }
            for index in range(3)
        ]
        self.control = self.base / "shard.json"
        self.control.write_text(json.dumps({"version": 1, "cases": cases}))
        self.binary = self.base / "bin"
        self.binary.mkdir()
        for name in (
            "srun",
            "salloc",
            "squeue",
            "scontrol",
            "scancel",
            "timeout",
            "docker",
            "getent",
        ):
            path = self.binary / name
            path.write_text(f"#!{sys.executable}\n" + SCHEDULER)
            path.chmod(0o755)
        self.state = self.base / "scheduler"
        self.state.mkdir()
        self.env = {
            **os.environ,
            "PATH": f"{self.binary}:{os.environ['PATH']}",
            "SCHEDULER_STATE": str(self.state),
            "COLLX_JOB_ROOT": str(self.job),
            "COLLX_SHARD_SKU": "h200-dgxc",
            "COLLX_BENCH": "nccl-ep",
            "COLLX_SHARD_FILE": str(self.control),
            "COLLX_NODES": "1",
            "COLLX_GPUS_PER_NODE": "8",
            "COLLX_SCALE_UP_DOMAIN": "8",
            "COLLX_IMAGE_DIGEST": "sha256:" + "a" * 64,
            "COLLECTIVEX_EXECUTION_ID": "fixture",
            "COLLECTIVEX_CANONICAL_GHA": "0",
            "COLLECTIVEX_OPERATOR_CONFIG": str(self.base / "no-operator-config"),
        }

    def cli(self, **updates):
        return subprocess.run(
            [sys.executable, str(self.feature / "ci.py"), "execute"],
            env={**self.env, **updates},
            text=True,
            capture_output=True,
            timeout=30,
        )

    def calls(self):
        return [json.loads(line) for line in (self.state / "calls").read_text().splitlines()]

    def results(self):
        return [
            json.loads(path.read_text())
            for path in sorted((self.feature / "results").glob("*.json"))
        ]

    def docker_profile(self):
        path = self.feature / "configs/platform_config.json"
        document = json.loads(path.read_text())
        document["platforms"]["mi325x-tw"] = {
            "arch": "gfx942",
            "image": "some/image:tag",
            "image_platform": "linux/amd64",
        }
        path.write_text(json.dumps(document))
        document = json.loads(self.control.read_text())
        for case in document["cases"]:
            case["backend"], case["transport"], case["scale_up_transport"] = "mori", "xgmi", "xgmi"
        self.control.write_text(json.dumps(document))

    def test_docker_runs_direct_torchrun_and_retains_the_cold_start_retry(self):
        self.docker_profile()
        result = self.cli(COLLX_SHARD_SKU="mi325x-tw", COLLX_BENCH="mori", DOCKER_FAIL_FIRST="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        launches = [
            call["args"]
            for call in self.calls()
            if call["tool"] == "docker" and call["args"][0] == "run"
        ]
        self.assertEqual(len(launches), 4)
        self.assertTrue(
            all("torchrun" in args and "--nproc-per-node=8" in args for args in launches)
        )
        self.assertTrue(all("MORI_ENABLE_SDMA=1" in args for args in launches))
        self.assertTrue(all("bash" not in args for args in launches))
        self.assertFalse(any(call["tool"] == "salloc" for call in self.calls()))
        self.assertEqual([row["case"] for row in self.results()], ["case-0", "case-1", "case-2"])
        removed = [
            call["args"]
            for call in self.calls()
            if call["tool"] == "docker" and call["args"][0] == "rm"
        ]
        self.assertEqual(len(removed), 4)

    def test_docker_swap_retains_runner_identity_and_both_layouts(self):
        self.docker_profile()
        result = self.cli(
            COLLX_SHARD_SKU="mi325x-tw",
            COLLX_BENCH="swap-blocks",
            COLLX_GPUS_PER_NODE="1",
            COLLX_VENDOR="amd",
            COLLECTIVEX_SOURCE_SHA="fixture",
            COLLX_IMAGE_REFRESH="0",
            COLLX_SWAP_IMAGE="vllm/vllm-openai-rocm:fixture",
            COLLX_SWAP_BLOCK_BYTES="8",
            COLLX_SWAP_NUM_BLOCKS="1 2",
            COLLX_SWAP_MAX_PAYLOAD_BYTES="16",
            COLLX_SWAP_WARMUP="0",
            COLLX_SWAP_ITERATIONS="1",
            COLLX_SWAP_SEED="0",
            COLLX_SWAP_DEVICE="0",
            COLLX_SWAP_TIME="45",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        launches = [
            call["args"]
            for call in self.calls()
            if call["tool"] == "docker" and call["args"][0] == "run"
        ]
        self.assertEqual(len(launches), 2)
        self.assertTrue(
            all(
                args[args.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
                for args in launches
            )
        )
        self.assertEqual([row["layout"] for row in self.results()], ["contiguous", "random"])

    def test_failed_case_does_not_suppress_later_cases_or_partial_artifacts(self):
        result = self.cli(FAIL_CASE="case-1")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(
            [(row["case"], row["outcome"]["status"]) for row in self.results()],
            [("case-0", "success"), ("case-1", "invalid"), ("case-2", "success")],
        )
        self.assertFalse((self.job / "jobid").exists())
        self.assertEqual(list((self.base / "stage").iterdir()), [])
        self.assertEqual(
            [call["args"] for call in self.calls() if call["tool"] == "scancel"], [["321"]]
        )

    def test_throttled_node_is_released_and_excluded_before_retry(self):
        result = self.cli(FAIL_FIRST_HEALTH="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        requests = [call["args"] for call in self.calls() if call["tool"] == "salloc"]
        self.assertEqual(len(requests), 2)
        self.assertIn("--exclude=node321", requests[1])
        self.assertEqual(
            [call["args"] for call in self.calls() if call["tool"] == "scancel"], [["321"], ["322"]]
        )
        self.assertEqual(len(self.results()), 3)

    def test_preparation_failure_releases_the_allocation_without_running_cases(self):
        result = self.cli(FAIL_PREPARE="1")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(self.results(), [])
        self.assertFalse((self.job / "jobid").exists())
        self.assertEqual(
            [call["args"] for call in self.calls() if call["tool"] == "scancel"], [["321"]]
        )

    def test_sigterm_preserves_143_and_collects_completed_results_before_cleanup(self):
        with (self.base / "process.log").open("w") as output:
            process = subprocess.Popen(
                [sys.executable, str(self.feature / "ci.py"), "execute"],
                env={**self.env, "SLEEP_CASE": "case-1"},
                stdout=output,
                stderr=output,
            )
            try:
                deadline = time.monotonic() + 15
                while not (self.state / "running").exists() and process.poll() is None:
                    if time.monotonic() > deadline:
                        self.fail("benchmark step never started")
                    time.sleep(0.05)
                process.send_signal(signal.SIGTERM)
                self.assertEqual(process.wait(timeout=15), 143)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        self.assertEqual([row["case"] for row in self.results()], ["case-0"])
        self.assertFalse((self.job / "jobid").exists())
        self.assertEqual(list((self.base / "stage").iterdir()), [])


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.env = {
            **os.environ,
            "PATH": f"{self.bin}:{os.environ['PATH']}",
            "RECORD": str(self.root / "record.json"),
            "PAYLOAD": "literal $value",
        }
        self.allocation = SlurmAllocation(self.root, self.env)

    def executable(self, name, body):
        path = self.bin / name
        path.write_text(f"#!{sys.executable}\nimport json, os, pathlib, sys\n" + body)
        path.chmod(0o755)

    def test_library_step_preserves_bare_flags_and_literal_arguments(self):
        self.executable(
            "srun",
            """
pathlib.Path(os.environ['RECORD']).write_text(json.dumps({
    'args': sys.argv[1:], 'payload': os.environ['PAYLOAD'], 'stdin': sys.stdin.read(),
}))
print('step output')
""",
        )
        self.allocation.job_id = "321"
        path = self.root / "step.log"
        injected = self.root / "injected"
        payload = f"spaces; $(touch {injected}) 'quoted'"
        self.allocation.step(
            {
                "nodes": 2,
                "ntasks_per_node": 4,
                "container_writable": True,
                "container_mounts": "/some path:/ix",
                "no_container_entrypoint": True,
            },
            ["python3", "bench.py", payload],
            path,
            stdin=b"node input",
        )
        record = json.loads((self.root / "record.json").read_text())
        self.assertEqual(record["args"][-3:], ["python3", "bench.py", payload])
        self.assertEqual(
            set(record["args"][:-3]),
            {
                "--jobid=321",
                "--nodes=2",
                "--ntasks-per-node=4",
                "--container-writable",
                "--container-mounts=/some path:/ix",
                "--no-container-entrypoint",
            },
        )
        self.assertEqual((record["payload"], record["stdin"]), ("literal $value", "node input"))
        self.assertEqual(path.read_text(), "step output\n")
        self.assertFalse(injected.exists())

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
        self.executable(
            "scancel", "pathlib.Path(os.environ['RECORD']).write_text(json.dumps(sys.argv[1:]))\n"
        )
        self.executable(
            "squeue",
            """
counter = pathlib.Path(os.environ['RECORD'] + '.queries')
if not counter.exists():
    counter.write_text('queried')
    print('321')
""",
        )
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
        env = network_environment(
            {
                "COLLX_RDMA_DEVICES": "mlx5_1:2,mlx5_10",
                "COLLX_SOCKET_IFNAME": "eth0",
                "COLLX_IB_GID_INDEX": "3",
                "COLLX_RDMA_LINK_LAYER": "roce",
                "COLLX_RAIL_ISOLATED": "1",
                "NCCL_NET_PLUGIN": "stale",
            },
            2,
            "nvlink-rdma",
        )
        self.assertEqual(env["NCCL_IB_HCA"], "=mlx5_1:2,mlx5_10")
        self.assertEqual(env["UCCL_IB_HCA"], "=mlx5_1:2,mlx5_10")
        self.assertEqual(env["MORI_RDMA_DEVICES"], "mlx5_1,mlx5_10")
        self.assertEqual(env["UCCL_IB_GID_INDEX"], "3")
        self.assertEqual(env["NCCL_CROSS_NIC"], "0")
        self.assertNotIn("NCCL_NET_PLUGIN", env)

    def test_efa_does_not_retain_verbs_selectors(self):
        env = network_environment(
            {
                "COLLX_RDMA_DEVICES": "rdmap1s0,rdmap2s0",
                "COLLX_RDMA_FABRIC": "efa",
                "NCCL_IB_HCA": "wrong",
                "NVSHMEM_IB_ENABLE_IBGDA": "1",
            },
            2,
            "nvlink-rdma",
        )
        self.assertEqual(env["NCCL_NET_PLUGIN"], "ofi")
        self.assertEqual(env["NVSHMEM_LIBFABRIC_PROVIDER"], "efa")
        self.assertNotIn("NCCL_IB_HCA", env)
        self.assertNotIn("NVSHMEM_IB_ENABLE_IBGDA", env)

    def test_different_node_interfaces_are_resolved_on_each_rank(self):
        output = "\n".join(
            [
                "[collectivex-private] socket-interface-selected=eth0",
                "[collectivex-private] socket-interface-selected=eth1",
                "[collectivex-private] rdma-link-layer=infiniband",
                "[collectivex-private] rdma-link-layer=infiniband",
            ]
        )
        env = validated_selectors(output, 2, {"COLLX_SOCKET_IFNAME": "stale"})
        self.assertNotIn("COLLX_SOCKET_IFNAME", env)
        self.assertEqual(env["COLLX_RDMA_LINK_LAYER"], "infiniband")
        with self.assertRaisesRegex(RuntimeError, "disagree"):
            validated_selectors(output.replace("infiniband", "roce", 1), 2, {})
