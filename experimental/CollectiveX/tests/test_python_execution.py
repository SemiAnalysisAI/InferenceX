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


ROOT = Path(__file__).resolve().parents[1]

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
        if operation == 'address': print('compute-zero')
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
