"""Native collector acceptance uses independent, hand-computed two-node traces."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from infx.results.power.native_multinode import record_begin, record_end, run
from infx.results.power.single_node import integrate_power

REPO = Path(__file__).resolve().parents[1]


def _package(tmp_path, vendor="amd"):
    root = tmp_path / "native_power"
    for rank, role, watts in ((0, "prefill", 100), (1, "decode", 300)):
        node = root / f"node-{rank}"
        record_begin(node, vendor=vendor, node=f"host-{rank}", rank=rank, role=role,
                     gpu_indices=[0], num_nodes=2, job_id="job-123", revision="revision-abc",
                     clock_synchronized=True)
        record_end(node, collector_exit_code=0)
        manifest = json.loads((node / "manifest.json").read_text())
        manifest.update(collection_start_unix=0, collection_end_unix=5)
        (node / "manifest.json").write_text(json.dumps(manifest))
        for ending in ("", "_end"):
            if vendor == "amd":
                (node / f"gpu_metrics_devices{ending}.json").write_text(json.dumps([
                    {"gpu": 0, "uuid": f"uuid-{rank}"}, {"gpu": 1, "uuid": f"unused-{rank}"}]))
            else:
                (node / f"gpu_metrics_identity{ending}.csv").write_text(
                    f"index, uuid, pci.bus_id\n0, uuid-{rank}, 0000:01:00.0\n1, unused-{rank}, 0000:02:00.0\n")
        # The spare physical GPU is visible but does not belong to the server.
        (node / "gpu_metrics.csv").write_text("timestamp,gpu,power\n" + "".join(
            f"{tick},0,{watts}\n{tick},1,900\n" for tick in range(5)))
    bench = tmp_path / "result.json"
    bench.write_text(json.dumps({"benchmark_start_time_unix": 1, "benchmark_end_time_unix": 3,
                                "duration": 2, "completed": 2, "total_input_tokens": 20,
                                "total_output_tokens": 10}))
    agg = tmp_path / "agg.json"
    agg.write_text(json.dumps({"avg_power_w": 999, "prefill_gpu_energy_j": 999}))
    return root, bench, agg


@pytest.mark.parametrize("vendor", ["amd", "nvidia"])
def test_native_whole_fleet_and_role_energy_use_only_serving_devices(tmp_path, vendor):
    root, bench, agg = _package(tmp_path, vendor)
    assert run(root, bench, agg, expected_prefill_gpus=1, expected_decode_gpus=1, require_power=True) == 0
    actual = json.loads(agg.read_text())
    assert actual["power_valid"] == 1
    assert actual["total_gpu_energy_j"] == 800
    assert actual["avg_power_w"] == 200
    assert actual["p90_power_w"] == 200
    assert actual["p75_power_w"] == 200
    assert actual["joules_per_successful_query"] == 400
    assert actual["prefill_gpu_energy_j"] == 200
    assert actual["decode_gpu_energy_j"] == 600
    assert actual["prefill_joules_per_input_token"] == 10
    assert actual["decode_joules_per_output_token"] == 60
    audit = json.loads((tmp_path / "power_validation_result.json").read_text())
    assert audit["observed_gpu_count"] == 2
    assert audit["per_gpu_role"] == {"uuid-0": "prefill", "uuid-1": "decode"}
    assert len(audit["nodes"][0]["telemetry_sha256"]) == 64


@pytest.mark.parametrize(("field", "value", "reason"), [
    ("clock_synchronized", False, "native_clock_not_synchronized"),
    ("lifecycle", "collecting", "native_collector_incomplete"),
    ("collection_end_unix", 2, "native_collection_window_mismatch"),
    ("job_id", "other-job", "native_run_identity_mismatch"),
    ("role", "prefill", "native_role_gpu_count_mismatch"),
    ("expected_num_nodes", 3, "native_node_topology_mismatch"),
])
def test_native_invalid_evidence_clears_stale_metrics_and_writes_audit(tmp_path, field, value, reason):
    root, bench, agg = _package(tmp_path)
    path = root / "node-1/manifest.json"
    manifest = json.loads(path.read_text()); manifest[field] = value
    path.write_text(json.dumps(manifest))
    assert run(root, bench, agg, expected_prefill_gpus=1, expected_decode_gpus=1, require_power=True) == 1
    actual = json.loads(agg.read_text())
    assert actual["power_valid"] == 0
    assert "avg_power_w" not in actual
    assert "prefill_gpu_energy_j" not in actual
    audit = json.loads((tmp_path / "power_validation_result.json").read_text())
    assert reason in audit["reasons"]


def test_native_device_replacement_cannot_preserve_validity(tmp_path):
    root, bench, agg = _package(tmp_path)
    (root / "node-1/gpu_metrics_devices_end.json").write_text('[{"gpu":0,"uuid":"replacement"}]')
    assert run(root, bench, agg, expected_prefill_gpus=1, expected_decode_gpus=1, require_power=True) == 1
    assert "native_device_identity_changed" in json.loads((tmp_path / "power_validation_result.json").read_text())["reasons"]


def test_native_aggregate_nodes_do_not_invent_prefill_decode_metrics(tmp_path):
    root, bench, agg = _package(tmp_path)
    for path in root.glob("*/manifest.json"):
        manifest = json.loads(path.read_text()); manifest["role"] = "aggregate"
        path.write_text(json.dumps(manifest))
    assert run(root, bench, agg, expected_prefill_gpus=0, expected_decode_gpus=0,
               expected_aggregate_gpus=2, require_power=True) == 0
    actual = json.loads(agg.read_text())
    assert actual["avg_power_w"] == 200
    assert "prefill_gpu_energy_j" not in actual


def test_utc_context_replays_in_a_different_timezone(tmp_path):
    csv = tmp_path / "gpu_metrics.csv"
    csv.write_text("timestamp,index,power.draw [W]\n2026/01/01 00:00:00,0,100\n2026/01/01 00:00:02,0,100\n")
    (tmp_path / "gpu_metrics_context.json").write_text('{"timestamp_timezone":"UTC"}')
    script = "from pathlib import Path; from infx.results.power.single_node import integrate_power; " + \
        f"r=integrate_power(Path({str(csv)!r}),start_unix=1767225600,end_unix=1767225602,expected_num_gpus=1); assert r.power_valid; assert r.total_gpu_energy_j == 200"
    subprocess.run([sys.executable, "-c", script], cwd=REPO,
                   env={**os.environ, "TZ": "America/Los_Angeles"}, check=True, timeout=10)


@pytest.mark.parametrize("clock_value, synchronized", [
    ("yes", True), ("true", True), ("no", False), ("false", False), ("", False),
])
def test_native_supervisor_reaps_monitor_and_writes_completion(tmp_path, clock_value, synchronized):
    binary = tmp_path / "bin"; binary.mkdir()
    (binary / "python3").symlink_to(sys.executable)
    fake = binary / "nvidia-smi"
    fake.write_text(f'''#!{sys.executable}
import datetime, os, sys, time
if any("index,uuid" in arg for arg in sys.argv):
    print("index, uuid, pci.bus_id"); print("0, gpu-0, 0000:01:00.0")
else:
    with open({str(tmp_path / 'monitor.pid')!r}, "w") as stream: stream.write(str(os.getpid()))
    if "-l" in sys.argv: print("timestamp,index,power.draw [W]", flush=True)
    while True:
        print(datetime.datetime.now(datetime.timezone.utc).strftime("%Y/%m/%d %H:%M:%S.%f") + ",0,100", flush=True)
        if "-l" not in sys.argv: break
        time.sleep(0.1)
''')
    fake.chmod(0o755)
    control = tmp_path / "control"; control.mkdir()
    node = tmp_path / "node-0"
    process = subprocess.Popen(["bash", str(REPO / "benchmarks/native_power_collect.sh"),
                                str(node), str(control), "nvidia", "0", "aggregate", "0", "1"],
                               env={**os.environ, "PATH": f"{binary}:{os.environ['PATH']}",
                                    "SLURM_JOB_ID": "test-job", "POWERX_COLLECTOR_REVISION": "revision",
                                    "POWERX_CLOCK_SYNCHRONIZED": clock_value}, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 10
        while not (control / "ready-0").exists():
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.02)
        (control / "stop").write_text("stop")
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, (stdout, stderr)
        assert (control / "done-0").read_text().strip() == "0"
        manifest = json.loads((node / "manifest.json").read_text())
        assert manifest["lifecycle"] == "complete"
        assert manifest["clock_synchronized"] is synchronized
        assert (node / "gpu_metrics_identity_end.csv").exists()
    finally:
        if process.poll() is None:
            process.terminate(); process.communicate(timeout=10)


def test_amd_stop_coverage_uses_slowest_gpu_and_normalizes_milliseconds(tmp_path):
    csv = tmp_path / "gpu_metrics.csv"
    csv.write_text('timestamp,gpu,vcn_activity,socket_power\n'
                   '1700000002000,0,"[0, 0]",250\n'
                   '1700000001000,1,"[0, 0]",250\n'
                   '1700000009000,1,"[0, 0]",N/A\n')
    result = subprocess.run(["bash", "-c", 'source "$1"; GPU_METRICS_CSV="$2"; _amd_monitor_min_covered_tick',
                             "test", str(REPO / "benchmarks/benchmark_lib.sh"), str(csv)],
                            capture_output=True, text=True, check=True, timeout=10)
    assert result.stdout.strip() == "1700000001"


def test_amd_shutdown_kills_both_pipeline_processes_when_stream_dies(tmp_path):
    binary = tmp_path / "bin"; binary.mkdir()
    fake = binary / "amd-smi"
    fake.write_text(f'''#!{sys.executable}
import json, sys, time
if "-w" in sys.argv:
    print("timestamp,gpu,socket_power", flush=True)
    while True:
        print(str(int(time.time())) + ",0,250", flush=True); time.sleep(.1)
else:
    print("[]")
''')
    fake.chmod(0o755)
    csv = tmp_path / "gpu_metrics.csv"
    script = '''source "$1"
start_gpu_monitor --output "$2"
source_pid=$GPU_MONITOR_SOURCE_PID
sink_pid=$GPU_MONITOR_PID
kill "$sink_pid"
wait "$sink_pid" 2>/dev/null || true
AMD_MONITOR_STOP_TIMEOUT_S=0
stop_gpu_monitor
if kill -0 "$source_pid" 2>/dev/null; then echo 'source leaked'; exit 1; fi
if kill -0 "$sink_pid" 2>/dev/null; then echo 'sink leaked'; exit 1; fi
[[ ! -p "$2.pipe.$$" ]]
'''
    subprocess.run(["bash", "-c", script, "test", str(REPO / "benchmarks/benchmark_lib.sh"), str(csv)],
                   env={**os.environ, "PATH": f"{binary}:{os.environ['PATH']}"},
                   capture_output=True, text=True, check=True, timeout=10)


def test_amd_multinode_selects_equal_local_tensor_ranks(tmp_path):
    arguments = tmp_path / "collector.args"
    # Mock only the downstream collector command; exercise actual topology routing.
    script = f'''source {str(REPO / 'benchmarks/multi_node/amd_utils/power.sh')!r}
bash() {{ printf '%s\\0' "$@" > {str(arguments)!r}; }}
start_amd_multinode_power
wait "$POWERX_COLLECTOR_PID"
'''
    subprocess.run(["bash", "-c", script], env={**os.environ, "BENCH_INPUT_LEN": "8192",
                   "BENCH_OUTPUT_LEN": "1024", "PREFILL_TP_SIZE": "12", "DECODE_TP_SIZE": "12",
                   "GPUS_PER_NODE": "8", "xP": "1", "yD": "1", "NNODES": "4", "NODE_RANK": "1",
                   "WS_PATH": str(tmp_path), "BENCHMARK_LOGS_DIR": str(tmp_path), "SLURM_JOB_ID": "123",
                   "IS_AGENTIC": "0", "EVAL_ONLY": "false", "DRY_RUN": "0"},
                   capture_output=True, text=True, timeout=10, check=True)
    args = arguments.read_bytes().decode().split("\0")
    assert args[-6:-1] == ["amd", "1", "prefill", "0,1,2,3,4,5", "4"]


def test_result_processor_discovers_staged_native_package(tmp_path, monkeypatch):
    from infx.results.fixed_sequence import aggregate_power_result

    root, bench, agg = _package(tmp_path)
    logs = tmp_path / "LOGS"
    logs.mkdir()
    root.rename(logs / "native_power")
    monkeypatch.chdir(tmp_path)
    env = {"IS_MULTINODE": "true", "PREFILL_GPUS": "1", "DECODE_GPUS": "1",
           "RESULT_FILENAME": "result", "REQUIRE_POWER": "1"}
    assert aggregate_power_result(env, bench, agg) == 0
    assert json.loads(agg.read_text())["total_gpu_energy_j"] == 800
    assert (tmp_path / "power_validation_result.json").is_file()

    # Two competing formats must not silently select one package.
    (logs / "power").mkdir()
    assert aggregate_power_result(env, bench, agg) == 1
    invalid = json.loads(agg.read_text())
    assert invalid["power_valid"] == 0
    assert "total_gpu_energy_j" not in invalid
