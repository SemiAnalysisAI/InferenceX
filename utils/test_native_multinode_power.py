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
    from infx.results.power.audit import audit_summary
    summary = audit_summary(audit, "power_validation_result.json")["power_audit"]
    assert summary["sample_count"] == 10  # Five samples on each of two participating GPUs.
    assert summary["producer_sha"] == "revision-abc"


def test_native_parse_failure_has_a_public_reason_and_retained_detail(tmp_path):
    from infx.results.power.audit import audit_summary
    root, bench, agg = _package(tmp_path)
    (root / "node-1/manifest.json").write_text("not JSON")
    assert run(root, bench, agg, expected_prefill_gpus=1, expected_decode_gpus=1, require_power=True) == 1
    audit = json.loads((tmp_path / "power_validation_result.json").read_text())
    summary = audit_summary(audit, "power_validation_result.json")
    assert "native_node_invalid" in summary["power_invalid_reasons"]
    assert audit["node_errors"][0]["node"] == "node-1"
    assert audit["node_errors"][0]["detail"]


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


@pytest.mark.parametrize('role', ['prefill', 'decode'])
def test_native_single_role_preserves_whole_fleet_and_role_metrics(tmp_path, role):
    root, bench, agg = _package(tmp_path)
    for path in root.glob('*/manifest.json'):
        manifest = json.loads(path.read_text())
        manifest['role'] = role
        path.write_text(json.dumps(manifest))
    assert run(root, bench, agg, expected_prefill_gpus=2 if role == 'prefill' else 0,
               expected_decode_gpus=2 if role == 'decode' else 0, require_power=True) == 0
    actual = json.loads(agg.read_text())
    assert actual['total_gpu_energy_j'] == 800
    assert actual[f'{role}_gpu_energy_j'] == 800
    assert actual[f'{role}_avg_power_w'] == 200
    opposite = 'decode' if role == 'prefill' else 'prefill'
    assert f'{opposite}_gpu_energy_j' not in actual
    assert json.loads((tmp_path / 'power_validation_result.json').read_text())['power_valid']


def test_native_amd_abort_skips_legacy_tail_wait(tmp_path):
    result = subprocess.run(['bash', '-c', '''source "$1"
kill() { return 0; }; wait() { return 0; }
sleep() { echo unexpected-tail-wait >&2; }
_write_amd_smi_sidecar() { return 0; }
GPU_MONITOR_PID=999 GPU_MONITOR_VENDOR=amd AMD_MONITOR_STOP_TIMEOUT_S=0
GPU_METRICS_CSV="$2/missing.csv"
stop_gpu_monitor
''', 'bash', str(REPO / 'benchmarks/benchmark_lib.sh'), str(tmp_path)],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert 'unexpected-tail-wait' not in result.stderr


def test_native_amd_abort_publishes_receipt_before_reaper_deadline(tmp_path):
    binary = tmp_path / 'bin'
    binary.mkdir()
    (binary / 'python3').symlink_to(sys.executable)
    fake = binary / 'amd-smi'
    fake.write_text(f'''#!{sys.executable}
import sys, time
if "-w" in sys.argv:
    print("timestamp,gpu,socket_power", flush=True)
    while True:
        print(str(int(time.time())) + ",0,100", flush=True)
        time.sleep(0.1)
else:
    print("[]")
''')
    fake.chmod(0o755)
    control = tmp_path / 'control'
    control.mkdir()
    node = tmp_path / 'node-0'
    process = subprocess.Popen(['bash', '-c', '\n'.join([
        'source "$1"',
        'bash "$4" "$2" "$3" amd 0 aggregate 0 1 &',
        'POWERX_COLLECTOR_PID=$! POWERX_CONTROL_DIR=$3 POWERX_NUM_NODES=1',
        'POWERX_BARRIER_TIMEOUT_S=5',
        'powerx_wait_collectors ready || exit 1',
        'POWERX_BARRIER_TIMEOUT_S=0',
        'powerx_reap_collector',
    ]), 'bash', str(REPO / 'benchmarks/native_power_lifecycle.sh'), str(node), str(control), str(REPO / 'benchmarks/native_power_collect.sh')],
        env={**os.environ, 'PATH': f'{binary}:/usr/bin:/bin', 'SLURM_JOB_ID': 'test-job'},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=8)
        assert process.returncode == 143, (stdout, stderr)
        assert (control / 'done-0').read_text().strip() == '143'
        assert json.loads((node / 'manifest.json').read_text())['lifecycle'] == 'failed'
    finally:
        import signal
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()


def test_native_prefill_only_rejects_aggregate_role_devices(tmp_path):
    root, bench, agg = _package(tmp_path)
    manifest_path = root / 'node-1/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['role'] = 'aggregate'
    manifest_path.write_text(json.dumps(manifest))
    assert run(root, bench, agg, expected_prefill_gpus=2, expected_decode_gpus=0,
               require_power=True) == 1
    audit = json.loads((tmp_path / 'power_validation_result.json').read_text())
    assert 'native_role_gpu_count_mismatch' in audit['reasons']
    assert json.loads(agg.read_text())['power_valid'] == 0
    assert 'prefill_avg_power_w' not in json.loads(agg.read_text())


@pytest.mark.parametrize('tick', [2, 4])
def test_native_audit_retains_boundary_noise_without_relaxing_in_window_errors(tmp_path, tick):
    root, bench, agg = _package(tmp_path)
    for rank in [0, 1]:
        with (root / f'node-{rank}/gpu_metrics.csv').open('a') as stream:
            stream.write(f'{tick},0,N/A\n')
            if rank == 0:
                stream.write(f'{tick},0,0\n')
    assert run(root, bench, agg, expected_prefill_gpus=1, expected_decode_gpus=1,
               require_power=True) == int(tick == 2)
    audit = json.loads((tmp_path / 'power_validation_result.json').read_text())
    assert audit['boundary_degenerate_rows'] == ({'uuid-0': 2, 'uuid-1': 1} if tick == 4 else {})
    if tick == 4:
        assert json.loads(agg.read_text())['total_gpu_energy_j'] == 800
