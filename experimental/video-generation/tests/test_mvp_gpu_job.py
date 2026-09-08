"""CPU-only supervisor tests. Fakes here are NEVER H3 benchmark evidence."""

import copy
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluator import mvp_gpu_job as gpu
from evaluator.mvp_runner import _slots


GPU = "GPU-12345678-abcd-abcd-abcd-123456789abc"
REV = "a" * 40
SHA = "b" * 64


@pytest.fixture
def spec(tmp_path):
    plan = json.loads((Path(__file__).parents[1] / "mvp/h3-smoke.plan.json").read_text())
    model = tmp_path / "model"
    model.mkdir()
    weight = b"CPU TEST PLACEHOLDER, NOT MODEL WEIGHTS"
    (model / "weights.safetensors").write_bytes(weight)
    return {
        "schema_version": "0.1.0", "job_id": "cpu-supervisor-test",
        "authorization": {"compute_approved": True, "model_license_reviewed": True, "approval_reference": "CPU unit-test fakes only; no GPU authorized"},
        "allocation": {"mode": "cooperative_shared", "label": "CPU unit-test fake"},
        "gpu_uuids": [GPU], "port": 30280, "lock_directory": str(tmp_path / "locks"),
        "baseline": {"python": sys.executable, "source": str(tmp_path / "source-a"), "revision": REV, "source_sha256": SHA},
        "candidate": {"python": sys.executable, "source": str(tmp_path / "source-b"), "revision": REV, "source_sha256": SHA},
        "model": {"path": str(model), "revision": plan["model_revision"],
                  "files": [{"path": "weights.safetensors", "size_bytes": len(weight), "sha256": hashlib.sha256(weight).hexdigest()}]},
        "server": {"ulysses_degree": 1, "dit_cpu_offload": False, "tp_size": 1, "encoder_parallel": "auto", "performance_mode": "speed"},
        "plan": plan,
        "policy": {"policy_id": "uncalibrated-cpu-test", "calibration_status": "uncalibrated", "max_latency_increase_fraction": 0.1,
                   "min_video_psnr_db": 30, "min_audio_spectral_cosine": 0.95, "max_audio_rms_ratio_error": 0.1,
                   "max_memory_increase_fraction": 0.1},
        "limits": {"job_seconds": 100, "startup_seconds": 20, "request_seconds": 5, "cleanup_seconds": 0.2,
                   "telemetry_interval_seconds": 0.1, "command_seconds": 0.1, "max_idle_memory_mib": 1000},
    }


def snapshot(*, used=50, pid=None):
    return {"at": "2026-09-01T00:00:00Z", "monotonic_seconds": time.monotonic(),
            "gpus": [{"uuid": GPU, "index": 0, "name": "CPU TEST FAKE", "memory_total_mib": 100000,
                      "memory_used_mib": used, "utilization_percent": 50 if pid else 0, "driver_version": "fixture", "mig_mode": "Disabled"}],
            "compute_apps": [{"gpu_uuid": GPU, "pid": pid, "memory_used_mib": used}] if pid else []}


def controlled_receipt(spec):
    roles = {}
    for label in ("baseline", "candidate"):
        roles[label] = {
            "status": "complete", "cleanup": {"status": "clean", "idle_after": True},
            "source_identity": {**spec[label], "python_sha256": "c" * 64, "python_version": "3.12.1", "packages": {"torch": "2.13.0", "sglang": "test"}},
            "process_identity": {"pid": 500, "pgid": 500, "session_id": 500, "start_ticks": 100, "launch_nonce": label},
            "telemetry_summary": {"qualified": True, "measurement_sample_count": 2, "observed_owned_compute_by_gpu": {GPU: 2},
                                  "gpu_identity": [{"uuid": GPU, "name": "CPU TEST FAKE"}], "observed_memory_peak_mib_by_gpu": {GPU: 100}},
        }
    return {"schema_version": "0.1.0", "bundle_type": "controlled_gpu_job", "job_id": spec["job_id"],
            "status": "complete", "measurement_status": "complete", "evidence_kind": "controlled_h3_gpu",
            "spec_sha256": gpu._digest(spec), "plan_sha256": gpu._digest(spec["plan"]), "roles": roles,
            "model_identity": {"manifest_sha256": gpu._digest(spec["model"]["files"])}}


def comparison(spec, status="pass"):
    return {"bundle_type": "mvp_comparison", "evidence_kind": "operator_endpoint", "plan_sha256": gpu._digest(spec["plan"]),
            "overall_status": status, "measurement": {"performance_mode": "same_configuration_class_regression"}}


def test_preview_never_probes_or_starts_compute(spec, monkeypatch):
    monkeypatch.setattr(gpu, "_command", lambda *a, **k: pytest.fail("preview executed a process"))
    monkeypatch.setattr(gpu.GpuProbe, "snapshot", lambda *a: pytest.fail("preview accessed a GPU"))
    spec["authorization"]["compute_approved"] = False
    result = gpu.preview_gpu_job(spec)
    assert result["evidence_kind"] == "gpu_job_preview_no_execution"
    assert result["spec_sha256"] == gpu._digest(gpu.validate_gpu_job(spec))
    args = result["commands"]["baseline"]
    assert args[:3] == [sys.executable, "-c", gpu._LAUNCH]
    assert args[args.index("--model-path") + 1] == spec["model"]["path"]
    assert args[args.index("--model-id") + 1] == "MiniMaxAI/MiniMax-H3"
    assert args[args.index("--revision") + 1] == spec["model"]["revision"]
    assert args[args.index("--dit-cpu-offload") + 1] == "false"
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert result["workload"]["total_requests"] == 9


@pytest.mark.parametrize("field,value", [("gpu_uuids", ["0"]), ("gpu_uuids", [GPU, GPU]), ("gpu_uuids", [{}]),
                                        ("port", 22), ("lock_directory", "/"), ("schema_version", "2")])
def test_invalid_topology_fails_closed(spec, field, value):
    spec[field] = value
    with pytest.raises(ValueError):
        gpu.validate_gpu_job(spec)


@pytest.mark.parametrize("field", ["baseline", "candidate", "server", "limits"])
def test_unsupported_controls_cannot_be_silently_ignored(spec, field):
    spec[field]["shell_command"] = "echo not permitted"
    with pytest.raises(ValueError):
        gpu.validate_gpu_job(spec)


@pytest.mark.parametrize("field,value", [("compute_approved", False), ("model_license_reviewed", False), ("approval_reference", "")])
def test_authorization_blocks_before_weight_or_gpu_access(spec, tmp_path, monkeypatch, field, value):
    spec["authorization"][field] = value
    monkeypatch.setattr(gpu, "_model_manifest", lambda *a: pytest.fail("unauthorized weights accessed"))
    monkeypatch.setattr(gpu.GpuProbe, "snapshot", lambda *a: pytest.fail("unauthorized GPU access"))
    output = tmp_path / "no-output"
    with pytest.raises(ValueError, match="approval"):
        gpu.run_gpu_job(spec, output)
    assert not output.exists()


def test_model_manifest_verifies_all_bytes_and_inventory(spec):
    identity = gpu._model_manifest(spec, time.monotonic() + 10)
    assert identity["verified_files"] == 1
    assert identity["manifest_sha256"] == gpu._digest(spec["model"]["files"])
    (Path(spec["model"]["path"]) / "unrecorded.json").write_text("{}")
    with pytest.raises(ValueError, match="inventory"):
        gpu._model_manifest(spec, time.monotonic() + 10)


def test_corrupted_staged_weights_are_rejected(spec):
    (Path(spec["model"]["path"]) / "weights.safetensors").write_bytes(b"different bytes")
    with pytest.raises(ValueError, match="hash or size"):
        gpu._model_manifest(spec, time.monotonic() + 10)


def test_huggingface_snapshot_blob_symlinks_are_verified(spec, tmp_path):
    cached = tmp_path / "models--MiniMaxAI--MiniMax-H3"
    root = cached / "snapshots" / spec["model"]["revision"]
    root.mkdir(parents=True)
    blobs = cached / "blobs"
    blobs.mkdir()
    original = Path(spec["model"]["path"]) / "weights.safetensors"
    data = original.read_bytes()
    blob = blobs / "blobhash"
    blob.write_bytes(data)
    (root / "weights.safetensors").symlink_to(blob)
    spec["model"]["path"] = str(root)
    assert gpu._model_manifest(spec, time.monotonic() + 10)["verified_files"] == 1
    (root / "weights.safetensors").unlink()
    (root / "weights.safetensors").symlink_to(original)
    with pytest.raises(ValueError, match="escapes"):
        gpu._model_manifest(spec, time.monotonic() + 10)


def test_huggingface_blob_directory_cannot_redirect_authority(spec, tmp_path):
    root = tmp_path / "cache" / "snapshots" / spec["model"]["revision"]
    root.mkdir(parents=True)
    external = Path(spec["model"]["path"])
    (root.parent.parent / "blobs").symlink_to(external, target_is_directory=True)
    (root / "weights.safetensors").symlink_to(external / "weights.safetensors")
    spec["model"]["path"] = str(root)
    with pytest.raises(ValueError, match="blob directory"):
        gpu._model_manifest(spec, time.monotonic() + 10)


def test_runtime_env_does_not_inherit_secrets_or_perf_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "never-copy-this")
    monkeypatch.setenv("PYTHONPATH", "/untrusted")
    monkeypatch.setenv("SGLANG_ENABLE_CACHE_DIT", "1")
    monkeypatch.setenv("LD_PRELOAD", "/untrusted.so")
    actual = gpu._runtime_env("/pinned/source", [GPU], "nonce", tmp_path)
    assert not {"MINIMAX_API_KEY", "SGLANG_ENABLE_CACHE_DIT", "LD_PRELOAD"} & set(actual)
    assert actual["PYTHONPATH"] == "/pinned/source/python"
    assert actual["CUDA_VISIBLE_DEVICES"] == GPU
    assert actual["HF_HUB_OFFLINE"] == "1"
    if "HOME" in os.environ:
        assert actual["HOME"] == os.environ["HOME"]


@pytest.mark.parametrize("ambient", [None, "112", "999999", "auto", ""])
def test_native_thread_caps_are_explicit_and_identical_across_arms_and_clients(tmp_path, monkeypatch, ambient):
    expected = {name: "1" for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    )}
    for name in expected:
        if ambient is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, ambient)
    for role, source, devices in (
        ("baseline", "/pinned/baseline", [GPU]),
        ("candidate", "/pinned/candidate", [GPU]),
        ("client", "", []),
    ):
        actual = gpu._runtime_env(source, devices, role, tmp_path / role / "cache")
        assert {name: actual[name] for name in expected} == expected
    # Building a controlled child environment must not change the caller's policy.
    assert {name: os.environ.get(name) for name in expected} == {name: ambient for name in expected}


def test_identity_probe_records_observed_native_thread_limits(tmp_path, monkeypatch):
    expected = {name: "1" for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    )}
    for name in expected:
        monkeypatch.setenv(name, "112")
    actual = gpu._runtime_env(str(tmp_path / "source"), [], "cpu-only-identity", tmp_path / "cache")
    observed = subprocess.run(
        [sys.executable, "-c", gpu._IDENTITY], env=actual,
        check=True, capture_output=True, text=True, timeout=10,
    )
    assert json.loads(observed.stdout)["cpu_native_thread_limits"] == expected
    assert json.loads(observed.stdout)["compilation_worker_limit"] == "2"


def test_compilation_workers_do_not_inherit_host_cpu_count(tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_JOBS", "224")
    for role in ("baseline", "candidate", "client"):
        actual = gpu._runtime_env("/pinned/source", [GPU], role, tmp_path / role)
        assert actual["MAX_JOBS"] == "2"
    assert os.environ["MAX_JOBS"] == "224"


@pytest.mark.parametrize("name,relative", [
    ("SGLANG_CACHE_DIR", "sglang"),
    ("SGLANG_JIT_CACHE_DIR", "sglang/jit"),
    ("FLASHINFER_WORKSPACE_BASE", "flashinfer"),
])
def test_runtime_native_caches_are_role_private(tmp_path, monkeypatch, name, relative):
    monkeypatch.setenv(name, "/untrusted/shared-cache")
    monkeypatch.setenv("HF_TOKEN", "never-copy-this")
    baseline_cache = tmp_path / "baseline" / "cache"
    candidate_cache = tmp_path / "candidate" / "cache"
    baseline = gpu._runtime_env("/pinned/source", [GPU], "baseline", baseline_cache)
    candidate = gpu._runtime_env("/pinned/source", [GPU], "candidate", candidate_cache)
    assert baseline[name] == str(baseline_cache / relative)
    assert candidate[name] == str(candidate_cache / relative)
    assert baseline[name] != candidate[name]
    for actual in (baseline, candidate):
        assert "HF_TOKEN" not in actual
        assert actual["HF_HUB_OFFLINE"] == "1"
        assert actual["TRANSFORMERS_OFFLINE"] == "1"
        assert actual.get("HOME") == os.environ.get("HOME")
    assert not baseline_cache.exists()
    assert not candidate_cache.exists()


def test_source_tree_requires_exact_clean_committed_manifest(tmp_path):
    source = tmp_path / "source"
    package = source / "python/sglang/cli"
    package.mkdir(parents=True)
    (package / "main.py").write_text("def main(): pass\n")
    (source / "python/sglang/__init__.py").write_text("")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=CPU Test", "-c", "user.email=cpu-test@example.invalid", "commit", "-qm", "test fixture"], check=True)
    manifest = gpu.source_file_manifest(source)
    assert manifest["source_sha256"] == gpu._digest(manifest["files"])
    assert len(manifest["revision"]) == 40
    (package / "main.py").write_text("changed\n")
    with pytest.raises(ValueError, match="clean"):
        gpu.source_file_manifest(source)


def test_gpu_lease_is_exclusive_and_quarantine_is_persistent(tmp_path):
    directory = tmp_path / "lease"
    with gpu.GpuLease(directory, [GPU], "first") as lease:
        with pytest.raises(BlockingIOError):
            with gpu.GpuLease(directory, [GPU], "second"):
                pytest.fail("contended GPU lease acquired")
        lease.quarantine("unit-test simulated cleanup uncertainty")
    with pytest.raises(RuntimeError, match="quarantined"):
        with gpu.GpuLease(directory, [GPU], "third"):
            pytest.fail("quarantined GPU lease acquired")


def test_gpu_lease_rejects_symlink_and_shared_writable_directory(tmp_path):
    target = tmp_path / "real"
    target.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="lock directory"):
        with gpu.GpuLease(link, [GPU], "test"):
            pass
    target.chmod(0o777)
    with pytest.raises(ValueError, match="lock directory"):
        with gpu.GpuLease(target, [GPU], "test"):
            pass


def test_nvidia_probe_parses_observed_metrics_and_pid(spec, monkeypatch):
    outputs = iter([
        f"0, {GPU}, Test H200, 141312, 70000, 95, 590.1, 400, 60, Disabled\n".encode(),
        f"{GPU}, 1234, 69000\n".encode(),
    ])
    calls = []
    def command(argv, **kwargs):
        calls.append(argv)
        return next(outputs)
    monkeypatch.setattr(gpu, "_command", command)
    actual = gpu.GpuProbe([GPU], 0.1).snapshot()
    assert actual["gpus"][0]["memory_used_mib"] == 70000
    assert actual["compute_apps"][0]["pid"] == 1234
    assert all("--id=" + GPU in call for call in calls)
    assert not gpu._idle(actual, 1000)


@pytest.mark.parametrize("gpu_row", [f"0, {GPU}, X, 100, N/A, 0, d, 1, 1, Disabled\n",
                                  f"0, {GPU}, X, 100, 0, 0, d, 1, 1, Enabled\n", ""])
def test_incomplete_or_mig_gpu_metrics_fail_closed(monkeypatch, gpu_row):
    monkeypatch.setattr(gpu, "_command", lambda *a, **k: gpu_row.encode())
    with pytest.raises((ValueError, RuntimeError)):
        gpu.GpuProbe([GPU], 0.1).snapshot()


def sampler_fixture(tmp_path, *, owner=True, error=False):
    class Probe:
        devices = [GPU]
        timeout = 0.01
        count = 0
        def snapshot(self):
            self.count += 1
            if error:
                raise RuntimeError("telemetry unavailable")
            result = snapshot(used=100 * self.count, pid=1234)
            if self.count == 3:
                sampler.done.set()
            return result
    sampler = gpu._Sampler(Probe(), SimpleNamespace(observe=lambda pid: {"pid": pid, "pgid": 1, "session_id": 1, "start_ticks": 1} if owner else None), tmp_path / "samples.jsonl", 0.001)
    sampler.begin_measurement()
    sampler._loop()
    sampler.end_measurement()
    return sampler


def test_telemetry_records_real_sample_max_not_a_manufactured_peak(tmp_path):
    sampler = sampler_fixture(tmp_path)
    summary = sampler.summary()
    assert summary["qualified"] is True
    assert summary["observed_memory_peak_mib_by_gpu"] == {GPU: 200}
    assert summary["measurement_sample_count"] == 2
    assert summary["observed_owned_compute_by_gpu"] == {GPU: 2}
    assert "not exact" in summary["memory_semantics"]
    assert len((tmp_path / "samples.jsonl").read_text().splitlines()) == 2


@pytest.mark.parametrize("owner,error", [(False, False), (True, True)])
def test_foreign_compute_and_missing_telemetry_cannot_qualify(tmp_path, owner, error):
    sampler = sampler_fixture(tmp_path, owner=owner, error=error)
    assert sampler.failed.is_set()
    assert sampler.summary()["qualified"] is False
    assert sampler.summary()["errors"]


def test_unowned_pid_diagnostics_record_identity_uid_and_cgroup_without_secrets(monkeypatch, tmp_path):
    foreign = {"pid": 1234, "ppid": 999, "pgid": 998, "session_id": 997, "start_ticks": 123, "state": "S"}
    monkeypatch.setattr(gpu, "_proc_identity", lambda pid: copy.deepcopy(foreign))
    reads = []
    def read(pid, name):
        reads.append((pid, name))
        return ("Name:\tsecret-process-name\nUid:\t1003\t1004\t1005\t1006\n"
                if name == "status" else "0::/docker/foreign-test-container\n")
    monkeypatch.setattr(gpu, "_diagnostic_proc_file", read)
    sampler = sampler_fixture(tmp_path, owner=False)
    assert sampler.failed.is_set() and not sampler.summary()["qualified"]
    saved = json.loads((tmp_path / "samples.jsonl").read_text())
    entry = saved["unowned_compute_apps"][0]
    diagnostic = entry["process_diagnostic"]
    assert saved["owned_compute_apps"] == []
    assert diagnostic["identity"] == {"status": "observed", **foreign}
    assert diagnostic["uid"] == {"status": "observed", "real": 1003, "effective": 1004, "saved": 1005, "filesystem": 1006}
    assert diagnostic["cgroup"]["entries"] == [{"hierarchy_id": 0, "controllers": "", "path": "/docker/foreign-test-container"}]
    assert diagnostic["lifetime_check"] == "same_start_ticks"
    assert diagnostic["ownership_established"] is False and diagnostic["diagnostic_only"] is True
    assert reads == [(1234, "status"), (1234, "cgroup")]
    assert "secret-process-name" not in json.dumps(saved)
    assert sampler.errors == ["foreign or PID-namespace-invisible GPU process detected; no ownership established"]


def test_unowned_pid_disappearing_during_diagnostics_still_aborts(monkeypatch, tmp_path):
    identities = iter([{"pid": 1234, "ppid": 999, "pgid": 998, "session_id": 997, "start_ticks": 123, "state": "S"}, None])
    monkeypatch.setattr(gpu, "_proc_identity", lambda pid: next(identities))
    def missing(*args):
        raise FileNotFoundError("sensitive diagnostic exception text")
    monkeypatch.setattr(gpu, "_diagnostic_proc_file", missing)
    sampler = sampler_fixture(tmp_path, owner=False)
    diagnostic = sampler.samples[0]["unowned_compute_apps"][0]["process_diagnostic"]
    assert sampler.failed.is_set() and not sampler.summary()["qualified"]
    assert diagnostic["uid"]["status"] == diagnostic["cgroup"]["status"] == "missing"
    assert diagnostic["identity_after"]["status"] == "missing"
    assert diagnostic["lifetime_check"] == "disappeared_during_reads"
    assert "sensitive" not in json.dumps(diagnostic)


def test_permission_denied_nvml_pid_is_retained_and_aborted(monkeypatch, tmp_path):
    def denied(*args):
        raise PermissionError("do not retain this exception message")
    monkeypatch.setattr(gpu, "_proc_identity", denied)
    monkeypatch.setattr(gpu, "_diagnostic_proc_file", denied)
    owner = SimpleNamespace(observe=denied)
    probe = SimpleNamespace(snapshot=lambda: snapshot(pid=1234), devices=[GPU], timeout=0.01)
    sampler = gpu._Sampler(probe, owner, tmp_path / "denied.jsonl", 0.1)
    sampler._loop()
    assert sampler.failed.is_set() and not sampler.summary()["qualified"]
    entry = json.loads((tmp_path / "denied.jsonl").read_text())["unowned_compute_apps"][0]
    assert entry["ownership_observation"] == "permission_denied"
    diagnostic = entry["process_diagnostic"]
    assert all(diagnostic[key]["status"] == "permission_denied" for key in ("identity", "identity_after", "uid", "cgroup"))
    assert diagnostic["lifetime_check"] == "unverified"
    assert "do not retain" not in json.dumps(diagnostic)
    assert sampler.errors == ["foreign or PID-namespace-invisible GPU process detected; no ownership established"]


def test_diagnostics_cannot_promote_pid_even_when_later_identity_matches_owned_group(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu, "_unowned_process_diagnostic", lambda pid: {
        "identity": {"pid": pid, "pgid": 1, "session_id": 1, "start_ticks": 1},
        "diagnostic_only": True, "ownership_established": False})
    sampler = sampler_fixture(tmp_path, owner=False)
    assert sampler.failed.is_set() and sampler.samples[0]["owned_compute_apps"] == []
    assert len(sampler.samples[0]["unowned_compute_apps"]) == 1


def test_diagnostics_explicitly_mark_pid_reuse(monkeypatch):
    identities = iter([{"pid": 1234, "ppid": 9, "pgid": 9, "session_id": 9, "start_ticks": ticks, "state": "S"}
                       for ticks in (123, 456)])
    monkeypatch.setattr(gpu, "_proc_identity", lambda pid: next(identities))
    monkeypatch.setattr(gpu, "_diagnostic_proc_file", lambda pid, name: "Uid:\t1\t1\t1\t1\n" if name == "status" else "0::/\n")
    result = gpu._unowned_process_diagnostic(1234)
    assert result["lifetime_check"] == "pid_reused_during_reads"
    assert result["ownership_established"] is False


def test_diagnostic_proc_reader_never_opens_command_or_environment():
    for name in ("cmdline", "environ", "exe", "../status"):
        with pytest.raises(ValueError, match="unsupported"):
            gpu._diagnostic_proc_file(1234, name)


def test_diagnostic_proc_reader_has_bounded_input(monkeypatch):
    from io import StringIO
    monkeypatch.setattr(Path, "open", lambda path, **kwargs: StringIO("x" * 65537))
    with pytest.raises(ValueError, match="size limit"):
        gpu._diagnostic_proc_file(1234, "cgroup")


def test_unowned_pid_already_missing_is_explicit(monkeypatch):
    monkeypatch.setattr(gpu, "_proc_identity", lambda pid: None)
    def missing(*args):
        raise FileNotFoundError
    monkeypatch.setattr(gpu, "_diagnostic_proc_file", missing)
    result = gpu._unowned_process_diagnostic(1234)
    assert all(result[key]["status"] == "missing" for key in ("identity", "identity_after", "uid", "cgroup"))
    assert result["lifetime_check"] == "unverified" and result["ownership_established"] is False


@pytest.fixture
def fake_process(monkeypatch, tmp_path):
    state = {"identity": {"pid": 12345, "ppid": os.getpid(), "pgid": 12345, "session_id": 12345, "start_ticks": 99, "state": "S"}, "signals": []}
    class Popen:
        pid = 12345
        def __init__(self, argv, **kwargs):
            state["kwargs"] = kwargs
        def wait(self, timeout):
            state["waited"] = True
            return -15
    monkeypatch.setattr(gpu.subprocess, "Popen", Popen)
    monkeypatch.setattr(gpu, "_proc_identity", lambda pid: copy.deepcopy(state["identity"]))
    monkeypatch.setattr(gpu, "_group_members", lambda pgid: [copy.deepcopy(state["identity"])] if state["identity"]["state"] != "Z" else [])
    def killpg(pgid, sig):
        state["signals"].append((pgid, sig))
        state["identity"]["state"] = "Z"
    monkeypatch.setattr(gpu.os, "killpg", killpg)
    owned = gpu.OwnedProcess(["fake-command"], cwd=tmp_path, env={}, stdout=tmp_path / "out", stderr=tmp_path / "err", nonce="fake")
    return owned, state


def test_cleanup_only_signals_verified_own_session(fake_process):
    owned, state = fake_process
    assert state["kwargs"]["start_new_session"] is True
    assert state["kwargs"]["stdin"] is subprocess.DEVNULL
    assert owned.close(0.1)["status"] == "clean"
    assert state["signals"] == [(12345, signal.SIGTERM)]
    assert state["waited"] is True
    assert owned.close(0.1)["status"] == "clean"
    assert len(state["signals"]) == 1


@pytest.mark.parametrize("field", ["start_ticks", "pgid", "session_id"])
def test_identity_mismatch_refuses_all_group_signals(fake_process, field):
    owned, state = fake_process
    state["identity"][field] += 1
    with pytest.raises(RuntimeError, match="identity"):
        owned.close(0.1)
    assert state["signals"] == []


def test_attempt_watchdog_preempts_native_decoder_hang(tmp_path, monkeypatch):
    clock = [1.0]
    monkeypatch.setattr(gpu.time, "monotonic", lambda: clock[0])
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps({"event": "attempt_started", "slot_id": "m1"}) + "\n")
    monitor = gpu._AttemptMonitor(path, 2)
    monitor.check()
    clock[0] = 3.1
    with pytest.raises(TimeoutError, match="per-attempt"):
        monitor.check()


def test_attempt_watchdog_accepts_completed_slots_and_partial_append(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps({"event": "attempt_started", "slot_id": "m1"}) + "\n" + '{"event":')
    monitor = gpu._AttemptMonitor(path, 2)
    monitor.check()
    assert monitor.active_slot == "m1"
    path.write_text(json.dumps({"event": "attempt_started", "slot_id": "m1"}) + "\n" + json.dumps({"event": "attempt_finished", "record": {"slot_id": "m1"}}) + "\n")
    monitor.check()
    assert monitor.active_slot is None


def test_signal_cancels_submission_and_restores_handlers():
    previous = signal.getsignal(signal.SIGTERM)
    with gpu._Supervisor({"job_seconds": 10, "cleanup_seconds": 0.1}) as supervisor:
        supervisor._on_signal(signal.SIGTERM, None)
        with pytest.raises(gpu.JobCancelled, match="signal"):
            supervisor.check()
    assert signal.getsignal(signal.SIGTERM) is previous


def test_clean_group_with_residual_gpu_compute_is_cleanup_failure():
    owner = SimpleNamespace(close=lambda **kwargs: {"status": "clean", "remaining_owned_pids": []})
    sampler = SimpleNamespace(phase="measurement", done=threading.Event(), stop=lambda **kwargs: None, end_measurement=lambda: None)
    probe = SimpleNamespace(snapshot=lambda **kwargs: snapshot(used=2000, pid=999))
    result = gpu._cleanup_role(owner, sampler, probe, {"cleanup_seconds": 0.1, "max_idle_memory_mib": 1000})
    assert result["status"] == "failed"
    assert result["idle_after"] is False


def test_completed_hardware_measurement_is_not_uncalibrated_ci_pass(spec):
    receipt = controlled_receipt(spec)
    result = gpu._gate(spec, receipt, comparison(spec))
    assert receipt["measurement_status"] == "complete"
    assert result["ci_accepted"] is False
    assert result["regression_status"] == "inconclusive"
    assert result["release_qualified"] is False
    assert any("calibrated" in reason for reason in result["acceptance_reasons"])


def test_strict_ci_requires_calibrated_complete_dedicated_evidence(spec, monkeypatch):
    spec["allocation"]["mode"] = "dedicated_ci"
    monkeypatch.setattr(gpu, "_calibration", lambda *args, **kwargs: (True, "CPU unit-test calibration fake"))
    result = gpu._gate(spec, controlled_receipt(spec), comparison(spec), verified_evidence={})
    assert result["ci_accepted"] is True
    assert result["regression_status"] == "pass"
    assert result["release_qualified"] is False


@pytest.mark.parametrize("change", ["fixture", "imported", "hardware", "descriptive", "cleanup", "telemetry", "source", "incomplete", "mixed_dependencies"])
def test_no_false_green_when_required_evidence_changes(spec, monkeypatch, change):
    spec["allocation"]["mode"] = "dedicated_ci"
    monkeypatch.setattr(gpu, "_calibration", lambda *args, **kwargs: (True, "CPU fake"))
    receipt, compared = controlled_receipt(spec), comparison(spec)
    role = receipt["roles"]["candidate"]
    if change in {"fixture", "imported"}:
        compared["evidence_kind"] = "fixture" if change == "fixture" else "imported_media"
    elif change == "hardware":
        role["telemetry_summary"]["gpu_identity"][0]["uuid"] = "wrong-device"
    elif change == "descriptive":
        compared["measurement"]["performance_mode"] = "descriptive_only"
    elif change == "cleanup":
        role["cleanup"]["status"] = "failed"
    elif change == "telemetry":
        role["telemetry_summary"]["qualified"] = False
    elif change == "source":
        role["source_identity"]["source_sha256"] = "d" * 64
    elif change == "incomplete":
        receipt["measurement_status"] = "incomplete"
    elif change == "mixed_dependencies":
        role["source_identity"]["packages"]["torch"] = "different"
    result = gpu._gate(spec, receipt, compared)
    assert result["ci_accepted"] is False
    assert result["regression_status"] == "inconclusive"


def test_memory_regression_is_observed_and_never_hidden_by_uncalibrated_policy(spec):
    receipt = controlled_receipt(spec)
    receipt["roles"]["candidate"]["telemetry_summary"]["observed_memory_peak_mib_by_gpu"][GPU] = 150
    result = gpu._gate(spec, receipt, comparison(spec))
    assert result["memory_increase_fraction"] == 0.5
    assert result["memory_gate_status"] == "fail"
    assert result["regression_status"] == "fail"
    assert result["ci_accepted"] is False


def test_calibrated_label_without_calibration_jobs_is_not_evidence(spec):
    spec["policy"]["calibration_status"] = "operator_calibrated"
    assert gpu._calibration(spec)[0] is False
    spec["policy"]["calibration_evidence"] = [{"job_path": "/missing", "sha256": SHA}]
    assert gpu._calibration(spec)[0] is False


def test_saved_evidence_hash_tampering_blocks_re_evaluation(spec, tmp_path):
    directory = tmp_path / "saved"
    directory.mkdir()
    receipt = controlled_receipt(spec)
    gpu._write(directory / "spec.json", spec)
    data = directory / "run.json"
    data.write_text("{}")
    receipt["roles"]["baseline"].update(run_path="run.json", run_sha256=gpu._hash(data))
    gpu._write(directory / "gpu-job.json", receipt)
    data.write_text('{"tampered":true}')
    with pytest.raises(ValueError):
        gpu.evaluate_gpu_job(directory)


@pytest.mark.skipif(platform.system() != "Linux", reason="real /proc process ownership is Linux-only; no GPU needed")
def test_real_owned_session_cleanup_does_not_kill_foreign_process(tmp_path):
    foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    owned = gpu.OwnedProcess([sys.executable, "-c", "import time; time.sleep(30)"], cwd=tmp_path, env={},
                            stdout=tmp_path / "owned.out", stderr=tmp_path / "owned.err", nonce="cpu-only")
    try:
        assert owned.close(0.5)["status"] == "clean"
        assert foreign.poll() is None
    finally:
        foreign.terminate()
        foreign.wait(timeout=3)


@pytest.mark.skipif(platform.system() != "Linux", reason="real /proc process ownership is Linux-only; no GPU needed")
def test_unreaped_leader_preserves_identity_while_child_remains(tmp_path):
    owned = gpu.OwnedProcess([sys.executable, "-c", "import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])"],
                            cwd=tmp_path, env={}, stdout=tmp_path / "out", stderr=tmp_path / "err", nonce="cpu-only")
    end = time.monotonic() + 3
    while owned.running() and time.monotonic() < end:
        time.sleep(0.02)
    assert gpu._proc_identity(owned.identity["pid"])["state"] == "Z"
    assert owned.close(0.5)["status"] == "clean"


def saved_job(spec, directory, *, day=3, candidate_invalid=False):
    """Forge CPU test metadata to exercise integrity rules, NEVER model evidence."""
    spec = gpu.validate_gpu_job(copy.deepcopy(spec))
    directory.mkdir()
    gpu._write(directory / "spec.json", spec)
    receipt = controlled_receipt(spec)
    receipt.update(execution_id=directory.name, failures=[], cleanup_status="clean",
                   started_at=f"2026-08-{day:02d}T00:00:00Z", finished_at=f"2026-08-{day:02d}T01:00:00Z",
                   comparison_path="comparison.json")
    receipt["model_identity"]["revision"] = spec["model"]["revision"]
    compared = comparison(spec, "fail" if candidate_invalid else "pass")
    compared.update(policy=spec["policy"], checks=[{"name": "test.control", "status": "pass"}], slots=[])
    runs = {}
    for number, label in enumerate(("baseline", "candidate")):
        run_dir = directory / label
        (run_dir / "artifacts").mkdir(parents=True)
        metadata = directory / "supervisor" / label
        metadata.mkdir(parents=True)
        role = receipt["roles"][label]
        role.update(client_exit_code=1 if candidate_invalid and label == "candidate" else 0,
                    client_cleanup={"status": "clean"}, gpu_before=snapshot(),
                    run_path=f"{label}/run.json", telemetry_path=f"supervisor/{label}/telemetry.jsonl")
        role["cleanup"]["gpu_after"] = snapshot()
        role["process_identity"].update(pid=500+number, pgid=500+number, session_id=500+number, launch_nonce=f"{directory.name}-{label}")
        records = []
        for index, slot in enumerate(_slots(spec["plan"])):
            relative = f"artifacts/{index}.mp4"
            artifact = run_dir / relative
            artifact.write_bytes(f"CPU metadata test only {directory.name}-{label}-{index}".encode())
            invalid = candidate_invalid and label == "candidate" and index == 1
            records.append({**{key: slot[key] for key in ("slot_id", "case_id", "prompt", "seed", "repetition", "phase")},
                            "status": "succeeded", "attempted": True, "artifact_path": relative, "sha256": gpu._hash(artifact),
                            "media": {"valid": not invalid}, "latency_seconds": 0.001,
                            "submit_to_terminal_seconds": 0.0002, "submit_to_media_seconds": 0.0005, "media_validation_seconds": 0.0003})
        config = {"runtime": "sglang", "runtime_revision": spec[label]["revision"], "model_id": spec["plan"]["model_id"],
                  "model_revision": spec["model"]["revision"], "hardware_label": GPU,
                  "client_source_sha256": "d" * 64, "client_environment": {"python": "3.12.1"},
                  "media_evaluator": {"source_sha256": "e" * 64}, "measurement_semantics": {"latency": "real-test-timer"}, "limits": {"timeout_seconds": 5}}
        config["configuration_sha256"] = gpu._digest(config)
        scheduled = sum(row["phase"] == "measurement" for row in records)
        valid = sum(row["phase"] == "measurement" and row["media"]["valid"] for row in records)
        run = {"bundle_type": "mvp_run", "bundle_version": "0.1.0", "run_id": f"{directory.name}-{label}", "evidence_kind": "operator_endpoint",
               "status": "complete" if valid == scheduled else "partial", "started_at": receipt["started_at"], "finished_at": receipt["finished_at"],
               "plan": spec["plan"], "plan_sha256": gpu._digest(spec["plan"]), "configuration": config, "records": records,
               "summary": {"scheduled": scheduled, "valid": valid, "failed": scheduled-valid},
               "measurement": {"boundary": "submit_to_validated_media", "concurrency": 1, "wall_seconds": 0.01}}
        gpu._write(run_dir / "run.json", run)
        role["run_sha256"] = gpu._hash(run_dir / "run.json")
        compared[label] = {"run_id": run["run_id"], "run_bundle_sha256": role["run_sha256"]}
        runs[label] = run
        samples = []
        for stamp in (1.01, 1.09):
            sample = snapshot(used=100, pid=role["process_identity"]["pid"])
            sample.update(monotonic_seconds=stamp, phase="measurement", unowned_compute_apps=[])
            sample["owned_compute_apps"] = [{**sample["compute_apps"][0], "process_identity": role["process_identity"]}]
            samples.append(sample)
        telemetry = metadata / "telemetry.jsonl"
        telemetry.write_bytes(b"".join(gpu.canonical_json_bytes(sample)+b"\n" for sample in samples))
        role["telemetry_sha256"] = gpu._hash(telemetry)
        role["telemetry_summary"] = gpu.summarize_gpu_samples(samples, [GPU], spec["limits"]["telemetry_interval_seconds"],
                                                            window_start=1, window_end=1.1, command_seconds=spec["limits"]["command_seconds"])
    for left, right in zip(runs["baseline"]["records"][1:], runs["candidate"]["records"][1:]):
        status = "pass" if right["media"]["valid"] else "fail"
        compared["slots"].append({"slot_id": left["slot_id"], "baseline": left, "candidate": right,
                                  "checks": [{"name": "candidate.valid_media", "status": status}]})
    gpu._write(directory / "comparison.json", compared)
    receipt["comparison_sha256"] = gpu._hash(directory / "comparison.json")
    gpu._write(directory / "gpu-job.json", receipt)
    return directory


def test_saved_measurement_integrity_recomputes_raw_telemetry(spec, tmp_path):
    from evaluator.mvp_gpu_evidence import verify_measurement_job
    directory = saved_job(spec, tmp_path / "current")
    verified = verify_measurement_job(directory, deadline=time.monotonic() + 5)
    assert verified["receipt"]["measurement_status"] == "complete"
    assert gpu.evaluate_gpu_job(directory)["regression_status"] == "inconclusive"
    receipt = gpu._read(directory / "gpu-job.json")
    receipt["roles"]["candidate"]["telemetry_summary"]["observed_memory_peak_mib_by_gpu"][GPU] = 1
    gpu._write(directory / "gpu-job.json", receipt)
    with pytest.raises(ValueError, match="telemetry summary"):
        verify_measurement_job(directory, deadline=time.monotonic() + 5)


def test_saved_known_media_regression_yields_fail_not_inconclusive(spec, tmp_path):
    directory = saved_job(spec, tmp_path / "candidate-defect", candidate_invalid=True)
    result = gpu.evaluate_gpu_job(directory)
    assert result["measurement_status"] == "complete"
    assert result["regression_status"] == "fail"
    assert result["ci_accepted"] is False


@pytest.mark.parametrize("mutation", ["artifact", "run_hash", "stage_timing", "pid", "missing_slot", "comparison_link", "denominator", "cadence"])
def test_saved_corrupt_or_incomplete_evidence_cannot_qualify(spec, tmp_path, mutation):
    directory = saved_job(spec, tmp_path / "bad")
    receipt = gpu._read(directory / "gpu-job.json")
    role = receipt["roles"]["candidate"]
    run_path = directory / role["run_path"]
    run = gpu._read(run_path)
    if mutation == "artifact":
        (run_path.parent / run["records"][1]["artifact_path"]).write_bytes(b"changed")
    elif mutation == "run_hash":
        role["run_sha256"] = "0" * 64
    elif mutation in {"stage_timing", "missing_slot", "denominator"}:
        if mutation == "stage_timing":
            run["records"][1].pop("submit_to_terminal_seconds")
        elif mutation == "missing_slot":
            run["records"].pop()
        else:
            run["summary"]["failed"] = 9000
        gpu._write(run_path, run)
        role["run_sha256"] = gpu._hash(run_path)
    elif mutation == "comparison_link":
        compared = gpu._read(directory / "comparison.json")
        compared["baseline"]["run_bundle_sha256"] = "0" * 64
        gpu._write(directory / "comparison.json", compared)
        receipt["comparison_sha256"] = gpu._hash(directory / "comparison.json")
    else:
        telemetry = directory / role["telemetry_path"]
        samples = [json.loads(line) for line in telemetry.read_text().splitlines()]
        if mutation == "pid":
            samples[0]["owned_compute_apps"][0]["process_identity"]["pgid"] = 999999
        else:
            samples[-1]["monotonic_seconds"] = 999
        telemetry.write_bytes(b"".join(gpu.canonical_json_bytes(sample)+b"\n" for sample in samples))
        role["telemetry_sha256"] = gpu._hash(telemetry)
    gpu._write(directory / "gpu-job.json", receipt)
    with pytest.raises(ValueError):
        gpu.evaluate_gpu_job(directory)


def calibrated_jobs(spec, tmp_path):
    first = saved_job(spec, tmp_path / "prior-one", day=1)
    second = saved_job(spec, tmp_path / "prior-two", day=2)
    calibrated = copy.deepcopy(spec)
    calibrated["allocation"]["mode"] = "dedicated_ci"
    calibrated["policy"]["calibration_status"] = "operator_calibrated"
    calibrated["policy"]["calibration_evidence"] = [{"job_path": str(path / "gpu-job.json"), "sha256": gpu._hash(path / "gpu-job.json")} for path in (first, second)]
    current = saved_job(calibrated, tmp_path / "independent-current", day=3)
    return current, first, second


def test_calibrated_ci_pass_has_verified_nonrecursive_prior_evidence(spec, tmp_path):
    current, first, second = calibrated_jobs(spec, tmp_path)
    result = gpu.evaluate_gpu_job(current)
    assert result["calibration"]["verified"] is True
    assert result["ci_accepted"] is True
    assert result["regression_status"] == "pass"
    assert result["release_qualified"] is False


@pytest.mark.parametrize("mutation", ["duplicate", "future", "different_cell", "different_build", "raw_media", "missing_prior"])
def test_bad_calibration_never_becomes_green(spec, tmp_path, mutation):
    current, first, second = calibrated_jobs(spec, tmp_path)
    prior_path = first / "gpu-job.json"
    prior = gpu._read(prior_path)
    if mutation == "raw_media":
        (first / "baseline/artifacts/1.mp4").write_bytes(b"changed")
    elif mutation == "future":
        prior["finished_at"] = "2026-08-04T00:00:00Z"
        gpu._write(prior_path, prior)
    elif mutation == "duplicate":
        prior["execution_id"] = gpu._read(second / "gpu-job.json")["execution_id"]
        gpu._write(prior_path, prior)
    elif mutation in {"different_cell", "different_build"}:
        prior_spec = gpu._read(first / "spec.json")
        if mutation == "different_cell":
            prior_spec["server"]["performance_mode"] = "memory"
        else:
            prior_spec["baseline"]["revision"] = "d" * 40
        gpu._write(first / "spec.json", prior_spec)
        prior["spec_sha256"] = gpu._digest(prior_spec)
        gpu._write(prior_path, prior)
    else:
        prior_path.unlink()
    if mutation != "missing_prior":
        current_spec = gpu._read(current / "spec.json")
        current_spec["policy"]["calibration_evidence"][0]["sha256"] = gpu._hash(prior_path)
        gpu._write(current / "spec.json", current_spec)
        receipt = gpu._read(current / "gpu-job.json")
        receipt["spec_sha256"] = gpu._digest(current_spec)
        compared = gpu._read(current / "comparison.json")
        compared["policy"] = current_spec["policy"]
        gpu._write(current / "comparison.json", compared)
        receipt["comparison_sha256"] = gpu._hash(current / "comparison.json")
        gpu._write(current / "gpu-job.json", receipt)
    result = gpu.evaluate_gpu_job(current)
    assert result["ci_accepted"] is False
    assert result["regression_status"] == "inconclusive"
    assert result["calibration"]["verified"] is False


def test_portable_calibration_evidence_remains_hash_bound(spec, tmp_path):
    import shutil
    current, first, second = calibrated_jobs(spec, tmp_path)
    for prior in (first, second):
        digest = gpu._hash(prior / "gpu-job.json")
        shutil.copytree(prior, current / "calibration" / digest)
        (prior / "gpu-job.json").unlink()
    assert gpu.evaluate_gpu_job(current)["ci_accepted"] is True


def test_occupied_gpu_preflight_never_spawns_a_runtime(spec, tmp_path, monkeypatch):
    directory = tmp_path / "occupied-job"
    directory.mkdir()
    supervisor = SimpleNamespace(deadline=time.monotonic()+10, total_deadline=time.monotonic()+10,
                                 cancelled=threading.Event(), spawn=lambda *a, **k: pytest.fail("occupied GPU launched a runtime"))
    monkeypatch.setattr(gpu, "_source_identity", lambda *a, **k: {"revision": REV, "source_sha256": SHA})
    probe = SimpleNamespace(snapshot=lambda: snapshot(used=5000, pid=9876))
    receipt = {"roles": {}}
    with pytest.raises(RuntimeError, match="existing compute"):
        gpu._role(spec, "baseline", directory, supervisor, probe, receipt)
    assert receipt["roles"]["baseline"]["cleanup"]["status"] == "not_started"


@pytest.mark.parametrize("startup_exits", [False, True])
def test_role_safe_cleanup_preserves_known_candidate_failure(spec, tmp_path, monkeypatch, startup_exits):
    directory = tmp_path / "candidate-job"
    directory.mkdir()
    identity = {"pid": 500, "pgid": 500, "session_id": 500, "start_ticks": 100, "launch_nonce": "fake"}
    cleanups = []
    class Process:
        def __init__(self):
            self.identity = identity
        def running(self):
            return not startup_exits
        def close(self, **kwargs):
            cleanups.append(kwargs["deadline"])
            return {"status": "clean", "remaining_owned_pids": []}
        def check_output_budget(self):
            pass
    class Sampler:
        def __init__(self, probe, owner, path, interval):
            path.touch()
            self.done = threading.Event()
            self.failed = threading.Event()
        def start(self): pass
        def begin_measurement(self): pass
        def end_measurement(self): pass
        def stop(self, **kwargs): pass
        def summary(self): return {"qualified": True}
    def spawn(argv, **kwargs):
        if "evaluator.cli" in argv:
            run_dir = directory / "candidate"
            run_dir.mkdir()
            gpu._write(run_dir / "run.json", {"status": "partial", "evidence_kind": "operator_endpoint", "plan_sha256": gpu._digest(spec["plan"]),
                                             "finished_at": "2026-09-01T00:00:00Z", "summary": {"valid": 7, "failed": 1, "scheduled": 8}})
        return Process()
    supervisor = SimpleNamespace(deadline=time.monotonic()+10, total_deadline=time.monotonic()+10,
                                 cancelled=threading.Event(), check=lambda: None, spawn=spawn)
    monkeypatch.setattr(gpu, "_source_identity", lambda *a, **k: {"revision": REV, "source_sha256": SHA})
    monkeypatch.setattr(gpu, "source_file_manifest", lambda *a, **k: {"revision": REV, "source_sha256": SHA})
    monkeypatch.setattr(gpu, "_port_available", lambda *a: None)
    monkeypatch.setattr(gpu, "_owned_listener", lambda *a: True)
    monkeypatch.setattr(gpu, "_health", lambda *a: True)
    monkeypatch.setattr(gpu, "_Sampler", Sampler)
    monkeypatch.setattr(gpu, "_wait_client", lambda *a, **k: 1)
    probe = SimpleNamespace(snapshot=lambda **kwargs: snapshot())
    receipt = {"roles": {}}
    if startup_exits:
        with pytest.raises(RuntimeError, match="before readiness"):
            gpu._role(spec, "candidate", directory, supervisor, probe, receipt)
    else:
        result = gpu._role(spec, "candidate", directory, supervisor, probe, receipt)
        assert result["status"] == "complete"
        assert result["run_status"] == "partial"
        assert result["client_exit_code"] == 1
    assert receipt["roles"]["candidate"]["cleanup"]["status"] == "clean"
    assert cleanups and max(cleanups) <= supervisor.total_deadline


def test_cancelled_preflight_hashing_stops_without_finishing_weights(spec):
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(gpu.JobCancelled, match="pinned files"):
        gpu._model_manifest(spec, time.monotonic()+60, cancelled)


def test_invalid_calibration_cannot_hide_a_known_comparison_failure(spec, monkeypatch):
    def invalid(*args, **kwargs):
        raise FileNotFoundError("missing calibration fixture")
    monkeypatch.setattr(gpu, "_calibration", invalid)
    result = gpu._gate(spec, controlled_receipt(spec), comparison(spec, "fail"), verified_evidence={})
    assert result["regression_status"] == "fail"
    assert result["ci_accepted"] is False
    assert result["calibration"]["verified"] is False


@pytest.mark.parametrize("reader", [gpu._hash, gpu._read])
def test_special_files_cannot_block_evidence_reads(tmp_path, reader):
    fifo = tmp_path / "gpu-job.json"
    os.mkfifo(fifo)
    started = time.monotonic()
    with pytest.raises(ValueError, match="regular file"):
        reader(fifo)
    assert time.monotonic() - started < 1


@pytest.mark.parametrize("reader", [gpu._hash, gpu._read])
def test_final_evidence_symlink_is_not_followed(tmp_path, reader):
    target = tmp_path / "target.json"
    target.write_text("{}")
    link = tmp_path / "gpu-job.json"
    link.symlink_to(target)
    with pytest.raises(OSError):
        reader(link)


def test_calibration_preflight_verifies_known_cell_before_gpu_work(spec, tmp_path):
    from evaluator.mvp_gpu_evidence import preflight_calibration
    current, first, second = calibrated_jobs(spec, tmp_path)
    frozen = gpu._read(current / "spec.json")
    result = preflight_calibration(frozen, current, started_at="2026-08-03T00:00:00Z", execution_id="not-yet-run", deadline=time.monotonic()+5)
    assert result["status"] == "passed"
    assert result["performed_before_gpu_lease"] is True
    assert result["reference_count"] == 2
    assert result["verified_prior_execution_ids"] == ["prior-one", "prior-two"]
    frozen["server"]["performance_mode"] = "memory"
    with pytest.raises(ValueError, match="current cell"):
        preflight_calibration(frozen, current, started_at="2026-08-03T00:00:00Z", execution_id="not-yet-run", deadline=time.monotonic()+5)


@pytest.mark.parametrize("bad", ["missing", "malformed", "raw_artifact", "wrong_cell", "future", "duplicate"])
def test_bad_calibration_run_preflight_never_reaches_weights_gpu_or_lease(spec, tmp_path, monkeypatch, bad):
    current, first, second = calibrated_jobs(spec, tmp_path)
    frozen = gpu._read(current / "spec.json")
    references = frozen["policy"]["calibration_evidence"]
    if bad == "missing":
        references[0]["job_path"] = str(tmp_path / "does-not-exist/gpu-job.json")
    elif bad == "malformed":
        references[0] = {"bad": "not a reference"}
    elif bad == "raw_artifact":
        (first / "baseline/artifacts/1.mp4").write_bytes(b"changed after prior receipt")
    elif bad == "wrong_cell":
        frozen["server"]["performance_mode"] = "memory"
    elif bad == "future":
        receipt = gpu._read(first / "gpu-job.json")
        receipt["finished_at"] = "2099-01-01T00:00:00Z"
        gpu._write(first / "gpu-job.json", receipt)
        references[0]["sha256"] = gpu._hash(first / "gpu-job.json")
    else:
        references[1] = copy.deepcopy(references[0])
    monkeypatch.setattr(gpu, "_require_linux", lambda: None)
    monkeypatch.setattr(gpu, "_model_manifest", lambda *a, **k: pytest.fail("bad calibration reached current weights"))
    monkeypatch.setattr(gpu, "GpuProbe", lambda *a, **k: pytest.fail("bad calibration constructed a GPU probe"))
    monkeypatch.setattr(gpu, "GpuLease", lambda *a, **k: pytest.fail("bad calibration constructed a GPU lease"))
    monkeypatch.setattr(gpu._Supervisor, "spawn", lambda *a, **k: pytest.fail("bad calibration launched a process"))
    output = tmp_path / "blocked-before-gpu"
    result = gpu.run_gpu_job(frozen, output)
    assert result["status"] == "failed"
    assert result["ci_accepted"] is False
    assert result["roles"] == {}
    assert result["calibration_preflight"]["status"] == "failed"
    assert result["calibration_preflight"]["performed_before_gpu_lease"] is True
    assert gpu._read(output / "gpu-job.json")["calibration_preflight"] == result["calibration_preflight"]


def test_calibration_preflight_observes_hard_work_deadline(spec, tmp_path):
    from evaluator.mvp_gpu_evidence import preflight_calibration
    current, _, _ = calibrated_jobs(spec, tmp_path)
    with pytest.raises(TimeoutError):
        preflight_calibration(gpu._read(current / "spec.json"), current, started_at="2026-08-03T00:00:00Z",
                              execution_id="not-yet-run", deadline=time.monotonic()-1)
