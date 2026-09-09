"""Control-path tests use a fake scheduler. They do not claim GPU execution."""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import ci


def config(tmp_path):
    return {"schema_version": 1, "task_id": "h3-test", "workspace": {"host": str(tmp_path), "container": "/work"},
            "runtime": {"entry": str(tmp_path / "entry.sh"), "entry_sha256": "a" * 64,
                        "rootfs": str(tmp_path / "rootfs"), "ready_marker": str(tmp_path / "ready"), "python": "/opt/harness/bin/python"},
            "spec": {"path": str(tmp_path / "spec.json"), "sha256": "b" * 64},
            "resources": {"gpus": 4, "cpus": 32, "memory_gb": 256, "minutes": 90},
            "allocation_receipts": [], "mode": "smoke"}


def allocation(tmp_path, job="123"):
    identity = {"JobId": job, "JobName": "owned-holder", "Comment": "h3:owner-nonce", "WorkDir": str(tmp_path),
                "Account": "sa-shared", "Partition": "main", "UserId": f"tester({os.getuid()})"}
    receipt = {"task_id": "h3-test", "identity": identity}
    record = {**identity, "JobState": "RUNNING", "NumNodes": "1", "NodeList": "h200-node",
              "AllocTRES": "cpu=64,mem=512G,node=1,gres/gpu=8", "NumCPUs": "64", "OverSubscribe": "NO",
              "EndTime": (datetime.now(timezone.utc) + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S")}
    return receipt, record


def save_receipt(root, receipt):
    path = root / "old" / "allocation.json"
    path.parent.mkdir(parents=True)
    ci.write(path, receipt)
    return path


def test_h100_site_keeps_full_allocation_separate_from_participating_gpus(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNNER_NAME", "h3-test-runner")
    cfg = config(tmp_path)
    cfg.update(mode="serving-smoke", site={"cluster": "h100-dgxc", "partition": "hpc-gpu-1", "account": "customer", "gpu_model": "H100"})
    cfg["resources"].update(gpus=4, allocated_gpus=8)
    ci.validate_config(cfg)
    commands = []
    def run(argv, **kwargs):
        commands.append(argv)
        return SimpleNamespace(stdout="salloc: Granted job allocation 123", stderr="", returncode=0)
    monkeypatch.setattr(ci.subprocess, "run", run)
    monkeypatch.setattr(ci, "command", lambda argv: "tester")
    receipt = ci.allocate(cfg, tmp_path)
    assert "--partition=hpc-gpu-1" in commands[0] and "--account=customer" in commands[0]
    assert "--exclusive" in commands[0] and "--gres=gpu:8" in commands[0]
    assert receipt["site"] == cfg["site"]
    _, record = allocation(tmp_path)
    record.update(receipt["identity"])
    ci.verify_identity(receipt, record, cfg["task_id"])
    step = ci.step_argv(cfg, receipt, record, tmp_path, tmp_path)
    assert "--gpus-per-task=4" in step
    record["Account"] = "other"
    with pytest.raises(ValueError, match="identity differs"):
        ci.verify_identity(receipt, record, cfg["task_id"])


def test_amd_granted_full_node_requires_explicit_gpu_evidence(tmp_path):
    cfg = config(tmp_path)
    cfg.update(mode="serving-smoke", site=dict(ci.AMD_SITE))
    cfg["resources"].update(gpus=4, allocated_gpus=8)
    ci.validate_config(cfg)
    receipt, record = allocation(tmp_path)
    record.update(Account=ci.AMD_SITE["account"], Partition="compute", AllocTRES="cpu=128,mem=512G,node=1,billing=128", TresPerNode="gres/gpu:8")
    assert ci.allocated_gpu_count(record) == 8
    assert ci.capacity(record, cfg["resources"]) is None
    assert "--gres=gpu:8" in ci.step_argv(cfg, receipt, record, tmp_path, tmp_path)
    record["OverSubscribe"] = "OK"
    assert ci.allocated_gpu_count(record) is None
    assert ci.capacity(record, cfg["resources"]) == "insufficient allocated GPU/CPU/memory capacity"
    record.update(OverSubscribe="NO", AllocTRES="cpu=128,mem=512G,node=1,gres/gpu=4")
    assert ci.allocated_gpu_count(record) == 4


@pytest.mark.parametrize("change", [
    {"site": {"cluster": "h100-dgxc", "partition": "hpc-gpu-1", "account": "customer", "gpu_model": "H200"}},
    {"site": {"cluster": "unknown", "partition": "main", "account": "customer", "gpu_model": "H200"}},
    {"resources": {"gpus": 4, "allocated_gpus": 2, "cpus": 32, "memory_gb": 512, "minutes": 90}},
])
def test_invalid_hardware_or_allocation_budget_is_rejected(tmp_path, change):
    cfg = config(tmp_path)
    cfg.update(change)
    with pytest.raises(ValueError):
        ci.validate_config(cfg)


def test_explicit_concurrency_selection_requires_serving_mode(tmp_path):
    cfg = config(tmp_path)
    cfg["concurrencies"] = [4]
    with pytest.raises(ValueError, match="requires serving-smoke"):
        ci.validate_config(cfg)
    cfg["mode"] = "serving-smoke"
    ci.validate_config(cfg)
    cfg["concurrencies"] = [1, 1]
    with pytest.raises(ValueError, match="unique concurrency"):
        ci.validate_config(cfg)


@pytest.mark.parametrize("mode", ["smoke", "serving-smoke"])
def test_allocation_submits_from_receipted_work_directory(tmp_path, monkeypatch, mode):
    run_dir = tmp_path / "results"
    run_dir.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    scheduler = bin_dir / "salloc"
    scheduler.write_text('#!/bin/sh\npwd > "$H3_TEST_SCHEDULER_CWD"\necho "salloc: Granted job allocation 123"\n')
    scheduler.chmod(0o755)
    observed = tmp_path / "scheduler-cwd"
    monkeypatch.setenv("RUNNER_NAME", "h3-test-runner")
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("H3_TEST_SCHEDULER_CWD", str(observed))
    cfg = config(tmp_path)
    cfg["mode"] = mode
    cfg["resources"]["gpus"] = 2
    receipt = ci.allocate(cfg, run_dir)
    assert observed.read_text().strip() == receipt["identity"]["WorkDir"] == str(run_dir)
    argv = json.loads((run_dir / "allocation-command.json").read_text())
    assert ("--exclusive" in argv) is (mode == "smoke")
    assert ("--gres=gpu:2" if mode == "serving-smoke" else "--gres=gpu:8") in argv


def test_reuse_checks_identity_and_retains_active_step_evidence(tmp_path, monkeypatch):
    receipt, record = allocation(tmp_path)
    save_receipt(tmp_path, receipt)
    calls = []
    def command(argv, **kwargs):
        calls.append(argv)
        if "--steps" in argv:
            assert "--format=%i|%N" in argv
            return "123.0|h200-node\n"
        return "123\n"
    monkeypatch.setattr(ci, "command", command)
    monkeypatch.setattr(ci, "job_record", lambda job: record)
    found = ci.recover(config(tmp_path), tmp_path)
    assert found["action"] == "reuse"
    assert "123.0" in found["active_steps"]
    assert not any(argv[0] in {"salloc", "scancel"} for argv in calls)
    record["Comment"] = "another-task"
    with pytest.raises(ValueError, match="identity differs"):
        ci.recover(config(tmp_path), tmp_path)


def test_expired_capacity_records_reason_before_new_allocation(tmp_path, monkeypatch):
    receipt, record = allocation(tmp_path)
    save_receipt(tmp_path, receipt)
    record["EndTime"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    monkeypatch.setattr(ci, "command", lambda argv, **kw: "123\n")
    monkeypatch.setattr(ci, "job_record", lambda job: record)
    found = ci.recover(config(tmp_path), tmp_path)
    assert found["action"] == "allocate"
    assert "remaining" in found["reasons"][0]["reason"]


def test_pending_or_unknown_submission_never_allocates(tmp_path, monkeypatch):
    receipt, record = allocation(tmp_path)
    path = save_receipt(tmp_path, receipt)
    record["JobState"] = "PENDING"
    monkeypatch.setattr(ci, "command", lambda argv, **kw: "123\n")
    monkeypatch.setattr(ci, "job_record", lambda job: record)
    assert ci.recover(config(tmp_path), tmp_path)["action"] == "wait"
    path.unlink()
    ci.write(path.with_name("allocation-intent.json"), {})
    with pytest.raises(ValueError, match="Unresolved allocation intent"):
        ci.recover(config(tmp_path), tmp_path)


def test_scheduler_failure_does_not_mean_empty_queue(tmp_path, monkeypatch):
    def failed(argv):
        raise subprocess.CalledProcessError(1, argv)
    monkeypatch.setattr(ci, "command", failed)
    with pytest.raises(subprocess.CalledProcessError):
        ci.recover(config(tmp_path), tmp_path)


def test_step_uses_bound_suballocation_and_persistent_entry(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    receipt, record = allocation(tmp_path)
    monkeypatch.setenv("SLURM_GPUS_PER_NODE", "8")
    monkeypatch.setenv("SLURM_TRES_PER_TASK", "gres/gpu:8")
    monkeypatch.setenv("SALLOC_PARTITION", "another-provider")
    env = ci.environment()
    assert not any(key.startswith(("SLURM_", "SALLOC_")) for key in env)
    argv = ci.step_argv(cfg, receipt, record, tmp_path / "results", tmp_path / "package")
    assert "--gpus-per-task=4" in argv and "--exclusive" in argv and "--exact" in argv
    assert "--cpus-per-task=32" in argv and "--cpu-bind=verbose,cores" in argv
    assert "--time=85" in argv
    assert argv[-4:] == ["python3", str(tmp_path / "package" / "ci.py"), "--enter", str(tmp_path / "results")]
    assert not any("container-image" in arg or "overlap" in arg for arg in argv)


def test_cleanup_only_cancels_bound_owned_step(tmp_path, monkeypatch):
    receipt, record = allocation(tmp_path)
    ci.write(tmp_path / "binding.json", {"job_id": "123", "step_id": "7"})
    queues = iter(["123.7\n123.4\n", "123.4\n"])
    calls = []
    def command(argv, **kw):
        calls.append(argv)
        return next(queues) if argv[0] == "squeue" else ""
    monkeypatch.setattr(ci, "command", command)
    monkeypatch.setattr(ci, "job_record", lambda job: record)
    assert ci.drain_step(receipt, "h3-test", tmp_path)["status"] == "ended"
    assert [call for call in calls if call[0] == "scancel"] == [["scancel", "123.7"]]


@pytest.mark.parametrize("finishes", [True, False])
def test_allocation_cleanup_waits_for_slurm_epilog_with_finite_deadline(tmp_path, monkeypatch, finishes):
    receipt, record = allocation(tmp_path)
    elapsed = [0]
    monkeypatch.setattr(ci, "time", SimpleNamespace(monotonic=lambda: elapsed[0], sleep=lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds)))
    monkeypatch.setattr(ci, "job_record", lambda job: record)
    def command(argv):
        if argv[0] == "scancel":
            return ""
        return "" if finishes and elapsed[0] >= 20 else "COMPLETING\n"
    monkeypatch.setattr(ci, "command", command)
    if finishes:
        assert ci.stop_allocation(receipt, "h3-test")["status"] == "released"
    else:
        with pytest.raises(RuntimeError, match="terminal state"):
            ci.stop_allocation(receipt, "h3-test")
        assert elapsed[0] <= 120


@pytest.mark.parametrize("comparison_status", ["pass", "fail", "inconclusive"])
def test_smoke_completion_is_separate_from_regression(comparison_status):
    receipt = {"regression_status": "inconclusive", "ci_accepted": False}
    verified = {"comparison": {"overall_status": comparison_status, "checks": [], "slots": [{
        role: {"status": "succeeded", "media": {"valid": True}, "analysis_error": None}
        for role in ("baseline", "candidate")}]}, "runs": {
        role: {"summary": {"scheduled": 1, "valid": 1}} for role in ("baseline", "candidate")}}
    assert ci.smoke_exit(verified, receipt, "smoke") == 0
    assert receipt["ci_accepted"] is False
    assert ci.smoke_exit(verified, receipt, "regression") == 2
    receipt["regression_status"] = "fail"
    assert ci.smoke_exit(verified, receipt, "smoke") == 0
    assert ci.smoke_exit(verified, receipt, "regression") == 1
    verified["runs"]["candidate"]["summary"]["valid"] = 0
    assert ci.smoke_exit(verified, receipt, "smoke") == 1


@pytest.mark.parametrize("failure", ["decode", "analysis", "warmup"])
def test_smoke_rejects_fresh_media_failure_despite_recorded_success(failure):
    observation = {"status": "succeeded", "media": {"valid": True}, "analysis_error": None}
    comparison = {"slots": [{role: dict(observation) for role in ("baseline", "candidate")}], "checks": []}
    verified = {"comparison": comparison, "runs": {
        role: {"summary": {"scheduled": 1, "valid": 1}} for role in ("baseline", "candidate")}}
    if failure == "decode":
        comparison["slots"][0]["candidate"]["media"] = {"valid": False}
    elif failure == "analysis":
        comparison["slots"][0]["candidate"]["analysis_error"] = "decoder failed"
    else:
        comparison["checks"] = [{"name": "candidate.warmup", "status": "inconclusive"}]
    assert ci.smoke_exit(verified, {"regression_status": "inconclusive"}, "smoke") == 1


def test_changed_runtime_blocks_before_scheduler(tmp_path):
    cfg = config(tmp_path)
    Path(cfg["runtime"]["rootfs"]).mkdir()
    Path(cfg["runtime"]["ready_marker"]).touch()
    Path(cfg["runtime"]["entry"]).write_text("changed")
    with pytest.raises(ValueError, match="entry script changed"):
        ci.prepared_spec(cfg)


def test_staging_reuses_identical_source_and_refuses_drift(tmp_path):
    source = tmp_path / "source"
    (source / "evaluator").mkdir(parents=True)
    (source / "ci.py").write_text("entry")
    (source / "evaluator" / "__init__.py").write_text("")
    (source / "runtime-patches").mkdir()
    (source / "runtime-patches" / "timing.patch").write_text("CPU patch fixture")
    dest = tmp_path / "package"
    original = ci.stage_package(source, dest)
    assert ci.stage_package(source, dest) == original
    assert (dest / "runtime-patches" / "timing.patch").read_text() == "CPU patch fixture"
    (dest / "runtime-patches" / "timing.patch").write_text("tampered")
    with pytest.raises(ValueError, match="source differs"):
        ci.stage_package(source, dest)


@pytest.mark.parametrize("reused", [False, True])
def test_failure_collects_original_outputs_and_preserves_holder(tmp_path, monkeypatch, reused):
    cfg = config(tmp_path)
    entry = Path(cfg["runtime"]["entry"])
    entry.write_text("entry")
    Path(cfg["runtime"]["ready_marker"]).write_text("pinned image and preparation identity")
    cfg["runtime"]["entry_sha256"] = ci.digest(entry)
    receipt, record = allocation(tmp_path)
    monkeypatch.setenv("GITHUB_RUN_ID", "456")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setenv("H3_SOURCE_SHA", "a" * 40)
    monkeypatch.setattr(ci, "prepared_spec", lambda cfg: {"test_only": True})
    monkeypatch.setattr(ci, "command", lambda argv, **kw: "a" * 40 if "rev-parse" in argv else "")
    monkeypatch.setattr(ci, "stage_package", lambda source, destination: {})
    decision = {"action": "reuse", "receipt": receipt} if reused else {"action": "allocate"}
    monkeypatch.setattr(ci, "recover", lambda cfg, results: decision)
    monkeypatch.setattr(ci, "allocate", lambda cfg, path: receipt)
    monkeypatch.setattr(ci, "job_record", lambda job: record)
    monkeypatch.setattr(ci, "drain_step", lambda *args: {"status": "ended"})
    canceled = []
    monkeypatch.setattr(ci, "stop_allocation", lambda receipt, task: canceled.append(receipt["identity"]["JobId"]) or {"status": "released"})
    def fail(argv, log, seconds):
        log.write_text("test-only infrastructure failure")
        (log.parent / "partial.mp4").write_bytes(b"retained failed-output bytes; not video evidence")
        raise RuntimeError("server exited before readiness")
    monkeypatch.setattr(ci, "run_step", fail)
    output = tmp_path / "download"
    assert ci.launch(cfg, output) == 2
    assert (output / "partial.mp4").read_bytes().startswith(b"retained")
    state = ci.read(output / "ci.json")
    assert "server exited" in state["error"] and state["ci_accepted"] is False
    assert canceled == ([] if reused else ["123"])
    assert "partial.mp4" in (output / "SHA256SUMS").read_text()
    manifest = ci.read(output / "manifest.json")
    assert manifest["git_commit"] == "a" * 40
    assert manifest["slurm_allocation"]["identity"]["JobId"] == "123"
    assert manifest["evidence"]["ci.json"] == ci.digest(output / "ci.json")
    assert (output / "runtime-readiness.record").read_text() == "pinned image and preparation identity"
    assert manifest["evidence"]["runtime-entry.sh"] == ci.digest(entry)


def test_timeout_stops_only_its_local_process_group(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        ci.run_step([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path / "log", 0.1)


def test_missing_weights_fail_preparation_before_any_slurm_call(tmp_path, monkeypatch):
    from evaluator import mvp_gpu_job
    cfg = config(tmp_path)
    Path(cfg["runtime"]["rootfs"]).mkdir()
    Path(cfg["runtime"]["ready_marker"]).touch()
    Path(cfg["runtime"]["entry"]).write_text("entry")
    cfg["runtime"]["entry_sha256"] = ci.digest(cfg["runtime"]["entry"])
    (tmp_path / "source").mkdir()
    spec = {"baseline": {"source": "/work/source"}, "candidate": {"source": "/work/source"},
            "authorization": {"compute_approved": True, "model_license_reviewed": True, "approval_reference": "test-only"},
            "limits": {"job_seconds": 4500}, "model": {"path": "/work/models", "files": [{"path": "model.safetensors", "size_bytes": 4}]}}
    ci.write(cfg["spec"]["path"], spec)
    cfg["spec"]["sha256"] = ci.digest(cfg["spec"]["path"])
    monkeypatch.setattr(mvp_gpu_job, "validate_gpu_job", lambda spec: spec)
    monkeypatch.setattr(ci, "command", lambda *args, **kwargs: pytest.fail("Preparation must not call Slurm"))
    with pytest.raises(ValueError, match="stage weights before allocating"):
        ci.prepared_spec(cfg)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "model.safetensors").write_bytes(b"test")
    assert ci.prepared_spec(cfg)["model"] == spec["model"]


def test_entry_failure_retains_step_binding_before_runtime(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    receipt, record = allocation(tmp_path)
    Path(cfg["runtime"]["entry"]).write_text("changed runtime entry")
    ci.write(tmp_path / "context.json", {"config": cfg, "allocation": receipt, "node": record["NodeList"]})
    for key, value in {"SLURM_JOB_ID": "123", "SLURM_STEP_ID": "9", "SLURMD_NODENAME": "h200-node"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(ci.os, "sched_getaffinity", lambda pid: {2, 3, 4, 5}, raising=False)
    with pytest.raises(ValueError, match="Entry changed"):
        ci.enter(tmp_path)
    binding = ci.read(tmp_path / "binding.json")
    assert binding["job_id"] == "123" and binding["step_id"] == "9"
    assert binding["cpu_affinity"] == [2, 3, 4, 5]


def test_export_excludes_only_known_caches_and_preserves_media(tmp_path):
    source, target = tmp_path / "source", tmp_path / "export"
    cache = source / "gpu" / "supervisor" / "baseline" / "cache"
    cache.mkdir(parents=True)
    (cache / "kernel-link").symlink_to("/not-a-readable-cache-target")
    media = source / "gpu" / "baseline" / "outputs" / "sample.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"test-only media bytes")
    report = source / "report" / "index.html"
    report.parent.mkdir()
    report.write_text("test-only report")
    ci.collect(source, target)
    assert (target / "gpu/baseline/outputs/sample.mp4").read_bytes() == b"test-only media bytes"
    assert (target / "report/index.html").read_text() == "test-only report"
    assert not (target / "gpu/supervisor/baseline/cache").exists()
    assert (cache / "kernel-link").is_symlink()
    assert "kernel-link" not in (target / "SHA256SUMS").read_text()


@pytest.mark.parametrize('assignment,success', [('4,5', True), ('4-5', True), ('0,1', False)])
def test_entry_resolves_device_minors_instead_of_nvml_indices(tmp_path, assignment, success):
    driver = tmp_path / 'driver'
    ids = ['GPU-12441e19-6453-d8c4-69a8-9fe1cd8b770c', 'GPU-994fd357-abc0-57a9-12d5-f7d40e741530']
    for minor, identity in zip((4, 5), ids):
        info = driver / f'pci-{minor}' / 'information'
        info.parent.mkdir(parents=True)
        info.write_text(f'Model: NVIDIA H200\nDevice Minor: {minor}\nGPU UUID: {identity}\n')
    entry = (Path(ci.__file__).parent / 'runtime-entry.example.sh').read_text()
    program = entry.split("h3_gpu_uuids=$(python3 - <<'PY'\n", 1)[1].split('\nPY\n)', 1)[0]
    program = program.replace('/proc/driver/nvidia/gpus', str(driver))
    result = subprocess.run([sys.executable, '-c', program], env={**os.environ, 'SLURM_STEP_GPUS': assignment},
                            capture_output=True, text=True)
    if success:
        assert result.returncode == 0
        assert result.stdout.strip() == ','.join(ids)
    else:
        assert result.returncode != 0
        assert 'lack NVIDIA UUIDs' in result.stderr


@pytest.mark.parametrize("decision", [{"action": "allocate"}, {"action": "reuse", "receipt": {"identity": {"JobId": "999"}}}])
def test_required_lease_never_replaces_or_borrows_another_allocation(tmp_path, monkeypatch, decision):
    cfg = config(tmp_path)
    entry = Path(cfg["runtime"]["entry"])
    entry.write_text("entry")
    Path(cfg["runtime"]["ready_marker"]).write_text("synthetic readiness")
    cfg["runtime"]["entry_sha256"] = ci.digest(entry)
    monkeypatch.setenv("GITHUB_RUN_ID", "456")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setenv("H3_SOURCE_SHA", "a" * 40)
    monkeypatch.setattr(ci, "prepared_spec", lambda cfg: {})
    monkeypatch.setattr(ci, "command", lambda argv, **kw: "a" * 40 if "rev-parse" in argv else "")
    monkeypatch.setattr(ci, "stage_package", lambda *args: {})
    monkeypatch.setattr(ci, "recover", lambda *args: decision)
    monkeypatch.setattr(ci, "allocate", lambda *args: pytest.fail("must not replace the required allocation"))
    monkeypatch.setattr(ci, "run_step", lambda *args: pytest.fail("must not enter a different allocation"))
    monkeypatch.setattr(ci, "stop_allocation", lambda *args: pytest.fail("outer owner releases its allocation"))
    output = tmp_path / "output"
    assert ci.launch(cfg, output, required_allocation="123") == 2
    assert "must reuse its original allocation" in ci.read(output / "ci.json")["error"]
