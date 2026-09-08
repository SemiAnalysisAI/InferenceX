"""Synthetic contract fixtures only: these tests do not execute a GPU model."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from evaluator import mvp_gpu_report as viewer


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_job(tmp_path, *, evidence="fixture", receipt_kind="no_gpu_measurement", failed=False):
    """Deliberately fake media/identity, explicitly labeled unit-test fixtures."""
    job = tmp_path / "job"
    job.mkdir()
    plan = {
        "plan_id": "synthetic-schema-fixture-not-a-model-run", "model_id": "fixture/no-model", "model_revision": "f" * 40,
        "generation": {"width": 32, "height": 32, "fps": 24, "frame_count": 4, "audio_sample_rate_hz": 32000, "audio_channels": 2},
        "cases": [{"case_id": f"fixture-{index}", "prompt": f"SYNTHETIC TEST ONLY {index}", "seed": index, "requires_motion": True, "requires_sound": True} for index in (1, 2, 3)],
        "repetitions": 1, "warmup_runs": 1,
    }
    spec = {
        "schema_version": "0.1.0", "job_id": "harness-only-schema-fixture",
        "allocation": {"mode": "cooperative_shared", "label": "FAKE TEST DEVICE"},
        "gpu_uuids": ["GPU-SYNTHETIC-TEST-ONLY"], "plan": plan,
        "policy": {"policy_id": "uncalibrated-test-policy", "calibration_status": "uncalibrated"},
        "authorization": {"compute_approved": False, "model_license_reviewed": False, "approval_reference": "synthetic test only"},
        "DO_NOT_EXPORT_SECRET": "pretend-sensitive-extra",
    }
    for label, revision in (("baseline", "a" * 40), ("candidate", "b" * 40)):
        spec[label] = {"source": f"/fixture/{label}", "python": "/fixture/python", "revision": revision, "source_sha256": "a" * 64}
    write_json(job / "spec.json", spec)
    receipt = {
        "schema_version": "0.1.0", "bundle_type": "controlled_gpu_job", "job_id": spec["job_id"],
        "status": "complete", "measurement_status": "complete", "evidence_kind": receipt_kind,
        "regression_status": "inconclusive", "ci_accepted": False, "release_qualified": False,
        "cleanup_status": "clean",
        "started_at": "2026-09-01T01:00:00Z", "finished_at": "2026-09-01T01:05:00Z",
        "spec_sha256": digest(spec), "roles": {}, "failures": [],
    }
    for label in ("baseline", "candidate"):
        directory = job / label
        (directory / "artifacts").mkdir(parents=True)
        config = {"runtime": "fixture-runtime", "runtime_revision": spec[label]["revision"], "model_id": "fixture/no-model", "model_revision": "f" * 40, "hardware_label": "FAKE TEST DEVICE", "identity_verification": "synthetic_fixture"}
        records = []
        slots = [dict(plan["cases"][0], slot_id="warmup-001", phase="warmup", repetition=0)]
        slots += [dict(case, slot_id=f"measurement-r001-c{index:03d}", phase="measurement", repetition=1) for index, case in enumerate(plan["cases"], 1)]
        for index, slot in enumerate(slots):
            relative = f"artifacts/{slot['slot_id']}.mp4"
            data = f"NOT A VIDEO, SYNTHETIC STRUCTURAL TEST {slot['slot_id']}".encode()
            (directory / relative).write_bytes(data)
            records.append({**slot, "status": "succeeded", "attempted": True, "artifact_path": relative,
                            "sha256": hashlib.sha256(data).hexdigest(), "latency_seconds": float(index * 10) if index else 99.0,
                            "submit_to_terminal_seconds": float(index * 10 - 2) if index else 97.0,
                            "submit_to_media_seconds": float(index * 10 - 1) if index else 98.0,
                            "media_validation_seconds": 1.0, "media": {"valid": True, "video": {"width": 32, "height": 32}, "audio": {"present": True, "channels": 2}}, "error": None})
        if failed and label == "candidate":
            records[-1].update(status="failed", attempted=False, artifact_path=None, sha256=None, media=None, latency_seconds=0.0, submit_to_terminal_seconds=None, submit_to_media_seconds=None, media_validation_seconds=None, error="not_started_after_fixture_abort")
        run = {
            "bundle_version": "0.1.0", "bundle_type": "mvp_run", "run_id": f"{label}-SYNTHETIC", "evidence_kind": evidence,
            "plan_id": plan["plan_id"], "plan": plan, "plan_sha256": digest(plan), "configuration": config, "configuration_sha256": digest(config),
            "status": "partial" if failed and label == "candidate" else "complete", "started_at": "2026-09-01T01:00:00Z", "finished_at": "2026-09-01T01:05:00Z",
            "measurement": {"boundary": "submit_to_validated_media", "concurrency": 1, "wall_seconds": 65.0},
            "records": records, "summary": {"scheduled": 9999, "valid": 9999, "latency_median_seconds": 0.000001},
        }
        run_hash = write_json(directory / "run.json", run)
        telemetry_path = f"supervisor/{label}/telemetry.jsonl"
        telemetry_file = job / telemetry_path
        telemetry_file.parent.mkdir(parents=True)
        gpu = {"uuid": "GPU-SYNTHETIC-TEST-ONLY", "name": "SYNTHETIC GPU NOT HARDWARE", "index": 0, "memory_total_mib": 100000, "memory_used_mib": 40000}
        app = {"gpu_uuid": gpu["uuid"], "pid": 123, "memory_used_mib": 40000}
        telemetry_file.write_bytes(b"\n".join(json.dumps({"monotonic_seconds": index, "phase": "startup" if index < 6 else "measurement", "gpus": [gpu], "compute_apps": [app], "owned_compute_apps": [app], "unowned_compute_apps": [], "synthetic_test_only": True}).encode() for index in range(10)) + b"\n")
        receipt["roles"][label] = {
            "status": "complete", "run_path": f"{label}/run.json", "run_sha256": run_hash,
            "source_identity": {**spec[label], "python_sha256": "b" * 64, "sglang_module": f"/fixture/{label}/python/sglang/__init__.py", "python_version": "TEST", "packages": {"torch": "TEST ONLY"}, "env": {"secret": "pretend-sensitive-extra"}},
            "process_identity": {"pid": 123, "pgid": 123, "start_ticks": 999, "session_id": 123, "launch_nonce": "c" * 32, "environment": "pretend-sensitive-extra"},
            "telemetry_path": telemetry_path, "telemetry_sha256": hashlib.sha256(telemetry_file.read_bytes()).hexdigest(),
            "telemetry_summary": {"sample_count": 10, "measurement_sample_count": 4, "gpu_identity": [{"uuid": "GPU-SYNTHETIC-TEST-ONLY", "name": "SYNTHETIC GPU NOT HARDWARE", "index": 0, "memory_total_mib": 100000}], "observed_memory_peak_mib_by_gpu": {"GPU-SYNTHETIC-TEST-ONLY": 40000}, "observed_owned_compute_by_gpu": {"GPU-SYNTHETIC-TEST-ONLY": 4}, "errors": []},
            "cleanup": {"status": "clean", "idle_after": True, "remaining_owned_pids": []},
        }
    write_json(job / "gpu-job.json", receipt)
    return job, spec, receipt


def change_run(job, receipt, label, edit):
    path = job / label / "run.json"
    run = json.loads(path.read_text())
    edit(run)
    receipt["roles"][label]["run_sha256"] = write_json(path, run)
    write_json(job / "gpu-job.json", receipt)


def render(tmp_path, job):
    output = tmp_path / "report"
    report = viewer.write_gpu_report(job, output)
    return report, output, (output / "index.html").read_text()


def test_empty_job_shows_no_invented_runs_or_measurements(tmp_path):
    job = tmp_path / "empty"
    job.mkdir()
    result, output, page = render(tmp_path, job)
    assert result["status"] == "not_started"
    assert result["roles"]["candidate"]["summary"]["scheduled"] is None
    assert result["roles"]["candidate"]["summary"]["latency_median_seconds"] is None
    assert "No verified GPU timing" in page
    assert "deliberately contains no sample video" in page
    assert "<video" not in page
    assert list((output / "assets").iterdir()) == []
    assert not result["ci_accepted"] and not result["release_qualified"]


def test_missing_roles_retain_frozen_denominators(tmp_path):
    job, spec, receipt = fixture_job(tmp_path)
    for label in ("baseline", "candidate"):
        (job / label / "run.json").unlink()
    result, _, page = render(tmp_path, job)
    assert result["status"] == "incomplete"
    assert result["roles"]["candidate"]["summary"]["scheduled"] == 3
    assert result["roles"]["candidate"]["summary"]["not_recorded"] == 3
    assert "No generated artifact available" in page


@pytest.mark.parametrize("evidence", ["fixture", "imported_media"])
def test_fixture_and_imported_media_cannot_masquerade_as_gpu_timing(tmp_path, evidence):
    job, _, _ = fixture_job(tmp_path, evidence=evidence, receipt_kind="controlled_h3_gpu")
    result, output, page = render(tmp_path, job)
    role = result["roles"]["candidate"]
    assert role["summary"]["latency_median_seconds"] is None
    assert role["summary"]["valid_clips_per_second"] is None
    assert "Harness-only / imported evidence" in page
    assert role["summary"]["scheduled"] == role["summary"]["valid"] == 3
    assert page.count("<video controls") == 6
    assert len(list((output / "assets").iterdir())) == 4  # duplicate contents deduplicated


@pytest.mark.parametrize("evidence", ["operator_endpoint", "live_h3"])
def test_bound_controlled_schema_recomputes_metrics_and_preserves_missing_slots(tmp_path, evidence):
    job, _, _ = fixture_job(tmp_path, evidence=evidence, receipt_kind="controlled_h3_gpu", failed=True)
    result, _, page = render(tmp_path, job)
    summary = result["roles"]["candidate"]["summary"]
    assert summary["scheduled"] == 3 and summary["valid"] == 2
    assert summary["latency_median_seconds"] == 15.0
    assert summary["latency_count"] == 2
    assert summary["not_started"] == 1
    assert summary["valid_clips_per_second"] == pytest.approx(2 / 65)
    assert result["roles"]["candidate"]["observations"][-1]["latency_seconds"] is None
    assert result["roles"]["candidate"]["observations"][0]["submit_to_terminal_seconds"] == 8.0
    assert result["same_gpu_uuid_set"] is True
    assert "99.000 s" not in page  # warmup timing is not a measurement
    assert "NOT exact peak" in page
    assert "not GPU kernel latency" in page


def test_html_is_script_free_escaped_portable_and_sanitized(tmp_path):
    job, spec, receipt = fixture_job(tmp_path)
    attack = '</script><img src=x onerror="alert(1)">'
    spec["job_id"] = attack
    write_json(job / "spec.json", spec)
    receipt["spec_sha256"] = digest(spec)
    receipt["failures"] = [attack]
    receipt["roles"]["candidate"]["source_identity"]["revision"] = attack
    write_json(job / "gpu-job.json", receipt)
    result, output, page = render(tmp_path, job)
    assert "<script" not in page and "<img" not in page
    assert "&lt;img" in page
    assert "Content-Security-Policy" in page
    assert 'src="assets/' in page
    assert 'src="http' not in page
    payload = (output / "evidence.json").read_text()
    assert "pretend-sensitive-extra" not in page + payload
    assert "DO_NOT_EXPORT_SECRET" not in payload
    assert result["report"]["scripts"] is False


def test_run_hash_mismatch_cannot_supply_values_or_media(tmp_path):
    job, _, _ = fixture_job(tmp_path, evidence="operator_endpoint", receipt_kind="controlled_h3_gpu")
    path = job / "candidate" / "run.json"
    path.write_text(path.read_text() + " ")
    result, _, page = render(tmp_path, job)
    assert result["roles"]["candidate"]["summary"]["valid"] == 0
    assert result["roles"]["candidate"]["summary"]["not_recorded"] == 3
    assert "Document SHA256 does not match its receipt" in page


def test_spec_hash_mismatch_disables_gpu_timing(tmp_path):
    job, spec, _ = fixture_job(tmp_path, evidence="operator_endpoint", receipt_kind="controlled_h3_gpu")
    spec["job_id"] = "tampered"
    write_json(job / "spec.json", spec)
    result, _, _ = render(tmp_path, job)
    assert not result["roles"]["candidate"]["gpu_timing_presented"]


@pytest.mark.parametrize("raw", ['{"a":1,"a":2}', '{"value":NaN}', '{"value":1e999}', '[1,2]', '{broken'])
def test_malformed_receipt_creates_error_page_not_success(tmp_path, raw):
    job = tmp_path / "malformed"
    job.mkdir()
    (job / "gpu-job.json").write_text(raw)
    result, _, page = render(tmp_path, job)
    assert result["status"] == "invalid"
    assert result["issues"]
    assert "No verified GPU timing" in page


@pytest.mark.parametrize("unsafe", ["../outside.mp4", "/outside.mp4", "artifacts/../outside.mp4", "artifacts\\outside.mp4"])
def test_media_path_traversal_is_never_read(tmp_path, unsafe):
    job, _, receipt = fixture_job(tmp_path)
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"SENSITIVE OUTSIDE DATA")
    def edit(run):
        run["records"][1].update(artifact_path=unsafe, sha256=hashlib.sha256(outside.read_bytes()).hexdigest())
    change_run(job, receipt, "candidate", edit)
    result, output, page = render(tmp_path, job)
    assert result["roles"]["candidate"]["observations"][0]["artifact_path"] is None
    assert not any(path.read_bytes() == b"SENSITIVE OUTSIDE DATA" for path in (output / "assets").iterdir())


@pytest.mark.parametrize("directory", [False, True])
def test_media_symlinks_are_rejected_even_when_target_inside_job(tmp_path, directory):
    job, _, receipt = fixture_job(tmp_path)
    if directory:
        (job / "candidate" / "link").symlink_to(job / "candidate" / "artifacts", target_is_directory=True)
        change_run(job, receipt, "candidate", lambda run: run["records"][1].update(artifact_path="link/measurement-r001-c001.mp4"))
    else:
        target = job / "candidate" / "artifacts" / "measurement-r001-c001.mp4"
        data = target.read_bytes()
        original = job / "candidate" / "original.mp4"
        original.write_bytes(data)
        target.unlink()
        target.symlink_to(original)
    result, _, _ = render(tmp_path, job)
    assert result["roles"]["candidate"]["observations"][0]["artifact_path"] is None


def test_receipt_symlink_is_rejected_without_disclosing_target(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    secret = tmp_path / "secret.json"
    secret.write_text('{"secret":"OUTSIDE_PRIVATE"}')
    (job / "gpu-job.json").symlink_to(secret)
    result, output, page = render(tmp_path, job)
    assert result["status"] == "invalid"
    assert "OUTSIDE_PRIVATE" not in page + (output / "evidence.json").read_text()


def test_existing_output_is_never_overwritten(tmp_path):
    job = tmp_path / "job"
    output = tmp_path / "report"
    job.mkdir()
    output.mkdir()
    (output / "index.html").write_text("USER CONTENT")
    with pytest.raises(FileExistsError):
        viewer.write_gpu_report(job, output)
    assert (output / "index.html").read_text() == "USER CONTENT"


@pytest.mark.parametrize("root_link", [False, True])
def test_symlink_root_or_output_rejected(tmp_path, root_link):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        viewer.write_gpu_report(alias if root_link else real, tmp_path / "report" if root_link else alias / "report")


def test_json_size_is_bounded(tmp_path, monkeypatch):
    job = tmp_path / "job"
    job.mkdir()
    (job / "gpu-job.json").write_text('{"padding":"' + 'a' * 200 + '"}')
    monkeypatch.setattr(viewer, "MAX_JSON_BYTES", 64)
    result, _, page = render(tmp_path, job)
    assert result["status"] == "invalid"
    assert "byte limit" in page


def test_media_size_and_total_budget_are_bounded(tmp_path, monkeypatch):
    job, _, _ = fixture_job(tmp_path)
    monkeypatch.setattr(viewer, "MAX_TOTAL_MEDIA_BYTES", 1)
    result, output, page = render(tmp_path, job)
    assert not list((output / "assets").iterdir())
    assert result["roles"]["candidate"]["summary"]["valid"] == 0
    assert "Total media exceeds" in page


def test_bad_media_hash_leaves_no_unverified_asset(tmp_path):
    job, _, receipt = fixture_job(tmp_path)
    change_run(job, receipt, "candidate", lambda run: run["records"][1].update(sha256="0" * 64))
    result, output, _ = render(tmp_path, job)
    assert not (output / "assets" / ("0" * 64 + ".mp4")).exists()
    assert result["roles"]["candidate"]["observations"][0]["valid"] is False


def test_nonregular_json_file_does_not_block(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    (job / "gpu-job.json").mkdir()
    result, _, page = render(tmp_path, job)
    assert result["status"] == "invalid"
    assert "No verified GPU timing" in page


def test_stale_comparison_is_not_presented(tmp_path):
    job, _, receipt = fixture_job(tmp_path)
    comparison = {"bundle_type": "mvp_comparison", "baseline": {"run_bundle_sha256": "a" * 64}, "candidate": {"run_bundle_sha256": "a" * 64}, "overall_status": "pass"}
    receipt.update(comparison_path="comparison.json", comparison_sha256=write_json(job / "comparison.json", comparison))
    write_json(job / "gpu-job.json", receipt)
    result, _, page = render(tmp_path, job)
    assert result["comparison_status"] == "inconclusive"
    assert "not bound to both observed run bundles" in page


def test_bound_comparison_checks_are_escaped_and_keep_units(tmp_path):
    job, _, receipt = fixture_job(tmp_path)
    comparison = {"bundle_type": "mvp_comparison", "baseline": {"run_bundle_sha256": receipt["roles"]["baseline"]["run_sha256"]}, "candidate": {"run_bundle_sha256": receipt["roles"]["candidate"]["run_sha256"]}, "overall_status": "fail",
                  "checks": [{"name": "<unsafe>", "status": "fail", "observed": 0.2, "threshold": 0.1, "unit": "fraction", "reason": "synthetic fixture"}]}
    receipt.update(comparison_path="comparison.json", comparison_sha256=write_json(job / "comparison.json", comparison))
    write_json(job / "gpu-job.json", receipt)
    result, _, page = render(tmp_path, job)
    assert result["comparison_status"] == "fail"
    assert "&lt;unsafe&gt;" in page and "<unsafe>" not in page
    assert "fraction" in page
    assert result["ci_status"] == "inconclusive"


def test_telemetry_tamper_is_explicit(tmp_path):
    job, _, receipt = fixture_job(tmp_path)
    (job / receipt["roles"]["candidate"]["telemetry_path"]).write_text("tampered")
    result, _, page = render(tmp_path, job)
    assert not result["roles"]["candidate"]["telemetry"]["file_sha256_verified"]
    assert "Telemetry SHA256 does not match" in page


def test_unverified_different_gpu_uuid_is_not_same_device_regression(tmp_path):
    job, _, receipt = fixture_job(tmp_path)
    receipt["roles"]["candidate"]["telemetry_summary"]["gpu_identity"][0]["uuid"] = "GPU-OTHER-SYNTHETIC"
    write_json(job / "gpu-job.json", receipt)
    result, _, page = render(tmp_path, job)
    assert result["same_gpu_uuid_set"] is None
    assert "GPU identity missing" in page
    assert "Cross-GPU results are descriptive" in page


def test_display_limit_preserves_full_denominators_and_export(tmp_path, monkeypatch):
    job, _, _ = fixture_job(tmp_path)
    monkeypatch.setattr(viewer, "MAX_DISPLAY_SLOTS", 1)
    result, _, page = render(tmp_path, job)
    assert result["roles"]["candidate"]["summary"]["scheduled"] == 3
    assert len(result["roles"]["candidate"]["observations"]) == 3
    assert "first 1 of 3 scheduled slots" in page


def test_not_bound_run_withholds_gpu_metrics(tmp_path):
    job, _, receipt = fixture_job(tmp_path, evidence="operator_endpoint", receipt_kind="controlled_h3_gpu")
    del receipt["roles"]["candidate"]["run_sha256"]
    write_json(job / "gpu-job.json", receipt)
    result, _, page = render(tmp_path, job)
    assert not result["roles"]["candidate"]["gpu_timing_presented"]
    assert "GPU timings withheld" in page


def test_inconsistent_configuration_hash_withholds_gpu_metrics(tmp_path):
    job, _, receipt = fixture_job(tmp_path, evidence="operator_endpoint", receipt_kind="controlled_h3_gpu")
    change_run(job, receipt, "candidate", lambda run: run["configuration"].update(configuration_sha256="0" * 64))
    result, _, page = render(tmp_path, job)
    assert not result["roles"]["candidate"]["gpu_timing_presented"]
    assert "configuration hash is absent or mismatched" in page


def test_unknown_slots_cannot_qualify_timing_population(tmp_path):
    job, _, receipt = fixture_job(tmp_path, evidence="operator_endpoint", receipt_kind="controlled_h3_gpu")
    change_run(job, receipt, "candidate", lambda run: run["records"].append(copy.deepcopy(run["records"][1])))
    result, _, page = render(tmp_path, job)
    assert not result["roles"]["candidate"]["gpu_timing_presented"]
    assert "duplicate" in page


def test_impossible_serial_wall_time_is_not_a_throughput_result(tmp_path):
    job, _, receipt = fixture_job(tmp_path, evidence="operator_endpoint", receipt_kind="controlled_h3_gpu")
    change_run(job, receipt, "candidate", lambda run: run["measurement"].update(wall_seconds=1.0))
    result, _, page = render(tmp_path, job)
    assert result["roles"]["candidate"]["summary"]["valid_clips_per_second"] is None
    assert "throughput withheld" in page


def test_real_cpu_encoded_fixture_has_playable_media_but_no_gpu_timing(tmp_path):
    """Also produces an explicitly CPU-fixture-labeled artifact for visual QA."""
    pytest.importorskip("av")
    pytest.importorskip("numpy")
    from test_mvp_fixtures import FIXTURE_POLICY, _fixture_plan, _fixture_run
    from evaluator.mvp_compare import compare_runs

    job = tmp_path / "cpu-fixture-job"
    job.mkdir()
    plan = _fixture_plan()
    _fixture_run(job / "baseline", "baseline", plan)
    _fixture_run(job / "candidate", "candidate", plan, baseline_dir=job / "baseline", defect="muted")
    spec = {"schema_version": "0.1.0", "job_id": "CPU FIXTURE VISUAL QA — NOT AN H3 RUN", "plan": plan,
            "allocation": {"mode": "cooperative_shared", "label": "CPU-only encoded moving shapes and tones"}, "policy": FIXTURE_POLICY}
    write_json(job / "spec.json", spec)
    receipt = {"schema_version": "0.1.0", "bundle_type": "controlled_gpu_job", "status": "complete", "measurement_status": "complete",
               "evidence_kind": "no_gpu_measurement", "spec_sha256": digest(spec), "roles": {}, "regression_status": "inconclusive", "ci_accepted": False,
               "acceptance_reasons": ["CPU fixture only. No GPU exists in this test and no model inference ran."]}
    for label in ("baseline", "candidate"):
        receipt["roles"][label] = {"status": "complete", "run_path": f"{label}/run.json", "run_sha256": hashlib.sha256((job / label / "run.json").read_bytes()).hexdigest(), "cleanup": {"status": "not_applicable", "reason": "No GPU resources used by CPU fixture"}}
    comparison = compare_runs(job / "baseline", job / "candidate", policy=FIXTURE_POLICY)
    receipt.update(comparison_path="comparison.json", comparison_sha256=write_json(job / "comparison.json", comparison))
    write_json(job / "gpu-job.json", receipt)
    result, output, page = render(tmp_path, job)
    assert result["comparison_status"] == "fail"
    assert result["roles"]["baseline"]["summary"]["valid"] == 2
    assert result["roles"]["candidate"]["summary"]["valid"] == 1
    assert result["roles"]["candidate"]["summary"]["latency_median_seconds"] is None
    assert result["slot_comparisons"]
    assert "CPU FIXTURE VISUAL QA" in page and "Harness-only / imported evidence" in page
    assert "Recorded paired fidelity metrics" in page
    assert "Recorded supervisor acceptance" in page
    assert all(path.stat().st_size > 1000 for path in (output / "assets").iterdir())


@pytest.mark.parametrize("field,value", [
    ("revision", "0" * 40), ("source_sha256", "0" * 64),
    ("python", "/not/the/pinned/python"), ("sglang_module", "/wrong/sglang/__init__.py"),
])
def test_gpu_label_requires_matching_observed_runtime_pins(tmp_path, field, value):
    job, _, receipt = fixture_job(tmp_path, evidence="operator_endpoint", receipt_kind="controlled_h3_gpu")
    receipt["roles"]["candidate"]["source_identity"][field] = value
    write_json(job / "gpu-job.json", receipt)
    result, _, page = render(tmp_path, job)
    assert result["roles"]["candidate"]["gpu_timing_presented"] is False
    assert "Observed source/Python identity does not match" in page


@pytest.mark.parametrize("field,value", [("pid", 1), ("pgid", 12), ("session_id", 12), ("start_ticks", 0), ("launch_nonce", "not-a-launch-nonce")])
def test_gpu_label_requires_plausible_owned_process_receipt(tmp_path, field, value):
    job, _, receipt = fixture_job(tmp_path, evidence="operator_endpoint", receipt_kind="controlled_h3_gpu")
    receipt["roles"]["candidate"]["process_identity"][field] = value
    write_json(job / "gpu-job.json", receipt)
    result, _, page = render(tmp_path, job)
    assert result["roles"]["candidate"]["gpu_timing_presented"] is False
    assert "plausible owned process/session" in page


def test_telemetry_identity_annotations_preserve_timing_without_clearing_cleanup_failure(tmp_path):
    job, _, receipt = fixture_job(tmp_path, evidence="operator_endpoint", receipt_kind="controlled_h3_gpu")
    for role in receipt["roles"].values():
        telemetry = job / role["telemetry_path"]
        samples = [json.loads(line) for line in telemetry.read_text().splitlines()]
        for sample in samples:
            for app in sample["owned_compute_apps"]:
                app["process_identity"] = {"pid": app["pid"], "pgid": 123, "session_id": 123, "start_ticks": 999}
        telemetry.write_text("\n".join(json.dumps(sample) for sample in samples) + "\n")
        role["telemetry_sha256"] = hashlib.sha256(telemetry.read_bytes()).hexdigest()
    receipt.update(status="failed", measurement_status="incomplete", cleanup_status="failed",
                   failures=["owned runtime cleanup did not establish idle GPUs"])
    receipt["roles"]["candidate"]["cleanup"].update(status="failed", idle_after=False)
    write_json(job / "gpu-job.json", receipt)
    result, _, _ = render(tmp_path, job)
    assert result["issues"] == []
    assert all(role["telemetry"]["samples_consistent"] and role["gpu_timing_presented"] for role in result["roles"].values())
    assert result["status"] == "failed" and result["ci_accepted"] is False
    assert result["failures"] == receipt["failures"]
    assert result["roles"]["candidate"]["cleanup"]["status"] == "failed"


@pytest.mark.parametrize("defect", ["missing", "overlap", "duplicate", "wrong_pid", "memory"])
def test_telemetry_ownership_partition_rejects_mismatched_observations(tmp_path, defect):
    job, _, receipt = fixture_job(tmp_path, evidence="operator_endpoint", receipt_kind="controlled_h3_gpu")
    role = receipt["roles"]["candidate"]
    telemetry = job / role["telemetry_path"]
    samples = [json.loads(line) for line in telemetry.read_text().splitlines()]
    sample = samples[0]
    app = sample["owned_compute_apps"][0]
    if defect == "missing":
        sample["owned_compute_apps"] = []
    elif defect == "overlap":
        sample["unowned_compute_apps"] = [dict(app, ownership_observation="not_owned")]
    elif defect == "duplicate":
        sample["compute_apps"].append(dict(app, pid=124))
        sample["owned_compute_apps"].append(dict(app))
    elif defect == "wrong_pid":
        app["pid"] = 124
    else:
        app["memory_used_mib"] += 1
    telemetry.write_text("\n".join(json.dumps(sample) for sample in samples) + "\n")
    role["telemetry_sha256"] = hashlib.sha256(telemetry.read_bytes()).hexdigest()
    write_json(job / "gpu-job.json", receipt)
    result, _, page = render(tmp_path, job)
    assert result["roles"]["candidate"]["telemetry"]["samples_consistent"] is False
    assert result["roles"]["candidate"]["gpu_timing_presented"] is False
    assert "Telemetry compute ownership partition is inconsistent" in page


def test_gpu_label_requires_owned_compute_during_measurement(tmp_path):
    job, _, receipt = fixture_job(tmp_path, evidence="operator_endpoint", receipt_kind="controlled_h3_gpu")
    role = receipt["roles"]["candidate"]
    telemetry = job / role["telemetry_path"]
    samples = [json.loads(line) for line in telemetry.read_text().splitlines()]
    for sample in samples:
        sample["owned_compute_apps"] = []
        sample["compute_apps"] = []
    telemetry.write_text("\n".join(json.dumps(sample) for sample in samples) + "\n")
    role["telemetry_sha256"] = hashlib.sha256(telemetry.read_bytes()).hexdigest()
    role["telemetry_summary"]["observed_owned_compute_by_gpu"] = {"GPU-SYNTHETIC-TEST-ONLY": 0}
    write_json(job / "gpu-job.json", receipt)
    result, _, page = render(tmp_path, job)
    assert result["roles"]["candidate"]["telemetry"]["samples_consistent"] is True
    assert result["roles"]["candidate"]["gpu_timing_presented"] is False
    assert "do not establish owned compute" in page


def test_gpu_label_requires_summary_to_match_raw_telemetry(tmp_path):
    job, _, receipt = fixture_job(tmp_path, evidence="operator_endpoint", receipt_kind="controlled_h3_gpu")
    receipt["roles"]["candidate"]["telemetry_summary"]["observed_memory_peak_mib_by_gpu"] = {"GPU-SYNTHETIC-TEST-ONLY": 1}
    write_json(job / "gpu-job.json", receipt)
    result, _, page = render(tmp_path, job)
    assert result["roles"]["candidate"]["gpu_timing_presented"] is False
    assert "does not match its hash-verified samples" in page


@pytest.mark.parametrize("edit", [
    lambda row: row.update(submit_to_terminal_seconds=20.0),
    lambda row: row.update(submit_to_media_seconds=-1.0),
    lambda row: row.update(media_validation_seconds=5.0),
    lambda row: row.update(submit_to_terminal_seconds=None),
])
def test_invalid_nested_timings_are_never_presented(tmp_path, edit):
    job, _, receipt = fixture_job(tmp_path, evidence="operator_endpoint", receipt_kind="controlled_h3_gpu")
    change_run(job, receipt, "candidate", lambda run: edit(run["records"][1]))
    result, _, page = render(tmp_path, job)
    assert result["roles"]["candidate"]["gpu_timing_presented"] is False
    assert result["roles"]["candidate"]["summary"]["latency_median_seconds"] is None
    assert "Invalid nested client timing boundaries" in page


def test_recorded_failed_ci_verdict_is_not_overwritten_as_inconclusive(tmp_path):
    job, _, receipt = fixture_job(tmp_path)
    receipt["regression_status"] = "fail"
    write_json(job / "gpu-job.json", receipt)
    result, _, page = render(tmp_path, job)
    assert result["ci_status"] == "fail"
    assert 'Recorded CI verdict: <span class="badge bad">fail</span>' in page
    assert not result["release_qualified"]


def test_recorded_ci_pass_cannot_be_green_with_unqualified_evidence(tmp_path):
    job, _, receipt = fixture_job(tmp_path)
    receipt.update(regression_status="pass", ci_accepted=True)
    write_json(job / "gpu-job.json", receipt)
    result, _, page = render(tmp_path, job)
    assert result["ci_status"] == "inconclusive"
    assert result["ci_accepted"] is False
    assert 'Recorded CI verdict: <span class="badge good">pass</span>' not in page
    assert "recorded pass is not shown as passing" in page


def test_nonfinalized_run_never_presents_gpu_timing(tmp_path):
    job, _, receipt = fixture_job(tmp_path, evidence="operator_endpoint", receipt_kind="controlled_h3_gpu")
    change_run(job, receipt, "candidate", lambda run: run.update(finished_at=None))
    result, _, page = render(tmp_path, job)
    assert result["roles"]["candidate"]["gpu_timing_presented"] is False
    assert "not finalized" in page


def qualified_fixture_job(tmp_path):
    """Structurally qualified synthetic receipt; never actual GPU/CI evidence."""
    job, spec, receipt = fixture_job(tmp_path, evidence="operator_endpoint", receipt_kind="controlled_h3_gpu")
    spec["policy"]["calibration_status"] = "operator_calibrated"
    spec["allocation"]["mode"] = "dedicated_ci"
    write_json(job / "spec.json", spec)
    receipt["spec_sha256"] = digest(spec)
    receipt.update(regression_status="pass", ci_accepted=True)
    comparison = {"bundle_type": "mvp_comparison", "overall_status": "pass", "policy": spec["policy"],
                  "baseline": {"run_bundle_sha256": receipt["roles"]["baseline"]["run_sha256"]},
                  "candidate": {"run_bundle_sha256": receipt["roles"]["candidate"]["run_sha256"]}}
    receipt.update(comparison_path="comparison.json", comparison_sha256=write_json(job / "comparison.json", comparison))
    write_json(job / "gpu-job.json", receipt)
    return job, receipt


@pytest.mark.parametrize("changes", [
    {"status": "failed"}, {"status": "running"}, {"status": "aborted"},
    {"failures": ["Synthetic infrastructure failure"]}, {"failures": None},
    {"cleanup_status": "failed"}, {"cleanup_status": None},
    {"finished_at": None}, {"started_at": None},
    {"finished_at": "not a timestamp"},
    {"finished_at": "2026-09-01T01:05:00"},
    {"finished_at": "2026-09-01T00:59:59Z"},
])
def test_otherwise_qualified_ci_pass_requires_finalized_clean_success(tmp_path, changes):
    job, receipt = qualified_fixture_job(tmp_path)
    receipt.update(changes)
    write_json(job / "gpu-job.json", receipt)
    result, _, page = render(tmp_path, job)
    assert result["ci_status"] == "inconclusive"
    assert result["ci_accepted"] is False
    assert result["release_qualified"] is False
    assert 'Recorded CI verdict: <span class="badge good">pass</span>' not in page
    assert "recorded pass is not shown as passing" in page


def test_recorded_ci_pass_requires_finalization_fields_to_be_present(tmp_path):
    job, receipt = qualified_fixture_job(tmp_path)
    del receipt["finished_at"]
    write_json(job / "gpu-job.json", receipt)
    result, _, _ = render(tmp_path, job)
    assert result["ci_status"] == "inconclusive" and result["ci_accepted"] is False


@pytest.mark.parametrize("condition", ["cooperative_shared", "uncalibrated"])
def test_otherwise_qualified_receipt_cannot_promote_shared_or_uncalibrated_job(tmp_path, condition):
    job, receipt = qualified_fixture_job(tmp_path)
    spec = json.loads((job / "spec.json").read_text())
    if condition == "cooperative_shared":
        spec["allocation"]["mode"] = condition
    else:
        spec["policy"]["calibration_status"] = condition
        comparison = json.loads((job / "comparison.json").read_text())
        comparison["policy"] = spec["policy"]
        receipt["comparison_sha256"] = write_json(job / "comparison.json", comparison)
    write_json(job / "spec.json", spec)
    receipt["spec_sha256"] = digest(spec)
    write_json(job / "gpu-job.json", receipt)
    result, _, page = render(tmp_path, job)
    assert result["ci_status"] == "inconclusive" and result["ci_accepted"] is False
    assert 'Recorded CI verdict: <span class="badge good">pass</span>' not in page


def test_recorded_accepted_ci_is_labeled_reported_not_independently_qualified(tmp_path):
    job, _ = qualified_fixture_job(tmp_path)
    result, _, page = render(tmp_path, job)
    assert result["ci_status"] == "pass" and result["ci_accepted"] is True
    assert result["release_qualified"] is False
    assert 'Recorded CI verdict: <span class="badge good">pass</span>' in page
    assert "does not independently qualify CI" in page
