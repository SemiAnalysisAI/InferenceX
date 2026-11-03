"""Synthetic contract fixtures only: these tests do not execute a GPU model."""

import copy
import hashlib
import json

import pytest

from evaluator import gpu_report as viewer


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_job(tmp_path, *, evidence="fixture", receipt_kind="no_gpu_measurement", failed=False, warmup_case=None):
    """Deliberately fake media/identity, explicitly labeled unit-test fixtures."""
    job = tmp_path / "job"
    job.mkdir()
    plan = {
        "plan_id": "synthetic-schema-fixture-not-a-model-run", "model_id": "fixture/no-model", "model_revision": "f" * 40,
        "generation": {"width": 32, "height": 32, "fps": 24, "frame_count": 4, "audio_sample_rate_hz": 32000, "audio_channels": 2},
        "cases": [{"case_id": f"fixture-{index}", "prompt": f"SYNTHETIC TEST ONLY {index}", "seed": index, "requires_motion": True, "requires_sound": True} for index in (1, 2, 3)],
        "repetitions": 1, "warmup_runs": 1,
    }
    if warmup_case is not None:
        plan["warmup_case"] = warmup_case
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
        slots = [dict(warmup_case or plan["cases"][0], slot_id="warmup-001", phase="warmup", repetition=0)]
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


@pytest.mark.parametrize("evidence", [
    "imported_media",
], ids=['imported_media'])
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


def test_warmup_slot_uses_the_frozen_warmup_case(tmp_path):
    warmup = {"case_id": "fixture-warmup", "prompt": "SYNTHETIC WARMUP ONLY", "seed": 0, "requires_motion": True, "requires_sound": True}
    job, _, _ = fixture_job(tmp_path, warmup_case=warmup)
    result, _, _ = render(tmp_path, job)
    for role in result["roles"].values():
        assert [row["case_id"] for row in role["warmups"]] == ["fixture-warmup"]
        assert role["summary"]["warmups_valid"] == 1
    assert not any("frozen slot" in issue["message"] for issue in result["issues"])


def test_malformed_warmup_case_withholds_the_schedule(tmp_path):
    job, _, _ = fixture_job(tmp_path, warmup_case={"case_id": "fixture-warmup"})
    result, _, _ = render(tmp_path, job)
    assert result["roles"]["candidate"]["summary"]["scheduled"] is None
    assert any("Frozen cases require" in issue["message"] for issue in result["issues"])


@pytest.mark.parametrize("unsafe", [
    "/outside.mp4",
    "artifacts/../outside.mp4",
], ids=['/outside.mp4', 'artifacts/../outside.mp4'])
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


@pytest.mark.parametrize("directory", [
    True,
], ids=['True'])
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


def test_bad_media_hash_leaves_no_unverified_asset(tmp_path):
    job, _, receipt = fixture_job(tmp_path)
    change_run(job, receipt, "candidate", lambda run: run["records"][1].update(sha256="0" * 64))
    result, output, _ = render(tmp_path, job)
    assert not (output / "assets" / ("0" * 64 + ".mp4")).exists()
    assert result["roles"]["candidate"]["observations"][0]["valid"] is False


def test_stale_comparison_is_not_presented(tmp_path):
    job, _, receipt = fixture_job(tmp_path)
    comparison = {"bundle_type": "mvp_comparison", "baseline": {"run_bundle_sha256": "a" * 64}, "candidate": {"run_bundle_sha256": "a" * 64}, "overall_status": "pass"}
    receipt.update(comparison_path="comparison.json", comparison_sha256=write_json(job / "comparison.json", comparison))
    write_json(job / "gpu-job.json", receipt)
    result, _, page = render(tmp_path, job)
    assert result["comparison_status"] == "inconclusive"
    assert "not bound to both observed run bundles" in page


def test_telemetry_tamper_is_explicit(tmp_path):
    job, _, receipt = fixture_job(tmp_path)
    (job / receipt["roles"]["candidate"]["telemetry_path"]).write_text("tampered")
    result, _, page = render(tmp_path, job)
    assert not result["roles"]["candidate"]["telemetry"]["file_sha256_verified"]
    assert "Telemetry SHA256 does not match" in page


def test_unknown_slots_cannot_qualify_timing_population(tmp_path):
    job, _, receipt = fixture_job(tmp_path, evidence="operator_endpoint", receipt_kind="controlled_h3_gpu")
    change_run(job, receipt, "candidate", lambda run: run["records"].append(copy.deepcopy(run["records"][1])))
    result, _, page = render(tmp_path, job)
    assert not result["roles"]["candidate"]["gpu_timing_presented"]
    assert "duplicate" in page


@pytest.mark.parametrize("field,value", [
    ("source_sha256", "0" * 64),
], ids=['source_sha256-0000000000000000000000000000000000000000000000000000000000000000'])
def test_gpu_label_requires_matching_observed_runtime_pins(tmp_path, field, value):
    job, _, receipt = fixture_job(tmp_path, evidence="operator_endpoint", receipt_kind="controlled_h3_gpu")
    receipt["roles"]["candidate"]["source_identity"][field] = value
    write_json(job / "gpu-job.json", receipt)
    result, _, page = render(tmp_path, job)
    assert result["roles"]["candidate"]["gpu_timing_presented"] is False
    assert "Observed source/Python identity does not match" in page


@pytest.mark.parametrize("field,value", [
    ("pgid", 12),
], ids=['pgid-12'])
def test_gpu_label_requires_plausible_owned_process_receipt(tmp_path, field, value):
    job, _, receipt = fixture_job(tmp_path, evidence="operator_endpoint", receipt_kind="controlled_h3_gpu")
    receipt["roles"]["candidate"]["process_identity"][field] = value
    write_json(job / "gpu-job.json", receipt)
    result, _, page = render(tmp_path, job)
    assert result["roles"]["candidate"]["gpu_timing_presented"] is False
    assert "plausible owned process/session" in page


@pytest.mark.parametrize("defect", [
    "overlap",
    "wrong_pid",
], ids=['overlap', 'wrong_pid'])
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


@pytest.mark.parametrize("field", [
    "gpu_identity",
], ids=['gpu_identity'])
def test_gpu_label_requires_summary_to_match_raw_telemetry(tmp_path, field):
    job, _, receipt = fixture_job(tmp_path, evidence="operator_endpoint", receipt_kind="controlled_h3_gpu")
    summary = receipt["roles"]["candidate"]["telemetry_summary"]
    if field == "gpu_identity":
        summary[field][0]["uuid"] = "GPU-OTHER-SYNTHETIC"
    else:
        summary[field] = {"GPU-SYNTHETIC-TEST-ONLY": 1}
    write_json(job / "gpu-job.json", receipt)
    result, _, page = render(tmp_path, job)
    assert result["roles"]["candidate"]["gpu_timing_presented"] is False
    assert "does not match its hash-verified samples" in page
    if field == "gpu_identity":
        assert result["same_gpu_uuid_set"] is None
        assert "GPU identity missing" in page
        assert "Cross-GPU results are descriptive" in page


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
    {"cleanup_status": "failed"},
    {"finished_at": None},
], ids=['changes5', 'changes7'])
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


@pytest.mark.parametrize("condition", [
    "cooperative_shared",
], ids=['cooperative_shared'])
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
