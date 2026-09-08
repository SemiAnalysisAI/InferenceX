import copy
import hashlib
import json
from pathlib import Path

import pytest

from evaluator import mvp_compare
from evaluator.mvp_compare import compare_runs
from evaluator.mvp_report import write_report


POLICY = {
    "policy_id": "explicit-test-policy",
    "calibration_status": "fixture_control",
    "max_latency_increase_fraction": 0.1,
    "min_video_psnr_db": 40.0,
    "min_audio_spectral_cosine": 0.99,
    "max_audio_rms_ratio_error": 0.05,
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_bundle(directory, document):
    (directory / "run.json").write_text(json.dumps(document), encoding="utf-8")


def bundle(tmp_path, name, *, warmups=1):
    directory = tmp_path / name
    directory.mkdir()
    (directory / "media").mkdir()
    plan = {
        "plan_id": "test-plan",
        "model_id": "fixture/no-model",
        "model_revision": "fixed-model-revision",
        "generation": {
            "width": 320, "height": 180, "fps": 24, "frame_count": 48,
            "duration_seconds": 2, "audio_sample_rate_hz": 32000, "audio_channels": 2,
        },
        "repetitions": 1,
        "warmup_runs": warmups,
        "cases": [
            {"case_id": f"case-{index}", "prompt": f"test prompt {index}", "seed": index,
             "requires_motion": True, "requires_sound": True}
            for index in (1, 2)
        ],
    }
    configuration = {
        "runtime": "test-runtime", "runtime_revision": name,
        "model_id": plan["model_id"], "model_revision": plan["model_revision"],
        "hardware_label": "same-test-device", "endpoint": "http://localhost:1234",
        "identity_verification": "operator_declared",
    }
    records = []
    slots = [(f"warmup-{index:03d}", plan["cases"][(index - 1) % 2], "warmup", 0) for index in range(1, warmups + 1)]
    slots += [(f"measurement-r001-c{index:03d}", case, "measurement", 1) for index, case in enumerate(plan["cases"], 1)]
    for slot_id, case, phase, repetition in slots:
        relative = f"media/{slot_id}.mp4"
        data = f"mock-{name}-{slot_id}".encode()
        (directory / relative).write_bytes(data)
        records.append({
            "slot_id": slot_id, "case_id": case["case_id"], "prompt": case["prompt"],
            "seed": case["seed"], "repetition": repetition, "phase": phase,
            "status": "succeeded", "attempted": True,
            "artifact_path": relative, "sha256": hashlib.sha256(data).hexdigest(),
            "latency_seconds": 1000.0 if phase == "warmup" else 10.0,
            "media": {"valid": True}, "error": None,
        })
    document = {
        "bundle_version": "0.1.0", "bundle_type": "mvp_run", "run_id": name,
        "plan_id": plan["plan_id"], "plan_sha256": digest(plan), "plan": plan,
        "configuration": configuration, "configuration_sha256": digest(configuration),
        "evidence_kind": "fixture",
        "status": "complete", "started_at": "2026-09-01T00:00:00+00:00", "finished_at": "2026-09-01T00:30:00+00:00",
        "measurement": {"boundary": "submit_to_validated_media", "concurrency": 1,
                        "wall_seconds": 20.5, "timing_evidence": "synthetic_fixture_control"},
        "records": records,
        "summary": {"valid": 9999, "latency_median_seconds": 0.00001},
    }
    write_bundle(directory, document)
    return directory, document


@pytest.fixture
def media_stub(monkeypatch):
    analysis = {"valid": True, "video": {"present": True, "width": 320, "height": 180, "frame_count": 48},
                "audio": {"present": True, "channels": 2, "sample_rate_hz": 32000}, "checks": []}
    comparison = {
        "compatible": True,
        "metrics": {
            "video_mae": 0.0, "video_psnr_db": None, "video_identical": True,
            "video_compared_frames": 48, "video_total_frames": 48, "video_sample_coverage_fraction": 1.0,
            "audio_spectral_cosine": 1.0, "audio_spectral_cosine_channels": [1.0, 1.0],
            "audio_rms_ratio": 1.0, "audio_rms_ratio_channels": [1.0, 1.0],
        },
        "checks": [], "notes": [],
    }
    monkeypatch.setattr(mvp_compare, "analyze_media", lambda path, expected=None: copy.deepcopy(analysis))
    monkeypatch.setattr(mvp_compare, "compare_media", lambda left, right: copy.deepcopy(comparison))
    return analysis, comparison


def test_recomputed_summary_pairs_only_measurements_and_retains_explicit_limits(tmp_path, media_stub):
    baseline, _ = bundle(tmp_path, "baseline")
    candidate, _ = bundle(tmp_path, "candidate")
    result = compare_runs(baseline, candidate, policy=POLICY)
    assert result["overall_status"] == "pass"
    assert result["release_qualified"] is False
    assert result["summary"]["measurement_slots"] == 2
    assert result["candidate"]["summary"]["valid"] == 2
    assert result["candidate"]["summary"]["latency_median_seconds"] == 10.0
    assert result["candidate"]["summary"]["valid_clips_per_second"] == pytest.approx(2 / 20.5)
    assert result["measurement"]["timing_evidence"]["candidate"] == "synthetic_fixture_control"
    json.dumps(result, allow_nan=False)


def test_latency_regression_uses_declared_threshold(tmp_path, media_stub):
    baseline, _ = bundle(tmp_path, "baseline")
    candidate, document = bundle(tmp_path, "candidate")
    for record in document["records"]:
        if record["phase"] == "measurement":
            record["latency_seconds"] = 12.0
    document["measurement"]["wall_seconds"] = 24.5
    write_bundle(candidate, document)
    result = compare_runs(baseline, candidate, policy=POLICY)
    assert result["overall_status"] == "fail"
    assert result["measurement"]["latency_increase_fraction"] == pytest.approx(0.2)
    assert next(check for check in result["checks"] if check["name"] == "performance.median_latency")["status"] == "fail"


def test_optional_client_timing_boundaries_are_nested_and_legacy_records_still_work(tmp_path, media_stub):
    baseline, _ = bundle(tmp_path, "baseline")
    candidate, document = bundle(tmp_path, "candidate")
    for record in document["records"]:
        record.update(submit_to_terminal_seconds=7.0, submit_to_media_seconds=8.0,
                      media_validation_seconds=1.5)
    write_bundle(candidate, document)
    assert compare_runs(baseline, candidate, policy=POLICY)["overall_status"] == "pass"


@pytest.mark.parametrize("updates,match", [
    ({"submit_to_terminal_seconds": -1}, "finite nonnegative"),
    ({"submit_to_terminal_seconds": True}, "finite nonnegative"),
    ({"submit_to_terminal_seconds": 9}, "follows completed media"),
    ({"submit_to_media_seconds": 11}, "exceeds total latency"),
    ({"media_validation_seconds": 3}, "download plus validation"),
    ({"submit_to_terminal_seconds": None}, "missing client timing"),
    ({"media_validation_seconds": None}, "missing client timing"),
])
def test_impossible_client_timing_boundaries_are_rejected(tmp_path, media_stub, updates, match):
    baseline, _ = bundle(tmp_path, "baseline")
    candidate, document = bundle(tmp_path, "candidate")
    document["records"][-1].update(submit_to_terminal_seconds=7.0,
                                  submit_to_media_seconds=8.0, media_validation_seconds=1.5)
    document["records"][-1].update(updates)
    write_bundle(candidate, document)
    with pytest.raises(ValueError, match=match):
        compare_runs(baseline, candidate, policy=POLICY)


def test_failed_attempt_may_have_only_terminal_timing(tmp_path, media_stub):
    baseline, _ = bundle(tmp_path, "baseline")
    candidate, document = bundle(tmp_path, "candidate")
    document["records"][-1].update(status="failed", artifact_path=None, sha256=None,
                                  media=None, error="provider failed", submit_to_terminal_seconds=7.0,
                                  submit_to_media_seconds=None, media_validation_seconds=None)
    write_bundle(candidate, document)
    assert compare_runs(baseline, candidate, policy=POLICY)["overall_status"] == "fail"


@pytest.mark.parametrize("field", ["hardware_label", "runtime"])
def test_cross_configuration_latency_is_descriptive_not_a_regression_gate(tmp_path, media_stub, field):
    baseline, _ = bundle(tmp_path, "baseline")
    candidate, document = bundle(tmp_path, "candidate")
    document["configuration"][field] = "different-class"
    document["configuration_sha256"] = digest(document["configuration"])
    for record in document["records"]:
        record["latency_seconds"] = 100.0
    document["measurement"]["wall_seconds"] = 200.5
    write_bundle(candidate, document)
    result = compare_runs(baseline, candidate, policy=POLICY)
    assert result["overall_status"] == "pass"
    assert result["measurement"]["performance_mode"] == "descriptive_only"
    assert next(check for check in result["checks"] if check["name"] == "performance.median_latency")["status"] == "descriptive"


def test_worst_audio_channel_cannot_be_hidden_by_aggregate(tmp_path, media_stub):
    baseline, _ = bundle(tmp_path, "baseline")
    candidate, _ = bundle(tmp_path, "candidate")
    media_stub[1]["metrics"]["audio_rms_ratio_channels"] = [1.0, 0.0]
    media_stub[1]["metrics"]["audio_spectral_cosine_channels"] = [1.0, 0.5]
    result = compare_runs(baseline, candidate, policy=POLICY)
    assert result["overall_status"] == "fail"
    names = {check["name"] for check in result["slots"][0]["checks"] if check["status"] == "fail"}
    assert names == {"fidelity.audio_rms_ratio_error", "fidelity.audio_spectral_cosine"}


@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
def test_missing_or_nonfinite_quality_metric_is_not_a_pass(tmp_path, media_stub, value):
    baseline, _ = bundle(tmp_path, "baseline")
    candidate, _ = bundle(tmp_path, "candidate")
    media_stub[1]["metrics"].update(video_identical=False, video_psnr_db=value)
    result = compare_runs(baseline, candidate, policy=POLICY)
    assert result["overall_status"] == "inconclusive"
    assert result["slots"][0]["metrics"]["video_psnr_db"] is None
    json.dumps(result, allow_nan=False)


def test_audio_undefined_from_silence_is_inconclusive(tmp_path, media_stub):
    baseline, _ = bundle(tmp_path, "baseline")
    candidate, _ = bundle(tmp_path, "candidate")
    media_stub[1]["metrics"]["audio_spectral_cosine_channels"] = [1.0, None]
    media_stub[1]["metrics"]["audio_rms_ratio_channels"] = [1.0, None]
    result = compare_runs(baseline, candidate, policy=POLICY)
    assert result["overall_status"] == "inconclusive"


def test_not_started_failure_is_retained_without_zero_latency_imputation(tmp_path, media_stub):
    baseline, _ = bundle(tmp_path, "baseline")
    candidate, document = bundle(tmp_path, "candidate")
    document["records"][-1].update(status="failed", attempted=False, latency_seconds=0,
                                  artifact_path=None, sha256=None, media=None, error="Not started after uncertain remote completion")
    write_bundle(candidate, document)
    result = compare_runs(baseline, candidate, policy=POLICY)
    summary = result["candidate"]["summary"]
    assert result["overall_status"] == "fail"
    assert summary["scheduled"] == 2
    assert summary["valid"] == summary["failed"] == summary["not_started"] == 1
    assert summary["technical_success_rate"] == 0.5
    assert summary["latency_median_seconds"] == summary["attempt_latency_median_seconds"] == 10.0
    assert result["slots"][-1]["metrics"].get("video_psnr_db") is None
    assert result["slots"][-1]["metrics"]["latency_increase_fraction"] is None
    assert result["slots"][-1]["candidate"]["latency_seconds"] is None


def test_baseline_failure_is_inconclusive_not_evidence_of_candidate_quality(tmp_path, media_stub):
    baseline, document = bundle(tmp_path, "baseline")
    candidate, _ = bundle(tmp_path, "candidate")
    document["records"][-1].update(status="failed", artifact_path=None, sha256=None, media=None, error="baseline failed")
    write_bundle(baseline, document)
    result = compare_runs(baseline, candidate, policy=POLICY)
    assert result["overall_status"] == "inconclusive"


def test_media_validation_is_recomputed_not_trusted(tmp_path, media_stub, monkeypatch):
    baseline, _ = bundle(tmp_path, "baseline")
    candidate, _ = bundle(tmp_path, "candidate")
    analysis = media_stub[0]
    monkeypatch.setattr(mvp_compare, "analyze_media", lambda path, expected=None: {
        **analysis, "valid": "candidate" not in path.parts,
        "checks": [{"name": "video.decode", "status": "failed", "detail": "corrupted frame"}],
    })
    result = compare_runs(baseline, candidate, policy=POLICY)
    assert result["overall_status"] == "fail"
    assert result["candidate"]["summary"]["valid"] == 0


def test_native_duration_is_resolved_from_frames_not_rounded_request(tmp_path, media_stub, monkeypatch):
    baseline, left = bundle(tmp_path, "baseline")
    candidate, right = bundle(tmp_path, "candidate")
    for directory, document in ((baseline, left), (candidate, right)):
        document["plan"]["generation"].update(duration_seconds=4, frame_count=107)
        document["plan_sha256"] = digest(document["plan"])
        write_bundle(directory, document)
    expected_values = []
    monkeypatch.setattr(mvp_compare, "analyze_media", lambda path, expected=None: expected_values.append(expected) or copy.deepcopy(media_stub[0]))
    compare_runs(baseline, candidate, policy=POLICY)
    assert all(expected["duration_seconds"] == 107 / 24 for expected in expected_values)


@pytest.mark.parametrize("mutation,match", [
    (lambda d: d["records"].pop(), "missing measurement"),
    (lambda d: d["records"].append(copy.deepcopy(d["records"][-1])), "duplicate slot"),
    (lambda d: d["records"][-1].update(prompt="changed prompt"), "prompt differs"),
    (lambda d: d["records"][-1].update(repetition=True), "repetition differs"),
    (lambda d: d["measurement"].update(concurrency=True), "concurrency=1"),
    (lambda d: d.update(plan_sha256="0" * 64), "plan SHA256"),
    (lambda d: d["configuration"].update(model_revision="changed"), "does not match the frozen plan"),
    (lambda d: d["configuration"].update(runtime_revision="changed"), "configuration SHA256"),
    (lambda d: d["records"][-1].update(sha256="0" * 64), "SHA256 mismatch"),
    (lambda d: d["records"][-1].update(artifact_path="../outside.mp4"), "escapes"),
    (lambda d: d["records"][-1].update(artifact_path="media/missing.mp4"), "missing artifact"),
    (lambda d: d["records"].pop(0), "missing warmup"),
    (lambda d: d["records"][0].update(slot_id="warmup-999"), "unexpected warmup"),
    (lambda d: d["records"].append(d["records"].pop(0)), "warmup slots must precede"),
    (lambda d: d["measurement"].update(wall_seconds=0.000001), "shorter than summed"),
    (lambda d: d["records"][-1].update(attempted=False), "not-started slot"),
    (lambda d: d.update(finished_at=None), "not finalized"),
    (lambda d: d.pop("configuration_sha256"), "declare its configuration SHA256"),
    (lambda d: d["records"].insert(1, d["records"].pop()), "frozen execution order"),
])
def test_malformed_or_tampered_bundles_are_refused(tmp_path, media_stub, mutation, match):
    baseline, _ = bundle(tmp_path, "baseline")
    candidate, document = bundle(tmp_path, "candidate")
    mutation(document)
    write_bundle(candidate, document)
    with pytest.raises(ValueError, match=match):
        compare_runs(baseline, candidate, policy=POLICY)


def test_artifact_symlink_may_not_escape_run_directory(tmp_path, media_stub):
    baseline, _ = bundle(tmp_path, "baseline")
    candidate, document = bundle(tmp_path, "candidate")
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")
    link = candidate / "media" / "link.mp4"
    link.symlink_to(outside)
    document["records"][-1].update(artifact_path="media/link.mp4", sha256=hashlib.sha256(b"outside").hexdigest())
    write_bundle(candidate, document)
    with pytest.raises(ValueError, match="inside its run directory"):
        compare_runs(baseline, candidate, policy=POLICY)


def test_duplicate_json_keys_are_rejected(tmp_path, media_stub):
    baseline, _ = bundle(tmp_path, "baseline")
    candidate, document = bundle(tmp_path, "candidate")
    (candidate / "run.json").write_text('{"run_id":"hidden",' + json.dumps(document)[1:], encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key: run_id"):
        compare_runs(baseline, candidate, policy=POLICY)


@pytest.mark.parametrize("field,value,match", [
    ("repetitions", 10001, "10000 total slots"),
    ("generation", {}, "generation.width"),
    ("warmup_runs", -1, "nonnegative"),
])
def test_invalid_plan_cannot_omit_expected_checks(tmp_path, media_stub, field, value, match):
    baseline, _ = bundle(tmp_path, "baseline")
    candidate, document = bundle(tmp_path, "candidate")
    document["plan"][field] = value
    document["plan_sha256"] = digest(document["plan"])
    write_bundle(candidate, document)
    with pytest.raises(ValueError, match=match):
        compare_runs(baseline, candidate, policy=POLICY)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), True, -0.1])
def test_threshold_policy_has_no_implicit_or_nonfinite_defaults(tmp_path, media_stub, value):
    policy = {**POLICY, "max_latency_increase_fraction": value}
    with pytest.raises(ValueError, match="explicit, finite"):
        compare_runs(tmp_path / "unused", tmp_path / "unused", policy=policy)


def test_report_is_portable_script_free_escaped_and_hash_checked(tmp_path, media_stub):
    baseline, _ = bundle(tmp_path, "baseline")
    candidate, _ = bundle(tmp_path, "candidate")
    result = compare_runs(baseline, candidate, policy=POLICY)
    result["slots"][0]["prompt"] = '<script>alert("x")</script> & injection'
    destination = tmp_path / "presentation" / "report.html"
    write_report(result, destination)
    content = destination.read_text(encoding="utf-8")
    assert "Synthetic fixture evidence" in content
    assert "not an H3 result" in content
    assert '<script>' not in content
    assert '&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; injection' in content
    assert 'src="http' not in content
    assert "report.comparison.json" in content
    portable = json.loads(destination.with_suffix(".comparison.json").read_text())
    artifact = portable["slots"][0]["baseline"]["artifact_path"]
    assert not Path(artifact).is_absolute()
    assert (destination.parent / artifact).is_file()
    assert Path(result["slots"][0]["baseline"]["artifact_path"]).is_absolute()
    with pytest.raises(FileExistsError):
        write_report(result, destination)
    Path(result["slots"][0]["baseline"]["artifact_path"]).write_bytes(b"tampered after comparison")
    with pytest.raises(ValueError, match="changed after comparison"):
        write_report(result, destination.with_name("tampered.html"))
    assert not destination.with_name("tampered.html").exists()
    assert not destination.with_name("tampered.comparison.json").exists()


def test_report_refuses_existing_asset_collision(tmp_path, media_stub):
    baseline, _ = bundle(tmp_path, "baseline")
    candidate, _ = bundle(tmp_path, "candidate")
    result = compare_runs(baseline, candidate, policy=POLICY)
    destination = tmp_path / "report.html"
    assets = tmp_path / "report_assets"
    assets.mkdir()
    key = result["slots"][0]["baseline"]["sha256"]
    (assets / f"{key}.mp4").write_bytes(b"wrong content")
    with pytest.raises(ValueError, match="different content"):
        write_report(result, destination)
    assert not destination.exists()


def test_imported_media_compares_fidelity_without_fake_performance(tmp_path, media_stub):
    baseline, left = bundle(tmp_path, "baseline", warmups=0)
    candidate, right = bundle(tmp_path, "candidate", warmups=0)
    for directory, document in ((baseline, left), (candidate, right)):
        document["evidence_kind"] = "imported_media"
        document["measurement"].update(boundary="not_measured_imported_media", wall_seconds=None)
        for record in document["records"]:
            record["latency_seconds"] = None
        document["provenance"] = {
            "source_url": "https://example.com/public-sample.mp4", "source_path": "/private/path/sample.mp4",
            "transformation": {"kind": "deliberate_audio_mute", "compressed_video_bitexact": True,
                               "decoded_video_identical": True, "audio_sample_count_preserved": True,
                               "full_stream_verification": {"details": "DETAILED_JSON_ONLY" * 1000}},
        }
        write_bundle(directory, document)
    result = compare_runs(baseline, candidate, policy=POLICY)
    assert result["overall_status"] == "pass"
    assert result["comparison_scope"] == "media_fidelity_only"
    assert result["measurement"]["performance_mode"] == "not_measured_imported_media"
    assert result["candidate"]["summary"]["latency_median_seconds"] is None
    assert result["candidate"]["summary"]["valid_clips_per_second"] is None
    assert all(check["status"] == "not_applicable" for check in result["checks"])
    output = tmp_path / "imported.html"
    write_report(result, output)
    html = output.read_text()
    assert "no H3 inference run or timing measurement" in html
    assert "Not measured" in html
    assert "pairing ID seed (not a generation seed)" in html
    assert "compressed video bit-exact: yes" in html
    assert "DETAILED_JSON_ONLY" not in html
    assert html.index("Side-by-side evidence") < html.index("Configurations and pins")
    assert "DETAILED_JSON_ONLY" in output.with_suffix(".comparison.json").read_text()
    assert "/private/path" not in output.with_suffix(".comparison.json").read_text()


def test_live_self_comparison_is_not_independent_evidence(tmp_path, media_stub):
    directory, document = bundle(tmp_path, "baseline")
    document["evidence_kind"] = "live_h3"
    write_bundle(directory, document)
    with pytest.raises(ValueError, match="distinct baseline and candidate"):
        compare_runs(directory, directory, policy=POLICY)


def _pin_client(document, media_stub, monkeypatch):
    code = {"implementation_version": "test-v1", "source_sha256": "1" * 64}
    monkeypatch.setattr(mvp_compare, "_active_media_code", lambda: code)
    media_stub[0]["implementation"] = {
        "version": "test-v1", "pyav_version": "17.1.0", "numpy_version": "2.5.2",
        "ffmpeg_libraries": {"libavcodec": "62.28.101"},
    }
    document["configuration"].update(
        limits={"poll_interval_seconds": 0.5},
        client_environment={"python": "3.13", "av": "17.1.0", "numpy": "2.5.2"},
        client_source_sha256="2" * 64,
        measurement_semantics={"latency": "submit through validated media"},
        media_evaluator={**code, "pyav_version": "17.1.0", "numpy_version": "2.5.2",
                         "ffmpeg_libraries": {"libavcodec": "62.28.101"}},
    )
    document["configuration_sha256"] = digest(document["configuration"])


@pytest.mark.parametrize("field", ["limits", "client_environment", "client_source_sha256", "measurement_semantics", "media_evaluator"])
@pytest.mark.parametrize("change", ["different", "missing"])
def test_client_measurement_changes_cannot_performance_gate(tmp_path, media_stub, monkeypatch, field, change):
    baseline, left = bundle(tmp_path, "baseline")
    candidate, right = bundle(tmp_path, "candidate")
    for document in (left, right):
        _pin_client(document, media_stub, monkeypatch)
    if change == "missing":
        right["configuration"].pop(field)
    elif isinstance(right["configuration"][field], dict):
        right["configuration"][field] = {**right["configuration"][field], "changed": True}
    else:
        right["configuration"][field] = "3" * 64
    right["configuration_sha256"] = digest(right["configuration"])
    write_bundle(baseline, left)
    write_bundle(candidate, right)
    result = compare_runs(baseline, candidate, policy=POLICY)
    assert result["measurement"]["performance_mode"] == "descriptive_only"
    assert next(check for check in result["checks"] if check["name"] == "performance.median_latency")["status"] == "descriptive"


def test_live_performance_needs_pinned_client_but_not_matching_endpoints(tmp_path, media_stub, monkeypatch):
    baseline, left = bundle(tmp_path, "baseline")
    candidate, right = bundle(tmp_path, "candidate")
    for directory, document in ((baseline, left), (candidate, right)):
        document["evidence_kind"] = "live_h3"
        write_bundle(directory, document)
    result = compare_runs(baseline, candidate, policy=POLICY)
    assert result["measurement"]["performance_mode"] == "descriptive_only"
    for directory, document in ((baseline, left), (candidate, right)):
        _pin_client(document, media_stub, monkeypatch)
        document["configuration"]["endpoint"] = "http://localhost:9999" if directory == candidate else "http://localhost:8888"
        document["configuration_sha256"] = digest(document["configuration"])
        write_bundle(directory, document)
    result = compare_runs(baseline, candidate, policy=POLICY)
    assert result["overall_status"] == "pass"
    assert result["measurement"]["performance_mode"] == "same_configuration_class_regression"
    media_stub[0]["implementation"]["pyav_version"] = "different-fresh-evaluator"
    result = compare_runs(baseline, candidate, policy=POLICY)
    assert result["measurement"]["performance_mode"] == "descriptive_only"


def test_evaluator_unavailable_is_counted_without_inventing_model_failure(tmp_path, media_stub, monkeypatch):
    baseline, _ = bundle(tmp_path, "baseline", warmups=0)
    candidate, _ = bundle(tmp_path, "candidate", warmups=0)

    def analyze(path, expected=None):
        if "candidate" in path.parts:
            raise RuntimeError("evaluation service unavailable")
        return copy.deepcopy(media_stub[0])

    monkeypatch.setattr(mvp_compare, "analyze_media", analyze)
    result = compare_runs(baseline, candidate, policy=POLICY)
    summary = result["candidate"]["summary"]
    assert result["overall_status"] == "inconclusive"
    assert summary["evaluator_unavailable"] == 2
    assert summary["known_invalid_completed"] == 0
    assert summary["invalid_completed"] == 2
    assert summary["verified_technical_success_fraction"] == 0
    destination = tmp_path / "unavailable.html"
    write_report(result, destination)
    assert "Media evaluation unavailable for 2 observation(s)" in destination.read_text()


@pytest.mark.parametrize("suffix", [".html", ".comparison.json"])
@pytest.mark.parametrize("symlink", [False, True])
def test_preexisting_report_outputs_are_never_overwritten(tmp_path, media_stub, suffix, symlink):
    baseline, _ = bundle(tmp_path, "baseline", warmups=0)
    candidate, _ = bundle(tmp_path, "candidate", warmups=0)
    result = compare_runs(baseline, candidate, policy=POLICY)
    destination = tmp_path / "protected.html"
    protected = destination.with_suffix(suffix)
    if symlink:
        owner = tmp_path / "original-content.txt"
        owner.write_text("original-content")
        protected.symlink_to(owner)
    else:
        protected.write_text("original-content")
    with pytest.raises(FileExistsError):
        write_report(result, destination)
    assert protected.read_text() == "original-content"
    assert not (tmp_path / "protected_assets").exists()


@pytest.mark.parametrize("suffix", [".html", ".comparison.json"])
def test_report_exclusive_creation_handles_post_preflight_race(tmp_path, media_stub, monkeypatch, suffix):
    baseline, _ = bundle(tmp_path, "baseline", warmups=0)
    candidate, _ = bundle(tmp_path, "candidate", warmups=0)
    result = compare_runs(baseline, candidate, policy=POLICY)
    destination = tmp_path / "racing.html"
    racing = destination.with_suffix(suffix)
    original_open = Path.open

    def race(path, mode="r", *args, **kwargs):
        if path == racing and mode == "x":
            with original_open(path, "w", encoding="utf-8") as stream:
                stream.write("concurrent-owner-content")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", race)
    with pytest.raises(FileExistsError):
        write_report(result, destination)
    assert racing.read_text() == "concurrent-owner-content"
    other = destination.with_suffix(".comparison.json") if suffix == ".html" else destination
    assert not other.exists()
    assert not (tmp_path / "racing_assets").exists()
