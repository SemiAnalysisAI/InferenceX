"""CPU-only exporter tests reuse the supervisor's synthetic evidence fixture."""
import json

import pytest

from evaluator import mvp_gpu_job as gpu
from evaluator.mvp_result import write_result
from test_mvp_gpu_job import GPU, saved_job, spec  # noqa: F401

COMMIT = "a" * 40
PRODUCER = {"git_commit": "b" * 40, "ci": {"run_id": "999"}}
SOURCE_CI = {"databaseId": 123, "runAttempt": 1, "headSha": COMMIT, "url": "https://github.com/SemiAnalysisAI/InferenceX/actions/runs/123", "status": "completed", "conclusion": "success", "jobs": [{"name": "h3-video / H3 video H200 smoke", "status": "completed", "conclusion": "success"}]}


def _seal(root):
    (root / "SHA256SUMS").write_text("".join(f"{gpu._hash(path)}  {path.relative_to(root).as_posix()}\n" for path in sorted(root.rglob("*")) if path.is_file() and path.name != "SHA256SUMS"))


@pytest.fixture
def bundle(spec, tmp_path, request):
    root = tmp_path / "artifact"
    root.mkdir()
    spec["plan"]["cases"] = spec["plan"]["cases"][:1]
    spec["plan"]["repetitions"] = 1
    spec["allocation"] = {"mode": "dedicated_ci", "label": "Slurm 456.0 on test-node"}
    if getattr(request, "param", None):
        spec["serving"] = request.param
    saved_job(spec, root / "gpu")
    receipt = gpu._read(root / "gpu/gpu-job.json")
    receipt.update(regression_status="inconclusive", ci_accepted=False, release_qualified=False)
    gpu._write(root / "gpu/gpu-job.json", receipt)
    for role in ("baseline", "candidate"):
        (root / f"gpu/{role}/events.jsonl").write_text("")
        for name in ("runtime.stdout.log", "runtime.stderr.log", "client.stderr.log", "client.stdout.json"):
            (root / f"gpu/supervisor/{role}/{name}").write_text("synthetic test log\n")
    (root / "report").mkdir()
    (root / "report/index.html").write_text("<html>Synthetic CPU test report</html>")
    allocation = {"identity": {"JobId": "456"}}
    ci_identity = {"repository": "SemiAnalysisAI/InferenceX", "workflow_sha": COMMIT, "run_url": SOURCE_CI["url"]}
    ci = {"schema_version": 1, "source_sha": COMMIT, "run_id": "123", "run_attempt": "1", "ci": ci_identity,
          "allocation": allocation, "slurm_job": {"JobId": "456", "NodeList": "test-node", "AllocTRES": "cpu=8,gres/gpu=8"},
          "step_cleanup": {"status": "ended", "step_id": "456.0"}, "allocation_cleanup": {"status": "released"}, "exit_code": 0}
    gpu._write(root / "ci.json", ci)
    gpu._write(root / "allocation.json", allocation)
    gpu._write(root / "binding.json", {"job_id": "456", "step_id": "0", "node": "test-node", "gpu_uuids": [GPU]})
    gpu._write(root / "step-result.json", {"exit_code": 0})
    gpu._write(root / "manifest.json", {"schema_version": 1, "git_commit": COMMIT, "run_id": "123", "run_attempt": "1", "ci": ci_identity,
        "slurm_allocation": allocation, "workload_plan": spec["plan"], "exit_code": 0,
        "evidence": {"ci.json": gpu._hash(root / "ci.json")}})
    _seal(root)
    return root


def test_export_preserves_execution_provenance_and_withholds_missing_power(bundle):
    original = {path.relative_to(bundle): path.read_bytes() for path in bundle.rglob("*") if path.is_file()}
    result = write_result(bundle, producer=PRODUCER, source_ci=SOURCE_CI)
    assert result["schema_version"] == "1.0.0"
    assert result["status"] == "complete" and result["workload_status"] == "passed"
    assert result["regression_status"] == "inconclusive" and result["release_qualified"] is False
    assert result["execution"]["ci"]["git_commit"] == COMMIT != result["producer"]["git_commit"]
    assert result["execution"]["ci"]["run_id"] == "123" != result["producer"]["ci"]["run_id"]
    assert result["hardware"]["selected_gpu_count"] == 1 and result["hardware"]["reserved_gpu_count"] == 8
    assert result["hardware"]["tdp"]["watts_per_gpu"] is None
    baseline = result["roles"]["baseline"]
    assert baseline["metrics"]["latency_seconds"]["mean"] == 0.001
    assert baseline["metrics"]["valid_clips_per_second"] == 100.0
    assert baseline["power"]["phases"]["measurement"]["valid"] is False
    assert baseline["power"]["phases"]["measurement"]["aggregate"] is None
    assert all((bundle / path).read_bytes() == content for path, content in original.items())
    assert all((bundle / item["path"]).is_file() and gpu._hash(bundle / item["path"]) == item["sha256"] for item in result["files"])


@pytest.mark.parametrize("mutation", ["tamper", "extra_file", "symlink", "missing_telemetry", "unknown_version", "nonzero_exit", "slurm_mismatch", "trusted_ci_mismatch", "missing_report_asset"])
def test_invalid_bundle_writes_failed_result_and_propagates_error(bundle, mutation):
    trusted = dict(SOURCE_CI)
    if mutation == "tamper":
        (bundle / "report/index.html").write_text("changed")
    elif mutation == "extra_file":
        (bundle / "unlisted.txt").write_text("unsealed")
    elif mutation == "symlink":
        (bundle / "media-link").symlink_to(bundle / "report/index.html")
    elif mutation == "trusted_ci_mismatch":
        trusted["headSha"] = "c" * 40
    else:
        if mutation == "missing_report_asset":
            (bundle / "report/index.html").write_text("<video src='assets/missing.mp4'></video>")
        elif mutation == "missing_telemetry":
            (bundle / "gpu/supervisor/baseline/telemetry.jsonl").unlink()
        else:
            path = bundle / ("step-result.json" if mutation == "nonzero_exit" else "manifest.json" if mutation == "unknown_version" else "binding.json")
            value = json.loads(path.read_text())
            value.update({"exit_code": 2} if mutation == "nonzero_exit" else {"schema_version": 99} if mutation == "unknown_version" else {"job_id": "987"})
            gpu._write(path, value)
        _seal(bundle)
    with pytest.raises(ValueError):
        write_result(bundle, producer=PRODUCER, source_ci=trusted)
    failed = json.loads((bundle / "result.json").read_text())
    assert failed["status"] == "failed" and failed["invalid_reasons"]
    assert failed["release_qualified"] is False
    assert all(role["metrics"]["status"] == "withheld" for role in failed["roles"].values())


def test_frontend_schema_rejects_unknown_version_and_invalid_power_values(bundle):
    from copy import deepcopy
    from pathlib import Path
    import jsonschema

    schema = json.loads((Path(__file__).parents[1] / "result.schema.json").read_text())
    result = write_result(bundle, producer=PRODUCER, source_ci=SOURCE_CI)
    jsonschema.Draft202012Validator(schema).validate(result)
    invalid = deepcopy(result)
    invalid["schema_version"] = "2.0.0"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(invalid)
    invalid = deepcopy(result)
    invalid["roles"]["baseline"]["power"]["phases"]["measurement"]["aggregate"] = {"energy_j": 99, "avg_power_w": 99, "observed_peak_power_w": 99, "joules_per_valid_clip": 99}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(invalid)


def test_later_hardware_profile_never_backfills_historical_limits_and_html_escapes(bundle):
    profile = {"schema_version": 1, "observation_kind": "read_only_inventory", "observed_at": "2026-09-10T00:00:00Z",
               "gpu_uuids": [GPU], "slurm": {"gpu_uuids": [GPU]},
               "power_configuration": {"gpus": [{"uuid": GPU, "configured_limit_w": 700}]},
               "tdp": {"status": "verified", "watts_per_gpu": 700, "hardware_variant": "synthetic SXM test",
                       "source_url": "https://example.org/test-hardware", "evidence": "<script>bad()</script>"}}
    result = write_result(bundle, producer=PRODUCER, source_ci=SOURCE_CI, hardware_profile=profile)
    assert result["hardware"]["tdp"]["watts_per_gpu"] == 700
    assert result["hardware"]["configured_power_limits"]["watts_by_gpu"] is None
    assert result["hardware"]["later_hardware_observation"]["observed_at"] == "2026-09-10T00:00:00Z"
    power_report = (bundle / "power-report.html").read_text()
    assert "<script>bad()</script>" not in power_report and "&lt;script&gt;bad()&lt;/script&gt;" in power_report
    assert "report/index.html" in power_report and "Baseline samples and coverage" in power_report


def test_same_run_export_keeps_workflow_pending_and_verifies_completed_gpu_job(bundle):
    from copy import deepcopy

    trusted = deepcopy(SOURCE_CI) | {"status": "in_progress", "conclusion": None}
    trusted["jobs"][0]["name"] = "p1.500 | H3 video H200 smoke"
    producer = {"git_commit": COMMIT, "mode": "same_run_export", "ci": {
        "run_id": "123", "run_attempt": "1", "repository": "SemiAnalysisAI/InferenceX"}}
    result = write_result(bundle, producer=producer, source_ci=trusted)
    assert result["workload_status"] == "passed"
    assert result["execution"]["ci"]["external_ci_verification"] == "passed"
    assert result["execution"]["ci"]["workflow_status_at_export"] == "in_progress"
    assert result["execution"]["ci"]["workflow_conclusion_at_export"] is None


@pytest.mark.parametrize("mismatch", ["run", "attempt", "repository", "commit", "mode", "source", "pending_job", "failed_job"])
def test_same_run_exception_does_not_skip_execution_identity_or_job_status(bundle, mismatch):
    from copy import deepcopy

    trusted = deepcopy(SOURCE_CI) | {"status": "in_progress", "conclusion": None}
    producer = {"git_commit": COMMIT, "mode": "same_run_export", "ci": {
        "run_id": "123", "run_attempt": "1", "repository": "SemiAnalysisAI/InferenceX"}}
    if mismatch in {"run", "attempt", "repository"}:
        producer["ci"][{"run": "run_id", "attempt": "run_attempt", "repository": "repository"}[mismatch]] = "wrong"
    elif mismatch == "commit":
        producer["git_commit"] = "e" * 40
    elif mismatch == "mode":
        producer["mode"] = "historical_replay"
    elif mismatch == "source":
        trusted["headSha"] = "f" * 40
    elif mismatch == "pending_job":
        trusted["jobs"][0]["status"] = "in_progress"
    else:
        trusted["jobs"][0]["conclusion"] = "failure"
    with pytest.raises(ValueError):
        write_result(bundle, producer=producer, source_ci=trusted)
    assert json.loads((bundle / "result.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("change", ["different_setting", "wrong_device"])
def test_contemporaneous_limits_preserve_each_snapshot_without_claiming_stability(bundle, change):
    receipt_path = bundle / "gpu/gpu-job.json"
    receipt = gpu._read(receipt_path)
    for role in receipt["roles"].values():
        role.update(started_at="2026-08-03T00:00:01Z", finished_at="2026-08-03T00:59:59Z")
        for when, timestamp in (("before", "2026-08-03T00:00:00Z"), ("after", "2026-08-03T01:00:00Z")):
            role[f"power_configuration_{when}"] = {"status": "recorded", "observed_at": timestamp, "gpus": [{
                "uuid": GPU, "configured_limit_w": 600, "enforced_limit_w": 600, "default_limit_w": 700, "maximum_limit_w": 700}]}
    after = receipt["roles"]["baseline"]["power_configuration_after"]["gpus"][0]
    after["configured_limit_w" if change == "different_setting" else "uuid"] = 700 if change == "different_setting" else "GPU-WRONG"
    gpu._write(receipt_path, receipt)
    _seal(bundle)
    result = write_result(bundle, producer=PRODUCER, source_ci=SOURCE_CI)
    limits = result["hardware"]["configured_power_limits"]
    assert limits["watts_by_gpu"] is None
    assert limits["by_role"]["baseline"]["before"]["gpus"][0]["configured_limit_w"] == 600
    if change == "different_setting":
        assert limits["status"] == "recorded"
        assert limits["by_role"]["baseline"]["after"]["gpus"][0]["configured_limit_w"] == 700
        assert limits["by_role"]["baseline"]["same_observed_values"] is False
    else:
        assert limits["status"] == "partial"
        assert limits["by_role"]["baseline"]["after"]["gpus"] is None
    assert result["workload_status"] == "passed"


@pytest.mark.parametrize('bundle', [{'concurrency': 2, 'delivery_deadline_seconds': 1}], indirect=True)
def test_serving_export_keeps_contract_media_and_deployment_identity(bundle):
    from pathlib import Path
    import jsonschema
    result = write_result(bundle, producer=PRODUCER, source_ci=SOURCE_CI)
    jsonschema.Draft202012Validator(json.loads((Path(__file__).parents[1] / 'result.schema.json').read_text())).validate(result)
    assert result['schema_version'] == '1.0.0'
    assert result['execution']['deployment']['replica_count'] == 1
    assert result['execution']['deployment']['gpus_per_replica'] == 1
    assert result['execution']['deployment']['configured_batch_size'] is None
    assert result['hardware']['reserved_gpu_count'] == 8
    stats = result['roles']['baseline']['metrics']['serving']
    assert stats['concurrency'] == 2
    assert stats['deadline_met_valid_clips'] == 1
    assert stats['client_ready_latency_seconds']['p50'] == .0005
    assert stats['client_ready_latency_seconds']['p90'] is None
    assert stats['capacity_qualified'] is False
    assert result['roles']['baseline']['records'][1]['job_id'] == 'fixture-1'
    media = result['roles']['baseline']['records'][1]['media_file']
    assert gpu._hash(bundle / media['path']) == media['sha256']
