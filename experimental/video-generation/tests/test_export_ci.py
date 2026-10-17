"""Exporter trust and failure paths, with GitHub/download collaborators faked."""

from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys

import pytest

import export_ci


REPO = "SemiAnalysisAI/InferenceX"
SHA = "a" * 40
REAL_API = export_ci.api


@pytest.fixture
def github(monkeypatch):
    run = {"id": 123, "repository": {"full_name": REPO}, "head_repository": {"full_name": REPO},
           "head_sha": SHA, "run_attempt": 1, "event": "workflow_dispatch", "status": "completed",
           "conclusion": "success", "html_url": f"https://github.com/{REPO}/actions/runs/123"}
    jobs = {"jobs": [{"id": 456, "name": "h3-video / p1 | H3 video H200 smoke",
                       "status": "completed", "conclusion": "success"}]}
    artifact = {"id": 789, "name": "h3-video-123-1", "expired": False, "size_in_bytes": 1000,
                "workflow_run": {"id": 123, "head_sha": SHA}}
    responses = {"actions/runs/123": run, "actions/runs/123/attempts/1/jobs": jobs,
                 "actions/runs/123/artifacts": {"artifacts": [artifact]}}
    monkeypatch.setattr(export_ci, "api", lambda path: deepcopy(responses[path]))
    monkeypatch.setenv("GITHUB_SHA", SHA)
    monkeypatch.setenv("GITHUB_RUN_ID", "999")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setenv("GITHUB_REPOSITORY", REPO)
    return run, jobs, artifact


@pytest.mark.parametrize("value", ["", "0", "123,123", "1,2,3", "../123", "1;touch x", "1, 2", "123\n", "1" * 21])
def test_source_ids_reject_ambiguous_or_non_numeric_dispatch_input(value):
    with pytest.raises(ValueError):
        export_ci.source_ids(value)


def test_source_ids_preserve_order_without_shell_interpretation():
    assert export_ci.source_ids("34293342829,34291306687") == ["34293342829", "34291306687"]


@pytest.mark.parametrize("hardware", ["H100", "B200", "MI355X"])
def test_serving_sources_accept_verified_cross_hardware_generation(github, hardware):
    github[1]["jobs"][0]["name"] = f"h3-video / p1.5 | H3 video {hardware} smoke"
    source, artifact = export_ci.verified_execution("123")
    assert source["headSha"] == SHA and artifact["id"] == 789


def test_inventory_reuse_requires_the_hardware_job_and_artifact(github):
    with pytest.raises(ValueError):
        export_ci.verified_execution("123", inventory=True)
    github[1]["jobs"][0]["name"] = "h3-video / p1.500 | H3 H200 hardware inventory"
    github[2]["name"] = "h3-hardware-123-1"
    source_ci, artifact = export_ci.verified_execution("123", inventory=True)
    assert source_ci["databaseId"] == 123 and artifact["name"] == "h3-hardware-123-1"


def test_authentication_stays_out_of_saved_public_ci_receipts(github, monkeypatch, tmp_path):
    run, jobs, artifact = github
    sentinel = "CPU-test-authentication-sentinel"
    for value in (run, jobs["jobs"][0], artifact):
        value["unexpected_auth_field"] = sentinel
    responses = iter((run, jobs, {"artifacts": [artifact]}))
    def response(request, timeout):
        assert request.get_header("Authorization") == "Bearer " + sentinel
        assert request.full_url.startswith("https://api.github.com/repos/" + REPO + "/")
        return BytesIO(json.dumps(next(responses)).encode())
    monkeypatch.setenv("GH_TOKEN", sentinel)
    monkeypatch.setattr(export_ci, "api", REAL_API)
    monkeypatch.setattr(export_ci, "urlopen", response)
    source_ci, source_artifact = export_ci.verified_execution("123")
    path = tmp_path / "public-ci.json"
    export_ci.ci.write(path, {"source_ci": source_ci, "artifact": source_artifact})
    assert sentinel not in path.read_text()
    assert source_ci["databaseId"] == 123 and source_artifact["id"] == 789


@pytest.mark.parametrize("job_name", ["h3-video / p1.500 | H3 video H200 smoke", "p1 | H3 video H200 smoke"])
def test_verified_execution_returns_exact_attempt_and_artifact(github, job_name):
    github[1]["jobs"][0]["name"] = job_name
    source_ci, artifact = export_ci.verified_execution("123")
    assert (source_ci["databaseId"], source_ci["headSha"], source_ci["runAttempt"]) == (123, SHA, 1)
    assert source_ci["jobs"][0]["id"] == 456
    assert artifact["id"] == 789


@pytest.mark.parametrize("defect", ["fork", "repository", "event", "failed", "other_in_progress",
                                    "run_id", "job_failed", "job_not_completed", "job_name", "duplicate_job",
                                    "expired", "artifact_commit", "artifact_run", "empty_artifact", "oversized_artifact"])
def test_verified_execution_rejects_wrong_execution_or_artifact_identity(github, defect):
    run, jobs, artifact = github
    if defect == "fork":
        run["head_repository"]["full_name"] = "someone/InferenceX"
    elif defect == "repository":
        run["repository"]["full_name"] = "someone/InferenceX"
    elif defect == "event":
        run["event"] = "pull_request"
    elif defect == "failed":
        run["conclusion"] = "failure"
    elif defect == "other_in_progress":
        run.update(status="in_progress", conclusion=None)
    elif defect == "run_id":
        run["id"] = 999
    elif defect == "job_failed":
        jobs["jobs"][0]["conclusion"] = "failure"
    elif defect == "job_not_completed":
        jobs["jobs"][0]["status"] = "in_progress"
    elif defect == "job_name":
        jobs["jobs"][0]["name"] = "unrelated H3 video H200 smoke fixture"
    elif defect == "duplicate_job":
        jobs["jobs"].append(deepcopy(jobs["jobs"][0]))
    elif defect == "expired":
        artifact["expired"] = True
    elif defect == "artifact_commit":
        artifact["workflow_run"]["head_sha"] = "b" * 40
    elif defect == "artifact_run":
        artifact["workflow_run"]["id"] = 124
    elif defect == "empty_artifact":
        artifact["size_in_bytes"] = 0
    else:
        artifact["size_in_bytes"] = 2 * 1024**3 + 1
    with pytest.raises(ValueError):
        export_ci.verified_execution("123")


@pytest.mark.parametrize("mismatch", [None, "commit", "attempt"])
def test_current_run_exception_requires_exact_producer_and_finished_h3_job(github, monkeypatch, mismatch):
    run, _, _ = github
    run.update(status="in_progress", conclusion=None)
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    if mismatch == "commit":
        monkeypatch.setenv("GITHUB_SHA", "b" * 40)
    elif mismatch == "attempt":
        monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    if mismatch:
        with pytest.raises(ValueError):
            export_ci.verified_execution("123")
    else:
        source_ci, _ = export_ci.verified_execution("123")
        assert source_ci["status"] == "in_progress"
        assert source_ci["jobs"][0]["conclusion"] == "success"


def fake_download(monkeypatch):
    original = b"raw runtime log\n"
    seal = f"{hashlib.sha256(original).hexdigest()}  runtime.log\n".encode()
    def download(argv, **kwargs):
        assert kwargs == {"check": True, "timeout": 180}
        assert argv[:5] == ["gh", "run", "download", "123", "--repo"]
        destination = Path(argv[argv.index("--dir") + 1])
        destination.mkdir(parents=True)
        (destination / "runtime.log").write_bytes(original)
        (destination / "SHA256SUMS").write_bytes(seal)
    monkeypatch.setattr(export_ci.subprocess, "run", download)
    monkeypatch.setattr(export_ci.ci, "command", lambda argv, **kwargs: SHA)
    return original, seal


def test_current_run_export_keeps_independent_github_metadata_for_manifest_join(github, monkeypatch, tmp_path):
    run, _, _ = github
    run.update(status="in_progress", conclusion=None)
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    fake_download(monkeypatch)
    observed = []
    def write_result(target, **kwargs):
        observed.append(kwargs)
        (target / "result.json").write_text('{"status":"complete"}')
        return {"status": "complete"}
    monkeypatch.setattr(export_ci, "write_result", write_result)
    assert export_ci.publish(["123"], tmp_path / "output", None) == 0
    assert observed[0]["producer"]["mode"] == "same_run_export"
    assert observed[0]["source_ci"]["databaseId"] == 123
    assert observed[0]["source_ci"]["headSha"] == SHA
    assert observed[0]["source_ci"]["runAttempt"] == 1


def test_failed_result_export_preserves_raw_files_old_seal_and_error_seal(github, monkeypatch, tmp_path):
    original, seal = fake_download(monkeypatch)
    def write_result(target, **kwargs):
        (target / "result.json").write_text('{"status":"failed"}')
        raise ValueError("Media reference is missing")
    monkeypatch.setattr(export_ci, "write_result", write_result)
    output = tmp_path / "output"
    assert export_ci.publish(["123"], output, None) == 2
    target = output / "source-123"
    assert (target / "runtime.log").read_bytes() == original
    assert (target / "source-SHA256SUMS").read_bytes() == seal
    assert json.loads((target / "export-error.json").read_text())["exit_code"] == 2
    assert json.loads((output / "index.json").read_text())["status"] == "partial"
    checksums = dict(line.split("  ", 1)[::-1] for line in (target / "SHA256SUMS").read_text().splitlines())
    assert {"runtime.log", "source-SHA256SUMS", "result.json", "export-error.json"} <= checksums.keys()
    for name, expected in checksums.items():
        assert hashlib.sha256((target / name).read_bytes()).hexdigest() == expected


def seal(root):
    files = [(path.name, hashlib.sha256(path.read_bytes()).hexdigest()) for path in root.iterdir() if path.name != "SHA256SUMS"]
    (root / "SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(files)))


def hardware_artifact(root):
    from inventory_ci import classify_tdp

    root.mkdir()
    xml = "<nvidia_smi_log><gpu><uuid>GPU-fixture</uuid><product_name>CPU TEST</product_name></gpu></nvidia_smi_log>"
    identity = {"run_id": "999", "run_attempt": "1", "source_sha": SHA}
    binding = {"job_id": "7", "step_id": "0", "gpu_uuids": ["GPU-fixture"], "node": "fixture-node"}
    profile = {**identity, "git_commit": SHA, "ci": {"repository": REPO, "run_id": "999", "run_attempt": "1"},
               "slurm": deepcopy(binding), "gpu_uuids": ["GPU-fixture"], "tdp": classify_tdp(xml, ["GPU-fixture"]),
               "raw": {"inventory": "nvidia-smi.xml", "topology": "topology.txt"}}
    files = {"hardware-profile.json": profile, "binding.json": binding,
             "ci.json": {**identity, "phase": "complete", "exit_code": 0,
                         "step_cleanup": {"status": "ended", "step_id": "7.0"},
                         "allocation_cleanup": {"status": "released"}, "allocation_reused": False,
                         "allocation": {"identity": {"JobId": "7"}}},
             "manifest.json": {**identity, "exit_code": 0, "slurm_allocation": {"identity": {"JobId": "7"}}},
             "step-result.json": {"exit_code": 0, "inventory_completed": True}}
    for name, value in files.items():
        (root / name).write_text(json.dumps(value))
    (root / "nvidia-smi.xml").write_text(xml)
    (root / "topology.txt").write_text("CPU TEST topology placeholder")
    seal(root)
    return profile


def test_verified_hardware_preserves_source_and_joins_slurm_identity(tmp_path):
    root = tmp_path / "hardware"
    hardware_artifact(root)
    original = (root / "hardware-profile.json").read_bytes()
    profile = export_ci.verified_hardware(root, "999", "1", SHA)
    assert profile["slurm"]["job_id"] == "7"
    assert profile["raw"]["inventory"] == "hardware/nvidia-smi.xml"
    assert (root / "hardware-profile.json").read_bytes() == original


def test_reclassify_retained_raw_pci_without_rewriting_the_inventory(tmp_path):
    root = tmp_path / "hardware"
    hardware_artifact(root)
    (root / "nvidia-smi.xml").write_text('<nvidia_smi_log><gpu><uuid>GPU-fixture</uuid><product_name>NVIDIA H200</product_name>'
        '<pci><pci_device_id>233510DE</pci_device_id><pci_sub_system_id>18BE10DE</pci_sub_system_id></pci></gpu></nvidia_smi_log>')
    seal(root)
    original = (root / "hardware-profile.json").read_bytes()
    profile = export_ci.verified_hardware(root, "999", "1", SHA)
    assert profile["tdp"]["status"] == "verified" and profile["tdp"]["watts_per_gpu"] == 700
    assert profile["recorded_tdp_classification"]["status"] == "unknown"
    assert (root / "hardware-profile.json").read_bytes() == original


def test_cpu_export_reuses_independently_verified_inventory_commit(github, monkeypatch, tmp_path):
    fake_download(monkeypatch)
    original_download = export_ci.subprocess.run
    execution = export_ci.verified_execution
    def admission(run_id, *, inventory=False):
        if inventory:
            assert run_id == "456"
            return {"runAttempt": 2, "headSha": "b" * 40}, {"name": "h3-hardware-456-2"}
        return execution(run_id)
    def download(argv, **kwargs):
        if argv[3] == "456":
            hardware_artifact(Path(argv[argv.index("--dir") + 1]))
        else:
            original_download(argv, **kwargs)
    observed = []
    def verify(root, run_id, attempt, sha):
        observed.append((run_id, attempt, sha))
        return {"gpu_uuids": ["GPU-fixture"]}
    monkeypatch.setattr(export_ci, "verified_execution", admission)
    monkeypatch.setattr(export_ci.subprocess, "run", download)
    monkeypatch.setattr(export_ci, "verified_hardware", verify)
    monkeypatch.setattr(export_ci, "write_result", lambda *args, **kwargs: {"status": "complete"})
    monkeypatch.setattr(export_ci.ci, "allocate", lambda *args, **kwargs: pytest.fail("CPU export allocated GPUs"))
    output = tmp_path / "output"
    assert export_ci.publish(["123"], output, None, hardware_run_id="456") == 0
    assert observed == [("456", "2", "b" * 40)]
    assert (output / "source-123/hardware/nvidia-smi.xml").is_file()


@pytest.mark.parametrize("defect", ["checksum", "cleanup", "step", "gpu", "missing_raw", "invented_tdp"])
def test_verified_hardware_rejects_bad_seal_or_unproven_teardown(tmp_path, defect):
    root = tmp_path / "hardware"
    hardware_artifact(root)
    if defect == "checksum":
        (root / "nvidia-smi.xml").write_text("changed after seal")
    elif defect == "missing_raw":
        (root / "topology.txt").unlink()
        seal(root)
    else:
        name = "hardware-profile.json" if defect in ("gpu", "invented_tdp") else "ci.json"
        value = json.loads((root / name).read_text())
        if defect == "cleanup":
            value["allocation_cleanup"]["status"] = "failed"
        elif defect == "step":
            value["step_cleanup"]["step_id"] = "8.0"
        elif defect == "gpu":
            value["gpu_uuids"] = ["GPU-other"]
        else:
            value["tdp"]["watts_per_gpu"] = 700
        (root / name).write_text(json.dumps(value))
        seal(root)
    with pytest.raises(ValueError):
        export_ci.verified_hardware(root, "999", "1", SHA)


@pytest.mark.parametrize("field", ["git_commit", "run_id", "run_attempt"])
def test_hardware_profile_mismatch_fails_before_download_and_retains_error(github, monkeypatch, tmp_path, field):
    monkeypatch.setattr(export_ci.ci, "command", lambda argv, **kwargs: SHA)
    monkeypatch.setattr(export_ci.subprocess, "run", lambda *a, **k: pytest.fail("Mismatched hardware must not start download"))
    hardware = tmp_path / "hardware"
    profile = hardware_artifact(hardware)
    if field == "git_commit":
        profile[field] = "b" * 40
    else:
        profile["ci"][field] = "2"
    (hardware / "hardware-profile.json").write_text(json.dumps(profile))
    seal(hardware)
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["export_ci.py", "--source-run-ids", "123", "--hardware", str(hardware), "--output", str(output)])
    assert export_ci.main() == 2
    error = json.loads((output / "export-error.json").read_text())
    assert error["exit_code"] == 2
    assert "Hardware" in error["error"]
