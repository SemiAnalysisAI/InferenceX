"""CPU-only matrix and receipt tests; fixture bytes are not generated media."""

import copy
import subprocess
import sys
import time
from pathlib import Path

import pytest

from evaluator import mvp_gpu_job as gpu, mvp_gpu_evidence as evidence, mvp_serving_smoke as smoke
from test_mvp_gpu_job import saved_job, spec  # noqa: F401


def saved_single(spec, directory):
    directory.parent.mkdir(parents=True, exist_ok=True)
    saved_job(spec, directory)
    receipt = gpu._read(directory / "gpu-job.json")
    receipt.update(bundle_type="controlled_serving_smoke", comparison_path=None, ci_accepted=False, release_qualified=False)
    del receipt["roles"]["candidate"]
    gpu._write(directory / "gpu-job.json", receipt)
    (directory / "baseline/events.jsonl").write_text("")
    return receipt


def test_single_runtime_smoke_cannot_be_accepted_as_paired_evidence(spec, tmp_path):
    spec["serving"] = {"concurrency": 2}
    directory = tmp_path / "single"
    saved_single(spec, directory)
    verified = evidence.verify_measurement_job(directory, deadline=time.monotonic() + 5, serving_smoke=True)
    assert set(verified["runs"]) == {"baseline"}
    assert verified["comparison"] is None
    with pytest.raises(ValueError, match="identity is not verified"):
        evidence.verify_measurement_job(directory, deadline=time.monotonic() + 5)
    (directory / "baseline/artifacts/1.mp4").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="media bytes"):
        evidence.verify_measurement_job(directory, deadline=time.monotonic() + 5, serving_smoke=True)


@pytest.mark.parametrize("fail_second", [False, True])
@pytest.mark.parametrize("requests", [4, 20])
def test_matrix_preserves_scheduled_requests_without_doubling_roles(spec, tmp_path, monkeypatch, fail_second, requests):
    spec["plan"]["cases"] = spec["plan"]["cases"][:1]
    spec["plan"]["repetitions"] = requests
    spec["serving"] = {"concurrency": 1}
    submitted = []
    def execute(current, directory, *, serving_smoke):
        assert serving_smoke is True
        submitted.append(copy.deepcopy(current))
        if fail_second and len(submitted) == 2:
            raise RuntimeError("CPU fake runtime startup failure")
        return saved_single(current, directory)
    monkeypatch.setattr(gpu, "run_gpu_job", execute)
    result = smoke.run_matrix(spec, tmp_path)
    assert result["completion"]["scheduled"] == requests * 3
    assert result["requests_per_configuration"] == requests
    assert [s["serving"]["concurrency"] for s in submitted] == ([1, 2] if fail_second else [1, 2, 4])
    assert all(s["plan"] == spec["plan"] for s in submitted)
    assert result["completion"]["valid"] == requests * (1 if fail_second else 3)
    assert result["completion"]["not_started"] == (requests * 2 if fail_second else 0)
    assert result["status"] == ("failed" if fail_second else "complete")
    assert not result["ci_accepted"]
    report = (tmp_path / "report/index.html").read_text()
    assert report.count("<video ") == requests * (1 if fail_second else 3)
    assert "../serving-smoke.json" in report


def test_single_runtime_supervisor_never_launches_candidate(spec, tmp_path, monkeypatch):
    spec["serving"] = {"concurrency": 2}
    monkeypatch.setattr(gpu, "_require_linux", lambda: None)
    monkeypatch.setattr(gpu, "GpuProbe", lambda *args: None)
    calls = []
    def role(current, label, directory, supervisor, probe, receipt):
        calls.append(label)
        receipt["roles"][label] = {"cleanup": {"status": "clean"}}
    monkeypatch.setattr(gpu, "_role", role)
    def verify(directory, *, deadline, serving_smoke):
        assert serving_smoke is True
        return {"CPU fixture verifier": True}
    monkeypatch.setattr(evidence, "verify_measurement_job", verify)
    result = gpu.run_gpu_job(spec, tmp_path / "supervised", serving_smoke=True)
    assert calls == ["baseline"]
    assert result["comparison_path"] is None
    assert result["status"] == "complete" and result["measurement_verified"] is True
    assert result["ci_accepted"] is False


def test_interrupted_client_intent_is_counted_as_unfinished(spec, tmp_path, monkeypatch):
    spec["plan"]["cases"] = spec["plan"]["cases"][:1]
    spec["plan"]["repetitions"] = 4
    spec["serving"] = {"concurrency": 1}
    def execute(current, directory, *, serving_smoke):
        receipt = saved_single(current, directory)
        run_path = directory / "baseline/run.json"
        run = gpu._read(run_path)
        run.update(records=[], status="running", finished_at=None)
        gpu._write(run_path, run)
        (directory / "baseline/events.jsonl").write_text('{"event":"attempt_started","slot_id":"measurement-r001-c001"}\n')
        return receipt
    monkeypatch.setattr(gpu, "run_gpu_job", execute)
    result = smoke.run_matrix(spec, tmp_path)
    assert result["status"] == "failed"
    assert result["completion"] == {"scheduled": 12, "attempted": 1, "completed": 0,
                                     "valid": 0, "failed": 12, "not_started": 11, "unfinished": 1}
    assert [cell["status"] for cell in result["cells"]] == ["failed", "not_started", "not_started"]


def test_login_preflight_does_not_need_staged_power_module(tmp_path):
    package = Path(smoke.__file__).resolve().parents[1]
    subprocess.run([sys.executable, "-I", "-c",
                    "import sys; sys.path.insert(0, sys.argv[1]); from evaluator.mvp_serving_smoke import validate_spec",
                    str(package)], cwd=tmp_path, check=True, capture_output=True, text=True)
