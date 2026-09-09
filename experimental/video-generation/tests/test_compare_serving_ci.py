import json
from pathlib import Path
import shutil

import pytest

import ci
import compare_serving_ci as fidelity


SHA = "a" * 40


def source(root, run_id="123"):
    run = root / "gpu/c1/baseline/run.json"
    run.parent.mkdir(parents=True)
    ci.write(run, {"configuration": {"serving": {"concurrency": 1}}})
    ci.write(root / "ci.json", {"run_id": run_id, "run_attempt": "1", "source_sha": SHA})
    ci.write(root / "manifest.json", {"run_id": run_id, "run_attempt": "1", "git_commit": SHA,
             "mode": "serving-smoke"})
    ci.write(root / "serving-smoke.json", {"bundle_type": "h3_serving_smoke_matrix", "schema_version": "1.0.0",
             "cells": [{"concurrency": 1, "run": {"path": "gpu/c1/baseline/run.json", "sha256": ci.digest(run)}}]})
    seal(root)
    return {"databaseId": int(run_id), "runAttempt": 1, "headSha": SHA}


def seal(root):
    (root / "SHA256SUMS").write_text("".join(f"{sha}  {path}\n" for path, sha in ci.inventory(root).items()))


def test_selects_sealed_c1_and_rejects_tampered_bytes_or_ci_identity(tmp_path):
    metadata = source(tmp_path)
    selected = fidelity.selected_run(tmp_path, metadata)
    assert selected == tmp_path / "gpu/c1/baseline"
    with pytest.raises(ValueError, match="CI identity"):
        fidelity.selected_run(tmp_path, {**metadata, "headSha": "b" * 40})
    (selected / "run.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        fidelity.selected_run(tmp_path, metadata)


@pytest.mark.parametrize("mutation", ["non_c1", "duplicate_c1", "escape"])
def test_rejects_wrong_or_ambiguous_cell_even_when_outer_seal_is_updated(tmp_path, mutation):
    metadata = source(tmp_path)
    path = tmp_path / "serving-smoke.json"
    matrix = ci.read(path)
    if mutation == "non_c1":
        run = tmp_path / "gpu/c1/baseline/run.json"
        ci.write(run, {"configuration": {"serving": {"concurrency": 2}}})
        matrix["cells"][0]["run"]["sha256"] = ci.digest(run)
    elif mutation == "duplicate_c1":
        matrix["cells"].append(matrix["cells"][0])
    else:
        matrix["cells"][0]["run"]["path"] = "../outside/run.json"
    ci.write(path, matrix)
    seal(tmp_path)
    with pytest.raises(ValueError):
        fidelity.selected_run(tmp_path, metadata)


def test_publication_retains_uncalibrated_failure_and_original_seals(tmp_path, monkeypatch):
    originals = tmp_path / "originals"
    metadata = {run: source(originals / run, run) for run in ("123", "456")}
    monkeypatch.setenv("GITHUB_SHA", SHA)
    monkeypatch.setenv("GITHUB_RUN_ID", "789")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setenv("GITHUB_REPOSITORY", "SemiAnalysisAI/InferenceX")
    monkeypatch.setattr(ci, "command", lambda *args: SHA)
    monkeypatch.setattr(fidelity.export_ci, "verified_execution", lambda run: (metadata[run], {"name": "original"}))
    def download(argv, **kwargs):
        assert argv[:3] == ["gh", "run", "download"]
        shutil.copytree(originals / argv[3], Path(argv[argv.index("--dir") + 1]))
    monkeypatch.setattr(fidelity.subprocess, "run", download)
    def compare(left, right, *, policy):
        assert left.parts[-4:] == ("source-123", "gpu", "c1", "baseline")
        assert right.parts[-4:] == ("source-456", "gpu", "c1", "baseline")
        assert policy["calibration_status"] == "uncalibrated"
        return {"overall_status": "fail", "release_qualified": False, "policy": policy,
                "summary": {"matched_valid_pairs": 20}}
    monkeypatch.setattr(fidelity, "compare_runs", compare)
    def report(result, path):
        assert path.suffix == ".html" and result["overall_status"] == "fail"
        path.parent.mkdir()
        path.write_text("fixture report")
    monkeypatch.setattr(fidelity, "write_report", report)
    output = tmp_path / "published"
    fidelity.publish(["123", "456"], output)
    assert (output / "source-123/SHA256SUMS").read_bytes() == (originals / "123/SHA256SUMS").read_bytes()
    result = json.loads((output / "comparison.json").read_text())
    assert result["source_artifacts"][1]["ci"]["databaseId"] == 456
    receipt = ci.read(output / "reprocessing.json")
    assert receipt["status"] == "complete" and receipt["threshold_outcome"] == "fail"
    assert receipt["release_qualified"] is False and receipt["generation_executed"] is False
