"""Native video reuse admission without a GPU or producer checkout."""

import hashlib
import json
import shutil
import sys

import pytest

from infx.workflows import sweep_runs

RUN, ATTEMPT, HEAD = "123", "2", "b" * 40


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(bundle):
    matrix = json.loads((bundle / "serving-smoke.json").read_text())
    for cell in matrix["cells"]:
        for key in ("run", "receipt", "power"):
            path = bundle / cell[key]["path"]
            if path.is_file():
                cell[key]["sha256"] = sha(path)
    write(bundle / "serving-smoke.json", matrix)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["evidence"] = {p.relative_to(bundle).as_posix(): sha(p)
                            for p in bundle.rglob("*.json") if p != manifest_path}
    write(manifest_path, manifest)
    (bundle / "SHA256SUMS").write_text("".join(
        f"{sha(p)}  {p.relative_to(bundle).as_posix()}\n"
        for p in sorted(bundle.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"))


@pytest.fixture
def sweep(tmp_path):
    case = {"case_id": "motion", "prompt": "A moving ball", "seed": 1}
    plan = {"model_id": "test/video", "model_revision": "4" * 40, "repetitions": 2, "cases": [case]}
    plan_sha = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    identity = {"plan_sha256": plan_sha, "engine": {"name": "sglang", "revision": "7" * 40}}
    rows = []
    for row_id in ("1", "2"):
        fingerprint = row_id * 64
        name = f"h3-results-{RUN}-{ATTEMPT}-{fingerprint}"
        bundle = tmp_path / name
        cells = []
        for concurrency in (1, 4):
            prefix = f"gpu/c{concurrency}"
            records = []
            for repetition in (1, 2):
                slot = f"measurement-r{repetition:03d}-c001"
                media_path = f"artifacts/{slot}.mp4"
                media = bundle / prefix / "baseline" / media_path
                media.parent.mkdir(parents=True, exist_ok=True)
                media.write_bytes(f"fixture:{row_id}:{concurrency}:{repetition}".encode())
                records.append({**case, "repetition": repetition, "slot_id": slot,
                    "phase": "measurement", "status": "succeeded", "media": {"valid": True},
                    "artifact_path": media_path, "sha256": sha(media)})
            write(bundle / prefix / "baseline/run.json", {
                "status": "complete", "plan": plan, "plan_sha256": plan_sha,
                "configuration": {"runtime": "sglang", "runtime_revision": "7" * 40},
                "summary": {"scheduled": 2, "valid": 2}, "records": records})
            write(bundle / prefix / "power.json", {"valid": True, "status": "valid"})
            write(bundle / prefix / "gpu-job.json", {"status": "complete", "measurement_verified": True, "power_verified": True})
            cells.append({"concurrency": concurrency, "status": "complete", "verified": True,
                **{key: {"path": f"{prefix}/{filename}", "sha256": sha(bundle / prefix / filename)}
                   for key, filename in (("run", "baseline/run.json"), ("receipt", "gpu-job.json"), ("power", "power.json"))}})
        write(bundle / "serving-smoke.json", {**identity, "plan": plan, "status": "complete", "power_verified": True,
                                              "cells": cells, "gpu_uuids": ["GPU-a", "GPU-b"]})
        ci = {"artifact_name": name, "run_id": RUN, "run_attempt": ATTEMPT, "ci": {"head_sha": HEAD}}
        write(bundle / "ci.json", {**ci, "source_sha": "a" * 40})
        write(bundle / "manifest.json", {**ci, **identity, "git_commit": "a" * 40,
            "recipe_fingerprint": fingerprint, "workload_plan": plan})
        seal(bundle)
        rows.append({"recipe-fingerprint": fingerprint, "framework": "sglang", "model": "test/video",
                     "gpus": 2, "conc-list": [1, 4]})
    write(tmp_path / "video-sweep-manifest/sweep_manifest.json", {
        "run-id": RUN, "run-attempt": ATTEMPT, "head": HEAD, "video-full-sweep": True,
        "matrix": {"single_node": {"video": rows}}})
    return tmp_path


def test_discovery_requires_native_bundle_and_full_sweep_manifest():
    bundle = f"h3-results-{RUN}-{ATTEMPT}-" + "1" * 64
    assert not sweep_runs.has_reusable_result_artifacts({bundle})
    assert sweep_runs.has_reusable_result_artifacts({bundle, "video-sweep-manifest"})
    assert not sweep_runs.has_reusable_result_artifacts({"h3-results-123-2", "video-sweep-manifest"})
    assert sweep_runs.has_reusable_result_artifacts({"results_bmk"})
    assert sweep_runs.has_reusable_result_artifacts({"bmk_agentic_fixture"})


def test_complete_sweep_accepts_distinct_checkout_and_workflow_head(sweep, monkeypatch, capsys):
    from infx.results.video_artifacts import validate_video_artifacts
    from infx.workflows import validate_reusable_sweep_artifacts
    assert validate_video_artifacts(sweep, run_id=RUN, run_attempt=ATTEMPT, head_sha=HEAD) == 2
    monkeypatch.setattr(sys, "argv", ["validate", "--artifacts-dir", str(sweep)])
    assert validate_reusable_sweep_artifacts.main() == 0
    assert "2 complete native video deployment(s)" in capsys.readouterr().out


@pytest.mark.parametrize("fault,match", [
    ("row", "coverage mismatch"), ("attempt", "trusted run/head/attempt"), ("partial", "full video sweep"),
    ("null_ref", "missing path or SHA256"), ("media", "reference mismatch"),
    ("population", "request population"), ("gpu", "GPU count mismatch"), ("power", "unverified native"),
])
def test_resealed_invalid_video_evidence_fails_closed(sweep, fault, match):
    from infx.results.video_artifacts import validate_video_artifacts
    bundle = sweep / f"h3-results-{RUN}-{ATTEMPT}-{'2' * 64}"
    if fault == "row":
        shutil.rmtree(bundle)
    elif fault in {"attempt", "partial"}:
        path = sweep / "video-sweep-manifest/sweep_manifest.json"
        data = json.loads(path.read_text())
        data["run-attempt" if fault == "attempt" else "video-full-sweep"] = "1" if fault == "attempt" else False
        write(path, data)
    else:
        path = bundle / "serving-smoke.json"
        matrix = json.loads(path.read_text())
        if fault == "null_ref":
            matrix["cells"][0]["receipt"] = {"path": "missing.json", "sha256": None}
        elif fault == "gpu":
            matrix["gpu_uuids"] = ["GPU-a", "GPU-a"]
        elif fault == "media":
            next(bundle.rglob("*.mp4")).write_bytes(b"changed after measurement")
        else:
            target = bundle / matrix["cells"][0]["run" if fault == "population" else "power"]["path"]
            data = json.loads(target.read_text())
            if fault == "population":
                data["summary"] = {"scheduled": 1, "valid": 1}
                data["records"] = data["records"][:1]
            else:
                data["valid"] = False
            write(target, data)
        write(path, matrix)
        seal(bundle)
    with pytest.raises(ValueError, match=match):
        validate_video_artifacts(sweep, run_id=RUN, run_attempt=ATTEMPT, head_sha=HEAD)
