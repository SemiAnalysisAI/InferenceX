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


@pytest.mark.parametrize("entrypoint", ["pin", "latest", "merge", "stage"])
@pytest.mark.parametrize("fault", [None, "row", "attempt", "head", "run", "partial", "mixed", "name", "metadata"])
def test_reuse_entrypoints_validate_retained_video_bytes(sweep, monkeypatch, entrypoint, fault):
    import io
    import zipfile

    from infx import github
    from infx.workflows import merge_with_reuse, reuse, stage_results

    run = {"id": int(RUN), "run_attempt": int(ATTEMPT), "head_sha": HEAD,
           "event": "pull_request", "status": "completed", "conclusion": "success",
           "path": ".github/workflows/run-sweep.yml", "created_at": "2026-10-18T01:00:00Z"}
    if fault == "row":
        shutil.rmtree(sweep / f"h3-results-{RUN}-{ATTEMPT}-{'2' * 64}")
    elif fault in {"attempt", "head", "run"}:
        key, value = {"attempt": ("run_attempt", 3), "head": ("head_sha", "c" * 40),
                      "run": ("id", 124)}[fault]
        run[key] = value
    elif fault == "partial":
        path = sweep / "video-sweep-manifest/sweep_manifest.json"
        value = json.loads(path.read_text())
        value["video-full-sweep"] = False
        write(path, value)
    elif fault == "mixed":
        shutil.rmtree(sweep / "video-sweep-manifest")
        write(sweep / "results_bmk/result.json", [])
    elif fault == "name":
        (sweep / f"h3-results-{RUN}-{ATTEMPT}-{'2' * 64}").rename(sweep / "h3-results-malformed")
        write(sweep / "results_bmk/result.json", [])
    elif fault == "metadata":
        run.pop("run_attempt")
    artifacts, archives = [], {}
    for index, folder in enumerate(sorted(sweep.iterdir()), start=1):
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as archive:
            for path in folder.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(folder))
        raw = data.getvalue()
        artifacts.append({"id": index, "name": folder.name, "expired": False,
                          "size_in_bytes": len(raw),
                          "digest": "sha256:" + hashlib.sha256(raw).hexdigest()})
        archives[index] = raw
    artifacts.append({"id": 99, "name": "changelog-metadata", "expired": False})
    downloads = []

    def paginate(repo, path, token, item_key="", params=None):
        if path.endswith("/artifacts"):
            return artifacts
        if path.endswith("/commits"):
            return [{"sha": run["head_sha"]}]
        if path.endswith("/timeline"):
            return [{"event": "labeled", "label": {"name": "full-sweep-enabled"},
                     "created_at": "2026-10-18T00:00:00Z"}]
        if path.endswith("/runs"):
            return [run]
        raise AssertionError(path)

    def download(command, **kwargs):
        assert command[:2] == ["gh", "api"]
        assert kwargs["env"]["GH_TOKEN"] == "test-token"
        artifact_id = int(command[2].split("/")[-2])
        downloads.append(artifact_id)
        return archives[artifact_id]

    monkeypatch.setattr(github, "paginate", paginate)
    monkeypatch.setattr(github, "api", lambda *args, **kwargs: run)
    monkeypatch.setattr(github.subprocess, "check_output", download)

    def invoke():
        if entrypoint == "pin":
            reuse.validate_reusable_run("example/project", "run-sweep.yml", 7, run,
                                        "test-token", allow_failed=True)
        elif entrypoint == "latest":
            assert reuse.find_latest_successful_pr_run("example/project", "run-sweep.yml",
                       "feature", {run["head_sha"]}, "test-token") == run
        elif entrypoint == "merge":
            assert merge_with_reuse.find_eligible_run("example/project", 7, "feature", "test-token") == run["id"]
        else:
            assert stage_results.resolve_run("example/project", 7, {"head": {"ref": "feature"}},
                       "test-token", str(run["id"])) == run

    if fault:
        with pytest.raises(RuntimeError, match="video"):
            invoke()
    else:
        invoke()
        assert len(downloads) == 3


@pytest.mark.parametrize("fault", [None, "json_only", "traversal", "absolute", "duplicate", "collision", "symlink", "special", "digest"])
def test_retained_artifact_download_preserves_files_and_rejects_unsafe_zip(tmp_path, monkeypatch, fault):
    import io
    import stat
    import warnings
    import zipfile

    from infx.klaud.github import VerificationError, download_files, download_json

    output = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("manifest.json", "{}")
            archive.writestr("data.bin", b"sealed media")
            name = {"traversal": "../escape.json", "absolute": "/escape.json",
                    "duplicate": "manifest.json", "collision": "data.bin/child.json"}.get(fault)
            if name:
                archive.writestr(name, "{}")
            if fault in {"symlink", "special"}:
                member = zipfile.ZipInfo("unsafe.json")
                member.create_system = 3
                member.external_attr = ((stat.S_IFLNK if fault == "symlink" else stat.S_IFIFO) | 0o644) << 16
                archive.writestr(member, "../escape.json")
    data = output.getvalue()
    metadata = {"id": 9, "expired": False, "size_in_bytes": len(data),
                "digest": "sha256:" + ("0" * 64 if fault == "digest" else hashlib.sha256(data).hexdigest())}
    monkeypatch.setattr(sweep_runs.github.subprocess, "check_output", lambda *args, **kwargs: data)
    destination = tmp_path / "downloaded"
    if fault not in {None, "json_only"}:
        with pytest.raises(VerificationError):
            download_files("example/project", metadata, destination)
        assert not destination.exists()
    elif fault == "json_only":
        download_json("example/project", metadata, destination)
        assert [p.name for p in destination.iterdir()] == ["manifest.json"]
    else:
        download_files("example/project", metadata, destination)
        assert (destination / "data.bin").read_bytes() == b"sealed media"
        assert (destination / "manifest.json").read_text() == "{}"
