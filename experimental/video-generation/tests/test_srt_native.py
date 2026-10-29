import csv
import hashlib
import json
import os
from pathlib import Path

import pytest

from evaluator import srt_bundle, srt_selected
from specs import h3_spec

# frozen-c124 measures one prompt four times: the four-clip cells these fixtures describe.
PLAN = h3_spec.plan_for(h3_spec.load_spec(Path(__file__).resolve().parents[1] / "specs/h3.yaml"), "frozen-c124")
PLAN_SHA = hashlib.sha256(json.dumps(PLAN, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _run(concurrency, scheduled=4, valid=4, status="complete"):
    records = []
    for repetition in range(1, PLAN["repetitions"] + 1):
        for index, case in enumerate(PLAN["cases"], 1):
            slot_id = f"measurement-r{repetition:03d}-c{index:03d}"
            records.append({**case, "slot_id": slot_id, "repetition": repetition,
                "phase": "measurement", "attempted": True, "status": "succeeded", "media": {"valid": True},
                "artifact_path": f"artifacts/{slot_id}.mp4", "sha256": hashlib.sha256(b"clip").hexdigest(),
                "timing_window": {"end_monotonic_seconds": 150.0}})
    return {
        "bundle_type": "mvp_run", "status": status, "plan_id": PLAN["plan_id"], "plan_sha256": PLAN_SHA,
        "plan": PLAN, "configuration": {"runtime": "sglang", "runtime_revision": "7" * 40, "model_revision": PLAN["model_revision"]},
        "started_at": "2026-10-17T00:00:00Z", "finished_at": "2026-10-17T00:10:00Z",
        "measurement": {"boundary": "submit_to_downloaded_media", "concurrency": concurrency, "warmup_runs": 1,
                        "wall_seconds": 60.0, "start_monotonic_seconds": 100.0, "end_monotonic_seconds": 160.0},
        "serving": {"mode": "closed_loop", "concurrency": concurrency, "client_ready_latency_seconds": {"p50": 118.0}},
        "records": records[:valid],
        "summary": {"scheduled": scheduled, "completed": valid, "valid": valid, "failed": scheduled - valid,
                    "not_started": 0, "latency_median_seconds": 119.0, "valid_clips_per_second": valid / 60},
    }  # fmt: skip


def test_matrix_runs_every_concurrency_against_one_server_and_publishes_windows(tmp_path, monkeypatch):
    seen = []

    def fake_run_plan(plan, output, **kwargs):
        seen.append((kwargs["serving_concurrency"], output, kwargs["runtime"], kwargs["runtime_revision"]))
        output.mkdir(parents=True)
        run = _run(kwargs["serving_concurrency"])
        (output / "run.json").write_text(json.dumps(run))
        return run

    monkeypatch.setattr(srt_selected, "run_plan", fake_run_plan)
    monkeypatch.setattr(srt_selected.time, "monotonic", lambda: 50.0)
    monkeypatch.setattr(srt_selected.time, "time", lambda: 1_000_050.0)
    logs = tmp_path / "logs"
    windows = logs / "power" / "windows"
    windows.mkdir(parents=True)
    runs = srt_selected.run_matrix(
        PLAN, runtime="sglang", runtime_revision="7" * 40, endpoint="http://10.0.0.1:30283",
        output=logs / "h3", concurrencies=[1, 2], gpus=2, window_dir=windows,
        request_seconds=900,
    )

    assert [(c, r, rev) for c, _, r, rev in seen] == [(1, "sglang", "7" * 40), (2, "sglang", "7" * 40)]
    assert seen[0][1] == logs / "h3/gpu/c1/baseline"
    assert sorted(runs) == [1, 2]
    contract = json.loads((logs / "h3/gpu/c2/benchmark_concurrency_2.json").read_text())
    assert contract["max_concurrency"] == 2 and contract["energy_normalization"] == "clips"
    assert contract["benchmark_start_time_unix"] == pytest.approx(1_000_100.0)
    assert contract["duration"] == pytest.approx(60.0) and contract["completed"] == 4
    window = json.loads((windows / "benchmark_concurrency_2.json").read_text())
    assert window["result_path"] == "h3/gpu/c2/benchmark_concurrency_2.json"
    assert (window["benchmark_type"], window["concurrency"], window["status"]) == ("custom", 2, "completed")
    assert window["benchmark_start_time_unix"] == contract["benchmark_start_time_unix"]


@pytest.mark.parametrize("concurrencies", [[1, 1], [], [0]])
def test_matrix_requires_distinct_positive_concurrencies(tmp_path, concurrencies):
    with pytest.raises(ValueError, match="distinct positive"):
        srt_selected.run_matrix(
            PLAN, runtime="sglang", runtime_revision="7" * 40, endpoint="http://10.0.0.1:30283",
            output=tmp_path, concurrencies=concurrencies, gpus=2, window_dir=tmp_path,
            request_seconds=900,
        )


def test_benchmark_window_rejects_an_empty_measured_block():
    run = _run(1)
    run["measurement"]["end_monotonic_seconds"] = run["measurement"]["start_monotonic_seconds"]
    with pytest.raises(ValueError, match="positive duration"):
        srt_selected.benchmark_window(run, utc_minus_monotonic=0.0)


def _native_job(tmp_path, concurrencies=(1, 2), power_valid=True):
    logs = tmp_path / "logs"
    h3, power = logs / "h3", logs / "power"
    (h3 / "server").mkdir(parents=True)
    # The record srt-yaml-serve seals before it execs the engine.
    bound = {"schema_version": "0.1.0", "plan": PLAN, "serving": {"mode": "closed_loop"},
             "gpu_uuids": ["GPU-11111111-1111-1111-1111-111111111111", "GPU-22222222-2222-2222-2222-222222222222"],
             "baseline": {"name": "sglang", "revision": "7" * 40, "model_path": "/model",
                          "model_revision": PLAN["model_revision"], "argv": ["/usr/bin/python3"]}}
    (h3 / "server/spec.json").write_text(json.dumps(bound))
    (h3 / "server/launch.json").write_text(json.dumps({"slurm_job_id": "77", "argv": ["x"]}))
    power.mkdir()
    with (power / "samples.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["schema_version", "timestamp_unix", "scrape_seq", "hostname", "gpu_index", "gpu_uuid", "power_w"])
        for seq in range(70):
            for index, uuid in enumerate(bound["gpu_uuids"]):
                writer.writerow([1, 1000.0 + seq, seq, "node", index, uuid, 600.0 + index])
    (power / "native-job-status.txt").write_text("77|COMPLETED|0:0\n")
    (power / "power-producer-sha.txt").write_text("a" * 40 + "\n")
    (power / "exporter-image.sha256").write_text("sha256:" + "e" * 64 + "\n")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    for concurrency in concurrencies:
        cell = h3 / "gpu" / f"c{concurrency}"
        (cell / "baseline/artifacts").mkdir(parents=True)
        run = _run(concurrency)
        (cell / "baseline/run.json").write_text(json.dumps(run))
        for record in run["records"]:
            (cell / "baseline" / record["artifact_path"]).write_bytes(b"clip")
        (cell / f"benchmark_concurrency_{concurrency}.json").write_text(json.dumps({
            "max_concurrency": concurrency, "benchmark_start_time_unix": 1005.0, "benchmark_end_time_unix": 1065.0,
            "duration": 60.0, "completed": 4, "scheduled": 4, "energy_normalization": "clips",
            "clock": {"utc_minus_monotonic_seconds": 905.0}}))
        (workspace / f"r_conc{concurrency}.json").write_text(json.dumps({"max_concurrency": concurrency, "power_valid": int(power_valid)}))
        (workspace / f"power_validation_r_conc{concurrency}.json").write_text(json.dumps({
            "power_valid": power_valid, "reasons": [] if power_valid else ["manifest_missing"],
            "benchmark_window": {"start_time_unix": 1005.0, "end_time_unix": 1065.0},
            "expected_gpu_count": 2, "observed_gpu_count": 2,
            "per_gpu_role": {f"node/{u}": "h3-server" for u in bound["gpu_uuids"]},
            "per_gpu_energy_j": {f"node/{u}": 36000.0 + i for i, u in enumerate(bound["gpu_uuids"])},
            "per_gpu_max_sample_gap_s": {f"node/{u}": 1.0 for u in bound["gpu_uuids"]},
            "metrics": {"avg_total_gpu_power_w": 1201.0, "total_gpu_energy_j": 72001.0,
                        "joules_per_successful_query": 18000.25} if power_valid else {},
            "producer": {"producer_git_commit": "a" * 40, "exporter_image_sha256": "e" * 64},
            "integration_method": "per_device_trapezoidal_with_linear_boundary_interpolation",
        }))
    return logs, workspace


def _checksums(output):
    return dict(line.split("  ", 1)[::-1] for line in (output / "SHA256SUMS").read_text().splitlines())


def test_bundle_joins_cells_power_and_identity_into_the_app_layout(tmp_path):
    logs, workspace = _native_job(tmp_path)
    output = tmp_path / "h3-results"
    environ = {"GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "2", "GITHUB_REPOSITORY": "SemiAnalysisAI/InferenceX",
               "GITHUB_SHA": "b" * 40, "IMAGE": "img:tag", "PATH": os.environ["PATH"],
               "VIDEO_ARTIFACT_NAME": "h3-results-123-2-" + "c" * 64}
    result = srt_bundle.build_bundle(h3_dir=logs / "h3", power_dir=logs / "power", workspace=workspace,
                                         result_filename="r", concurrencies=[1, 2], output=output, environ=environ)
    assert result["status"] == "complete" and result["power_verified"] and all(c["verified"] for c in result["cells"])
    matrix = json.loads((output / "serving-smoke.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    ci = json.loads((output / "ci.json").read_text())
    checksums = _checksums(output)
    assert matrix["evidence_kind"] == "srt_slurm_h3_service" and matrix["gpu_uuids"] == ["GPU-11111111-1111-1111-1111-111111111111", "GPU-22222222-2222-2222-2222-222222222222"]
    assert matrix["completion"]["valid"] == 8 and matrix["scheduled"] == 8
    assert (manifest["run_id"], manifest["run_attempt"], ci["run_id"]) == ("123", "2", "123")
    assert manifest["git_commit"] == ci["source_sha"] and manifest["workload_plan"] == matrix["plan"]
    assert manifest["artifact_name"] == ci["artifact_name"] == "h3-results-123-2-" + "c" * 64
    assert manifest["plan_sha256"] == matrix["plan_sha256"] == PLAN_SHA
    assert manifest["engine"] == matrix["engine"] == {"name": "sglang", "revision": "7" * 40}
    assert manifest["evidence"]["serving-smoke.json"] == checksums["serving-smoke.json"]
    assert ci["slurm_job"] == {"job_id": "77", "state": "COMPLETED"} and ci["power"]["producer_git_commit"] == "a" * 40
    for cell in matrix["cells"]:
        for ref in ("run", "receipt", "power"):
            path = output / cell[ref]["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == cell[ref]["sha256"] == checksums[cell[ref]["path"]]
        receipt = json.loads((output / cell["receipt"]["path"]).read_text())
        assert receipt["job_id"] == f"github-123-2-c{cell['concurrency']}" and receipt["measurement_verified"]
        assert receipt["spec_sha256"] == checksums[f"gpu/c{cell['concurrency']}/spec.json"]
        spec = json.loads((output / f"gpu/c{cell['concurrency']}/spec.json").read_text())
        assert spec["serving"]["concurrency"] == cell["concurrency"] and spec["gpu_uuids"] == matrix["gpu_uuids"]
        power = json.loads((output / cell["power"]["path"]).read_text())
        assert power["valid"] and power["phases"] == cell["power"]["phases"]
        assert power["phases"]["measurement"]["aggregate"]["joules_per_valid_clip"] == 18000.25
        assert power["phases"]["measurement"]["aggregate"]["observed_peak_power_w"] == 1201.0
        assert len(power["sample_series"]) == 70 and set(power["per_gpu"]) == set(matrix["gpu_uuids"])
    assert "gpu/c1/baseline/artifacts/measurement-r001-c001.mp4" in checksums


def test_bundle_seals_observed_slurm_allocation(tmp_path):
    logs, workspace = _native_job(tmp_path, concurrencies=(1,))
    allocation = "cpu=16,gres/gpu=8,mem=128G,node=1"
    (logs / "power/native-job-status.txt").write_text(f"77|COMPLETED|0:0|{allocation}\n")
    output = tmp_path / "h3-results"

    srt_bundle.build_bundle(
        h3_dir=logs / "h3", power_dir=logs / "power", workspace=workspace,
        result_filename="r", concurrencies=[1], output=output, environ={"PATH": os.environ["PATH"]},
    )

    ci = json.loads((output / "ci.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    checksums = _checksums(output)
    assert ci["slurm_job"] == {"job_id": "77", "state": "COMPLETED", "AllocTRES": allocation}
    assert manifest["evidence"]["ci.json"] == checksums["ci.json"]
    assert hashlib.sha256((output / "ci.json").read_bytes()).hexdigest() == checksums["ci.json"]


def test_bundle_reports_invalid_power_without_verifying_the_cell(tmp_path):
    logs, workspace = _native_job(tmp_path, concurrencies=(1,), power_valid=False)
    output = tmp_path / "h3-results"
    result = srt_bundle.build_bundle(h3_dir=logs / "h3", power_dir=logs / "power", workspace=workspace,
                                         result_filename="r", concurrencies=[1], output=output, environ={"PATH": os.environ["PATH"]})
    assert result["status"] == "complete" and result["power_verified"] is False
    [cell] = json.loads((output / "serving-smoke.json").read_text())["cells"]
    assert cell["status"] == "complete" and cell["verified"] is False
    power = json.loads((output / cell["power"]["path"]).read_text())
    assert power["valid"] is False and power["invalid_reasons"] == ["manifest_missing"]
    assert power["phases"]["measurement"]["aggregate"] is None


def test_bundle_marks_a_missing_or_short_cell_failed(tmp_path):
    logs, workspace = _native_job(tmp_path, concurrencies=(1,))
    short = _run(1, valid=3)
    (logs / "h3/gpu/c1/baseline/run.json").write_text(json.dumps(short))
    output = tmp_path / "h3-results"
    result = srt_bundle.build_bundle(h3_dir=logs / "h3", power_dir=logs / "power", workspace=workspace,
                                         result_filename="r", concurrencies=[1, 4], output=output, environ={"PATH": os.environ["PATH"]})
    assert result["status"] == "failed"
    cells = {c["concurrency"]: c for c in json.loads((output / "serving-smoke.json").read_text())["cells"]}
    assert cells[1]["status"] == "failed" and cells[1]["completion"]["valid"] == 3
    assert cells[4]["status"] == "failed" and "run" not in cells[4]
    assert any("c4: run.json missing" in f for f in result["failures"])


@pytest.mark.parametrize("field,value", [("plan_sha256", "a" * 64),
                                        ("configuration", {"runtime": "other", "runtime_revision": "9" * 40})])
def test_bundle_rejects_cells_with_different_workload_or_engine(tmp_path, field, value):
    logs, workspace = _native_job(tmp_path)
    path = logs / "h3/gpu/c2/baseline/run.json"
    run = json.loads(path.read_text())
    run[field] = value
    path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match="identity differs"):
        srt_bundle.build_bundle(h3_dir=logs / "h3", power_dir=logs / "power", workspace=workspace,
                                   result_filename="r", concurrencies=[1, 2], output=tmp_path / "bundle")


def _sweep(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    rows = []
    for index, runner in enumerate(("cluster:h200", "cluster:b200"), start=1):
        fixture = tmp_path / runner.split(":")[1]
        fixture.mkdir()
        logs, workspace = _native_job(fixture)
        fingerprint = str(index) * 64
        name = f"h3-results-123-2-{fingerprint}"
        env = {"GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "2", "GITHUB_HEAD_SHA": "c" * 40,
               "GITHUB_SHA": "b" * 40, "RECIPE_FINGERPRINT": fingerprint, "VIDEO_ARTIFACT_NAME": name}
        srt_bundle.build_bundle(h3_dir=logs / "h3", power_dir=logs / "power", workspace=workspace,
                                   result_filename="r", concurrencies=[1, 2], output=artifacts / name, environ=env)
        rows.append({"recipe-fingerprint": fingerprint, "runner": runner, "framework": "sglang",
                     "gpus": 2, "conc-list": [1, 2], "model": PLAN["model_id"]})
    directory = artifacts / "video-sweep-manifest"
    directory.mkdir()
    (directory / "sweep_manifest.json").write_text(json.dumps({"run-id": 123, "run-attempt": 2,
        "head": "c" * 40, "video-full-sweep": True, "matrix": {"single_node": {"video": rows}}}))
    return artifacts


def test_reuse_validates_every_selected_native_deployment_and_preserves_checkout_identity(tmp_path):
    from infx.results.video_artifacts import validate_video_artifacts
    artifacts = _sweep(tmp_path)
    assert validate_video_artifacts(artifacts, run_id="123", run_attempt="2", head_sha="c" * 40) == 2
    with pytest.raises(ValueError, match="trusted run/head"):
        validate_video_artifacts(artifacts, run_id="124", run_attempt="2", head_sha="c" * 40)
    with pytest.raises(ValueError, match="trusted run/head/attempt"):
        validate_video_artifacts(artifacts, run_id="123", run_attempt="3", head_sha="c" * 40)


@pytest.mark.parametrize("fault,match", [("row", "coverage mismatch"), ("media", "checksum mismatch"),
                                        ("cell", "concurrency coverage mismatch"), ("head", "CI identity mismatch")])
def test_reuse_rejects_incomplete_or_altered_native_sweeps(tmp_path, fault, match):
    import shutil
    from infx.results.video_artifacts import validate_video_artifacts
    artifacts = _sweep(tmp_path)
    bundle = artifacts / ("h3-results-123-2-" + "2" * 64)
    if fault == "row":
        shutil.rmtree(bundle)
    elif fault == "media":
        media = next(bundle.rglob("*.mp4"))
        media.write_bytes(b"corrupted")
    else:
        path = bundle / ("serving-smoke.json" if fault == "cell" else "manifest.json")
        data = json.loads(path.read_text())
        if fault == "cell":
            data["cells"].pop()
        else:
            data["ci"]["head_sha"] = "d" * 40
        path.write_text(json.dumps(data))
        manifest_path = bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["evidence"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        # The manifest does not include itself in evidence.
        manifest["evidence"].pop("manifest.json", None)
        manifest_path.write_text(json.dumps(manifest))
        (bundle / "SHA256SUMS").write_text("".join(
            f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(bundle).as_posix()}\n"
            for p in sorted(bundle.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"))
    with pytest.raises(ValueError, match=match):
        validate_video_artifacts(artifacts, run_id="123", run_attempt="2", head_sha="c" * 40)


def test_startup_failure_preserves_declared_plan_and_engine_identity(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    identity = {"plan_sha256": PLAN_SHA, "workload_plan": PLAN,
                "engine": {"name": "vllm-omni", "revision": "a" * 40}}
    (workspace / "video-identity.json").write_text(json.dumps(identity))
    output = tmp_path / "bundle"
    result = srt_bundle.build_bundle(h3_dir=tmp_path / "absent", power_dir=tmp_path / "power",
        workspace=workspace, result_filename="r", concurrencies=[1, 2], output=output)
    manifest = json.loads((output / "manifest.json").read_text())
    matrix = json.loads((output / "serving-smoke.json").read_text())
    assert result["status"] == "failed" and result["power_verified"] is False
    assert {key: manifest[key] for key in identity} == identity
    assert matrix["engine"] == identity["engine"] and matrix["plan"] == identity["workload_plan"]
    assert "video-identity.json" in manifest["evidence"]


@pytest.mark.parametrize("fault,match", [("power", "unverified native"), ("plan", "selected workload")])
def test_reuse_rechecks_sealed_receipts_and_selected_workload(tmp_path, fault, match):
    from infx.results.video_artifacts import validate_video_artifacts
    artifacts = _sweep(tmp_path)
    bundle = artifacts / ("h3-results-123-2-" + "2" * 64)
    matrix_path = bundle / "serving-smoke.json"
    matrix = json.loads(matrix_path.read_text())
    cell = matrix["cells"][0]
    field = "power" if fault == "power" else "run"
    path = bundle / cell[field]["path"]
    value = json.loads(path.read_text())
    if fault == "power":
        value["valid"] = False
    else:
        value["plan"]["model_id"] = "another/model"
    path.write_text(json.dumps(value))
    cell[field]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    matrix_path.write_text(json.dumps(matrix))
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["evidence"] = {p.relative_to(bundle).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in bundle.rglob("*.json") if p != manifest_path}
    manifest_path.write_text(json.dumps(manifest))
    (bundle / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(bundle).as_posix()}\n"
        for p in sorted(bundle.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"))
    with pytest.raises(ValueError, match=match):
        validate_video_artifacts(artifacts, run_id="123", run_attempt="2", head_sha="c" * 40)


@pytest.mark.parametrize("fault,match", [
    ("null_ref", "missing path or SHA256"), ("missing_ref", "reference mismatch"),
    ("population", "request population"), ("missing_media", "reference mismatch"),
    ("duplicate_gpu", "GPU count mismatch"), ("duplicate_slot", "request population"),
])
def test_reuse_rejects_resealed_missing_evidence_and_wrong_population(tmp_path, fault, match):
    from infx.results.video_artifacts import validate_video_artifacts
    artifacts = _sweep(tmp_path)
    bundle = artifacts / ("h3-results-123-2-" + "2" * 64)
    matrix_path = bundle / "serving-smoke.json"
    matrix = json.loads(matrix_path.read_text())
    cell = matrix["cells"][0]
    run_path = bundle / cell["run"]["path"]
    run = json.loads(run_path.read_text())
    if fault in {"null_ref", "missing_ref"}:
        cell["receipt"] = {"path": "missing.json", "sha256": None if fault == "null_ref" else "a" * 64}
    elif fault == "duplicate_gpu":
        matrix["gpu_uuids"] = [matrix["gpu_uuids"][0]] * 2
    else:
        if fault == "population":
            run["summary"].update(scheduled=1, valid=1)
            run["records"] = run["records"][:1]
        elif fault == "duplicate_slot":
            run["records"][1] = run["records"][0]
        else:
            run["records"][0]["artifact_path"] = "artifacts/missing.mp4"
        run_path.write_text(json.dumps(run))
        cell["run"]["sha256"] = hashlib.sha256(run_path.read_bytes()).hexdigest()
    matrix_path.write_text(json.dumps(matrix))
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["evidence"] = {p.relative_to(bundle).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in bundle.rglob("*.json") if p != manifest_path}
    manifest_path.write_text(json.dumps(manifest))
    (bundle / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(bundle).as_posix()}\n"
        for p in sorted(bundle.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"))
    with pytest.raises(ValueError, match=match):
        validate_video_artifacts(artifacts, run_id="123", run_attempt="2", head_sha="c" * 40)


def test_partial_native_sweep_cannot_qualify_for_reuse(tmp_path):
    from infx.results.video_artifacts import validate_video_artifacts
    artifacts = _sweep(tmp_path)
    path = artifacts / "video-sweep-manifest/sweep_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["video-full-sweep"] = False
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="requires a full video sweep"):
        validate_video_artifacts(artifacts, run_id="123", run_attempt="2", head_sha="c" * 40)
