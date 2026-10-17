"""The native srt-slurm path: per-concurrency client cells, server record, result bundle."""

import csv
import hashlib
import json
import os
from pathlib import Path

import pytest

from evaluator import mvp_srt, mvp_srt_bundle

SPEC = json.loads((Path(__file__).resolve().parents[1] / "specs/h3-serving-2gpu.json").read_text())


def _run(concurrency, scheduled=4, valid=4, status="complete"):
    return {
        "bundle_type": "mvp_run", "status": status, "plan_id": SPEC["plan"]["plan_id"], "plan_sha256": "f" * 64,
        "plan": SPEC["plan"], "configuration": {"runtime": "sglang", "runtime_revision": "7" * 40},
        "started_at": "2026-10-17T00:00:00Z", "finished_at": "2026-10-17T00:10:00Z",
        "measurement": {"boundary": "submit_to_downloaded_media", "concurrency": concurrency, "warmup_runs": 1,
                        "wall_seconds": 60.0, "start_monotonic_seconds": 100.0, "end_monotonic_seconds": 160.0},
        "serving": {"mode": "closed_loop", "concurrency": concurrency, "client_ready_latency_seconds": {"p50": 118.0}},
        "records": [{"phase": "measurement", "attempted": True, "timing_window": {"end_monotonic_seconds": 150.0}}] * valid,
        "summary": {"scheduled": scheduled, "completed": valid, "valid": valid, "failed": scheduled - valid,
                    "not_started": 0, "latency_median_seconds": 119.0, "valid_clips_per_second": valid / 60},
    }  # fmt: skip


def test_matrix_runs_every_concurrency_against_one_server_and_publishes_windows(tmp_path, monkeypatch):
    seen = []

    def fake_run_plan(plan, output, **kwargs):
        seen.append((kwargs["serving_concurrency"], output))
        output.mkdir(parents=True)
        run = _run(kwargs["serving_concurrency"])
        (output / "run.json").write_text(json.dumps(run))
        return run

    monkeypatch.setattr(mvp_srt, "run_plan", fake_run_plan)
    monkeypatch.setattr(mvp_srt.time, "monotonic", lambda: 50.0)
    monkeypatch.setattr(mvp_srt.time, "time", lambda: 1_000_050.0)
    logs = tmp_path / "logs"
    windows = logs / "power" / "windows"
    windows.mkdir(parents=True)
    runs = mvp_srt.run_srt_matrix(SPEC, "http://10.0.0.1:30283", logs / "h3", [1, 2], window_dir=windows)

    assert [c for c, _ in seen] == [1, 2]
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


def test_matrix_requires_distinct_concurrencies():
    with pytest.raises(ValueError, match="distinct"):
        mvp_srt.run_srt_matrix(SPEC, "http://10.0.0.1:30283", Path("/nonexistent"), [1, 1], window_dir=None)


def test_benchmark_window_rejects_an_empty_measured_block():
    run = _run(1)
    run["measurement"]["end_monotonic_seconds"] = run["measurement"]["start_monotonic_seconds"]
    with pytest.raises(ValueError, match="positive duration"):
        mvp_srt.benchmark_window(run, utc_minus_monotonic=0.0)


def test_serve_role_binds_the_host_and_records_the_bound_spec(tmp_path, monkeypatch):
    launched = {}

    class FakeProcess:
        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    def fake_popen(argv, env):
        launched["argv"] = argv
        launched["env"] = env
        return FakeProcess()

    monkeypatch.setattr(mvp_srt, "cuda_devices", lambda: ["GPU-11111111-1111-1111-1111-111111111111", "GPU-22222222-2222-2222-2222-222222222222"])
    monkeypatch.setattr(mvp_srt.subprocess, "Popen", fake_popen)
    monkeypatch.setenv("SLURM_JOB_ID", "77")
    record = tmp_path / "server"
    assert mvp_srt.serve_role(SPEC, "baseline", 30283, host="0.0.0.0", record=record) == 0
    argv = launched["argv"]
    assert argv[argv.index("--host") + 1] == "0.0.0.0" and argv[argv.index("--num-gpus") + 1] == "2"
    # The launch preamble reads this before importing SGLang (Slurm job 100550 died without it).
    assert launched["env"]["VGBENCH_GPU_UUIDS"] == (
        "GPU-11111111-1111-1111-1111-111111111111,GPU-22222222-2222-2222-2222-222222222222"
    )
    bound = json.loads((record / "spec.json").read_text())
    assert bound["gpu_uuids"] == ["GPU-11111111-1111-1111-1111-111111111111", "GPU-22222222-2222-2222-2222-222222222222"] and bound["port"] == 30283
    launch = json.loads((record / "launch.json").read_text())
    assert launch["slurm_job_id"] == "77" and launch["argv"] == argv


def _native_job(tmp_path, concurrencies=(1, 2), power_valid=True):
    logs = tmp_path / "logs"
    h3, power = logs / "h3", logs / "power"
    (h3 / "server").mkdir(parents=True)
    bound = {**SPEC, "gpu_uuids": ["GPU-11111111-1111-1111-1111-111111111111", "GPU-22222222-2222-2222-2222-222222222222"]}
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
        (cell / "baseline/run.json").write_text(json.dumps(_run(concurrency)))
        (cell / "baseline/artifacts/measurement-001.mp4").write_bytes(b"clip")
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
               "GITHUB_SHA": "b" * 40, "IMAGE": "img:tag", "PATH": os.environ["PATH"]}
    result = mvp_srt_bundle.build_bundle(h3_dir=logs / "h3", power_dir=logs / "power", workspace=workspace,
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
    assert "gpu/c1/baseline/artifacts/measurement-001.mp4" in checksums


def test_bundle_reports_invalid_power_without_verifying_the_cell(tmp_path):
    logs, workspace = _native_job(tmp_path, concurrencies=(1,), power_valid=False)
    output = tmp_path / "h3-results"
    result = mvp_srt_bundle.build_bundle(h3_dir=logs / "h3", power_dir=logs / "power", workspace=workspace,
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
    result = mvp_srt_bundle.build_bundle(h3_dir=logs / "h3", power_dir=logs / "power", workspace=workspace,
                                         result_filename="r", concurrencies=[1, 4], output=output, environ={"PATH": os.environ["PATH"]})
    assert result["status"] == "failed"
    cells = {c["concurrency"]: c for c in json.loads((output / "serving-smoke.json").read_text())["cells"]}
    assert cells[1]["status"] == "failed" and cells[1]["completion"]["valid"] == 3
    assert cells[4]["status"] == "failed" and "run" not in cells[4]
    assert any("c4: run.json missing" in f for f in result["failures"])
