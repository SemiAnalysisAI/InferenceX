"""Per-concurrency video results joined to the native power package."""

import json
from pathlib import Path

import pytest
import yaml

from infx.results import video
from infx.config import repository_root
from infx.tests.results.power.test_aggregate_power_multinode import (
    WINDOW_END,
    WINDOW_START,
    build_package,
)

RUN = {
    "bundle_type": "mvp_run", "status": "complete", "plan_id": "h3-serving-2gpu-c124-v1",
    "plan_sha256": "f" * 64,
    "plan": {"model_id": "MiniMaxAI/MiniMax-H3", "model_revision": "4" * 40,
             "generation": {"width": 1344, "height": 768, "frame_count": 107, "fps": 24}},
    "configuration": {"runtime": "sglang", "runtime_revision": "7" * 40},
    "started_at": "2026-10-17T00:00:00Z", "finished_at": "2026-10-17T00:10:00Z",
    "measurement": {"boundary": "submit_to_downloaded_media", "concurrency": 4, "warmup_runs": 1,
                    "wall_seconds": 60.0, "start_monotonic_seconds": 10.0, "end_monotonic_seconds": 70.0},
    "serving": {"mode": "closed_loop", "concurrency": 4, "valid_video_seconds_per_second": 0.3,
                "client_ready_latency_seconds": {"p50": 118.0}},
    "records": [],
    "summary": {"scheduled": 4, "completed": 4, "valid": 4, "failed": 0, "not_started": 0,
                "technical_success_rate": 1.0, "latency_median_seconds": 119.0,
                "latency_mean_seconds": 119.2, "latency_max_seconds": 120.0,
                "valid_clips_per_second": 4 / 60},
}  # fmt: skip
BENCH = {
    "schema_version": 1, "benchmark_type": "video-generation", "max_concurrency": 4,
    "benchmark_start_time_unix": WINDOW_START, "benchmark_end_time_unix": WINDOW_END,
    "duration": WINDOW_END - WINDOW_START, "completed": 4, "scheduled": 4, "valid": 4, "failed": 0,
    "energy_normalization": "clips", "clock": {"utc_minus_monotonic_seconds": 990.0},
}  # fmt: skip
IDENTITY = {"cluster": "h200-dgxc", "model": "MiniMaxAI/MiniMax-H3", "image": "img:tag",
            "framework": "sglang", "precision": "bf16", "spec": "h3-serving-2gpu.json"}  # fmt: skip


def test_aggregate_carries_identity_throughput_and_the_measured_window():
    aggregate = video.build_aggregate(
        RUN, BENCH, concurrency=4, gpu_count=2, identity=IDENTITY,
        run_path="h3/gpu/c4/baseline/run.json", run_sha256="a" * 64,
    )
    assert aggregate["scenario_type"] == "video-generation" and aggregate["cluster"] == "h200-dgxc"
    assert (aggregate["max_concurrency"], aggregate["num_gpus"], aggregate["valid"]) == (4, 2, 4)
    assert aggregate["clips_per_hour"] == pytest.approx(240.0)
    assert aggregate["latency_median_seconds"] == 119.0
    assert aggregate["energy_normalization"] == "clips"
    assert aggregate["duration"] == WINDOW_END - WINDOW_START


@pytest.mark.parametrize(("run_edit", "bench_edit"), [
    ({"measurement": {**RUN["measurement"], "concurrency": 2}}, {}),
    ({}, {"completed": 3}),
])  # fmt: skip
def test_aggregate_refuses_a_window_that_disagrees_with_the_run(run_edit, bench_edit):
    with pytest.raises(ValueError, match="c4"):
        video.build_aggregate(
            {**RUN, **run_edit}, {**BENCH, **bench_edit}, concurrency=4, gpu_count=2,
            identity=IDENTITY, run_path="x", run_sha256="a" * 64,
        )


def _service_package(tmp_path, service="h3-server"):
    """A self-consistent native package whose GPUs are planned by the pool service."""
    pkg = build_package(tmp_path, bench_extra={**BENCH, "model_id": "MiniMaxAI/MiniMax-H3"})
    manifest = json.loads((pkg.power_dir / "manifest.json").read_text())
    for device in manifest["expected_devices"]:
        for assignment in device["assignments"]:
            assignment.update(worker_role=service, worker_index=0, worker_process=0, het_group=None)
    (pkg.power_dir / "manifest.json").write_text(json.dumps(manifest))
    return pkg


def _cell(pkg, logs_root, concurrency=4):
    cell = logs_root / "h3" / "gpu" / f"c{concurrency}"
    (cell / "baseline").mkdir(parents=True)
    (cell / "baseline/run.json").write_text(json.dumps(RUN))
    bench = cell / f"benchmark_concurrency_{concurrency}.json"
    # The package's window names the original result; make it this cell's window contract
    # (a window file is named after the result it measured, as write_window names it).
    original = json.loads(pkg.original_result.read_text())
    bench.write_text(json.dumps(original))
    [window] = pkg.windows_dir.glob("*.json")
    data = json.loads(window.read_text())
    data["result_path"] = bench.relative_to(logs_root).as_posix()
    window.unlink()
    (pkg.windows_dir / bench.name).write_text(json.dumps(data))
    manifest = json.loads((pkg.power_dir / "manifest.json").read_text())
    for validation in manifest["window_validations"]:
        validation["window_file"] = f"windows/{bench.name}"
    (pkg.power_dir / "manifest.json").write_text(json.dumps(manifest))
    return cell


def test_service_planned_gpus_validate_with_clip_normalized_energy(tmp_path):
    pkg = _service_package(tmp_path)
    _cell(pkg, pkg.logs_root)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    rc = video.run(
        logs_root=pkg.logs_root, workspace=workspace, result_filename="h3_video", concurrencies=[4],
        gpu_count=4, expected_producer_sha="a" * 40, identity=IDENTITY, require_power=True,
    )
    assert rc == 0
    aggregate = json.loads((workspace / "h3_video_conc4.json").read_text())
    sidecar = json.loads((workspace / "power_validation_h3_video_conc4.json").read_text())
    assert sidecar["power_valid"] is True and sidecar["reasons"] == []
    assert set(sidecar["per_gpu_role"].values()) == {"h3-server"}
    assert aggregate["power_valid"] == 1
    assert aggregate["joules_per_successful_query"] == pytest.approx(aggregate["total_gpu_energy_j"] / 4)
    assert "joules_per_output_token" not in aggregate
    assert aggregate["power_audit"]["source"] == "power_validation_h3_video_conc4.json"
    summary = json.loads((workspace / "h3_video.json").read_text())
    assert summary["cells"]["4"]["status"] == "complete" and summary["status"] == "complete"


@pytest.mark.parametrize("runtime", ["sglang", "vllm-omni"])
def test_wan_service_role_is_admitted_by_native_power(tmp_path, runtime):
    recipe = repository_root() / f"benchmarks/single_node/srt-slurm-recipes/wan/{runtime}/h200-bf16/serving-client.yaml"
    [service] = yaml.safe_load(recipe.read_text())["services"]
    assert service["name"] == video.SERVICE
    pkg = _service_package(tmp_path, service=service["name"])
    _cell(pkg, pkg.logs_root)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    assert video.run(
        logs_root=pkg.logs_root, workspace=workspace, result_filename="wan_video", concurrencies=[4],
        gpu_count=4, expected_producer_sha="a" * 40, identity=IDENTITY, require_power=True,
    ) == 0
    sidecar = json.loads((workspace / "power_validation_wan_video_conc4.json").read_text())
    assert sidecar["power_valid"] and set(sidecar["per_gpu_role"].values()) == {service["name"]}


def test_missing_power_never_publishes_as_measured_and_fails_only_when_required(tmp_path):
    pkg = _service_package(tmp_path)
    _cell(pkg, pkg.logs_root)
    (pkg.power_dir / "manifest.json").unlink()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    common = dict(
        logs_root=pkg.logs_root, workspace=workspace, result_filename="h3_video", concurrencies=[4],
        gpu_count=4, expected_producer_sha="a" * 40, identity=IDENTITY,
    )
    assert video.run(**common, require_power=False) == 0
    aggregate = json.loads((workspace / "h3_video_conc4.json").read_text())
    assert aggregate["power_valid"] == 0 and "total_gpu_energy_j" not in aggregate
    sidecar = json.loads((workspace / "power_validation_h3_video_conc4.json").read_text())
    assert "manifest_missing" in sidecar["reasons"]
    assert video.run(**common, require_power=True) == 1


def test_an_incomplete_cell_fails_the_point(tmp_path):
    pkg = _service_package(tmp_path)
    cell = _cell(pkg, pkg.logs_root)
    short = {**RUN, "summary": {**RUN["summary"], "valid": 3, "failed": 1}}
    (cell / "baseline/run.json").write_text(json.dumps(short))
    workspace = tmp_path / "ws"
    workspace.mkdir()
    rc = video.run(
        logs_root=pkg.logs_root, workspace=workspace, result_filename="h3_video", concurrencies=[4],
        gpu_count=4, expected_producer_sha="a" * 40, identity=IDENTITY, require_power=False,
    )
    assert rc == 1
    summary = json.loads((workspace / "h3_video.json").read_text())
    assert summary["cells"]["4"]["status"] == "missing"


def test_a_missing_run_fails_the_point_and_still_summarizes(tmp_path):
    pkg = _service_package(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    rc = video.run(
        logs_root=pkg.logs_root, workspace=workspace, result_filename="h3_video", concurrencies=[1],
        gpu_count=4, expected_producer_sha="a" * 40, identity=IDENTITY, require_power=False,
    )
    assert rc == 1
    summary = json.loads((workspace / "h3_video.json").read_text())
    assert summary["cells"]["1"]["status"] == "missing" and summary["status"] == "failed"
