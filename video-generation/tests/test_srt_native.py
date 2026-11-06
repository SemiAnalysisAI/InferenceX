import csv
import hashlib
import json
import os
from pathlib import Path

import pytest

from evaluator import srt_bundle, srt_selected
from specs import video_spec

# frozen-c124 measures one prompt four times: the four-clip cells these fixtures describe.
PLAN = video_spec.plan_for(
    video_spec.load_spec(Path(__file__).resolve().parents[1] / "specs/h3.yaml"), "frozen-c124"
)
PLAN_SHA = video_spec.plan_sha256(PLAN)


def _run(concurrency, scheduled=4, valid=4, status="complete"):
    records = []
    for repetition in range(1, PLAN["repetitions"] + 1):
        for index, case in enumerate(PLAN["cases"], 1):
            slot_id = f"measurement-r{repetition:03d}-c{index:03d}"
            records.append(
                {
                    **case,
                    "slot_id": slot_id,
                    "job_id": f"c{concurrency}-{slot_id}",
                    "repetition": repetition,
                    "phase": "measurement",
                    "attempted": True,
                    "status": "succeeded",
                    "media": {"valid": True},
                    "artifact_path": f"artifacts/{slot_id}.mp4",
                    "sha256": hashlib.sha256(b"clip").hexdigest(),
                    "timing_window": {"end_monotonic_seconds": 150.0},
                }
            )
    return {
        "bundle_type": "mvp_run",
        "status": status,
        "plan_id": PLAN["plan_id"],
        "plan_sha256": PLAN_SHA,
        "plan": PLAN,
        "configuration": {
            "runtime": "sglang",
            "runtime_revision": "7" * 40,
            "model_revision": PLAN["model_revision"],
        },
        "started_at": "2026-10-17T00:00:00Z",
        "finished_at": "2026-10-17T00:10:00Z",
        "measurement": {
            "boundary": "submit_to_downloaded_media",
            "concurrency": concurrency,
            "warmup_runs": 1,
            "wall_seconds": 60.0,
            "start_monotonic_seconds": 100.0,
            "end_monotonic_seconds": 160.0,
        },
        "serving": {
            "mode": "closed_loop",
            "concurrency": concurrency,
            "client_ready_latency_seconds": {"p50": 118.0},
        },
        "records": records[:valid],
        "summary": {
            "scheduled": scheduled,
            "completed": valid,
            "valid": valid,
            "failed": scheduled - valid,
            "not_started": 0,
            "latency_median_seconds": 119.0,
            "valid_clips_per_second": valid / 60,
        },
    }


def test_matrix_runs_every_concurrency_against_one_server_and_publishes_windows(
    tmp_path, monkeypatch
):
    seen, identities = [], []

    def fake_run_plan(plan, output, **kwargs):
        seen.append(
            (kwargs["serving_concurrency"], output, kwargs["runtime"], kwargs["runtime_revision"])
        )
        identities.append(kwargs["numerical_identity"])
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
    (logs / "video/server").mkdir(parents=True)
    launch = {
        "runtime": "sglang",
        "runtime_revision": "7" * 40,
        "model_revision": PLAN["model_revision"],
        "plan_sha256": PLAN_SHA,
        "image_digest": "c" * 64,
        "gpu_uuids": ["GPU-0", "GPU-1"],
        "argv": ["python3", "-m", "sglang"],
        "environment": {"PYTHONUNBUFFERED": "1"},
    }
    (logs / "video/server/launch.json").write_text(json.dumps(launch))
    runs = srt_selected.run_matrix(
        PLAN,
        runtime="sglang",
        runtime_revision="7" * 40,
        endpoint="http://10.0.0.1:30283",
        output=logs / "video",
        concurrencies=[1, 2],
        gpus=2,
        window_dir=windows,
        request_seconds=900,
    )

    assert [(c, r, rev) for c, _, r, rev in seen] == [
        (1, "sglang", "7" * 40),
        (2, "sglang", "7" * 40),
    ]
    assert seen[0][1] == logs / "video/gpu/c1/baseline"
    assert (
        identities[0]
        == identities[1]
        == {
            "runtime_revision": "7" * 40,
            "image_digest": "c" * 64,
            "server_argv": launch["argv"],
            "server_environment": launch["environment"],
            "gpu_topology": {"gpu_uuids": launch["gpu_uuids"]},
        }
    )
    assert sorted(runs) == [1, 2]
    contract = json.loads((logs / "video/gpu/c2/benchmark_concurrency_2.json").read_text())
    assert contract["max_concurrency"] == 2 and contract["energy_normalization"] == "clips"
    assert contract["benchmark_start_time_unix"] == pytest.approx(1_000_100.0)
    assert contract["duration"] == pytest.approx(60.0) and contract["completed"] == 4
    window = json.loads((windows / "benchmark_concurrency_2.json").read_text())
    assert window["result_path"] == "video/gpu/c2/benchmark_concurrency_2.json"
    assert (window["benchmark_type"], window["concurrency"], window["status"]) == (
        "custom",
        2,
        "completed",
    )
    assert window["benchmark_start_time_unix"] == contract["benchmark_start_time_unix"]


def _server_log(records):
    """An SGLang log whose startup and per-request lines attest each record ran the plan."""
    generation = PLAN["generation"]
    steps = generation["num_inference_steps"]
    features_off = {
        "cache_dit_config": None,
        "quantization": None,
        "component_quantizations": {},
        "component_precisions": {},
        "lora_path": None,
        "disable_conditioning_cache": True,
    }
    lines = [
        f"[10-17 00:00:00] server_args: {json.dumps(features_off)}",
        "[10-17 00:00:01] Loading H3DiT from 13 safetensors file(s) , param_dtype: torch.bfloat16",
        "[10-17 00:00:02] Using torch_sdpa attention backend",
    ]
    for record in records:
        fields = {
            "width": generation["width"],
            "height": generation["height"],
            "num_frames": generation["frame_count"],
            "fps": generation["fps"],
            "seed": record["seed"],
            "infer_steps": steps,
            "flow_shift": generation["flow_shift"],
            "output_file_path": f"outputs/{record['job_id']}.mp4",
        }
        lines += ["[10-17 00:01:00] Sampling params:", *(f"  {k}: {v}" for k, v in fields.items())]
        lines.append("\r".join(f"denoise: | {step}/{steps} [" for step in range(steps + 1)))
    return "\n".join(lines) + "\n"


def _native_job(tmp_path, concurrencies=(1, 2), power_valid=True):
    logs = tmp_path / "logs"
    video, power = logs / "video", logs / "power"
    (video / "server").mkdir(parents=True)
    # The record srt_selected.serve seals before it execs the engine.
    bound = {
        "schema_version": "0.1.0",
        "plan": PLAN,
        "serving": {"mode": "closed_loop"},
        "gpu_uuids": [
            "GPU-11111111-1111-1111-1111-111111111111",
            "GPU-22222222-2222-2222-2222-222222222222",
        ],
        "baseline": {
            "name": "sglang",
            "revision": "7" * 40,
            "model_path": "/model",
            "model_revision": PLAN["model_revision"],
            "argv": ["/usr/bin/python3"],
        },
    }
    (video / "server/spec.json").write_text(json.dumps(bound))
    (video / "server/launch.json").write_text(json.dumps({"slurm_job_id": "77", "argv": ["x"]}))
    power.mkdir()
    with (power / "samples.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "schema_version",
                "timestamp_unix",
                "scrape_seq",
                "hostname",
                "gpu_index",
                "gpu_uuid",
                "power_w",
            ]
        )
        for seq in range(70):
            for index, uuid in enumerate(bound["gpu_uuids"]):
                writer.writerow([1, 1000.0 + seq, seq, "node", index, uuid, 600.0 + index])
    job = {"job_id": "77", "state": "succeeded", "raw": "COMPLETED|0:0", "alloc_tres": None}
    (power / "native-job-status.json").write_text(json.dumps(job))
    (power / "power-producer-sha.txt").write_text("a" * 40 + "\n")
    (power / "exporter-image.sha256").write_text("sha256:" + "e" * 64 + "\n")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    served = []
    for concurrency in concurrencies:
        cell = video / "gpu" / f"c{concurrency}"
        (cell / "baseline/artifacts").mkdir(parents=True)
        run = _run(concurrency)
        served += run["records"]
        (cell / "baseline/run.json").write_text(json.dumps(run))
        for record in run["records"]:
            (cell / "baseline" / record["artifact_path"]).write_bytes(b"clip")
        (cell / f"benchmark_concurrency_{concurrency}.json").write_text(
            json.dumps(
                {
                    "max_concurrency": concurrency,
                    "benchmark_start_time_unix": 1005.0,
                    "benchmark_end_time_unix": 1065.0,
                    "duration": 60.0,
                    "completed": 4,
                    "scheduled": 4,
                    "energy_normalization": "clips",
                    "clock": {"utc_minus_monotonic_seconds": 905.0},
                }
            )
        )
        (workspace / f"r_conc{concurrency}.json").write_text(
            json.dumps({"max_concurrency": concurrency, "power_valid": int(power_valid)})
        )
        (workspace / f"power_validation_r_conc{concurrency}.json").write_text(
            json.dumps(
                {
                    "power_valid": power_valid,
                    "reasons": [] if power_valid else ["manifest_missing"],
                    "benchmark_window": {"start_time_unix": 1005.0, "end_time_unix": 1065.0},
                    "expected_gpu_count": 2,
                    "observed_gpu_count": 2,
                    "per_gpu_role": {f"node/{u}": "video-server" for u in bound["gpu_uuids"]},
                    "per_gpu_energy_j": {
                        f"node/{u}": 36000.0 + i for i, u in enumerate(bound["gpu_uuids"])
                    },
                    "per_gpu_max_sample_gap_s": {f"node/{u}": 1.0 for u in bound["gpu_uuids"]},
                    "metrics": {
                        "avg_total_gpu_power_w": 1201.0,
                        "total_gpu_energy_j": 72001.0,
                        "joules_per_successful_query": 18000.25,
                    }
                    if power_valid
                    else {},
                    "producer": {
                        "producer_git_commit": "a" * 40,
                        "exporter_image_sha256": "e" * 64,
                    },
                    "integration_method": "per_device_trapezoidal_with_linear_boundary_interpolation",
                }
            )
        )
    (logs / "service_video-server.out").write_text(_server_log(served))
    return logs, workspace


def _checksums(output):
    return dict(
        line.split("  ", 1)[::-1] for line in (output / "SHA256SUMS").read_text().splitlines()
    )


def test_bundle_joins_cells_power_and_identity_into_the_app_layout(tmp_path):
    logs, workspace = _native_job(tmp_path)
    output = tmp_path / "h3-results"
    environ = {
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_REPOSITORY": "SemiAnalysisAI/InferenceX",
        "GITHUB_SHA": "b" * 40,
        "IMAGE": "img:tag",
        "PATH": os.environ["PATH"],
        "VIDEO_ARTIFACT_NAME": "h3-results-123-2-" + "c" * 64,
    }
    result = srt_bundle.build_bundle(
        video_dir=logs / "video",
        power_dir=logs / "power",
        server_log=logs / "service_video-server.out",
        precision="bf16",
        workspace=workspace,
        result_filename="r",
        concurrencies=[1, 2],
        output=output,
        environ=environ,
    )
    assert (
        result["status"] == "complete"
        and result["power_verified"]
        and all(c["verified"] for c in result["cells"])
    )
    matrix = json.loads((output / "serving-smoke.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    ci = json.loads((output / "ci.json").read_text())
    checksums = _checksums(output)
    assert matrix["evidence_kind"] == "srt_slurm_h3_service" and matrix["gpu_uuids"] == [
        "GPU-11111111-1111-1111-1111-111111111111",
        "GPU-22222222-2222-2222-2222-222222222222",
    ]
    assert matrix["completion"]["valid"] == 8 and matrix["scheduled"] == 8
    assert (manifest["run_id"], manifest["run_attempt"], ci["run_id"]) == ("123", "2", "123")
    assert ci["phase"] == "complete"
    assert (
        manifest["git_commit"] == ci["source_sha"] and manifest["workload_plan"] == matrix["plan"]
    )
    assert manifest["artifact_name"] == ci["artifact_name"] == "h3-results-123-2-" + "c" * 64
    assert manifest["plan_sha256"] == matrix["plan_sha256"] == PLAN_SHA
    assert manifest["engine"] == matrix["engine"] == {"name": "sglang", "revision": "7" * 40}
    assert manifest["evidence"]["serving-smoke.json"] == checksums["serving-smoke.json"]
    assert (
        ci["slurm_job"] == {"job_id": "77", "state": "succeeded", "raw": "COMPLETED|0:0"}
        and ci["power"]["producer_git_commit"] == "a" * 40
    )
    for cell in matrix["cells"]:
        for ref in ("run", "receipt", "power", "attestation"):
            path = output / cell[ref]["path"]
            assert (
                hashlib.sha256(path.read_bytes()).hexdigest()
                == cell[ref]["sha256"]
                == checksums[cell[ref]["path"]]
            )
        receipt = json.loads((output / cell["receipt"]["path"]).read_text())
        assert (
            receipt["job_id"] == f"github-123-2-c{cell['concurrency']}"
            and receipt["measurement_verified"]
        )
        assert receipt["spec_sha256"] == checksums[f"gpu/c{cell['concurrency']}/spec.json"]
        spec = json.loads((output / f"gpu/c{cell['concurrency']}/spec.json").read_text())
        assert (
            spec["serving"]["concurrency"] == cell["concurrency"]
            and spec["gpu_uuids"] == matrix["gpu_uuids"]
        )
        power = json.loads((output / cell["power"]["path"]).read_text())
        assert power["valid"] and power["phases"] == cell["power"]["phases"]
        assert power["phases"]["measurement"]["aggregate"]["joules_per_valid_clip"] == 18000.25
        assert power["phases"]["measurement"]["aggregate"]["observed_peak_power_w"] == 1201.0
        assert len(power["sample_series"]) == 70 and set(power["per_gpu"]) == set(
            matrix["gpu_uuids"]
        )
    assert "gpu/c1/baseline/artifacts/measurement-r001-c001.mp4" in checksums


def test_bundle_counts_malformed_power_samples_instead_of_hiding_them(tmp_path):
    logs, workspace = _native_job(tmp_path, concurrencies=(1,))
    with (logs / "power/samples.csv").open("a", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([1, 1070.0, 70, "node", 0, "GPU-x", "n/a"])
        writer.writerow([1, 1071.0, 71, "node", 0, "GPU-x", "nan"])
    output = tmp_path / "h3-results"
    srt_bundle.build_bundle(
        video_dir=logs / "video",
        power_dir=logs / "power",
        server_log=logs / "service_video-server.out",
        precision="bf16",
        workspace=workspace,
        result_filename="r",
        concurrencies=[1],
        output=output,
        environ={"PATH": os.environ["PATH"]},
    )
    power = json.loads((output / "gpu/c1/power.json").read_text())
    assert power["malformed_sample_rows"] == 2
    assert len(power["sample_series"]) == 70


def test_power_sample_is_valid_only_when_it_reads_every_server_gpu(tmp_path):
    logs, workspace = _native_job(tmp_path, concurrencies=(1,))
    samples = logs / "power/samples.csv"
    lines = samples.read_text().splitlines(keepends=True)
    samples.write_text("".join(line for line in lines if not line.startswith("1,1005.0,5,node,1,")))
    output = tmp_path / "h3-results"
    srt_bundle.build_bundle(
        video_dir=logs / "video",
        power_dir=logs / "power",
        server_log=logs / "service_video-server.out",
        precision="bf16",
        workspace=workspace,
        result_filename="r",
        concurrencies=[1],
        output=output,
        environ={"PATH": os.environ["PATH"]},
    )
    series = json.loads((output / "gpu/c1/power.json").read_text())["sample_series"]
    assert [sample["valid"] for sample in series[4:7]] == [True, False, True]


def test_bundle_withholds_verification_when_the_engine_log_cannot_attest_the_work(tmp_path):
    logs, workspace = _native_job(tmp_path, concurrencies=(1,))
    (logs / "service_video-server.out").unlink()
    output = tmp_path / "h3-results"
    result = srt_bundle.build_bundle(
        video_dir=logs / "video",
        power_dir=logs / "power",
        server_log=logs / "service_video-server.out",
        precision="bf16",
        workspace=workspace,
        result_filename="r",
        concurrencies=[1],
        output=output,
        environ={"PATH": os.environ["PATH"]},
    )
    [cell] = json.loads((output / "serving-smoke.json").read_text())["cells"]
    receipt = json.loads((output / cell["receipt"]["path"]).read_text())
    attestation = json.loads((output / cell["attestation"]["path"]).read_text())
    assert result["status"] == "complete" and result["power_verified"] is False
    assert cell["verified"] is False and receipt["measurement_verified"] is False
    assert receipt["power_verified"] is True and receipt["attestation"]["valid"] is False
    assert attestation["valid"] is False and attestation["server_log"] == {"sha256": None}


def test_bundle_marks_a_missing_or_short_cell_failed(tmp_path):
    logs, workspace = _native_job(tmp_path, concurrencies=(1,))
    short = _run(1, valid=3)
    (logs / "video/gpu/c1/baseline/run.json").write_text(json.dumps(short))
    output = tmp_path / "h3-results"
    result = srt_bundle.build_bundle(
        video_dir=logs / "video",
        power_dir=logs / "power",
        server_log=logs / "service_video-server.out",
        precision="bf16",
        workspace=workspace,
        result_filename="r",
        concurrencies=[1, 4],
        output=output,
        environ={"PATH": os.environ["PATH"]},
    )
    assert result["status"] == "failed"
    assert json.loads((output / "ci.json").read_text())["phase"] == "failed"
    cells = {
        c["concurrency"]: c
        for c in json.loads((output / "serving-smoke.json").read_text())["cells"]
    }
    assert cells[1]["status"] == "failed" and cells[1]["completion"]["valid"] == 3
    assert cells[4]["status"] == "failed" and "run" not in cells[4]
    assert any("c4: run.json missing" in f for f in result["failures"])


def _bundle(tmp_path, artifacts, fingerprint, attempt):
    name = f"h3-results-123-{attempt}-{fingerprint}"
    fixture = tmp_path / name
    fixture.mkdir()
    logs, workspace = _native_job(fixture)
    env = {
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": attempt,
        "GITHUB_HEAD_SHA": "c" * 40,
        "GITHUB_SHA": "b" * 40,
        "RECIPE_FINGERPRINT": fingerprint,
        "VIDEO_ARTIFACT_NAME": name,
    }
    srt_bundle.build_bundle(
        video_dir=logs / "video",
        power_dir=logs / "power",
        server_log=logs / "service_video-server.out",
        precision="bf16",
        workspace=workspace,
        result_filename="r",
        concurrencies=[1, 2],
        output=artifacts / name,
        environ=env,
    )


def _sweep(tmp_path, attempts=("2", "2")):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    rows = []
    for index, (runner, attempt) in enumerate(
        zip(("cluster:h200", "cluster:b200"), attempts, strict=True), start=1
    ):
        fingerprint = str(index) * 64
        _bundle(tmp_path, artifacts, fingerprint, attempt)
        rows.append(
            {
                "recipe-fingerprint": fingerprint,
                "runner": runner,
                "framework": "sglang",
                "gpus": 2,
                "conc-list": [1, 2],
                "model": PLAN["model_id"],
            }
        )
    directory = artifacts / "video-sweep-manifest"
    directory.mkdir()
    (directory / "sweep_manifest.json").write_text(
        json.dumps(
            {
                "run-id": 123,
                "run-attempt": 2,
                "head": "c" * 40,
                "video-full-sweep": True,
                "matrix": {"single_node": {"video": rows}},
            }
        )
    )
    return artifacts


def _reseal(bundle):
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["evidence"] = {
        p.relative_to(bundle).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in bundle.rglob("*.json")
        if p != manifest_path
    }
    manifest_path.write_text(json.dumps(manifest))
    (bundle / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(bundle).as_posix()}\n"
            for p in sorted(bundle.rglob("*"))
            if p.is_file() and p.name != "SHA256SUMS"
        )
    )


def test_reuse_validates_every_selected_native_deployment_and_preserves_checkout_identity(tmp_path):
    from infx.results.video_artifacts import validate_video_artifacts

    artifacts = _sweep(tmp_path)
    assert (
        validate_video_artifacts(artifacts, run_id="123", run_attempt="2", head_sha="c" * 40) == 2
    )
    with pytest.raises(ValueError, match="trusted run/head"):
        validate_video_artifacts(artifacts, run_id="124", run_attempt="2", head_sha="c" * 40)
    with pytest.raises(ValueError, match="trusted run/head/attempt"):
        validate_video_artifacts(artifacts, run_id="123", run_attempt="3", head_sha="c" * 40)
    path = artifacts / "video-sweep-manifest/sweep_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["video-full-sweep"] = False
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="requires a full video sweep"):
        validate_video_artifacts(artifacts, run_id="123", run_attempt="2", head_sha="c" * 40)


def test_startup_failure_preserves_declared_plan_and_engine_identity(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    identity = {
        "plan_sha256": PLAN_SHA,
        "workload_plan": PLAN,
        "engine": {"name": "vllm-omni", "revision": "a" * 40},
    }
    (workspace / "video-identity.json").write_text(json.dumps(identity))
    output = tmp_path / "bundle"
    result = srt_bundle.build_bundle(
        video_dir=tmp_path / "absent",
        power_dir=tmp_path / "power",
        server_log=tmp_path / "absent.out",
        precision="bf16",
        workspace=workspace,
        result_filename="r",
        concurrencies=[1, 2],
        output=output,
    )
    manifest = json.loads((output / "manifest.json").read_text())
    matrix = json.loads((output / "serving-smoke.json").read_text())
    assert result["status"] == "failed" and result["power_verified"] is False
    assert {key: manifest[key] for key in identity} == identity
    assert matrix["engine"] == identity["engine"] and matrix["plan"] == identity["workload_plan"]
    assert "video-identity.json" in manifest["evidence"]


@pytest.mark.parametrize(
    "fault,match",
    [
        ("population", "request population"),
        ("missing_media", "reference mismatch"),
        ("power", "unverified native"),
        ("attestation", "work attestation failed"),
        ("head", "CI identity mismatch"),
    ],
)
def test_reuse_rejects_resealed_bundles_with_altered_evidence(tmp_path, fault, match):
    from infx.results.video_artifacts import validate_video_artifacts

    artifacts = _sweep(tmp_path)
    bundle = artifacts / ("h3-results-123-2-" + "2" * 64)
    matrix_path = bundle / "serving-smoke.json"
    matrix = json.loads(matrix_path.read_text())
    cell = matrix["cells"][0]
    reference = (
        None if fault == "head" else cell[fault if fault in ("power", "attestation") else "run"]
    )
    path = bundle / ("manifest.json" if reference is None else reference["path"])
    doc = json.loads(path.read_text())
    if fault == "population":
        doc["summary"].update(scheduled=1, valid=1)
        doc["records"] = doc["records"][:1]
    elif fault == "missing_media":
        doc["records"][0]["artifact_path"] = "artifacts/missing.mp4"
    elif fault in ("power", "attestation"):
        doc["valid"] = False
    else:
        doc["ci"]["head_sha"] = "d" * 40
    path.write_text(json.dumps(doc))
    if reference is not None:
        reference["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        matrix_path.write_text(json.dumps(matrix))
    _reseal(bundle)
    with pytest.raises(ValueError, match=match):
        validate_video_artifacts(artifacts, run_id="123", run_attempt="2", head_sha="c" * 40)


@pytest.mark.parametrize(
    "fault,match",
    [
        ("newest_corrupted", "checksum mismatch"),
        ("after_manifest", "after the sweep manifest"),
        ("unselected", "extra="),
    ],
)
def test_reuse_counts_only_the_newest_attempt_of_each_selected_deployment(tmp_path, fault, match):
    from infx.results.video_artifacts import validate_video_artifacts

    # Deployment 1 ran only in attempt 1; deployment 2 ran in attempts 1 and 2.
    artifacts = _sweep(tmp_path, attempts=("1", "2"))
    _bundle(tmp_path, artifacts, "2" * 64, "1")
    assert (
        validate_video_artifacts(artifacts, run_id="123", run_attempt="2", head_sha="c" * 40) == 2
    )
    if fault == "newest_corrupted":
        newest = artifacts / ("h3-results-123-2-" + "2" * 64)
        next(newest.rglob("*.mp4")).write_bytes(b"corrupted")
    elif fault == "after_manifest":
        _bundle(tmp_path, artifacts, "2" * 64, "3")
    else:
        _bundle(tmp_path, artifacts, "3" * 64, "2")
    with pytest.raises(ValueError, match=match):
        validate_video_artifacts(artifacts, run_id="123", run_attempt="2", head_sha="c" * 40)
