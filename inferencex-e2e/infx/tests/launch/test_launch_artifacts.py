"""Workspace staging behavior of the launcher artifact helpers."""

import json
import os
import stat
import sys
import tarfile

import pytest

from infx.launch.artifacts import (
    ArtifactError,
    bundle_server_logs,
    collect_agentic_power_results,
    copy_agentic_results,
    copy_fixed_sequence_results,
    copy_to_workspace,
)
from infx.launch.backends.base import JobState, JobStatus
from infx.launch.drivers.srt import collect
from infx.launch.drivers.srt.collect import cleanup_outputs


def test_fixed_sequence_results_use_bounded_point_names(tmp_path):
    logs, workspace = tmp_path / "logs", tmp_path / "ws"
    workspace.mkdir()
    point = logs / "cfg_isl1024_osl128"
    (point / "nested").mkdir(parents=True)
    (point / "results_concurrency_4_gpus_8.json").write_text("agg")
    (point / "nested" / "results_concurrency_16_gpus_12_ctx_2_gen_3.json").write_text("disagg")
    (logs / "unrelated").mkdir()
    (logs / "unrelated" / "results_concurrency_1_gpus_1.json").write_text("ignored")

    copy_fixed_sequence_results(logs, workspace, "stem")

    assert {path.name: path.read_text() for path in workspace.iterdir()} == {
        "stem_cfg_isl1024_osl128_conc4_gpus_8.json": "agg",
        "stem_cfg_isl1024_osl128_conc16_gpus_12_ctx_2_gen_3.json": "disagg",
    }


def test_fixed_sequence_results_reject_unparseable_point(tmp_path):
    point = tmp_path / "logs" / "isl_osl"
    point.mkdir(parents=True)
    (point / "results_concurrency_x_gpus_8.json").write_text("{}")
    with pytest.raises(ArtifactError):
        copy_fixed_sequence_results(tmp_path / "logs", tmp_path, "stem")


def test_copy_onto_same_inode_is_a_noop(tmp_path):
    source = tmp_path / "result.json"
    source.write_text("payload")
    link = tmp_path / "mounted.json"
    os.link(source, link)
    source.chmod(stat.S_IRUSR)
    copy_to_workspace(source, link)
    copy_to_workspace(source, source)
    assert link.read_text() == "payload"


def test_agentic_results_require_a_match(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "other_conc4.json").write_text("{}")
    with pytest.raises(ArtifactError):
        copy_agentic_results(tmp_path / "src", tmp_path, "point")


def test_bundle_skips_empty_logs_and_archives_contents(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    archive = tmp_path / "logs.tar.gz"
    bundle_server_logs(logs, archive)
    assert not archive.exists()
    (logs / "worker.log").write_text("x")
    bundle_server_logs(logs, archive)
    with tarfile.open(archive) as tar:
        assert "./worker.log" in tar.getnames()


def test_cleanup_outputs_retries_then_removes_nfs_files(tmp_path, monkeypatch):
    (tmp_path / "outputs" / "job").mkdir(parents=True)
    (tmp_path / "keep" / ".nfs0001").parent.mkdir()
    (tmp_path / "keep" / ".nfs0001").write_text("")
    (tmp_path / "keep" / "log").write_text("")
    real_rmtree, calls = collect.shutil.rmtree, []

    def flaky_rmtree(path):
        calls.append(path)
        if len(calls) < 3:
            raise OSError("Device or resource busy")
        real_rmtree(path)

    monkeypatch.setattr(collect.shutil, "rmtree", flaky_rmtree)
    sleeps = []
    cleanup_outputs(tmp_path, sleep=sleeps.append)
    assert len(calls) == 3 and sleeps == [10, 10]
    assert not (tmp_path / "outputs").exists()
    assert sorted(p.name for p in (tmp_path / "keep").iterdir()) == ["log"]


@pytest.mark.parametrize(("status", "rc"), [
    (JobStatus(JobState.SUCCEEDED, "COMPLETED|0:0", 0), 0),
    (JobStatus(JobState.FAILED, "FAILED|1:0", 1), 1),
])  # fmt: skip
def test_power_collection_records_the_job_status_and_stages_results_either_way(tmp_path, status, rc):
    logs, source, workspace = tmp_path / "logs", tmp_path / "src", tmp_path / "ws"
    source.mkdir()
    workspace.mkdir()
    (source / "point_conc4.json").write_text("{}")

    assert collect_agentic_power_results(
        status, "42", logs, source, workspace, "point", "sha", [4], results_python="true",
    ) == rc  # fmt: skip

    assert (logs / "power" / "native-job-status.txt").read_text() == f"42|{status.raw}\n"
    assert (workspace / "point_conc4.json").exists()


@pytest.mark.parametrize("require_power", [False, True])
@pytest.mark.parametrize("has_cpu", [False, True])
@pytest.mark.parametrize("disagg", [False, True])
def test_agentic_collection_requires_declared_cpu_and_keeps_gpu_valid(tmp_path, has_cpu, disagg, require_power):
    from infx.tests.results.power.test_aggregate_power_multinode import (
        PRODUCER_SHA,
        _cpu_rows,
        add_cpu_package,
        build_package,
    )

    pkg = build_package(tmp_path / "captured")
    result_dir = pkg.logs_root / "agentic/conc_4"
    result_dir.mkdir(parents=True)
    formal_result = result_dir / "agentic_power_concurrency_4.json"
    formal_result.write_bytes(pkg.original_result.read_bytes())
    [window_path] = pkg.windows_dir.glob("*.json")
    window = json.loads(window_path.read_text())
    window["result_path"] = str(formal_result.relative_to(pkg.logs_root))
    window_path.unlink()
    window_path = pkg.windows_dir / formal_result.name
    window_path.write_text(json.dumps(window))
    manifest_path = pkg.power_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["window_validations"][0]["window_file"] = f"windows/{formal_result.name}"
    manifest_path.write_text(json.dumps(manifest))
    if not disagg:
        manifest_path = pkg.power_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for device in manifest["expected_devices"]:
            device["assignments"][0].update(worker_role="agg", het_group=0)
        manifest_path.write_text(json.dumps(manifest))
    if has_cpu:
        add_cpu_package(pkg, _cpu_rows())
    source, workspace = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    workspace.mkdir()
    (source / "point_conc4.json").write_text(
        json.dumps(
            {
                "disagg": disagg,
                "num_prefill_gpu": 2 if disagg else 4,
                "num_decode_gpu": 2 if disagg else 0,
            }
        )
    )
    status = JobStatus(JobState.SUCCEEDED, "COMPLETED|0:0", 0)
    assert collect_agentic_power_results(
        status,
        "42",
        pkg.logs_root,
        source,
        workspace,
        "point",
        PRODUCER_SHA,
        [4],
        results_python=sys.executable,
        expected_cpu_source="acpi",
        require_power=require_power,
    ) == int(require_power and not has_cpu)
    result = json.loads((workspace / "point_conc4.json").read_text())
    assert result["power_valid"] == 1
    assert result["cpu_power_valid"] == int(has_cpu)
    assert result["total_gpu_energy_j"] == 84000.0
    assert (result_dir / "power_validation.json").is_file()
    assert result["power_audit"]["source"] == "LOGS/agentic/conc_4/power_validation.json"
    cpu = result["power_audit"]["cpu"]
    if has_cpu:
        assert cpu["sensor_kind"] == "grace_socket"
        assert cpu["source"] == "acpi"
        assert cpu["expected_sockets"] == cpu["observed_sockets"] == 4
        assert cpu["reason_codes"] == []
    else:
        assert cpu["reason_codes"] == ["cpu_artifacts_missing"]
