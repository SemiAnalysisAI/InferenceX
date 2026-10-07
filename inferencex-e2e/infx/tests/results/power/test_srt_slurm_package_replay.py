"""Replay packages written by the pinned srt-slurm producer through the power consumers.

srt-slurm 641a07f2 writes one ``power/{manifest.json, samples.csv, windows/*.json}``
package for every exporter kind; only ``source_metric``, ``power_scope``,
``temperature_metric`` and the exporter identity differ. The AMD package here is
written by the real ``PowerTelemetrySession`` fed fake rocm/device-metrics-exporter
scrapes, the DCGM one by the default mapping, so the consumers are checked against
the bytes a cluster ships rather than a hand-built imitation.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from infx.results import fixed_sequence
from infx.results.power import multinode
from infx.results.power.window import write_window

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "utils/srt-slurm/src"))
from srtctl.core.power.contract import (  # noqa: E402
    MANIFEST_FILENAME,
    SAMPLES_FILENAME,
    sha256_file,
)
from srtctl.core.power.manifest import ExpectedWindow  # noqa: E402
from srtctl.core.power.mapping import DCGM_EXPORTER_COMMAND_TEMPLATE  # noqa: E402
from srtctl.core.power.samples import read_samples  # noqa: E402
from srtctl.core.power.session import (  # noqa: E402
    PowerEndpoint,
    PowerSessionSettings,
    PowerTelemetrySession,
)
from srtctl.core.power.topology import build_expected_devices  # noqa: E402
from srtctl.core.power.validate_artifacts import validate_power_artifacts  # noqa: E402
from srtctl.core.schema import TelemetryExporterConfig  # noqa: E402
from srtctl.core.topology import Process  # noqa: E402

PRODUCER_SHA = "641a07f2d465847fe51d8d8db275366651d9ebef"
AMD_IMAGE = (
    "ghcr.io#semianalysisai/amd-device-metrics-exporter@sha256:"
    "db82192b0a7387bb4b2238fc2f5d0e2267ada14d996cc26061d881f1645b9bdc"
)
AMD_SCOPE = "gpu_device_power_as_reported_by_amd_device_metrics_exporter"
# The default_gpu_exporter block the MI300X/MI325X/MI355X clusters run with.
AMD_EXPORTER = TelemetryExporterConfig.Schema().load(
    {
        "container_image": AMD_IMAGE,
        "port": 19500,
        "command": "env AMD_GPU_GET_CACHE_TTL=0s /home/amd/tools/entrypoint.sh",
        "kind": "custom",
        "gpu_labels": {"index": "gpu_id", "identity": "serial_number"},
        "gpu_metrics": {
            "power": {"metric": "gpu_power_usage", "scope": AMD_SCOPE},
            "gpu_util": {"metric": "gpu_gfx_activity"},
            "temperature": {"metric": "gpu_junction_temperature"},
        },
    }
)
DCGM_EXPORTER = TelemetryExporterConfig.Schema().load(
    {"container_image": "dcgm-exporter", "port": 9401}
)

GPUS = range(4)
CONCURRENCY = 4
T0 = 1_700_000_000.0
SCRAPES = 70
RESULT_STEM = "qwen3.5_8k1k_fp8_sglang_tp4_conc4"
# A 40-request 8k1k point whose formal window sits inside the sampled span.
BENCH = {
    "model_id": "Qwen/Qwen3.5-397B-A17B-FP8",
    "max_concurrency": CONCURRENCY,
    "benchmark_start_time_unix": T0 + 5.0,
    "benchmark_end_time_unix": T0 + 65.0,
    "duration": 60.0,
    "completed": 40,
    "total_input_tokens": 40 * 8192,
    "total_output_tokens": 40 * 1024,
    "total_token_throughput": 6144.0,
    "output_throughput": 682.67,
}
# Every GPU draws a constant 500 W + index, so the 60 s window integrates exactly.
TOTAL_ENERGY_J = sum(500 + index for index in GPUS) * 60.0
AVG_POWER_W = TOTAL_ENERGY_J / 60.0 / len(GPUS)


def _amd_scrape() -> str:
    """One rocm/device-metrics-exporter body: lowercase metrics, gpu_id and serial labels."""
    lines = []
    for index in GPUS:
        labels = (
            f'gpu_id="{index}",serial_number="SN{index}",card_model="MI355X",'
            f'gpu_partition_id="NA",hostname="exporter-lies"'
        )
        lines += [
            f"gpu_power_usage{{{labels}}} {500 + index}",
            f"gpu_gfx_activity{{{labels}}} {10 * index}",
            f"gpu_junction_temperature{{{labels}}} {60 + index}",
        ]
    return "\n".join(lines) + "\n"


def _dcgm_scrape() -> str:
    lines = []
    for index in GPUS:
        labels = f'gpu="{index}",UUID="GPU-{index}",device="nvidia{index}"'
        lines += [
            f"DCGM_FI_DEV_POWER_USAGE{{{labels}}} {500 + index}",
            f"DCGM_FI_DEV_GPU_UTIL{{{labels}}} {10 * index}",
            f"DCGM_FI_PROF_SM_ACTIVE{{{labels}}} 0.5",
            f"DCGM_FI_DEV_GPU_TEMP{{{labels}}} {60 + index}",
        ]
    return "\n".join(lines) + "\n"


EXPORTERS = {
    "amd": (AMD_EXPORTER, AMD_EXPORTER.command, _amd_scrape, "gpu_junction_temperature"),
    "dcgm": (
        DCGM_EXPORTER,
        DCGM_EXPORTER_COMMAND_TEMPLATE.format(port=DCGM_EXPORTER.port),
        _dcgm_scrape,
        "DCGM_FI_DEV_GPU_TEMP",
    ),
}


class _Clock:
    """Head-node clock the session reads, advanced one second per scrape."""

    def __init__(self) -> None:
        self.now = T0

    def time(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.now


class _FakeResponse:
    status_code = 200

    def __init__(self, body: str) -> None:
        self.text = body

    def raise_for_status(self) -> None:
        return None


def _produce_package(tmp_path: Path, exporter_name: str, *, clock_sync_failures: tuple[str, ...] = ()) -> Path:
    """Run the pinned producer end to end and return the job log directory."""
    exporter, command, scrape, _ = EXPORTERS[exporter_name]
    logs = tmp_path / "LOGS"
    clock = _Clock()
    settings = PowerSessionSettings(
        power_dir=logs / "power",
        log_dir=logs,
        job_id="12345",
        run_name=RESULT_STEM,
        sample_interval_seconds=1.0,
        startup_timeout_seconds=30.0,
        request_timeout_seconds=0.5,
        collector_join_timeout_seconds=5.0,
        required=True,
        exporter_port=exporter.port,
        exporter_image=exporter.container_image,
        exporter_command=command,
        producer_git_commit=PRODUCER_SHA,
        mapping=exporter.power_mapping,
    )
    worker = Process(
        node="node-a",
        gpu_indices=frozenset(GPUS),
        sys_port=8081,
        http_port=30000,
        endpoint_mode="agg",
        endpoint_index=0,
        node_rank=0,
        het_group=None,
    )
    with patch("srtctl.core.power.session.time", clock):
        session = PowerTelemetrySession(
            settings=settings,
            expected_devices=build_expected_devices([worker]),
            expected_windows=[ExpectedWindow("custom", CONCURRENCY)],
            nodes=["node-a"],
            endpoints=[PowerEndpoint("node-a", f"http://node-a:{exporter.port}/metrics")],
        )
        session.initialize()
        session.record_clock_sync_failures(clock_sync_failures)
        with patch("srtctl.core.power.session.requests.get", return_value=_FakeResponse(scrape())):
            for second in range(SCRAPES):
                clock.now = T0 + second
                session.collect_once()
        result = logs / f"{RESULT_STEM}.json"
        result.write_text(json.dumps(BENCH))
        write_window(result, CONCURRENCY, session.windows_dir)
        clock.now = T0 + SCRAPES
        outcome = session.stop_and_finalize()
    report = validate_power_artifacts(power_dir=logs / "power", result_root=logs)
    if clock_sync_failures:
        assert not outcome.publication_valid and "clock_sync_unverified" in outcome.reason_codes
        assert not report.ok and "stored publication_valid is false" in report.failures
    else:
        assert outcome.publication_valid, outcome.reason_codes
        assert report.ok, report.failures
    return logs


def _consume(logs: Path, *, sha: str = PRODUCER_SHA, require_power: bool = True):
    """Run the multinode consumer the way the launcher does on a renamed workspace copy."""
    bench = logs.parent / "renamed_by_launcher.json"
    shutil.copy(logs / f"{RESULT_STEM}.json", bench)
    agg = logs.parent / "agg.json"
    agg.write_text(json.dumps({"hw": "mi355x"}))
    sidecar = logs.parent / "power_validation.json"
    code = multinode.run(
        logs / "power",
        bench,
        agg,
        prefill_gpus=0,
        decode_gpus=0,
        aggregate_gpus=len(GPUS),
        expected_producer_sha=sha,
        logs_root=logs,
        validation_result=sidecar,
        require_power=require_power,
    )
    return code, json.loads(agg.read_text()), json.loads(sidecar.read_text())


@pytest.mark.parametrize("exporter_name", sorted(EXPORTERS))
def test_pinned_producer_package_publishes_with_temperature(tmp_path, exporter_name):
    logs = _produce_package(tmp_path, exporter_name)
    manifest = json.loads((logs / "power" / MANIFEST_FILENAME).read_text())
    exporter, _, _, temperature_metric = EXPORTERS[exporter_name]
    assert "power_profile" not in manifest
    assert manifest["source_metric"] == exporter.power_mapping.power_metric
    assert manifest["temperature_metric"] == temperature_metric
    assert manifest["dcgm_exporter"]["container_image_resolved"] == exporter.container_image

    code, agg, sidecar = _consume(logs)

    assert code == 0, sidecar["failures"]
    assert sidecar["reasons"] == []
    assert sidecar["producer"]["producer_git_commit"] == PRODUCER_SHA
    assert agg["power_valid"] == 1
    assert agg["total_gpu_energy_j"] == pytest.approx(TOTAL_ENERGY_J)
    assert agg["avg_power_w"] == pytest.approx(AVG_POWER_W)
    assert agg["joules_per_output_token"] == pytest.approx(
        TOTAL_ENERGY_J / BENCH["total_output_tokens"]
    )
    # The App ingests samples.csv from the retained package: the bytes the producer
    # hashed are untouched and every row still carries its temperature.
    samples = logs / "power" / SAMPLES_FILENAME
    assert sha256_file(samples) == manifest["samples_sha256"]
    rows, reasons = read_samples(samples)
    assert reasons == ()
    assert len(rows) == manifest["sample_row_count"] == SCRAPES * len(GPUS)
    assert {row.gpu_index: row.temperature_c for row in rows} == {
        index: 60.0 + index for index in GPUS
    }


@pytest.mark.parametrize("require_power", [True, False])
def test_producer_pin_mismatch_blocks_publication(tmp_path, require_power):
    logs = _produce_package(tmp_path, "amd")

    code, agg, sidecar = _consume(logs, sha="b" * 40, require_power=require_power)

    assert code == int(require_power)
    assert agg["power_valid"] == 0
    assert "avg_power_w" not in agg
    assert "producer_commit_mismatch" in sidecar["reasons"]
    assert sidecar["producer"]["producer_git_commit"] == PRODUCER_SHA
    assert sidecar["producer"]["expected_producer_git_commit"] == "b" * 40


def test_single_node_result_processor_publishes_the_amd_package(tmp_path, monkeypatch):
    logs = _produce_package(tmp_path, "amd")
    monkeypatch.chdir(tmp_path)
    shutil.copy(logs / f"{RESULT_STEM}.json", tmp_path / f"{RESULT_STEM}.json")
    env = {
        "RUNNER_TYPE": "mi355x-amds",
        "FRAMEWORK": "sglang",
        "PRECISION": "fp8",
        "SPEC_DECODING": "none",
        "RESULT_FILENAME": RESULT_STEM,
        "ISL": "8192",
        "OSL": "1024",
        "DISAGG": "false",
        "MODEL_PREFIX": "qwen3.5",
        "IMAGE": "test-image",
        "TP": "4",
        "EP_SIZE": "1",
        "DP_ATTENTION": "false",
        "GPU_COUNT": "4",
        "POWER_ARTIFACT_DIR": str(logs / "power"),
        "POWER_RESULT_ROOT": str(logs),
        "POWER_PRODUCER_SHA": PRODUCER_SHA,
        "REQUIRE_POWER": "1",
    }

    assert fixed_sequence.process_result(env) == 0

    agg = json.loads((tmp_path / f"agg_{RESULT_STEM}.json").read_text())
    assert agg["is_multinode"] is False
    assert agg["tp"] == 4
    assert agg["power_valid"] == 1
    assert agg["power_invalid_reasons"] == []
    assert agg["avg_power_w"] == pytest.approx(AVG_POWER_W)
    assert agg["power_audit"]["producer_sha"] == PRODUCER_SHA
    assert agg["power_audit"]["expected_gpu_count"] == len(GPUS)
    assert agg["power_audit"]["observed_gpu_count"] == len(GPUS)


def test_clock_sync_refusal_is_named_not_a_verdict_mismatch(tmp_path):
    # h200-dgxc, 2026-10-07: the producer kept collecting but marked the package
    # unpublishable because worker-11 could not prove NTP synchronisation.
    logs = _produce_package(tmp_path, "dcgm", clock_sync_failures=("worker-11",))
    code, agg, sidecar = _consume(logs)
    assert code == 1
    assert sidecar["power_valid"] is False
    assert "producer_verdict_mismatch" not in sidecar["reasons"]
    assert "package_recompute_invalid" in sidecar["reasons"]
    assert any(failure.startswith("clock_sync_unverified: worker-11") for failure in sidecar["failures"])
    assert agg["power_valid"] == 0
    assert "avg_power_w" not in agg
