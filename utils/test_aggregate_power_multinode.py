"""Tests for the multinode srt-slurm power artifact consumer.

The builder writes a fully self-consistent v1 artifact package (manifest,
samples, window, original result, renamed workspace copy). Each tamper test
breaks exactly one contract gate and asserts the package is rejected in both
modes: best-effort keeps exit 0 but publishes power_valid=0 and no energy
metrics; REQUIRE_POWER fails the run after the audit sidecar exists.
"""

import csv
import json
import shutil
from pathlib import Path

import pytest

import aggregate_power_multinode as apm

PRODUCER_SHA = "a" * 40
WINDOW_START = 1000.0
WINDOW_END = 1060.0
FIRST_TS = 998.0
LAST_TS = 1062.0
RESULT_STEM = "my_result"

# (hostname, gpu_index, role, het_group, constant power W)
DEVICES = (
    ("node-d", 0, "decode", 1, 300.0),
    ("node-d", 1, "decode", 1, 300.0),
    ("node-p", 0, "prefill", 0, 400.0),
    ("node-p", 1, "prefill", 0, 400.0),
)

BENCH_FIELDS = {
    "model_id": "test-model",
    "max_concurrency": 4,
    "benchmark_start_time_unix": WINDOW_START,
    "benchmark_end_time_unix": WINDOW_END,
    "duration": 60.0,
    "completed": 8,
    "total_input_tokens": 32768,
    "total_output_tokens": 4096,
}


class Package:
    def __init__(self, root: Path):
        self.root = root
        self.logs_root = root / "LOGS"
        self.power_dir = self.logs_root / "power"
        self.windows_dir = self.power_dir / "windows"
        self.original_result = self.logs_root / f"{RESULT_STEM}.json"
        self.bench_result = root / "renamed_by_launcher.json"
        self.agg_result = root / "agg_renamed_by_launcher.json"
        self.validation_result = root / "power_validation.json"

    def run(
        self,
        *,
        prefill_gpus=2,
        decode_gpus=2,
        aggregate_gpus=0,
        sha=PRODUCER_SHA,
        require_power=False,
    ):
        return apm.run(
            self.power_dir,
            self.bench_result,
            self.agg_result,
            prefill_gpus=prefill_gpus,
            decode_gpus=decode_gpus,
            aggregate_gpus=aggregate_gpus,
            expected_producer_sha=sha,
            logs_root=self.logs_root,
            validation_result=self.validation_result,
            require_power=require_power,
        )

    def agg(self):
        return json.loads(self.agg_result.read_text())

    def sidecar(self):
        return json.loads(self.validation_result.read_text())


def _uuid(host, idx):
    return f"GPU-{host}-{idx}"


def _rows(power_fn=None):
    rows = []
    seq = 0
    ts = FIRST_TS
    while ts <= LAST_TS:
        for host, idx, _role, _het, watts in DEVICES:
            power = power_fn(host, idx, ts) if power_fn else watts
            rows.append([1, repr(ts), seq, host, idx, _uuid(host, idx), repr(power)])
        seq += 1
        ts += 1.0
    return rows, seq


def build_package(tmp_path, power_fn=None, publication_valid=True, bench_extra=None) -> Package:
    pkg = Package(tmp_path)
    pkg.windows_dir.mkdir(parents=True)

    rows, scrapes = _rows(power_fn)
    with open(pkg.power_dir / "samples.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        # Model the producer's wire format independently of the consumer's parser.
        writer.writerow([
            "schema_version", "timestamp_unix", "scrape_seq", "hostname",
            "gpu_index", "gpu_uuid", "power_w",
        ])
        writer.writerows(rows)

    observed = [
        {
            "hostname": host,
            "gpu_index": idx,
            "gpu_uuids": [_uuid(host, idx)],
            "first_sample_time_unix": FIRST_TS,
            "last_sample_time_unix": LAST_TS,
        }
        for host, idx, _role, _het, _w in sorted(DEVICES)
    ]
    gaps = {f"{host}/{_uuid(host, idx)}": 1.0 for host, idx, _r, _h, _w in sorted(DEVICES)}
    manifest = {
        "schema_version": 1,
        "producer": "srt-slurm.dcgm-power",
        "producer_version": "1.0",
        "producer_git_commit": PRODUCER_SHA,
        "source_metric": "DCGM_FI_DEV_POWER_USAGE",
        "unit": "W",
        "power_scope": "gpu_device_board_as_reported_by_dcgm",
        "timestamp_source": "head_node_unix_clock",
        "job_id": "12345",
        "run_name": "canary",
        "sample_interval_seconds": 1.0,
        "request_timeout_seconds": 2.0,
        "max_scrape_duration_seconds": 0.05,
        "required": True,
        "started_at_unix": 990.0,
        "stopped_at_unix": 1070.0,
        "status": "complete",
        "publication_valid": publication_valid,
        "dcgm_exporter": {
            "container_image_resolved": "/squash/dcgm-exporter.sqsh",
            "container_image_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "port": 9401,
            "command": "dcgm-exporter",
        },
        "expected_devices": [
            {
                "hostname": host,
                "gpu_index": idx,
                "assignments": [
                    {
                        "worker_role": role,
                        "worker_index": 0,
                        "worker_process": 0,
                        "het_group": het,
                    }
                ],
            }
            for host, idx, role, het, _w in sorted(DEVICES)
        ],
        "observed_devices": observed,
        "expected_windows": [{"benchmark_type": "sa-bench", "concurrency": 4}],
        "scrape_count": scrapes,
        "sample_row_count": len(rows),
        "window_validations": [
            {
                "benchmark_type": "sa-bench",
                "concurrency": 4,
                "window_file": f"windows/{RESULT_STEM}.json",
                "power_coverage_valid": True,
                "reason_codes": [],
                "per_device_max_sample_gap_seconds": gaps,
            }
        ],
        "artifact_errors": [],
        "reason_codes": [],
    }
    (pkg.power_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    window = {
        "schema_version": 1,
        "clock_source": "head_node_unix_clock",
        "benchmark_type": "sa-bench",
        "concurrency": 4,
        "status": "completed",
        "benchmark_start_time_unix": WINDOW_START,
        "benchmark_end_time_unix": WINDOW_END,
        "duration": 60.0,
        "reason": None,
        "result_path": f"{RESULT_STEM}.json",
    }
    (pkg.windows_dir / f"{RESULT_STEM}.json").write_text(json.dumps(window, indent=2))

    bench_fields = dict(BENCH_FIELDS, **(bench_extra or {}))
    pkg.original_result.write_text(json.dumps(bench_fields, indent=2))
    pkg.bench_result.write_text(json.dumps(bench_fields, indent=2))
    pkg.agg_result.write_text(json.dumps({"hw": "gb200", "conc": 4}, indent=2))
    return pkg


def _edit_manifest(pkg, **changes):
    manifest = json.loads((pkg.power_dir / "manifest.json").read_text())
    manifest.update(changes)
    (pkg.power_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _rewrite_samples(pkg, mutate):
    path = pkg.power_dir / "samples.csv"
    with open(path, newline="") as handle:
        rows = list(csv.reader(handle))
    header, body = rows[0], mutate(rows[1:])
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(body)


def assert_invalid(pkg, expected_reason, **run_kwargs):
    """Both modes must reject: metrics withheld always, exit code differs."""
    assert pkg.run(require_power=False, **run_kwargs) == 0
    agg = pkg.agg()
    assert agg["power_metric_schema_version"] == 2
    assert agg["power_valid"] == 0
    for key in apm.WHOLE_METRIC_KEYS + apm.ROLE_METRIC_KEYS:
        assert key not in agg
    sidecar = pkg.sidecar()
    assert sidecar["power_valid"] is False
    assert expected_reason in sidecar["reasons"]
    assert pkg.run(require_power=True, **run_kwargs) == 1
    return sidecar


class TestValidPackage:
    def test_emits_all_metrics_exactly(self, tmp_path):
        pkg = build_package(tmp_path)
        assert pkg.run() == 0

        agg = pkg.agg()
        assert agg["power_metric_schema_version"] == 2
        assert agg["power_valid"] == 1
        assert agg["avg_power_w"] == 350.0
        assert agg["avg_total_gpu_power_w"] == 1400.0
        assert agg["total_gpu_energy_j"] == 84000.0
        assert agg["joules_per_successful_query"] == 10500.0
        assert agg["joules_per_input_token"] == 2.563477
        assert agg["joules_per_output_token"] == 20.507812
        assert agg["joules_per_total_token"] == 2.278646
        assert agg["prefill_gpu_energy_j"] == 48000.0
        assert agg["decode_gpu_energy_j"] == 36000.0
        assert agg["prefill_avg_power_w"] == 400.0
        assert agg["decode_avg_power_w"] == 300.0
        assert agg["prefill_joules_per_input_token"] == 1.464844
        assert agg["decode_joules_per_output_token"] == 8.789062

        sidecar = pkg.sidecar()
        assert sidecar["power_valid"] is True
        assert sidecar["reasons"] == []
        assert sidecar["producer"]["stored_publication_valid"] is True
        assert sidecar["producer"]["recomputed_publication_valid"] is True
        assert sidecar["selected_window"]["window_file"] == f"windows/{RESULT_STEM}.json"
        # Role, energy, and gap maps share the hostname/uuid key namespace so
        # role-level sums can be re-derived from the sidecar alone.
        assert sidecar["per_gpu_role"] == {
            "node-d/GPU-node-d-0": "decode",
            "node-d/GPU-node-d-1": "decode",
            "node-p/GPU-node-p-0": "prefill",
            "node-p/GPU-node-p-1": "prefill",
        }
        assert set(sidecar["per_gpu_energy_j"]) == set(sidecar["per_gpu_role"])

    def test_role_and_deployment_power_use_their_own_gpu_counts(self, tmp_path):
        """Role means must not divide by the whole deployment's GPU count."""

        def ramp(host, idx, ts):
            if (host, idx) == ("node-d", 0):
                return 300.0 + (ts - FIRST_TS)
            return dict(((h, i), w) for h, i, _r, _g, w in DEVICES)[(host, idx)]

        pkg = build_package(tmp_path, power_fn=ramp)
        assert pkg.run() == 0
        agg = pkg.agg()
        # Decode GPU means are 332 W and 300 W; both prefill GPUs draw 400 W.
        assert agg["prefill_avg_power_w"] == pytest.approx(400.0)
        assert agg["decode_avg_power_w"] == pytest.approx(316.0)
        assert agg["avg_total_gpu_power_w"] == pytest.approx(1432.0)
        assert agg["avg_power_w"] == pytest.approx(358.0)

    def test_strict_mode_passes_on_valid_package(self, tmp_path):
        pkg = build_package(tmp_path)
        assert pkg.run(require_power=True) == 0
        assert pkg.agg()["power_valid"] == 1

    def test_aggregate_topology_emits_only_whole_deployment_metrics(self, tmp_path):
        pkg = build_package(tmp_path)
        manifest_path = pkg.power_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for device in manifest["expected_devices"]:
            for assignment in device["assignments"]:
                assignment["worker_role"] = "agg"
                assignment["het_group"] = None
        manifest_path.write_text(json.dumps(manifest, indent=2))

        assert pkg.run(
            prefill_gpus=0,
            decode_gpus=0,
            aggregate_gpus=4,
            require_power=True,
        ) == 0

        agg = pkg.agg()
        assert agg["power_valid"] == 1
        assert agg["avg_power_w"] == 350.0
        assert agg["total_gpu_energy_j"] == 84000.0
        assert set(apm.ROLE_METRIC_KEYS).isdisjoint(agg)
        assert set(pkg.sidecar()["per_gpu_role"].values()) == {"agg"}

    def test_agentx_adapter_consumes_a_real_custom_benchmark_package(self, tmp_path):
        from utils.agentic.aggregation.power_adapter import run_multinode_agentic_power

        pkg = build_package(tmp_path)
        result_dir = pkg.logs_root / "agentic" / "conc_4"
        result_dir.mkdir(parents=True)
        stem = "agentic_power_concurrency_4"
        formal_result = result_dir / f"{stem}.json"
        pkg.original_result.replace(formal_result)

        old_window = pkg.windows_dir / f"{RESULT_STEM}.json"
        window = json.loads(old_window.read_text())
        window.update(
            {
                "benchmark_type": "custom",
                "result_path": f"agentic/conc_4/{stem}.json",
            }
        )
        old_window.unlink()
        (pkg.windows_dir / f"{stem}.json").write_text(json.dumps(window, indent=2))

        manifest_path = pkg.power_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["expected_windows"] = [
            {"benchmark_type": "custom", "concurrency": 4}
        ]
        manifest["window_validations"][0].update(
            {
                "benchmark_type": "custom",
                "window_file": f"windows/{stem}.json",
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2))
        pkg.agg_result.write_text(
            json.dumps(
                {
                    "hw": "h200",
                    "conc": 4,
                    "disagg": True,
                    "num_prefill_gpu": 2,
                    "num_decode_gpu": 2,
                }
            )
        )

        assert run_multinode_agentic_power(
            result_dir=result_dir,
            agg_result=pkg.agg_result,
            power_dir=pkg.power_dir,
            logs_root=pkg.logs_root,
            expected_producer_sha=PRODUCER_SHA,
            require_power=True,
        ) == 0

        agg = pkg.agg()
        assert agg["power_valid"] == 1
        assert agg["prefill_avg_power_w"] == 400.0
        assert agg["decode_avg_power_w"] == 300.0
        validation = json.loads((result_dir / "power_validation.json").read_text())
        assert validation["selected_window"]["window_file"] == f"windows/{stem}.json"

    def test_trapezoid_matches_hand_computed_ramp(self, tmp_path):
        # node-d/0 ramps linearly 300 -> 364 W across the samples; the
        # trapezoid over [1000, 1060] must equal the analytic integral
        # (mean of boundary powers x duration) to well under 0.1%.
        def ramp(host, idx, ts):
            if (host, idx) == ("node-d", 0):
                return 300.0 + (ts - FIRST_TS)
            return dict(((h, i), w) for h, i, _r, _g, w in DEVICES)[(host, idx)]

        pkg = build_package(tmp_path, power_fn=ramp)
        assert pkg.run() == 0
        energy = pkg.sidecar()["per_gpu_energy_j"]["node-d/GPU-node-d-0"]
        assert energy == pytest.approx(19_920.0, rel=1e-9)


class TestVerdictAndIdentityGates:
    def test_stored_verdict_false_is_verdict_mismatch(self, tmp_path):
        pkg = build_package(tmp_path, publication_valid=False)
        sidecar = assert_invalid(pkg, "producer_verdict_mismatch")
        assert sidecar["producer"]["stored_publication_valid"] is False
        assert sidecar["producer"]["recomputed_publication_valid"] is True

    def test_producer_sha_mismatch_invalid_in_both_modes(self, tmp_path):
        pkg = build_package(tmp_path)
        assert_invalid(pkg, "producer_commit_mismatch", sha="b" * 40)

    def test_missing_producer_pin_is_invalid(self, tmp_path):
        pkg = build_package(tmp_path)
        assert_invalid(pkg, "producer_pin_missing", sha=None)

    def test_missing_power_dir_is_invalid_not_crash(self, tmp_path):
        pkg = build_package(tmp_path)
        shutil.rmtree(pkg.power_dir)
        assert_invalid(pkg, "power_artifacts_missing")


class TestSampleGates:
    def test_duplicate_row_rejected(self, tmp_path):
        pkg = build_package(tmp_path)
        _rewrite_samples(pkg, lambda body: body + [body[10]])
        assert_invalid(pkg, "package_recompute_invalid")

    def test_uuid_reuse_rejected(self, tmp_path):
        pkg = build_package(tmp_path)

        def reuse(body):
            return [
                row[:5] + [_uuid("node-d", 0)] + row[6:]
                if row[3] == "node-d" and row[4] == "1"
                else row
                for row in body
            ]

        _rewrite_samples(pkg, reuse)
        sidecar = assert_invalid(pkg, "package_recompute_invalid")
        assert any("gpu_uuid_changed" in failure for failure in sidecar["failures"])

    def test_non_monotonic_timestamps_rejected(self, tmp_path):
        pkg = build_package(tmp_path)

        def swap(body):
            body = list(body)
            body[0], body[4] = (
                body[0][:1] + body[4][1:2] + body[0][2:],
                body[4][:1] + body[0][1:2] + body[4][2:],
            )
            return body

        _rewrite_samples(pkg, swap)
        sidecar = assert_invalid(pkg, "package_recompute_invalid")
        assert any("timestamp_non_monotonic" in failure for failure in sidecar["failures"])

    def test_sampling_gap_over_fixed_three_seconds_rejected(self, tmp_path):
        pkg = build_package(tmp_path)

        def drop(body):
            return [
                row
                for row in body
                if not (row[3] == "node-p" and row[4] == "0" and 20 <= int(row[2]) <= 24)
            ]

        _rewrite_samples(pkg, drop)
        sidecar = assert_invalid(pkg, "package_recompute_invalid")
        assert any("sample_gap_exceeded" in failure for failure in sidecar["failures"])

    def test_non_bracketing_device_rejected(self, tmp_path):
        pkg = build_package(tmp_path)

        def truncate(body):
            return [
                row
                for row in body
                if not (row[3] == "node-d" and row[4] == "0" and float(row[1]) <= WINDOW_START)
            ]

        _rewrite_samples(pkg, truncate)
        sidecar = assert_invalid(pkg, "package_recompute_invalid")
        assert any(
            "measurement_window_not_bracketed" in failure for failure in sidecar["failures"]
        )

    def test_overflowing_power_metric_is_invalid_not_stale(self, tmp_path):
        # 9e307 W rows are finite sample-wise, so the recompute stays valid,
        # but the trapezoid sum overflows to inf; the verdict must flip to
        # invalid instead of dying mid-patch and leaving the agg unpatched.
        pkg = build_package(tmp_path, power_fn=lambda host, idx, ts: 9e307)
        assert_invalid(pkg, "non_finite_power_metric")

        # The sidecar must stay strict RFC 8259 JSON: the overflowed per-GPU
        # energies are nulled, never serialized as bare Infinity tokens.
        def reject_constant(value):
            raise AssertionError(f"non-finite JSON constant in sidecar: {value}")

        sidecar = json.loads(
            pkg.validation_result.read_text(), parse_constant=reject_constant
        )
        assert set(sidecar["per_gpu_energy_j"].values()) == {None}


class TestWindowAndResultGates:
    def test_result_path_escape_rejected(self, tmp_path):
        pkg = build_package(tmp_path)
        window_path = pkg.windows_dir / f"{RESULT_STEM}.json"
        window = json.loads(window_path.read_text())
        window["result_path"] = "../evil.json"
        window_path.write_text(json.dumps(window))
        sidecar = assert_invalid(pkg, "package_recompute_invalid")
        assert any(
            "measurement_window_result_path_invalid" in failure
            for failure in sidecar["failures"]
        )

    def test_workspace_copy_content_mismatch_rejected(self, tmp_path):
        pkg = build_package(tmp_path)
        tampered = dict(BENCH_FIELDS, completed=9)
        pkg.bench_result.write_text(json.dumps(tampered, indent=2))
        assert_invalid(pkg, "result_content_mismatch")

    def test_no_completed_window_for_concurrency_rejected(self, tmp_path):
        pkg = build_package(tmp_path)
        tampered = dict(BENCH_FIELDS, max_concurrency=8)
        pkg.original_result.write_text(json.dumps(tampered, indent=2))
        pkg.bench_result.write_text(json.dumps(tampered, indent=2))
        sidecar = assert_invalid(pkg, "window_for_result_missing")
        # The package is still internally consistent (window start/end/duration
        # match the original result); only the result<->window binding fails.
        assert sidecar["producer"]["recomputed_publication_valid"] is True


class TestTopologyGates:
    def test_role_counts_must_match_workflow_env(self, tmp_path):
        pkg = build_package(tmp_path)
        assert_invalid(pkg, "topology_env_mismatch", prefill_gpus=4, decode_gpus=2)

    def test_aggregate_role_count_must_match_aggregate_topology(self, tmp_path):
        pkg = build_package(tmp_path)
        manifest_path = pkg.power_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for device in manifest["expected_devices"]:
            for assignment in device["assignments"]:
                assignment["worker_role"] = "agg"
                assignment["het_group"] = None
        manifest_path.write_text(json.dumps(manifest, indent=2))

        assert_invalid(
            pkg,
            "topology_env_mismatch",
            prefill_gpus=0,
            decode_gpus=0,
            aggregate_gpus=8,
        )

    def test_roles_sharing_het_group_rejected(self, tmp_path):
        pkg = build_package(tmp_path)
        manifest = json.loads((pkg.power_dir / "manifest.json").read_text())
        for device in manifest["expected_devices"]:
            for assignment in device["assignments"]:
                assignment["het_group"] = 0
        (pkg.power_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        sidecar = assert_invalid(pkg, "topology_env_mismatch")
        assert any("share a het group" in failure for failure in sidecar["failures"])

    def test_none_het_groups_valid_for_non_het_deployments(self, tmp_path):
        # Real GB200 1P1D runs launch as one plain Slurm job (no het
        # components): the producer reports het_group=None for every device
        # and the package must stay publishable in both modes.
        pkg = build_package(tmp_path)
        manifest = json.loads((pkg.power_dir / "manifest.json").read_text())
        for device in manifest["expected_devices"]:
            for assignment in device["assignments"]:
                assignment["het_group"] = None
        (pkg.power_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        assert pkg.run(require_power=True) == 0
        agg = pkg.agg()
        assert agg["power_valid"] == 1
        assert agg["prefill_gpu_energy_j"] == 48000.0
        assert agg["decode_gpu_energy_j"] == 36000.0
        assert agg["prefill_avg_power_w"] == 400.0
        assert agg["decode_avg_power_w"] == 300.0

    def test_mixed_none_and_real_het_groups_rejected(self, tmp_path):
        pkg = build_package(tmp_path)
        manifest = json.loads((pkg.power_dir / "manifest.json").read_text())
        for device in manifest["expected_devices"]:
            for assignment in device["assignments"]:
                if assignment["worker_role"] == "decode":
                    assignment["het_group"] = None
        (pkg.power_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        sidecar = assert_invalid(pkg, "topology_env_mismatch")
        assert any(
            "het group None while other devices" in failure
            for failure in sidecar["failures"]
        )


class TestManifestGates:
    def test_wire_contract_mismatch_rejected(self, tmp_path):
        pkg = build_package(tmp_path)
        _edit_manifest(pkg, producer="someone-else.power")
        assert_invalid(pkg, "package_recompute_invalid")

    def test_stored_evidence_mismatch_rejected(self, tmp_path):
        pkg = build_package(tmp_path)
        _edit_manifest(pkg, sample_row_count=1)
        sidecar = assert_invalid(pkg, "package_recompute_invalid")
        assert any("sample_row_count" in failure for failure in sidecar["failures"])

    def test_incomplete_status_rejected(self, tmp_path):
        pkg = build_package(tmp_path)
        _edit_manifest(pkg, status="incomplete", publication_valid=False)
        assert_invalid(pkg, "package_recompute_invalid")
