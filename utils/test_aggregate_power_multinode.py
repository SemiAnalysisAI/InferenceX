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

from infx.results.power import multinode as apm

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
        assert agg["p75_power_w"] == 350.0
        assert agg["p75_total_gpu_power_w"] == 1400.0
        assert agg["p90_power_w"] == 350.0
        assert agg["p90_total_gpu_power_w"] == 1400.0
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
        assert agg["p75_total_gpu_power_w"] == pytest.approx(1447.0)
        assert agg["p75_power_w"] == pytest.approx(361.75)
        assert agg["p90_total_gpu_power_w"] == pytest.approx(1456.0)
        assert agg["p90_power_w"] == pytest.approx(364.0)


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
        from infx.results.agentic.power_adapter import run_multinode_agentic_power

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


@pytest.mark.parametrize("utilization", [("", ""), ("75.5", "0.9")])
def test_v2_samples_preserve_energy(tmp_path, utilization):
    """The pinned producer's optional utilization columns preserve board energy."""
    pkg = build_package(tmp_path)
    path = pkg.power_dir / "samples.csv"
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(rows[0] + ["gpu_util_pct", "sm_active"])
        writer.writerows([[2, *row[1:], *utilization] for row in rows[1:]])
    assert pkg.run(require_power=True) == 0
    assert pkg.agg()["power_valid"] == 1
    assert pkg.agg()["total_gpu_energy_j"] == pytest.approx(84000)


@pytest.mark.parametrize("row", [
    [1, 1, 0, "node", 0, "GPU-0", 300, "", ""],
    [2, 1, 0, "node", 0, "GPU-0", 300, "nan", ""],
    [2, 1, 0, "node", 0, "GPU-0", 300, 101, ""],
    [2, 1, 0, "node", 0, "GPU-0", 300, 50, 1.1],
    [2, 1, 0, "node", 0, "GPU-0", 300],
])
def test_v2_samples_reject_mixed_versions_and_invalid_utilization(tmp_path, row):
    path = tmp_path / "samples.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["schema_version", "timestamp_unix", "scrape_seq", "hostname",
                         "gpu_index", "gpu_uuid", "power_w", "gpu_util_pct", "sm_active"])
        writer.writerow(row)
    rows, reasons = apm.read_samples(path)
    assert not rows
    assert reasons == ("samples_csv_malformed",)


# --- Grace CPU-side power leg (power/cpu/) -----------------------------------

CPU_HOSTS = tuple(sorted({host for host, *_ in DEVICES}))
CPU_HEADER_V1 = [
    "schema_version", "timestamp_unix", "hostname", "source", "sensor",
    "socket_id", "power_w", "total_power_w",
]
CPU_HEADER_V2 = CPU_HEADER_V1[:7] + ["cpu_rail_w", "soc_w", "dram_w", "total_power_w"]
GRACE_W = {"node-d": 500.0, "node-p": 700.0}
MODULE_W = 1500.0
DCGM_W = 300.0
# Component rails never feed a headline metric; the values are chosen so any
# leak into a published key is visible against the Grace/module constants.
RAILS_W = {"cpu_rail": 200.0, "soc": 50.0, "dram": 30.0}
GRACE_KINDS = {"grace": None, **RAILS_W}

_SENSORS = {
    # (source, v1 firmware OEM label, v2 collector sensor name)
    "grace": ("acpi", "Grace Power Socket {s}", "CPU{s}:cpuSidePowerUsageW"),
    "module": ("acpi", "Module Power Socket {s}", "Module Power Socket {s}"),
    "dcgm": ("dcgm", "CPU{s}:cpuPowerUsageW", "CPU{s}:cpuPowerUsageW"),
    "cpu_rail": ("acpi", "CPU Power Socket {s}", "CPU{s}:cpuRailPowerUsageW"),
    "soc": ("acpi", "SysIO Power Socket {s}", "CPU{s}:socPowerUsageW"),
    "dram": ("acpi", "DRAM Power Socket {s}", "CPU{s}:dramPowerUsageW"),
}


def _cpu_row(ts, host, socket, kind, watts, fmt):
    source, label_v1, label_v2 = _SENSORS[kind]
    sensor = (label_v1 if fmt == "v1" else label_v2).format(s=socket)
    if fmt == "v1":
        return [1, repr(ts), host, source, sensor, socket, repr(watts), ""]
    rails = ["", "", ""] if source == "dcgm" else [repr(w) for w in RAILS_W.values()]
    return [2, repr(ts), host, source, sensor, socket, repr(watts), *rails, ""]


def _cpu_rows(kinds_by_host=None, fmt="v1", *, hosts=CPU_HOSTS):
    """One row per (scrape, host, socket, kind); ``None`` watts means the Grace constant."""
    kinds_by_host = kinds_by_host or {host: GRACE_KINDS for host in hosts}
    rows = []
    ts = FIRST_TS
    while ts <= LAST_TS:
        for host in hosts:
            for socket in (0, 1):
                for kind, watts in kinds_by_host.get(host, {}).items():
                    watts = GRACE_W[host] if watts is None else watts
                    rows.append(_cpu_row(ts, host, socket, kind, watts, fmt))
        ts += 1.0
    return rows


def add_cpu_package(pkg, rows, header=CPU_HEADER_V1, manifest_text=None):
    cpu_dir = pkg.power_dir / "cpu"
    cpu_dir.mkdir()
    with open(cpu_dir / "samples.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    if manifest_text is None:
        manifest_text = json.dumps(
            {
                "schema_version": 1,
                "producer": "srt-slurm.cpu-power",
                "producer_git_commit": PRODUCER_SHA,
                "started_at_unix": 990.0,
                "stopped_at_unix": 1070.0,
                "nodes": {
                    host: {"resolved_mode": "acpi", "scrape_count": 65, "error_count": 0}
                    for host in CPU_HOSTS
                },
            },
            indent=2,
        )
    (cpu_dir / "cpu_manifest.json").write_text(manifest_text)
    return cpu_dir


def _gpu_fields(agg):
    return {k: v for k, v in agg.items() if k != "cpu_power_valid" and k not in apm.CPU_METRIC_KEYS}


def _reference_agg(tmp_path):
    """The GPU-only aggregate every CPU-leg outcome must reproduce field for field."""
    reference = build_package(tmp_path / "reference")
    assert reference.run() == 0
    return reference.agg()


def assert_grace_keys(agg):
    assert agg["cpu_power_valid"] == 1
    assert agg["avg_cpu_socket_power_w"] == 600.0
    assert agg["avg_total_cpu_power_w"] == 2400.0
    assert agg["total_cpu_energy_j"] == 144000.0


class TestCpuSidePower:
    @pytest.mark.parametrize("fmt, header", [("v1", CPU_HEADER_V1), ("v2", CPU_HEADER_V2)])
    def test_grace_total_emits_cpu_keys_and_provenance(self, tmp_path, fmt, header):
        from infx.results.power.audit import audit_summary

        pkg = build_package(tmp_path)
        kinds = None if fmt == "v1" else {host: {"grace": None} for host in CPU_HOSTS}
        rows = _cpu_rows(kinds, fmt)
        add_cpu_package(pkg, rows, header)
        assert pkg.run(require_power=True) == 0

        agg = pkg.agg()
        assert_grace_keys(agg)
        assert "avg_total_module_power_w" not in agg
        assert "total_module_energy_j" not in agg
        assert _gpu_fields(agg) == _reference_agg(tmp_path)

        cpu = pkg.sidecar()["cpu"]
        assert cpu["cpu_power_valid"] is True
        assert cpu["reason_codes"] == []
        assert cpu["sensor_kind"] == "grace_socket"
        assert cpu["source"] == "acpi"
        assert cpu["expected_sockets"] == 4
        assert cpu["observed_sockets"] == 4
        assert cpu["sample_row_count"] == len(rows)
        assert set(cpu["per_series_energy_j"]) == {
            f"{host}/socket{socket}/grace_socket" for host in CPU_HOSTS for socket in (0, 1)
        }
        assert audit_summary(pkg.sidecar(), "power_validation.json")["power_audit"]["cpu"] == {
            "sensor_kind": "grace_socket",
            "source": "acpi",
            "expected_sockets": 4,
            "observed_sockets": 4,
            "sample_row_count": len(rows),
            "reason_codes": [],
        }

    def test_dcgm_only_package_uses_cpu_rail_kind(self, tmp_path):
        pkg = build_package(tmp_path)
        add_cpu_package(pkg, _cpu_rows({host: {"dcgm": DCGM_W} for host in CPU_HOSTS}))
        assert pkg.run() == 0
        agg = pkg.agg()
        assert agg["cpu_power_valid"] == 1
        assert agg["avg_cpu_socket_power_w"] == 300.0
        assert agg["avg_total_cpu_power_w"] == 1200.0
        assert agg["total_cpu_energy_j"] == 72000.0
        cpu = pkg.sidecar()["cpu"]
        assert (cpu["sensor_kind"], cpu["source"]) == ("dcgm_cpu_rail", "dcgm")

    def test_module_on_every_socket_is_preferred_and_grace_keys_stay_grace(self, tmp_path):
        pkg = build_package(tmp_path)
        add_cpu_package(
            pkg, _cpu_rows({host: {**GRACE_KINDS, "module": MODULE_W} for host in CPU_HOSTS})
        )
        assert pkg.run() == 0
        agg = pkg.agg()
        assert_grace_keys(agg)
        assert agg["avg_total_module_power_w"] == 6000.0
        assert agg["total_module_energy_j"] == 360000.0
        cpu = pkg.sidecar()["cpu"]
        assert cpu["sensor_kind"] == "module"
        assert set(cpu["per_series_energy_j"]) == {
            f"{host}/socket{socket}/{kind}"
            for host in CPU_HOSTS
            for socket in (0, 1)
            for kind in ("module", "grace_socket")
        }

    def test_module_on_some_sockets_falls_back_to_grace(self, tmp_path):
        pkg = build_package(tmp_path)
        kinds = {"node-d": {**GRACE_KINDS, "module": MODULE_W}, "node-p": GRACE_KINDS}
        add_cpu_package(pkg, _cpu_rows(kinds))
        assert pkg.run() == 0
        agg = pkg.agg()
        assert_grace_keys(agg)
        assert "avg_total_module_power_w" not in agg
        assert "total_module_energy_j" not in agg
        assert pkg.sidecar()["cpu"]["sensor_kind"] == "grace_socket"

    def test_package_without_cpu_dir_emits_no_cpu_fields(self, tmp_path):
        pkg = build_package(tmp_path)
        assert pkg.run() == 0
        agg = pkg.agg()
        assert "cpu_power_valid" not in agg
        assert set(apm.CPU_METRIC_KEYS).isdisjoint(agg)
        assert "cpu" not in pkg.sidecar()

    def test_stale_cpu_keys_are_stripped_on_rerun(self, tmp_path):
        pkg = build_package(tmp_path)
        pkg.agg_result.write_text(
            json.dumps({"hw": "gb200", "conc": 4, "cpu_power_valid": 1, "total_cpu_energy_j": 1.0})
        )
        assert pkg.run() == 0
        assert _gpu_fields(pkg.agg()) == _reference_agg(tmp_path)
        assert "cpu_power_valid" not in pkg.agg()
        assert "total_cpu_energy_j" not in pkg.agg()

    def _drop(self, rows, host, socket, predicate):
        return [
            row
            for row in rows
            if not (row[2] == host and row[5] == socket and predicate(float(row[1])))
        ]

    @pytest.fixture
    def tampered(self, request, tmp_path):
        kind = request.param
        pkg = build_package(tmp_path)
        rows = _cpu_rows()
        header, manifest_text = CPU_HEADER_V1, None
        if kind == "header":
            header = [*CPU_HEADER_V1[:6], "watts", "total_power_w"]
        elif kind == "socket":
            rows = self._drop(rows, "node-p", 1, lambda ts: True)
        elif kind == "gap":
            rows = self._drop(rows, "node-d", 0, lambda ts: 1020.0 <= ts <= 1024.0)
        elif kind == "unbracketed":
            rows = self._drop(rows, "node-d", 0, lambda ts: ts <= WINDOW_START)
        elif kind == "mixed":
            rows = _cpu_rows({"node-d": {"grace": None}, "node-p": {"dcgm": DCGM_W}})
        elif kind == "manifest":
            manifest_text = "{broken"
        elif kind == "malformed":
            rows = [*rows, [1, repr(FIRST_TS), "node-d", "acpi", "Grace Power Socket 0", 0, "n/a", ""]]
        cpu_dir = add_cpu_package(pkg, rows, header, manifest_text)
        if kind == "samples_missing":
            (cpu_dir / "samples.csv").unlink()
        return pkg

    @pytest.mark.parametrize(
        "tampered, reason",
        [
            ("header", "cpu_samples_header_mismatch"),
            ("socket", "cpu_socket_count_mismatch"),
            ("gap", "cpu_sample_gap_exceeded"),
            ("unbracketed", "cpu_window_not_bracketed"),
            ("mixed", "cpu_sensor_kind_mixed"),
            ("manifest", "cpu_manifest_invalid"),
            ("malformed", "cpu_samples_malformed"),
            ("samples_missing", "cpu_samples_missing"),
        ],
        indirect=["tampered"],
    )
    def test_cpu_leg_failures_leave_gpu_fields_untouched(self, tmp_path, tampered, reason):
        pkg = tampered
        # REQUIRE_POWER guards the GPU leg only; a broken CPU leg never fails the run.
        assert pkg.run(require_power=True) == 0
        agg = pkg.agg()
        assert agg["cpu_power_valid"] == 0
        assert set(apm.CPU_METRIC_KEYS).isdisjoint(agg)
        assert _gpu_fields(agg) == _reference_agg(tmp_path)
        cpu = pkg.sidecar()["cpu"]
        assert cpu["cpu_power_valid"] is False
        assert reason in cpu["reason_codes"]
        assert pkg.sidecar()["power_valid"] is True

    @pytest.mark.parametrize(
        "gpu_gap, sha, gpu_reason",
        [
            (True, PRODUCER_SHA, "package_recompute_invalid"),
            (False, "b" * 40, "producer_commit_mismatch"),
            (False, None, "producer_pin_missing"),
        ],
    )
    def test_cpu_leg_survives_an_invalid_gpu_leg(self, tmp_path, gpu_gap, sha, gpu_reason):
        """No GPU verdict, the producer pin included, reaches cpu_power_valid."""
        pkg = build_package(tmp_path)
        add_cpu_package(pkg, _cpu_rows())
        if gpu_gap:
            _rewrite_samples(
                pkg,
                lambda body: [
                    row
                    for row in body
                    if not (row[3] == "node-p" and row[4] == "0" and 20 <= int(row[2]) <= 24)
                ],
            )
        assert pkg.run(sha=sha) == 0
        agg = pkg.agg()
        assert agg["power_valid"] == 0
        assert "total_gpu_energy_j" not in agg
        assert_grace_keys(agg)
        sidecar = pkg.sidecar()
        assert gpu_reason in sidecar["reasons"]
        assert sidecar["cpu"]["reason_codes"] == []

    def test_window_unavailable_when_result_binds_to_no_window(self, tmp_path):
        pkg = build_package(tmp_path)
        add_cpu_package(pkg, _cpu_rows())
        tampered = dict(BENCH_FIELDS, max_concurrency=8)
        pkg.original_result.write_text(json.dumps(tampered, indent=2))
        pkg.bench_result.write_text(json.dumps(tampered, indent=2))
        assert pkg.run() == 0
        assert pkg.agg()["cpu_power_valid"] == 0
        assert "cpu_window_unavailable" in pkg.sidecar()["cpu"]["reason_codes"]
