import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from infx.results.power.publication import build_manifest

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "docs/fixtures/powerx-manifest-v2/artifacts"


@pytest.fixture
def evidence(tmp_path):
    root = tmp_path / "artifacts"
    shutil.copytree(GOLDEN, root)
    manifest = json.loads((root / "required-power-sweep-manifest/sweep_manifest.json").read_text())
    return root, manifest


def _edit(path, **fields):
    payload = json.loads(path.read_text())
    payload.update(fields)
    path.write_text(json.dumps(payload))


def test_builds_the_shared_golden_manifest(evidence):
    root, manifest = evidence
    sweep = {key: value for key, value in manifest.items() if key not in {"schema-version", "points", "publication"}}
    assert build_manifest(sweep, root) == manifest


@pytest.mark.parametrize("file", ["bmk_agentic_golden/agg.json", "agentic_golden/power_validation.json", "agentic_golden/gpu_metrics.csv", "agentic_golden/gpu_metrics_identity.csv", "agentic_golden/power_node.txt"])
def test_missing_evidence_does_not_publish_manifest(evidence, file, tmp_path):
    root, _ = evidence
    (root / file).unlink()
    output = tmp_path / "output.json"
    result = subprocess.run([
        sys.executable, "-m", "infx.results.power.publication", "--sweep",
        str(root / "required-power-sweep-manifest/sweep_manifest.json"),
        "--artifacts", str(root), "--output", str(output),
    ], cwd=REPO, capture_output=True, text=True)
    assert result.returncode != 0
    assert not output.exists()


@pytest.mark.parametrize("energy", [0, -1, None, float("nan")])
def test_rejects_invalid_device_energy(evidence, energy):
    root, manifest = evidence
    _edit(root / "agentic_golden/power_validation.json", per_gpu_energy_j={"0": energy})
    with pytest.raises(ValueError, match="energy"):
        build_manifest(manifest, root)


def test_rejects_a_partial_expected_concurrency_set(evidence):
    root, manifest = evidence
    manifest["matrix"]["single_node"]["agentic"][0]["conc"] = [1, 8]
    with pytest.raises(ValueError, match="concurrency 8"):
        build_manifest(manifest, root)


def test_rejects_device_energy_inconsistent_with_aggregate(evidence):
    root, manifest = evidence
    _edit(root / "agentic_golden/power_validation.json", per_gpu_energy_j={"0": 500})
    with pytest.raises(ValueError, match="energy differs"):
        build_manifest(manifest, root)


@pytest.mark.parametrize("missing_role", [None, "prefill", "decode", "window", "topology", "node-count", "role-energy"])
def test_requires_both_physical_sides_of_disaggregated_deployment(evidence, missing_role):
    root, manifest = evidence
    row = manifest["matrix"]["single_node"]["agentic"].pop()
    row["disagg"] = True
    row["node-count"] = 2
    row["prefill"] = {"num-worker": 1, "tp": 1, "ep": 1}
    row["decode"] = {"num-worker": 1, "tp": 1, "ep": 1}
    manifest["matrix"]["multi_node"]["agentic"] = [row]
    _edit(root / "bmk_agentic_golden/agg.json", disagg=True, is_multinode=True,
          num_gpus=2, num_prefill_gpu=1, num_decode_gpu=1,
          prefill_tp=1, prefill_ep=1, prefill_num_workers=1,
          decode_tp=1, decode_ep=1, decode_num_workers=1,
          prefill_gpu_energy_j=400, decode_gpu_energy_j=600,
          prefill_joules_per_input_token=1, decode_joules_per_output_token=1.2)
    energies = {"prefill-node/GPU-p": 400, "decode-node/GPU-d": 600}
    roles = {"prefill-node/GPU-p": "prefill", "decode-node/GPU-d": "decode"}
    if missing_role in {"prefill", "decode"}:
        roles[f"{missing_role}-node/GPU-{missing_role[0]}"] = "aggregate"
    _edit(root / "agentic_golden/power_validation.json", observed_gpu_count=2,
          expected_gpu_count=2, per_gpu_energy_j=energies, per_gpu_role=roles,
          selected_window={"window_file": "windows/golden.json"})
    central = root / "power_audit_golden/LOGS/power"
    central.mkdir(parents=True)
    (central / "manifest.json").write_text('{"status":"complete"}')
    (central / "windows").mkdir()
    (central / "windows/golden.json").write_text(json.dumps({
        "status": "completed", "concurrency": 1,
        "benchmark_start_time_unix": 1700000000, "benchmark_end_time_unix": 1700000002,
    }))
    (central / "samples.csv").write_text("hostname,gpu_uuid,power_w\nprefill-node,GPU-p,200\ndecode-node,GPU-d,300\n")
    if missing_role == "window":
        (central / "windows/golden.json").unlink()
    elif missing_role == "topology":
        row["prefill"]["tp"] = 2
    elif missing_role == "node-count":
        row["node-count"] = 3
    elif missing_role == "role-energy":
        _edit(root / "bmk_agentic_golden/agg.json", prefill_gpu_energy_j=500)
    if missing_role:
        with pytest.raises((ValueError, FileNotFoundError)):
            build_manifest(manifest, root)
    else:
        point = build_manifest(manifest, root)["points"][0]
        assert [(device["node"], device["role"], device["energy_j"]) for device in point["devices"]] == [
            ("prefill-node", "prefill", 400), ("decode-node", "decode", 600),
        ]


def test_workflow_metadata_declares_required_intent_and_optional_compatibility(tmp_path):
    workflow = yaml.safe_load((REPO / ".github/workflows/run-sweep.yml").read_text())
    step = next(step for step in workflow["jobs"]["upload-changelog-metadata"]["steps"]
                if step.get("id") == "metadata")
    for required in (True, False):
        matrix = {"single_node": {"agentic": [{"require-power": required}]}, "changelog_metadata": {"config-keys": ["example"]}}
        result = subprocess.run(["bash", "-eo", "pipefail", "-c", step["run"]], cwd=tmp_path,
            env={**os.environ, "SWEEP_MATRIX": json.dumps(matrix), "SWEEP_HEAD": "b" * 40,
                 "SWEEP_LABELS": "[]", "FULL_SWEEP": "false", "GITHUB_RUN_ID": "123",
                 "GITHUB_RUN_ATTEMPT": "1", "GITHUB_OUTPUT": str(tmp_path / "output"),
                 "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}"},
            capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert json.loads((tmp_path / "changelog_metadata.json").read_text())["require-power"] is required


def test_amd_physical_identity_comes_from_list_payload(evidence):
    root, manifest = evidence
    (root / "agentic_golden/gpu_metrics_identity.csv").unlink()
    (root / "agentic_golden/gpu_metrics_devices.json").write_text(json.dumps({
        "gpu_data": [{"gpu": 0, "uuid": "physical-amd-0", "bdf": "0000:01:00.0"}]
    }))
    point = build_manifest(manifest, root)["points"][0]
    assert point["devices"] == [{"node": "golden-node", "gpu_uuid": "physical-amd-0", "role": "aggregate", "energy_j": 1000}]


def test_fixed_sequence_uses_declared_parallelism_and_named_audit(evidence):
    root, manifest = evidence
    row = manifest["matrix"]["single_node"]["agentic"].pop()
    row.pop("scenario-type")
    row.update(isl=1024, osl=1024)
    manifest["matrix"]["single_node"]["1k1k"] = [row]
    from infx.results.fixed_sequence import build_result
    raw = build_result(
        {"model_id": "Qwen/Qwen3.5", "max_concurrency": 1,
         "total_token_throughput": 500, "output_throughput": 250,
         "ttft_p50_ms": 200, "tpot_p50_ms": 20},
        {"RUNNER_TYPE": "h100", "FRAMEWORK": "sglang", "PRECISION": "fp8",
         "SPEC_DECODING": "none", "ISL": "1024", "OSL": "1024", "DISAGG": "false",
         "MODEL_PREFIX": "qwen3.5", "IMAGE": "example/serving:golden", "TP": "1",
         "EP_SIZE": "1", "DP_ATTENTION": "false", "RECIPE_FINGERPRINT": "a" * 64},
    )
    raw.update(power_metric_schema_version=2, power_valid=1, avg_power_w=500,
               avg_total_gpu_power_w=500, total_gpu_energy_j=1000, joules_per_output_token=2)
    (root / "bmk_agentic_golden").rename(root / "bmk_golden")
    (root / "bmk_golden/agg.json").write_text(json.dumps(raw))
    (root / "agentic_golden").rename(root / "power_audit_golden")
    (root / "power_audit_golden/power_validation.json").rename(root / "power_audit_golden/power_validation_golden.json")
    point = build_manifest(manifest, root)["points"][0]
    assert point["identity"]["benchmark_type"] == "single_turn"
    assert point["identity"]["isl"] == 1024
    assert point["topology"]["num_gpus"] == 1


@pytest.mark.parametrize("change", [None, "schema", "window", "artifact", "source", "head", "attempt"])
def test_reused_source_contract_is_checked_before_dispatch(evidence, tmp_path, change):
    root, manifest = evidence
    source = root / "required-power-sweep-manifest/sweep_manifest.json"
    if change == "schema":
        manifest.pop("schema-version")
    elif change == "window":
        manifest["points"][0]["measurement_window"]["end_time_unix"] += 1
    elif change == "artifact":
        (root / "agentic_golden/gpu_metrics.csv").write_text("changed telemetry\n")
    elif change == "source":
        manifest["run-id"] = 124
    elif change == "head":
        manifest["head"] = "c" * 40
    elif change == "attempt":
        manifest["run-attempt"] = 2
    source.write_text(json.dumps(manifest))
    output = tmp_path / "validated.json"
    result = subprocess.run([
        sys.executable, "-m", "infx.results.power.publication", "--sweep", str(source),
        "--artifacts", str(root), "--output", str(output), "--verify-existing", "--expected-run-id", "123", "--expected-head", "b" * 40,
        "--expected-run-attempt", "1",
    ], cwd=REPO, capture_output=True, text=True)
    assert result.returncode == (0 if change is None else 1), result.stderr
    assert output.exists() is (change is None)


def test_replacement_requires_an_exact_snapshot_policy(evidence):
    root, manifest = evidence
    policy = {"mode": "replacement", "replacement_scope": [{
        "curve_scope": "model-hardware-workload", "previous_snapshot_workflow_run_id": 122,
        "removed_point_identities": ["previous-recipe-concurrency-8"],
    }]}
    assert build_manifest(manifest, root, publication=policy)["publication"] == policy
    policy["replacement_scope"][0]["removed_point_identities"] = []
    with pytest.raises(ValueError, match="exact removed identities"):
        build_manifest(manifest, root, publication=policy)
