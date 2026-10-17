import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from infx.results.power.publication import build_manifest

REPO = Path(__file__).resolve().parents[2]
GOLDEN = Path(__file__).resolve().parent / "fixtures/powerx-manifest-v2"


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
    row["node-count"] = 3
    row["compute-node-count"] = 2
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
    shutil.copyfile(root / "agentic_golden/power_validation.json", central.parent / "power_validation.json")
    shutil.copyfile(root / "agentic_golden/power_source.json", central.parent.parent / "power_source.json")
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
        row["compute-node-count"] = 3
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


@pytest.fixture(params=["single", "aggregate", "disaggregated"])
def native_evidence(tmp_path, monkeypatch, request):
    from infx.tests.results.power import test_aggregate_power_multinode as native

    shape = request.param
    devices = (
        (("node-a", 0, "agg", 0, 500.0), ("node-a", 1, "agg", 0, 500.0))
        if shape != "disaggregated" else
        (("node-p", 0, "prefill", 0, 400.0), ("node-d", 0, "decode", 1, 600.0))
    )
    monkeypatch.setattr(native, "DEVICES", devices)
    root = tmp_path / "native"
    evidence = root / "power_audit_native"
    package = native.build_package(evidence)
    result = json.loads((GOLDEN / "bmk_agentic_golden/agg.json").read_text())
    result.update(conc=4, users=4, tp=2, num_gpus=2, is_multinode=shape != "single",
                  disagg=shape == "disaggregated",
                  power_audit={"source": "results/power_validation.json"})
    sweep = json.loads((GOLDEN / "required-power-sweep-manifest/sweep_manifest.json").read_text())
    row = sweep["matrix"]["single_node"]["agentic"].pop()
    row.update(conc=4, tp=2, **{"num-gpus": 2})
    if shape != "single":
        row.update(disagg=shape == "disaggregated", **{"node-count": 3 if shape == "disaggregated" else 1, "compute-node-count": 2 if shape == "disaggregated" else 1})
        row["prefill"] = {"num-worker": 1, "tp": 1 if shape == "disaggregated" else 2, "ep": 1}
        row["decode"] = {"num-worker": 1 if shape == "disaggregated" else 0, "tp": 1 if shape == "disaggregated" else 2, "ep": 1}
        result.update(prefill_tp=row["prefill"]["tp"], prefill_ep=1, prefill_num_workers=1,
                      decode_tp=1 if shape == "disaggregated" else 0,
                      decode_ep=1 if shape == "disaggregated" else 0,
                      decode_num_workers=1 if shape == "disaggregated" else 0,
                      num_prefill_gpu=row["prefill"]["tp"], num_decode_gpu=1 if shape == "disaggregated" else 0)
        sweep["matrix"]["multi_node"]["agentic"] = [row]
    else:
        sweep["matrix"]["single_node"]["agentic"] = [row]
    package.agg_result.write_text(json.dumps(result))
    package.validation_result = evidence / "results/power_validation.json"
    package.validation_result.parent.mkdir()
    assert package.run(prefill_gpus=1 if shape == "disaggregated" else 0,
                       decode_gpus=1 if shape == "disaggregated" else 0,
                       aggregate_gpus=0 if shape == "disaggregated" else 2,
                       require_power=True) == 0
    target = root / "bmk_agentic_native"
    target.mkdir()
    shutil.copyfile(package.agg_result, target / "agg.json")
    shutil.copyfile(GOLDEN / "agentic_golden/power_source.json", evidence / "power_source.json")
    return root, sweep


def test_native_artifact_cli_binds_physical_identity_and_audit(native_evidence, tmp_path):
    root, sweep = native_evidence
    source = tmp_path / "plan.json"
    source.write_text(json.dumps(sweep))
    output = tmp_path / "manifest.json"
    command = [sys.executable, "-m", "infx.results.power.publication", "--sweep", str(source),
               "--artifacts", str(root), "--output", str(output),
               "--expected-run-id", "123", "--expected-head", "b" * 40,
               "--expected-run-attempt", "1"]
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    point = json.loads(output.read_text())["points"][0]
    assert sum(device["energy_j"] for device in point["devices"]) == 60000
    assert {Path(file["path"]).name for file in point["artifacts"]} >= {
        "samples.csv", "manifest.json", "power_validation.json", "power_source.json"}
    command[command.index("--sweep") + 1] = str(output)
    result = subprocess.run([*command, "--verify-existing"], cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_fixed_sequence_manifest_keeps_each_native_point(native_evidence, tmp_path):
    from infx.results.power import multinode

    root, sweep = native_evidence
    topology = "multi_node" if sweep["matrix"]["multi_node"].get("agentic") else "single_node"
    row = sweep["matrix"][topology]["agentic"].pop()
    row.pop("scenario-type")
    row.update(conc=[4, 8], isl=8192, osl=1024)
    sweep["matrix"][topology]["8k1k"] = [row]
    results = root / "bmk_native"
    (root / "bmk_agentic_native").rename(results)
    first = results / "agg_renamed_by_launcher.json"
    (results / "agg.json").rename(first)
    _edit(first, isl=8192, osl=1024, power_audit=None)
    audit = root / "power_audit_native"
    (audit / "results/power_validation.json").rename(audit / "results/power_validation_c4.json")
    package = audit / "LOGS/power"
    window = json.loads((package / "windows/my_result.json").read_text())
    window.update(concurrency=8, result_path="result_c8.json")
    (package / "windows/result_c8.json").write_text(json.dumps(window))
    native_manifest = json.loads((package / "manifest.json").read_text())
    native_manifest["expected_windows"].append({"benchmark_type": "sa-bench", "concurrency": 8})
    native_manifest["window_validations"].append({
        **native_manifest["window_validations"][0], "concurrency": 8,
        "window_file": "windows/result_c8.json",
    })
    (package / "manifest.json").write_text(json.dumps(native_manifest))
    raw = json.loads((audit / "LOGS/my_result.json").read_text())
    raw["max_concurrency"] = 8
    raw_path = audit / "LOGS/result_c8.json"
    raw_path.write_text(json.dumps(raw))
    second = results / "agg_result_c8.json"
    second.write_text(first.read_text())
    _edit(second, conc=8, power_audit={"source": "results/power_validation_c8.json"})
    disagg = row.get("disagg", False)
    assert multinode.run(package, raw_path, second, prefill_gpus=int(disagg),
                         decode_gpus=int(disagg), aggregate_gpus=0 if disagg else 2,
                         expected_producer_sha="a" * 40, logs_root=audit / "LOGS",
                         validation_result=audit / "results/power_validation_c8.json",
                         require_power=True) == 0
    source = tmp_path / "fixed-plan.json"
    source.write_text(json.dumps(sweep))
    output = tmp_path / "fixed-manifest.json"
    completed = subprocess.run([
        sys.executable, "-m", "infx.results.power.publication", "--sweep", str(source),
        "--artifacts", str(root), "--output", str(output),
    ], cwd=REPO, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    points = json.loads(output.read_text())["points"]
    assert [(p["identity"]["benchmark_type"], p["identity"]["concurrency"]) for p in points] == [
        ("single_turn", 4), ("single_turn", 8),
    ]
    assert len({next(a["path"] for a in p["artifacts"] if "power_validation" in a["path"]) for p in points}) == 2


@pytest.mark.parametrize("failure", ["missing", "head", "attempt", "checkout", "tree", "duplicate", "symlink"])
def test_publication_rejects_unbound_source_evidence(evidence, failure):
    root, sweep = evidence
    receipt = root / "agentic_golden/power_source.json"
    if failure == "missing":
        receipt.unlink()
    elif failure == "head":
        _edit(receipt, head="c" * 40)
    elif failure == "attempt":
        _edit(receipt, **{"run-attempt": 2})
    elif failure == "checkout":
        _edit(receipt, **{"checkout-sha": "c" * 40})
    elif failure == "tree":
        _edit(receipt, **{"checkout-tree": "invalid"})
    elif failure == "duplicate":
        shutil.copyfile(root / "bmk_agentic_golden/agg.json", root / "bmk_agentic_golden/duplicate.json")
    elif failure == "symlink":
        original = receipt.read_text()
        receipt.unlink()
        (root / "other.json").write_text(original)
        receipt.symlink_to(root / "other.json")
    with pytest.raises((ValueError, FileNotFoundError)):
        build_manifest(sweep, root)


def test_retry_preserves_old_successes_and_selects_newest_bound_copy(evidence):
    root, sweep = evidence
    sweep["run-attempt"] = 2
    previous = build_manifest(sweep, root)
    assert previous["run-attempt"] == 2
    shutil.copytree(root / "bmk_agentic_golden", root / "bmk_agentic_retry")
    shutil.copytree(root / "agentic_golden", root / "agentic_retry")
    _edit(root / "agentic_retry/power_source.json", **{"run-attempt": 2})
    current = build_manifest(sweep, root)
    paths = [file["path"] for file in current["points"][0]["artifacts"]]
    assert "bmk_agentic_retry/agg.json" in paths
    assert "agentic_retry/power_source.json" in paths
    assert not any("golden" in path for path in paths)


def test_merge_plan_requires_the_same_required_scope(evidence):
    from infx.results.power.publication import compare_required_scope

    _, sweep = evidence
    merged = json.loads(json.dumps(sweep["matrix"]))
    merged["single_node"]["agentic"][0]["queue-token"] = "new-run-token"
    compare_required_scope(sweep["matrix"], merged)
    merged["single_node"]["agentic"][0]["conc"] = [1, 8]
    with pytest.raises(ValueError, match="merged plan"):
        compare_required_scope(sweep["matrix"], merged)
