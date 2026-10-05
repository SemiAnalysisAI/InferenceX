"""Local failure artifacts and CI routing; no scheduler or model calls."""

import sys

import pytest

import ci
from test_ci import config
from test_wan_runner import wan_plan  # noqa: F401


def prepared_config(tmp_path, plan):
    cfg = config(tmp_path)
    cfg.update(mode="serving-smoke", concurrencies=[1])
    ci.write(cfg["spec"]["path"], {"plan": plan})
    cfg["spec"]["sha256"] = ci.digest(cfg["spec"]["path"])
    return cfg


def test_wan_routing_requires_a_sealed_serving_plan(wan_plan, tmp_path):
    cfg = prepared_config(tmp_path, wan_plan)
    assert ci.prepared_artifact_prefix(cfg) == "video-serving"
    cfg["mode"] = "smoke"
    with pytest.raises(ValueError, match="serving-smoke"):
        ci.prepared_artifact_prefix(cfg)
    cfg["mode"] = "serving-smoke"
    ci.write(cfg["spec"]["path"], {"plan": {**wan_plan, "repetitions": 2}})
    with pytest.raises(ValueError, match="changed"):
        ci.prepared_artifact_prefix(cfg)


def test_preparation_failure_keeps_model_and_planned_counts_without_gpu_identity(wan_plan, tmp_path, monkeypatch):
    cfg = prepared_config(tmp_path, wan_plan)
    path, output = tmp_path / "site.json", tmp_path / "output"
    ci.write(path, cfg)
    for name, value in {"GITHUB_RUN_ID": "10001005", "GITHUB_RUN_ATTEMPT": "1",
                        "H3_SOURCE_SHA": "a" * 40, "GITHUB_REPOSITORY": "SemiAnalysisAI/InferenceX"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(sys, "argv", ["ci.py", "--config", str(path), "--output", str(output)])
    monkeypatch.setattr(ci, "command", lambda argv, **kw: "a" * 40 if "rev-parse" in argv else "")
    monkeypatch.setattr(ci, "allocate", lambda *args, **kw: pytest.fail("preparation failure must not allocate"))
    monkeypatch.setattr(ci, "recover", lambda *args, **kw: pytest.fail("preparation failure must not inspect GPUs"))
    assert ci.main() == 2
    manifest = ci.read(output / "manifest.json")
    assert manifest["workload_plan"] == wan_plan
    assert manifest["slurm_allocation"] is None
    assert manifest["execution_started"] is False
    matrix = ci.read(output / "serving-smoke.json")
    assert matrix["bundle_type"] == "video_serving_smoke_matrix"
    assert matrix["completion"] == {"scheduled": 4, "attempted": 0, "completed": 0,
                                    "valid": 0, "failed": 4, "not_started": 4, "unfinished": 0}
    assert "gpu_uuids" not in matrix
    assert "runtime" not in matrix
    assert "run" not in matrix["cells"][0]
    expected = {name: digest for digest, name in (line.split("  ") for line in (output / "SHA256SUMS").read_text().splitlines())}
    assert ci.inventory(output) == expected


def test_failure_receipt_never_rewrites_existing_execution(wan_plan, tmp_path):
    cfg = prepared_config(tmp_path, wan_plan)
    output = tmp_path / "artifact"
    output.mkdir()
    ci.write(output / "manifest.json", {"existing": "preserve"})
    ci.record_wan_preparation_failure(cfg, output, "later failure")
    assert ci.read(output / "manifest.json") == {"existing": "preserve"}
    assert not (output / "serving-smoke.json").exists()


@pytest.mark.parametrize("collection_failed", [False, True])
def test_preparation_fallback_preserves_a_persistent_run(wan_plan, tmp_path, monkeypatch, collection_failed):
    cfg = prepared_config(tmp_path, wan_plan)
    for name, value in {"GITHUB_RUN_ID": "10001005", "GITHUB_RUN_ATTEMPT": "1",
                        "H3_SOURCE_SHA": "a" * 40, "GITHUB_REPOSITORY": "SemiAnalysisAI/InferenceX"}.items():
        monkeypatch.setenv(name, value)
    run_dir = tmp_path / "results" / cfg["task_id"] / "github-10001005-1"
    run_dir.mkdir(parents=True)
    ci.write(run_dir / "ci.json", {"phase": "failed", "execution_started": True})
    ci.write(run_dir / "manifest.json", {"slurm_allocation": {"identity": {"JobId": "123"}}})
    ci.write(run_dir / "serving-smoke.json", {"completion": {"attempted": 2, "not_started": 2}})
    output = tmp_path / "artifact"
    output.mkdir()
    if collection_failed:
        copyfile = ci.shutil.copyfile

        def interrupted_copy(source, target):
            if source.name == "manifest.json":
                raise OSError("collection interrupted before manifest copy")
            return copyfile(source, target)

        monkeypatch.setattr(ci.shutil, "copyfile", interrupted_copy)
        with pytest.raises(OSError, match="collection interrupted"):
            ci.collect(run_dir, output)

    ci.record_wan_preparation_failure(cfg, output, "collection failure or existing-run retry")

    assert ci.read(run_dir / "serving-smoke.json")["completion"] == {"attempted": 2, "not_started": 2}
    assert not (output / "manifest.json").exists()
    assert not (output / "serving-smoke.json").exists()
    if collection_failed:
        assert ci.read(output / "ci.json") == {"phase": "failed", "execution_started": True}
    else:
        assert list(output.iterdir()) == []
