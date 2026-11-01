"""Execute the PR workflow's canary selection with video and mixed matrices."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("llm", [False, True])
def test_canary_leaves_every_video_deployment_scheduled_once(tmp_path, llm):
    video = [{"scenario-type": "video-generation", "conc-list": [1, 2, 4], "gpus": 2,
              "runner": runner, "run-eval": False} for runner in ("cluster:a", "cluster:b")]
    fixed = [{"conc": 4, "run-eval": False}] if llm else []
    matrix = {"single_node": {"video": video, "1k1k": fixed}}
    workflow = yaml.safe_load((ROOT / ".github/workflows/run-sweep.yml").read_text())
    step = workflow["jobs"]["canary-select"]["steps"][0]
    output = tmp_path / "output"
    result = subprocess.run(["bash", "-e", "-c", step["run"]], text=True, capture_output=True,
                            env={**os.environ, "SEARCH_SPACE": json.dumps(matrix), "GITHUB_OUTPUT": str(output)})
    assert result.returncode == 0, result.stderr
    outputs = {key: json.loads(value) for key, value in
               (line.split("=", 1) for line in output.read_text().splitlines())}
    canary = outputs["canary-config"]
    remaining = outputs["remaining-search-space-config"]["single_node"]
    assert len(canary) == 1
    assert canary[0] == video[0]
    assert remaining["1k1k"] == fixed
    assert [r for r in canary if r.get("scenario-type") == "video-generation"] + remaining["video"] == video


@pytest.mark.parametrize("video,llm,retired,expected", [
    (True, False, False, ("true", "false", "false")),
    (False, True, False, ("false", "true", "false")),
    (True, True, False, ("true", "true", "false")),
    (False, False, True, ("false", "false", "true")),
])
def test_merge_ingest_routes_video_and_llm_independently(tmp_path, video, llm, retired, expected):
    workspace = tmp_path / "workspace"
    benchmarks = workspace / "inferencex-e2e/benchmarks"
    benchmarks.mkdir(parents=True)
    (benchmarks / "check_env.sh").symlink_to(ROOT / "inferencex-e2e/benchmarks/check_env.sh")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    uv = bindir / "uv"
    uv.write_text('#!/bin/sh\nprintf "%s" "$FIXTURE_PLAN"\n')
    uv.chmod(0o755)
    plan = {"single_node": {"video": [{"x": 1}] if video else [], "1k1k": [{"x": 1}] if llm else []},
            "multi_node": {}, "changelog_metadata": {"entries": [{"retired": retired}]}, "evals": [], "agentic_evals": [],
            "multinode_evals": [], "multinode_agentic_evals": []}
    workflow = yaml.safe_load((ROOT / ".github/workflows/merge-ingest.yml").read_text())
    step = next(step for step in workflow["jobs"]["ingest"]["steps"] if step.get("id") == "plan")
    output = tmp_path / "output"
    result = subprocess.run(["bash", "-e", "-c", step["run"]], text=True, capture_output=True,
        env={**os.environ, "PATH": str(bindir) + os.pathsep + os.environ["PATH"], "FIXTURE_PLAN": json.dumps(plan),
             "GITHUB_WORKSPACE": str(workspace), "RUNNER_TEMP": str(tmp_path), "GITHUB_OUTPUT": str(output),
             "PUSH_BEFORE": "a", "PUSH_AFTER": "b"})
    assert result.returncode == 0, result.stderr
    outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert (outputs["video"], outputs["llm"], outputs["retirement-only"]) == expected


def test_video_reuse_discovery_requires_the_selected_matrix_manifest():
    from infx.workflows.sweep_runs import has_reusable_result_artifacts
    name = "h3-results-123-2-" + "a" * 64
    assert not has_reusable_result_artifacts({name})
    assert has_reusable_result_artifacts({name, "video-sweep-manifest"})
    assert not has_reusable_result_artifacts({"h3-results-123-2", "video-sweep-manifest"})


@pytest.mark.parametrize("labels,full", [([], False), (["full-sweep-enabled"], True),
    (["full-sweep-fail-fast"], True), (["full-sweep-enabled", "evals-only"], False),
    (["full-sweep-enabled", "agentx-fast"], False)])
def test_selected_video_manifest_distinguishes_full_sweep_from_partial(tmp_path, labels, full):
    workflow = yaml.safe_load((ROOT / ".github/workflows/run-sweep.yml").read_text())
    step = workflow["jobs"]["upload-changelog-metadata"]["steps"][0]
    result = subprocess.run(["bash", "-e", "-c", step["run"]], cwd=tmp_path, text=True, capture_output=True,
        env={**os.environ, "SWEEP_MATRIX": json.dumps({"changelog_metadata": {}, "single_node": {"video": [{}]}}),
             "SWEEP_HEAD": "a" * 40, "SWEEP_LABELS": json.dumps(labels), "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "2"})
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "sweep_manifest.json").read_text())
    assert manifest["video-full-sweep"] is full
    assert (manifest["run-id"], manifest["run-attempt"]) == (123, 2)


@pytest.mark.parametrize("video,llm", [(True, False), (True, True), (False, True)])
def test_sweep_setup_routes_only_llm_rows_to_benchmark_aggregation(tmp_path, video, llm):
    workspace = tmp_path / "workspace"
    (workspace / "inferencex-e2e").mkdir(parents=True)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    uv = bindir / "uv"
    uv.write_text('#!/bin/sh\ncase "$*" in *infx.matrix.plan*) printf "%s" "$FIXTURE_PLAN";; *) cat;; esac\n')
    uv.chmod(0o755)
    plan = {"single_node": {"video": [{"scenario-type": "video-generation"}] if video else [],
                            "1k1k": [{"conc": 1}] if llm else []}, "multi_node": {}}
    workflow = yaml.safe_load((ROOT / ".github/workflows/run-sweep.yml").read_text())
    step = next(step for step in workflow["jobs"]["setup"]["steps"] if step.get("id") == "setup")
    output = tmp_path / "output"
    result = subprocess.run(["bash", "-e", "-c", step["run"]], cwd=workspace, text=True, capture_output=True,
        env={**os.environ, "PATH": str(bindir) + os.pathsep + os.environ["PATH"], "FIXTURE_PLAN": json.dumps(plan),
             "GITHUB_WORKSPACE": str(workspace), "GITHUB_OUTPUT": str(output), "GITHUB_BASE_REF": "main",
             "PR_HEAD_SHA": "a" * 40, "ALL_EVALS": "false", "EVALS_ONLY": "false",
             "GITHUB_EVENT_NAME": "pull_request", "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1",
             "PRIORITY_CRITERIA": ""})
    assert result.returncode == 0, result.stderr
    outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert outputs["has-benchmark-results"] == str(llm).lower()
    assert json.loads(outputs["search-space-config"]) == plan


def test_empty_llm_collection_cannot_create_a_reusable_aggregate(tmp_path, monkeypatch):
    import sys
    from infx.results import collect_results
    monkeypatch.chdir(tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setattr(sys, "argv", ["collect_results", str(results), "bmk"])
    collect_results.main()
    assert not (tmp_path / "agg_bmk.json").exists()
    point = {"model": "test/model", "throughput": 12.5}
    (results / "point.json").write_text(json.dumps(point))
    collect_results.main()
    assert json.loads((tmp_path / "agg_bmk.json").read_text()) == [point]
