"""Execute the PR workflow's canary selection with video and mixed matrices."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[4]


def test_canary_leaves_every_video_deployment_scheduled_once(tmp_path):
    video = [{"scenario-type": "video-generation", "conc-list": [1, 2, 4], "gpus": 2,
              "runner": runner, "run-eval": False} for runner in ("cluster:a", "cluster:b")]
    fixed = [{"conc": 4, "run-eval": False}]
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


def test_merge_ingest_routes_video_and_llm_independently(tmp_path):
    workspace = tmp_path / "workspace"
    benchmarks = workspace / "inferencex-e2e/benchmarks"
    benchmarks.mkdir(parents=True)
    (benchmarks / "check_env.sh").symlink_to(ROOT / "inferencex-e2e/benchmarks/check_env.sh")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    uv = bindir / "uv"
    uv.write_text('#!/bin/sh\nprintf "%s" "$FIXTURE_PLAN"\n')
    uv.chmod(0o755)
    plan = {"single_node": {"video": [{"x": 1}], "1k1k": [{"x": 1}]},
            "multi_node": {}, "changelog_metadata": {"entries": [{"retired": False}]}, "evals": [], "agentic_evals": [],
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
    assert (outputs["video"], outputs["llm"], outputs["retirement-only"]) == ("true", "true", "false")


@pytest.mark.parametrize("agentx_fast,manifest", [("false", True), ("true", False)])
def test_trusted_external_dispatch_records_reusable_video_matrix(tmp_path, agentx_fast, manifest):
    workflow = yaml.safe_load((ROOT / ".github/workflows/e2e-tests.yml").read_text())
    step = next(s for s in workflow["jobs"]["get-jobs"]["steps"] if s.get("id") == "source-identity")
    plan = {"single_node": {"video": [{"recipe-fingerprint": "f"}]}, "changelog_metadata": {}}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    output = tmp_path / "output"
    head, merge = "a" * 40, "b" * 40
    result = subprocess.run(["bash", "-e", "-c", step["run"]], text=True, capture_output=True,
        env={**os.environ, "RUNNER_TEMP": str(tmp_path), "GITHUB_OUTPUT": str(output), "GITHUB_RUN_ID": "123",
             "GITHUB_RUN_ATTEMPT": "2", "SOURCE_PR_NUMBER": "7", "CHANGELOG_HEAD_REF": head, "MERGE_SHA": merge,
             "EVALS_ONLY": "false", "AGENTX_FAST": agentx_fast, "ALL_EVALS": "false", "FAIL_FAST": "true"})
    assert result.returncode == 0, result.stderr
    if not manifest:
        assert not (tmp_path / "sweep_manifest.json").exists()
        return
    assert output.read_text() == "video-manifest=true\n"
    assert json.loads((tmp_path / "sweep_manifest.json").read_text()) == {
        "head": head, "run-id": 123, "run-attempt": 2, "video-full-sweep": True, "matrix": plan}
