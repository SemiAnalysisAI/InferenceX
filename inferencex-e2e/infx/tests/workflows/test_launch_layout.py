"""Execute GPU launch steps against current and historical checkout layouts."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("layout", ["historical", "nested", "historical-with-stale-submodule"])
@pytest.mark.parametrize(
    "workflow,step_name",
    [
        ("benchmark-tmpl.yml", "Launch job script"),
        ("benchmark-multinode-tmpl.yml", "Launch multi-node job script"),
        ("profile.yml", "Launch + Profile (single-node sglang/vllm)"),
        ("speedbench-al.yml", "Collect AL matrix"),
    ],
)
def test_launch_uses_measured_project_root(tmp_path, layout, workflow, step_name):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    project = checkout / "inferencex-e2e" if layout == "nested" else checkout
    (project / "configs").mkdir(parents=True)
    (project / "configs" / "runners.yaml").write_text("{}\n")
    if layout == "historical-with-stale-submodule":
        stale = checkout / "inferencex-e2e" / "utils" / "srt-slurm"
        stale.mkdir(parents=True)
        (stale / ".git").write_text("gitdir: ../../../.git/modules/utils/srt-slurm\n")
        (stale / "leftover.py").write_text("# Retained when checking out an older revision.\n")

    runners = project / "runners"
    runners.mkdir()
    (runners / "launch_fixture.sh").write_text(
        """#!/bin/bash
python3 - <<'PY'
import json
import os
from pathlib import Path

Path(os.environ["LAUNCH_CAPTURE"]).write_text(json.dumps({
    "cwd": str(Path.cwd()),
    "workspace": os.environ["GITHUB_WORKSPACE"],
}))
result = os.environ["RESULT_FILENAME"]
Path(result + ".json").write_text("{}\\n")
Path(result + "_conc1.json").write_text('{"num_requests_successful": 1}\\n')
Path("profile_" + result + ".trace.json.gz").write_bytes(b"fixture trace")
Path("speedbench-reference-al.yaml").write_text("fixture: 1\\n")
PY
"""
    )
    config = yaml.safe_load((ROOT / ".github" / "workflows" / workflow).read_text())
    step = next(
        step
        for job in config["jobs"].values()
        for step in job.get("steps", [])
        if step.get("name") == step_name
    )
    capture = tmp_path / "launch.json"
    github_env = tmp_path / "github-env"
    github_output = tmp_path / "github-output"
    summary = tmp_path / "summary"
    completed = subprocess.run(
        ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", step["run"]],
        cwd=checkout,
        env={
            **os.environ,
            "LAUNCH_CAPTURE": str(capture),
            "GITHUB_WORKSPACE": str(checkout),
            "GITHUB_ENV": str(github_env),
            "GITHUB_OUTPUT": str(github_output),
            "GITHUB_STEP_SUMMARY": str(summary),
            "RUNNER_NAME": "fixture_01",
            "RESULT_FILENAME_BASE": "layout-test",
            "RESULT_FILENAME": "layout-test",
            "RECIPE_FINGERPRINT": "fixture-recipe",
            "TP": "2",
            "PP_SIZE": "1",
            "PCP_SIZE": "1",
            "DCP_SIZE": "1",
            "EP_SIZE": "1",
            "DP_ATTENTION": "false",
            "EXP_NAME": "layout-test",
            "FRAMEWORK": "vllm",
            "PRECISION": "fp8",
            "CONC": "1",
            "CONC_LIST": "1",
            "EVAL_CONC": "1",
            "EVAL_ONLY": "false",
            "SCENARIO_TYPE": "fixed-sequence",
            "PREFILL_ADDITIONAL_SETTINGS": "[]",
            "DECODE_ADDITIONAL_SETTINGS": "[]",
        },
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(capture.read_text()) == {
        "cwd": str(project.resolve()),
        "workspace": str(project.resolve()),
    }
    exported = dict(line.split("=", 1) for line in github_env.read_text().splitlines())
    assert exported["INFERENCEX_E2E_ROOT"] == str(project.resolve())
    if workflow == "speedbench-al.yml":
        assert "fixture: 1" in summary.read_text()
    else:
        assert (project / (exported["RESULT_FILENAME"] + ".json")).read_text() == "{}\n"
    if workflow == "profile.yml":
        output = dict(line.split("=", 1) for line in github_output.read_text().splitlines())
        assert Path(output["trace"]).read_bytes() == b"fixture trace"
