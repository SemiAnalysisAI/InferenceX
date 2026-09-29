"""Execute the GPU workflows' launch steps against a measured checkout."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[4]

# Records how the launcher ran, then writes the files each workflow checks after it.
CAPTURE = """import json
import os
import sys
from pathlib import Path

Path(os.environ["LAUNCH_CAPTURE"]).write_text(json.dumps({
    "argv": sys.argv[1:],
    "cwd": str(Path.cwd()),
    "workspace": os.environ["GITHUB_WORKSPACE"],
    "settings": [os.environ.get("FIXTURE_SETTINGS"), os.environ.get("FIXTURE_MULTI_NODE_SETTINGS")],
}))
result = os.environ["RESULT_FILENAME"]
Path(result + ".json").write_text("{}\\n")
Path(result + "_conc1.json").write_text('{"num_requests_successful": 1}\\n')
Path("profile_" + result + ".trace.json.gz").write_bytes(b"fixture trace")
Path("speedbench-reference-al.yaml").write_text("fixture: 1\\n")
"""


def measured_checkout(root: Path, launcher: str) -> tuple[Path, Path]:
    """A checkout whose ``infx.launch`` runs ``launcher``; returns (checkout, project)."""
    checkout = root / "checkout"
    project = checkout / "inferencex-e2e"
    (project / "configs").mkdir(parents=True)
    (project / "configs" / "runners.yaml").write_text("{}\n")
    (project / "benchmarks" / "multi_node").mkdir(parents=True)
    (project / "benchmarks" / "runtime_settings.sh").write_text("export FIXTURE_SETTINGS=1\n")
    (project / "benchmarks" / "multi_node" / "runtime_settings.sh").write_text(
        "export FIXTURE_MULTI_NODE_SETTINGS=1\n"
    )
    # A leftover shell launcher never runs: every revision launches through infx.launch.
    (project / "runners").mkdir()
    (project / "runners" / "launch_fixture.sh").write_text("#!/bin/bash\nexit 97\n")
    package = project / "infx" / "launch"
    package.mkdir(parents=True)
    (project / "infx" / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "__main__.py").write_text(launcher)
    # The measured revision's own result-name helper; every launchable revision ships it.
    results = project / "infx" / "results"
    results.mkdir()
    (results / "__init__.py").write_text("")
    shutil.copy(ROOT / "inferencex-e2e/infx/results/result_filename.py", results)
    return checkout, project


def workflow_step(workflow: str, step_name: str) -> dict:
    config = yaml.safe_load((ROOT / ".github" / "workflows" / workflow).read_text())
    return next(
        step
        for job in config["jobs"].values()
        for step in job.get("steps", [])
        if step.get("name") == step_name
    )


@pytest.mark.parametrize(
    "workflow,step_name",
    [
        ("benchmark-tmpl.yml", "Launch job script"),
        ("benchmark-multinode-tmpl.yml", "Launch multi-node job script"),
        ("profile.yml", "Launch + Profile (single-node sglang/vllm)"),
        ("speedbench-al.yml", "Collect AL matrix"),
    ],
)
def test_launch_runs_the_python_entrypoint_from_the_project_root(tmp_path, workflow, step_name):
    checkout, project = measured_checkout(tmp_path, CAPTURE)
    step = workflow_step(workflow, step_name)
    launch_capture = tmp_path / "launch.json"
    github_env = tmp_path / "github-env"
    github_output = tmp_path / "github-output"
    summary = tmp_path / "summary"
    completed = subprocess.run(
        ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", step["run"]],
        cwd=checkout,
        env={
            **os.environ,
            "LAUNCH_CAPTURE": str(launch_capture),
            "INFERENCEX_LAUNCH_PYTHON": sys.executable,
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
            "IS_AGENTIC": "0",
            "SCENARIO_TYPE": "fixed-sequence",
            "PREFILL_ADDITIONAL_SETTINGS": "[]",
            "DECODE_ADDITIONAL_SETTINGS": "[]",
        },
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    recorded = json.loads(launch_capture.read_text())
    assert recorded["argv"] == ["run"]
    assert (recorded["cwd"], recorded["workspace"]) == (str(project.resolve()), str(project.resolve()))
    multi_node = "1" if workflow == "benchmark-multinode-tmpl.yml" else None
    assert recorded["settings"] == ["1", multi_node]
    exported = dict(line.split("=", 1) for line in github_env.read_text().splitlines())
    assert exported["INFERENCEX_E2E_ROOT"] == str(project.resolve())
    if workflow == "speedbench-al.yml":
        assert "fixture: 1" in summary.read_text()
    else:
        assert (project / (exported["RESULT_FILENAME"] + ".json")).read_text() == "{}\n"
    if workflow == "profile.yml":
        output = dict(line.split("=", 1) for line in github_output.read_text().splitlines())
        assert Path(output["trace"]).read_bytes() == b"fixture trace"


@pytest.mark.parametrize(
    "agentic,eval_only,conc,conc_list,launches",
    [
        (True, False, "4", "4", True),
        (True, False, "4", "4 8", False),
        (True, False, "4", "8", False),
        (True, True, "4", "4 8", True),
        (False, False, "", "4 8", True),
    ],
)
def test_multinode_launch_isolates_agentx_throughput(
    tmp_path, agentic, eval_only, conc, conc_list, launches
):
    """Reject invalid throughput jobs before launching, without changing eval batching."""
    marker = tmp_path / "launched"
    launcher = "import os, pathlib, sys\npathlib.Path(os.environ['LAUNCH_MARKER']).touch()\nsys.exit(77)\n"
    checkout, _ = measured_checkout(tmp_path, launcher)
    step = workflow_step("benchmark-multinode-tmpl.yml", "Launch multi-node job script")
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", step["run"]],
        cwd=checkout,
        env={
            **os.environ,
            "LAUNCH_MARKER": str(marker),
            "INFERENCEX_LAUNCH_PYTHON": sys.executable,
            "GITHUB_WORKSPACE": str(checkout),
            "GITHUB_ENV": str(tmp_path / "github-env"),
            "RESULT_FILENAME_BASE": "agentx-isolation",
            "RECIPE_FINGERPRINT": "",
            "PREFILL_ADDITIONAL_SETTINGS": "[]",
            "DECODE_ADDITIONAL_SETTINGS": "[]",
            "IS_AGENTIC": "1" if agentic else "0",
            "SCENARIO_TYPE": "agentic-coding" if agentic else "fixed-seq-len",
            "EVAL_ONLY": "true" if eval_only else "false",
            "CONC": conc,
            "CONC_LIST": conc_list,
            "EVAL_CONC": "4",
            "RUNNER_NAME": "fixture_01",
            "VALIDATION_BENCHMARK_LIB": str(ROOT / "inferencex-e2e/benchmarks/benchmark_lib.sh"),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == (77 if launches else 1), result.stdout + result.stderr
    assert marker.exists() is launches
    if not launches:
        assert "AgentX" in result.stderr
