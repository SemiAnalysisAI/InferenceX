"""Execute the GPU workflows' launch steps against a measured checkout."""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[4]

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

# Like the launcher, the stub handles the signals itself: background jobs start ignoring SIGINT.
SIGNALLED = """import json
import os
import signal
import sys
import time
from pathlib import Path


def stop(signum, frame):
    Path("job_event.tmp").write_text(json.dumps({"signal": signum}))
    os.replace("job_event.tmp", "job_event.json")
    sys.exit(128 + signum)


for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(signum, stop)
Path(os.environ["LAUNCH_READY"]).touch()
time.sleep(60)
"""
LAUNCH_STEPS = [
    ("benchmark-tmpl.yml", "Launch job script"),
    ("benchmark-multinode-tmpl.yml", "Launch multi-node job script"),
    ("profile.yml", "Launch + Profile (single-node sglang/vllm)"),
    ("speedbench-al.yml", "Collect AL matrix"),
]


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
    (project / "runners").mkdir()
    (project / "runners" / "launch_fixture.sh").write_text("#!/bin/bash\nexit 97\n")
    package = project / "infx" / "launch"
    package.mkdir(parents=True)
    (project / "infx" / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "__main__.py").write_text(launcher)
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


def step_env(tmp_path: Path, checkout: Path, **extra: str) -> dict[str, str]:
    """What Actions and the earlier steps give a launch step."""
    return {
        **os.environ,
        "INFERENCEX_LAUNCH_PYTHON": sys.executable,
        "INFERENCEX_TOOLING": str(ROOT / "inferencex-e2e"),
        "GITHUB_WORKSPACE": str(checkout),
        "GITHUB_ENV": str(tmp_path / "github-env"),
        "GITHUB_OUTPUT": str(tmp_path / "github-output"),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
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
        **extra,
    }


@pytest.mark.parametrize("workflow,step_name", LAUNCH_STEPS)
def test_launch_runs_the_python_entrypoint_from_the_project_root(tmp_path, workflow, step_name):
    checkout, project = measured_checkout(tmp_path, CAPTURE)
    step = workflow_step(workflow, step_name)
    launch_capture = tmp_path / "launch.json"
    completed = subprocess.run(
        ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", step["run"]],
        cwd=checkout,
        env=step_env(tmp_path, checkout, LAUNCH_CAPTURE=str(launch_capture)),
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
    exported = dict(line.split("=", 1) for line in (tmp_path / "github-env").read_text().splitlines())
    assert exported["INFERENCEX_E2E_ROOT"] == str(project.resolve())
    if workflow == "speedbench-al.yml":
        assert "fixture: 1" in (tmp_path / "summary").read_text()
    else:
        assert (project / (exported["RESULT_FILENAME"] + ".json")).read_text() == "{}\n"
    if workflow == "profile.yml":
        output = dict(line.split("=", 1) for line in (tmp_path / "github-output").read_text().splitlines())
        assert Path(output["trace"]).read_bytes() == b"fixture trace"


@pytest.mark.parametrize("workflow,step_name", LAUNCH_STEPS)
def test_a_signal_to_the_step_shell_reaches_the_launcher_and_its_status_ends_the_step(
    tmp_path, workflow, step_name
):
    checkout, project = measured_checkout(tmp_path, SIGNALLED)
    ready, log = tmp_path / "ready", tmp_path / "step.log"
    with log.open("w") as output:
        shell = subprocess.Popen(
            ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", workflow_step(workflow, step_name)["run"]],
            cwd=checkout, env=step_env(tmp_path, checkout, LAUNCH_READY=str(ready)),
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
        )  # fmt: skip
    try:
        deadline = time.monotonic() + 30
        while not ready.exists():
            assert shell.poll() is None and time.monotonic() < deadline, log.read_text()
            time.sleep(0.05)
        os.kill(shell.pid, signal.SIGINT)
        record = project / "job_event.json"
        deadline = time.monotonic() + 2
        while not record.exists():
            assert time.monotonic() < deadline, "the launcher never got the signal:\n" + log.read_text()
            time.sleep(0.05)

        assert json.loads(record.read_text()) == {"signal": signal.SIGINT}
        assert shell.wait(timeout=10) == 128 + signal.SIGINT, log.read_text()
    finally:
        if shell.poll() is None:
            os.killpg(shell.pid, signal.SIGKILL)
            shell.wait()
