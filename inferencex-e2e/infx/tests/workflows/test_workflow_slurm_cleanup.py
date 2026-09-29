"""Execute the self-hosted workflows' shell Slurm cleanup against fake scancel/squeue."""

import subprocess
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[4]
RUNNER = "fixture-cluster_00"
LAUNCH = "infx.launch run"

# A stale job of this runner leaves the queue one poll after scancel, so each
# cancel must wait through one squeue round. scancel records the workspace
# listing at the time it ran.
FAKES = {
    "scancel": (
        'printf \'%s|%s\\n\' "$*" "$(ls "$GITHUB_WORKSPACE" | tr "\\n" " ")" >> "$FAKE_DIR/scancel.log"\n'
        'touch "$FAKE_DIR/cancelled"\n'
    ),
    "squeue": (
        'printf \'%s\\n\' "$*" >> "$FAKE_DIR/squeue.log"\n'
        '[ -e "$FAKE_DIR/cancelled" ] && [ -e "$FAKE_DIR/polled" ] && exit 0\n'
        '[ -e "$FAKE_DIR/cancelled" ] && touch "$FAKE_DIR/polled"\n'
        "echo 123\n"
    ),
    "sleep": "exit 0\n",
}


def launch_job_steps(workflow: str) -> list[dict]:
    """Steps of the workflow's job that runs ``python -m infx.launch run``."""
    config = yaml.safe_load((ROOT / ".github" / "workflows" / workflow).read_text())
    [steps] = [
        job["steps"]
        for job in config["jobs"].values()
        if any(LAUNCH in step.get("run", "") for step in job.get("steps", []))
    ]
    return steps


def run_steps(tmp_path: Path, steps: list[dict]) -> dict[str, list[str]]:
    """Run each step's script as Actions would; return the fakes' call logs."""
    fakes = tmp_path / "bin"
    fakes.mkdir(exist_ok=True)
    for name, body in FAKES.items():
        (fakes / name).write_text(f"#!/bin/bash\n{body}")
        (fakes / name).chmod(0o755)
    for state in ("cancelled", "polled", "scancel.log", "squeue.log"):
        (tmp_path / state).unlink(missing_ok=True)
    env = {
        "PATH": f"{fakes}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "USER": "runner",
        "RUNNER_NAME": RUNNER,
        "GITHUB_WORKSPACE": str(tmp_path / "workspace"),
        "FAKE_DIR": str(tmp_path),
    }
    for step in steps:
        assert "${{" not in step["run"], step["name"]
        completed = subprocess.run(
            ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", step["run"]],
            cwd=tmp_path / "workspace", env=env, capture_output=True, text=True, timeout=30,
        )  # fmt: skip
        assert completed.returncode == 0, completed.stdout + completed.stderr
    return {
        name: (tmp_path / f"{name}.log").read_text().splitlines() if (tmp_path / f"{name}.log").exists() else []
        for name in ("scancel", "squeue")
    }


def assert_cancelled_and_waited(calls: dict[str, list[str]]) -> None:
    assert any(f"--name={RUNNER}" in line.split("|")[0] for line in calls["scancel"]), calls
    polls = [line for line in calls["squeue"] if f"--name={RUNNER}" in line]
    assert len(polls) >= 2, calls  # it kept polling until the job left the queue


@pytest.mark.parametrize(
    "workflow", ["benchmark-tmpl.yml", "benchmark-multinode-tmpl.yml", "profile.yml", "speedbench-al.yml"]
)
def test_stale_runner_jobs_are_cancelled_before_checkout(tmp_path, workflow):
    workspace = tmp_path / "workspace"
    (workspace / "speedbench_results").mkdir(parents=True)
    steps = launch_job_steps(workflow)
    checkout = next(
        index for index, step in enumerate(steps) if str(step.get("uses", "")).startswith("actions/checkout")
    )
    # Conditional pre-run steps (the MI355X ownership repair) need sudo; they cancel nothing.
    pre_checkout = [step for step in steps[:checkout] if "run" in step and "if" not in step]

    calls = run_steps(tmp_path, pre_checkout)

    assert_cancelled_and_waited(calls)
    if workflow == "speedbench-al.yml":
        # Cancelled while the stale outputs were still there, then they are removed.
        assert all("speedbench_results" in line.split("|")[1] for line in calls["scancel"])
        assert not (workspace / "speedbench_results").exists()


@pytest.mark.parametrize("workflow", ["benchmark-tmpl.yml", "benchmark-multinode-tmpl.yml", "speedbench-al.yml"])
def test_post_run_cancellation_does_not_need_the_launcher_python(tmp_path, workflow):
    (tmp_path / "workspace").mkdir()
    steps = launch_job_steps(workflow)
    launch = next(index for index, step in enumerate(steps) if LAUNCH in step.get("run", ""))
    # With INFERENCEX_LAUNCH_PYTHON unset only unconditional always() steps run.
    post_run = [step for step in steps[launch + 1:] if step.get("if") == "always()" and "run" in step]

    assert_cancelled_and_waited(run_steps(tmp_path, post_run))
