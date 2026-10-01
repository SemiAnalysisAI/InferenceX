"""Slurm CLI wrappers of the Slurm backend against fake salloc/srun/squeue/sacct/scontrol binaries."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from infx.clusters import load_inventory
from infx.launch.backends import slurm
from infx.launch.backends.base import Job, JobState
from infx.launch.backends.slurm.cli import (
    ContainerSpec,
    Resources,
    SlurmError,
    final_status,
    queue_state,
    salloc,
    srun_argv,
    stream_log,
)
from infx.launch.lifecycle import Lifecycle
from infx.launch.request import LaunchRequest


@pytest.fixture
def fake_bin(tmp_path, monkeypatch):
    """Directory of fake Slurm binaries placed ahead of only the system tool dirs."""
    binaries = tmp_path / "bin"
    binaries.mkdir()
    monkeypatch.setenv("PATH", f"{binaries}:/usr/bin:/bin")

    def install(name: str, body: str) -> None:
        binary = binaries / name
        binary.write_text(f"#!/bin/bash\n{body}\n")
        binary.chmod(0o755)

    return install


def recorder(log: Path) -> str:
    """Bash snippet appending the invocation's argv as a JSON line to ``log``."""
    return (
        f"{sys.executable} -c 'import json,sys; print(json.dumps(sys.argv[1:]))' \"$@\" >> {log}"
    )


def calls(log: Path) -> list[list[str]]:
    """Recorded invocations, one argv per line."""
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def test_salloc_returns_granted_job_and_requests_no_shell(fake_bin, tmp_path):
    log = tmp_path / "salloc.log"
    fake_bin("salloc", f"{recorder(log)}\n"
             "echo 'salloc: Pending job allocation 5150' >&2\n"
             "echo 'salloc: Granted job allocation 5150' >&2")
    resources = Resources(partition="h200", account="bench", time_min=180, job_name="runner_0",
                          gres="gpu:h200:8", exclude=("n1", "n2"))
    assert salloc(resources, extra=("--exclusive",)) == Job("5150")
    [argv] = calls(log)
    assert argv[-1] == "--no-shell"
    assert {"--partition=h200", "--account=bench", "--time=180", "--job-name=runner_0",
            "--gres=gpu:h200:8", "--exclude=n1,n2", "--exclusive", "--nodes=1"} <= set(argv)


@pytest.mark.parametrize("body", [
    "echo 'salloc: Job allocation 12 has been revoked.' >&2; exit 0",
    "echo 'salloc: Granted job allocation 12' >&2; exit 1",
])
def test_salloc_without_a_grant_fails(fake_bin, body):
    fake_bin("salloc", body)
    with pytest.raises(SlurmError, match="salloc failed"):
        salloc(Resources(partition="p", account=None, time_min=10, job_name="n"))


@pytest.mark.parametrize("record,state,exit_code", [
    ("COMPLETED|0:0", JobState.SUCCEEDED, 0),
    ("FAILED|1:0", JobState.FAILED, 1),
    ("COMPLETED|0:15", JobState.FAILED, 0),
    ("CANCELLED by 1001|0:0", JobState.CANCELLED, 0),
])
def test_final_status_maps_allocation_accounting(fake_bin, tmp_path, record, state, exit_code):
    log = tmp_path / "sacct.log"
    fake_bin("sacct", f"{recorder(log)}\necho '{record}'")
    status = final_status(Job("42"), delay_s=0)
    assert (status.state, status.exit_code) == (state, exit_code)
    assert calls(log) == [["-X", "-n", "-P", "-j", "42", "--format=State,ExitCode"]]


@pytest.mark.parametrize("readings,state", [
    (["COMPLETING|0:0", "COMPLETING|0:0", "COMPLETED|0:0"], JobState.SUCCEEDED),
    (["RUNNING|0:0"] * 3, JobState.RUNNING),
])
def test_final_status_retries_unsettled_accounting_up_to_its_attempts(fake_bin, tmp_path, readings, state):
    counter = tmp_path / "count"
    fake_bin("sacct", f"n=$(cat {counter} 2>/dev/null || echo 0); echo $((n+1)) > {counter}\n"
             f"readings=({' '.join(repr(reading) for reading in readings)}); echo \"${{readings[$n]}}\"")
    assert final_status(Job("42"), attempts=3, delay_s=0).state is state
    assert counter.read_text().strip() == "3"


@pytest.mark.parametrize("controller,state", [
    ("JobId=42 JobName=x JobState=COMPLETED Reason=None ExitCode=0:0", JobState.SUCCEEDED),
    ("JobId=42 JobName=x JobState=FAILED Reason=NonZeroExitCode ExitCode=3:0", JobState.FAILED),
    ("JobId=420 JobName=x JobState=COMPLETED ExitCode=0:0", JobState.UNKNOWN),
])
def test_final_status_falls_back_to_controller_without_sacct(fake_bin, controller, state):
    fake_bin("scontrol", f"echo '{controller}'")
    assert final_status(Job("42"), attempts=3, delay_s=0).state is state


def test_queue_state_reads_only_the_named_job(fake_bin):
    fake_bin("squeue", "echo '420|RUNNING'; echo '42|PENDING'")
    assert queue_state(Job("42")) == "PENDING"
    assert queue_state(Job("7")) is None


def test_stream_log_fails_when_job_dies_before_its_log(fake_bin, tmp_path):
    log = tmp_path / "scontrol.log"
    fake_bin("squeue", "exit 0")
    fake_bin("scontrol", recorder(log))
    with pytest.raises(SlurmError, match="ended before creating"):
        stream_log(Job("77"), tmp_path / "missing.log", wait_s=0)
    assert calls(log) == [["show", "job", "77"]]


def slurm_backend() -> slurm.SlurmBackend:
    """A Slurm backend for runner ``c_0`` of a minimal cluster."""
    record = {"gpus-per-node": 8, "arch": "x86_64", "scheduler": "slurm",
              "slurm": {"partition": "p", "exclusive": True}}  # fmt: skip
    cluster = load_inventory({"labels": {"cluster:c": ["c_0"]}, "clusters": {"c": record}}).clusters["c"]
    return slurm.SlurmBackend(cluster, LaunchRequest.from_env({"RUNNER_NAME": "c_0"}), Lifecycle())


def test_backend_cancel_waits_until_a_listed_job_leaves_the_queue(fake_bin, tmp_path, monkeypatch):
    cancelled = tmp_path / "cancelled"
    fake_bin("squeue", f'[ -e {cancelled} ] || echo "4242|RUNNING"')
    fake_bin("scancel", f"touch {cancelled}")
    monkeypatch.setattr(slurm, "CANCEL_POLL_S", 0)
    backend = slurm_backend()

    backend.cancel(backend.attach("4242"), wait_s=5)

    assert cancelled.exists()
    assert queue_state(Job("4242")) is None


def test_workflow_cleanup_cancels_the_runners_jobs_and_those_srtctl_submitted(fake_bin, tmp_path, monkeypatch):
    scancel, squeue = tmp_path / "scancel.log", tmp_path / "squeue.log"
    fake_bin("scancel", recorder(scancel))
    fake_bin("squeue", recorder(squeue))
    monkeypatch.setenv("USER", "runner")

    slurm.SlurmBackend.cleanup(None, "b300-dsxe_03")

    assert calls(scancel) == [
        ["--user=runner", "--name=b300-dsxe_03"], ["--user=runner", "--name=inferencex-b300-dsxe_03"],
    ]
    [query] = calls(squeue)
    assert "--name=b300-dsxe_03,inferencex-b300-dsxe_03" in query


def test_a_container_step_killed_by_a_signal_reports_the_shell_exit_code():
    step = subprocess.Popen(["bash", "-c", "kill -KILL $$"])
    step.wait()

    status = slurm_backend().state(slurm.SlurmJob("42", step=step))

    assert (status.state, status.exit_code) == (JobState.FAILED, 137)


def test_srun_rejects_env_value_that_would_split_export():
    with pytest.raises(ValueError, match="UCX_NET_DEVICES"):
        srun_argv(None, ["true"], container=ContainerSpec(image="i", env={"UCX_NET_DEVICES": "a,b"}))
