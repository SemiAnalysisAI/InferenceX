"""Slurm CLI wrappers of the Slurm backend against fake salloc/srun/squeue/sacct/scontrol binaries."""

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infx.clusters import load_inventory
from infx.launch.backends import slurm
from infx.launch.backends.base import Job, JobState
from infx.launch.backends.slurm import cli
from infx.launch.backends.slurm.cli import (
    ContainerSpec,
    Resources,
    SlurmError,
    final_status,
    queue_state,
    salloc,
    srun_argv,
    wait_for_log,
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


@pytest.mark.parametrize("record,state,exit_code,nodes", [
    ("COMPLETED|0:0|gpu[01-02]", JobState.SUCCEEDED, 0, "gpu[01-02]"),
    ("FAILED|1:0|gpu03", JobState.FAILED, 1, "gpu03"),
    ("COMPLETED|0:15", JobState.FAILED, 143, None),
    ("FAILED|0:9|gpu03", JobState.FAILED, 137, "gpu03"),
    ("OUT_OF_MEMORY|0:125|gpu03", JobState.FAILED, 0, "gpu03"),
    ("CANCELLED by 1001|0:0|None assigned", JobState.CANCELLED, 0, None),
])
def test_final_status_maps_allocation_accounting(fake_bin, tmp_path, record, state, exit_code, nodes):
    log = tmp_path / "sacct.log"
    fake_bin("sacct", f"{recorder(log)}\necho '{record}'")
    status = final_status(Job("42"), delay_s=0)
    assert (status.state, status.exit_code, status.nodes) == (state, exit_code, nodes)
    assert calls(log) == [["-X", "-n", "-P", "-j", "42", "--format=State,ExitCode,NodeList,Submit,Start,End"]]


def test_final_status_reads_the_accounting_times_as_epoch_seconds(fake_bin):
    fake_bin("sacct", 'if [ "$SLURM_TIME_FORMAT" = "%s" ]; then echo "FAILED|1:0|gpu03|1791547200|1791548100|Unknown"; '
             'else echo "FAILED|1:0|gpu03|2026-10-09T12:00:00|2026-10-09T12:15:00|Unknown"; fi')
    status = final_status(Job("42"), delay_s=0)
    assert (status.submitted_at, status.started_at, status.ended_at) == (
        datetime(2026, 10, 9, 12, 0, tzinfo=UTC), datetime(2026, 10, 9, 12, 15, tzinfo=UTC), None,
    )


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


@pytest.mark.parametrize("controller,state,nodes", [
    ("JobId=42 JobName=x JobState=COMPLETED Reason=None ExitCode=0:0 NodeList=gpu07", JobState.SUCCEEDED, "gpu07"),
    ("JobId=42 JobName=x JobState=FAILED Reason=NonZeroExitCode ExitCode=3:0 NodeList=(null)", JobState.FAILED, None),
    ("JobId=420 JobName=x JobState=COMPLETED ExitCode=0:0 NodeList=gpu07", JobState.UNKNOWN, None),
])
def test_final_status_falls_back_to_controller_without_sacct(fake_bin, controller, state, nodes):
    fake_bin("scontrol", f"echo '{controller}'")
    status = final_status(Job("42"), attempts=3, delay_s=0)
    assert (status.state, status.nodes) == (state, nodes)


def test_queue_state_reads_only_the_named_job(fake_bin):
    fake_bin("squeue", "echo '420|RUNNING'; echo '42|PENDING'")
    assert queue_state(Job("42")) == "PENDING"
    assert queue_state(Job("7")) is None


def test_queue_state_retries_a_failed_squeue_instead_of_reporting_the_job_gone(fake_bin, tmp_path, monkeypatch):
    failed = tmp_path / "failed"
    fake_bin("squeue", f"if [ ! -e {failed} ]; then touch {failed}; "
             "echo 'slurm_load_jobs error: Socket timed out' >&2; exit 1; fi; echo '42|RUNNING'")
    monkeypatch.setattr(cli, "SQUEUE_RETRY_S", 0)
    assert queue_state(Job("42")) == "RUNNING"


def test_waiting_for_a_log_fails_when_the_job_dies_before_it(fake_bin, tmp_path):
    log = tmp_path / "scontrol.log"
    fake_bin("squeue", "exit 0")
    fake_bin("scontrol", recorder(log))
    with pytest.raises(SlurmError, match="ended before creating"):
        wait_for_log(Job("77"), tmp_path / "missing.log", wait_s=0)
    assert calls(log) == [["show", "job", "77"]]


def test_waiting_for_a_log_accepts_one_created_just_before_the_job_left_the_queue(fake_bin, tmp_path, capsys):
    batch_log, scontrol = tmp_path / "batch.log", tmp_path / "scontrol.log"
    fake_bin("squeue", f"echo started > {batch_log}")
    fake_bin("scontrol", recorder(scontrol))
    wait_for_log(Job("77"), batch_log, wait_s=0)
    assert calls(scontrol) == []
    assert "ERROR" not in capsys.readouterr().err


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


def test_backend_cancel_scancels_even_when_squeue_fails(fake_bin, tmp_path, monkeypatch):
    log = tmp_path / "scancel.log"
    fake_bin("squeue", "exit 1")
    fake_bin("scancel", recorder(log))
    monkeypatch.setattr(cli, "SQUEUE_RETRY_S", 0)
    backend = slurm_backend()

    backend.cancel(backend.attach("4242"))

    assert calls(log) == [["4242"]]


def test_backend_cancel_skips_a_job_squeue_reports_as_purged(fake_bin, tmp_path, monkeypatch):
    queries, scancel = tmp_path / "squeue.log", tmp_path / "scancel.log"
    fake_bin("squeue", f"{recorder(queries)}\n"
             "echo 'slurm_load_jobs error: Invalid job id specified' >&2; exit 1")
    fake_bin("scancel", recorder(scancel))
    monkeypatch.setattr(cli, "SQUEUE_RETRY_S", 0)
    backend = slurm_backend()

    backend.cancel(backend.attach("4242"))

    assert len(calls(queries)) == 1
    assert calls(scancel) == []


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


def test_workflow_cleanup_stops_waiting_for_a_job_that_never_leaves_the_queue(fake_bin, monkeypatch, capsys):
    fake_bin("scancel", "exit 0")
    fake_bin("squeue", "echo 4242")
    monkeypatch.setattr(cli, "CANCEL_NAMED_WAIT_S", 0)

    def stuck(seconds: float) -> None:
        raise AssertionError("cleanup kept waiting past its deadline")

    monkeypatch.setattr(cli.time, "sleep", stuck)

    slurm.SlurmBackend.cleanup(None, "b300-dsxe_03")

    assert "WARNING: jobs 4242 still in the queue" in capsys.readouterr().err


def test_a_container_step_killed_by_a_signal_reports_the_shell_exit_code():
    step = subprocess.Popen(["bash", "-c", "kill -KILL $$"])
    step.wait()

    status = slurm_backend().state(slurm.SlurmJob("42", step=step))

    assert (status.state, status.exit_code) == (JobState.FAILED, 137)


@pytest.mark.parametrize(("allocation", "kind", "retriable"), [
    ("NODE_FAIL|0:0|gpu01", "JobNodeFail", True),
    ("RUNNING|0:0|gpu01", "JobFailed", False),
])  # fmt: skip
def test_a_container_step_on_a_lost_allocation_is_recorded_as_retriable(fake_bin, allocation, kind, retriable):
    fake_bin("sacct", f"echo '{allocation}'")
    step = subprocess.Popen(["bash", "-c", "exit 1"])
    step.wait()
    backend = slurm_backend()

    backend.state(slurm.SlurmJob("42", step=step))

    error = backend.life.event.finish(1).error
    assert (error.type, error.retriable) == (kind, retriable)


def test_a_running_allocation_records_no_end_from_the_controllers_projection(fake_bin):
    fake_bin("sacct", "echo 'sacct: error: Problem talking to the database' >&2; exit 1")
    # A running job's EndTime is its start plus the time limit, not when it ended.
    fake_bin("scontrol", "echo 'JobId=42 JobState=RUNNING ExitCode=0:0 NodeList=gpu01 "
             "SubmitTime=1791547200 StartTime=1791547260 EndTime=1791633660'")
    step = subprocess.Popen(["bash", "-c", "exit 0"])
    step.wait()
    backend = slurm_backend()

    backend.state(slurm.SlurmJob("42", step=step))

    record = backend.life.event.finish(0)
    assert (record.slurm_started_at, record.slurm_ended_at, record.queue_s) == (
        datetime(2026, 10, 9, 12, 1, tzinfo=UTC), None, 60.0,
    )


def test_srun_rejects_env_value_that_would_split_export():
    with pytest.raises(ValueError, match="UCX_NET_DEVICES"):
        srun_argv(None, ["true"], container=ContainerSpec(image="i", env={"UCX_NET_DEVICES": "a,b"}))
