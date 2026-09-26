"""Exercise the actual C48 watcher with only scheduler, clock and file inputs doubled."""

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "c48_watcher", ROOT / "runners/watch_glm52_b200_c48_cleanup.py"
)
watcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watcher)


def run_watch(
    tmp_path,
    monkeypatch,
    *,
    marker="",
    source="sweep",
    owner="valid",
    scheduler="normal",
    terminal_at=None,
):
    log = tmp_path / "logs/sweep_123.log"
    log.parent.mkdir()
    log.write_text(marker if source == "sweep" else "")
    if source == "benchmark":
        (log.parent / "benchmark.out").write_text(marker)
    clock = [0]
    calls = []
    cancelled = [False]

    def run(args, **kwargs):
        calls.append((clock[0], args))
        if args[0] == "scontrol":
            if scheduler == "unavailable":
                return subprocess.CompletedProcess(
                    args, 1, "", "controller unavailable"
                )
            uid = os.getuid() + (
                owner == "wrong_uid"
                or owner == "changes_at_deadline"
                and clock[0] >= 300
            )
            output = log if owner != "wrong_output" else log.parent / "another.log"
            job = "123" if owner != "wrong_job" else "124"
            name = "runner" if owner != "wrong_name" else "other-runner"
            terminal = (cancelled[0] and scheduler != "stuck_after_cancel") or (
                terminal_at is not None and clock[0] >= terminal_at
            )
            state = (
                "CANCELLED"
                if cancelled[0] and terminal
                else "COMPLETED"
                if terminal
                else "RUNNING"
            )
            text = f"JobId={job} UserId=task({uid}) JobName={name} StdOut={output} JobState={state} ExitCode=0:0"
            return subprocess.CompletedProcess(args, 0, text, "")
        if args[0] == "squeue":
            terminal = (
                cancelled[0]
                and scheduler != "stuck_after_cancel"
                or terminal_at is not None
                and clock[0] >= terminal_at
            )
            return subprocess.CompletedProcess(
                args,
                0,
                "" if terminal and scheduler != "terminal_active_steps" else "123.0\n",
                "",
            )
        if args[0] == "scancel":
            assert args == ("scancel", "123")
            cancelled[0] = True
            return subprocess.CompletedProcess(args, 0, "", "")
        raise AssertionError(args)

    monkeypatch.setattr(watcher.subprocess, "run", run)
    monkeypatch.setattr(watcher.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        watcher.time, "sleep", lambda _: clock.__setitem__(0, clock[0] + 100)
    )
    result = watcher.watch("123", log, "runner")
    receipt = json.loads((log.parent / "c48-cleanup-watcher.json").read_text())
    return result, receipt, calls


@pytest.mark.parametrize(
    "marker,source",
    [
        ("2026-09-26 05:40:00 [INFO] Cleanup\n", "sweep"),
        (
            "2026-09-26 05:40:00 [INFO] Cleaning up 12 processes (5 running)...\n",
            "sweep",
        ),
        ("ERROR Benchmark failed with exit code 1\n", "sweep"),
        ("ERROR Critical process 'decode_3' exited with code -9\n", "sweep"),
        (
            "Terminal warmup failure for trace x; aborting run early (broadcasting ProfileCancelCommand).\n",
            "benchmark",
        ),
        ("Run aborted (failed): Benchmark aborted. Will exit non-zero.\n", "benchmark"),
    ],
)
def test_explicit_terminal_markers_arm_300_second_exact_job_cancel(
    tmp_path, monkeypatch, marker, source
):
    result, receipt, calls = run_watch(
        tmp_path, monkeypatch, marker=marker, source=source
    )
    assert result == 1  # forced cleanup is never a successful benchmark
    assert [(at, args) for at, args in calls if args[0] == "scancel"] == [
        (300, ("scancel", "123"))
    ]
    assert receipt["status"] == "terminal_no_active_steps"
    assert receipt["last_active_steps"] == []


def test_scored_request_errors_do_not_arm_or_cancel(tmp_path, monkeypatch):
    result, receipt, calls = run_watch(
        tmp_path,
        monkeypatch,
        marker="WARNING request timed out; scored request error 500\n",
        source="benchmark",
        terminal_at=400,
    )
    assert result == 0
    assert not any(args[0] == "scancel" for _, args in calls)
    assert not any(
        event["event"] == "cleanup_grace_armed" for event in receipt["events"]
    )


@pytest.mark.parametrize(
    "owner", ["wrong_uid", "wrong_output", "wrong_job", "wrong_name"]
)
def test_mismatched_ownership_never_cancels(tmp_path, monkeypatch, owner):
    result, receipt, calls = run_watch(
        tmp_path, monkeypatch, marker="INFO Cleanup\n", owner=owner
    )
    assert result == 1 and receipt["status"] == "ownership_mismatch"
    assert not any(args[0] == "scancel" for _, args in calls)


def test_scheduler_inspection_failure_is_bounded_and_never_guesses_owner(
    tmp_path, monkeypatch
):
    result, receipt, calls = run_watch(
        tmp_path, monkeypatch, marker="INFO Cleanup\n", scheduler="unavailable"
    )
    assert result == 1 and receipt["status"] == "cleanup_unverified"
    assert len([args for _, args in calls if args[0] == "scontrol"]) == 3
    assert not any(args[0] == "scancel" for _, args in calls)


def test_cancel_without_terminal_cleanup_fails_boundedly(tmp_path, monkeypatch):
    result, receipt, calls = run_watch(
        tmp_path, monkeypatch, marker="INFO Cleanup\n", scheduler="stuck_after_cancel"
    )
    assert result == 1 and receipt["status"] == "cleanup_unverified"
    assert len([args for _, args in calls if args[0] == "scancel"]) == 1
    assert calls[-1][0] == 500  # 300-second grace + bounded cancellation wait


def test_successful_native_cleanup_before_grace_does_not_cancel(tmp_path, monkeypatch):
    result, receipt, calls = run_watch(
        tmp_path, monkeypatch, marker="INFO Cleanup\n", terminal_at=100
    )
    assert result == 0 and receipt["status"] == "terminal_no_active_steps"
    assert not any(args[0] == "scancel" for _, args in calls)


def test_ownership_is_rechecked_before_cancellation(tmp_path, monkeypatch):
    result, receipt, calls = run_watch(
        tmp_path, monkeypatch, marker="INFO Cleanup\n", owner="changes_at_deadline"
    )
    assert result == 1 and receipt["status"] == "ownership_mismatch"
    assert not any(args[0] == "scancel" for _, args in calls)


def test_terminal_allocation_with_active_steps_does_not_verify_cleanup(
    tmp_path, monkeypatch
):
    result, receipt, calls = run_watch(
        tmp_path,
        monkeypatch,
        marker="INFO Cleanup\n",
        scheduler="terminal_active_steps",
        terminal_at=100,
    )
    assert result == 1 and receipt["status"] == "cleanup_unverified"
    assert not any(args[0] == "scancel" for _, args in calls)
    assert calls[-1][0] == 300
