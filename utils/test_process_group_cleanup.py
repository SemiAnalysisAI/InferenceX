"""Real process/pipe regressions for post-benchmark group teardown."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

LIBRARY = Path(__file__).resolve().parents[1] / "benchmarks/benchmark_lib.sh"


def stop_groups(status: int, *groups: int) -> subprocess.CompletedProcess[str]:
    # The client exits independently; the actual cleanup implementation must
    # preserve its code even when signaling and polling report success.
    return subprocess.run(
        [
            "bash",
            "-c",
            (
                'source "$1" --validation-only; shift; '
                'client_status=$1; shift; bash -c "exit $client_status"; status=$?; '
                'stop_background_process_groups "$status" 1 1 "$@"'
            ),
            "cleanup",
            str(LIBRARY),
            str(status),
            *map(str, groups),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=8,
    )


def await_file(path: Path) -> None:
    deadline = time.monotonic() + 5
    while not path.exists():
        assert time.monotonic() < deadline, "Child did not start"
        time.sleep(0.01)


@pytest.mark.parametrize("leader_exits, client_status", [(True, 0), (False, 19)])
def test_stubborn_descendant_releases_pipe_and_preserves_client_status(
    tmp_path: Path,
    leader_exits: bool,
    client_status: int,
) -> None:
    ready = tmp_path / "child-ready"
    child_code = (
        "import signal,pathlib,sys,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        'pathlib.Path(sys.argv[1]).touch(); print("worker output", flush=True); time.sleep(60)'
    )
    leader_code = (
        "import subprocess,sys,time; "
        'subprocess.Popen([sys.executable,"-c",sys.argv[1],sys.argv[2]]); '
        'time.sleep(0 if sys.argv[3]=="True" else 60)'
    )
    with (
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                leader_code,
                child_code,
                str(ready),
                str(leader_exits),
            ],
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ) as leader,
        subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"]
        ) as unrelated,
    ):
        try:
            await_file(ready)
            if leader_exits:
                assert leader.wait(timeout=3) == 0
            result = stop_groups(client_status, leader.pid)
            assert result.returncode == client_status, result.stderr
            assert "force-stopping owned process groups" in result.stdout
            # An orphan that retains stdout makes communicate hang even after
            # its leader exited. This tests the original tee-pipe failure.
            output, _ = leader.communicate(timeout=3)
            assert "worker output" in output
            assert unrelated.poll() is None
        finally:
            try:
                os.killpg(leader.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                # macOS can retain a zombie-only process group owned by init.
                pass
            unrelated.terminate()


def test_graceful_group_gets_term_without_kill(tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    stopped = tmp_path / "stopped"
    code = (
        "import signal,pathlib,sys,time; "
        "signal.signal(signal.SIGTERM, lambda *_: (pathlib.Path(sys.argv[2]).touch(), sys.exit(0))); "
        "pathlib.Path(sys.argv[1]).touch(); time.sleep(60)"
    )
    with subprocess.Popen(
        [sys.executable, "-c", code, str(ready), str(stopped)], start_new_session=True
    ) as leader:
        try:
            await_file(ready)
            result = stop_groups(0, leader.pid)
            assert result.returncode == 0, result.stderr
            assert leader.wait(timeout=2) == 0
            assert stopped.exists()
            assert "force-stopping" not in result.stdout
        finally:
            if leader.poll() is None:
                leader.kill()


@pytest.mark.parametrize("client_status, expected", [(0, 1), (23, 23)])
def test_refused_cleanup_cannot_hide_work_failure_or_report_success(
    client_status: int, expected: int
) -> None:
    result = stop_groups(client_status, 1)
    assert result.returncode == expected
    assert "refusing unsafe process-group cleanup" in result.stderr


def test_refuses_callers_own_group() -> None:
    result = subprocess.run(
        [
            "bash",
            "-c",
            (
                'source "$1" --validation-only; '
                'group=$(ps -o pgid= -p $$ | tr -d " "); stop_background_process_groups 0 1 1 "$group"'
            ),
            "cleanup",
            str(LIBRARY),
        ],
        start_new_session=True,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 1
    assert "refusing unsafe process-group cleanup" in result.stderr


@pytest.mark.parametrize("client_status, expected", [(0, 1), (23, 23)])
def test_signal_failure_is_bounded_and_preserves_work_failure(
    client_status: int, expected: int
) -> None:
    with subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    ) as leader:
        try:
            # Only mock the OS signal collaborator: the real liveness checks,
            # grace deadlines, and status selection all run against a live group.
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    (
                        'source "$1" --validation-only; '
                        'kill() { return 1; }; stop_background_process_groups "$2" 1 1 "$3"'
                    ),
                    "cleanup",
                    str(LIBRARY),
                    str(client_status),
                    str(leader.pid),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=6,
            )
            assert result.returncode == expected
            assert "still alive after KILL grace" in result.stderr
            assert leader.poll() is None
        finally:
            leader.kill()


@pytest.mark.parametrize("work_status", [0, 17])
def test_exit_handler_cleans_failed_startup_or_normal_work_and_umbp(
    tmp_path: Path, work_status: int
) -> None:
    ready = tmp_path / "group-ready"
    auxiliary_ready = tmp_path / "auxiliary-ready"
    auxiliary_stopped = tmp_path / "auxiliary-stopped"
    worker_code = (
        "import signal,pathlib,sys,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "pathlib.Path(sys.argv[1]).touch(); "
        'print("retained server output", flush=True); time.sleep(60)'
    )
    auxiliary_code = (
        "import signal,pathlib,sys,time; "
        "signal.signal(signal.SIGTERM, lambda *_: (pathlib.Path(sys.argv[2]).touch(), sys.exit(0))); "
        "pathlib.Path(sys.argv[1]).touch(); time.sleep(60)"
    )
    with (
        subprocess.Popen(
            [sys.executable, "-c", worker_code, str(ready)],
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ) as worker,
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                auxiliary_code,
                str(auxiliary_ready),
                str(auxiliary_stopped),
            ]
        ) as auxiliary,
    ):
        try:
            await_file(ready)
            await_file(auxiliary_ready)
            completed = subprocess.run(
                [
                    "bash",
                    "-c",
                    (
                        'source "$1" --validation-only; '
                        'owned_groups=("$3"); auxiliary_pid=$4; '
                        'trap \'exit_after_background_process_cleanup "$?" 1 1 '
                        '"$auxiliary_pid" "${owned_groups[@]}"\' EXIT; '
                        'bash -c "exit $2"; exit $?'
                    ),
                    "startup",
                    str(LIBRARY),
                    str(work_status),
                    str(worker.pid),
                    str(auxiliary.pid),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=7,
            )
            assert completed.returncode == work_status, completed.stderr
            assert "force-stopping owned process groups" in completed.stdout
            # EOF proves the failed startup cannot retain the outer tee pipe.
            output, _ = worker.communicate(timeout=2)
            assert output == "retained server output\n"
            assert worker.returncode == -signal.SIGKILL
            assert auxiliary.wait(timeout=2) == 0
            assert auxiliary_stopped.exists()
        finally:
            if worker.poll() is None:
                worker.kill()
            if auxiliary.poll() is None:
                auxiliary.kill()


@pytest.mark.parametrize("work_status, expected", [(0, 1), (17, 17)])
def test_exit_handler_runs_auxiliary_cleanup_even_if_group_cleanup_fails(
    work_status: int, expected: int
) -> None:
    with subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"]
    ) as auxiliary:
        try:
            completed = subprocess.run(
                [
                    "bash",
                    "-c",
                    (
                        'source "$1" --validation-only; auxiliary_pid=$3; '
                        'trap \'exit_after_background_process_cleanup "$?" 1 1 '
                        '"$auxiliary_pid" 1\' EXIT; exit "$2"'
                    ),
                    "startup",
                    str(LIBRARY),
                    str(work_status),
                    str(auxiliary.pid),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            assert completed.returncode == expected
            assert "refusing unsafe process-group cleanup" in completed.stderr
            assert auxiliary.wait(timeout=2) == -signal.SIGTERM
        finally:
            if auxiliary.poll() is None:
                auxiliary.kill()
