"""Exit-code precedence and signal handling of the launcher lifecycle."""

import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from infx.launch.lifecycle import Lifecycle

ROOT = Path(__file__).resolve().parents[3]


def test_workload_rc_wins_over_failing_cleanups():
    ran = []

    def broken():
        ran.append("broken")
        raise OSError("scancel failed")

    with Lifecycle() as life:
        life.callback(ran.append, "first-registered")
        life.callback(broken)
        life.callback(lambda: 7)
        life.record(3)
        life.record(5)
    assert life.returncode == 3
    assert ran == ["broken", "first-registered"]


def test_cleanup_failure_fails_an_otherwise_green_run():
    with Lifecycle() as life:
        life.callback(lambda: 2)
        life.record(0)
    assert life.returncode == 1


def test_body_exception_runs_cleanups_and_propagates():
    ran = []
    with pytest.raises(RuntimeError), Lifecycle() as life:
        life.callback(ran.append, "cleanup")
        raise RuntimeError("boom")
    assert ran == ["cleanup"]
    assert life.returncode == 1


def test_sigterm_during_body_runs_cleanups_then_exits_143(tmp_path):
    marker = tmp_path / "cleaned"
    script = textwrap.dedent(f"""
        import sys, time
        from pathlib import Path
        from infx.launch.lifecycle import Lifecycle
        with Lifecycle() as life:
            life.callback(Path({str(marker)!r}).write_text, "done")
            print("ready", flush=True)
            time.sleep(60)
        sys.exit(0)
    """)
    child = subprocess.Popen(
        [sys.executable, "-c", script], cwd=ROOT, stdout=subprocess.PIPE, text=True
    )
    assert child.stdout.readline().strip() == "ready"
    child.send_signal(signal.SIGTERM)
    assert child.wait(timeout=20) == 143
    assert marker.read_text() == "done"


def test_cleanups_run_with_the_launch_signals_blocked_and_the_mask_is_restored():
    launch_signals = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
    masks = []
    with Lifecycle() as life:
        life.callback(lambda: masks.append(signal.pthread_sigmask(signal.SIG_BLOCK, [])))
    assert launch_signals <= masks[0]
    assert not launch_signals & signal.pthread_sigmask(signal.SIG_BLOCK, [])


def test_signal_during_cleanups_is_held_off_and_dropped(tmp_path):
    marker = tmp_path / "cleaned"
    script = textwrap.dedent(f"""
        import os, signal, sys
        from pathlib import Path
        from infx.launch.lifecycle import Lifecycle
        with Lifecycle() as life:
            life.callback(Path({str(marker)!r}).touch)
            life.callback(os.kill, os.getpid(), signal.SIGTERM)
        sys.exit(life.returncode)
    """)
    child = subprocess.run([sys.executable, "-c", script], cwd=ROOT, timeout=20, check=False)
    assert child.returncode == 0
    assert marker.exists()
