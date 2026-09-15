"""Client lifecycle regressions, using short real processes without a GPU server."""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from infx.bench_serving import server_watch


@pytest.mark.parametrize('state', [None, ('Z', 'original'), ('S', 'reused')])
def test_dead_zombie_and_reused_pids_are_not_healthy(monkeypatch, state):
    monkeypatch.setattr(server_watch, 'process_state', lambda _: state)
    assert not server_watch.healthy({'123': 'original'})


def test_client_exit_code_preserved_and_healthy_server_left_alone():
    with subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']) as server:
        try:
            state = server_watch.snapshot(server.pid)
            assert server_watch.run(state, [sys.executable, '-c', 'raise SystemExit(7)'], 0.02) == 7
            assert server.poll() is None
        finally:
            server.terminate()


def test_required_worker_death_stops_client_while_wrapper_lives(tmp_path: Path):
    pidfile = tmp_path / 'client.pid'
    with subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']) as wrapper, \
            subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'], stdin=subprocess.PIPE) as worker:
        try:
            state = {**server_watch.snapshot(wrapper.pid), **server_watch.snapshot(worker.pid)}
            command = [sys.executable, '-c',
                       'import os,time,pathlib,sys,signal; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); '
                       'os.kill(int(sys.argv[2]), signal.SIGTERM); time.sleep(30)',
                       str(pidfile), str(worker.pid)]
            started = time.monotonic()
            assert server_watch.run(state, command, 0.02) == 1
            assert time.monotonic() - started < 5
            assert wrapper.poll() is None
            assert pidfile.exists()
            observed = server_watch.process_state(int(pidfile.read_text()))
            assert observed is None or observed[0].startswith('Z')
        finally:
            wrapper.terminate()
            if worker.poll() is None:
                worker.terminate()


def test_exited_client_leader_does_not_leave_owned_worker(tmp_path: Path):
    pidfile = tmp_path / 'worker.pid'
    with subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']) as server:
        try:
            command = [sys.executable, '-c',
                       'import pathlib,subprocess,sys; p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]); pathlib.Path(sys.argv[1]).write_text(str(p.pid))',
                       str(pidfile)]
            assert server_watch.run(server_watch.snapshot(server.pid), command, 0.02) == 0
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                state = server_watch.process_state(int(pidfile.read_text()))
                if state is None or state[0].startswith('Z'):
                    break
                time.sleep(0.02)
            else:
                pytest.fail('Client exited but its owned worker survived')
            assert server.poll() is None
        finally:
            server.terminate()


def test_cleanup_accepts_a_group_with_only_an_exited_process():
    with subprocess.Popen([sys.executable, '-c', 'pass'], start_new_session=True) as client:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            state = server_watch.process_state(client.pid)
            if state and state[0].startswith('Z'):
                break
            time.sleep(0.01)
        else:
            pytest.fail('Client did not exit')
        server_watch.stop(client)
        assert client.wait() == 0


@pytest.mark.parametrize('leader_exits', [False, True])
def test_cleanup_does_not_hide_permission_denial_for_a_live_group(monkeypatch, leader_exits):
    command = ('import subprocess,sys; subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"])'
               if leader_exits else 'import time; time.sleep(30)')
    killpg = os.killpg
    with subprocess.Popen([sys.executable, '-c', command], start_new_session=True) as client:
        def denied(*args):
            raise PermissionError('denied')
        try:
            if leader_exits:
                client.wait(timeout=3)
            monkeypatch.setattr(os, 'killpg', denied)
            with pytest.raises(PermissionError, match='denied'):
                server_watch.stop(client)
            if not leader_exits:
                assert client.poll() is None
        finally:
            killpg(client.pid, signal.SIGKILL)


def test_cleanup_kills_a_client_that_ignores_termination(monkeypatch, tmp_path):
    ready = tmp_path / 'ready'
    command = 'import signal,pathlib,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); pathlib.Path(sys.argv[1]).touch(); time.sleep(30)'
    with subprocess.Popen([sys.executable, '-c', command, str(ready)], start_new_session=True) as client:
        try:
            deadline = time.monotonic() + 3
            while not ready.exists():
                assert time.monotonic() < deadline, 'Client did not become ready'
                time.sleep(0.01)
            wait = client.wait
            monkeypatch.setattr(client, 'wait', lambda timeout=None: wait(0.05 if timeout is not None else None))
            server_watch.stop(client)
            assert client.returncode == -signal.SIGKILL
        finally:
            if client.poll() is None:
                client.kill()
