#!/usr/bin/env python3
"""One-run registered Python-stack observer; never changes serving settings."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import time
import stat
from faulthandler_hook.registration import SIGNAL, owns_job, proc

PIDS = re.compile(r'\((Worker_[^\s()]+|EngineCore(?:_[^\s()]+)?) pid=(\d+)\)')
THROUGHPUT = re.compile(r'Avg generation throughput: ([\d.]+).*?Running: (\d+) reqs')
ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
MAX_STACK_BYTES = 3 * 1024 * 1024
stop = False


def request_stop(_sig, _frame):
    global stop
    stop = True


def descendant(pid, root, start_ticks):
    current = proc(pid)
    if current['uid'] != proc(root)['uid']:
        raise RuntimeError('target UID does not match benchmark owner in this namespace')
    seen = set()
    node = current
    while node['pid'] != root:
        if node['pid'] <= 1 or node['pid'] in seen:
            raise RuntimeError('target not in this benchmark subtree')
        seen.add(node['pid'])
        node = proc(node['ppid'])
    if node['start_ticks'] != start_ticks:
        raise RuntimeError('benchmark root PID identity changed')
    return current


MAX_TOTAL_STACK_BYTES = 54 * 1024 * 1024
MAX_EVENT_BYTES = 2 * 1024 * 1024


def registered_target(pid, root, root_start, directory, job):
    actual = descendant(pid, root, root_start)
    receipt_path = directory / f"{pid}-{actual['start_ticks']}.json"
    receipt = json.loads(receipt_path.read_text())
    identity = receipt['identity']
    if (identity['pid'] != pid or identity['start_ticks'] != actual['start_ticks']
            or receipt['job'] != job or not owns_job(actual, job)
            or identity['uid_map'] != actual['uid_map']
            or receipt['signal'] != int(SIGNAL)
            or receipt['boot_id'] != Path('/proc/sys/kernel/random/boot_id').read_text().strip()):
        raise RuntimeError('registration identity does not match current job/process/kernel')
    fifo = Path(receipt['fifo'])
    if fifo.parent.resolve() != directory.resolve():
        raise RuntimeError('registration FIFO is outside this diagnostic directory')
    inode = fifo.stat()
    target_fd = (Path('/proc') / str(pid) / 'fd' / str(receipt['fd'])).stat()
    if (not stat.S_ISFIFO(inode.st_mode) or (inode.st_dev, inode.st_ino) != (receipt['device'], receipt['inode'])
            or (inode.st_dev, inode.st_ino) != (target_fd.st_dev, target_fd.st_ino)):
        raise RuntimeError('registered handler descriptor is no longer the observed FIFO')
    status = (Path('/proc') / str(pid) / 'status').read_text()
    caught = int(re.search(r'^SigCgt:\s*([0-9a-fA-F]+)', status, re.M)[1], 16)
    if not caught & (1 << (int(SIGNAL) - 1)):
        raise RuntimeError('registered signal is no longer caught')
    return actual, receipt


def capture_python(pid, root, root_start, directory, job, output, limit, readers):
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        raise RuntimeError('pidfd signal delivery is required; no unsafe PID-only fallback')
    pidfd = os.pidfd_open(pid)
    try:
        actual, receipt = registered_target(pid, root, root_start, directory, job)
        key = (pid, actual['start_ticks'])
        if key not in readers:
            readers[key] = os.open(receipt['fifo'], os.O_RDONLY | os.O_NONBLOCK)
        reader = readers[key]
        # No signal is sent until the full live registration/FD check passes.
        signal.pidfd_send_signal(pidfd, SIGNAL)
        seen = saved = 0
        began = last_data = time.monotonic()
        with output.open('wb') as stream:
            while not stop and time.monotonic() - began < 5:
                try:
                    data = os.read(reader, 65536)
                except BlockingIOError:
                    data = b''
                if data:
                    seen += len(data)
                    last_data = time.monotonic()
                    kept = data[:max(0, limit - saved)]
                    stream.write(kept)
                    saved += len(kept)
                elif seen and time.monotonic() - last_data >= 0.5:
                    break
                else:
                    time.sleep(0.01)
        return {'identity': actual, 'registered_at': receipt['registered_at'], 'bytes_read': seen,
                'bytes_saved': saved, 'observed_bytes_dropped': seen - saved,
                'timed_out': time.monotonic() - began >= 5,
                'possibly_truncated': True, 'python_stack_only': True,
                'limit_note': 'Nonblocking producer may drop bytes if FIFO fills; completeness is not claimed.'}
    finally:
        os.close(pidfd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=int, required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--registrations', type=Path, required=True)
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = proc(args.root)
    events = (args.output / 'events.jsonl').open('a', buffering=1)

    def event(kind, **fields):
        line = json.dumps({'at': time.time(), 'event': kind, **fields}) + '\n'
        if events.tell() + len(line.encode()) <= MAX_EVENT_BYTES:
            events.write(line)

    event('observer_started', root=root, tool='registered-faulthandler',
          trigger_seconds=60, rounds_max=2, capture_seconds=5, python_stack_only=True,
          stack_budget_bytes=MAX_TOTAL_STACK_BYTES, event_budget_bytes=MAX_EVENT_BYTES,
          output_limit_per_target=MAX_STACK_BYTES)
    readers = {}
    total_saved = 0
    offset, residual = 0, ''
    targets = {}
    seen_generation = False
    zero_since = None
    last_status = None
    last_round = None
    rounds = 0
    try:
        while not stop:
            try:
                current_root = proc(args.root)
                if current_root['start_ticks'] != root['start_ticks']:
                    break
            except FileNotFoundError:
                break
            try:
                with args.log.open('r', errors='replace') as source:
                    source.seek(offset)
                    # Read incrementally; do not copy full server logs or model data.
                    data = source.read(1024 * 1024)
                    offset = source.tell()
            except FileNotFoundError:
                data = ''
            lines = (residual + data).split('\n')
            residual = lines.pop()
            for line in lines:
                line = ANSI.sub('', line)
                for role, pid_text in PIDS.findall(line):
                    pid = int(pid_text)
                    if role not in targets:
                        try:
                            identity = descendant(pid, args.root, root['start_ticks'])
                            _, registration = registered_target(pid, args.root, root['start_ticks'], args.registrations, args.job)
                            readers[(pid, identity['start_ticks'])] = os.open(registration['fifo'], os.O_RDONLY | os.O_NONBLOCK)
                            targets[role] = identity
                            event('target_identified', role=role, identity=identity)
                        except (OSError, RuntimeError) as exc:
                            event('target_rejected', role=role, pid=pid, error=str(exc))
                            raise RuntimeError('Required worker registration unavailable; stop diagnostic') from exc
                match = THROUGHPUT.search(line)
                if match:
                    throughput, running = float(match[1]), int(match[2])
                    last_status = time.monotonic()
                    if throughput > 0:
                        seen_generation = True
                        zero_since = None
                    elif seen_generation and running > 0:
                        if zero_since is None:
                            zero_since = last_status
                            event('zero_throughput_started', running=running)
                    else:
                        zero_since = None
            now = time.monotonic()
            if (rounds < 2 and zero_since is not None and now - zero_since >= 60
                    and last_status is not None
                    and (last_round is None or now - last_round >= 30)):
                rounds += 1
                # Worker ranks first; the EngineCore is waiting for their response.
                selected = sorted(targets.items(), key=lambda item: (not item[0].startswith('Worker_'), item[0]))[:9]
                event('capture_round_started', round=rounds, targets=len(selected),
                      latest_status_age_seconds=now - last_status)
                for role, expected in selected:
                    if stop:
                        break
                    pid = expected['pid']
                    record = {'round': rounds, 'role': role, 'pid': pid}
                    try:
                        actual = descendant(pid, args.root, root['start_ticks'])
                        if actual['start_ticks'] != expected['start_ticks']:
                            raise RuntimeError('target PID identity changed')
                        record['identity'] = actual
                        path = args.output / f'round-{rounds}-{role}-{pid}.txt'
                        record['output'] = path.name
                        limit = min(MAX_STACK_BYTES, MAX_TOTAL_STACK_BYTES - total_saved)
                        if limit <= 0:
                            raise RuntimeError('total stack output budget exhausted')
                        record.update(capture_python(pid, args.root, root['start_ticks'],
                                      args.registrations, args.job, path, limit, readers))
                        total_saved += record['bytes_saved']
                    except (OSError, RuntimeError) as exc:
                        record['error'] = str(exc)
                    finally:
                        event('stack_capture', **record)
                last_round = time.monotonic()
                event('capture_round_finished', round=rounds)
            time.sleep(1 if data else 5)
    finally:
        for fd in readers.values():
            os.close(fd)
        event('observer_finished', rounds=rounds, signal_requested=stop)
        events.close()


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    main()
