"""Stop an owned benchmark/eval client when its ready server or worker exits."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess


def process_state(pid: int) -> tuple[str, str] | None:
    """A zombie is dead, and a reused PID is a different process."""
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return stat[0], stat[19]
    except FileNotFoundError:
        if Path('/proc').exists():
            return None
        # Local macOS checks; production runners use Linux /proc.
        result = subprocess.run(['ps', '-p', str(pid), '-o', 'stat=', '-o', 'lstart='],
                                capture_output=True, text=True, check=False)
        fields = result.stdout.strip().split(maxsplit=1)
        return (fields[0], fields[1]) if len(fields) == 2 else None


def snapshot(pid: int) -> dict[str, str]:
    if pid <= 0:
        raise ValueError('Expected a positive server PID')
    listing = subprocess.check_output(['ps', '-eo', 'pid=,ppid=,args='], text=True)
    processes = [line.strip().split(maxsplit=2) for line in listing.splitlines()]
    descendants = {pid}
    while True:
        expanded = descendants | {int(p[0]) for p in processes if len(p) == 3 and int(p[1]) in descendants}
        if expanded == descendants:
            break
        descendants = expanded
    # These are persistent required engine processes, not transient tokenizer or
    # HTTP request subprocesses. The root alone would miss a dead scheduler.
    worker = re.compile(r'sglang::scheduler|EngineCore|VllmWorker|TPWorker|trtllm-worker')
    required = {pid} | {int(p[0]) for p in processes if len(p) == 3 and int(p[0]) in descendants and worker.search(p[2])}
    result = {}
    for required_pid in required:
        state = process_state(required_pid)
        if not state or state[0].startswith(('Z', 'X')):
            raise ValueError('Server exited before the client started')
        result[str(required_pid)] = state[1]
    return result


def healthy(required: dict[str, str]) -> bool:
    return bool(required) and all((state := process_state(int(pid))) and state[1] == start
                                  and not state[0].startswith(('Z', 'X')) for pid, start in required.items())


def stop(client: subprocess.Popen) -> None:
    # The leader may already have exited while leaving its own workers behind.
    try:
        os.killpg(client.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        client.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(client.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        client.wait()


def run(required: dict[str, str], command: list[str], interval: float = 2) -> int:
    if not healthy(required):
        print('ERROR: ready server or required worker exited before client dispatch', flush=True)
        return 1
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        with subprocess.Popen(command, start_new_session=True) as client:
            try:
                while True:
                    try:
                        return client.wait(timeout=interval)
                    except subprocess.TimeoutExpired:
                        if not healthy(required):
                            print('ERROR: ready server or required worker exited; stopping its benchmark/eval client', flush=True)
                            return 1
            finally:
                stop(client)
    finally:
        signal.signal(signal.SIGTERM, previous)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    capture = commands.add_parser('capture')
    capture.add_argument('--pid', type=int, required=True)
    execute = commands.add_parser('run')
    execute.add_argument('--state', type=Path, required=True)
    execute.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.operation == 'capture':
        print(json.dumps(snapshot(args.pid)))
        return 0
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('Missing client command')
    return run(json.loads(args.state.read_text()), command)


if __name__ == '__main__':
    raise SystemExit(main())
