"""Record and stop only this wrapper's still-matching process or process group."""
import json
import os
from pathlib import Path
import signal
import sys
import time
from faulthandler_hook.registration import proc


def identity(pid):
    return proc(pid)


def matches(expected):
    try:
        actual = identity(expected['pid'])
        return all(actual[k] == expected[k] for k in ('pid', 'start_ticks', 'uid', 'pgid', 'sid', 'cgroup'))
    except (OSError, KeyError):
        return False


def stop_owned(expected, group=False):
    result = {'expected': expected, 'group': group, 'signals': []}
    # Check again immediately before EACH signal; a reused PID is never a target.
    for sig, grace in ((signal.SIGTERM, 5 if group else 1), (signal.SIGKILL, 1)):
        if not matches(expected):
            result['stopped_or_identity_changed'] = True
            break
        if group:
            if expected['pgid'] != expected['pid'] or expected['sid'] != expected['pid']:
                raise RuntimeError('benchmark is not the recorded dedicated session/group')
            os.killpg(expected['pgid'], sig)
        else:
            fd = os.pidfd_open(expected['pid'])
            try:
                if not matches(expected):
                    break
                signal.pidfd_send_signal(fd, sig)
            finally:
                os.close(fd)
        result['signals'].append(int(sig))
        until = time.monotonic() + grace
        while time.monotonic() < until and matches(expected):
            time.sleep(0.1)
    result['remaining_matching_leader'] = matches(expected)
    result['limit'] = 'If group leader is gone, surviving descendants remain owned by native Slurm cleanup.'
    return result


if __name__ == '__main__':
    mode, value, path = sys.argv[1:4]
    group = '--group' in sys.argv[4:]
    if mode == 'record':
        pid = int(value)
        for _ in range(20):
            record = identity(pid)
            if not group or (record['pgid'] == pid and record['sid'] == pid):
                break
            time.sleep(0.1)
        Path(path).write_text(json.dumps(record, indent=2) + '\n')
        if record['uid'] != proc(os.getpid())['uid'] or (group and (record['pgid'] != pid or record['sid'] != pid)):
            raise SystemExit('unexpected owner/session for launched process')
    elif mode == 'stop':
        saved = Path(value)
        result = stop_owned(json.loads(saved.read_text()), group) if saved.exists() else {'skipped': 'no recorded process identity; native cleanup owns remaining descendants'}
        Path(path).write_text(json.dumps(result, indent=2) + '\n')
    else:
        raise SystemExit('unknown owned process operation')
