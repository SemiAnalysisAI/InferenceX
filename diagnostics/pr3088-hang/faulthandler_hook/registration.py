"""Task-only SIGUSR2 registration; target writes never block or alter file limits."""
import faulthandler
import json
import os
from pathlib import Path
import re
import signal
import time

SIGNAL = signal.SIGUSR2
MAX_REGISTRATIONS = 64
_fd = None
_owner = None


def proc(pid):
    p = Path('/proc') / str(pid)
    fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'ppid': int(fields[1]), 'pgid': int(fields[2]), 'sid': int(fields[3]),
            'start_ticks': int(fields[19]),
            'uid': p.stat().st_uid, 'state': fields[0],
            'cgroup': (p / 'cgroup').read_text(), 'uid_map': (p / 'uid_map').read_text(),
            'caught_signals': int(re.search(r'^SigCgt:\s*([0-9a-fA-F]+)', (p / 'status').read_text(), re.M)[1], 16)}


def owns_job(identity, job):
    return re.search(r'(?:^|/)job_' + re.escape(job) + r'(?:[/_.]|$)', identity['cgroup']) is not None


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def register():
    global _fd, _owner
    directory, job = os.environ.get('POWERX_STACK_DIR'), os.environ.get('POWERX_DIAG_JOB')
    if not directory:
        return
    if not job or not job.isdecimal():
        raise RuntimeError('missing native diagnostic job identity')
    # Fork inherits the C handler and its descriptor. Close only the child's copy.
    if _owner is not None:
        faulthandler.unregister(SIGNAL)
        if _fd is not None:
            os.close(_fd)
        _fd = None
    if signal.getsignal(SIGNAL) != signal.SIG_DFL:
        raise RuntimeError('SIGUSR2 already has a Python handler')
    identity = proc(os.getpid())
    if identity.get('caught_signals', 0) & (1 << (int(SIGNAL) - 1)):
        raise RuntimeError('SIGUSR2 already has a native handler')
    if not owns_job(identity, job):
        raise RuntimeError('registration process is not in the declared Slurm job')
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    for slot in range(MAX_REGISTRATIONS):
        try:
            (root / f'slot-{slot}').mkdir()
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError('diagnostic registration limit reached')
    name = f"{identity['pid']}-{identity['start_ticks']}"
    fifo = root / f'{name}.fifo'
    os.mkfifo(fifo, 0o600)
    # RDWR retains a reader even after observer shutdown; NONBLOCK prevents a
    # full pipe from blocking serving. No SIGPIPE handler or global rlimit changes.
    _fd = os.open(fifo, os.O_RDWR | os.O_NONBLOCK | os.O_CLOEXEC)
    _owner = os.getpid()
    faulthandler.register(SIGNAL, file=_fd, all_threads=True, chain=False)
    info = os.fstat(_fd)
    receipt = {'identity': identity, 'job': job, 'signal': int(SIGNAL),
               'registered_at': time.time(), 'fd': _fd, 'fifo': str(fifo),
               'device': info.st_dev, 'inode': info.st_ino,
               'boot_id': boot_id(),
               'all_threads': True, 'nonblocking': True, 'python_stack_only': True}
    # Publishing this receipt is deliberately AFTER faulthandler.register.
    temp = root / f'{name}.json.tmp'
    temp.write_text(json.dumps(receipt) + '\n')
    temp.replace(root / f'{name}.json')


def install():
    register()
    if hasattr(os, 'register_at_fork'):
        os.register_at_fork(after_in_child=register)
