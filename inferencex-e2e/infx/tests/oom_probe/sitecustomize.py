import json
import mmap
import os
from pathlib import Path
import sys
import threading
import time


def memory_groups():
    mounts = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left, right = line.split(' - ', 1)
        cols, fs = left.split(), right.split()
        if fs[0] in ('cgroup', 'cgroup2'):
            mounts.append((cols[3], cols[4], fs[0], fs[2]))
    result = []
    for row in Path('/proc/self/cgroup').read_text().splitlines():
        _, controllers, group = row.split(':', 2)
        v2 = controllers == ''
        if not v2 and 'memory' not in controllers.split(','):
            continue
        for root, mount, fstype, opts in mounts:
            if (fstype == 'cgroup2') != v2 or (not v2 and 'memory' not in opts.split(',')):
                continue
            if group != root and not group.startswith(root.rstrip('/') + '/'):
                continue
            directory = Path(mount) / group.removeprefix(root).lstrip('/')
            while True:
                names = ('memory.max', 'memory.current', 'memory.events', 'memory.events.local', 'memory.swap.max') if v2 else ('memory.limit_in_bytes', 'memory.usage_in_bytes', 'memory.failcnt', 'memory.oom_control', 'memory.memsw.limit_in_bytes')
                files = {}
                for name in names:
                    try:
                        files[name] = directory.joinpath(name).read_text().strip()
                    except OSError:
                        pass
                if files:
                    result.append({'path': str(directory), 'files': files, 'version': 2 if v2 else 1})
                if directory == Path(mount) or directory.parent == directory:
                    break
                directory = directory.parent
    return result


def run_probe():
    expected = int(os.environ['INFX_OOM_EXPECTED_MIB']) * 2**20
    delay = int(os.environ['INFX_OOM_DELAY_S'])
    job = os.environ['SLURM_JOB_ID']
    step = os.environ['SLURM_STEP_ID']
    if not job.isdigit() or not step.isdigit():
        raise RuntimeError('probe requires numeric Slurm job and worker step')
    if not 0 <= delay <= 600:
        raise RuntimeError('delay must be between0 and600 seconds')
    output = Path(os.environ['INFX_OOM_OUTPUT_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    log = output / f'worker-{os.uname().nodename}-{os.getpid()}.jsonl'

    def emit(event, **values):
        row = {'event': event, 'time': time.time(), 'pid': os.getpid(), 'host': os.uname().nodename, 'job': job, 'step': step, **values}
        line = json.dumps(row, sort_keys=True)
        with log.open('a') as handle:
            handle.write(line + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        print('INFX_OOM_PROBE ' + line, flush=True)

    groups = memory_groups()
    finite = []
    for group in groups:
        key = 'memory.max' if group['version'] == 2 else 'memory.limit_in_bytes'
        value = group['files'].get(key, '')
        if value.isdigit() and 0 < int(value) < 2**60:
            finite.append((int(value), group['path']))
    meminfo = {k: int(v.split()[0]) * 1024 for k, v in (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
    cgroup = Path('/proc/self/cgroup').read_text()
    emit('armed', expected_bytes=expected, argv=sys.orig_argv, cgroup=cgroup, memory_groups=groups, memtotal_bytes=meminfo['MemTotal'], swap_total_bytes=meminfo['SwapTotal'], delay_seconds=delay)
    if not finite or min(x[0] for x in finite) != expected:
        raise RuntimeError(f'effective memory limit does not match expected {expected}: {finite}')
    if not 2**30 <= expected <= meminfo['MemTotal'] * 55 // 100:
        raise RuntimeError('probe cap must be1GiB through55% of physical node RAM')
    if meminfo['SwapTotal']:
        raise RuntimeError('probe requires a swap-free node to test resident-memory OOM deterministically')
    if f'job_{job}' not in cgroup or f'step_{step}' not in cgroup:
        raise RuntimeError('process is not in the expected Slurm worker step cgroup')
    if not any(limit == expected and f'step_{step}' in path for limit, path in finite):
        raise RuntimeError('expected memory cap must exist on the worker step, not only job ancestor')
    parent_limits = [limit for limit, path in finite if f'job_{job}' in path and f'step_{step}' not in path]
    if parent_limits and min(parent_limits) < expected * 5 // 4:
        raise RuntimeError('parent allocation must exceed worker cap by25% for helper isolation test')
    time.sleep(delay)
    Path('/proc/self/oom_score_adj').write_text('1000')
    target = expected * 11 // 10
    chunk = 64 * 2**20
    slab = b'\xa5' * chunk
    regions = []
    touched = 0
    started = time.monotonic()
    emit('pressure-start', target_bytes=target, memory_groups=memory_groups())
    while touched < target:
        if time.monotonic() - started > 600:
            emit('pressure-timeout', touched_bytes=touched)
            os._exit(87)
        region = mmap.mmap(-1, chunk, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        region.write(slab)
        regions.append(region)
        touched += chunk
        if touched % (4 * 2**30) == 0:
            emit('pressure-progress', touched_bytes=touched)
    emit('over-limit-survived', touched_bytes=touched, memory_groups=memory_groups())
    os._exit(86)


def guarded_probe():
    try:
        run_probe()
    except BaseException as error:
        print('INFX_OOM_PROBE ' + json.dumps({'event': 'probe-error', 'error': repr(error), 'pid': os.getpid()}), flush=True)
        os._exit(88)


if os.environ.get('INFX_OOM_PROBE') == '1' and any(sys.orig_argv[index:index + 2] == ['-m', 'sglang.launch_server'] for index in range(len(sys.orig_argv) - 1)):
    threading.Thread(target=guarded_probe, name='inference-worker-oom-probe', daemon=True).start()
