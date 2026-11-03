import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time

PROBE = r'''
import json, os, posixpath, sys
from pathlib import Path
pid = int(sys.argv[1])
p = Path('/proc') / str(pid)
r = {'pid': pid, 'host': os.uname().nodename, 'cgroup': p.joinpath('cgroup').read_text(), 'limits': []}
host_cgroup = sys.argv[2] if len(sys.argv) > 2 else ''
namespace_cgroup = sys.argv[3] if len(sys.argv) > 3 else ''
def group_for(text, controllers):
    return next((row.split(':', 2)[2] for row in text.splitlines() if row.split(':', 2)[1] == controllers), None)
try:
    env = dict(x.split(b'=', 1) for x in p.joinpath('environ').read_bytes().split(b'\0') if b'=' in x)
    r['devices'] = {k: env[k.encode()].decode(errors='replace') for k in ('CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES') if k.encode() in env}
except OSError as exc:
    r['devices_error'] = str(exc)
for row in r['cgroup'].splitlines():
    _, controllers, group = row.split(':', 2)
    v2 = controllers == ''
    if not v2 and 'memory' not in controllers.split(','): continue
    for mount in Path('/proc/self/mountinfo').read_text().splitlines():
        left, right = mount.split(' - ', 1)
        fields, fs = left.split(), right.split()
        if fs[0] != ('cgroup2' if v2 else 'cgroup'): continue
        if not v2 and 'memory' not in fs[2].split(','): continue
        root, target = fields[3:5]
        if group == root or group.startswith(root.rstrip('/') + '/'):
            relative = group.removeprefix(root).lstrip('/')
        elif root == '/..' or root.startswith('/../'):
            host_group, namespace_group = group_for(host_cgroup, controllers), group_for(namespace_cgroup, controllers)
            if host_group is None or namespace_group is None: continue
            host_root = posixpath.normpath(namespace_group.rstrip('/') + '/' + root.lstrip('/'))
            if host_group != host_root and not host_group.startswith(host_root.rstrip('/') + '/'): continue
            relative = host_group.removeprefix(host_root).lstrip('/')
            r['namespace_mapping'] = {'mount_root': root, 'host_mount_root': host_root, 'host_target': host_group}
        else: continue
        if '..' in Path(relative).parts: continue
        directory = Path(target) / relative
        while True:
            entry = {'path': str(directory), 'values': {}}
            names = ('memory.max', 'memory.current', 'memory.peak', 'memory.events') if v2 else ('memory.limit_in_bytes', 'memory.usage_in_bytes', 'memory.max_usage_in_bytes', 'memory.failcnt')
            for name in names:
                try: entry['values'][name] = directory.joinpath(name).read_text().strip()
                except OSError as exc: entry['values'][name] = {'error': str(exc)}
            r['limits'].append(entry)
            if directory == Path(target) or directory.parent == directory: break
            directory = directory.parent
print(json.dumps(r))
'''

REMOTE = r'''
import hashlib, json, os, re, subprocess, sys
from pathlib import Path
job, skip = sys.argv[1], set(json.loads(sys.argv[2]))
PROBE = __PROBE__
report = {'node': os.uname().nodename, 'containers': [], 'errors': []}
try:
    result = subprocess.run(['enroot', 'list', '-f'], capture_output=True, text=True, timeout=20)
except (OSError, subprocess.TimeoutExpired) as exc:
    result = None
    report['errors'].append(str(exc))
containers = []
if result is not None:
    if result.returncode: report['errors'].append('enroot list: ' + result.stderr[-2000:])
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2 or not re.search(r'(?:^|[_-])' + re.escape(job) + r'(?:$|[_.-])', parts[0]): continue
        if not parts[1].isdigit(): continue
        containers.append((parts[0], int(parts[1])))
job_pattern = re.compile(r'(?:^|/)job[_-]' + re.escape(job) + r'(?:/|$)')
worker_pattern = re.compile(r'(?:sglang|vllm|trtllm|tensorrt_llm|SGLANG)', re.I)
workers = []
job_processes = []
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit(): continue
    try:
        cg = proc.joinpath('cgroup').read_text()
        if not job_pattern.search(cg): continue
        argv = proc.joinpath('cmdline').read_bytes().split(b'\0')
        executable = os.path.basename(argv[0].decode(errors='replace')) if argv else ''
        command = b' '.join(argv).decode(errors='replace')
        candidate = {'pid': int(proc.name), 'mount_namespace': os.readlink(proc/'ns/mnt'), 'executable': executable, 'command_sha256': hashlib.sha256(command.encode()).hexdigest()}
        job_processes.append(candidate)
        if not ('python' in executable.lower() or worker_pattern.search(executable)): continue
        match = worker_pattern.search(command)
        if not match: continue
        if any(part in command for part in ('pip install', 'pip download', '-m infx.', '-m srtctl.')): continue
        workers.append({'pid': int(proc.name), 'mount_namespace': os.readlink(proc/'ns/mnt'), 'executable': executable, 'engine_marker': match.group(), 'command_sha256': hashlib.sha256(command.encode()).hexdigest()})
    except (OSError, ValueError): pass
for name, pid in containers:
    try:
        namespace = os.readlink('/proc/%s/ns/mnt' % pid)
        namespace_cgroup = Path('/proc/%s/cgroup' % pid).read_text()
    except OSError: continue
    members = [w for w in workers if w['mount_namespace'] == namespace]
    kind = 'worker' if members else 'helper'
    if str(pid) + ':' + kind in skip: continue
    if not members:
        helpers = [w for w in job_processes if w['mount_namespace'] == namespace]
        helpers.sort(key=lambda w: (w['executable'] in ('bash', 'sh', 'dash'), w['pid'] != pid, w['pid']))
        members = helpers[:1]
    if not members: continue
    record = {'name': name, 'container_pid': pid, 'mount_namespace': namespace, 'kind': kind, 'workers': []}
    # Each process in one Slurm step inherits its cgroup; retain one representative per path.
    seen = set()
    for worker in members:
        try: cg = Path('/proc/%s/cgroup' % worker['pid']).read_text()
        except OSError: continue
        if cg in seen: continue
        seen.add(cg)
        evidence = {**worker, 'kind': kind}
        for label, argv in [('host', ['python3', '-c', PROBE, str(worker['pid'])]), ('container', ['enroot', 'exec', str(pid), 'python3', '-c', PROBE, str(worker['pid']), cg, namespace_cgroup])]:
            try:
                result = subprocess.run(argv, capture_output=True, text=True, timeout=20)
                if result.returncode:
                    evidence[label] = {'error': result.stderr[-2000:], 'returncode': result.returncode}
                else:
                    evidence[label] = json.loads(result.stdout)
            except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                evidence[label] = {'error': str(exc)}
        record['workers'].append(evidence)
    report['containers'].append(record)
report['discovered_container_count'] = len(containers)
report['discovered_worker_count'] = len(workers)
print(json.dumps(report))
'''.replace('__PROBE__', repr(PROBE))


WATCH = "import hashlib\nimport json\nimport os\nfrom pathlib import Path\nimport re\nimport signal\nimport sys\nimport time\n\njob = sys.argv[1]\njob_pattern = re.compile(r'(?:^|/)job[_-]' + re.escape(job) + r'(?:/|$)')\nworker_pattern = re.compile(r'sglang|vllm|trtllm|tensorrt_llm', re.I)\nhelper_pattern = re.compile(r'dynamo\\.frontend|nginx|etcd|otelcol|exporter|tachometer|agentx|aiperf|nats-server|mooncake', re.I)\nopened = {}\nprevious = {}\nknown = {}\nstopped = False\nstarted = time.monotonic()\nlast_heartbeat = 0\nidle = 0\n\ndef stop(*_):\n    global stopped\n    stopped = True\n\nfor sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):\n    signal.signal(sig, stop)\n\ndef emit(value):\n    print(json.dumps({'epoch': time.time(), 'node': os.uname().nodename, 'job_id': job, **value}), flush=True)\n\ndef identity(proc):\n    raw = proc.joinpath('stat').read_text()\n    return raw[raw.rfind(')')+2:].split()[19]\n\ndef alive(record):\n    try:\n        p = Path('/proc') / str(record['pid'])\n        stat = p.joinpath('stat').read_text()\n        return identity(p) == record['starttime'] and stat[stat.rfind(')')+2:].split()[0] != 'Z'\n    except OSError:\n        return False\n\nmounts = []\nfor line in Path('/proc/self/mountinfo').read_text().splitlines():\n    left, right = line.split(' - ', 1)\n    fields, fs = left.split(), right.split()\n    if fs[0] == 'cgroup2' or (fs[0] == 'cgroup' and 'memory' in fs[2].split(',')):\n        mounts.append((fields[3], fields[4], fs[0] == 'cgroup2'))\n\nemit({'type': 'watch-start', 'interval_seconds': 1})\nwhile not stopped and time.monotonic() - started < 43200:\n    paths = set()\n    count = 0\n    new_process = False\n    for proc in Path('/proc').iterdir():\n        if not proc.name.isdigit():\n            continue\n        try:\n            cg = proc.joinpath('cgroup').read_text()\n            if not job_pattern.search(cg):\n                continue\n            count += 1\n            for row in cg.splitlines():\n                _, controllers, group = row.split(':', 2)\n                v2 = controllers == ''\n                if not v2 and 'memory' not in controllers.split(','):\n                    continue\n                for root, target, mount_v2 in mounts:\n                    if v2 != mount_v2 or (group != root and not group.startswith(root.rstrip('/')+'/')):\n                        continue\n                    directory = Path(target)/group.removeprefix(root).lstrip('/')\n                    while True:\n                        paths.add((str(directory), v2))\n                        if directory == Path(target) or directory.parent == directory:\n                            break\n                        directory = directory.parent\n            argv = proc.joinpath('cmdline').read_bytes().split(b'\\0')\n            command = b' '.join(argv).decode(errors='replace')\n            executable = os.path.basename(argv[0].decode(errors='replace')) if argv else ''\n            role = 'worker' if worker_pattern.search(command) else ('helper' if helper_pattern.search(command) else None)\n            if not role or executable in ('bash', 'sh', 'dash'):\n                continue\n            ident = (int(proc.name), identity(proc))\n            if ident not in known:\n                known[ident] = {'pid': ident[0], 'starttime': ident[1], 'role': role, 'executable': executable, 'command_sha256': hashlib.sha256(command.encode()).hexdigest(), 'cgroup': cg}\n                new_process = True\n        except (OSError, ValueError):\n            pass\n    idle = idle + 1 if not count else 0\n    for directory, v2 in paths:\n        files = ('memory.events', 'memory.events.local') if v2 else ('memory.failcnt', 'memory.oom_control')\n        for name in files:\n            path = str(Path(directory)/name)\n            if path in opened:\n                continue\n            try:\n                opened[path] = os.open(path, os.O_RDONLY)\n            except OSError:\n                pass\n    values = {}\n    for path, fd in opened.items():\n        try:\n            os.lseek(fd, 0, os.SEEK_SET)\n            values[path] = os.read(fd, 65536).decode().strip()\n        except OSError as exc:\n            values[path] = {'error': str(exc)}\n    changed = {key: value for key,value in values.items() if previous.get(key) != value}\n    heartbeat = time.monotonic() - last_heartbeat >= 30\n    if changed or new_process or heartbeat:\n        emit({'type': 'events', 'changes': changed, 'all_counters': values if heartbeat else None, 'processes': [{**record, 'alive': alive(record)} for record in known.values()], 'job_process_count': count})\n        last_heartbeat = time.monotonic() if heartbeat else last_heartbeat\n    previous = values\n    if idle >= 30:\n        break\n    time.sleep(1)\nemit({'type': 'watch-stop', 'job_empty_seconds': idle})\nfor fd in opened.values():\n    os.close(fd)\n"


def command(argv, **kwargs):
    return subprocess.run(argv, capture_output=True, text=True, timeout=45, check=True, **kwargs).stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--runner', required=True)
    parser.add_argument('--stop-file', type=Path, required=True)
    parser.add_argument('--later-seconds', type=int, default=180)
    parser.add_argument('--oom-watch', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    workspace = args.workspace.resolve()
    job_name = 'inferencex-' + args.runner
    seen = {}
    watchers = {}
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    metadata = {'run_id': os.environ.get('GITHUB_RUN_ID'), 'run_attempt': os.environ.get('GITHUB_RUN_ATTEMPT'), 'runner': args.runner, 'workspace': str(workspace), 'job_name': job_name, 'started_epoch': time.time()}
    metadata['measured_git_sha'] = command(['git', '-C', str(workspace), 'rev-parse', 'HEAD']).strip()
    (args.output/'monitor.json').write_text(json.dumps(metadata, indent=2))
    while not stopped and not args.stop_file.exists():
        now = time.time()
        try:
            lines = command(['squeue', '--noheader', '--name', job_name, '--states', 'RUNNING', '--format', '%i|%j|%N']).splitlines()
            for line in lines:
                job, name, nodelist = line.strip().split('|', 2)
                if name != job_name or not re.fullmatch(r'\d+', job): continue
                details = command(['scontrol', 'show', 'job', '-o', job])
                submitted = re.search(r'(?:^|\s)SubmitTime=(\S+)', details)
                if not submitted or datetime.fromisoformat(submitted.group(1)).timestamp() < metadata['started_epoch'] - 5:
                    continue
                match = re.search(r'(?:^|\s)WorkDir=(\S+)', details)
                if not match: continue
                workdir = Path(match.group(1)).resolve()
                run_component = re.compile(r'srt-slurm-' + re.escape(str(metadata['run_id'])) + '-' + re.escape(str(metadata['run_attempt'])) + '-' + re.escape(args.runner) + r'-[0-9]+')
                shared_run = any(run_component.fullmatch(part) for part in workdir.parts)
                if workdir != workspace and workspace not in workdir.parents and not shared_run:
                    raise ValueError('refusing job %s with WorkDir outside current workspace: %s' % (job, workdir))
                (args.output/('%s-job.txt' % job)).write_text(details)
                steps = subprocess.run(['scontrol', 'show', 'step', job], capture_output=True, text=True, timeout=30)
                (args.output/('%s-steps.txt' % job)).write_text(steps.stdout + steps.stderr)
                for node in command(['scontrol', 'show', 'hostnames', nodelist]).splitlines():
                    prefix = (job, node)
                    if args.oom_watch and prefix not in watchers:
                        stream = (args.output/('%s-%s-oom-events.jsonl' % prefix)).open('w')
                        errors = (args.output/('%s-%s-oom-watch-errors.log' % prefix)).open('w')
                        child = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', node, shlex.join(['python3', '-', job])], stdin=subprocess.PIPE, stdout=stream, stderr=errors, text=True)
                        child.stdin.write(WATCH)
                        child.stdin.close()
                        watchers[prefix] = (child, stream, errors)
                    skip = [key[2] + ':' + key[3] for key, state in seen.items() if key[:2] == prefix and (state['count'] >= 2 or now - state['first'] < args.later_seconds)]
                    remote_command = shlex.join(['python3', '-', job, json.dumps(skip)])
                    text = command(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', node, remote_command], input=REMOTE)
                    report = json.loads(text)
                    report.update(job_id=job, observed_epoch=now, workdir=str(workdir), measured_git_sha=metadata['measured_git_sha'], job_details=details, step_details=steps.stdout)
                    for container in report['containers']:
                        key = (*prefix, str(container['container_pid']), container['kind'])
                        state = seen.setdefault(key, {'first': now, 'count': 0})
                        state['count'] += 1
                        filename = '%s-%s-pid%s-%s-%s.json' % (job, re.sub(r'[^A-Za-z0-9_.-]', '_', node), key[2], key[3], state['count'])
                        (args.output/filename).write_text(json.dumps({**report, 'containers': [container]}, indent=2))
                    (args.output/('%s-%s-status.json' % (job, node))).write_text(json.dumps({k: v for k,v in report.items() if k != 'containers'}, indent=2))
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            with (args.output/'errors.jsonl').open('a') as output:
                output.write(json.dumps({'epoch': now, 'type': type(exc).__name__, 'error': str(exc), 'stderr': str(getattr(exc, 'stderr', '') or '')[-2000:]})+'\n')
        deadline = time.monotonic() + 30
        while not stopped and not args.stop_file.exists() and time.monotonic() < deadline:
            time.sleep(1)
    for child, stream, errors in watchers.values():
        if child.poll() is None:
            child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
        stream.close()
        errors.close()
    (args.output/'completed.json').write_text(json.dumps({'finished_epoch': time.time(), 'containers': [{'job': k[0], 'node': k[1], 'pid': k[2], 'kind': k[3], **v} for k,v in seen.items()]}, indent=2))


if __name__ == '__main__':
    main()
