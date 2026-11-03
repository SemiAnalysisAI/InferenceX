import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

import yaml


def step_id(cgroup):
    match = re.search(r'(?:^|/)step_([^/\n]+)', cgroup)
    return match.group(1) if match else None


def step_limit(probe, *, masked_namespace=False):
    if not isinstance(probe, dict) or 'error' in probe:
        return None
    finite = []
    for item in probe.get('limits', []):
        if not masked_namespace and not re.search(r'(?:^|/)step_', item['path']):
            continue
        values = item.get('values', item.get('files', {}))
        value = values.get('memory.max', values.get('memory.limit_in_bytes'))
        if isinstance(value, str) and value.isdigit() and int(value) < 2**60:
            finite.append(int(value))
    return min(finite) if finite else None


def names(details):
    result = {}
    for entry in re.split(r'(?=StepId=)', details):
        step = re.search(r'StepId=([0-9]+)\.([^\s]+)', entry)
        name = re.search(r'(?:^|\s)Name=([^\s]+)', entry)
        if step and name:
            result[step.group(2)] = name.group(1)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--available-mib', type=int, required=True)
    parser.add_argument('--gpus-per-node', type=int, required=True)
    parser.add_argument('--expected-worker-steps', required=True, help='JSON role:local-Slurm-step-count mapping, counting each node of a distributed step')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    expected_counts = json.loads(args.expected_worker_steps)
    recipe = yaml.safe_load(args.recipe.read_text())
    failures, missing, checks = [], [], []
    caps = {}
    for role, spec in recipe['roles'].items():
        mem = spec.get('srun_options', {}).get('mem')
        if not isinstance(mem, str) or not re.fullmatch(r'[1-9][0-9]*M', mem):
            raise ValueError('bound recipe role %s has no explicit MiB memory cap: %r' % (role, mem))
        cap = int(mem[:-1])
        possible_gpus = [g for g in range(1, args.gpus_per_node+1) if args.available_mib*g//args.gpus_per_node == cap]
        if len(possible_gpus) != 1:
            failures.append('role %s cap %s is not a unique GPU-proportional share' % (role, mem))
        caps[role] = {'bytes': cap*1048576, 'gpus': possible_gpus[0] if len(possible_gpus)==1 else None}
    if set(expected_counts) != set(caps):
        raise ValueError('expected-worker-steps roles must match bound recipe roles')
    observed = defaultdict(dict)
    helpers = {}
    for artifact in args.evidence.rglob('*.json'):
        data = json.loads(artifact.read_text())
        if not isinstance(data, dict) or not data.get('containers'):
            continue
        step_names = names(data.get('step_details', ''))
        for container in data['containers']:
            for process in container.get('workers', []):
                kind = process.get('kind', container.get('kind', 'worker'))
                host = process.get('host', {})
                step = step_id(host.get('cgroup', host.get('worker_cgroup', '')))
                key = (str(data.get('job_id')), data.get('node'), step)
                host_limit = step_limit(host)
                inside = process.get('container', {})
                same_pid = host.get('pid') == inside.get('pid') == process['pid']
                inside_limit = step_limit(inside, masked_namespace=same_pid and host_limit is not None)
                if not same_pid:
                    missing.append('host/container PID identity mismatch or absent for step %s' % (key,))
                check = {'artifact': str(artifact), 'kind': kind, 'pid': process['pid'], 'step': step, 'node': data.get('node'), 'host_step_limit_bytes': host_limit, 'container_step_limit_bytes': inside_limit}
                if kind == 'helper':
                    helpers[key] = check
                    if 'error' in host or not host.get('limits'):
                        missing.append('helper step %s lacks host cgroup evidence' % (key,))
                    if host_limit is not None and host_limit in {v['bytes'] for v in caps.values()}:
                        missing.append('helper step %s has a worker-sized cap; inspect its intended helper options' % (key,))
                    checks.append(check)
                    continue
                role = next((r for r in caps if step_names.get(step, '').startswith(r+'_')), None)
                if role is None and len(caps)==1:
                    role = next(iter(caps))
                check['role'] = role
                if role is None:
                    missing.append('cannot map worker step %s to a recipe role' % (key,))
                    checks.append(check)
                    continue
                check['expected_bytes'] = caps[role]['bytes']
                for label, value in [('host', host_limit), ('container', inside_limit)]:
                    if value is None:
                        missing.append('%s worker step %s has no readable finite step-local limit' % (label, key))
                    elif value != caps[role]['bytes']:
                        failures.append('%s worker step %s limit %s != expected %s' % (label, key, value, caps[role]['bytes']))
                devices = host.get('devices', {})
                visible = next((devices[k] for k in ('CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES') if devices.get(k)), None)
                if visible:
                    check['visible_devices'] = visible
                    count = len(visible.split(','))
                    if count != caps[role]['gpus']:
                        missing.append('worker step %s exposes %s devices but cap represents %s; inspect rank-local/MPI GPU visibility' % (key, count, caps[role]['gpus']))
                observed[role].setdefault(key, []).append(data.get('observed_epoch', 0))
                checks.append(check)
    for role, count in expected_counts.items():
        actual = observed[role]
        if len(actual) != count:
            missing.append('role %s has %s observed node/step identities, expected %s' % (role, len(actual), count))
        for key, timestamps in actual.items():
            if len(set(timestamps)) < 2 or max(timestamps)-min(timestamps) < 180:
                missing.append('worker step %s lacks startup and >=180s later observations' % (key,))
    if not helpers:
        missing.append('no helper container evidence')
    verdict = 'FAIL' if failures else ('INCOMPLETE' if missing else 'PASS')
    result = {'verdict': verdict, 'scope': 'cgroup limit evidence only; benchmark completion and OOM behavior require separate assessment', 'recipe': str(args.recipe), 'available_mib': args.available_mib, 'gpus_per_node': args.gpus_per_node, 'expected_caps': caps, 'failures': sorted(set(failures)), 'missing_evidence': sorted(set(missing)), 'checks': checks}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'checks'}, indent=2))
    return 0 if verdict == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
