"""Recover the one known job from the cancelled round-2 diagnostic."""
import json
import os
import subprocess
import time
from pathlib import Path

JOB = '2168'
TERMINAL = {'CANCELLED', 'FAILED', 'COMPLETED', 'TIMEOUT', 'NODE_FAIL', 'PREEMPTED', 'OUT_OF_MEMORY'}


def accounting():
    text = subprocess.check_output([
        'sacct', '-X', '-j', JOB, '--starttime=2026-09-12', '--noheader', '--parsable2',
        '--format=JobIDRaw,JobName%100,User,Submit,State,ElapsedRaw,AllocTRES,ReqTRES',
    ], text=True)
    rows = [line.split('|') for line in text.splitlines() if line.split('|')[0] == JOB]
    if len(rows) != 1 or len(rows[0]) != 8:
        raise RuntimeError('Prior diagnostic accounting missing or ambiguous; no allocation allowed')
    job, name, user, submitted, state, elapsed, allocated, requested = rows[0]
    if (name != 'b300-dsxe_00' or user != 'sa-gha-runner'
            or submitted != '2026-09-12T08:58:43' or 'gres/gpu=8' not in requested.split(',')):
        raise RuntimeError('Prior job identity does not match this task receipt; no cleanup or allocation allowed')
    return dict(job=job, name=name, user=user, submitted=submitted, state=state,
                elapsed_seconds=int(elapsed), allocated_tres=allocated, requested_tres=requested)


def main():
    record = accounting()
    print(json.dumps(record), flush=True)
    active = subprocess.check_output(['squeue', '--me', '--noheader', '--format=%i'], text=True).split()
    if JOB in active:
        subprocess.run(['scancel', JOB], check=True)
        for _ in range(15):
            if JOB not in subprocess.check_output(['squeue', '--me', '--noheader', '--format=%i'], text=True).split():
                break
            time.sleep(2)
        else:
            raise RuntimeError('Owned prior job did not terminate; no allocation allowed')
    record = accounting()
    Path(os.environ['RUNNER_TEMP'], 'powerx3040', 'prior2168.json').write_text(json.dumps(record, indent=2)+'\n')
    if record['state'].split()[0] not in TERMINAL:
        raise RuntimeError('Prior job is not terminal in accounting; no allocation allowed')
    if record['elapsed_seconds'] != 0 or 'gres/gpu=' in record['allocated_tres']:
        raise RuntimeError('Prior job consumed allocation; root must recompute the remaining budget')
    print('PRIOR_2168_TERMINAL_ZERO_GPU=PASS', flush=True)


if __name__ == '__main__':
    main()
