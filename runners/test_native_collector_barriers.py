"""CPU-only lifecycle checks with controlled external collector/Slurm processes."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
LIFECYCLE = ROOT / 'benchmarks/native_power_lifecycle.sh'
JOB = ROOT / 'benchmarks/multi_node/llm-d/job.slurm'


@pytest.mark.parametrize('failed_rank', [None, 1])
def test_stop_waits_for_every_collector_and_retains_failure(tmp_path, failed_rank):
    command = '''
source "$1"
export POWERX_CONTROL_DIR="$2" POWERX_NUM_NODES=2 POWERX_BARRIER_TIMEOUT_S=5
(while [[ ! -f "$2/stop" ]]; do sleep 0.01; done; sleep 0.1; echo 0 > "$2/done-0") &
POWERX_COLLECTOR_PID=$!
(while [[ ! -f "$2/stop" ]]; do sleep 0.01; done; sleep 0.2; echo "$3" > "$2/done-1") &
remote_pid=$!
powerx_stop_collectors
rc=$?
wait "$remote_pid"
exit "$rc"
'''
    result = subprocess.run(['bash', '-c', command, 'bash', str(LIFECYCLE),
                             str(tmp_path), '1' if failed_rank else '0'],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == (1 if failed_rank else 0), result.stderr
    assert (tmp_path / 'done-0').read_text().strip() == '0'
    assert (tmp_path / 'done-1').read_text().strip() == ('1' if failed_rank else '0')


def test_ready_barrier_rejects_collector_that_already_stopped(tmp_path):
    (tmp_path / 'ready-0').write_text('ready')
    (tmp_path / 'done-0').write_text('1')
    result = subprocess.run(['bash', '-c', 'source "$1"; POWERX_CONTROL_DIR="$2"; '
                             'POWERX_NUM_NODES=1; powerx_wait_collectors ready',
                             'bash', str(LIFECYCLE), str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 1
    assert 'stopped before benchmark readiness' in result.stderr


