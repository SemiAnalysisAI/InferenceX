"""Exercise llm-d shutdown with local stand-ins for the Slurm/container boundary."""
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / 'benchmarks/multi_node/llm-d/job.slurm'
SERVER = ROOT / 'benchmarks/multi_node/llm-d/server.sh'


@pytest.mark.parametrize('main_rc', [0, 7])
def test_llmd_completed_coordinator_does_not_cancel_allocation(tmp_path, main_rc):
    cwd = tmp_path / 'repo/benchmarks/multi_node/llm-d'
    cwd.mkdir(parents=True)
    model, logs, bindir = tmp_path / 'model', tmp_path / 'logs', tmp_path / 'bin'
    for path in (model, logs, bindir):
        path.mkdir()
    squash = tmp_path / 'image.sqsh'
    squash.write_text('fixture image')
    commands = {
        'scontrol': '#!/bin/sh\nprintf "node-a\\nnode-b\\n"\n',
        'scancel': '#!/bin/sh\necho unexpected-cancel >> "$CANCEL_RECEIPT"\n',
        'sleep': '#!/bin/sh\n/bin/sleep 0.01\n',
        'srun': '#!' + sys.executable + '\n' + r'''
import os, pathlib, subprocess, sys, time
args = sys.argv[1:]
if any(a.startswith('--container-image=') for a in args):
    marker = pathlib.Path(os.environ['BENCHMARK_LOGS_DIR']) / ('.bench_done.' + os.environ['SLURM_JOB_ID'])
    marker.write_text(os.environ['MAIN_RC'] + '\n')
    time.sleep(0.2)
    sys.exit(int(os.environ['MAIN_RC']))
if 'ip route' in ' '.join(args):
    print('127.0.0.1')
    sys.exit(0)
args = [arg for arg in args if not arg.startswith('--')]
sys.exit(subprocess.run(args).returncode)
''',
    }
    for name, script in commands.items():
        path = bindir / name
        path.write_text(script)
        path.chmod(0o755)
    defaults = dict.fromkeys('PREFILL_WORKERS DECODE_WORKERS PREFILL_DP_SIZE DECODE_DP_SIZE '
                            'BENCH_INPUT_LEN BENCH_OUTPUT_LEN BENCH_MAX_CONCURRENCY '
                            'BENCH_REQUEST_RATE BENCH_RANDOM_RANGE_RATIO BENCH_NUM_PROMPTS_MULTIPLIER '
                            'RUN_EVAL EVAL_ONLY EVAL_CONC EVAL_FRAMEWORK EVAL_LIMIT EVAL_SUITE '
                            'SWEBENCH_GEN_MODE SWEBENCH_USE_MODAL MODAL_TOKEN_ID MODAL_TOKEN_SECRET '
                            'IS_AGENTIC SCENARIO_TYPE FRAMEWORK PRECISION MODEL_PREFIX '
                            'RUNNER_TYPE RESULT_FILENAME SPEC_DECODING IS_MULTINODE CONFIG_FILE'.split(), '1')
    env = {**os.environ, **defaults, 'PATH': str(bindir) + os.pathsep + os.environ['PATH'],
           'SLURM_JOB_ID': 'local-fixture', 'SLURM_JOB_NODELIST': 'node-[a-b]',
           'NUM_NODES': '2', 'PREFILL_NODES': '1', 'DECODE_NODES': '1', 'GPUS_PER_NODE': '2',
           'MODEL_DIR': str(model), 'MODEL_NAME': 'fixture', 'BENCHMARK_LOGS_DIR': str(logs),
           'LLMD_CONTAINER_ENGINE': 'pyxis', 'LLMD_SQUASH_FILE': str(squash),
           'MAIN_RC': str(main_rc), 'CANCEL_RECEIPT': str(tmp_path / 'cancelled')}
    result = subprocess.run(['bash', str(JOB)], cwd=cwd, env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == main_rc, result.stderr + result.stdout
    assert not (tmp_path / 'cancelled').exists()


@pytest.mark.parametrize('node_rc', [0, 7])
def test_llmd_node_records_status_and_stops_owned_server(tmp_path, node_rc):
    function = re.search(r'^finish_llmd_node\(\) \{\n.*?^\}',
                         SERVER.read_text(), flags=re.MULTILINE | re.DOTALL).group()
    command = function + r'''
NODE_RANK=1 PREFILL_NODES=1
BENCH_DONE_MARKER="$1/done"
sleep 60 &
VLLM_PID=$!
trap finish_llmd_node EXIT
exit "$2"
'''
    result = subprocess.run(['bash', '-c', command, 'bash', str(tmp_path), str(node_rc)],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == node_rc, result.stderr
    assert (tmp_path / 'done').read_text().strip() == str(node_rc)
