"""Behavioral boundaries for the one approved Slurm lifecycle validation."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def validation(tmp_path):
    root = tmp_path / 'repo'
    scripts = root / 'benchmarks/multi_node/llm-d'
    scripts.mkdir(parents=True)
    shutil.copy(ROOT / 'benchmarks/multi_node/llm-d/submit.sh', scripts / 'submit.sh')
    recipes = scripts.parent / 'llm-d-recipes'
    recipes.mkdir()
    (recipes / 'dsv4-fp4-gb200-low-latency.yaml').write_text('slurm:\n  time_limit: "08:00:00"\n')
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    runner_temp = tmp_path / 'runner-temp'
    runner_temp.mkdir()
    out = root / 'LOGS/pr3052-validation'
    out.mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('SBATCH_', 'SLURM_'))}
    env.update(PATH=str(bindir) + os.pathsep + os.environ['PATH'], GITHUB_WORKSPACE=str(root),
               RUNNER_TEMP=str(runner_temp), GITHUB_RUN_ID='3052001', GITHUB_RUN_ATTEMPT='1',
               RUNNER_NAME='gb200-nv_0', TASK_JOB_NAME='powerx3052-3052001-1',
               TASK_JOB_USER=subprocess.check_output(['id', '-un'], text=True).strip(),
               VALIDATION_LOGS=str(out), VALIDATION_JOB_RECEIPT=str(runner_temp / 'powerx3052-3052001-1.job'),
               SLURM_ACCOUNT='benchmark', SLURM_PARTITION='batch', TIME_LIMIT='08:00:00',
               MODEL_PATH='/models/existing', MODEL_NAME='deepseek-ai/DeepSeek-V4-Pro', CONTAINER_IMAGE='existing',
               BENCHMARK_LOGS_DIR=str(root / 'benchmark_logs'), GPUS_PER_NODE='4',
               CONFIG_FILE='dsv4-fp4-gb200-low-latency.yaml', IS_AGENTIC='0', FRAMEWORK='llmd-vllm',
               MODEL_PREFIX='dsv4', PRECISION='fp4', IS_MULTINODE='true',
               ISL='8192', OSL='1024', CONC_LIST='1', PREFILL_NODES='2', DECODE_NODES='2', IMAGE='existing')
    stub = '''
import json, os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
out = pathlib.Path(os.environ['VALIDATION_LOGS'])
with (out / 'commands.jsonl').open('a') as f: f.write(json.dumps([name, *sys.argv[1:]]) + '\\n')
identity = os.environ['TASK_JOB_NAME'] + '|' + os.environ['TASK_JOB_USER']
if name == 'sbatch':
    print('4242')
elif name == 'scontrol':
    assert pathlib.Path(os.environ['VALIDATION_JOB_RECEIPT']).read_text().strip() == '4242'
    print('JobId=4242 JobName=' + os.environ['TASK_JOB_NAME'] + ' UserId=' + os.environ['TASK_JOB_USER'] + '(123) TimeLimit=' + os.environ.get('ACTUAL_LIMIT', '00:39:00') + ' Requeue=0 NumNodes=' + os.environ.get('ACTUAL_NODES', '4-4') + ' ReqTRES=cpu=16,node=4,gres/gpu=16')
elif name == 'squeue':
    mode = os.environ.get('QUEUE_MODE', 'empty')
    if mode == 'error': sys.exit(1)
    if mode == 'foreign': print('another-task|another-user')
    if mode == 'owned' and not (out / 'cancelled').exists():
        print('4242' if any(x.startswith('--name=') for x in sys.argv) else identity)
elif name == 'scancel':
    assert sys.argv[1:] == ['4242']
    (out / 'cancelled').touch()
elif name == 'date':
    count = out / 'clock-count'
    n = int(count.read_text()) if count.exists() else 0
    count.write_text(str(n + 1))
    print(1000 + n * 361)
elif name == 'sacct':
    state = os.environ.get('ACCOUNT_STATE', 'COMPLETED')
    if state != 'missing': print('4242|' + identity + '|' + state + '|0:0|start|end|00:00:01|gres/gpu=16|nodes')
else: sys.exit(99)
'''
    for name in ['sbatch', 'scontrol', 'squeue', 'scancel', 'sacct', 'enroot', 'srun', 'date']:
        path = bindir / name
        path.write_text('#!' + sys.executable + '\n' + stub)
        path.chmod(0o755)
    return scripts / 'submit.sh', env, out


def run_submit(validation, overrides=None, args=None):
    script, env, _ = validation
    return subprocess.run(['bash', str(script), *(args or ['2', '2', '8192', '1024', '1'])],
                          env={**env, **(overrides or {})}, text=True, capture_output=True, timeout=5)


def commands(out):
    path = out / 'commands.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_final_sbatch_cap_overrides_recipe_and_preserves_owned_receipt(validation):
    script, env, out = validation
    result = run_submit(validation)
    assert result.returncode == 0, result.stderr
    args = next(x for x in commands(out) if x[0] == 'sbatch')
    assert args[args.index('--time') + 1] == '00:39:00'
    assert args[args.index('-N') + 1] == '4'
    assert args[args.index('--gres=gpu:4')] == '--gres=gpu:4'
    assert '--no-requeue' in args and '--export=ALL' in args
    assert args[args.index('--job-name') + 1] == 'powerx3052-3052001-1'
    assert args[args.index('--chdir') + 1] == str(script.parent)
    assert args[-1] == str(script.parents[3] / 'runners/llmd_validation_job.sh')
    assert result.stdout.strip() == '4242'
    assert (out / 'job-id.txt').read_text().strip() == '4242'
    assert run_submit(validation).returncode != 0
    assert len([x for x in commands(out) if x[0] == 'sbatch']) == 1


@pytest.mark.parametrize('overrides,args', [
    ({'GPUS_PER_NODE': '8'}, None), ({'RUN_EVAL': 'true'}, None),
    ({'BENCH_NUM_PROMPTS_MULTIPLIER': '20'}, None), ({'GITHUB_RUN_ATTEMPT': '2'}, None),
    ({'SBATCH_ARRAY_INX': '0-3'}, None), ({'QUEUE_MODE': 'error'}, None),
    ({}, ['2', '2', '8192', '1024', '2']),
])
def test_invalid_requests_never_submit(validation, overrides, args):
    assert run_submit(validation, overrides, args).returncode != 0
    assert not any(x[0] == 'sbatch' for x in commands(validation[2]))


@pytest.mark.parametrize('queue,state,expected,cancelled', [
    ('owned', 'COMPLETED', 0, True), ('foreign', 'COMPLETED', 1, False),
    ('error', 'COMPLETED', 1, False), ('empty', 'missing', 1, False),
    ('empty', 'RUNNING', 1, False),
])
def test_cleanup_is_exact_owned_id_and_requires_terminal_accounting(validation, queue, state, expected, cancelled):
    _, env, out = validation
    Path(env['VALIDATION_JOB_RECEIPT']).write_text('4242\n')
    (out / 'request.txt').write_text(env['TASK_JOB_NAME'] + '|' + env['TASK_JOB_USER'] + '|4|4|00:39:00|C1|8192|1024|16|no-eval\n')
    result = subprocess.run(['bash', str(ROOT / 'runners/launch_gb200-nv.sh'), '--cleanup-validation'],
                            env={**env, 'QUEUE_MODE': queue, 'ACCOUNT_STATE': state},
                            text=True, capture_output=True, timeout=5)
    assert result.returncode == expected, result.stderr
    assert (out / 'cancelled').exists() == cancelled
    assert not any(x[0] in ['sbatch', 'srun', 'enroot'] for x in commands(out))


def test_missing_image_never_imports_or_allocates(validation):
    _, env, out = validation
    result = subprocess.run(['bash', str(ROOT / 'runners/launch_gb200-nv.sh')], env=env,
                            text=True, capture_output=True, timeout=5)
    assert result.returncode != 0
    assert 'refusing import' in result.stderr
    assert not any(x[0] in ['sbatch', 'srun', 'enroot'] for x in commands(out))


def test_wrong_effective_deadline_fails_after_preserving_job_id(validation):
    result = run_submit(validation, {'ACTUAL_LIMIT': '08:00:00'})
    assert result.returncode != 0
    assert (validation[2] / 'job-id.txt').read_text().strip() == '4242'


def test_cleanup_does_not_adopt_stale_workspace_request(validation):
    _, env, out = validation
    (out / 'request.txt').write_text('another-task\n')
    result = subprocess.run(['bash', str(ROOT / 'runners/launch_gb200-nv.sh'), '--cleanup-validation'],
                            env=env, text=True, capture_output=True, timeout=5)
    assert result.returncode != 0
    assert commands(out) == []


def test_interrupt_cleans_only_owned_allocation_and_preserves_failure(validation):
    _, env, out = validation
    # Calling cleanup directly would miss signal status propagation.
    prelude = (ROOT / 'runners/launch_gb200-nv.sh').read_text().split('\nexport SLURM_PARTITION="batch"', 1)[0]
    command = prelude + r'''
printf '%s\n' "$TASK_JOB_NAME|$TASK_JOB_USER|4|4|00:39:00|C1|8192|1024|16|no-eval" > "$VALIDATION_LOGS/request.txt"
printf '4242\n' > "$VALIDATION_JOB_RECEIPT"
kill -TERM "$$"
'''
    # The original source uses BASH_SOURCE to find the shared shell functions.
    script = out / 'interrupt-launcher.sh'
    script.write_text(command.replace('source "$(dirname "${BASH_SOURCE[0]}")/slurm_utils.sh"',
                                      'source "$SHARED_SLURM_UTILS"'))
    result = subprocess.run(['bash', str(script)],
                            env={**env, 'QUEUE_MODE': 'owned', 'SHARED_SLURM_UTILS': str(ROOT / 'runners/slurm_utils.sh')},
                            text=True, capture_output=True, timeout=5)
    assert result.returncode == 143, result.stderr
    assert (out / 'cancelled').exists()
    assert 'COMPLETED' in (out / 'cleanup-accounting.txt').read_text()
    assert [x for x in commands(out) if x[0] == 'scancel'] == [['scancel', '4242']]


@pytest.mark.parametrize('node_count,accepted', [('4', True), ('4-4', True), ('4-5', False), ('4-8', False)])
def test_effective_nodecount_accepts_only_exact_four(validation, node_count, accepted):
    result = run_submit(validation, {'ACTUAL_NODES': node_count})
    assert (result.returncode == 0) == accepted, result.stderr
    # Even rejected controller metadata must retain the ID for owned cleanup.
    assert (validation[2] / 'job-id.txt').read_text().strip() == '4242'
