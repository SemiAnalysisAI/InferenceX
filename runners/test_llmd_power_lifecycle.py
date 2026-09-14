"""CPU-only lifecycle checks with controlled external collector/Slurm processes."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / 'benchmarks/multi_node/llm-d/job.slurm'


@pytest.mark.parametrize('main_rc', [0, 7])
@pytest.mark.parametrize('enabled', [False, True])
def test_llmd_job_stages_native_evidence_and_keeps_main_step_status(tmp_path, main_rc, enabled):
    repo = tmp_path / 'repo'
    cwd = repo / 'benchmarks/multi_node/llm-d'
    cwd.mkdir(parents=True)
    model, logs, bindir = tmp_path / 'model', tmp_path / 'logs', tmp_path / 'bin'
    for path in (model, logs, bindir):
        path.mkdir()
    squash = tmp_path / 'image.sqsh'
    squash.write_text('fixture image')
    (bindir / 'scontrol').write_text('#!/bin/sh\nprintf "node-a\\nnode-b\\n"\n')
    (bindir / 'git').write_text('#!/bin/sh\necho 0123456789012345678901234567890123456789\n')
    (bindir / 'srun').write_text('#!' + sys.executable + '\n' + r'''
import json, os, pathlib, subprocess, sys
args = sys.argv[1:]
with open(os.environ['CALLS'], 'a') as f:
    f.write(json.dumps(args) + '\n')
if any(a.startswith('--container-image=') for a in args):
    if os.environ['POWERX_NATIVE_ENABLED'] != '1':
        assert not any('/powerx_native' in a for a in args)
        sys.exit(int(os.environ['MAIN_RC']))
    for rank in range(2):
        out = pathlib.Path(os.environ['POWERX_RAW_ROOT']) / f'node-{rank}'
        out.mkdir(parents=True, exist_ok=True)
        (out / 'manifest.json').write_text(json.dumps({'rank':rank,'synthetic':True}))
    sys.exit(int(os.environ['MAIN_RC']))
args = [a for a in args if not a.startswith('--')]
if 'ip route' in ' '.join(args):
    print('127.0.0.1')
    sys.exit(0)
for rank in range(2):
    subprocess.run(args, env={**os.environ,'SLURM_PROCID':str(rank)}, check=True)
''')
    for path in bindir.iterdir():
        path.chmod(0o755)
    defaults = dict.fromkeys('PREFILL_WORKERS DECODE_WORKERS PREFILL_DP_SIZE DECODE_DP_SIZE '
                            'BENCH_INPUT_LEN BENCH_OUTPUT_LEN BENCH_MAX_CONCURRENCY '
                            'BENCH_REQUEST_RATE BENCH_RANDOM_RANGE_RATIO BENCH_NUM_PROMPTS_MULTIPLIER '
                            'RUN_EVAL EVAL_ONLY EVAL_CONC EVAL_FRAMEWORK EVAL_LIMIT EVAL_SUITE '
                            'SWEBENCH_GEN_MODE SWEBENCH_USE_MODAL MODAL_TOKEN_ID MODAL_TOKEN_SECRET '
                            'IS_AGENTIC SCENARIO_TYPE FRAMEWORK PRECISION MODEL_PREFIX '
                            'RUNNER_TYPE RESULT_FILENAME SPEC_DECODING IS_MULTINODE CONFIG_FILE'.split(), '1')
    env = {**os.environ, **defaults, 'PATH':str(bindir) + os.pathsep + os.environ['PATH'],
           'SLURM_JOB_ID':f'local-{os.getpid()}-{main_rc}', 'SLURM_JOB_NODELIST':'node-[a-b]',
           'NUM_NODES':'2','PREFILL_NODES':'1','DECODE_NODES':'1','GPUS_PER_NODE':'2',
           'MODEL_DIR':str(model),'MODEL_NAME':'fixture','BENCHMARK_LOGS_DIR':str(logs),
           'LLMD_CONTAINER_ENGINE':'pyxis','LLMD_SQUASH_FILE':str(squash),
           'POWERX_NATIVE_ENABLED':'1' if enabled else '0','POWERX_RAW_ROOT':str(tmp_path/'raw'),
           'MAIN_RC':str(main_rc),'CALLS':str(tmp_path/'calls.jsonl')}
    result = subprocess.run(['bash', str(JOB)], cwd=cwd, env=env,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == main_rc, result.stderr + result.stdout
    if not enabled:
        assert not (repo / 'LOGS/native_power').exists()
        assert not (tmp_path / 'raw').exists()
        return
    for rank in range(2):
        saved = repo / f'LOGS/native_power/node-{rank}/manifest.json'
        assert json.loads(saved.read_text()) == {'rank':rank,'synthetic':True}
