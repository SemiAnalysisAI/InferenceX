"""CPU-only lifecycle checks with controlled external collector/Slurm processes."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(('image', 'expected_uri', 'cached'), [
    ('ghcr.io/tile-ai/tilert:0.1.5', 'docker://ghcr.io#tile-ai/tilert:0.1.5', None),
    ('vllm/vllm-openai:v0.26.0', 'docker://vllm/vllm-openai:v0.26.0', None),
    ('ghcr.io#tile-ai/tilert:0.1.5', 'docker://ghcr.io#tile-ai/tilert:0.1.5', None),
    ('docker://ghcr.io#tile-ai/tilert:0.1.5', 'docker://ghcr.io#tile-ai/tilert:0.1.5', None),
    ('registry.example:5000/team/image:tag', 'docker://registry.example:5000#team/image:tag', None),
    ('ghcr.io/tile-ai/tilert:0.1.5', None, 'prepared image'),
    pytest.param('ghcr.io/tile-ai/tilert:0.1.5', 'docker://ghcr.io#tile-ai/tilert:0.1.5',
                 'partial image', id='partial-cache'),
])
def test_tilert_import_uses_registry_and_reuses_squash(tmp_path, image, expected_uri, cached):
    bindir, squash = tmp_path / 'bin', tmp_path / 'squash'
    bindir.mkdir()
    squash.mkdir()
    image_file = squash / (image.translate(str.maketrans('/:@#', '____')) + '.sqsh')
    if cached is not None:
        image_file.write_text(cached)
    scripts = {
        'scontrol': '#!/bin/sh\nprintf "node-a\\nnode-b\\n"\n',
        'flock': '#!/bin/sh\nexit 0\n',
        'unsquashfs': '#!/bin/sh\ngrep -qxE "prepared image|imported image" "$2"\n',
        'enroot': '#!' + sys.executable + '\n' + '''
import json, os, sys
assert sys.argv[1:3] == ['import', '-o']
with open(os.environ['IMPORT_RECEIPT'], 'a') as receipt:
    receipt.write(json.dumps(sys.argv[4:]) + '\\n')
with open(sys.argv[3], 'x') as image:
    image.write('imported image')
''',
        'srun': '#!' + sys.executable + '\n' + '''
import os, subprocess, sys
args = sys.argv[1:]
if any(arg.startswith('--container-image=') for arg in args):
    sys.exit(0)
command = args[next(i for i, arg in enumerate(args) if not arg.startswith('--')):]
os.execvpe(command[0], command, os.environ)
''',
    }
    for name, script in scripts.items():
        path = bindir / name
        path.write_text(script)
        path.chmod(0o755)
    receipt = tmp_path / 'imports.jsonl'
    env = {**os.environ, 'PATH': str(bindir) + os.pathsep + os.environ['PATH'],
           'GITHUB_WORKSPACE': str(tmp_path), 'B200_SQUASH_DIR': str(squash),
           'IMAGE': image, 'DECODE_IMAGE': image, 'PREFILL_IMAGE': image,
           'MODEL_PATH': str(tmp_path), 'TILERT_WEIGHTS_DIR': str(tmp_path / 'weights'),
           'TILERT_IN_ALLOCATION': '1', 'SLURM_JOB_ID': '123',
           'SLURM_JOB_NODELIST': 'node-[a-b]', 'ISL': '1024', 'OSL': '1024',
           'REQUIRE_POWER': '0', 'IMPORT_RECEIPT': str(receipt), 'HOME': str(tmp_path)}
    result = subprocess.run(['bash', str(ROOT / 'benchmarks/multi_node/tilert_utils/submit.sh')],
                            env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    imports = [json.loads(line) for line in receipt.read_text().splitlines()] if receipt.exists() else []
    assert imports == ([] if expected_uri is None else [[expected_uri]])
    assert image_file.read_text() == ('prepared image' if expected_uri is None else 'imported image')


@pytest.mark.parametrize('prepared_path', ['', '/shared/hf/hub/snapshots/revision'])
def test_b200_tilert_preserves_prepared_model_path(prepared_path):
    env = {**os.environ, 'MODEL_PREFIX': 'glm5.1', 'PRECISION': 'fp8',
           'FRAMEWORK': 'tilert', 'IS_MULTINODE': 'true', 'SCENARIO_SUBDIR': '',
           'EXP_NAME': 'glm5.1_8k1k', 'GITHUB_WORKSPACE': str(ROOT),
           'MODEL_PATH': prepared_path}
    result = subprocess.run(
        ['bash', '-c', 'exec() { printf "%s\\n" "$MODEL_PATH"; exit; }; source "$1"',
         'bash', str(ROOT / 'runners/launch_b200-nscale-compat.sh')],
        env=env, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == (prepared_path or '/scratch/models/GLM-5.1-FP8')


@pytest.mark.parametrize(('decode_rc', 'interruption', 'enabled'),
                         [(0, None, True), (9, None, True), (0, 'TERM', True),
                          (0, 'HUP', True), (0, 'INT', True), (0, 'hang', True),
                          (0, None, False)])
def test_tilert_submit_keeps_decode_status_and_stages_both_roles(tmp_path, decode_rc, interruption, enabled):
    repo, bindir = tmp_path / 'repo', tmp_path / 'bin'
    for path in (repo, bindir):
        path.mkdir()
    commands = {
        'salloc':'#!' + sys.executable + '\n' + r'''
import json,os,pathlib,subprocess,sys
args=sys.argv[1:]
command=args[next(i for i, arg in enumerate(args) if not arg.startswith('--')):]
if '/' not in command[0]:
    # Slurm resolves its command before exec; do not silently skip EACCES.
    command[0]=next(str(pathlib.Path(p)/command[0]) for p in os.environ['PATH'].split(os.pathsep)
                    if (pathlib.Path(p)/command[0]).exists())
try:
    result=subprocess.run(command,env={**os.environ,'SLURM_JOB_ID':'123','SLURM_JOB_NODELIST':'node-[a-b]'})
finally:
    staged=pathlib.Path(os.environ['GITHUB_WORKSPACE'])/'LOGS/native_power'
    pathlib.Path(os.environ['RELEASE_RECEIPT']).write_text(json.dumps(
        [json.loads(p.read_text()) for p in sorted(staged.glob('node-*/manifest.json'))]))
sys.exit(result.returncode)
''',
        'squeue':'#!/bin/sh\necho unexpected job-name lookup >&2\nexit 98\n',
        'scontrol':'#!/bin/sh\nprintf "node-a\\nnode-b\\n"\n',
        'scancel':'#!/bin/sh\necho unexpected cancellation >&2\nexit 98\n',
        'git':'#!/bin/sh\necho 0123456789012345678901234567890123456789\n',
        'srun':'#!' + sys.executable + '\n' + r'''
import json, os, pathlib, signal, subprocess, sys, time
args=sys.argv[1:]
if any(a.startswith('--container-image=') for a in args):
    cache=os.environ['HF_HUB_CACHE_HOST_PATH']
    mounts=next(a.split('=',1)[1] for a in args if a.startswith('--container-mounts='))
    if cache:
        assert f'{cache}:{cache}' in mounts.split(',')
    assert f"{os.environ['MODEL_PATH']}:{os.environ['MODEL_PATH']}" in mounts.split(',')
    if os.environ['POWERX_NATIVE_ENABLED'] != '1':
        assert not any('/powerx_native' in a for a in args)
        sys.exit(0)
    rank=os.environ['POWERX_RANK']
    out=pathlib.Path(os.environ['POWERX_RAW_ROOT']) / f'node-{rank}'
    out.mkdir(parents=True,exist_ok=True)
    mode=os.environ.get('INTERRUPTION')
    manifest={'rank':int(rank),'synthetic':True}
    def finish(signum, frame):
        assert pathlib.Path(os.environ['POWERX_CONTROL_ROOT'],'stop').exists()
        manifest['terminated']=True
        (out/'manifest.json').write_text(json.dumps(manifest))
        sys.exit(143)
    signal.signal(signal.SIGTERM, signal.SIG_IGN if mode=='hang' and rank=='0' else finish)
    (out/'manifest.json').write_text(json.dumps(manifest))
    if mode:
        if rank=='1':
            deadline=time.monotonic()+5
            while not (out.parent/'node-0/manifest.json').exists():
                assert time.monotonic()<deadline
                time.sleep(0.01)
            if mode=='hang':
                sys.exit(7)
            os.kill(os.getppid(), getattr(signal, 'SIG'+mode))
        time.sleep(15)
        sys.exit(99)
    sys.exit(int(os.environ['DECODE_RC']) if rank=='0' else 0)
if 'timedatectl' in ' '.join(args):
    print('true')
    sys.exit(0)
if 'flock' in ' '.join(args):
    sys.exit(0)
args=[a for a in args if not a.startswith('--')]
sys.exit(subprocess.run(args).returncode)
''',
    }
    for name, script in commands.items():
        path = bindir / name
        path.write_text(script)
        path.chmod(0o755)
    # Allocation re-entry must not depend on the runner's writable user PATH.
    (bindir / 'env').write_text('not an executable\n')
    (bindir / 'env').chmod(0o600)
    env = {**os.environ, 'PATH':str(bindir)+os.pathsep+os.environ['PATH'],
           'GITHUB_WORKSPACE':str(repo),'B200_SQUASH_DIR':str(tmp_path/'squash'),
           'IMAGE':'synthetic-decode','PREFILL_IMAGE':'synthetic-prefill',
           'MODEL_PATH':str(repo),'MODEL_PREFIX':'fixture','PRECISION':'fp8',
           'HF_HUB_CACHE_HOST_PATH':str(tmp_path/'hf-cache') if enabled else '',
           'PREFILL_TP':'2','DECODE_TP':'2','SLURM_ACCOUNT':'fixture',
           'SLURM_PARTITION':'fixture','RUNNER_NAME':'fixture','REQUIRE_POWER':'1' if enabled else '0', 'ISL':'8192','OSL':'1024',
           'TILERT_WEIGHTS_DIR':str(tmp_path/'weights'),'TILERT_DECODE_DRAIN':'1',
           'POWERX_RAW_ROOT':str(tmp_path/'raw'),'DECODE_RC':str(decode_rc),
           'RELEASE_RECEIPT':str(tmp_path/'released'), 'INTERRUPTION':interruption or ''}
    result = subprocess.run(['bash', str(ROOT/'benchmarks/multi_node/tilert_utils/submit.sh')],
                            env=env,cwd=repo,capture_output=True,text=True,timeout=20)
    expected_rc={'TERM':143,'HUP':143,'INT':130,'hang':7}.get(interruption,decode_rc)
    assert result.returncode == expected_rc, result.stderr + result.stdout
    release_evidence=json.loads((tmp_path/'released').read_text())
    if not enabled:
        assert release_evidence == []
        assert not (repo / 'LOGS/native_power').exists()
        assert not (tmp_path / 'raw').exists()
        return
    assert [item['rank'] for item in release_evidence] == [0,1]
    if interruption in {'TERM','HUP','INT'}:
        assert all(item.get('terminated') for item in release_evidence)
    for rank in (0,1):
        assert json.loads((repo/f'LOGS/native_power/node-{rank}/manifest.json').read_text())['rank'] == rank


@pytest.mark.parametrize('node_rc', [0, 7])
def test_tilert_node_publishes_complete_owned_sentinel(tmp_path, node_rc):
    import re

    source = (ROOT / 'benchmarks/multi_node/tilert_utils/run_node.sh').read_text()
    function = re.search(r'^finish_tilert_node\(\) \{\n.*?^\}', source,
                         flags=re.MULTILINE | re.DOTALL).group()
    command = function + '''
TILERT_ROLE=prefill
DONE_SENTINEL="$1/done"
POWERX_HOST_UID=1000 POWERX_HOST_GID=1000
printf() {
    [[ ! -e "$DONE_SENTINEL" ]] || return 99
    builtin printf "$@"
}
chown() { [[ ! -e "$DONE_SENTINEL" ]]; }
trap finish_tilert_node EXIT
exit "$2"
'''
    result = subprocess.run(['bash', '-c', command, 'bash', str(tmp_path), str(node_rc)],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == node_rc, result.stderr
    assert (tmp_path / 'done').read_text().strip() == str(node_rc)


@pytest.mark.parametrize(('eval_rc', 'stage_rc'), [(0, 0), (7, 0), (0, 9), (7, 9)])
def test_tilert_eval_dispatches_shared_client_and_preserves_failure(tmp_path, eval_rc, stage_rc):
    source = (ROOT / 'benchmarks/multi_node/tilert_utils/run_node.sh').read_text()
    functions = source[source.index('run_bench_and_eval() {'):source.index('run_agentic_replay() {')]
    command = '''
source "$1/benchmarks/benchmark_lib.sh"
FUNCNEST=40
wait_for_server_ready() { return 0; }
run_server_client() { printf '%s\\n' "$@" > client-args; return "$CLIENT_RC"; }
append_lm_eval_summary() { touch staged; return "$STAGE_RC"; }
''' + functions + '\nrun_bench_and_eval\n'
    env = {**os.environ, 'RUN_EVAL': 'true', 'EVAL_ONLY': 'true',
           'POWERX_NATIVE_ENABLED': '0', 'ROUTER_PORT': '9876', 'ROUTER_PID': '1',
           'BENCHMARK_LOGS_DIR': str(tmp_path), 'CONC_LIST': '1', 'EVAL_CONC': '2',
           'MODEL_NAME': 'test-model', 'MODEL': 'test-model', 'EVAL_MAX_MODEL_LEN': '9472',
           'EVAL_FRAMEWORK': 'lm-eval', 'EVAL_SUITE': '', 'EVAL_TASKS_DIR': 'gsm8k',
           'EVAL_RESULT_DIR': str(tmp_path / 'eval'), 'OPENAI_API_KEY': 'EMPTY',
           'INFERENCEX_LM_EVAL_RUNTIME_READY': 'true', 'IS_AGENTIC': '0',
           'SCENARIO_TYPE': 'single_turn', 'PYTHONPYCACHEPREFIX': str(tmp_path / 'pycache'),
           'CLIENT_RC': str(eval_rc), 'STAGE_RC': str(stage_rc)}
    result = subprocess.run(['bash', '-c', command, 'bash', str(ROOT)],
                            env=env, cwd=tmp_path, capture_output=True, text=True, timeout=5)
    assert result.returncode == (eval_rc or stage_rc), result.stderr + result.stdout
    args = (tmp_path / 'client-args').read_text().splitlines()
    assert args[:3] == ['python3', '-m', 'lm_eval']
    model_args = args[args.index('--model_args') + 1].split(',')
    assert 'base_url=http://0.0.0.0:9876/v1/chat/completions' in model_args
    assert 'num_concurrent=2' in model_args
    assert (tmp_path / 'staged').exists()


def test_tilert_tcp_wait_preserves_caller_streams(tmp_path):
    import re
    import socket

    source = (ROOT / 'benchmarks/multi_node/tilert_utils/run_node.sh').read_text()
    function = re.search(r'^wait_for_tcp\(\) \{\n.*?^\}', source,
                         flags=re.MULTILINE | re.DOTALL).group()
    with socket.socket() as server:
        server.bind(('127.0.0.1', 0))
        server.listen(1)
        command = function + '''
exec 3>caller-fd
wait_for_tcp 127.0.0.1 "$1" 0
rc=$?
printf 'eval diagnostic\\n' >&2
printf 'caller stream\\n' >&3
exit "$rc"
'''
        result = subprocess.run(['bash', '-c', command, 'bash', str(server.getsockname()[1])],
                                cwd=tmp_path, capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert 'eval diagnostic' in result.stderr
    assert (tmp_path / 'caller-fd').read_text() == 'caller stream\n'
