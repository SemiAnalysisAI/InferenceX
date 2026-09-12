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


@pytest.mark.parametrize(('decode_rc', 'interruption'), [(0, None), (9, None),
                                                        (0, 'TERM'), (0, 'HUP'),
                                                        (0, 'INT'), (0, 'hang')])
def test_tilert_submit_keeps_decode_status_and_stages_both_roles(tmp_path, decode_rc, interruption):
    repo, bindir = tmp_path / 'repo', tmp_path / 'bin'
    for path in (repo, bindir):
        path.mkdir()
    commands = {
        'salloc':'#!' + sys.executable + '\n' + r'''
import json,os,pathlib,subprocess,sys
args=sys.argv[1:]
try:
    result=subprocess.run(args[args.index('env'):],env={**os.environ,'SLURM_JOB_ID':'123','SLURM_JOB_NODELIST':'node-[a-b]'})
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
    env = {**os.environ, 'PATH':str(bindir)+os.pathsep+os.environ['PATH'],
           'GITHUB_WORKSPACE':str(repo),'B200_SQUASH_DIR':str(tmp_path/'squash'),
           'IMAGE':'synthetic-decode','PREFILL_IMAGE':'synthetic-prefill',
           'MODEL_PATH':str(repo),'MODEL_PREFIX':'fixture','PRECISION':'fp8',
           'PREFILL_TP':'2','DECODE_TP':'2','SLURM_ACCOUNT':'fixture',
           'SLURM_PARTITION':'fixture','RUNNER_NAME':'fixture','REQUIRE_POWER':'1', 'ISL':'8192','OSL':'1024',
           'TILERT_WEIGHTS_DIR':str(tmp_path/'weights'),'TILERT_DECODE_DRAIN':'1',
           'POWERX_RAW_ROOT':str(tmp_path/'raw'),'DECODE_RC':str(decode_rc),
           'RELEASE_RECEIPT':str(tmp_path/'released'), 'INTERRUPTION':interruption or ''}
    result = subprocess.run(['bash', str(ROOT/'benchmarks/multi_node/tilert_utils/submit.sh')],
                            env=env,cwd=repo,capture_output=True,text=True,timeout=20)
    expected_rc={'TERM':143,'HUP':143,'INT':130,'hang':7}.get(interruption,decode_rc)
    assert result.returncode == expected_rc, result.stderr + result.stdout
    release_evidence=json.loads((tmp_path/'released').read_text())
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
