"""Keep optional collector failures separate from AMD serving outcomes."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _docker_container_env(required):
    job = (ROOT / 'benchmarks/multi_node/amd_utils/job.slurm').read_text()
    start = job.index('DOCKER_ENV_COMMON=(')
    block = job[start:job.index('\n)', start) + 2]
    # Execute the submit-shell array and the node-shell expansion used by docker.
    script = block + '\ndocker() { printf \'%s\\0\' "$@"; }; export -f docker;\n' + (
        'bash -c "docker run ${DOCKER_ENV_COMMON[*]} test-image"')
    result = subprocess.run(['bash', '-e', '-c', script],
                            env={'PATH': '/usr/bin:/bin', 'WS_PATH': '/workspace',
                                 'REQUIRE_POWER': required}, capture_output=True, check=True)
    args = result.stdout.decode().split('\0')
    return dict(args[index + 1].split('=', 1) for index, arg in enumerate(args) if arg == '-e')


@pytest.mark.parametrize('phase', ['ready', 'done'])
@pytest.mark.parametrize('required', ['', '1', 'true', 'YES'])
@pytest.mark.parametrize('serving_rc', [0, 7])
def test_amd_collector_failure_respects_requirement(tmp_path, phase, required, serving_rc):
    benchmark_root = tmp_path / 'benchmarks'
    scripts = benchmark_root / 'multi_node/amd_utils'
    scripts.mkdir(parents=True)
    for name in ['bench.sh', 'power.sh']:
        shutil.copyfile(ROOT / 'benchmarks/multi_node/amd_utils' / name, scripts / name)
    (benchmark_root / 'benchmark_lib.sh').write_text('''run_benchmark_serving() {
        printf 'called\\n' > "$CALL_RECEIPT"
        printf '1\\n' > "$POWERX_CONTROL_DIR/done-0"
        return "$SERVING_RC"
    }
''')
    control = tmp_path / 'control'
    control.mkdir()
    (control / 'ready-0').write_text('ready\n')
    if phase == 'ready':
        (control / 'done-0').write_text('1\n')
    receipt = tmp_path / 'called'
    result = subprocess.run(['bash', str(scripts / 'bench.sh'), '1', '1', '1', '1',
                             '/model', 'test', str(tmp_path / 'logs'), '8192', '1024', '1'],
                            env={**os.environ, 'POWERX_CONTROL_DIR': str(control), 'NNODES': '1',
                                 'REQUIRE_POWER': _docker_container_env(required).get('REQUIRE_POWER', ''), 'SERVING_RC': str(serving_rc),
                                 'CALL_RECEIPT': str(receipt), 'POWERX_HOST_UID': str(os.getuid()),
                                 'POWERX_HOST_GID': str(os.getgid())}, capture_output=True, text=True, timeout=10)
    expected = 1 if required and phase == 'ready' else serving_rc or int(bool(required))
    assert result.returncode == expected, result.stderr
    assert receipt.exists() == (not required or phase != 'ready')


@pytest.mark.parametrize('fault', ['topology', 'uneven_tp'])
@pytest.mark.parametrize('required', ['', '0', 'false', '1', 'true', 'YES'])
@pytest.mark.parametrize('serving_rc', [0, 7])
def test_amd_start_failure_respects_requirement(tmp_path, fault, required, serving_rc):
    scripts = tmp_path / 'amd_utils'
    scripts.mkdir()
    for name in ['server.sh', 'power.sh']:
        shutil.copyfile(ROOT / 'benchmarks/multi_node/amd_utils' / name, scripts / name)
    receipt = tmp_path / 'called'
    (scripts / 'server_sglang.sh').write_text('''printf 'called\\n' > "$CALL_RECEIPT"
exit "$SERVING_RC"
''')
    result = subprocess.run(['bash', str(scripts / 'server.sh')],
                            env={**os.environ, 'WS_PATH': str(scripts),
                                 'ENGINE': 'sglang-disagg', 'BENCH_INPUT_LEN': '8192',
                                 'BENCH_OUTPUT_LEN': '1024', 'EVAL_ONLY': 'false',
                                 'IS_AGENTIC': '0', 'DRY_RUN': '0',
                                 'GPUS_PER_NODE': '8', 'PREFILL_TP_SIZE': '9',
                                 'DECODE_TP_SIZE': '8', 'xP': '1', 'yD': '1',
                                 'NNODES': '2' if fault == 'topology' else '3',
                                 'NODE_RANK': '0', 'REQUIRE_POWER': _docker_container_env(required).get('REQUIRE_POWER', ''),
                                 'SERVING_RC': str(serving_rc), 'CALL_RECEIPT': str(receipt)},
                            capture_output=True, text=True, timeout=5)
    is_required = required in ['1', 'true', 'YES']
    assert result.returncode == (1 if is_required else serving_rc), result.stderr
    assert receipt.exists() == (not is_required)
    assert ('inconsistent AMD node topology' if fault == 'topology' else
            'uneven per-node TP layout') in result.stderr


@pytest.mark.parametrize('required', ['', '0', 'false', '1', 'true', 'YES'])
@pytest.mark.parametrize('serving_rc', [0, 7])
@pytest.mark.parametrize('staging_rc', [0, 9])
def test_amd_staging_failure_preserves_serving_outcome(tmp_path, required, serving_rc, staging_rc):
    tail = (ROOT / 'benchmarks/multi_node/amd_utils/job.slurm').read_text().split(
        'BENCHMARK_STEP_RC=$?', 1)[1]
    script = 'BENCHMARK_STEP_RC=$SERVING_RC\nstage_native_power() { return "$STAGING_RC"; }\n' + tail
    result = subprocess.run(['bash', '-euo', 'pipefail', '-c', script],
                            env={**os.environ, 'REQUIRE_POWER': required, 'SERVING_RC': str(serving_rc),
                                 'STAGING_RC': str(staging_rc), 'KEEP_CONTAINERS': '1'},
                            capture_output=True, text=True, timeout=5)
    expected = serving_rc or int(staging_rc != 0 and required in ['1', 'true', 'YES'])
    assert result.returncode == expected, result.stderr
