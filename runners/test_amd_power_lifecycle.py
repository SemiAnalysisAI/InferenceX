"""Keep optional collector failures separate from AMD serving outcomes.

Expected exit codes below follow the required-power rule the AMD scripts
implement: REQUIRE_POWER in {1, true, TRUE, yes, YES} makes a collector
failure fail the run with rc 1 but never masks a nonzero serving rc; any
other value downgrades the failure to a warning and the run rc is the
serving rc. Failures before the benchmark starts abort it when required.
"""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_job_slurm_forwards_require_power_into_the_container():
    job = (ROOT / 'benchmarks/multi_node/amd_utils/job.slurm').read_text()
    start = job.index('DOCKER_ENV_COMMON=(')
    block = job[start:job.index('\n)', start) + 2]
    # Execute the submit-shell array and the node-shell expansion used by docker.
    script = block + '\ndocker() { printf \'%s\\0\' "$@"; }; export -f docker;\n' + (
        'bash -c "docker run ${DOCKER_ENV_COMMON[*]} test-image"')
    result = subprocess.run(['bash', '-e', '-c', script],
                            env={'PATH': '/usr/bin:/bin', 'WS_PATH': '/workspace',
                                 'REQUIRE_POWER': 'sentinel-policy'}, capture_output=True, check=True)
    args = result.stdout.decode().split('\0')
    forwarded = dict(args[index + 1].split('=', 1) for index, arg in enumerate(args) if arg == '-e')
    assert forwarded['REQUIRE_POWER'] == 'sentinel-policy'


@pytest.mark.parametrize(('phase', 'required', 'serving_rc', 'expected_rc', 'benchmark_runs'), [
    ('ready', '', 0, 0, True),
    ('ready', '', 7, 7, True),
    ('ready', '1', 0, 1, False),
    ('ready', 'YES', 7, 1, False),
    ('done', '', 7, 7, True),
    ('done', '1', 0, 1, True),
    ('done', 'true', 7, 7, True),
])
def test_amd_collector_failure_respects_requirement(tmp_path, phase, required, serving_rc, expected_rc, benchmark_runs):
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
                                 'REQUIRE_POWER': required, 'SERVING_RC': str(serving_rc),
                                 'CALL_RECEIPT': str(receipt), 'POWERX_HOST_UID': str(os.getuid()),
                                 'POWERX_HOST_GID': str(os.getgid())}, capture_output=True, text=True, timeout=10)
    assert result.returncode == expected_rc, result.stderr
    assert receipt.exists() is benchmark_runs


@pytest.mark.parametrize('fault', ['topology', 'uneven_tp'])
@pytest.mark.parametrize(('required', 'serving_rc', 'expected_rc', 'benchmark_runs'), [
    ('', 0, 0, True),
    ('0', 7, 7, True),
    ('false', 0, 0, True),
    ('1', 0, 1, False),
    ('true', 7, 1, False),
    ('YES', 0, 1, False),
])
def test_amd_start_failure_respects_requirement(tmp_path, fault, required, serving_rc, expected_rc, benchmark_runs):
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
                                 'NODE_RANK': '0', 'REQUIRE_POWER': required,
                                 'SERVING_RC': str(serving_rc), 'CALL_RECEIPT': str(receipt)},
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == expected_rc, result.stderr
    assert receipt.exists() is benchmark_runs
    assert ('inconsistent AMD node topology' if fault == 'topology' else
            'uneven per-node TP layout') in result.stderr


@pytest.mark.parametrize(('required', 'serving_rc', 'staging_rc', 'expected_rc'), [
    ('', 0, 0, 0),
    ('', 0, 9, 0),
    ('', 7, 9, 7),
    ('false', 0, 9, 0),
    ('1', 0, 0, 0),
    ('1', 0, 9, 1),
    ('1', 7, 9, 7),
    ('YES', 0, 9, 1),
])
def test_amd_staging_failure_preserves_serving_outcome(tmp_path, required, serving_rc, staging_rc, expected_rc):
    tail = (ROOT / 'benchmarks/multi_node/amd_utils/job.slurm').read_text().split(
        'BENCHMARK_STEP_RC=$?', 1)[1]
    script = 'BENCHMARK_STEP_RC=$SERVING_RC\nstage_native_power() { return "$STAGING_RC"; }\n' + tail
    result = subprocess.run(['bash', '-euo', 'pipefail', '-c', script],
                            env={**os.environ, 'REQUIRE_POWER': required, 'SERVING_RC': str(serving_rc),
                                 'STAGING_RC': str(staging_rc), 'KEEP_CONTAINERS': '1'},
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == expected_rc, result.stderr
