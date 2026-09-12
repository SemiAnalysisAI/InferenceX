import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def test_native_control_receipt_is_owned_before_publication(tmp_path):
    source = (ROOT / 'benchmarks/native_power_collect.sh').read_text()
    function = source[source.index('write_control() {'):source.index('\nfinish() {')]
    result = subprocess.run(['bash', '-c', function + '''
control_dir=$1
POWERX_HOST_UID=1000 POWERX_HOST_GID=1000
chown() { [[ ! -e "$control_dir/done-0" ]]; }
write_control done-0 7
''', 'bash', str(tmp_path)], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / 'done-0').read_text() == '7\n'
    assert not list(tmp_path.glob('*.tmp'))


def _collector_with_monitor(tmp_path, *, alive, end_identity_rc=0):
    import shutil
    import sys

    scripts = tmp_path / 'repo/benchmarks'
    scripts.mkdir(parents=True)
    shutil.copyfile(ROOT / 'benchmarks/native_power_collect.sh', scripts / 'native_power_collect.sh')
    library = (ROOT / 'benchmarks/benchmark_lib.sh').read_text()
    functions = []
    for name in ['_background_process_is_running', '_write_amd_smi_sidecar']:
        start = library.index(name + '() {')
        functions.append(library[start:library.index('\n}', start) + 2])
    (scripts / 'benchmark_lib.sh').write_text('\n'.join(functions) + '''
GPU_MONITOR_PID="" GPU_MONITOR_VENDOR=""
start_gpu_monitor() {
    GPU_MONITOR_VENDOR=amd
    if [[ "$MONITOR_ALIVE" == 1 ]]; then
        sleep 30 &
        GPU_MONITOR_PID=$!
    else
        false &
        GPU_MONITOR_PID=$!
        wait "$GPU_MONITOR_PID" || true
    fi
}
stop_gpu_monitor() {
    kill "$GPU_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_MONITOR_PID" 2>/dev/null || true
}
amd-smi() {
    if [[ -f "$IDENTITY_CALLED" && "$END_IDENTITY_RC" != 0 ]]; then return "$END_IDENTITY_RC"; fi
    touch "$IDENTITY_CALLED"
    printf '[{"gpu":0,"uuid":"test-gpu"}]\\n'
}
''')
    control = tmp_path / 'control'
    control.mkdir()
    (control / 'stop').touch()
    power = tmp_path / 'power'
    result = subprocess.run(['bash', str(scripts / 'native_power_collect.sh'), str(power),
                             str(control), 'amd', '0', 'prefill', '0', '1'],
                            env={'PATH': f'{Path(sys.executable).parent}:/usr/bin:/bin',
                                 'PYTHONPATH': str(ROOT), 'MONITOR_ALIVE': str(int(alive)),
                                 'END_IDENTITY_RC': str(end_identity_rc),
                                 'IDENTITY_CALLED': str(tmp_path / 'identity-called')},
                            capture_output=True, text=True, timeout=5)
    return result, control, power


def test_dead_monitor_never_publishes_ready(tmp_path):
    import json

    result, control, power = _collector_with_monitor(tmp_path, alive=False)
    assert not (control / 'ready-0').exists(), result.stderr
    assert (control / 'done-0').read_text().strip() != '0'
    assert json.loads((power / 'manifest.json').read_text())['lifecycle'] == 'failed'


def test_amd_end_identity_failure_is_not_a_successful_done_receipt(tmp_path):
    import json

    result, control, power = _collector_with_monitor(tmp_path, alive=True, end_identity_rc=7)
    assert (control / 'ready-0').exists(), result.stderr
    assert (control / 'done-0').read_text().strip() != '0'
    assert json.loads((power / 'manifest.json').read_text())['lifecycle'] == 'failed'


def test_live_monitor_with_successful_identity_completes(tmp_path):
    import json

    result, control, power = _collector_with_monitor(tmp_path, alive=True)
    assert result.returncode == 0, result.stderr
    assert (control / 'ready-0').exists()
    assert (control / 'done-0').read_text().strip() == '0'
    assert json.loads((power / 'manifest.json').read_text())['lifecycle'] == 'complete'


def test_native_timezone_is_scoped_to_collector(tmp_path):
    import json
    import os
    import sys

    binary = tmp_path / 'bin'
    binary.mkdir()
    smi = binary / 'nvidia-smi'
    smi.write_text(f'''#!{sys.executable}
import os, sys, time
with open(os.environ['SMI_ENV_LOG'], 'a') as output:
    print(os.environ.get('TZ', ''), file=output, flush=True)
if '-l' in sys.argv:
    print('timestamp, index, power.draw [W]', flush=True)
    print('2026/09/12 00:00:00.000, 0, 100', flush=True)
    time.sleep(30)
else:
    print('index, uuid, pci.bus_id, name, driver_version')
    print('0, GPU-fixture, 0000:00:00.0, fixture, fixture')
''')
    smi.chmod(0o755)
    env = {**os.environ, 'PATH': f'{binary}:{Path(sys.executable).parent}:/usr/bin:/bin',
           'TZ': 'Pacific/Honolulu', 'SMI_ENV_LOG': str(tmp_path / 'ordinary-env')}
    ordinary = tmp_path / 'ordinary.csv'
    result = subprocess.run(['bash', '-c', '''source "$1"
start_gpu_monitor --output "$2"
stop_gpu_monitor
''', 'bash', str(ROOT / 'benchmarks/benchmark_lib.sh'), str(ordinary)],
                            env=env, capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert set((tmp_path / 'ordinary-env').read_text().splitlines()) == {'Pacific/Honolulu'}
    assert not ordinary.with_name('ordinary_context.json').exists()

    control = tmp_path / 'control'
    control.mkdir()
    (control / 'stop').touch()
    power = tmp_path / 'native'
    env['SMI_ENV_LOG'] = str(tmp_path / 'native-env')
    result = subprocess.run(['bash', str(ROOT / 'benchmarks/native_power_collect.sh'),
                             str(power), str(control), 'nvidia', '0', 'aggregate', '0', '1'],
                            env=env, capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert set((tmp_path / 'native-env').read_text().splitlines()) == {'UTC'}
    assert json.loads((power / 'gpu_metrics_context.json').read_text()) == {'timestamp_timezone': 'UTC'}
    assert (control / 'done-0').read_text().strip() == '0'
