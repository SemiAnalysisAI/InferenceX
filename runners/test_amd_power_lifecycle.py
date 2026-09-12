"""Keep optional collector failures separate from AMD serving outcomes."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


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
                                 'REQUIRE_POWER': required, 'SERVING_RC': str(serving_rc),
                                 'CALL_RECEIPT': str(receipt), 'POWERX_HOST_UID': str(os.getuid()),
                                 'POWERX_HOST_GID': str(os.getgid())}, capture_output=True, text=True, timeout=10)
    expected = 1 if required and phase == 'ready' else serving_rc or int(bool(required))
    assert result.returncode == expected, result.stderr
    assert receipt.exists() == (not required or phase != 'ready')
