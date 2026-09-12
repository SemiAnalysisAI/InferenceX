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
