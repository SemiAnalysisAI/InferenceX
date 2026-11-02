import json
import os
from pathlib import Path
import subprocess
import sys
from importlib.metadata import version

import mori
import mori.io
import mori.umbp

venv = Path(sys.argv[1]).resolve()
package = Path(mori.__file__).resolve().parent
assert package.is_relative_to(venv), (package, venv)
assert subprocess.check_output(['git', '-C', str(venv / 'mori-source'), 'rev-parse', 'HEAD'], text=True).strip() == os.environ['MORI_SOURCE_COMMIT']
for name in ('umbp_master', 'umbp_standalone_server'):
    binary = package / name
    assert binary.is_file(), binary
    result = subprocess.run(['ldd', str(binary)], text=True, capture_output=True, check=True)
    assert 'not found' not in result.stdout, result.stdout
assert any(b'UMBP_DRAM_MAX_MR_BYTES' in p.read_bytes() for p in package.glob('*.so'))
print(json.dumps({'mori_source': os.environ['MORI_SOURCE_COMMIT'], 'mori_path': str(package), 'versions': {p: version(p) for p in ('amd-mori', 'vllm', 'torch', 'ai-dynamo', 'ai-dynamo-runtime')}, 'max_mr_bytes': os.environ.get('UMBP_DRAM_MAX_MR_BYTES')}), flush=True)
