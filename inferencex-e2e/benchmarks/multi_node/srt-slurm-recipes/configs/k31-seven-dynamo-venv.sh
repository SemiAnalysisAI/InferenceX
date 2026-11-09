#!/usr/bin/env bash
set -eo pipefail
source /infmax-workspace/benchmarks/check_env.sh
check_env_vars DYNAMO_VENV_ROOT DYNAMO_VERSION DYNAMO_ENGINE_GRPC_VERSION DYNAMO_SITE_PACKAGES DYNAMO_BASE_PYTHON SLURM_JOB_ID
mkdir -p /private-state
bash /configs/k31-umbp-tc-audit-pool.sh
venv_path="${DYNAMO_VENV_ROOT}/${SLURM_JOB_ID}"
mkdir -p "$DYNAMO_VENV_ROOT" "$venv_path/home"
(
    flock -x 200
    if [[ ! -f "$venv_path/.install-complete" ]]; then
        "$DYNAMO_BASE_PYTHON" -m venv --system-site-packages "$venv_path"
        base_sites=$("$DYNAMO_BASE_PYTHON" -c 'import site; print("\n".join(site.getsitepackages()))')
        "$venv_path/bin/python" - "$DYNAMO_SITE_PACKAGES" "$base_sites" <<'PY'
import site
import sys
from pathlib import Path
source = Path(sys.argv[1])
if not (source / 'dynamo').is_dir():
    raise RuntimeError(f'Missing validated Dynamo package snapshot: {source}')
(Path(site.getsitepackages()[0]) / 'k31-dynamo.pth').write_text(str(source) + '\n' + sys.argv[2] + '\n')
PY
        venv_site=$("$venv_path/bin/python" -c 'import site; print(site.getsitepackages()[0])')
        PYTHONPATH="/infmax-workspace${PYTHONPATH:+:$PYTHONPATH}" "$venv_path/bin/python" -m infx.bench dynamo-mum-compat --source "$DYNAMO_SITE_PACKAGES/dynamo" --target "$venv_site/dynamo"
        touch "$venv_path/.install-complete"
    fi
) 200>"${venv_path}.lock"
# Engine roles intentionally use the image's /opt/mum overlay via PYTHONPATH.
# Validate against the base interpreter under the same role environment, rather
# than assuming that every valid vLLM installation lives in runtime-venv.
# Run this even when another role has already initialized the shared venv.
"$venv_path/bin/python" - <<'PY'
import json
import os
import subprocess
from importlib.metadata import version
from pathlib import Path
import torch
import vllm
for name in ('ai-dynamo', 'ai-dynamo-runtime'):
    if version(name) != os.environ['DYNAMO_VERSION']:
        raise RuntimeError(f'{name}: unexpected version {version(name)}')
if version('grpcio') != os.environ['DYNAMO_ENGINE_GRPC_VERSION']:
    raise RuntimeError(f'grpcio: unexpected version {version("grpcio")}')
base = json.loads(subprocess.check_output([
    os.environ['DYNAMO_BASE_PYTHON'], '-c',
    'import json, torch, vllm; from pathlib import Path; '
    'print(json.dumps({"vllm": str(Path(vllm.__file__).resolve()), '
    '"torch": str(Path(torch.__file__).resolve())}))',
], text=True))
actual = {name: str(Path(module.__file__).resolve())
          for name, module in (('vllm', vllm), ('torch', torch))}
if actual != base:
    raise RuntimeError(f'Dynamo venv shadows engine packages: {actual=}, {base=}')
print({'engine_packages': actual,
       'versions': {p: version(p) for p in ('ai-dynamo', 'ai-dynamo-runtime', 'vllm', 'torch')}})
PY
"$venv_path/bin/python" -c 'import dynamo.frontend.main; print("Dynamo frontend package:", dynamo.frontend.main.__file__)'
