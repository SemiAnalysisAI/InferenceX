#!/usr/bin/env bash
set -eo pipefail
python3 - <<'PY'
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

package = Path('/infmax-workspace/benchmarks/single_node/srt-slurm-recipes/qwen3.8next/sglang/mi355x-fp4-mtp/hyperloom')
manifest = json.loads((package / 'manifest.json').read_text())
for entry in manifest['files']:
    path = package / entry['file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
        raise RuntimeError(f'Hyperloom asset hash mismatch: {path}')

digest = hashlib.sha256((package / 'manifest.json').read_bytes()).hexdigest()
marker = Path('/sgl-workspace/.qwen38-hyperloom-patches')
targets = [(Path('/sgl-workspace/sglang'), package / 'sglang.diff'),
           (Path('/sgl-workspace/aiter'), package / 'aiter.diff')]
if marker.exists():
    if marker.read_text().strip() != digest:
        raise RuntimeError('This container already contains a different Hyperloom patch set')
    for tree, patch in targets:
        subprocess.run(['git', '-c', 'safe.directory=*', '-C', str(tree), 'apply',
                        '--reverse', '--check', '--binary', str(patch)], check=True)
    print('Hyperloom patch set already applied and verified', flush=True)
else:
    for tree, patch in targets:
        subprocess.run(['git', '-c', 'safe.directory=*', '-C', str(tree), 'apply',
                        '--check', '--binary', str(patch)], check=True)
    for tree, patch in targets:
        subprocess.run(['git', '-c', 'safe.directory=*', '-C', str(tree), 'apply',
                        '--binary', str(patch)], check=True)
    # Rebuild the JIT sorting module against the patched AITER sources.
    aiter = Path('/sgl-workspace/aiter').resolve()
    for suffix in ['aiter/jit/module_moe_sorting_opus.so', 'aiter/jit/build/module_moe_sorting_opus']:
        artifact = aiter / suffix
        artifact.resolve().relative_to(aiter)
        if artifact.is_dir():
            shutil.rmtree(artifact)
        elif artifact.exists():
            artifact.unlink()

destination = Path('/opt/stack/configs/tables')
destination.mkdir(parents=True, exist_ok=True)
for entry in manifest['files']:
    if entry['file'].endswith('.csv'):
        shutil.copyfile(package / entry['file'], destination / entry['file'])
marker.write_text(digest + '\n')
print('Hyperloom SGLang/AITER patches and tuning tables ready', flush=True)
PY
