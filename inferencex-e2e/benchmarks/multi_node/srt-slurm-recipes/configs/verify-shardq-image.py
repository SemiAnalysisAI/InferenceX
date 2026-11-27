import hashlib
import base64
import csv
import importlib.metadata as md
import importlib.util
import json
import os
from pathlib import Path

root = Path('/opt/k31-unified')
audit = json.loads((root / 'source-audit.json').read_text())
frozen = json.loads(Path('/opt/k31-shardq-r48-manifest.json').read_text())
assert len(frozen) == 58, len(frozen)
pkg = Path(importlib.util.find_spec('vllm').origin).parent
source_sha = os.environ.get('K31_VLLM_SOURCE_SHA')
record_sha = os.environ.get('K31_VLLM_RECORD_SHA256')
assert bool(source_sha) == bool(record_sha), 'Source SHA and RECORD SHA must be supplied together'
if source_sha:
    dist = md.distribution('vllm')
    records = [p for p in dist.files if p.name == 'RECORD' and p.parent.name.endswith('.dist-info')]
    assert len(records) == 1, records
    record = Path(dist.locate_file(records[0]))
    assert hashlib.sha256(record.read_bytes()).hexdigest() == record_sha, 'Wheel RECORD mismatch'
    checked = set()
    with record.open(newline='') as handle:
        for name, digest, size in csv.reader(handle):
            if not name.startswith('vllm/'):
                continue
            target = Path(dist.locate_file(name)).resolve()
            assert target.is_relative_to(pkg.resolve()), name
            algorithm, encoded = digest.split('=', 1)
            assert algorithm == 'sha256', name
            data = target.read_bytes()
            actual = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
            assert actual == encoded and len(data) == int(size), name
            checked.add(target)
    assert checked, 'Empty vLLM wheel'
    assert all(p.resolve() in checked for p in pkg.rglob('*') if p.suffix in ('.py', '.so'))
    manifest = json.loads((pkg / '_source_manifest.json').read_text())
    assert manifest['source_sha'] == source_sha, manifest['source_sha']
else:
    expected = json.loads((root / 'vllm-python-source.json').read_text())
    for name, sha in expected.items():
        target = pkg / name
        sha = frozen.get(str(target), {}).get('sha256', sha)
        assert hashlib.sha256(target.read_bytes()).hexdigest() == sha, str(target)
    allowed = set(expected) | {'_version.py'}
    allowed.update(str(Path(p).relative_to(pkg)) for p in frozen if Path(p).is_relative_to(pkg))
    assert {str(p.relative_to(pkg)) for p in pkg.rglob('*.py')} <= allowed
for name, sha in json.loads((root / 'base-native.json').read_text()).items():
    assert hashlib.sha256((pkg / name).read_bytes()).hexdigest() == sha, name
aiter = Path(importlib.util.find_spec('aiter').origin).parent
for name, sha in audit['aiter_source_hashes'].items():
    target = aiter / name
    sha = frozen.get(str(target), {}).get('sha256', sha)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == sha, str(target)
for name, item in frozen.items():
    if source_sha and Path(name).is_relative_to(pkg):
        continue
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == item['sha256'], name
mori = Path(importlib.util.find_spec('mori').origin).parent
for name in ('libmori_pybinds.so', 'libmori_io.so', 'libmori_application.so', 'libmori_umbp_pybinds.so'):
    assert (mori / name).is_file(), name
assert not (mori / 'io/dmabuf_allocator.py').exists()
from mori.io import hip_gem_lookup, make_hip_gem_mem_pool
assert callable(hip_gem_lookup) and callable(make_hip_gem_mem_pool)
versions = {n: md.version(n) for n in ('vllm', 'amd-aiter', 'flydsl', 'amd-mori', 'vllm-router', 'torch')}
assert versions['vllm'] == '0.0.0+k31.' + (source_sha or audit['source'])[:12]
assert versions['flydsl'] == '0.5.0' and versions['vllm-router'] == '0.1.17'
assert versions['amd-mori'] == audit['mori_version']
for forbidden in ('/opt/k31-pr26', '/opt/k31-r10', '/opt/mum', '/opt/k31-role'):
    assert not Path(forbidden).exists(), forbidden
print(json.dumps({'base_source': audit['source'], 'source': source_sha or audit['source'], 'record_sha256': record_sha, 'frozen_files': len(frozen), 'versions': versions, 'runtime_source_overlays': False}))
