"""Verify full-sweep coverage using the dispatched matrix and result validators."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

from infx.workflows import validate_reusable_sweep_artifacts as reuse

from . import github
from .github import VerificationError


def expected_evals(matrix: dict) -> set[tuple]:
    expected = set()
    for bucket in ('evals', 'agentic_evals', 'multinode_evals', 'multinode_agentic_evals'):
        for entry in matrix.get(bucket, []):
            multi = entry.get('prefill') is not None
            row = {key.replace('-', '_'): value for key, value in entry.items()}
            row.update(is_multinode=multi, hw=entry['runner'], model_prefix=entry['model-prefix'],
                       eval_suite=entry.get('eval-suite') or 'gsm8k',
                       isl=entry.get('isl', 0), osl=entry.get('osl', 0),
                       dp_attention=entry.get('dp-attn', False))
            for role in ('prefill', 'decode') if multi else ():
                for key, value in entry[role].items():
                    name = {'dp-attn': 'dp_attention', 'num-worker': 'num_workers'}.get(key, key.replace('-', '_'))
                    row[f'{role}_{name}'] = value
            concs = entry['conc'] if multi and entry.get('eval-all-concs') else [
                entry['eval-conc'] if multi else entry['conc']]
            expected.update(reuse.eval_key({**row, 'conc': conc}) for conc in concs)
    return expected


def check_coverage(directory: Path, manifest: dict, run: dict, family: str) -> None:
    if (manifest['head'] != run['head_sha'] or manifest['run-id'] != run['id']
            or manifest['run-attempt'] != run['run_attempt'] or manifest['full-sweep'] is not True):
        raise VerificationError('Full-sweep provenance mismatch')
    matrix = manifest['matrix']
    if matrix['changelog_metadata']['head_ref'] != run['head_sha']:
        raise VerificationError('Matrix was generated from a different head')
    entries = matrix['changelog_metadata']['entries']
    key = family.split(':', 1)[1]
    # An image refresh needs the complete family, not an append-only/scenario subset.
    if len(entries) != 1 or entries[0]['config-keys'] != [key] or any(
            entries[0].get(field) for field in ('append-only', 'scenario-type', 'evals-only', 'eval-min-prefill-ep')):
        raise VerificationError('Final changelog does not select the complete candidate family')
    expected = set()
    for group in ('single_node', 'multi_node'):
        for rows in matrix.get(group, {}).values():
            for entry in rows:
                concs = entry['conc'] if isinstance(entry['conc'], list) else [entry['conc']]
                expected.update((entry['recipe-fingerprint'], int(conc), entry['image']) for conc in concs)
    if not expected:
        raise VerificationError('Empty full-sweep matrix')
    paths = list((directory / 'results_bmk').glob('*.json'))
    fixed = [row for _, row in reuse.json_rows(paths) if row.get('scenario_type') != 'agentic-coding']
    agentic = [row for _, row in reuse.json_rows(reuse.agentic_point_files(directory))]
    actual = [(row['recipe_fingerprint'], int(row['conc']), row['image']) for row in fixed + agentic]
    errors = reuse.duplicate_identity_errors('benchmark', actual)
    errors += reuse.validate_identity_set('benchmark', expected, set(actual))
    reuse.dedupe_reran_evals(directory)
    eval_rows, eval_errors = reuse.raw_eval_key_rows(directory)
    errors += eval_errors + reuse.validate_eval_artifacts(directory)
    errors += reuse.validate_identity_set('eval', expected_evals(matrix), {row[:-1] for row in eval_rows})
    if errors:
        # Only a fixed failure code escapes: no raw artifact data in public diagnostics.
        raise VerificationError('Full-sweep result coverage or consistency failed')


def verify_sweep(repository: str, run: dict, family: str) -> None:
    if run['status'] != 'completed' or run['conclusion'] != 'success':
        raise VerificationError('Final sweep has not passed')
    inventory = github.artifacts(repository, run['id'])
    manifests = [a for a in inventory if a['name'] == 'klaud-sweep-manifest' and not a['expired']]
    if len(manifests) != 1:
        raise VerificationError('Missing final-sweep manifest')
    with tempfile.TemporaryDirectory(prefix='klaud-validation-') as temp:
        directory = Path(temp)
        github.download_json(repository, manifests[0], directory / 'klaud-sweep-manifest')
        manifest = json.loads((directory / 'klaud-sweep-manifest/sweep_manifest.json').read_text())
        for artifact in inventory:
            name = artifact['name']
            wanted = (name == 'results_bmk' or name.startswith('bmk_agentic_')
                      or name.startswith('eval_') and not name.startswith(('eval_server_logs_', 'eval_gpu_metrics_')))
            if wanted:
                if '/' in name or '\\' in name or name in ('.', '..'):
                    raise VerificationError('Invalid artifact name')
                github.download_json(repository, artifact, directory / name)
        check_coverage(directory, manifest, run, family)
