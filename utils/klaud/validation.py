"""Verify full-sweep coverage using the dispatched matrix and result validators."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace

from utils import validate_reusable_sweep_artifacts as reuse

from . import github
from .github import VerificationError
from .models import OwnedCandidate, utc


def benchmark_entries(matrix: dict) -> list[dict]:
    return [entry for group in ('single_node', 'multi_node')
            for rows in matrix.get(group, {}).values() for entry in rows]


def benchmark_points(entries: list[dict]) -> set[tuple]:
    return {(entry['recipe-fingerprint'], int(conc), entry['image']) for entry in entries
            for conc in (entry['conc'] if isinstance(entry['conc'], list) else [entry['conc']])}


def canonical_matrix(repository: str, head: str, family: str) -> dict:
    """Generate the unfiltered family from exact-head YAML using trusted local code.

    Candidate source is data only. A future generator-policy change can require manual
    inspection of an older run; it cannot silently waive points or required evals.
    """
    from infx.matrix.generate import generate_test_config_sweep, mark_eval_entries
    from infx.matrix.validation import load_config_files, load_runner_file
    from utils.process_changelog import recipe_fingerprint

    OwnedCandidate(id='0' * 16 + '-' + '0' * 16, family=family, base=head)
    source, key = family.split(':', 1)
    with tempfile.TemporaryDirectory(prefix='klaud-matrix-') as temp:
        master, runners = Path(temp) / 'master.yaml', Path(temp) / 'runners.yaml'
        master.write_bytes(github.file_at(repository, head, source))
        runners.write_bytes(github.file_at(repository, head, 'configs/runners.yaml'))
        entries = generate_test_config_sweep(SimpleNamespace(config_keys=[key]),
                                             load_config_files([str(master)]), load_runner_file(str(runners)))
    evals = [dict(row, **{'eval-only': True}) for row in mark_eval_entries(deepcopy(entries), include_agentic=True)
             if row.get('run-eval')]
    return {'single_node': {'all': [{**row, 'recipe-fingerprint': recipe_fingerprint(row)} for row in entries]},
            'evals': evals}


def check_matrix(matrix: dict, canonical: dict, head: str, family: str) -> None:
    from utils.process_changelog import recipe_fingerprint
    metadata = matrix['changelog_metadata']
    if metadata['head_ref'] != head:
        raise VerificationError('Matrix was generated from a different head')
    entries = metadata['entries']
    if len(entries) != 1 or entries[0]['config-keys'] != [family.split(':', 1)[1]]:
        raise VerificationError('Final changelog must select exactly the candidate family')
    expected = benchmark_points(benchmark_entries(canonical))
    for entry in benchmark_entries(matrix):
        measured = {key: value for key, value in entry.items() if key not in ('priority', 'queue-token')}
        if recipe_fingerprint(measured) != entry['recipe-fingerprint']:
            raise VerificationError('Matrix settings do not match their recipe fingerprint')
    count = sum(len(entry['conc']) if isinstance(entry['conc'], list) else 1 for entry in benchmark_entries(matrix))
    if count != len(benchmark_points(benchmark_entries(matrix))):
        raise VerificationError('Duplicate point in final matrix')
    if not expected or benchmark_points(benchmark_entries(matrix)) != expected:
        raise VerificationError('Final matrix omits or changes canonical benchmark points')
    if expected_evals(matrix) != expected_evals(canonical):
        raise VerificationError('Final matrix does not match canonical default evals')


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


def check_coverage(directory: Path, manifest: dict, run: dict, family: str, canonical: dict) -> None:
    if (manifest['head'] != run['head_sha'] or manifest['run-id'] != run['id']
            or manifest['run-attempt'] != run['run_attempt'] or manifest['full-sweep'] is not True):
        raise VerificationError('Full-sweep provenance mismatch')
    matrix = manifest['matrix']
    check_matrix(matrix, canonical, run['head_sha'], family)
    expected = benchmark_points(benchmark_entries(matrix))
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


def select_artifacts(inventory: list[dict], run: dict, attempt: dict) -> list[dict]:
    """Newest same-name artifact within this run; never overlay two archives.

    Retained point artifacts from successful jobs in an earlier attempt are allowed.
    The final manifest must be current. An earlier aggregate is retained only when
    none of its point producers reran. Coverage/raw agreement still must pass.
    """
    selected: dict[str, dict] = {}
    for artifact in inventory:
        name = artifact['name']
        wanted = (name in ('klaud-sweep-manifest', 'results_bmk') or name.startswith('bmk_agentic_')
                  or name.startswith('eval_') and not name.startswith(('eval_server_logs_', 'eval_gpu_metrics_')))
        if not wanted:
            continue
        if '/' in name or '\\' in name or name in ('.', '..'):
            raise VerificationError('Invalid artifact name')
        producer = artifact.get('workflow_run', {})
        if producer.get('id') != run['id'] or producer.get('head_sha') != run['head_sha']:
            raise VerificationError('Artifact run or head mismatch')
        if utc(artifact['created_at']) < utc(run['created_at']) or utc(artifact['created_at']) > utc(run['updated_at']):
            raise VerificationError('Artifact outside run lifetime')
        previous = selected.get(name)
        if previous is None or (utc(artifact['created_at']), artifact['id']) > (utc(previous['created_at']), previous['id']):
            selected[name] = artifact
    for name, artifact in selected.items():
        if artifact['expired']:
            raise VerificationError('Latest result artifact has expired')
        if utc(artifact['created_at']) < utc(attempt['run_started_at']):
            if name == 'klaud-sweep-manifest':
                raise VerificationError('Manifest belongs to an earlier run attempt')
            if name in ('results_bmk', 'eval_results_all'):
                prefix = 'bmk_' if name == 'results_bmk' else 'eval_'
                if any(a['name'].startswith(prefix) and a['name'] != name
                       and not a['name'].startswith(('eval_server_logs_', 'eval_gpu_metrics_'))
                       and utc(a['created_at']) >= utc(attempt['run_started_at']) for a in inventory):
                    raise VerificationError('Stale aggregate after a point producer reran')
    if 'klaud-sweep-manifest' not in selected:
        raise VerificationError('Missing final-sweep manifest')
    return list(selected.values())


def verify_sweep(repository: str, run: dict, family: str) -> tuple[dict, list[dict], list[dict]]:
    if run['status'] != 'completed' or run['conclusion'] != 'success':
        raise VerificationError('Final sweep has not passed')
    attempt = github.read(repository, f'actions/runs/{run["id"]}/attempts/{run["run_attempt"]}')
    inventory = select_artifacts(github.artifacts(repository, run['id']), run, attempt)
    canonical = canonical_matrix(repository, run['head_sha'], family)
    with tempfile.TemporaryDirectory(prefix='klaud-validation-') as temp:
        directory = Path(temp)
        for artifact in inventory:
            github.download_json(repository, artifact, directory / artifact['name'])
        manifest = json.loads((directory / 'klaud-sweep-manifest/sweep_manifest.json').read_text())
        check_coverage(directory, manifest, run, family, canonical)
        paths = list((directory / 'results_bmk').glob('*.json'))
        fixed = [row for _, row in reuse.json_rows(paths) if row.get('scenario_type') != 'agentic-coding']
        agentic = [row for _, row in reuse.json_rows(reuse.agentic_point_files(directory))]
        evals = [row for _, row in reuse.json_rows((directory / 'eval_results_all').glob('*.json'))]
        return canonical, fixed + agentic, evals
