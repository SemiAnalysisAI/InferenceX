"""Select Klaud Cold candidates or check capacity before a benchmark dispatch."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import re
import shlex
import subprocess
from types import SimpleNamespace

from .api import PUBLIC, ReadError, capacity_context, fetch_capacity, fetch_catalog
from .github import VerificationError, read as github_read
from .models import CandidateOutcome, OwnedCandidate, Ownership, PRReview, Policy, identity


def observation_key(row: dict) -> tuple:
    return tuple(row.get(key) for key in (
        'model', 'hardware', 'framework', 'precision', 'spec_method', 'disagg',
        'benchmark_type', 'isl', 'osl', 'image'))


def live_families(root: Path) -> dict[tuple, set[str]]:
    """Resolve public identities from the canonical generator, excluding archives."""
    from infx.matrix.generate import _hardware_family, generate_test_config_sweep
    from infx.matrix.validation import load_config_files, load_runner_file

    runners = load_runner_file(str(root / 'configs/runners.yaml'))
    index: dict[tuple, set[str]] = {}
    paths = sorted((root / 'configs').glob('*-master.yaml'))
    if not paths:
        raise ReadError('live-configs-unavailable')
    for path in paths:
        configs = load_config_files([str(path)])
        for key in configs:
            family = f'{path.relative_to(root).as_posix()}:{key}'
            try:
                entries = generate_test_config_sweep(SimpleNamespace(config_keys=[key]), configs, runners)
            except ValueError:
                # One unrenderable family must not hide unrelated working recipes.
                print(f'::warning::Klaud live-family-unrenderable: {family}')
                continue
            for entry in entries:
                agentic = entry.get('scenario-type') == 'agentic-coding'
                row = {'model': entry['model-prefix'], 'hardware': _hardware_family(entry['runner']),
                       'framework': entry['framework'], 'precision': entry['precision'],
                       'spec_method': entry['spec-decoding'], 'disagg': entry.get('disagg', False),
                       'benchmark_type': 'agentic_traces' if agentic else 'single_turn',
                       'isl': None if agentic else entry['isl'], 'osl': None if agentic else entry['osl'],
                       'image': entry['image']}
                index.setdefault(observation_key(row), set()).add(family)
    return index


def choose(items: list[dict], available: set[str], occupied: set[str],
           families: dict[tuple, set[str]]) -> list[dict]:
    # Preserve claims made before the agent's spelling was corrected.
    occupied = {re.sub(r'^klaud[e]?/auto-', 'klaud/auto-', branch) for branch in occupied}
    selected = []
    seen = set()
    valid = [item for item in items if item['needs-review']]
    # Prefer the latest matching baseline within each live family before shuffling.
    for item in sorted(valid, key=lambda item: (item['source']['date'], item['source-id']), reverse=True):
        row = item['source']
        if not any(cluster == row['hardware'] or cluster.startswith(row['hardware'] + '-')
                   for cluster in available):
            continue
        legacy = identity({key: row[key] for key in ('model', 'hardware', 'framework', 'precision', 'spec_method', 'disagg')})[:16]
        release_key = identity([row['image'], item['release']])[:16]
        for family in sorted(families.get(observation_key(row), ())):
            prefix = 'klaud/auto-' + identity(family)[:16] + '-'
            branch = prefix + release_key
            claims = {branch, prefix, f'klaud/auto-{legacy}-{release_key}', f'klaud/auto-{legacy}-'}
            if family in seen or claims & occupied:
                continue
            selected.append({'id': branch.removeprefix('klaud/auto-'), 'family': family, 'source': row,
                             'release': item['release'], 'review-reasons': item['review-reasons'], 'branch': branch})
            seen.add(family)
    random.shuffle(selected)
    return selected


def plan(root: Path, directory: Path) -> None:
    policy = Policy()
    repository = os.environ['GITHUB_REPOSITORY']
    items, issues = fetch_catalog(policy)
    if issues:
        raise ReadError('public-feed-invalid: ' + ', '.join(issues))
    refs = github_read(repository, 'git/matching-refs/heads/klaud', paginate=True)
    occupied = {ref['ref'].removeprefix('refs/heads/') for page in refs for ref in page}
    pulls = github_read(repository, 'pulls?state=open&per_page=100', paginate=True)
    occupied.update(pr['head']['ref'] for page in pulls for pr in page)
    occupied.update(pr['head']['ref'].rsplit('-', 1)[0] + '-' for page in pulls for pr in page
                    if re.match(r'^klaud[e]?/auto-', pr['head']['ref']))
    capacity = capacity_context(policy)
    available = set(capacity['eligible-telemetry-clusters'])
    candidates = choose(items, available, occupied, live_families(root))
    base = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True, timeout=30).strip()
    contexts = [{**candidate, 'base': base, 'repository': repository,
                 'public-api': {'schema': PUBLIC + '/api/openapi.json',
                                'images': PUBLIC + '/api/v1/latest-images',
                                'releases': PUBLIC + '/api/v1/framework-releases'}} for candidate in candidates]
    open_prs = [{'number': pr['number'], 'title': pr['title'], 'url': pr['html_url'],
                 'author': pr['user']['login'], 'draft': pr['draft'],
                 'branch': pr['head']['ref'], 'head': pr['head']['sha']} for page in pulls for pr in page]
    if candidates:
        for pr in open_prs:
            try:
                pages = github_read(repository, f'pulls/{pr["number"]}/files?per_page=100', paginate=True)
            except (subprocess.SubprocessError, ValueError):
                pr.update({'files': [], 'files-complete': False})
                continue
            pr['files'] = sorted({name for page in pages for file in page
                                  for name in (file['filename'], file.get('previous_filename')) if name})
            pr['files-complete'] = sum(len(page) for page in pages) < 3000
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'candidates.json').write_text(json.dumps(contexts, indent=2, allow_nan=False) + '\n')
    (directory / 'open-prs.json').write_text(json.dumps(open_prs, indent=2) + '\n')
    # Private selection hints for the read-only reviewer; excluded from artifacts.
    (directory / 'capacity.json').write_text(json.dumps(capacity) + '\n')
    if filename := os.environ.get('GITHUB_OUTPUT'):
        with open(filename, 'a') as output:
            output.write(f'has_candidates={str(bool(candidates)).lower()}\n')
            output.write(f'review_schema={json.dumps(PRReview.model_json_schema(by_alias=True))}\n')


def denied_bash_category(denial: dict) -> str:
    """Retain only fixed categories, never command text, arguments or paths."""
    command = denial.get('tool_input', {})
    command = command.get('command') if isinstance(command, dict) else None
    if not isinstance(command, str):
        return 'unknown'
    try:
        words = shlex.split(command)
    except ValueError:
        return 'unknown'
    if not words:
        return 'unknown'
    if words[0] in {'cd', 'env', 'bash', 'sh'} or '=' in words[0]:
        return 'shell-wrapper'
    if words[0] in {'cat', 'ls', 'find', 'sed', 'awk', 'rg', 'grep', 'head', 'tail', 'pwd', 'wc', 'jq'}:
        return 'file-read-or-filter'
    if words[0] in {'python', 'python3', 'uv'}:
        return 'python-or-uv'
    if words[0] == 'git':
        return 'git'
    if words[0] == 'gh':
        return 'github-cli'
    return 'other'


def execution_diagnostics(path: Path) -> dict:
    try:
        messages = json.loads(path.read_text())
        if not isinstance(messages, list):
            raise ValueError('Expected execution message list')
    except (OSError, ValueError):
        return {'execution-log': 'unavailable'}
    result = next((message for message in reversed(messages)
                   if isinstance(message, dict) and message.get('type') == 'result'), {})
    metrics = {key: value for key in ('duration_ms', 'num_turns', 'total_cost_usd')
               if (type(value := result.get(key)) is int or type(value) is float and math.isfinite(value)) and value >= 0}
    denials = result.get('permission_denials', [])
    if not isinstance(denials, list):
        denials = []
    tools = {'Read', 'Glob', 'Grep', 'Bash', 'WebFetch', 'WebSearch', 'Write', 'Edit', 'Agent', 'Task', 'StructuredOutput'}
    denied_tools = Counter(denial['tool_name'] if isinstance(denial.get('tool_name'), str) and denial['tool_name'] in tools else 'other'
                           for denial in denials if isinstance(denial, dict))
    subtypes = {'success', 'error_max_turns', 'error_during_execution', 'error_max_budget_usd',
                'error_max_structured_output_retries'}
    subtype = result.get('subtype')
    return {'execution-log': 'available', 'result-present': bool(result), **metrics,
            'termination': subtype if isinstance(subtype, str) and subtype in subtypes else 'unknown',
            'is-error': result.get('is_error') if type(result.get('is_error')) is bool else None,
            'permission-denials': dict(denied_tools),
            'denied-bash-categories': dict(Counter(denied_bash_category(denial) for denial in denials
                                                 if isinstance(denial, dict) and denial.get('tool_name') == 'Bash'))}


def check_stop() -> dict:
    """Require a verified terminal outcome, not merely an agent's final response."""
    from .lifecycle import current_session
    try:
        session = current_session()
        pulls = session.pulls()
        if pulls and session.handed_off(pulls[0]):
            return {}
        outcome = CandidateOutcome.model_validate_json(
            (Path(os.environ['KLAUD_EVIDENCE']) / 'outcome.json').read_text())
        session.verify(outcome)
    except (KeyError, TypeError, ValueError, subprocess.SubprocessError, OSError):
        return {'decision': 'block', 'reason': 'No verified terminal outcome. Continue monitoring and repairs, then run the documented finish command to validate or complete failure/deferral cleanup and reporting. Inspect read failures; never dispatch replacements or cancel healthy work just to stop.'}
    return {}


def save_diagnostics(execution_file: Path, outcome_file: Path, action_outcome: str, output: Path) -> bool:
    diagnostics = {'action-outcome': action_outcome, **execution_diagnostics(execution_file)}
    try:
        outcome = CandidateOutcome.model_validate_json(outcome_file.read_text())
        if action_outcome != 'success':
            raise ValueError('Action did not complete successfully')
        from .lifecycle import current_session
        current_session().verify(outcome)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        # Never echo invalid structured output, which could contain private data.
        outcome = CandidateOutcome(outcome='unexpected-error', phase='unknown',
                                   pull_request=None, run_ids=[], repairs_used=None)
        diagnostics['outcome-report'] = 'unavailable-or-invalid'
    else:
        diagnostics['outcome-report'] = 'available'
    diagnostics['candidate-outcome'] = outcome.model_dump(by_alias=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(diagnostics, indent=2) + '\n')
    if filename := os.environ.get('GITHUB_STEP_SUMMARY'):
        repository = os.environ['GITHUB_REPOSITORY']
        prefix = f'https://github.com/{repository}'
        pr = f'[#{outcome.pull_request}]({prefix}/pull/{outcome.pull_request})' if outcome.pull_request else '—'
        runs = ', '.join(f'[{run}]({prefix}/actions/runs/{run})' for run in outcome.run_ids) or '—'
        repairs = str(outcome.repairs_used) if outcome.repairs_used is not None else 'unknown / 未知'
        with open(filename, 'a') as summary:
            summary.write('| Outcome / 结果 | Phase / 阶段 | PR | Repairs / 修复 | Runs / 运行 |\n'
                          '| --- | --- | --- | --- | --- |\n'
                          f'| {outcome.outcome} | {outcome.phase} | {pr} | {repairs} | {runs} |\n')
    return outcome.outcome != 'unexpected-error'


def select(directory: Path, max_candidates: int, execution_file: Path | None = None) -> None:
    contexts = json.loads((directory / 'candidates.json').read_text())
    review = PRReview(decisions=[])
    deferred = None
    if contexts:
        if os.environ.get('KLAUD_REVIEW_OUTCOME', 'success') != 'success':
            deferred = 'review-action-failed'
        else:
            try:
                review = PRReview.model_validate_json(os.environ.get('KLAUD_PR_REVIEW', ''))
                ids = [decision.candidate_id for decision in review.decisions]
                if len(set(ids)) != len(ids) or not set(ids) <= {candidate['id'] for candidate in contexts}:
                    raise ValueError('Duplicate or unknown candidate IDs')
                expected = {candidate['id']: candidate['family'] for candidate in contexts}
                if any(decision.family != expected[decision.candidate_id] for decision in review.decisions
                       if decision.decision == 'proceed'):
                    raise ValueError('Review changed the resolved live family')
            except ValueError:
                deferred = 'invalid-review-output'
                review = PRReview(decisions=[])
    decisions = {decision.candidate_id: decision for decision in review.decisions}
    selected = []
    available = set()
    if any(decision.decision == 'proceed' for decision in review.decisions):
        try:
            available = fetch_capacity(Policy())
        except ReadError:
            deferred = 'capacity-unavailable'
    capacity_deferred = []
    families = {decision.family for decision in review.decisions if decision.decision != 'proceed'}
    for candidate in contexts:
        decision = decisions.get(candidate['id'])
        if decision is None or decision.decision != 'proceed' or decision.family in families:
            continue
        if not set(decision.telemetry_clusters) <= available:
            capacity_deferred.append(candidate['id'])
            continue
        selected.append({**candidate, 'pr-review': decision.model_dump(by_alias=True)})
        families.add(decision.family)
        if len(selected) >= max_candidates:
            break
    for candidate in selected:
        target = directory / candidate['id']
        target.mkdir(parents=True, exist_ok=True)
        (target / 'candidate.json').write_text(json.dumps(candidate, indent=2, allow_nan=False) + '\n')
    ownership = Ownership(run_id=int(os.environ['GITHUB_RUN_ID']), candidates=[
        OwnedCandidate.model_validate({key: candidate[key] for key in ('id', 'family', 'base')})
        for candidate in selected])
    (directory / 'ownership.json').write_text(ownership.model_dump_json(by_alias=True) + '\n')
    candidates = [candidate['id'] for candidate in selected]
    (directory / 'selection.json').write_text(json.dumps(
        {'candidates': candidates, 'deferred-reason': deferred, 'capacity-deferred-candidates': capacity_deferred,
         **review.model_dump(by_alias=True)}, indent=2) + '\n')
    if execution_file is not None:
        (directory / 'review-diagnostics.json').write_text(json.dumps(execution_diagnostics(execution_file), indent=2) + '\n')
    summary = f'Klaud Cold: selected {len(candidates)} of {len(contexts)} eligible candidates.'
    if deferred:
        summary += f' Invocation deferred: {deferred}; no candidates launched.'
        print(f'::warning::{summary}')
    if capacity_deferred:
        summary += f' {len(capacity_deferred)} reviewed candidates deferred by the latest capacity check.'
    if filename := os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(filename, 'a') as output:
            output.write(summary + '\n\nSee selection.json and review-diagnostics.json in the klaud-plan artifact.\n')
    if filename := os.environ.get('GITHUB_OUTPUT'):
        with open(filename, 'a') as output:
            output.write(f'selected={str(bool(candidates)).lower()}\ncandidates={json.dumps(candidates)}\n')


def main() -> int:
    parser = argparse.ArgumentParser(prog='python -m utils.klaud', description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd(), help='InferenceX checkout used to resolve the candidate base SHA')
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('plan', help='Prepare candidates and open PRs for overlap review')
    prepare.add_argument('--directory', type=Path, required=True, help='Output directory for candidate context')
    selection = commands.add_parser('select', help='Validate KLAUD_PR_REVIEW and select nonoverlapping candidates')
    selection.add_argument('--directory', type=Path, required=True)
    selection.add_argument('--max-candidates-per-run', type=int, required=True, help='Maximum candidates to select (1-256)')
    selection.add_argument('--execution-file', type=Path, help='Claude execution log; retain numeric metrics and fixed denial categories')
    capacity = commands.add_parser('check-capacity', help='Exit 0 with available nodes below 80%% utilization; otherwise nonzero, without printing telemetry')
    capacity.add_argument('--cluster', required=True, action='append', help='Exact telemetry cluster; repeat for every possible recipe target')
    commands.add_parser('check-stop', help='Claude Stop hook: require finished runs and validated, closed or handed-off PRs')
    commands.add_parser('recover', help='Reconcile interrupted sessions from completed autosweeps')
    finish = commands.add_parser('finish', help='Verify validation or finish owned cleanup and reporting')
    finish.add_argument('--outcome-file', type=Path, required=True)
    commands.add_parser('outcome-schema', help='Print the public-safe candidate outcome schema')
    diagnostics = commands.add_parser('diagnostics', help='Save sanitized candidate outcomes, termination metrics and permission categories')
    diagnostics.add_argument('--execution-file', type=Path, required=True)
    diagnostics.add_argument('--structured-outcome-file', type=Path, required=True)
    diagnostics.add_argument('--output', type=Path, required=True)
    diagnostics.add_argument('--outcome', choices=['success', 'failure', 'cancelled', 'skipped', 'unknown'], default='unknown')
    args = parser.parse_args()
    try:
        if args.command == 'recover':
            from .lifecycle import recover
            recover()
            return 0
        if args.command == 'finish':
            from .lifecycle import current_session
            outcome = CandidateOutcome.model_validate_json(args.outcome_file.read_text())
            outcome = current_session().finish(outcome)
            (Path(os.environ['KLAUD_EVIDENCE']) / 'outcome.json').write_text(
                outcome.model_dump_json(by_alias=True) + '\n')
            return 0
        if args.command == 'outcome-schema':
            print(json.dumps(CandidateOutcome.model_json_schema(by_alias=True)))
            return 0
        if args.command == 'check-stop':
            print(json.dumps(check_stop()))
            return 0
        if args.command == 'diagnostics':
            return 0 if save_diagnostics(args.execution_file, args.structured_outcome_file,
                                        args.outcome, args.output) else 1
        if args.command == 'plan':
            plan(args.root, args.directory)
            return 0
        if args.command == 'select':
            if not 1 <= args.max_candidates_per_run <= 256:
                parser.error('--max-candidates-per-run must be between 1 and 256')
            select(args.directory, args.max_candidates_per_run, args.execution_file)
            return 0
        return 0 if set(args.cluster) <= fetch_capacity(Policy()) else 1
    except ReadError as error:
        if args.command == 'plan':
            # ReadError contains fixed codes only, never response bodies or credentials.
            message = f'Klaud Cold preparation failed: {error}'
            print(f'::error::{message}')
            if filename := os.environ.get('GITHUB_STEP_SUMMARY'):
                with open(filename, 'a') as output:
                    output.write(message + '\n')
        return 1
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        reason = str(error) if isinstance(error, VerificationError) else 'State unavailable or invalid; inspect GitHub before retrying'
        print(f'::error::Klaud: {reason}.')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
