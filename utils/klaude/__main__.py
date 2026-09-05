"""Select Klaud Cold candidates or check capacity before a benchmark dispatch."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import shlex
import subprocess

from .api import PUBLIC, ReadError, capacity_context, fetch_capacity, fetch_catalog
from .models import PRReview, Policy, identity


def github_read(repository: str, path: str, *, paginate: bool = False) -> list:
    args = ['gh', 'api', '--method', 'GET']
    if paginate:
        args.extend(['--paginate', '--slurp'])
    return json.loads(subprocess.check_output(
        [*args, f'repos/{repository}/{path}'], text=True, timeout=60))


def choose(items: list[dict], available: set[str], occupied: set[str]) -> list[dict]:
    selected = []
    seen = set()
    valid = [item for item in items if item['needs-review']]
    for item in sorted(valid, key=lambda item: (item['source']['date'], item['source-id'])):
        row = item['source']
        if not any(cluster == row['hardware'] or cluster.startswith(row['hardware'] + '-')
                   for cluster in available):
            continue
        family = identity({key: row[key] for key in ('model', 'hardware', 'framework', 'precision', 'spec_method', 'disagg')})[:16]
        prefix = 'klaude/auto-' + family + '-'
        branch = prefix + identity([row['image'], item['release']])[:16]
        if family in seen or branch in occupied or prefix in occupied:
            continue
        selected.append({'id': branch.removeprefix('klaude/auto-'), 'source': row,
                         'release': item['release'], 'review-reasons': item['review-reasons'], 'branch': branch})
        seen.add(family)
    random.shuffle(selected)
    return selected


def plan(root: Path, directory: Path) -> None:
    policy = Policy()
    repository = os.environ['GITHUB_REPOSITORY']
    items, issues = fetch_catalog(policy)
    if issues:
        raise ValueError('Public image/release feed is unavailable or stale')
    refs = github_read(repository, 'git/matching-refs/heads/klaude/auto-', paginate=True)
    occupied = {ref['ref'].removeprefix('refs/heads/') for page in refs for ref in page}
    pulls = github_read(repository, 'pulls?state=open&per_page=100', paginate=True)
    occupied.update(pr['head']['ref'] for page in pulls for pr in page)
    occupied.update(pr['head']['ref'].rsplit('-', 1)[0] + '-' for page in pulls for pr in page
                    if pr['head']['ref'].startswith('klaude/auto-'))
    capacity = capacity_context(policy)
    available = set(capacity['eligible-telemetry-clusters'])
    candidates = choose(items, available, occupied)
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


def review_diagnostics(path: Path) -> dict:
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
    return {'execution-log': 'available', 'result-present': bool(result), **metrics,
            'permission-denials': dict(denied_tools),
            'denied-bash-categories': dict(Counter(denied_bash_category(denial) for denial in denials
                                                 if isinstance(denial, dict) and denial.get('tool_name') == 'Bash'))}


def select(directory: Path, max_candidates: int, execution_file: Path | None = None) -> None:
    contexts = json.loads((directory / 'candidates.json').read_text())
    review = PRReview(decisions=[])
    deferred = None
    if contexts:
        if os.environ.get('KLAUDE_REVIEW_OUTCOME', 'success') != 'success':
            deferred = 'review-action-failed'
        else:
            try:
                review = PRReview.model_validate_json(os.environ.get('KLAUDE_PR_REVIEW', ''))
                ids = [decision.candidate_id for decision in review.decisions]
                if len(set(ids)) != len(ids) or not set(ids) <= {candidate['id'] for candidate in contexts}:
                    raise ValueError('Duplicate or unknown candidate IDs')
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
    candidates = [candidate['id'] for candidate in selected]
    (directory / 'selection.json').write_text(json.dumps(
        {'candidates': candidates, 'deferred-reason': deferred, 'capacity-deferred-candidates': capacity_deferred,
         **review.model_dump(by_alias=True)}, indent=2) + '\n')
    if execution_file is not None:
        (directory / 'review-diagnostics.json').write_text(json.dumps(review_diagnostics(execution_file), indent=2) + '\n')
    summary = f'Klaud Cold: selected {len(candidates)} of {len(contexts)} eligible candidates.'
    if deferred:
        summary += f' Invocation deferred: {deferred}; no candidates launched.'
        print(f'::warning::{summary}')
    if capacity_deferred:
        summary += f' {len(capacity_deferred)} reviewed candidates deferred by the latest capacity check.'
    if filename := os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(filename, 'a') as output:
            output.write(summary + '\n\nSee selection.json and review-diagnostics.json in the klaude-plan artifact.\n')
    if filename := os.environ.get('GITHUB_OUTPUT'):
        with open(filename, 'a') as output:
            output.write(f'selected={str(bool(candidates)).lower()}\ncandidates={json.dumps(candidates)}\n')


def main() -> int:
    parser = argparse.ArgumentParser(prog='python -m utils.klaude', description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd(), help='InferenceX checkout used to resolve the candidate base SHA')
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('plan', help='Prepare candidates and open PRs for overlap review')
    prepare.add_argument('--directory', type=Path, required=True, help='Output directory for candidate context')
    selection = commands.add_parser('select', help='Validate KLAUDE_PR_REVIEW and select nonoverlapping candidates')
    selection.add_argument('--directory', type=Path, required=True)
    selection.add_argument('--max-candidates-per-run', type=int, required=True, help='Maximum candidates to select (1-256)')
    selection.add_argument('--execution-file', type=Path, help='Claude execution log; retain numeric metrics and fixed denial categories')
    capacity = commands.add_parser('check-capacity', help='Exit 0 with available nodes below 20%% utilization; otherwise nonzero, without printing telemetry')
    capacity.add_argument('--cluster', required=True, action='append', help='Exact telemetry cluster; repeat for every possible recipe target')
    args = parser.parse_args()
    try:
        if args.command == 'plan':
            plan(args.root, args.directory)
            return 0
        if args.command == 'select':
            if not 1 <= args.max_candidates_per_run <= 256:
                parser.error('--max-candidates-per-run must be between 1 and 256')
            select(args.directory, args.max_candidates_per_run, args.execution_file)
            return 0
        return 0 if set(args.cluster) <= fetch_capacity(Policy()) else 1
    except ReadError:
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
