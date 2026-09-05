"""Select Klaude candidates or check capacity before a benchmark dispatch."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from .api import PUBLIC, ReadError, fetch_capacity, fetch_catalog
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
    return selected


def plan(root: Path, directory: Path) -> None:
    policy = Policy()
    repository = os.environ['GITHUB_REPOSITORY']
    items, issues = fetch_catalog(policy)
    if issues:
        raise ValueError('Public image/release feed is unavailable or stale')
    occupied = {ref['ref'].removeprefix('refs/heads/') for ref in github_read(repository, 'git/matching-refs/heads/klaude/auto-')}
    pulls = github_read(repository, 'pulls?state=open&per_page=100', paginate=True)
    occupied.update(pr['head']['ref'] for page in pulls for pr in page)
    occupied.update(pr['head']['ref'].rsplit('-', 1)[0] + '-' for page in pulls for pr in page
                    if pr['head']['ref'].startswith('klaude/auto-'))
    candidates = choose(items, fetch_capacity(policy), occupied)
    base = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True, timeout=30).strip()
    contexts = [{**candidate, 'base': base, 'repository': repository,
                 'public-api': {'schema': PUBLIC + '/api/openapi.json',
                                'images': PUBLIC + '/api/v1/latest-images',
                                'releases': PUBLIC + '/api/v1/framework-releases'}} for candidate in candidates]
    open_prs = [{'number': pr['number'], 'title': pr['title'], 'url': pr['html_url'],
                 'author': pr['user']['login'], 'draft': pr['draft'],
                 'branch': pr['head']['ref'], 'head': pr['head']['sha']} for page in pulls for pr in page]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'candidates.json').write_text(json.dumps(contexts, indent=2, allow_nan=False) + '\n')
    (directory / 'open-prs.json').write_text(json.dumps(open_prs, indent=2) + '\n')
    if filename := os.environ.get('GITHUB_OUTPUT'):
        with open(filename, 'a') as output:
            output.write(f'has_candidates={str(bool(candidates)).lower()}\n')
            output.write(f'review_schema={json.dumps(PRReview.model_json_schema(by_alias=True))}\n')


def select(directory: Path, max_candidates: int) -> None:
    contexts = json.loads((directory / 'candidates.json').read_text())
    review = PRReview.model_validate_json(os.environ['KLAUDE_PR_REVIEW']) if contexts else PRReview(decisions=[])
    known = {candidate['id'] for candidate in contexts}
    decisions = {decision.candidate_id: decision for decision in review.decisions}
    if len(decisions) != len(review.decisions) or not decisions.keys() <= known:
        raise ValueError('PR review contains duplicate or unknown candidate IDs')
    selected = []
    families = {decision.family for decision in review.decisions if decision.decision != 'proceed'}
    for candidate in contexts:
        decision = decisions.get(candidate['id'])
        if decision is None or decision.decision != 'proceed' or decision.family in families:
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
        {'candidates': candidates, **review.model_dump(by_alias=True)}, indent=2) + '\n')
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
    capacity = commands.add_parser('check-capacity', help='Exit 0 below 20% node utilization; otherwise nonzero, without printing telemetry')
    capacity.add_argument('--cluster', required=True)
    args = parser.parse_args()
    try:
        if args.command == 'plan':
            plan(args.root, args.directory)
            return 0
        if args.command == 'select':
            if not 1 <= args.max_candidates_per_run <= 256:
                parser.error('--max-candidates-per-run must be between 1 and 256')
            select(args.directory, args.max_candidates_per_run)
            return 0
        return 0 if args.cluster in fetch_capacity(Policy()) else 1
    except ReadError:
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
