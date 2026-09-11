"""Finish owned sessions and reconcile interrupted ones in the next autosweep."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
from urllib.parse import urlencode

from . import github
from .github import VerificationError
from .models import CandidateOutcome, OwnedCandidate, Ownership, utc
from .validation import verify_sweep
from . import claims

BOT = 'Klaud-Cold'
RELEASE = {'capacity-deferred', 'readiness-blocked'}
SWEEP_LABELS = {'sweep-enabled', 'full-sweep-enabled', 'non-canary-full-sweep-enabled',
                'full-sweep-fail-fast', 'full-sweep-fail-fast-no-canary', 'all-evals', 'evals-only', 'agentx-fast'}


class PendingCleanup(VerificationError):
    """GitHub has accepted a transition but its child jobs are not terminal yet."""


def terminal(run: dict) -> bool:
    return run['status'] == 'completed' and bool(run.get('conclusion'))


class Session:
    def __init__(self, repository: str, parent: dict, candidate: OwnedCandidate, *, recovering: bool = False):
        self.repository, self.parent, self.candidate = repository, parent, candidate
        self.recovering = recovering
        self.branch = f'klaud/auto-{candidate.id}'
        self.marker = f'<!-- klaud-outcome:{parent["id"]}:{candidate.id}\n'
        self.validations = {}

    def validation(self, run: dict):
        key = (run['id'], run['head_sha'], run['run_attempt'])
        if key not in self.validations:
            self.validations[key] = verify_sweep(self.repository, run, self.candidate.family)
        return self.validations[key]

    def pulls(self) -> list[dict]:
        query = urlencode({'state': 'all', 'head': self.repository.split('/')[0] + ':' + self.branch,
                           'per_page': 100})
        pulls = [pull for pull in github.items(self.repository, 'pulls?' + query)
                 if utc(pull['created_at']) >= utc(self.parent['created_at'])
                 and (not terminal(self.parent) or utc(pull['created_at']) <= utc(self.parent['updated_at']))]
        if len(pulls) > 1:
            raise VerificationError('Ambiguous candidate PR')
        if pulls:
            pull = pulls[0]
            if pull['user']['login'] != BOT or pull['head']['repo']['full_name'] != self.repository:
                raise VerificationError('Candidate ownership mismatch')
        return pulls

    def runs(self) -> list[dict]:
        query = {'per_page': 100, 'created': '>=' + self.parent['created_at']}
        targeted = github.items(self.repository, 'actions/workflows/e2e-tests.yml/runs?' + urlencode(
            {**query, 'event': 'workflow_dispatch'}), 'workflow_runs')
        title = f'e2e Test - klaud-{self.parent["id"]}-{self.candidate.id}'
        targeted = [run for run in targeted if run['display_title'] == title
                    and run['actor']['login'] == BOT and run['head_repository']['full_name'] == self.repository]
        sweeps = github.items(self.repository, 'actions/workflows/run-sweep.yml/runs?' + urlencode(
            {**query, 'event': 'pull_request', 'branch': self.branch}), 'workflow_runs')
        pulls = self.pulls()
        sweeps = [run for run in sweeps if pulls and run['head_repository']['full_name'] == self.repository
                  and utc(run['created_at']) >= utc(pulls[0]['created_at'])
                  and (not run.get('pull_requests') or any(
                      pr['number'] == pulls[0]['number'] for pr in run['pull_requests']))]
        if self.recovering and any(utc(run['created_at']) > utc(self.parent['updated_at'])
                                   and run['actor']['login'] != BOT for run in sweeps):
            raise VerificationError('New maintainer runs require explicit handoff')
        return targeted + sweeps

    def handed_off(self, pull: dict) -> bool:
        return any(label['name'] == 'klaud-handoff' for label in pull['labels'])

    def refresh(self, pull: dict) -> dict:
        self.check_parent()
        current = github.read(self.repository, f'pulls/{pull["number"]}')
        if (current['head']['sha'] != pull['head']['sha'] or current['merged_at']
                or current['user']['login'] != BOT or self.handed_off(current)):
            raise VerificationError('Ownership changed; leave maintainer work intact')
        return current

    def check_parent(self) -> None:
        if self.recovering and not terminal(github.read(self.repository, f'actions/runs/{self.parent["id"]}')):
            raise VerificationError('Parent resumed; leave active session intact')

    def report(self, pull: dict) -> dict | None:
        comments = github.items(self.repository, f'issues/{pull["number"]}/comments?per_page=100')
        matches = [comment for comment in comments if comment['user']['login'] == BOT
                   and comment['body'].startswith(self.marker)]
        if not matches:
            return None
        body = max(matches, key=lambda c: c['id'])['body']
        return json.loads(body[len(self.marker):].split('\n-->', 1)[0])

    def pending(self, pull: dict) -> CandidateOutcome | None:
        marker = self.marker.replace('klaud-outcome:', 'klaud-cleanup:')
        comments = github.items(self.repository, f'issues/{pull["number"]}/comments?per_page=100')
        matches = [c for c in comments if c['user']['login'] == BOT and c['body'].startswith(marker)]
        if not matches:
            return None
        body = max(matches, key=lambda c: c['id'])['body']
        record = json.loads(body[len(marker):].split('\n-->', 1)[0])
        if record['head'] != pull['head']['sha']:
            raise VerificationError('PR head changed after cleanup was requested')
        outcome = CandidateOutcome.model_validate(record['outcome'])
        if outcome.outcome in ('validated', 'handoff'):
            raise VerificationError('Invalid pending cleanup category')
        return outcome

    def branch_exists(self) -> bool:
        refs = github.items(self.repository, 'git/matching-refs/heads/' + self.branch)
        return any(ref['ref'] == 'refs/heads/' + self.branch for ref in refs)

    def retry_released(self, pull: dict) -> bool:
        marker = f'<!-- klaud-retry-release:{self.parent["id"]}:{self.candidate.id}:{pull["head"]["sha"]} -->'
        comments = github.items(self.repository, f'issues/{pull["number"]}/comments?per_page=100')
        for comment in comments:
            actor = comment['user']['login']
            if actor != BOT and comment['body'].startswith(marker):
                permission = github.read(self.repository, f'collaborators/{actor}/permission')['permission']
                if permission in ('admin', 'maintain', 'write'):
                    return True
        return False

    def verify(self, outcome: CandidateOutcome, *, require_report: bool = True,
               require_ready: bool = True) -> None:
        pulls = self.pulls()
        pull = pulls[0] if pulls else None
        if outcome.outcome == 'handoff':
            if not pull or pull['number'] != outcome.pull_request or not self.handed_off(pull):
                raise VerificationError('No explicit maintainer handoff')
            return
        if pull and (self.handed_off(pull) or pull['merged_at']):
            raise VerificationError('Maintainer owns the PR')
        runs = self.runs()
        if any(not terminal(run) for run in runs):
            raise PendingCleanup('Owned jobs are unfinished')
        if set(outcome.run_ids) != {run['id'] for run in runs}:
            raise VerificationError('Outcome omits or invents owned runs')
        if outcome.pull_request != (pull['number'] if pull else None):
            raise VerificationError('Outcome PR mismatch')
        if outcome.outcome == 'validated':
            if not pull or pull['state'] != 'open' or (require_ready and pull['draft']) or not any(
                    label['name'] == 'full-sweep-enabled' for label in pull['labels']):
                raise VerificationError('Validated PR must remain ready for review')
            if {label['name'] for label in pull['labels']} & SWEEP_LABELS != {'full-sweep-enabled'}:
                raise VerificationError('Validated PR has incompatible sweep labels')
            finals = [run for run in runs if run['event'] == 'pull_request'
                      and run['head_sha'] == pull['head']['sha'] and run['conclusion'] != 'skipped']
            if not finals:
                raise VerificationError('No exact-head final sweep')
            latest = max(finals, key=lambda run: run['created_at'])
            if latest['conclusion'] != 'success':
                raise VerificationError('Latest exact-head final sweep failed')
            receipt = self.report(pull) if require_report else None
            proof = {'head': pull['head']['sha'], 'run-id': latest['id'], 'run-attempt': latest['run_attempt']}
            if receipt and receipt.get('validation') == proof:
                inventory = github.artifacts(self.repository, latest['id'])
                if not any(a['name'] == 'klaud-sweep-manifest' and not a['expired'] for a in inventory):
                    raise VerificationError('Verified sweep artifacts have expired')
            if not receipt or receipt.get('validation') != proof:
                self.validation(latest)
        else:
            if pull and (pull['state'] != 'closed' or not pull['draft'] or any(
                    label['name'] in SWEEP_LABELS for label in pull['labels'])):
                raise VerificationError('Failure/deferral PR cleanup is incomplete')
            if pull and self.branch_exists() != (outcome.outcome not in RELEASE) and not (
                    outcome.outcome not in RELEASE and self.retry_released(pull)):
                raise VerificationError('Incorrect cleanup branch disposition')
        if pull and require_report:
            receipt = self.report(pull)
            if (not receipt or receipt.get('head') != pull['head']['sha']
                    or receipt['outcome'] != outcome.model_dump(by_alias=True)):
                raise VerificationError('Missing verified completion report')

    def finish(self, outcome: CandidateOutcome) -> CandidateOutcome:
        pulls = self.pulls()
        pull = pulls[0] if pulls else None
        runs = self.runs()
        if pull and self.handed_off(pull):
            return CandidateOutcome(outcome='handoff', phase='cleanup', pull_request=pull['number'],
                                    run_ids=[r['id'] for r in runs], repairs_used=outcome.repairs_used)
        if pull and pull['merged_at']:
            raise VerificationError('Merged PR belongs to maintainers')
        if outcome.outcome == 'handoff':
            raise VerificationError('No explicit maintainer handoff')
        outcome = outcome.model_copy(update={'pull_request': pull['number'] if pull else None,
                                            'run_ids': sorted(run['id'] for run in runs)})
        if outcome.outcome == 'validated' and pull:
            # Reviews start at this transition: prove AND publish results first.
            self.verify(outcome, require_report=False, require_ready=False)
            from .reporting import publish_final
            final = max((run for run in runs if run['event'] == 'pull_request'
                         and run['head_sha'] == pull['head']['sha'] and run['conclusion'] != 'skipped'),
                        key=lambda run: run['created_at'])
            publish_final(self, final, self.validation(final))
            current = self.refresh(pull)
            if current['draft']:
                subprocess_ready(self.repository, pull['number'], undo=False)
        if outcome.outcome != 'validated':
            if pull and not self.report(pull):
                self.refresh(pull)
                if not self.pending(pull):
                    pending = self.marker.replace('klaud-outcome:', 'klaud-cleanup:')
                    request = {'head': pull['head']['sha'], 'outcome': outcome.model_dump(by_alias=True)}
                    body = (pending + json.dumps(request) + '\n-->\n'
                            f'Klaud Cold: **{outcome.outcome}**. Finishing cleanup; owned child runs will be stopped and checked before closure.')
                    github.write(self.repository, f'issues/{pull["number"]}/comments', 'POST', {'body': body})
            for run in runs:
                if not terminal(run):
                    self.check_parent()
                    if pull:
                        self.refresh(pull)
                    github.write(self.repository, f'actions/runs/{run["id"]}/cancel', 'POST')
            if any(not terminal(run) for run in self.runs()):
                raise PendingCleanup('Cancellation requested; wait for terminal jobs then finish again')
            if pull:
                current = self.refresh(pull)
                for label in current['labels']:
                    if label['name'] in SWEEP_LABELS:
                        self.refresh(pull)
                        github.write(self.repository, f'issues/{pull["number"]}/labels/{label["name"]}', 'DELETE')
                current = self.refresh(pull)
                if not current['draft']:
                    subprocess_ready(self.repository, pull['number'])
                self.refresh(pull)
                if current['state'] != 'closed':
                    github.write(self.repository, f'pulls/{pull["number"]}', 'PATCH', {'state': 'closed'})
                if outcome.outcome in RELEASE and self.branch_exists():
                    self.refresh(pull)
                    ref = github.read(self.repository, 'git/ref/heads/' + self.branch)
                    if ref['object']['sha'] != pull['head']['sha']:
                        raise VerificationError('Branch moved during cleanup')
                    github.write(self.repository, 'git/refs/heads/' + self.branch, 'DELETE')
        # PR transitions can enqueue skipped runs. Wait for them before reporting completion.
        outcome = outcome.model_copy(update={'run_ids': sorted(run['id'] for run in self.runs())})
        self.verify(outcome, require_report=False)
        if pull:
            self.refresh(pull)
            proof = None
            if outcome.outcome == 'validated':
                final = max((run for run in self.runs() if run['event'] == 'pull_request'
                             and run['head_sha'] == pull['head']['sha'] and run['conclusion'] != 'skipped'),
                            key=lambda run: run['created_at'])
                proof = {'head': pull['head']['sha'], 'run-id': final['id'], 'run-attempt': final['run_attempt']}
            record = {'head': pull['head']['sha'], 'outcome': outcome.model_dump(by_alias=True), 'validation': proof}
            if self.report(pull) != record:
                links = ', '.join(f'[{run}](https://github.com/{self.repository}/actions/runs/{run})' for run in outcome.run_ids) or '—'
                body = (self.marker + json.dumps(record) + '\n-->\n'
                        f'Klaud Cold: **{outcome.outcome}**. All owned runs are terminal. '
                        f'Repairs: {outcome.repairs_used if outcome.repairs_used is not None else "unknown"}. Runs: {links}.\n\n'
                        + ('The full sweep is verified; this PR remains ready for review.' if proof else
                           'PR closed; branch deleted for retry.' if outcome.outcome in RELEASE else
                           'PR closed; the exact-candidate branch is retained. See the attempt report for the failure and retry condition.'))
                github.write(self.repository, f'issues/{pull["number"]}/comments', 'POST', {'body': body})
        claims.release_family(self.repository, self.candidate, self.parent['id'])
        return outcome


def subprocess_ready(repository: str, number: int, *, undo: bool = True) -> None:
    subprocess.run(['gh', 'pr', 'ready', str(number), '--repo', repository] + (['--undo'] if undo else []),
                   check=True, capture_output=True, timeout=60)


def current_session() -> Session:
    context = json.loads((Path(os.environ['KLAUD_EVIDENCE']) / 'candidate.json').read_text())
    candidate = OwnedCandidate.model_validate({key: context[key] for key in ('id', 'family', 'base')})
    repository = os.environ['GITHUB_REPOSITORY']
    parent = github.read(repository, f'actions/runs/{int(os.environ["GITHUB_RUN_ID"])}')
    return Session(repository, parent, candidate)


def release_candidate(session: Session, expected_head: str) -> None:
    """Explicit maintainer retry after a verified closed session, never an agent action."""
    actor = json.loads(subprocess.check_output(['gh', 'api', 'user'], text=True, timeout=60))['login']
    permission = github.read(session.repository, f'collaborators/{actor}/permission')['permission']
    if actor == BOT or permission not in ('admin', 'maintain', 'write'):
        raise VerificationError('Only a repository maintainer can release a retained candidate')
    session.check_parent()
    pull = session.pulls()[0]
    if pull['head']['sha'] != expected_head or pull['state'] != 'closed' or pull['merged_at']:
        raise VerificationError('Retry release requires the reviewed closed PR head')
    receipt = session.report(pull)
    if not receipt:
        raise VerificationError('Finish session cleanup before releasing its retry claim')
    outcome = CandidateOutcome.model_validate(receipt['outcome'])
    if outcome.outcome in ('validated', 'handoff') or outcome.outcome in RELEASE:
        raise VerificationError('This outcome has no retained failure claim to release')
    session.verify(outcome)
    if not session.branch_exists() and session.retry_released(pull):
        return  # A previous invocation already completed the approved deletion.
    session.refresh(pull)
    branch = github.read(session.repository, 'git/ref/heads/' + session.branch)
    if branch['object']['sha'] != expected_head:
        raise VerificationError('Retained branch changed; leave maintainer work intact')
    # The outcome remains historically true. This explicit receipt records why the
    # absence of its retained branch must no longer be treated as incomplete cleanup.
    github.write(session.repository, f'issues/{pull["number"]}/comments', 'POST', {'body':
        f'<!-- klaud-retry-release:{session.parent["id"]}:{session.candidate.id}:{expected_head} -->\n'
        'A maintainer approved releasing this completed candidate for a fresh selection after reviewing its blocker. '
        'The previous results remain historical evidence.'})
    session.refresh(pull)
    if github.read(session.repository, 'git/ref/heads/' + session.branch)['object']['sha'] != expected_head:
        raise VerificationError('Retained branch changed; leave maintainer work intact')
    github.write(session.repository, 'git/refs/heads/' + session.branch, 'DELETE')


def reconcile(session: Session) -> bool:
    pulls = session.pulls()
    pull = pulls[0] if pulls else None
    if pull and (pull['merged_at'] or session.handed_off(pull)):
        return False
    record = session.report(pull) if pull else None
    if record:
        session.verify(CandidateOutcome.model_validate(record['outcome']))
        return False
    runs = session.runs()
    if not pull and not runs:
        return False
    pending = session.pending(pull) if pull else None
    # Preserve a completed successful sweep even if the agent died before reporting.
    final = [run for run in runs if pull and run['event'] == 'pull_request'
             and run['head_sha'] == pull['head']['sha'] and run['conclusion'] != 'skipped']
    if not pending and any(not terminal(run) for run in runs):
        # An interruption is not permission to stop healthy child work or block planning.
        # The durable ownership receipt keeps this family excluded until a later pass.
        raise PendingCleanup('Owned work is still running; retained for the next recovery pass')
    validated = bool(final and terminal(max(final, key=lambda r: r['created_at']))
                     and max(final, key=lambda r: r['created_at'])['conclusion'] == 'success')
    if validated and not pending:
        try:
            session.validation(max(final, key=lambda r: r['created_at']))
        except VerificationError:
            # A completed but uncertifiable run is a terminal inspection outcome,
            # not an indefinitely open draft. API/ownership uncertainty still bubbles
            # up and preserves work; no incompatibility is inferred from this failure.
            validated = False
    outcome = pending or CandidateOutcome(outcome='validated' if validated else 'unexpected-error',
                               phase='final-sweep' if validated else 'cleanup',
                               pull_request=pull['number'] if pull else None,
                               run_ids=[r['id'] for r in runs], repairs_used=None)
    # Unknown interruption retains the exact-candidate claim for manual review.
    session.finish(outcome)
    return True


def recover() -> None:
    repository = os.environ['GITHUB_REPOSITORY']
    inventory = github.items(repository, 'actions/artifacts?name=klaud-ownership&per_page=100', 'artifacts')
    # Refs cover a crash before artifact upload; artifacts cover legacy sessions.
    # Validate the entire inventory before mutating any owned work.
    owners = [(owner, None) for owner in claims.family_owners(repository)]
    for artifact in inventory:
        if artifact['expired']:
            continue
        parent = github.read(repository, f'actions/runs/{artifact["workflow_run"]["id"]}')
        if (parent['head_branch'] != 'main' or parent['path'] != '.github/workflows/klaud-plan.yml'
                or parent['event'] not in ('schedule', 'workflow_dispatch')):
            continue
        with tempfile.TemporaryDirectory(prefix='klaud-ownership-') as temp:
            github.download_json(repository, artifact, Path(temp))
            owner = Ownership.model_validate_json((Path(temp) / 'ownership.json').read_text())
        if owner.run_id != parent['id']:
            raise VerificationError('Ownership parent mismatch')
        owners.append((owner, artifact))
    parents = {owner.run_id: github.read(repository, f'actions/runs/{owner.run_id}') for owner, _ in owners}
    if any(parent['head_branch'] != 'main' or parent['path'] != '.github/workflows/klaud-plan.yml'
           or parent['event'] not in ('schedule', 'workflow_dispatch') for parent in parents.values()):
        raise VerificationError('Untrusted ownership parent')
    results = {}
    blocked = set()
    for owner, _ in sorted(owners, key=lambda pair: pair[0].run_id, reverse=True):
        parent = parents[owner.run_id]
        for candidate in owner.candidates:
            key = (parent['id'], candidate.id)
            if key in results:
                continue
            results[key] = False
            if not terminal(parent):
                blocked.add(candidate.family)
                continue
            session = Session(repository, parent, candidate, recovering=True)
            try:
                if not claims.recovery_lease(repository, parent['id'], candidate):
                    blocked.add(candidate.family)
                    continue
                changed = reconcile(session)
                claims.release_family(repository, candidate, parent['id'])
            except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
                blocked.add(candidate.family)
                reason = str(error) if isinstance(error, VerificationError) else 'state unavailable or invalid'
                status = ('pending / 等待后续收尾' if isinstance(error, PendingCleanup) else 'needs inspection / 需要检查') + f' ({reason})'
            else:
                results[key] = True
                status = 'reconciled / 已收尾' if changed else None
            if status and (summary := os.environ.get('GITHUB_STEP_SUMMARY')):
                with open(summary, 'a') as output:
                    output.write(f'Klaud `{candidate.id}`: {status}. '
                                 f'[Parent run / 父运行](https://github.com/{repository}/actions/runs/{parent["id"]})\n\n')
    for owner, artifact in owners:
        if artifact and all(results[(owner.run_id, candidate.id)] for candidate in owner.candidates):
            current = github.read(repository, f'actions/runs/{owner.run_id}')
            if terminal(current) and current['run_attempt'] == parents[owner.run_id]['run_attempt']:
                github.write(repository, f'actions/artifacts/{artifact["id"]}', 'DELETE')
    # Candidate failures are isolated, but their entire families remain unavailable.
    (Path(os.environ['RUNNER_TEMP']) / 'klaud-recovery.json').write_text(json.dumps(sorted(blocked)))
