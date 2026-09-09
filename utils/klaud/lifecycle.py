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

    def verify(self, outcome: CandidateOutcome, *, require_report: bool = True) -> None:
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
            if not pull or pull['state'] != 'open' or pull['draft'] or not any(
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
                verify_sweep(self.repository, latest, self.candidate.family)
        else:
            if pull and (pull['state'] != 'closed' or not pull['draft'] or any(
                    label['name'] in SWEEP_LABELS for label in pull['labels'])):
                raise VerificationError('Failure/deferral PR cleanup is incomplete')
            if pull and self.branch_exists() != (outcome.outcome not in RELEASE):
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
        if outcome.outcome != 'validated':
            if pull and not self.report(pull):
                self.refresh(pull)
                if not self.pending(pull):
                    pending = self.marker.replace('klaud-outcome:', 'klaud-cleanup:')
                    request = {'head': pull['head']['sha'], 'outcome': outcome.model_dump(by_alias=True)}
                    body = (pending + json.dumps(request) + '\n-->\n'
                            f'Klaud Cold: **{outcome.outcome}**. Finishing cleanup; owned child runs will be stopped and checked before closure.\n\n---\n\n'
                            f'Klaud Cold：**{outcome.outcome}**。正在完成清理；将先停止并确认自有子运行的状态，再关闭 PR。')
                    github.write(self.repository, f'issues/{pull["number"]}/comments', 'POST', {'body': body})
            for run in runs:
                if not terminal(run):
                    self.check_parent()
                    if pull:
                        self.refresh(pull)
                    github.write(self.repository, f'actions/runs/{run["id"]}/cancel', 'POST')
            if self.recovering:
                for run in self.runs():
                    if not terminal(run):
                        # Wait for cancellation without an extra agent or custom workflow timeout.
                        subprocess.run(['gh', 'run', 'watch', str(run['id']), '--repo', self.repository,
                                        '--interval', '15'], check=True, capture_output=True)
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
        # Label changes/closure can enqueue skipped runs. Never report completed cleanup before they finish.
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
                           'PR closed; the exact-candidate branch is retained for manual review. The interruption does not prove image incompatibility.')
                        + '\n\n---\n\n'
                        f'Klaud Cold：**{outcome.outcome}**。所有自有运行均已结束。'
                        f'修复次数：{outcome.repairs_used if outcome.repairs_used is not None else "未知"}。运行：{links}。\n\n'
                        + ('完整 sweep 已通过验证；PR 保持就绪，等待审查。' if proof else
                           'PR 已关闭；分支已删除，后续可以重试。' if outcome.outcome in RELEASE else
                           'PR 已关闭；保留该候选的分支，等待人工审查。运行中断不能证明镜像不兼容。'))
                github.write(self.repository, f'issues/{pull["number"]}/comments', 'POST', {'body': body})
        return outcome


def subprocess_ready(repository: str, number: int) -> None:
    subprocess.run(['gh', 'pr', 'ready', str(number), '--undo', '--repo', repository],
                   check=True, capture_output=True, timeout=60)


def current_session() -> Session:
    context = json.loads((Path(os.environ['KLAUD_EVIDENCE']) / 'candidate.json').read_text())
    candidate = OwnedCandidate.model_validate({key: context[key] for key in ('id', 'family', 'base')})
    repository = os.environ['GITHUB_REPOSITORY']
    parent = github.read(repository, f'actions/runs/{int(os.environ["GITHUB_RUN_ID"])}')
    return Session(repository, parent, candidate)


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
    if not pending and final and not terminal(max(final, key=lambda r: r['created_at'])):
        session.refresh(pull)
        latest = max(final, key=lambda r: r['created_at'])
        # A dispatched final sweep can still deliver complete evidence without an agent.
        subprocess.run(['gh', 'run', 'watch', str(latest['id']), '--repo', session.repository,
                        '--interval', '15'], check=True, capture_output=True)
        return reconcile(session)
    validated = bool(final and terminal(max(final, key=lambda r: r['created_at']))
                     and max(final, key=lambda r: r['created_at'])['conclusion'] == 'success')
    outcome = pending or CandidateOutcome(outcome='validated' if validated else 'unexpected-error',
                               phase='final-sweep' if validated else 'cleanup',
                               pull_request=pull['number'] if pull else None,
                               run_ids=[r['id'] for r in runs], repairs_used=None)
    # Unknown interruption retains the exact-candidate claim for manual review.
    while True:
        try:
            session.finish(outcome)
            break
        except PendingCleanup:
            session.check_parent()
            for run in session.runs():
                if not terminal(run):
                    subprocess.run(['gh', 'run', 'watch', str(run['id']), '--repo', session.repository,
                                    '--interval', '15'], check=True, capture_output=True)
    return True


def recover() -> None:
    repository = os.environ['GITHUB_REPOSITORY']
    inventory = github.items(repository, 'actions/artifacts?name=klaud-ownership&per_page=100', 'artifacts')
    seen = set()
    failed = False
    for artifact in sorted(inventory, key=lambda a: a['id'], reverse=True):
        if artifact['expired']:
            continue
        parent = github.read(repository, f'actions/runs/{artifact["workflow_run"]["id"]}')
        if (parent['head_branch'] != 'main'
                or parent['path'] != '.github/workflows/klaud-plan.yml'
                or parent['event'] not in ('schedule', 'workflow_dispatch')):
            continue
        with tempfile.TemporaryDirectory(prefix='klaud-ownership-') as temp:
            github.download_json(repository, artifact, Path(temp))
            ownership = Ownership.model_validate_json((Path(temp) / 'ownership.json').read_text())
        if ownership.run_id != parent['id']:
            raise VerificationError('Ownership parent mismatch')
        resolved = terminal(parent)
        for candidate in ownership.candidates:
            if candidate.id in seen:
                resolved = False
                continue
            seen.add(candidate.id)
            if not terminal(parent):
                continue
            session = Session(repository, parent, candidate, recovering=True)
            try:
                changed = reconcile(session)
            except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
                failed = True
                resolved = False
                reason = str(error) if isinstance(error, VerificationError) else 'state unavailable or invalid'
                status = f'needs inspection / 需要检查 ({reason})'
            else:
                status = 'reconciled / 已收尾' if changed else None
            if status and (summary := os.environ.get('GITHUB_STEP_SUMMARY')):
                with open(summary, 'a') as output:
                    output.write(f'Klaud `{candidate.id}`: {status}. '
                                 f'[Parent run / 父运行](https://github.com/{repository}/actions/runs/{parent["id"]})\n\n')
        if resolved:
            current = github.read(repository, f'actions/runs/{parent["id"]}')
            if terminal(current) and current.get('run_attempt') == parent.get('run_attempt'):
                # The public plan, outcome artifacts and PR reports remain. Retire only the
                # tiny pending-work receipt so future sweeps do not rescan completed history.
                github.write(repository, f'actions/artifacts/{artifact["id"]}', 'DELETE')
    if failed:
        raise VerificationError('Some owned sessions could not be reconciled; new dispatches withheld')
