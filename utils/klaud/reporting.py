"""Public, typed progress records and the single PR/comment renderer.

The agent supplies short observations; this module owns formatting and arithmetic.
Records are persisted in the same owned comment before waiting, so recovery does
not depend on an SDK transcript or final structured response.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import re
import unicodedata
from typing import Annotated, Literal

from pydantic import AfterValidator, Field, field_validator, model_validator

from . import github
from .github import VerificationError
from .models import Contract, identity

Number = Annotated[float, Field(ge=0, allow_inf_nan=False)]
SHA = Annotated[str, Field(pattern=r'^[0-9a-f]{40}$')]


def public_prose(value: str) -> str:
    if any(char in value for char in ('@', '<', '>', '\n', '|')):
        raise ValueError('Use one sentence of public prose without mentions, HTML or tables')
    if any(unicodedata.name(char, '').startswith(('CJK UNIFIED IDEOGRAPH', 'CJK COMPATIBILITY IDEOGRAPH')) for char in value):
        raise ValueError('Klaud PR bodies and comments must be English-only')
    return value


Prose = Annotated[str, Field(min_length=1, max_length=700, description='One concise English sentence'),
                  AfterValidator(public_prose)]


class Values(Contract):
    total_tps_gpu: Number | None = None
    output_tps_gpu: Number | None = None
    ttft_ms: Number | None = None
    tpot_ms: Number | None = None
    latency_statistic: Literal['mean', 'median'] = 'mean'
    request_errors: Annotated[int, Field(ge=0)] | None = None


class Point(Contract):
    # Hash of canonical generated settings excluding image/point name, including
    # concurrency. Dataset is separate because it is runtime evidence.
    key: str = Field(pattern=r'^[0-9a-f]{64}$')
    label: str = Field(pattern=r'^[A-Za-z0-9_.:+ /=-]{1,150}$')
    conc: int = Field(gt=0)
    scenario: Literal['fixed-seq-len', 'agentic-coding']
    dataset: str | None = Field(default=None, pattern=r'^[A-Za-z0-9_./:+-]{1,200}$')
    values: Values
    result: Literal['passed', 'failed', 'cancelled', 'pending', 'unavailable']
    run_id: Annotated[int, Field(gt=0)] | None = None
    head: SHA | None = None
    run_attempt: Annotated[int, Field(gt=0)] | None = None


class Evaluation(Contract):
    key: str = Field(pattern=r'^[0-9a-f]{64}$')
    suite: str = Field(pattern=r'^[A-Za-z0-9_.-]{1,100}$')
    metric: str = Field(pattern=r'^[A-Za-z0-9_,.-]{1,100}$')
    label: str = Field(default='', pattern=r'^[A-Za-z0-9_.:+ /=-]{0,150}$')
    score: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] | None = None
    samples: Annotated[int, Field(gt=0)] | None = None
    result: Literal['passed', 'failed', 'unavailable']
    run_id: Annotated[int, Field(gt=0)] | None = None
    head: SHA | None = None
    run_attempt: Annotated[int, Field(gt=0)] | None = None


class Baseline(Contract):
    family: str = Field(pattern=r'^configs/[^/:]+-master\.yaml:[^\s:]+$')
    date: str = Field(pattern=r'^\d{4}-\d{2}-\d{2}$')
    image: str = Field(pattern=r'^[A-Za-z0-9_./:@+-]+$')
    goal: Prose
    sources: list[str]
    points: list[Point]
    evals: list[Evaluation] = Field(default_factory=list)

    @field_validator('sources')
    @classmethod
    def public_sources(cls, urls: list[str]) -> list[str]:
        from urllib.parse import urlsplit
        for url in urls:
            parsed = urlsplit(url)
            if (parsed.scheme != 'https' or parsed.netloc != 'inferencex.semianalysis.com'
                    or not parsed.path.startswith('/api/') or any(c in url for c in '\n<>()')):
                raise ValueError('Baseline sources must be public dashboard API URLs')
        return urls

    @model_validator(mode='after')
    def distinct(self):
        unique_points(self.points)
        unique_evals(self.evals)
        return self


class Attempt(Contract):
    kind: Literal['initial', 'repair', 'infrastructure-retry', 'final']
    number: int = Field(ge=0)
    head: SHA
    image: str = Field(pattern=r'^[A-Za-z0-9_./:@+-]+$')
    run_id: int = Field(gt=0)
    run_attempt: int = Field(gt=0)
    status: Literal['queued', 'running', 'passed', 'failed', 'cancelled', 'deferred']
    change: Prose
    finding: Prose
    next: Prose
    benchmarks_expected: int = Field(ge=0)
    benchmarks_passed: int = Field(ge=0)
    evals_expected: int = Field(ge=0)
    evals_passed: int = Field(ge=0)
    points: list[Point]
    evals: list[Evaluation] = Field(default_factory=list)

    @model_validator(mode='after')
    def consistent(self):
        unique_points(self.points)
        unique_evals(self.evals)
        if self.benchmarks_passed > self.benchmarks_expected or self.evals_passed > self.evals_expected:
            raise ValueError('Passed coverage exceeds expected coverage')
        if self.kind == 'initial' and self.number != 0 or self.kind == 'repair' and not 1 <= self.number <= 5:
            raise ValueError('Initial update is separate from the five-repair budget')
        if any(p.run_id not in (None, self.run_id) or p.head not in (None, self.head)
               or p.run_attempt is not None and p.run_attempt > self.run_attempt for p in [*self.points, *self.evals]):
            raise ValueError('Attempt point provenance mismatch')
        return self


def unique_points(points: list[Point]) -> None:
    if len({point.key for point in points}) != len(points):
        raise ValueError('Duplicate comparison point')


def unique_evals(rows: list[Evaluation]) -> None:
    if len({(row.key, row.suite, row.metric) for row in rows}) != len(rows):
        raise ValueError('Duplicate comparison eval')


def point_key(entry: dict) -> str:
    return identity({key: value for key, value in entry.items()
                     if key not in ('image', 'exp-name', 'recipe-fingerprint', 'priority', 'queue-token')})


def point_label(entry: dict) -> str:
    if entry.get('prefill') is not None:
        shape = '/'.join(f'{role[0].upper()}TP{entry[role]["tp"]}x{entry[role]["num-worker"]}' for role in ('prefill', 'decode'))
    else:
        shape = f'TP{entry["tp"]} EP{entry.get("ep", 1)}'
    workload = f'{entry["isl"]}/{entry["osl"]}' if 'isl' in entry else 'AgentX'
    return f'{workload} c{entry["conc"]} {shape} {point_key(entry)[:6]}'


def values(row: dict) -> Values:
    """Collector/API metrics are seconds; only this presentation layer converts ms."""
    metrics = row.get('metrics', row)
    if 'request_metrics' in metrics:
        request = metrics['request_metrics']
        throughput = request['throughput']['per_gpu']
        ttft = request['latency']['ttft'].get('mean')
        tpot = request['latency']['tpot'].get('mean')
        return Values(total_tps_gpu=throughput.get('total_tput_tps'),
                      output_tps_gpu=throughput.get('output_tput_tps'),
                      ttft_ms=ttft * 1000 if ttft is not None else None,
                      tpot_ms=tpot * 1000 if tpot is not None else None,
                      request_errors=row.get('request_accounting', {}).get('records_error_dropped'))
    return Values(total_tps_gpu=metrics.get('tput_per_gpu'), output_tps_gpu=metrics.get('output_tput_per_gpu'),
                  ttft_ms=metrics['mean_ttft'] * 1000 if metrics.get('mean_ttft') is not None else None,
                  tpot_ms=metrics['mean_tpot'] * 1000 if metrics.get('mean_tpot') is not None else None,
                  request_errors=metrics.get('errors'))


def number(value: float | int | None) -> str:
    return 'N/A' if value is None else f'{value:,.2f}'.rstrip('0').rstrip('.')


def delta(old: float | None, new: float | None, comparable: bool = True) -> str:
    if not comparable or old is None or new is None or old == 0:
        return 'N/A'
    return f'{(new / old - 1) * 100:+.1f}%'


def table(header: list[str], rows: list[list[str]]) -> str:
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join('---' for _ in header) + ' |',
                      *('| ' + ' | '.join(row) + ' |' for row in rows)])


def point_table(points: list[Point], baseline: Baseline | None = None) -> str:
    old = {point.key: point for point in baseline.points} if baseline else {}
    rows = []
    for point in points:
        previous = old.get(point.key)
        comparable = bool(previous and previous.result == point.result == 'passed'
                          and previous.scenario == point.scenario
                          and (point.scenario != 'agentic-coding' or point.dataset and point.dataset == previous.dataset)
                          and previous.values.latency_statistic == point.values.latency_statistic)
        a, b = previous.values if previous else Values(), point.values
        reason = '' if comparable else ('; N/A: no matched baseline' if not previous else
                  '; N/A: point unavailable' if previous.result != 'passed' or point.result != 'passed' else
                  '; N/A: dataset/statistic mismatch')
        rows.append([point.label, f'{point.result}; errors={number(b.request_errors)}{reason}', number(b.output_tps_gpu),
                     delta(a.output_tps_gpu, b.output_tps_gpu, comparable),
                     delta(a.ttft_ms, b.ttft_ms, comparable), delta(a.tpot_ms, b.tpot_ms, comparable)])
    return table(['Point', 'Result', 'Output tok/s/GPU', 'Δ output', 'Δ TTFT', 'Δ TPOT'], rows)


def eval_table(rows: list[Evaluation], baseline: Baseline | None) -> str:
    previous = {(row.key, row.suite, row.metric): row for row in baseline.evals} if baseline else {}
    values = []
    for row in rows:
        old = previous.get((row.key, row.suite, row.metric))
        change = (f'{(row.score - old.score) * 100:+.2f}' if old and old.score is not None
                  and row.score is not None and old.result == row.result == 'passed'
                  and old.samples is not None and old.samples == row.samples else 'N/A')
        values.append([f'{row.label} {row.suite}/{row.metric}'.strip(), number(old.score * 100 if old and old.score is not None else None),
                       number(row.score * 100 if row.score is not None else None), change,
                       f'{old.samples if old and old.samples else "N/A"}/{row.samples or "N/A"}', row.result])
    return table(['Eval suite / metric', 'Baseline %', 'Updated %', 'Δ pp', 'n (old/new)', 'Result'], values)


def baseline_table(points: list[Point]) -> str:
    rows = [[p.label, number(p.values.total_tps_gpu), number(p.values.output_tps_gpu),
             number(p.values.ttft_ms), number(p.values.tpot_ms)] for p in points]
    return table(['Point', 'Total tok/s/GPU', 'Output tok/s/GPU', 'TTFT ms', 'TPOT ms'], rows)


def render_body(baseline: Baseline) -> str:
    statistic = '/'.join(sorted({point.values.latency_statistic for point in baseline.points})) or 'N/A'
    sources = ', '.join(f'[API {i + 1}]({url})' for i, url in enumerate(baseline.sources)) or 'N/A'
    return (f'{baseline.goal}\n\n**Baseline:** {baseline.date} · `{baseline.image}` · {sources}\n\n'
            + baseline_table(baseline.points[:12])
            + f'\n\n{min(12, len(baseline.points))}/{len(baseline.points)} baseline points shown. Latency: {statistic}. Missing values are N/A; all values and producer IDs are in the frozen baseline comments.\n\n'
            + ('Eval baseline: ' + '; '.join(f'{r.suite}/{r.metric} {number(r.score * 100 if r.score is not None else None)}%, n={r.samples or "N/A"}' for r in baseline.evals) if baseline.evals else 'Eval baseline: N/A (no matched published evidence).'))


def render_attempt(record: Attempt, baseline: Baseline | None, repository: str) -> str:
    titles = {'initial': 'Initial attempt', 'repair': f'Repair {record.number}/5',
              'infrastructure-retry': f'Infrastructure retry {record.number}', 'final': 'Final full sweep'}
    coverage = f'benchmarks {record.benchmarks_passed}/{record.benchmarks_expected}; evals {record.evals_passed}/{record.evals_expected}'
    details = point_table(record.points, baseline) + ('\n\n' + eval_table(record.evals, baseline) if record.evals else '')
    if len(record.points) > 12:
        details = '<details>\n<summary>All point results</summary>\n\n' + details + '\n\n</details>'
    return (f'### {titles[record.kind]}\n\n**Status:** {record.status} · {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}\n\n'
            f'**Measured:** `{record.image}` · `{record.head[:12]}` · '
            f'[run {record.run_id}](https://github.com/{repository}/actions/runs/{record.run_id}), attempt {record.run_attempt}\n\n'
            f'**Change:** {record.change}\n\n**Coverage:** {coverage}.\n\n{details}\n\n'
            'Throughput: higher is better; latency: lower is better. N/A means missing/zero baseline, '
            'failed point, or unproven matching identity, dataset or statistic. The run attempt may retain earlier successful jobs from this same run; absent per-point attempt provenance stays unknown. No significance claim.\n\n'
            f'**Finding:** {record.finding}\n\n**Next:** {record.next}')


def marker(session, name: str) -> str:
    return f'<!-- klaud-report:{session.parent["id"]}:{session.candidate.id}:{name}\n'


def stored(session, pull: dict, name: str) -> dict | None:
    matches = [c for c in github.items(session.repository, f'issues/{pull["number"]}/comments?per_page=100')
               if c['user']['login'] == 'Klaud-Cold' and c['body'].startswith(marker(session, name))]
    if len(matches) > 1:
        raise VerificationError('Ambiguous report comment')
    return matches[0] if matches else None


def decode(comment: dict, model, session=None, pull=None):
    value = json.loads(comment['body'].split('\n', 1)[1].split('\n-->', 1)[0])
    if 'record' in value:
        record = value['record']
        for name in value['chunks']:
            chunk = stored(session, pull, name)
            if not chunk:
                raise VerificationError('Incomplete persisted report')
            data = json.loads(chunk['body'].split('\n', 1)[1].split('\n-->', 1)[0])
            for key in ('points', 'evals'):
                record[key].extend(data[key])
        value = record
    return model.model_validate(value)


def baseline_for(session, pull: dict) -> Baseline | None:
    comment = stored(session, pull, 'baseline')
    return decode(comment, Baseline, session, pull) if comment else None


def initialize_body(session, pull: dict, record: Baseline) -> None:
    current = session.refresh(pull)
    body = current.get('body') or ''
    completed = '<!-- klaud-baseline-body -->'
    placeholder = '<!-- klaud-baseline -->'
    if completed in body:
        return
    if body.count(placeholder) != 1:
        raise VerificationError('Draft body needs one <!-- klaud-baseline --> placeholder; preserve other bots\' blocks')
    body = body.replace(placeholder, completed + '\n' + render_body(record) + '\n<!-- /klaud-baseline-body -->')
    github.write(session.repository, f'pulls/{pull["number"]}', 'PATCH', {'body': body})


def publish(session, record: Baseline | Attempt) -> None:
    pull = session.pulls()[0]
    session.refresh(pull)
    if isinstance(record, Baseline):
        if record.family != session.candidate.family:
            raise VerificationError('Baseline belongs to another family')
        name = 'baseline'
        previous = stored(session, pull, name)
        if previous:
            if decode(previous, Baseline, session, pull) != record:
                raise VerificationError('Baseline is frozen; do not silently replace it')
            initialize_body(session, pull, record)
            return
        text = 'Frozen public baseline; producer references and numeric evidence are recorded here.'
    else:
        runs = [run for run in session.runs() if run['id'] == record.run_id]
        if len(runs) != 1 or runs[0]['head_sha'] != record.head or runs[0]['run_attempt'] != record.run_attempt:
            raise VerificationError('Report does not describe an owned run at this head and attempt')
        if record.status == 'passed' and (runs[0]['status'] != 'completed' or runs[0]['conclusion'] != 'success'):
            raise VerificationError('Cannot publish a passing attempt before the entire owned run passes')
        name = f'run-{record.run_id}-{record.run_attempt}'
        previous = stored(session, pull, name)
        text = render_attempt(record, baseline_for(session, pull), session.repository)
    # Bounded comment chunks preserve every point without imposing a family-size cap.
    # Write the index last: interrupted publication is retried idempotently.
    packed = record.model_dump(by_alias=True)
    chunks = []
    if len(record.points) > 20 or len(record.evals) > 20:
        for offset in range(0, max(len(record.points), len(record.evals)), 20):
            part = {'points': packed['points'][offset:offset + 20], 'evals': packed['evals'][offset:offset + 20]}
            # Immutable content-addressed parts keep an existing index consistent
            # until the replacement index is published, even across interruptions.
            part_name = f'{name}-part-{offset // 20 + 1}-{identity(part)[:16]}'
            part_text = (baseline_table(record.points[offset:offset + 20]) if isinstance(record, Baseline) else
                         point_table(record.points[offset:offset + 20], baseline_for(session, pull)))
            if part['evals']:
                part_text += '\n\n' + eval_table(record.evals[offset:offset + 20], record if isinstance(record, Baseline) else baseline_for(session, pull))
            upsert(session, pull, part_name, json.dumps(part), part_text)
            chunks.append(part_name)
        packed.update(points=[], evals=[])
        packed = {'record': packed, 'chunks': chunks}
        if isinstance(record, Attempt):
            text = render_attempt(record.model_copy(update={'points': [], 'evals': []}), baseline_for(session, pull), session.repository)
        text += '\n\nFull numeric evidence follows in numbered report parts.'
    elif isinstance(record, Baseline):
        text += '\n\n' + baseline_table(record.points)
        if record.evals:
            text += '\n\n' + eval_table(record.evals, record)
    upsert(session, pull, name, json.dumps(packed), text)
    if isinstance(record, Baseline):
        initialize_body(session, pull, record)


def upsert(session, pull: dict, name: str, data: str, text: str) -> None:
    previous = stored(session, pull, name)
    # Only the typed allowlist enters the durable record, never raw API/log data.
    body = marker(session, name) + data + '\n-->\n' + text
    if len(body) > 60000:
        raise VerificationError('Report exceeds GitHub comment size; split the point evidence before publication')
    session.refresh(pull)
    path = f'issues/comments/{previous["id"]}' if previous else f'issues/{pull["number"]}/comments'
    github.write(session.repository, path, 'PATCH' if previous else 'POST', {'body': body})


def publish_final(session, run: dict, evidence: tuple[dict, list[dict], list[dict]]) -> None:
    """Normal finish and recovery publish the same artifact-derived final report."""
    from utils import validate_reusable_sweep_artifacts as reuse
    from .validation import benchmark_entries, expected_evals

    matrix, rows, eval_rows = evidence
    generated = {(entry['recipe-fingerprint'], int(conc)): {**entry, 'conc': int(conc)}
                 for entry in benchmark_entries(matrix)
                 for conc in (entry['conc'] if isinstance(entry['conc'], list) else [entry['conc']])}
    points = []
    for row in rows:
        entry = generated[(row['recipe_fingerprint'], int(row['conc']))]
        points.append(Point(key=point_key(entry), label=point_label(entry), conc=int(row['conc']),
                            scenario=entry.get('scenario-type', 'fixed-seq-len'),
                            dataset=(row.get('dataset') or {}).get('loader'), values=values(row), result='passed',
                            run_id=run['id'], head=run['head_sha']))
    evaluations = [Evaluation(key=identity(reuse.eval_key(row)), suite=row.get('eval_suite') or 'gsm8k', label=f'c{row["conc"]}',
                              metric=row.get('score_name') or 'em_strict', score=row.get('score', row.get('em_strict')),
                              samples=row.get('n_eff'), result='passed', run_id=run['id'], head=run['head_sha'])
                   for row in sorted(eval_rows, key=lambda row: (int(row['conc']), row.get('eval_suite', '')))]
    images = {row['image'] for row in rows}
    if len(images) != 1:
        raise VerificationError('Final report contains mixed images')
    publish(session, Attempt(kind='final', number=0, head=run['head_sha'], image=images.pop(),
            run_id=run['id'], run_attempt=run['run_attempt'], status='passed',
            change='Validate the complete updated-image family at this head.',
            finding='Complete benchmark coverage and required default eval artifacts verified. Measured results and deltas are shown above.',
            next='Finish will mark this completed PR ready for maintainer review; regressions remain visible.',
            benchmarks_expected=len(generated), benchmarks_passed=len(points),
            evals_expected=len(expected_evals(matrix)), evals_passed=len(expected_evals(matrix)),
            points=sorted(points, key=lambda point: (point.conc, point.label)), evals=evaluations))


def prepare_baseline(session, context: dict, model: str, goal: Prose) -> Baseline:
    """Fetch once, accepting only exact producer fingerprints of the selected old family.

    Legacy public rows without full fingerprints remain unavailable rather than
    matching merely by SKU/concurrency. Raw responses never enter the public record.
    """
    from .api import fetch
    from .validation import canonical_matrix, benchmark_entries

    matrix = canonical_matrix(session.repository, session.candidate.base, session.candidate.family)
    feed = fetch('benchmarks', model=model, date=context['source']['date'])
    info = fetch('workflow-info', date=context['source']['date'])
    # Public database bigint IDs are serialized as strings; URLs use decimal IDs.
    producers = {int(row['github_run_id']): row for row in info.payload['runs']}
    heads = {}
    for row in info.payload['runConfigs']:
        if row.get('head_sha'):
            heads.setdefault(int(row['github_run_id']), set()).add(row['head_sha'])
    old_image = context['source']['image']
    points = []
    for entry in benchmark_entries(matrix):
        if entry['image'] != old_image:
            raise VerificationError('Baseline source image no longer matches the selected base')
        for conc in (entry['conc'] if isinstance(entry['conc'], list) else [entry['conc']]):
            point = {**entry, 'conc': conc}
            matched = [row for row in feed.payload if row.get('recipe_fingerprint') == entry['recipe-fingerprint']
                       and row.get('image') == old_image and row.get('conc') == conc]
            published = matched[0] if len(matched) == 1 else None
            producer = re.fullmatch(r'https://github.com/' + re.escape(session.repository) + r'/actions/runs/(\d+)(?:/attempts/(\d+))?',
                                    (published or {}).get('run_url') or '')
            run_id = int(producer[1]) if producer else None
            run_attempt = int(producer[2]) if producer and producer[2] else None
            if (run_id not in producers or len(heads.get(run_id, ())) != 1
                    or run_attempt is not None and run_attempt > int(producers[run_id]['run_attempt'])):
                published = None
            points.append(Point(key=point_key(point), label=point_label(point), conc=int(conc),
                                scenario=entry.get('scenario-type', 'fixed-seq-len'),
                                # Dataset is not included in BenchmarkRow. Until a public
                                # producer dataset match is established, AgentX deltas are N/A.
                                values=values(published) if published else Values(),
                                result='passed' if published else 'unavailable',
                                run_id=run_id if published else None, head=next(iter(heads[run_id])) if published else None,
                                run_attempt=run_attempt if published else None))
    return Baseline(family=session.candidate.family, date=context['source']['date'], image=old_image,
                    goal=goal, sources=[feed.url, info.url], points=points)
