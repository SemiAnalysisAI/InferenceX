"""Public, typed progress records and the single PR/comment renderer.

The agent supplies short observations; this module owns formatting and arithmetic.
Records are persisted in the same owned comment before waiting, so recovery does
not depend on an SDK transcript or final structured response.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import AfterValidator, Field, field_validator, model_validator

from . import github
from .github import VerificationError
from .models import Contract, identity

if TYPE_CHECKING:
    from infx.klaud.lifecycle import Session


Number = Annotated[float, Field(ge=0, allow_inf_nan=False)]
SHA = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]


def public_prose(value: str) -> str:
    # Digest-pinned images are inline code, not GitHub mentions.
    prose = re.sub(r"`[A-Za-z0-9_./:+-]+@sha256:[0-9a-f]{64}`", "", value)
    if any(char in prose for char in ("@", "<", ">", "\n", "|")):
        raise ValueError("Use one sentence of public prose without mentions, HTML or tables")
    return value


Text = Annotated[str, Field(min_length=1, max_length=700), AfterValidator(public_prose)]


class Prose(Contract):
    en: Text = Field(description="One concise English sentence")
    zh: Text = Field(description="Natural Simplified Chinese translation; preserve identifiers")


class Values(Contract):
    total_tps_gpu: Number | None = None
    output_tps_gpu: Number | None = None
    ttft_ms: Number | None = None
    tpot_ms: Number | None = None
    latency_statistic: Literal["mean", "median"] = "mean"
    request_errors: Annotated[int, Field(ge=0)] | None = None


class Point(Contract):
    # Hash of canonical generated settings excluding image/point name, including
    # concurrency. Dataset is separate because it is runtime evidence.
    key: str = Field(pattern=r"^[0-9a-f]{64}$")
    label: str = Field(pattern=r"^[A-Za-z0-9_.:+ /=-]{1,150}$")
    conc: int = Field(gt=0)
    scenario: Literal["fixed-seq-len", "agentic-coding"]
    dataset: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_./:+-]{1,200}$")
    values: Values
    result: Literal["passed", "failed", "cancelled", "pending", "unavailable"]
    run_id: Annotated[int, Field(gt=0)] | None = None
    head: SHA | None = None
    run_attempt: Annotated[int, Field(gt=0)] | None = None


class Evaluation(Contract):
    key: str = Field(pattern=r"^[0-9a-f]{64}$")
    suite: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,100}$")
    metric: str = Field(pattern=r"^[A-Za-z0-9_,.-]{1,100}$")
    label: str = Field(default="", pattern=r"^[A-Za-z0-9_.:+ /=-]{0,150}$")
    score: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] | None = None
    samples: Annotated[int, Field(gt=0)] | None = None
    result: Literal["passed", "failed", "unavailable"]
    run_id: Annotated[int, Field(gt=0)] | None = None
    head: SHA | None = None
    run_attempt: Annotated[int, Field(gt=0)] | None = None


class Baseline(Contract):
    family: str = Field(pattern=r"^configs/[^/:]+-master\.yaml:[^\s:]+$")
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    image: str = Field(pattern=r"^[A-Za-z0-9_./:@+-]+$")
    goal: Prose
    sources: list[str]
    points: list[Point]
    evals: list[Evaluation] = Field(default_factory=list)

    @field_validator("sources")
    @classmethod
    def public_sources(cls, urls: list[str]) -> list[str]:
        from urllib.parse import urlsplit

        for url in urls:
            parsed = urlsplit(url)
            if (
                parsed.scheme != "https"
                or parsed.netloc != "inferencex.semianalysis.com"
                or not parsed.path.startswith("/api/")
                or any(c in url for c in "\n<>()")
            ):
                raise ValueError("Baseline sources must be public dashboard API URLs")
        return urls

    @model_validator(mode="after")
    def distinct(self) -> Self:
        unique_points(self.points)
        unique_evals(self.evals)
        return self


class Attempt(Contract):
    kind: Literal["initial", "repair", "infrastructure-retry", "final"]
    number: int = Field(ge=0)
    head: SHA
    image: str = Field(pattern=r"^[A-Za-z0-9_./:@+-]+$")
    run_id: int = Field(gt=0)
    run_attempt: int = Field(gt=0)
    status: Literal["queued", "running", "passed", "failed", "cancelled", "deferred"]
    change: Prose
    finding: Text | None = None  # Optional diagnostic evidence, not a visible summary paragraph.
    next: Prose
    benchmarks_expected: int = Field(ge=0)
    benchmarks_passed: int = Field(ge=0)
    evals_expected: int = Field(ge=0)
    evals_passed: int = Field(ge=0)
    points: list[Point]
    evals: list[Evaluation] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        unique_points(self.points)
        unique_evals(self.evals)
        if (
            self.benchmarks_passed > self.benchmarks_expected
            or self.evals_passed > self.evals_expected
        ):
            raise ValueError("Passed coverage exceeds expected coverage")
        if (self.kind == "initial" and self.number != 0) or (
            self.kind == "repair" and not 1 <= self.number <= 5
        ):
            raise ValueError("Initial update is separate from the five-repair budget")
        if any(
            p.run_id not in (None, self.run_id)
            or p.head not in (None, self.head)
            or (p.run_attempt is not None and p.run_attempt > self.run_attempt)
            for p in [*self.points, *self.evals]
        ):
            raise ValueError("Attempt point provenance mismatch")
        return self


def unique_points(points: list[Point]) -> None:
    if len({point.key for point in points}) != len(points):
        raise ValueError("Duplicate comparison point")


def unique_evals(rows: list[Evaluation]) -> None:
    if len({(row.key, row.suite, row.metric) for row in rows}) != len(rows):
        raise ValueError("Duplicate comparison eval")


def point_key(entry: dict) -> str:
    return identity(
        {
            key: value
            for key, value in entry.items()
            if key
            not in (
                "image",
                "exp-name",
                "recipe-fingerprint",
                "priority",
                "queue-token",
            )
        }
    )


def point_label(entry: dict) -> str:
    if entry.get("prefill") is not None:
        shape = "/".join(
            f"{role[0].upper()}TP{entry[role]['tp']}x{entry[role]['num-worker']}"
            for role in ("prefill", "decode")
        )
    else:
        shape = f"TP{entry['tp']} EP{entry.get('ep', 1)}"
    workload = f"{entry['isl']}/{entry['osl']}" if "isl" in entry else "AgentX"
    return f"{workload} c{entry['conc']} {shape} {point_key(entry)[:6]}"


def values(row: dict) -> Values:
    """Collector/API metrics are seconds; only this presentation layer converts ms."""
    metrics = row.get("metrics", row)
    if "request_metrics" in metrics:
        request = metrics["request_metrics"]
        throughput = request["throughput"]["per_gpu"]
        ttft = request["latency"]["ttft"].get("mean")
        tpot = request["latency"]["tpot"].get("mean")
        return Values(
            total_tps_gpu=throughput.get("total_tput_tps"),
            output_tps_gpu=throughput.get("output_tput_tps"),
            ttft_ms=ttft * 1000 if ttft is not None else None,
            tpot_ms=tpot * 1000 if tpot is not None else None,
            request_errors=row.get("request_accounting", {}).get("records_error_dropped"),
        )
    return Values(
        total_tps_gpu=metrics.get("tput_per_gpu"),
        output_tps_gpu=metrics.get("output_tput_per_gpu"),
        ttft_ms=metrics["mean_ttft"] * 1000 if metrics.get("mean_ttft") is not None else None,
        tpot_ms=metrics["mean_tpot"] * 1000 if metrics.get("mean_tpot") is not None else None,
        request_errors=metrics.get("errors"),
    )


def number(value: float | None) -> str:
    return "N/A" if value is None else f"{value:,.2f}".rstrip("0").rstrip(".")


def delta(old: float | None, new: float | None, comparable: bool = True) -> str:
    if not comparable or old is None or new is None or old == 0:
        return "N/A"
    return f"{(new / old - 1) * 100:+.1f}%".replace(".0%", "%")


def translated(english: str, chinese: str) -> str:
    return english + "\n\n<details>\n<summary>中文</summary>\n\n" + chinese + "\n\n</details>"


def short_label(label: str) -> str:
    def shape(match: re.Match) -> str:
        return "/".join(
            f"{int(n) // 1024}k" if int(n) > 0 and int(n) % 1024 == 0 else n for n in match.groups()
        )

    return re.sub(r"\b(\d+)/(\d+)\b", shape, label, count=1)


def point_layout(points: list[Point]) -> tuple[str, list[str], str]:
    """Factor out only shared settings; retain full labels for ambiguous concurrencies."""
    labels = [short_label(p.label) for p in points]
    shapes = [
        re.fullmatch(r"(.+) c\d+ (.+)", re.sub(r" [0-9a-f]{6}$", "", label)) for label in labels
    ]
    common = {match.groups() for match in shapes if match}
    statistics = {p.values.latency_statistic for p in points}
    latency = next(iter(statistics)).capitalize() + " latency" if len(statistics) == 1 else ""
    if (
        points
        and all(shapes)
        and len(common) == 1
        and len({p.conc for p in points}) == len(points)
        and latency
    ):
        workload, topology = common.pop()
        return (
            "Concurrency",
            [str(p.conc) for p in points],
            f"{workload} · {topology.replace(' EP', '/EP')} · {latency}",
        )
    if len(statistics) > 1:
        labels = [
            f"{label} · {p.values.latency_statistic}"
            for label, p in zip(labels, points, strict=False)
        ]
    return "Point", labels, latency


def table(header: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return ""
    alignment = [
        "---:" if header[0] == "Concurrency" else "---",
        *["---:"] * (len(header) - 1),
    ]
    return "\n".join(
        [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(alignment) + " |",
            *("| " + " | ".join(row) + " |" for row in rows),
        ]
    )


def note_lines(issues: dict[str, list[str]], total: int) -> str:
    return "  \n".join(
        f"**Note:** {'All rows' if len(labels) == total else ', '.join(labels)}: {reason}."
        for reason, labels in issues.items()
    )


def measured(new: float | None, old: float | None, comparable: bool) -> str:
    return "N/A" if new is None else f"{number(new)} ({delta(old, new, comparable)})"


def point_table(
    points: list[Point], baseline: Baseline | None = None, *, context: bool = True
) -> str:
    old = {point.key: point for point in baseline.points} if baseline else {}
    heading, labels, settings = point_layout(points)
    rows, issues = [], {}
    for point, label in zip(points, labels, strict=False):
        previous = old.get(point.key)
        comparable = bool(
            previous
            and previous.result == point.result == "passed"
            and previous.scenario == point.scenario
            and (
                point.scenario != "agentic-coding"
                or (point.dataset and point.dataset == previous.dataset)
            )
            and previous.values.latency_statistic == point.values.latency_statistic
        )
        a, b = previous.values if previous else Values(), point.values
        rows.append(
            [
                label,
                measured(b.output_tps_gpu, a.output_tps_gpu, comparable),
                measured(b.ttft_ms, a.ttft_ms, comparable),
                measured(b.tpot_ms, a.tpot_ms, comparable),
            ]
        )
        reasons = []
        if point.result != "passed":
            reasons.append(point.result)
        if b.request_errors:
            reasons.append(
                f"{number(b.request_errors)} request error" + ("s" if b.request_errors != 1 else "")
            )
        elif b.request_errors is None:
            reasons.append("request errors unavailable")
        if not comparable:
            reasons.append(
                "Δ N/A: no matched baseline"
                if not previous
                else "Δ N/A: point unavailable"
                if previous.result != "passed" or point.result != "passed"
                else "Δ N/A: dataset/scenario/statistic mismatch"
            )
        elif any(
            new is not None and (old is None or old == 0)
            for old, new in (
                (a.output_tps_gpu, b.output_tps_gpu),
                (a.ttft_ms, b.ttft_ms),
                (a.tpot_ms, b.tpot_ms),
            )
        ):
            reasons.append("Δ N/A where baseline is missing or zero")
        for reason in reasons:
            issues.setdefault(reason, []).append(f"c{label}" if heading == "Concurrency" else label)
    return "\n\n".join(
        part
        for part in (
            settings if context else "",
            table([heading, "Output tok/s/GPU ↑", "TTFT ms ↓", "TPOT ms ↓"], rows),
            note_lines(issues, len(points)),
        )
        if part
    )


def eval_table(rows: list[Evaluation], baseline: Baseline | None, *, compare: bool = True) -> str:
    previous = {(row.key, row.suite, row.metric): row for row in baseline.evals} if baseline else {}
    values, issues = [], {}
    for row in rows:
        old = previous.get((row.key, row.suite, row.metric))
        comparable = bool(
            old
            and old.score is not None
            and row.score is not None
            and old.result == row.result == "passed"
            and old.samples is not None
            and old.samples == row.samples
        )
        change = f"{(row.score - old.score) * 100:+.2f} pp" if comparable else "N/A"
        score = f"{number(row.score * 100)}%" if row.score is not None else "N/A"
        if compare and row.score is not None:
            score += f" ({change})"
        samples = number(row.samples)
        if compare:
            samples = (
                f"{samples} each"
                if old and row.samples is not None and old.samples == row.samples
                else f"{number(old.samples if old else None)}/{samples} (old/new)"
            )
        label = f"{row.suite}/{row.metric}" + (f" · {short_label(row.label)}" if row.label else "")
        values.append([label, score, samples])
        if row.result != "passed":
            issues.setdefault(row.result, []).append(label)
        if compare and not comparable:
            reason = (
                "Δ N/A: no matched eval baseline"
                if not old
                else "Δ N/A: eval score unavailable"
                if old.score is None or row.score is None
                else "Δ N/A: eval unavailable"
                if old.result != "passed" or row.result != "passed"
                else "Δ N/A: sample counts missing or different"
            )
            issues.setdefault(reason, []).append(label)
    return "\n\n".join(
        part
        for part in (
            table(["Eval", "Score ↑", "Samples"], values),
            note_lines(issues, len(rows)),
        )
        if part
    )


def baseline_table(points: list[Point], *, context: bool = True) -> str:
    heading, labels, settings = point_layout(points)
    rows = [
        [
            label,
            number(p.values.total_tps_gpu),
            number(p.values.output_tps_gpu),
            number(p.values.ttft_ms),
            number(p.values.tpot_ms),
        ]
        for label, p in zip(labels, points, strict=False)
    ]
    issues = {}
    for label, point in zip(labels, points, strict=False):
        if point.result != "passed":
            issues.setdefault(point.result, []).append(
                f"c{label}" if heading == "Concurrency" else label
            )
        if point.values.request_errors:
            reason = f"{number(point.values.request_errors)} request error" + (
                "s" if point.values.request_errors != 1 else ""
            )
            issues.setdefault(reason, []).append(label)
    return "\n\n".join(
        part
        for part in (
            settings if context else "",
            table(
                [
                    heading,
                    "Total tok/s/GPU ↑",
                    "Output tok/s/GPU ↑",
                    "TTFT ms ↓",
                    "TPOT ms ↓",
                ],
                rows,
            ),
            note_lines(issues, len(points)),
        )
        if part
    )


def render_body(baseline: Baseline) -> str:
    _, _, settings = point_layout(baseline.points[:12])
    sources = ", ".join(f"[API {i + 1}]({url})" for i, url in enumerate(baseline.sources)) or "N/A"
    meta = " · ".join(part for part in (settings, f"Sources: {sources}") if part)
    english = (
        f"**Goal:** {baseline.goal.en}  \n**Baseline:** {baseline.date} · `{baseline.image}`  \n{meta}\n\n"
        + baseline_table(baseline.points[:12], context=False)
        + "\n\n"
        + (eval_table(baseline.evals, None, compare=False) if baseline.evals else "**Eval:** N/A")
    )
    if len(baseline.points) > 12:
        english += f"\n\n12/{len(baseline.points)} points shown; remaining rows are in the baseline report."
    chinese = (
        f"**目标：**{baseline.goal.zh}  \n**基线：**{baseline.date} · `{baseline.image}`  \n"
        + meta.replace("Mean latency", "平均延迟")
        .replace("Median latency", "中位延迟")
        .replace("Sources:", "来源：")
        + "；数值及异常说明见上表。"
    )
    return translated(english, chinese)


def render_attempt(record: Attempt, baseline: Baseline | None, repository: str) -> str:
    titles = {
        "initial": ("Initial attempt", "初次尝试"),
        "repair": (f"Repair {record.number}/5", f"修复 {record.number}/5"),
        "infrastructure-retry": (
            f"Infrastructure retry {record.number}",
            f"基础设施重试 {record.number}",
        ),
        "final": ("Final full sweep", "最终完整 sweep"),
    }
    status_zh = {
        "queued": "排队中",
        "running": "运行中",
        "passed": "已通过",
        "failed": "失败",
        "cancelled": "已取消",
        "deferred": "已延期",
    }[record.status]
    title, title_zh = titles[record.kind]
    timestamp = f"{datetime.now(UTC):%Y-%m-%d %H:%M UTC}"
    link = f"[Run {record.run_id} / attempt {record.run_attempt}](https://github.com/{repository}/actions/runs/{record.run_id}/attempts/{record.run_attempt})"
    _, _, settings = point_layout(record.points)
    meta = f"`{record.image}` · `{record.head[:12]}`" + (f" · {settings}" if settings else "")
    results = "\n\n".join(
        part
        for part in (
            point_table(record.points, baseline, context=False),
            eval_table(record.evals, baseline),
        )
        if part
    )
    english = (
        f"**{title} · {record.status.capitalize()}** · {link} · {timestamp}  \n"
        f"{meta}  \n**Change:** {record.change.en}\n\n"
        + (results + "\n\n" if results else "")
        + f"**Next:** {record.next.en}"
    )
    chinese = (
        f"**{title_zh} · {status_zh}** · {link} · {timestamp}  \n"
        + meta.replace("Mean latency", "平均延迟").replace("Median latency", "中位延迟")
        + f"  \n**变更：**{record.change.zh}实测数值及异常说明见上表。  \n**下一步：**{record.next.zh}"
    )
    return translated(english, chinese)


def marker(session: Session, name: str) -> str:
    return f"<!-- klaud-report:{session.parent['id']}:{session.candidate.id}:{name}\n"


def stored(session: Session, pull: dict, name: str) -> dict | None:
    matches = [
        c
        for c in github.items(session.repository, f"issues/{pull['number']}/comments?per_page=100")
        if c["user"]["login"] == "Klaud-Cold" and c["body"].startswith(marker(session, name))
    ]
    if len(matches) > 1:
        raise VerificationError("Ambiguous report comment")
    return matches[0] if matches else None


def decode[Record: Contract](
    comment: dict,
    model: type[Record],
    session: Session | None = None,
    pull: dict | None = None,
) -> Record:
    value = json.loads(comment["body"].split("\n", 1)[1].split("\n-->", 1)[0])
    if "record" in value:
        record = value["record"]
        for name in value["chunks"]:
            chunk = stored(session, pull, name)
            if not chunk:
                raise VerificationError("Incomplete persisted report")
            data = json.loads(chunk["body"].split("\n", 1)[1].split("\n-->", 1)[0])
            for key in ("points", "evals"):
                record[key].extend(data[key])
        value = record
    return model.model_validate(value)


def baseline_for(session: Session, pull: dict) -> Baseline | None:
    comment = stored(session, pull, "baseline")
    return decode(comment, Baseline, session, pull) if comment else None


def initialize_body(session: Session, pull: dict, record: Baseline) -> None:
    current = session.refresh(pull)
    body = current.get("body") or ""
    completed = "<!-- klaud-baseline-body -->"
    placeholder = "<!-- klaud-baseline -->"
    if completed in body:
        return
    if body.count(placeholder) != 1:
        raise VerificationError(
            "Draft body needs one <!-- klaud-baseline --> placeholder; preserve other bots' blocks"
        )
    body = body.replace(
        placeholder,
        completed + "\n" + render_body(record) + "\n<!-- /klaud-baseline-body -->",
    )
    github.write(session.repository, f"pulls/{pull['number']}", "PATCH", {"body": body})


def publish(session: Session, record: Baseline | Attempt) -> None:
    pull = session.pulls()[0]
    session.refresh(pull)
    if isinstance(record, Baseline):
        if record.family != session.candidate.family:
            raise VerificationError("Baseline belongs to another family")
        name = "baseline"
        previous = stored(session, pull, name)
        if previous:
            if decode(previous, Baseline, session, pull) != record:
                raise VerificationError("Baseline is frozen; do not silently replace it")
            initialize_body(session, pull, record)
            return
        text = f"**Baseline:** {record.date} · `{record.image}`"
    else:
        runs = [run for run in session.runs() if run["id"] == record.run_id]
        if (
            len(runs) != 1
            or runs[0]["head_sha"] != record.head
            or runs[0]["run_attempt"] != record.run_attempt
        ):
            raise VerificationError(
                "Report does not describe an owned run at this head and attempt"
            )
        if record.status == "passed" and (
            runs[0]["status"] != "completed" or runs[0]["conclusion"] != "success"
        ):
            raise VerificationError(
                "Cannot publish a passing attempt before the entire owned run passes"
            )
        name = f"run-{record.run_id}-{record.run_attempt}"
        previous = stored(session, pull, name)
        text = render_attempt(record, baseline_for(session, pull), session.repository)
    # Bounded comment chunks preserve every point without imposing a family-size cap.
    # Write the index last: interrupted publication is retried idempotently.
    packed = record.model_dump(by_alias=True)
    chunks = []
    if len(record.points) > 20 or len(record.evals) > 20:
        for offset in range(0, max(len(record.points), len(record.evals)), 20):
            part = {
                "points": packed["points"][offset : offset + 20],
                "evals": packed["evals"][offset : offset + 20],
            }
            # Immutable content-addressed parts keep an existing index consistent
            # until the replacement index is published, even across interruptions.
            part_name = f"{name}-part-{offset // 20 + 1}-{identity(part)[:16]}"
            part_text = (
                baseline_table(record.points[offset : offset + 20])
                if isinstance(record, Baseline)
                else point_table(record.points[offset : offset + 20], baseline_for(session, pull))
            )
            if part["evals"]:
                part_text += "\n\n" + eval_table(
                    record.evals[offset : offset + 20],
                    baseline_for(session, pull),
                    compare=not isinstance(record, Baseline),
                )
            upsert(session, pull, part_name, json.dumps(part), part_text)
            chunks.append(part_name)
        packed.update(points=[], evals=[])
        packed = {"record": packed, "chunks": chunks}
        if isinstance(record, Attempt):
            text = render_attempt(
                record.model_copy(update={"points": [], "evals": []}),
                baseline_for(session, pull),
                session.repository,
            )
        note = "\n\nResults continue in numbered report comments."
        text = (
            text.replace("\n\n<details>", note + "\n\n<details>", 1)
            if "<details>" in text
            else text + note
        )
    elif isinstance(record, Baseline):
        text += "\n\n" + baseline_table(record.points)
        if record.evals:
            text += "\n\n" + eval_table(record.evals, None, compare=False)
    if isinstance(record, Baseline):
        text = translated(
            text, f"**基线：**{record.date} · `{record.image}`；数值及异常说明见表格。"
        )
    upsert(session, pull, name, json.dumps(packed), text)
    if isinstance(record, Baseline):
        initialize_body(session, pull, record)


def upsert(session: Session, pull: dict, name: str, data: str, text: str) -> None:
    previous = stored(session, pull, name)
    # Only the typed allowlist enters the durable record, never raw API/log data.
    body = marker(session, name) + data + "\n-->\n" + text
    if len(body) > 60000:
        raise VerificationError(
            "Report exceeds GitHub comment size; split the point evidence before publication"
        )
    session.refresh(pull)
    path = f"issues/comments/{previous['id']}" if previous else f"issues/{pull['number']}/comments"
    github.write(session.repository, path, "PATCH" if previous else "POST", {"body": body})


def publish_final(
    session: Session, run: dict, evidence: tuple[dict, list[dict], list[dict]]
) -> None:
    """Normal finish and recovery publish the same artifact-derived final report."""
    from infx.workflows import validate_reusable_sweep_artifacts as reuse

    from .validation import benchmark_entries, expected_evals

    matrix, rows, eval_rows = evidence
    generated = {
        (entry["recipe-fingerprint"], int(conc)): {**entry, "conc": int(conc)}
        for entry in benchmark_entries(matrix)
        for conc in (entry["conc"] if isinstance(entry["conc"], list) else [entry["conc"]])
    }
    points = []
    for row in rows:
        entry = generated[(row["recipe_fingerprint"], int(row["conc"]))]
        points.append(
            Point(
                key=point_key(entry),
                label=point_label(entry),
                conc=int(row["conc"]),
                scenario=entry.get("scenario-type", "fixed-seq-len"),
                dataset=(row.get("dataset") or {}).get("loader"),
                values=values(row),
                result="passed",
                run_id=run["id"],
                head=run["head_sha"],
            )
        )
    evaluations = [
        Evaluation(
            key=identity(reuse.eval_key(row)),
            suite=row.get("eval_suite") or "gsm8k",
            label=f"c{row['conc']}",
            metric=row.get("score_name") or "em_strict",
            score=row.get("score", row.get("em_strict")),
            samples=row.get("n_eff"),
            result="passed",
            run_id=run["id"],
            head=run["head_sha"],
        )
        for row in sorted(eval_rows, key=lambda row: (int(row["conc"]), row.get("eval_suite", "")))
    ]
    images = {row["image"] for row in rows}
    if len(images) != 1:
        raise VerificationError("Final report contains mixed images")
    publish(
        session,
        Attempt(
            kind="final",
            number=0,
            head=run["head_sha"],
            image=images.pop(),
            run_id=run["id"],
            run_attempt=run["run_attempt"],
            status="passed",
            change=Prose(
                en="Validate the complete updated-image family at this head.",
                zh="验证此提交更新镜像后的完整配置族。",
            ),
            next=Prose(
                en="Mark ready for maintainer review.",
                zh="标记为就绪，等待维护者审查。",
            ),
            benchmarks_expected=len(generated),
            benchmarks_passed=len(points),
            evals_expected=len(expected_evals(matrix)),
            evals_passed=len(expected_evals(matrix)),
            points=sorted(points, key=lambda point: (point.conc, point.label)),
            evals=evaluations,
        ),
    )


def public_point(entry: dict) -> dict:
    """Project generated settings onto the public BenchmarkRow identity (not metrics)."""
    from infx.matrix.generate import _hardware_family

    agentic = entry.get("scenario-type") == "agentic-coding"
    multi = entry.get("prefill") is not None
    point = {
        "model": entry["model-prefix"],
        "hardware": _hardware_family(entry["runner"]),
        "framework": entry["framework"],
        "precision": entry["precision"],
        "spec_method": entry["spec-decoding"],
        "disagg": entry.get("disagg", False),
        "is_multinode": multi,
        "benchmark_type": "agentic_traces" if agentic else "single_turn",
        "isl": None if agentic else entry["isl"],
        "osl": None if agentic else entry["osl"],
        "offload_mode": "on" if entry.get("kv-offloading", "none") != "none" else "off",
        "conc": int(entry["conc"]),
        "image": entry["image"],
    }
    for role in ("prefill", "decode"):
        topology = entry[role] if multi else entry
        point.update(
            {
                f"{role}_tp": topology["tp"],
                f"{role}_ep": topology.get("ep", 1),
                f"{role}_dp_attention": topology.get("dp-attn", False),
                f"{role}_num_workers": topology["num-worker"] if multi else 0,
            }
        )
    return point


def matrix_points(matrix: dict) -> list[dict]:
    from .validation import benchmark_entries

    return [
        {**entry, "conc": int(conc)}
        for entry in benchmark_entries(matrix)
        for conc in (entry["conc"] if isinstance(entry["conc"], list) else [entry["conc"]])
    ]


def check_baseline_coverage(matrix: dict, baseline: Baseline | None) -> None:
    """Current-family completeness cannot replace the frozen original point roster."""
    if baseline is None or not baseline.points:
        raise VerificationError("Missing frozen baseline point roster")
    if {point.key for point in baseline.points} - {point_key(p) for p in matrix_points(matrix)}:
        raise VerificationError("Final matrix omits or changes frozen baseline points")


def prepare_baseline(session: Session, context: dict, model: str, goal: Prose) -> Baseline:
    """Freeze source-date rows against their own producer's complete family.

    Legacy fingerprints may be absent, but exact producer provenance and a unique
    workload/topology/concurrency match are required. Raw API data stays private.
    """
    from fnmatch import fnmatchcase

    from .api import fetch
    from .validation import canonical_matrix

    matrix = canonical_matrix(session.repository, session.candidate.base, session.candidate.family)
    feed = fetch("benchmarks", model=model, date=context["source"]["date"])
    info = fetch("workflow-info", date=context["source"]["date"])
    # Public database bigint IDs are serialized as strings; URLs use decimal IDs.
    producers = {int(row["github_run_id"]): row for row in info.payload["runs"]}
    heads = {}
    for row in info.payload["runConfigs"]:
        if row.get("head_sha"):
            heads.setdefault(int(row["github_run_id"]), set()).add(row["head_sha"])
    old_image = context["source"]["image"]
    entries = {point_key(entry): entry for entry in matrix_points(matrix)}
    if not entries or any(entry["image"] != old_image for entry in entries.values()):
        raise VerificationError("Baseline source image no longer matches the selected base")
    historical: dict[str, list[dict]] = {}
    published: dict[str, Point] = {}
    unverified: list[dict] = []
    family_runs = {
        int(change["workflow_run_id"])
        for change in info.payload["changelogs"]
        if any(
            fnmatchcase(session.candidate.family.split(":", 1)[1], key)
            for key in change["config_keys"]
        )
    }
    for row in feed.payload:
        # Do not filter ISL/OSL here: that would erase other curves in the original family.
        if any(
            row.get(key) != context["source"][key]
            for key in (
                "model",
                "hardware",
                "framework",
                "precision",
                "spec_method",
                "disagg",
                "image",
            )
        ):
            continue
        producer = re.fullmatch(
            r"https://github.com/"
            + re.escape(session.repository)
            + r"/actions/runs/(\d+)(?:/attempts/(\d+))?",
            row.get("run_url") or "",
        )
        run_id = int(producer[1]) if producer else None
        run_attempt = int(producer[2]) if producer and producer[2] else None
        if (
            run_id not in producers
            or len(heads.get(run_id, ())) != 1
            or (run_attempt is not None and run_attempt > int(producers[run_id]["run_attempt"]))
        ):
            # Classify after reconstructing the complete historical family, so an
            # unrelated sibling cannot block it and feed ordering cannot hide points.
            unverified.append(row)
            continue
        head = next(iter(heads[run_id]))
        if head not in historical:
            historical[head] = matrix_points(
                canonical_matrix(
                    session.repository, head, session.candidate.family, historical=True
                )
            )
        matches = [
            entry
            for entry in historical[head]
            if all(row.get(key) == value for key, value in public_point(entry).items())
        ]
        if not matches:  # A distinct sibling workload/topology is not this family's baseline.
            continue
        if row.get("recipe_fingerprint"):
            matches = [
                entry
                for entry in matches
                if entry["recipe-fingerprint"] == row["recipe_fingerprint"]
            ]
        elif run_id not in family_runs:
            raise VerificationError("Legacy baseline producer does not select the candidate family")
        if len(matches) != 1:
            raise VerificationError("Public baseline recipe identity is ambiguous or mismatched")
        entry = matches[0]
        key = point_key(entry)
        if key in published:
            raise VerificationError("Duplicate public baseline point")
        # Retain all original points, even if a current family or API response is smaller.
        entries.update(
            (point_key(point), point) for point in historical[head] if point["image"] == old_image
        )
        published[key] = Point(
            key=key,
            label=point_label(entry),
            conc=entry["conc"],
            scenario=entry.get("scenario-type", "fixed-seq-len"),
            # Dataset is not in BenchmarkRow; AgentX deltas remain N/A until proven.
            values=values(row),
            result="passed",
            run_id=run_id,
            head=head,
            run_attempt=run_attempt,
        )
    identities = [public_point(entry) for entry in entries.values()]
    if any(
        any(all(row.get(key) == value for key, value in point.items()) for point in identities)
        for row in unverified
    ):
        raise VerificationError("Public baseline producer provenance is unavailable")
    if not published:
        raise VerificationError("No verified public baseline points for the selected family")
    points = [
        published.get(key)
        or Point(
            key=key,
            label=point_label(entry),
            conc=entry["conc"],
            scenario=entry.get("scenario-type", "fixed-seq-len"),
            values=Values(),
            result="unavailable",
        )
        for key, entry in entries.items()
    ]
    return Baseline(
        family=session.candidate.family,
        date=context["source"]["date"],
        image=old_image,
        goal=goal,
        sources=[feed.url, info.url],
        points=sorted(
            points, key=lambda point: (point.label.split(" c")[0], point.conc, point.label)
        ),
    )
