"""GreenSpec on the serving client: subset selection, request payloads, and the artifact guard.

GreenSpec is an internal speculative-decoding bench cut from production chat traffic. A release
ships every subset in one flat ``greenspec.jsonl``: each line is a fixed set of metadata keys
(``META_KEYS``) followed by a complete chat-completions body with no ``model`` and no ``stream``.
The ``greenspec`` scenario replays one selection of it through AgentX as a ``raw_payload`` file:

``settings``   the selection and payload options a serving YAML asks for (overrides:).
``write_selection``  select the rows and write each as a ready request body.
``tee_filtered``     run AgentX with its console passed through ``ConsoleFilter``: no text that can
                     echo server error text, each progress bar at most once per interval.
``guard``      run after the client, before any upload. The benchmark repo is public and its
               artifacts are downloadable, so nothing derived from row content may leave the
               runner: keep only ``ALLOWLIST`` in the result dir, strip error messages from the
               aiperf export (a server error can echo request content), withhold the server logs,
               then fail if any ``question_id`` of the release, or any file over ``MAX_BYTES``,
               is still in what gets uploaded.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FAMILY = "greenspec"
TAG = "GreenSpec"  # how the family names itself in the job log
META_KEYS = ("question_id", "category", "bucket", "lite", "isl", "osl", "thinking_src", "recorded")
MAX_TOKEN_KEYS = ("max_tokens", "max_completion_tokens")
SUBSETS = ("diverse", "agentic_coding", "agentic_tools", "llm_task")
MODES = ("target-default", "recorded")
OVERRIDES = ("subset", "lite", "bucket", "mode", "max_tokens", "requests")
DEFAULT_MAX_TOKENS = 8192
BUCKET = re.compile(r"[a-z_]+__[0-9a-z+-]+")
# Relative to the result dir; everything else is removed before the upload.
ALLOWLIST = (
    "aiperf_artifacts/profile_export_aiperf.json",
    "aiperf_artifacts/server_metrics_export.json",  # engine counters and gauges, nothing of a row
    "benchmark_command.txt",
    "benchmark.log",
    "scenario.yaml",
    "serving.yaml",
    "dataset.txt",
    "spec_metrics_before.json",
    "spec_metrics_after.json",
)
MAX_BYTES = 16 * 1024 * 1024
# aiperf's tqdm bars: "{desc}: {n}/{total} |{bar}| {pct}% [{elapsed}<{remaining}]", or
# "{desc}: {pct}% |{bar}| [{elapsed}<{remaining}]" when the total is unknown.
BAR_RE = re.compile(
    r"(?P<desc>[A-Za-z][A-Za-z _-]{0,30}): +(?:(?P<n>[\d,]+)/(?P<total>[\d,]+) +)?"
    r"(?:(?P<pct0>\d+)% +)?\|[^|]*\| *(?:(?P<pct>\d+)% +)?"
    r"\[(?P<elapsed>[\d:]+)<(?P<eta>[\d:?]+)\]"
)
# aiperf's log record: "14:27:38.612 INFO     message (file.py:64)".
LOG_RE = re.compile(r"\d\d:\d\d:\d\d\.\d{3} +([A-Z]+) +")
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
WITHHELD_LEVELS = ("ERROR", "CRITICAL")
SERVER_LOG_STUB = (
    "server log withheld: GreenSpec rows are internal and a server log can echo them\n"
)
PROGRESS_INTERVAL = 60.0


class GreenSpecError(ValueError):
    """A selection problem, as one line."""


@dataclass(frozen=True)
class Settings:
    subset: str = "diverse"
    lite: bool = False
    bucket: str | None = None
    mode: str = "target-default"
    max_tokens: int = DEFAULT_MAX_TOKENS
    requests: int | None = None  # None: the whole selection

    @property
    def recorded(self) -> bool:
        return self.mode == "recorded"


def normalize(overrides: Mapping[str, Any]) -> Settings:
    """The overrides of a greenspec serving YAML as ``Settings``; one line per problem."""
    if unknown := sorted(set(overrides) - set(OVERRIDES)):
        raise GreenSpecError(
            f"overrides: greenspec takes {', '.join(OVERRIDES)}, not {', '.join(unknown)}"
        )
    subset = overrides.get("subset", "diverse")
    if subset not in SUBSETS:
        raise GreenSpecError(f"overrides.subset: {subset!r} is not one of {', '.join(SUBSETS)}")
    lite = overrides.get("lite", False)
    if not isinstance(lite, bool):
        raise GreenSpecError(f"overrides.lite: true or false, got {lite!r}")
    bucket = overrides.get("bucket")
    if bucket is not None and not (isinstance(bucket, str) and BUCKET.fullmatch(bucket)):
        raise GreenSpecError(
            f"overrides.bucket: <category>__<range> like llm_task__16k+, got {bucket!r}"
        )
    mode = overrides.get("mode", "target-default")
    if mode not in MODES:
        raise GreenSpecError(f"overrides.mode: {mode!r} is not one of {', '.join(MODES)}")
    max_tokens = overrides.get("max_tokens", DEFAULT_MAX_TOKENS)
    requests = overrides.get("requests")
    for field, value in (("max_tokens", max_tokens), ("requests", requests)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise GreenSpecError(f"overrides.{field}: a positive integer, got {value!r}")
    return Settings(subset, lite, bucket, mode, max_tokens, requests)


def settings(env: Mapping[str, str]) -> Settings | None:
    """The job's ``Settings`` when its serving YAML runs the greenspec scenario, else None."""
    import yaml  # not imported at the top: the client's first pass runs before the AgentX venv exists

    try:
        raw = yaml.safe_load(env.get("SERVING_YAML") or "")
    except yaml.YAMLError:
        return None
    if not isinstance(raw, dict) or raw.get("scenario") != FAMILY:
        return None
    return normalize(raw.get("overrides") or {})


def label(chosen: Settings) -> str:
    """The result label of a selection: greenspec-diverse, greenspec-agentic_tools-lite-recorded-mt4096-n200, ..."""
    parts = [FAMILY, chosen.subset]
    if chosen.bucket:
        parts.append(chosen.bucket.replace("__", "_").replace("+", "plus"))
    if chosen.lite:
        parts.append("lite")
    if chosen.recorded:
        parts.append("recorded")
    if chosen.max_tokens != DEFAULT_MAX_TOKENS:
        parts.append(f"mt{chosen.max_tokens}")
    if chosen.requests:
        parts.append(f"n{chosen.requests}")
    return "-".join(parts)


def iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def keep(row: dict[str, Any], chosen: Settings) -> bool:
    return (
        (chosen.subset == "diverse" or row["category"] == chosen.subset)
        and (not chosen.lite or row["lite"])
        and (chosen.bucket is None or row["bucket"] == chosen.bucket)
    )


def select(path: Path, chosen: Settings) -> list[dict[str, Any]]:
    return [row for row in iter_rows(path) if keep(row, chosen)]


SCHEMA_MAPS = ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas")
SCHEMA_LISTS = ("allOf", "anyOf", "oneOf", "prefixItems", "items")
SCHEMA_ONES = (
    "items",
    "additionalProperties",
    "additionalItems",
    "contains",
    "propertyNames",
    "not",
    "if",
    "then",
    "else",
    "unevaluatedItems",
    "unevaluatedProperties",
)


def fix_required(schema: Any) -> Any:
    """A copy of a JSON schema whose ``required`` keyword is a list wherever it was an empty dict.

    Some releases (v1.1.0) serialized "no required fields" as ``{}``. vLLM accepts it, SGLang
    rejects the request with a 400, and the chat template prints nothing for either. Only the
    keyword is rewritten: a property *named* ``required`` lives under ``properties`` and is a
    sub-schema, so the walk goes by schema structure and leaves names, enum, const and default alone.
    """
    if not isinstance(schema, dict):
        return schema
    out = dict(schema)
    if out.get("required") == {}:
        out["required"] = []
    for key in SCHEMA_MAPS:
        if isinstance(out.get(key), dict):
            out[key] = {name: fix_required(sub) for name, sub in out[key].items()}
    for key in SCHEMA_LISTS:
        if isinstance(out.get(key), list):
            out[key] = [fix_required(sub) for sub in out[key]]
    for key in SCHEMA_ONES:
        if isinstance(out.get(key), dict):
            out[key] = fix_required(out[key])
    return out


def fix_tools(tools: Any) -> Any:
    """``tools`` with every function's ``parameters`` run through :func:`fix_required`."""
    if not isinstance(tools, list):
        return tools
    fixed = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function, dict) and "parameters" in function:
            tool = {
                **tool,
                "function": {**function, "parameters": fix_required(function["parameters"])},
            }
        fixed.append(tool)
    return fixed


def payload(row: dict[str, Any], model: str, chosen: Settings) -> dict[str, Any]:
    """One row -> the request body the client sends for it.

    Target-default keeps messages, tools and the like and lets the served model choose its own
    sampling; recorded also replays the production sampling. The output cap only ever lowers the
    row's own cap.
    """
    body = {k: v for k, v in row.items() if k not in META_KEYS}
    if "tools" in body:
        body["tools"] = fix_tools(body["tools"])
    if chosen.recorded:
        body.update(row.get("recorded") or {})
    capped = [k for k in MAX_TOKEN_KEYS if k in body]
    if capped:
        for k in capped:
            body[k] = min(body[k], chosen.max_tokens)
    else:
        body["max_tokens"] = chosen.max_tokens
    return {"model": model, **body, "stream": True, "stream_options": {"include_usage": True}}


def write_selection(dataset: Path, chosen: Settings, model: str, out: Path) -> int:
    """Write the selected rows as one request body per line to ``out``; return how many."""
    rows = select(dataset, chosen)
    if not rows:
        raise GreenSpecError(f"selection {label(chosen)}: no row of the release matches")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(
            json.dumps(payload(row, model, chosen), ensure_ascii=False) + "\n" for row in rows
        )
    return len(rows)


class ConsoleFilter:
    """aiperf's console for the job log, minus what can echo server error text.

    The Error Summary table is dropped (its Message column is the raw server response) and
    replaced by its codes and counts; an ERROR or CRITICAL record keeps its timestamp and level and
    loses its text and continuation lines (a traceback, wrapped text). An unterminated table stays
    dropped. Each progress bar is rebuilt from its numbers and printed at most once per
    ``interval`` seconds, plus once at 100%.
    """

    def __init__(
        self,
        write: Callable[[str], None],
        interval: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._write = write
        self._interval = interval
        self._clock = clock
        self._bars: dict[str, tuple[float, str]] = {}
        self._in_table = False
        self._dropping = False
        self._errors: list[str] = []

    def feed(self, segment: str) -> None:
        text = ANSI_RE.sub("", segment).rstrip()
        if not text:  # tqdm's \r next to a record's \n leaves empty segments
            return
        bar = BAR_RE.match(text.lstrip())
        if bar:
            self._dropping = False  # a bar ends the continuation of an ERROR record
            self._bar(bar)
            # tqdm writes no newline, so a log record can follow on the same line.
            text = text.lstrip()[bar.end() :]
            if not text.strip():
                return
        if self._in_table:
            self._table_line(text)
            return
        log = LOG_RE.match(text)
        if log:
            self._dropping = log.group(1) in WITHHELD_LEVELS
            if self._dropping:
                self._write(f"{text[: log.end()]}<withheld: may echo server error text>")
                return
        elif "Error Summary" in text:
            self._in_table, self._dropping = True, False
            return
        elif self._dropping:
            return
        self._write(text)

    def close(self) -> None:
        if self._in_table:
            self._in_table = False
            self._report_errors()

    def _bar(self, m: re.Match[str]) -> None:
        desc, pct = m["desc"].strip(), m["pct"] or m["pct0"]
        now, last = self._clock(), self._bars.get(desc)
        if last and (last[1] == pct == "100" or (pct != "100" and now - last[0] < self._interval)):
            return
        self._bars[desc] = (now, pct)
        count = f" {m['n']}/{m['total']}" if m["n"] else ""
        self._write(
            f"GreenSpec progress: {desc}{count} ({pct}%) elapsed {m['elapsed']} eta {m['eta']}"
        )

    def _table_line(self, text: str) -> None:
        if text.lstrip().startswith(("└", "╰")):
            self._in_table = False
            self._report_errors()
            return
        cells = [c.strip() for c in text.split("│")]
        if len(cells) >= 4 and (cells[1].isdigit() or cells[1] == "N/A"):
            count = cells[-2].replace(",", "")
            if count.isdigit():
                self._errors.append(f"{cells[1]} x{count}")

    def _report_errors(self) -> None:
        listed = ", ".join(self._errors) or "no rows parsed"
        self._write(
            f"GreenSpec: aiperf error summary withheld (messages can echo requests): {listed}"
        )
        self._errors = []


def filter_console(stream: Any, write: Callable[[str], None], interval: float) -> None:
    """Feed a binary stream to ``ConsoleFilter`` one \\r- or \\n-delimited segment at a time.

    A segment the filter fails on is dropped rather than ending the read: aiperf writes into this
    pipe and would die of a broken pipe.
    """
    console = ConsoleFilter(write, interval)

    def feed(raw: bytes) -> None:
        try:
            console.feed(raw.decode("utf-8", errors="replace"))
        except Exception:  # noqa: BLE001
            write("GreenSpec: console line dropped (filter error)")

    pending = b""
    while chunk := stream.read1(65536):
        *segments, pending = re.split(rb"[\r\n]", pending + chunk)
        for segment in segments:
            feed(segment)
    if pending:
        feed(pending)
    console.close()


def tee_filtered(
    argv: Sequence[str], log: Path, env: Mapping[str, str], interval: float = PROGRESS_INTERVAL
) -> int:
    """``argv 2>&1 | filter | tee log``: the job log and ``log`` get the filtered console only."""
    from infx.bench import proc

    proc.echo(argv, env)
    with log.open("w", encoding="utf-8") as sink:
        try:
            child = subprocess.Popen(
                argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
            )
        except OSError as error:
            print(f"ERROR: cannot run {argv[0]}: {error}", file=sys.stderr, flush=True)
            return 127

        def write(line: str) -> None:
            for stream in (sys.stdout, sink):
                print(line, file=stream, flush=True)

        with child:
            if child.stdout is not None:
                filter_console(child.stdout, write, interval)
    return proc.status(child.returncode)


def _redact_errors(export_path: Path) -> None:
    export = json.loads(export_path.read_text())
    for entry in export.get("error_summary") or []:
        details = entry.get("error_details")
        if isinstance(details, dict) and "message" in details:
            details["message"] = "<withheld>"
    export_path.write_text(json.dumps(export, indent=2))


def guard(
    dataset: Path,
    result_dir: Path,
    uploads: Sequence[Path],
    server_logs: Sequence[Path],
    keep_log_dir: Path | None = None,
) -> list[str]:
    """Prune, redact and scan everything a GreenSpec job uploads. Returns the problems.

    ``result_dir`` keeps only ``ALLOWLIST``; ``uploads`` (the aggregate result) are scanned as they
    are; each of ``server_logs`` is moved to ``keep_log_dir`` (node-local) if given and replaced by a stub.
    """
    if result_dir.is_dir():
        for p in sorted(result_dir.rglob("*"), reverse=True):
            rel = p.relative_to(result_dir).as_posix()
            if p.is_dir():
                if not any(p.iterdir()):
                    p.rmdir()
            elif rel not in ALLOWLIST:
                if p.parent == result_dir and p.suffix == ".log" and keep_log_dir is not None:
                    keep_log_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(p), str(keep_log_dir / p.name))
                else:
                    p.unlink()
        export = result_dir / "aiperf_artifacts" / "profile_export_aiperf.json"
        if export.is_file():
            _redact_errors(export)
    for log in server_logs:
        if log.is_file():
            if keep_log_dir is not None:
                keep_log_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(log), str(keep_log_dir / log.name))
            # Swap in a new inode: the server may still hold the log open, and a truncate
            # would let its next writes land in the stub (after a hole of NULs).
            tmp = log.with_name(log.name + ".stub")
            tmp.write_text(SERVER_LOG_STUB)
            tmp.replace(log)
    qids = {r["question_id"] for r in iter_rows(dataset)}
    files = [p for p in result_dir.rglob("*") if p.is_file()] if result_dir.is_dir() else []
    files += [p for p in uploads if p.is_file()]
    problems = []
    for p in files:
        size = p.stat().st_size
        if size > MAX_BYTES:
            problems.append(f"{p}: {size} bytes > {MAX_BYTES}")
            continue
        text = p.read_text(errors="replace")
        if any(q in text for q in qids):
            problems.append(f"{p}: contains a GreenSpec question_id")
    return problems
