"""Report an InferenceX-app ingest outcome on the PR whose changelog entry it published.

InferenceX-app sends ``ingest-results-completed`` after every dispatched merge
ingest, on success and failure. The PR is resolved here from the Merge Ingest
run's commit, never taken from the payload, and one comment per merge run is
updated in place.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from infx import github

APP_REPOSITORY = "SemiAnalysisAI/InferenceX-app"
MERGE_WORKFLOW = ".github/workflows/merge-ingest.yml"
BOT_LOGIN = "github-actions[bot]"
CATEGORIES = ("benchmark", "eval", "trace", "power")
OUTCOMES = ("new", "duplicate", "skipped", "unmapped", "failed")
DETAIL_KEYS = (
    "benchmarkFiles",
    "rawBenchmarkRows",
    "coveredByPerJobFiles",
    "purgedPoints",
    "failedRunRows",
    "runStatsNew",
    "runStatsDuplicate",
    "evalSamplesNew",
    "gpuMetricSamplesNew",
    "powerPublicationPoints",
    "changelogEntriesWritten",
)
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/+-]{0,79}")
MAX_NAMES = 20


def _count(value: Any) -> int:
    """Return ``value`` as a non-negative int count; raise RuntimeError otherwise."""
    if type(value) is not int or value < 0:
        raise RuntimeError("Report counts must be non-negative integers")
    return value


def render_report(report: Any) -> list[str]:
    """Return markdown lines for the payload's count report, validating every value.

    Raises RuntimeError on a malformed report. A missing report renders a note.
    """
    if report is None:
        return ["No ingest report was produced; the ingest stopped before counting rows."]
    if not isinstance(report, dict) or not isinstance(report.get("counts"), dict):
        raise RuntimeError("Callback report must be an object with counts")
    counts = report["counts"]
    lines = [
        f"| Category | {' | '.join(OUTCOMES)} |",
        f"| --- | {' | '.join('---:' for _ in OUTCOMES)} |",
    ]
    for category in CATEGORIES:
        row = counts.get(category)
        if not isinstance(row, dict):
            raise RuntimeError(f"Callback report is missing {category} counts")
        lines.append(f"| {category} | {' | '.join(str(_count(row.get(o))) for o in OUTCOMES)} |")
    detail = report.get("detail") or {}
    if not isinstance(detail, dict):
        raise RuntimeError("Callback report detail must be an object")
    extras = [f"{key} {_count(detail[key])}" for key in DETAIL_KEYS if key in detail]
    if extras:
        lines += ["", f"<sub>{', '.join(extras)}</sub>"]
    unmapped = report.get("unmapped") or {}
    if not isinstance(unmapped, dict):
        raise RuntimeError("Callback report unmapped must be an object")
    for kind in ("models", "hardware", "precisions", "datasets"):
        names = unmapped.get(kind) or []
        if not isinstance(names, list):
            raise RuntimeError("Unmapped names must be lists")
        shown = [f"`{n}`" for n in names[:MAX_NAMES] if isinstance(n, str) and NAME.fullmatch(n)]
        if len(shown) != min(len(names), MAX_NAMES):
            raise RuntimeError("Unmapped names must be plain identifiers")
        if shown:
            more = f" (+{len(names) - MAX_NAMES} more)" if len(names) > MAX_NAMES else ""
            lines.append(f"\n⚠️ Unmapped {kind}: {', '.join(shown)}{more}")
    return lines


def merged_pr(repo: str, merge_run_id: str, token: str) -> int | None:
    """Return the PR merged by the Merge Ingest run ``merge_run_id``, or None if none matches.

    Raises RuntimeError when the run is not a Merge Ingest push to main.
    """
    run = github.api(repo, f"/actions/runs/{merge_run_id}", token)
    if (
        str(run.get("path", "")).split("@")[0] != MERGE_WORKFLOW
        or run.get("event") != "push"
        or run.get("head_branch") != "main"
    ):
        raise RuntimeError("Callback merge run is not a Merge Ingest push to main")
    sha = str(run.get("head_sha", ""))
    pulls = github.api(repo, f"/commits/{sha}/pulls", token)
    merged = [p for p in pulls if p.get("merged_at") and p.get("merge_commit_sha") == sha]
    return int(merged[0]["number"]) if merged else None


def report(repo: str, event: dict[str, Any], token: str, server_url: str) -> None:
    """Post or update the ingest comment for one ``ingest-results-completed`` event.

    Raises RuntimeError if the payload is malformed or from an unexpected source.
    """
    if event.get("action") != "ingest-results-completed":
        raise RuntimeError("Unsupported ingest callback type")
    payload = event["client_payload"]
    if payload.get("source-repository") != APP_REPOSITORY:
        raise RuntimeError("Unsupported ingest callback source")
    merge_run_id, source_run_id, app_run_id, app_attempt = (
        str(payload.get(key, ""))
        for key in ("merge-run-id", "source-run-id", "app-run-id", "app-run-attempt")
    )
    if not all(
        re.fullmatch(r"[0-9]+", v) for v in (merge_run_id, source_run_id, app_run_id, app_attempt)
    ):
        raise RuntimeError("Callback run identifiers must be numeric")
    succeeded = payload.get("ingest-succeeded")
    if succeeded not in {"true", "false"}:
        raise RuntimeError("Callback ingest-succeeded value must be true or false")
    target = str(payload.get("database-target", ""))
    if not re.fullmatch(r"[a-z-]+", target):
        raise RuntimeError("Callback database target must be a plain name")
    table = render_report(payload.get("report"))

    pr_number = merged_pr(repo, merge_run_id, token)
    if pr_number is None:
        print(f"No merged PR found for merge run {merge_run_id}; nothing to report.")
        return

    app_url = f"{server_url}/{APP_REPOSITORY}/actions/runs/{app_run_id}/attempts/{app_attempt}"
    merge_url = f"{server_url}/{repo}/actions/runs/{merge_run_id}"
    source_url = f"{server_url}/{repo}/actions/runs/{source_run_id}"
    marker = f"<!-- ingest-results-report merge-run={merge_run_id} -->"
    if succeeded == "true":
        status_en = f"✅ Results ingested into the `{target}` database."
        status_zh = f"✅ 结果已写入 `{target}` 数据库。"
    else:
        status_en = (
            f"❌ Ingest into the `{target}` database failed. "
            f"[Inspect the ingest workflow]({app_url})."
        )
        status_zh = (
            f"❌ 写入 `{target}` 数据库的 ingest 失败，请[查看 ingest workflow]({app_url})。"
        )
    links = (
        f"[Ingest workflow]({app_url}) · [Merge Ingest run]({merge_url}) · "
        f"[Benchmark source run]({source_url})"
    )
    body = "\n".join(
        [
            marker,
            f"**Ingest report** — {status_en}",
            "",
            links,
            "",
            *table,
            "",
            (
                "Counts cover only this run. Rows: benchmark_results, eval_results, agentic rows "
                "linked to a trace, gpu_metrics series. This comment updates if the ingest is rerun."
            ),
            "",
            "<details>",
            "<summary>中文</summary>",
            "",
            f"**Ingest 报告** — {status_zh}",
            "",
            (
                "上表仅统计本次运行写入的数据，各行单位依次为 benchmark_results 行、eval_results 行、"
                "已关联 trace 的 agentic 行和 gpu_metrics 序列。重新运行 ingest 时本评论会原地更新。"
            ),
            "",
            "</details>",
        ]
    )
    comments = github.paginate(repo, f"/issues/{pr_number}/comments", token, "")
    existing = next(
        (
            c
            for c in comments
            if c.get("user", {}).get("login") == BOT_LOGIN and marker in str(c.get("body"))
        ),
        None,
    )
    if existing:
        github.api(
            repo, f"/issues/comments/{existing['id']}", token, method="PATCH", data={"body": body}
        )
    else:
        github.api(repo, f"/issues/{pr_number}/comments", token, method="POST", data={"body": body})


def main() -> None:
    """Report the ingest callback from the GitHub Actions environment."""
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
    report(
        os.environ["GITHUB_REPOSITORY"],
        event,
        os.environ["GH_TOKEN"],
        os.environ["GITHUB_SERVER_URL"],
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
