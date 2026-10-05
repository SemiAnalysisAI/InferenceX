#!/usr/bin/env python3
"""Refresh the AgentX offload GitHub and unofficial-result inventory."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_REPO = "SemiAnalysisAI/InferenceX"
DEFAULT_BRANCH = "experiment/agentx-b200-offload"
CHART_BASE = (
    "https://inferencex.semianalysis.com/inference/minimax-m3"
    "?i_seq=agentic-traces&i_prec=fp4&i_pctl=p90&i_metric=y_tpPerGpu"
)
UNOFFICIAL_API = "https://inferencex.semianalysis.com/api/unofficial-run"


def gh_api(repo: str, path: str, fields: dict[str, str | int] | None = None) -> dict[str, Any]:
    command = ["gh", "api", f"repos/{repo}/{path}"]
    for key, value in (fields or {}).items():
        command.extend(["-f", f"{key}={value}"])
    command.extend(["--method", "GET"])
    return json.loads(subprocess.check_output(command, text=True))


def list_branch_runs(repo: str, branch: str) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    page = 1
    while True:
        batch = gh_api(
            repo,
            "actions/runs",
            {"branch": branch, "per_page": 100, "page": page},
        )["workflow_runs"]
        runs.extend(batch)
        if len(batch) < 100:
            return runs
        page += 1


def unofficial_counts(run_id: int) -> tuple[int, int, int]:
    url = f"{UNOFFICIAL_API}?runId={run_id}"
    try:
        with urllib.request.urlopen(url, timeout=45) as response:
            payload = json.load(response)
            return (
                response.status,
                len(payload.get("benchmarks", [])),
                len(payload.get("evaluations", [])),
            )
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return 404, 0, 0
        raise


def inspect_run(repo: str, run: dict[str, Any]) -> dict[str, Any]:
    run_id = int(run["id"])
    jobs = gh_api(repo, f"actions/runs/{run_id}/jobs", {"per_page": 100})["jobs"]
    artifacts = gh_api(repo, f"actions/runs/{run_id}/artifacts", {"per_page": 100})["artifacts"]
    api_status, benchmark_count, evaluation_count = unofficial_counts(run_id)
    title = str(run.get("display_title") or "")
    pure_nvme = "-nvme" in title and "-dram-nvme" not in title
    return {
        "workflow_run_id": run_id,
        "workflow_name": run.get("name"),
        "display_title": title,
        "status": run.get("status"),
        "conclusion": run.get("conclusion"),
        "created_at": run.get("created_at"),
        "updated_at": run.get("updated_at"),
        "html_url": run.get("html_url"),
        "job_ids": [int(job["id"]) for job in jobs],
        "artifact_ids": [int(artifact["id"]) for artifact in artifacts],
        "artifact_names": [artifact["name"] for artifact in artifacts],
        "unofficial_api_status": api_status,
        "benchmark_count": benchmark_count,
        "evaluation_count": evaluation_count,
        "renderable": benchmark_count > 0,
        "pure_nvme": pure_nvme,
    }


def chart_url(run_ids: list[int]) -> str:
    return f"{CHART_BASE}&unofficialruns={','.join(map(str, run_ids))}"


def markdown_table(runs: list[dict[str, Any]], *, chinese: bool = False) -> str:
    if chinese:
        lines = [
            "| Workflow run ID | 结论 | 图表 benchmark 数 | GitHub job 数 | 工作流 |",
            "| ---: | --- | ---: | ---: | --- |",
        ]
    else:
        lines = [
            "| Workflow run ID | Conclusion | Chart benchmarks | GitHub jobs | Workflow |",
            "| ---: | --- | ---: | ---: | --- |",
        ]
    for run in runs:
        run_id = run["workflow_run_id"]
        title = str(run["display_title"]).replace("|", "\\|")
        lines.append(
            f"| [{run_id}]({run['html_url']}) | {run['conclusion'] or 'none'} | "
            f"{run['benchmark_count']} | {len(run['job_ids'])} | `{title}` |"
        )
    return "\n".join(lines)


def english_doc(inventory: dict[str, Any]) -> str:
    return f"""# AgentX offload unofficial-run inventory

<div align=\"center\">

**English** | [中文](./UNOFFICIAL_RUNS_zh.md)

</div>

This generated index prevents the testing-branch workflow IDs and unofficial chart URLs
from being lost. The chart accepts **GitHub workflow run IDs**, not the individual job IDs.
Every individual job ID remains available in [`unofficial-runs.json`](./unofficial-runs.json).

Refresh all three generated files from the repository root:

```bash
python3 experiments/agentx-offload/update_unofficial_runs.py --write
```

The refresh queries every workflow on `{inventory['branch']}`, inventories its jobs and
artifacts, and asks the public unofficial-run API whether it currently returns benchmark
rows. `renderable` means the API returned at least one benchmark row; it does not mean the
run is scientifically valid. Consult [`runs.json`](./runs.json) for validity and study
conclusions.

Generated at: `{inventory['generated_at']}`

## URLs

All API-renderable benchmark points ({len(inventory['renderable_workflow_run_ids'])} runs):

<{inventory['all_renderable_url']}>

Pure-NVMe API-renderable points ({len(inventory['pure_nvme_renderable_workflow_run_ids'])} runs):

<{inventory['pure_nvme_renderable_url']}>

## Complete branch workflow list

Branch workflows: **{len(inventory['workflow_runs'])}**

Individual GitHub jobs: **{inventory['job_id_count']}**

{markdown_table(inventory['workflow_runs'])}
"""


def chinese_doc(inventory: dict[str, Any]) -> str:
    return f"""# AgentX 卸载 unofficial-run 清单

<div align=\"center\">

[English](./UNOFFICIAL_RUNS.md) | **中文**

</div>

本生成清单用于长期保存测试分支的工作流 ID 和 unofficial 图表 URL。
图表参数使用的是 **GitHub workflow run ID**，不是单个 job ID。所有单独的 job ID 均保存在
[`unofficial-runs.json`](./unofficial-runs.json) 中。

在仓库根目录运行以下命令可刷新三个生成文件：

```bash
python3 experiments/agentx-offload/update_unofficial_runs.py --write
```

刷新脚本会查询 `{inventory['branch']}` 上的全部工作流，记录其 job 和产物，
并调用公开的 unofficial-run API 判断当前是否返回 benchmark 记录。
`renderable` 只表示 API 至少返回一条 benchmark 记录，不代表该运行在科学上有效。
有效性和研究结论应以 [`runs.json`](./runs.json) 为准。

生成时间：`{inventory['generated_at']}`

## URL

全部可由 API 渲染的 benchmark 点
（{len(inventory['renderable_workflow_run_ids'])} 个运行）：

<{inventory['all_renderable_url']}>

纯 NVMe 且可由 API 渲染的点
（{len(inventory['pure_nvme_renderable_workflow_run_ids'])} 个运行）：

<{inventory['pure_nvme_renderable_url']}>

## 完整分支工作流列表

分支工作流：**{len(inventory['workflow_runs'])}**

GitHub 单独 job：**{inventory['job_id_count']}**

{markdown_table(inventory['workflow_runs'], chinese=True)}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    branch_runs = list_branch_runs(args.repo, args.branch)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        runs = list(executor.map(lambda run: inspect_run(args.repo, run), branch_runs))
    runs.sort(key=lambda run: run["workflow_run_id"])

    renderable_ids = [run["workflow_run_id"] for run in runs if run["renderable"]]
    pure_nvme_ids = [
        run["workflow_run_id"] for run in runs if run["renderable"] and run["pure_nvme"]
    ]
    inventory = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "repo": args.repo,
        "branch": args.branch,
        "workflow_run_ids": [run["workflow_run_id"] for run in runs],
        "job_id_count": sum(len(run["job_ids"]) for run in runs),
        "renderable_workflow_run_ids": renderable_ids,
        "pure_nvme_renderable_workflow_run_ids": pure_nvme_ids,
        "all_renderable_url": chart_url(renderable_ids),
        "pure_nvme_renderable_url": chart_url(pure_nvme_ids),
        "workflow_runs": runs,
    }

    base = Path(__file__).resolve().parent
    if args.write:
        (base / "unofficial-runs.json").write_text(
            json.dumps(inventory, indent=2, ensure_ascii=False) + "\n"
        )
        (base / "UNOFFICIAL_RUNS.md").write_text(english_doc(inventory))
        (base / "UNOFFICIAL_RUNS_zh.md").write_text(chinese_doc(inventory))
    else:
        print(json.dumps(inventory, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
