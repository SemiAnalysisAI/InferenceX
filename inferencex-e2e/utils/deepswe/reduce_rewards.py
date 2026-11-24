"""Reduce a Pier DeepSWE job to a native report and an ``inferencex-eval-v1`` result.

Reads ``<job-dir>/<task>__<id>/verifier/reward.json`` for every trial. Accuracy is the mean
reward over all trials (avg@n when ``--n-attempts`` is above 1). A trial without a reward is
errored; an errored trial, or fewer scored trials than the job selected, makes the run an
integration error, and a job that selected fewer than the 113 tasks is recorded with a ``limit``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TASK_NAME = "kimi_deepswe"
ADAPTER_NAME = "kimi-deepswe"
RESULT_FORMAT = "inferencex-eval-v1"
TOTAL_TASKS = 113


def _read_json(path: Path) -> dict[str, Any] | None:
    """The JSON object at ``path``, or None when it is missing or malformed."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _task_id(trial: Path) -> str:
    """The task id from the trial's config, else from the directory name."""
    config = _read_json(trial / "config.json") or {}
    name = config.get("task", {}).get("name") if isinstance(config.get("task"), dict) else None
    if not name:
        name = trial.name.rsplit("__", 1)[0]
    return str(name).split("/")[-1]


def collect(job_dir: Path) -> dict[str, Any]:
    """Per-task rewards, errored trials and the job's own statistics."""
    trials = sorted(path for path in job_dir.glob("*__*") if path.is_dir())
    by_task: dict[str, list[dict[str, Any]]] = {}
    errored: list[str] = []
    for trial in trials:
        task = _task_id(trial)
        reward = _read_json(trial / "verifier" / "reward.json")
        if reward is None or "reward" not in reward:
            errored.append(trial.name)
            by_task.setdefault(task, [])
            continue
        by_task.setdefault(task, []).append(
            {"trial": trial.name, "reward": float(reward["reward"]), **_tests(reward)}
        )
    result = _read_json(job_dir / "result.json") or {}
    return {
        "by_task": dict(sorted(by_task.items())),
        "errored_trials": errored,
        "n_trials": len(trials),
        "selected_tasks": _selected_tasks(job_dir, len(by_task)),
        "started_at": result.get("started_at"),
        "finished_at": result.get("finished_at"),
        "stats": result.get("stats", {}),
    }


def _selected_tasks(job_dir: Path, observed: int) -> int:
    """How many tasks the job selected: Pier's job config when it names them, else the trials."""
    config = _read_json(job_dir / "config.json") or {}
    datasets = config.get("datasets")
    if not isinstance(datasets, list):
        return observed
    total = 0
    for dataset in datasets:
        names = dataset.get("task_names") if isinstance(dataset, dict) else None
        if isinstance(names, list):
            total += len(names)
        elif isinstance(dataset, dict) and isinstance(dataset.get("n_tasks"), int):
            total += dataset["n_tasks"]
        else:
            return TOTAL_TASKS
    return total or observed


def _tests(reward: dict[str, Any]) -> dict[str, Any]:
    """The verifier's test ratios when the reward file carries them."""
    return {key: reward[key] for key in ("f2p", "p2p") if key in reward}


def summarize(collected: dict[str, Any], n_attempts: int) -> dict[str, Any]:
    """Accuracy, per-attempt pass@1 and the completion counts."""
    rewards = [entry["reward"] for entries in collected["by_task"].values() for entry in entries]
    attempts = []
    for k in range(n_attempts):
        kth = [
            entries[k]["reward"] for entries in collected["by_task"].values() if len(entries) > k
        ]
        attempts.append(sum(kth) / len(kth) if kth else 0.0)
    selected = collected["selected_tasks"]
    return {
        "accuracy": sum(rewards) / len(rewards) if rewards else 0.0,
        "pass1_by_attempt": attempts,
        "completed_trials": len(rewards),
        "expected_trials": selected * n_attempts,
        "selected_tasks": selected,
        "n_tasks": len(collected["by_task"]),
        "errored_trials": len(collected["errored_trials"]),
    }


def compatibility_result(model: str, summary: dict[str, Any], n_attempts: int) -> dict[str, Any]:
    """The lm-eval style result the eval collectors and ``validate_scores.py`` read."""
    result: dict[str, Any] = {
        "result_format": RESULT_FORMAT,
        "eval_adapter": ADAPTER_NAME,
        "model_name": model,
        "results": {
            TASK_NAME: {
                "exact_match,strict-match": summary["accuracy"],
                "exact_match_stderr,strict-match": 0.0,
            }
        },
        "configs": {
            TASK_NAME: {
                "metric_list": [{"metric": "exact_match"}],
                "filter_list": [{"name": "strict-match"}],
            }
        },
        "n-samples": {
            TASK_NAME: {
                "original": TOTAL_TASKS * n_attempts,
                "effective": summary["completed_trials"],
            }
        },
    }
    if summary["selected_tasks"] < TOTAL_TASKS:
        result["limit"] = summary["selected_tasks"]
    if summary["errored_trials"] or summary["completed_trials"] < summary["expected_trials"]:
        result["integration_error"] = {
            "type": "IncompleteRun",
            "message": (
                f"{summary['completed_trials']} of {summary['expected_trials']} trials scored, "
                f"{summary['errored_trials']} errored"
            ),
        }
    return result


def main(argv: list[str] | None = None) -> int:
    """Write ``deepswe_report.json`` and ``results_deepswe_<timestamp>.json`` to the output dir."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--job-dir", type=Path, required=True, help="Pier's jobs/<run> directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True, help="the served model name")
    parser.add_argument("--n-attempts", type=int, required=True, help="Pier n_attempts")
    args = parser.parse_args(argv)

    collected = collect(args.job_dir)
    summary = summarize(collected, args.n_attempts)
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S.%f")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": args.model,
        "task": TASK_NAME,
        "n_attempts": args.n_attempts,
        "summary": summary,
        **collected,
    }
    (args.output_dir / "deepswe_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    result = compatibility_result(args.model, summary, args.n_attempts)
    (args.output_dir / f"results_deepswe_{stamp}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"task": TASK_NAME, **summary}))
    return 1 if "integration_error" in result else 0


if __name__ == "__main__":
    sys.exit(main())
