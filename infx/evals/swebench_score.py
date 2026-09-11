#!/usr/bin/env python3
"""Score SWE-bench predictions and emit the repository's lm-eval result shape."""

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterator, Optional

DEFAULT_DATASET = "princeton-nlp/SWE-bench_Lite"
DEFAULT_TASK = "swebench_lite"

_FENCED_DIFF_RE = re.compile(
    r"```(?:diff|patch)?\s*\n(?P<body>.*?)```",
    re.DOTALL | re.IGNORECASE,
)
_DIFF_GIT_RE = re.compile(r"(?:^|\n)(diff --git .*)", re.DOTALL)

_DIFF_LINE_PREFIXES = (
    "diff ", "index ", "--- ", "+++ ", "@@", "+", "-", " ", "\\",
    "old mode ", "new mode ", "new file mode ", "deleted file mode ",
    "rename ", "copy ", "similarity ", "dissimilarity ",
    "Binary files ", "GIT binary patch",
)


def _trim_to_diff_body(text: str) -> str:
    """Drop trailing non-diff output from a generated patch."""
    lines = text.splitlines()
    out: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        if lines[i].startswith(_DIFF_LINE_PREFIXES):
            out.append(lines[i])
            i += 1
            continue
        if lines[i] == "":
            j = i
            while j < n and lines[j] == "":
                j += 1
            if j < n and lines[j].startswith(_DIFF_LINE_PREFIXES):
                out.extend(lines[i:j])
                i = j
                continue
        break
    return "\n".join(out)


def extract_patch(text: str) -> str:
    """Extract a unified diff from a model generation."""
    if not text:
        return ""

    def _finish(body: str) -> str:
        body = _trim_to_diff_body(body).strip("\n")
        return body + "\n" if body else ""

    for match in _FENCED_DIFF_RE.finditer(text):
        body = match.group("body")
        if "diff --git" in body or body.lstrip().startswith(("--- ", "+++ ")):
            return _finish(body)
    git_match = _DIFF_GIT_RE.search(text)
    if git_match:
        trimmed = _finish(git_match.group(1))
        if trimmed:
            return trimmed
    lone = _FENCED_DIFF_RE.search(text)
    if lone:
        body = lone.group("body").strip("\n")
        return body + "\n" if body else ""
    return text.strip("\n") + "\n" if text.strip() else ""


def _response_text(record: dict) -> str:
    """Read response text from supported lm-eval sample schemas."""
    for key in ("filtered_resps", "resps"):
        val = record.get(key)
        while isinstance(val, (list, tuple)) and val:
            val = val[0]
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _instance_id(record: dict) -> Optional[str]:
    doc = record.get("doc")
    if isinstance(doc, dict):
        for key in ("instance_id", "instance", "id"):
            val = doc.get(key)
            if isinstance(val, str) and val:
                return val
    val = record.get("instance_id")
    return val if isinstance(val, str) and val else None


def iter_samples(samples_dir: Path) -> Iterator[dict]:
    """Yield JSON records from every samples_*.jsonl under ``samples_dir``."""
    files = sorted(samples_dir.rglob("samples_*.jsonl"))
    if not files:
        raise FileNotFoundError(
            f"no samples_*.jsonl found under {samples_dir} -- did lm-eval run "
            "with --log_samples?"
        )
    for path in files:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def build_predictions(samples_dir: Path, model_name: str) -> list[dict]:
    """Turn lm-eval samples into swebench prediction rows (dedup by instance)."""
    by_instance: dict[str, dict] = {}
    skipped = 0
    for record in iter_samples(samples_dir):
        instance_id = _instance_id(record)
        if not instance_id:
            skipped += 1
            continue
        patch = extract_patch(_response_text(record))
        by_instance[instance_id] = {
            "instance_id": instance_id,
            "model_name_or_path": model_name,
            "model_patch": patch,
        }
    if skipped:
        print(f"WARN: skipped {skipped} sample(s) with no instance_id", file=sys.stderr)
    if not by_instance:
        raise ValueError("no usable predictions extracted from samples")
    return list(by_instance.values())


def write_predictions(predictions: list[dict], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as fh:
        for row in predictions:
            fh.write(json.dumps(row) + "\n")


def run_harness(
    predictions_path: Path,
    dataset_name: str,
    run_id: str,
    work_dir: Path,
    max_workers: int,
    namespace: Optional[str],
    modal: bool = False,
    timeout: Optional[int] = None,
) -> None:
    """Invoke the official swebench harness (local Docker, or Modal sandboxes)."""
    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", dataset_name,
        "--predictions_path", str(predictions_path),
        "--run_id", run_id,
    ]
    if timeout is not None:
        cmd += ["--timeout", str(timeout)]
    if modal:
        cmd += ["--modal", "true", "--max_workers", str(max_workers)]
    else:
        cmd += ["--max_workers", str(max_workers)]
        if namespace is not None:
            cmd += ["--namespace", namespace]
    print(f"[swebench] running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(work_dir), check=True)


def find_report(work_dir: Path, model_name: str, run_id: str) -> Path:
    """Locate the harness report JSON, tolerant to known layout variants."""
    sanitized = model_name.replace("/", "__")
    candidates = [
        work_dir / f"{sanitized}.{run_id}.json",
        work_dir / f"{model_name}.{run_id}.json",
        work_dir / "evaluation_results" / "results.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    for path in sorted(work_dir.rglob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and (
            "resolved_instances" in data or "resolved_ids" in data
        ):
            return path
    raise FileNotFoundError(
        f"could not locate a swebench report under {work_dir} "
        f"(looked for {[str(c) for c in candidates]})"
    )


def parse_resolved(report: dict) -> tuple[int, int]:
    """Return resolved and submitted counts from a harness report.

    Sliced runs use submitted instances rather than the full dataset size.
    """
    resolved: Optional[int] = None
    for key in ("resolved_instances", "resolved", "num_resolved"):
        if isinstance(report.get(key), int):
            resolved = report[key]
            break
    if resolved is None and isinstance(report.get("resolved_ids"), list):
        resolved = len(report["resolved_ids"])

    total: Optional[int] = None
    for key in ("submitted_instances", "completed_instances", "total_instances"):
        val = report.get(key)
        if isinstance(val, int) and val > 0:
            total = val
            break
    if total is None:
        for key in ("submitted_ids", "completed_ids"):
            if isinstance(report.get(key), list) and report[key]:
                total = len(report[key])
                break

    if resolved is None or total is None or total <= 0:
        raise ValueError(
            f"could not parse resolved/total from report keys {sorted(report)}"
        )
    return resolved, total


def build_results_json(
    task: str,
    resolved: int,
    total: int,
    model_name: str,
    lm_eval_version: str,
    report: Optional[dict],
) -> dict:
    """Publish resolved rate as the exact-match metric used by score validation."""
    rate = resolved / total
    stderr = math.sqrt(rate * (1.0 - rate) / total) if total else 0.0
    return {
        "lm_eval_version": lm_eval_version,
        "model_name": model_name,
        "results": {
            task: {
                "alias": task,
                "exact_match,resolved": rate,
                "exact_match_stderr,resolved": stderr,
            }
        },
        "configs": {
            task: {
                "metric_list": [{"metric": "exact_match"}],
                "filter_list": [{"name": "resolved"}],
            }
        },
        "n-samples": {task: {"effective": total, "original": total}},
        "swebench": {
            "resolved": resolved,
            "total": total,
            "resolved_rate": rate,
            "report": report,
        },
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Score SWE-bench patches from lm-eval samples")
    parser.add_argument("--samples-dir", default=None, help="dir containing lm-eval samples_*.jsonl (single-shot mode)")
    parser.add_argument(
        "--predictions-file", default=None,
        help="pre-built predictions.jsonl (agentic mode) -- skips samples parsing",
    )
    parser.add_argument("--out-dir", required=True, help="dir to write predictions + results JSON")
    parser.add_argument("--model-name", required=True, help="served model name (model_name_or_path)")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--task-name", default=DEFAULT_TASK)
    parser.add_argument("--run-id", default=None, help="harness run id (default: task name)")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--instance-timeout", type=int, default=None,
        help="per-instance test timeout in seconds (harness default 1800)",
    )
    parser.add_argument(
        "--namespace", default=None,
        help="local-Docker --namespace value (pass '' on arm/Mac to build images locally)",
    )
    parser.add_argument(
        "--modal", action="store_true",
        help="score on Modal remote sandboxes instead of local Docker (needs modal creds)",
    )
    parser.add_argument("--lm-eval-version", default="unknown")
    parser.add_argument(
        "--predictions-only", action="store_true",
        help="write predictions.jsonl and stop (no scoring; score elsewhere)",
    )
    parser.add_argument(
        "--no-run", action="store_true",
        help="skip the Docker harness; requires --report (offline/testing)",
    )
    parser.add_argument(
        "--report", default=None,
        help="path to a pre-computed harness report JSON (implies --no-run)",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or args.task_name

    if args.predictions_file:
        src = Path(args.predictions_file)
        text = src.read_text(encoding="utf-8", errors="replace")
        try:
            blob = json.loads(text)
            predictions = list(blob.values()) if isinstance(blob, dict) else blob
        except json.JSONDecodeError:
            predictions = [json.loads(line) for line in text.splitlines() if line.strip()]
        if not predictions:
            print(f"ERROR: no predictions in {src}", file=sys.stderr)
            return 1
        predictions_path = out_dir / "predictions.jsonl"
        write_predictions(predictions, predictions_path)
        print(f"[swebench] using {len(predictions)} pre-built predictions -> {predictions_path}")
    elif args.samples_dir:
        predictions = build_predictions(Path(args.samples_dir), args.model_name)
        predictions_path = out_dir / "predictions.jsonl"
        write_predictions(predictions, predictions_path)
        print(f"[swebench] wrote {len(predictions)} predictions -> {predictions_path}")
    else:
        print("ERROR: one of --samples-dir or --predictions-file is required", file=sys.stderr)
        return 1

    if args.predictions_only:
        print("[swebench] predictions-only: skipping scoring (score elsewhere)")
        return 0

    if args.report:
        report = json.loads(Path(args.report).read_text(encoding="utf-8", errors="replace"))
    elif args.no_run:
        print("ERROR: --no-run requires --report", file=sys.stderr)
        return 1
    else:
        run_harness(
            predictions_path, args.dataset_name, run_id,
            out_dir, args.max_workers, args.namespace, modal=args.modal,
            timeout=args.instance_timeout,
        )
        report_path = find_report(out_dir, args.model_name, run_id)
        report = json.loads(report_path.read_text(encoding="utf-8", errors="replace"))
        # Artifact collection requires a stable name.
        staged = out_dir / f"swebench_report_{args.task_name}.json"
        if report_path.resolve() != staged.resolve():
            staged.write_text(json.dumps(report, indent=2), encoding="utf-8")

    resolved, total = parse_resolved(report)

    results = build_results_json(
        args.task_name, resolved, total, args.model_name,
        args.lm_eval_version, report,
    )
    results_path = out_dir / f"results_{args.task_name}.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(
        f"[swebench] {args.task_name}: resolved {resolved}/{total} "
        f"= {resolved / total:.4f} -> {results_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
