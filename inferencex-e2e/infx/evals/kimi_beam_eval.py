#!/usr/bin/env python3
"""Run the stock Kimi Vendor Verifier BEAM (1M) scripts and project their outcome."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

UTC = timezone.utc  # noqa: UP017

TASK_NAME = "kimi_beam_1m"
SUPPORTED_TASK_NAMES = (TASK_NAME,)
ADAPTER_NAME = "kimi-beam"
RESULT_FORMAT = "inferencex-eval-v1"
NATIVE_REPORT_FILENAME = "kimi_beam_report.json"
COMPATIBILITY_GLOB = "results_kimi_beam_*.json"
ANSWERS_FILENAME = "samples_kimi_beam_1m_answers.jsonl"
SCORES_FILENAME = "samples_kimi_beam_1m_scores.jsonl"
TOTAL_QUESTIONS = 700
LFS_POINTER_VERSION = b"version https://git-lfs.github.com/spec/v1"

TEMPERATURE = "1.0"
TOP_P = "0.95"
MAX_TOKENS = "32768"
# The bundled Kimi-K2.6 tokenizer counts at least 2.7 % fewer tokens than the Kimi K3 engine, so
# a budget sized to the full 1,048,576 window makes the engine reject every question of the
# largest chats with a context-length error.
MAX_CONTEXT_TOKENS = "1000000"
DEFAULT_CONCURRENCY = "16"
DEFAULT_THINKING_JSON = '{"thinking":{"type":"enabled","keep":"all","effort":"max"}}'
JUDGE_TEMPERATURE = "0.3"
JUDGE_CONCURRENCY = "32"


def prepare_compatibility_path(output_dir: Path) -> Path:
    """Remove stale projections and return a timestamped collector artifact path."""
    for stale_path in output_dir.glob(COMPATIBILITY_GLOB):
        stale_path.unlink()
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S.%f")
    return output_dir / f"results_kimi_beam_{timestamp}.json"


def _positive_int(name: str, value: str) -> int:
    """Parse ``value`` as a positive decimal integer named ``name`` in errors."""
    if not value.isdecimal() or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def settings(environ: Mapping[str, str]) -> dict[str, Any]:
    """The caller-adjustable inputs: API keys, question limit, concurrency and thinking body.

    The sampling, context budget and judge settings are fixed by this adapter.
    """
    api_key = environ.get("BEAM_API_KEY") or "EMPTY"
    limit = environ.get("EVAL_LIMIT", "")
    return {
        "api_key": api_key,
        "judge_api_key": environ.get("JUDGE_API_KEY") or api_key,
        "limit": _positive_int("EVAL_LIMIT", limit) if limit else None,
        "concurrency": _positive_int(
            "BEAM_CONCURRENCY", environ.get("BEAM_CONCURRENCY") or DEFAULT_CONCURRENCY
        ),
        "thinking_json": environ.get("BEAM_THINKING_JSON", DEFAULT_THINKING_JSON),
    }


def build_generate_command(
    *,
    python: str,
    beam_dir: Path,
    base_url: str,
    model: str,
    answers_path: Path,
    concurrency: int,
    thinking_json: str,
    limit: int | None,
) -> list[str]:
    """``beam_generate.py`` with the fixed protocol; the API key travels in ``BEAM_API_KEY``."""
    command = [
        python,
        str(beam_dir / "beam_generate.py"),
        "--model",
        model,
        "--base-url",
        base_url,
        "--output",
        str(answers_path),
        "--chats-file",
        str(beam_dir / "data" / "beam_1m_chats.jsonl.gz"),
        "--questions-file",
        str(beam_dir / "data" / "beam_1m_questions.jsonl"),
        "--tokenizer",
        str(beam_dir / "tokenizer"),
        "--concurrency",
        str(concurrency),
        "--temperature",
        TEMPERATURE,
        "--top-p",
        TOP_P,
        "--max-tokens",
        MAX_TOKENS,
        "--max-context-tokens",
        MAX_CONTEXT_TOKENS,
        "--thinking-json",
        thinking_json,
    ]
    if limit is not None:
        command += ["--limit", str(limit)]
    return command


def build_judge_command(
    *,
    python: str,
    beam_dir: Path,
    base_url: str,
    model: str,
    answers_path: Path,
    scores_path: Path,
) -> list[str]:
    """``beam_judge.py`` judging with the model under test; the key travels in ``JUDGE_API_KEY``."""
    return [
        python,
        str(beam_dir / "beam_judge.py"),
        "--answers",
        str(answers_path),
        "--output",
        str(scores_path),
        "--judge-model",
        model,
        "--judge-base-url",
        base_url,
        "--judge-temperature",
        JUDGE_TEMPERATURE,
        "--concurrency",
        JUDGE_CONCURRENCY,
    ]


def check_beam_checkout(beam_dir: Path) -> None:
    """Raise ValueError unless the scripts, tokenizer and resolved data files are present."""
    for name in ("beam_generate.py", "beam_judge.py", "tokenizer/tokenizer_config.json"):
        if not (beam_dir / name).is_file():
            raise ValueError(f"verifier checkout lacks beam/{name}")
    for name in ("data/beam_1m_chats.jsonl.gz", "data/beam_1m_questions.jsonl"):
        path = beam_dir / name
        if not path.is_file():
            raise ValueError(f"verifier checkout lacks beam/{name}")
        with path.open("rb") as handle:
            if handle.read(len(LFS_POINTER_VERSION)) == LFS_POINTER_VERSION:
                raise ValueError(f"beam/{name} is a git-lfs pointer, not the data file")


def _row_key(row: Mapping[str, Any]) -> str:
    """The identity shared by an answer and its score."""
    return f"{row.get('chat_id')}|{row.get('question_type')}|{row.get('question_index')}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Every JSON object line of ``path``; a non-object line raises ValueError."""
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path.name} line {number} is not a JSON object")
            rows.append(row)
    return rows


def project(
    model: str,
    answers: Sequence[Mapping[str, Any]],
    scores: Sequence[Mapping[str, Any]],
    *,
    expected: int,
    limit: int | None,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Reduce the harness outputs to the native report and the collector result.

    Returns ``(native_report, compatibility_result, completed)``. A run is complete only when
    every expected question was answered without exhausting its retries and every answer was
    judged with a finite score in [0, 1]; otherwise the result is a zero-score integration error.
    """
    errored = [row for row in answers if row.get("error")]
    answered_keys = {_row_key(row) for row in answers}
    types_by_key = {_row_key(row): str(row.get("question_type")) for row in answers}
    valid: dict[str, float] = {}
    invalid = 0
    for row in scores:
        key = _row_key(row)
        score = row.get("score")
        if (
            key not in answered_keys
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0.0 <= score <= 1.0
        ):
            invalid += 1
            continue
        valid[key] = float(score)
    problems = []
    if len(answers) != expected:
        problems.append(f"{len(answers)} of {expected} questions answered")
    if errored:
        problems.append(f"{len(errored)} questions failed every retry")
    if invalid:
        problems.append(f"{invalid} scores are malformed or orphaned")
    if len(valid) != len(answers):
        problems.append(f"{len(valid)} of {len(answers)} answers judged")
    completed = not problems
    accuracy = sum(valid.values()) / len(valid) if valid else 0.0
    per_type: dict[str, list[float]] = {}
    for key, score in valid.items():
        per_type.setdefault(types_by_key[key], []).append(score)
    native = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": model,
        "task": TASK_NAME,
        "completed": completed,
        "limit": limit,
        "summary": {
            "expected": expected,
            "answered": len(answers),
            "errored": len(errored),
            "judged": len(valid),
            "accuracy": accuracy,
            "accuracy_by_question_type": {
                kind: sum(values) / len(values) for kind, values in sorted(per_type.items())
            },
        },
        "config": {
            "temperature": float(TEMPERATURE),
            "top_p": float(TOP_P),
            "max_tokens": int(MAX_TOKENS),
            "max_context_tokens": int(MAX_CONTEXT_TOKENS),
            "judge": "self",
            "judge_temperature": float(JUDGE_TEMPERATURE),
        },
        "integration_error": None,
    }
    if completed:
        return (
            native,
            _compatibility_result(model, accuracy, n_samples=len(valid), limit=limit),
            True,
        )
    error = RuntimeError("; ".join(problems))
    native["integration_error"] = {"type": type(error).__name__, "message": str(error)}
    return (
        native,
        _compatibility_result(model, 0.0, n_samples=0, limit=limit, integration_error=error),
        False,
    )


def _compatibility_result(
    model: str,
    score: float,
    *,
    n_samples: int,
    limit: int | None,
    integration_error: BaseException | None = None,
) -> dict[str, Any]:
    """The lm-eval style result for ``model`` with ``score`` over ``n_samples`` judged questions."""
    result: dict[str, Any] = {
        "result_format": RESULT_FORMAT,
        "eval_adapter": ADAPTER_NAME,
        "model_name": model,
        "results": {
            TASK_NAME: {
                "exact_match,strict-match": score,
                "exact_match_stderr,strict-match": 0.0,
            }
        },
        "configs": {
            TASK_NAME: {
                "metric_list": [{"metric": "exact_match"}],
                "filter_list": [{"name": "strict-match"}],
            }
        },
        "n-samples": {TASK_NAME: {"original": TOTAL_QUESTIONS, "effective": n_samples}},
    }
    if limit is not None:
        result["limit"] = limit
    if integration_error is not None:
        result["integration_error"] = {
            "type": type(integration_error).__name__,
            "message": str(integration_error),
        }
    return result


def _write_native_failure(path: Path, *, model: str, error: BaseException) -> None:
    """Write a native diagnostic envelope when the scripts cannot produce one."""
    _write_json(
        path,
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "model": model,
            "task": TASK_NAME,
            "completed": False,
            "summary": {"expected": TOTAL_QUESTIONS, "answered": 0, "errored": 0, "judged": 0},
            "integration_error": {"type": type(error).__name__, "message": str(error)},
        },
    )


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    """Write ``document`` to ``path`` as indented JSON."""
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def _run(command: Sequence[str], *, cwd: Path, env: Mapping[str, str]) -> None:
    """Run one harness script to completion; a non-zero exit raises RuntimeError."""
    print("+ " + " ".join(command), file=sys.stderr, flush=True)
    returncode = subprocess.run(command, cwd=cwd, env=env, check=False).returncode
    if returncode:
        raise RuntimeError(f"{Path(command[1]).name} exited with code {returncode}")


def run_evaluation(
    *,
    verifier_dir: Path,
    base_url: str,
    model: str,
    output_dir: Path,
    environ: Mapping[str, str],
    python: str = sys.executable,
) -> bool:
    """Generate and judge every question, then write the report and the collector result."""
    output_dir.mkdir(parents=True, exist_ok=True)
    native_report = output_dir / NATIVE_REPORT_FILENAME
    compatibility_path = prepare_compatibility_path(output_dir)
    answers_path = output_dir / ANSWERS_FILENAME
    scores_path = output_dir / SCORES_FILENAME
    for stale in (native_report, answers_path, scores_path):
        stale.unlink(missing_ok=True)
    limit: int | None = None
    native: dict[str, Any] | None = None
    compatibility = _compatibility_result(model, 0.0, n_samples=0, limit=None)
    completed = False
    try:
        inputs = settings(environ)
        limit = inputs["limit"]
        beam_dir = verifier_dir / "beam"
        check_beam_checkout(beam_dir)
        expected = min(limit, TOTAL_QUESTIONS) if limit else TOTAL_QUESTIONS
        env = {
            **environ,
            "BEAM_API_KEY": inputs["api_key"],
            "JUDGE_API_KEY": inputs["judge_api_key"],
        }
        _run(
            build_generate_command(
                python=python,
                beam_dir=beam_dir,
                base_url=base_url,
                model=model,
                answers_path=answers_path,
                concurrency=inputs["concurrency"],
                thinking_json=inputs["thinking_json"],
                limit=limit,
            ),
            cwd=beam_dir,
            env=env,
        )
        _run(
            build_judge_command(
                python=python,
                beam_dir=beam_dir,
                base_url=base_url,
                model=model,
                answers_path=answers_path,
                scores_path=scores_path,
            ),
            cwd=beam_dir,
            env=env,
        )
        native, compatibility, completed = project(
            model,
            read_jsonl(answers_path),
            read_jsonl(scores_path),
            expected=expected,
            limit=limit,
        )
    except (OSError, ValueError, RuntimeError) as error:
        compatibility = _compatibility_result(
            model, 0.0, n_samples=0, limit=limit, integration_error=error
        )
        if native is None:
            _write_native_failure(native_report, model=model, error=error)
    finally:
        if native is not None:
            _write_json(native_report, native)
        _write_json(compatibility_path, compatibility)
    return completed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments; ``--verifier-dir`` and ``--base-url`` are required unless failing."""
    parser = argparse.ArgumentParser(
        description="Run the pinned stock Kimi Vendor Verifier BEAM (1M) evaluation."
    )
    parser.add_argument("--verifier-dir", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-prefix", default="")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--task-name", choices=SUPPORTED_TASK_NAMES, default=TASK_NAME)
    parser.add_argument("--integration-error")
    args = parser.parse_args(argv)
    if args.integration_error is None:
        missing = [
            option
            for option, value in (
                ("--verifier-dir", args.verifier_dir),
                ("--base-url", args.base_url),
            )
            if value is None
        ]
        if missing:
            parser.error(f"{', '.join(missing)} required unless --integration-error is provided")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run the BEAM eval, or record the given integration error; 0 on pass, 1 on failure."""
    args = parse_args(argv)
    if args.integration_error is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        error = RuntimeError(args.integration_error)
        _write_native_failure(
            args.output_dir / NATIVE_REPORT_FILENAME, model=args.model, error=error
        )
        _write_json(
            prepare_compatibility_path(args.output_dir),
            _compatibility_result(
                args.model, 0.0, n_samples=0, limit=None, integration_error=error
            ),
        )
        return 0
    passed = run_evaluation(
        verifier_dir=args.verifier_dir,
        base_url=args.base_url,
        model=args.model,
        output_dir=args.output_dir,
        environ=os.environ,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
