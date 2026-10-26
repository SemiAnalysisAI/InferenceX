"""AA-LCR generation adapter: answer every question against the local endpoint.

Grading needs an LLM judge, so this adapter only generates. ``aa_lcr_judge.py``
grades ``samples_aa_lcr.jsonl`` on a GitHub-hosted runner, which keeps the
judge's API key off GPU runners. The published ``score`` is the fraction of
requests that returned a non-empty final answer, not accuracy.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import os
import sys
import time
import unicodedata
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RESULT_FORMAT = "inferencex-eval-v1"
TASK_NAME = "aa_lcr_generation"
REVISION = "9a77ef56b717057ade24ceab4d273712a0b4f19e"
BASE = f"https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR/resolve/{REVISION}"
FILES = {
    "AA-LCR_Dataset.csv": "aea5198982436fa774f616964bbeb34716de9e94368a123275b9355765c9b73b",
    "extracted_text/AA-LCR_extracted-text.zip": (
        "5e839249826f6b9bd5324f0d139089c9dc481ccb3f212a6dfad00c51045d9d8a"
    ),
}
PROMPT = """BEGIN INPUT DOCUMENTS

{documents_text}

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

{question}

END QUESTION
"""
REQUEST_TIMEOUT_S = 3600


def _fetch(name: str) -> bytes:
    """Download one pinned dataset file and verify its SHA-256."""
    request = urllib.request.Request(f"{BASE}/{name}")
    if token := os.environ.get("HF_TOKEN"):
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=300) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != FILES[name]:
        raise RuntimeError(f"{name} sha256 {digest} does not match the pinned {FILES[name]}")
    return data


def load_questions() -> list[dict[str, Any]]:
    """The 100 questions with their prompts, documents in ``data_source_filenames`` order."""
    rows = list(csv.DictReader(io.StringIO(_fetch("AA-LCR_Dataset.csv").decode("utf-8"))))
    archive = zipfile.ZipFile(io.BytesIO(_fetch("extracted_text/AA-LCR_extracted-text.zip")))
    # The archive stores UTF-8 names without the UTF-8 flag (so zipfile decodes
    # them as cp437), and some names are NFD while the CSV is NFC.
    members = {
        unicodedata.normalize(
            "NFC", info.filename if info.flag_bits & 0x800 else info.filename.encode("cp437").decode()
        ): info
        for info in archive.infolist()
    }
    questions = []
    for row in rows:
        folder = f"lcr/{row['document_category']}/{row['document_set_id']}"
        docs = [
            archive.read(members[unicodedata.normalize("NFC", f"{folder}/{name}")]).decode("utf-8")
            for name in row["data_source_filenames"].split(";")
        ]
        documents_text = "\n\n".join(
            f"BEGIN DOCUMENT {i + 1}:\n{doc}\nEND DOCUMENT {i + 1}" for i, doc in enumerate(docs)
        )
        questions.append({
            "question_id": row["question_id"],
            "document_set_id": row["document_set_id"],
            "question": row["question"],
            "answer": row["answer"],
            "prompt": PROMPT.format(documents_text=documents_text, question=row["question"]),
        })  # fmt: skip
    if len(questions) != 100:
        raise RuntimeError(f"expected 100 AA-LCR questions, found {len(questions)}")
    return questions


def _ask(base_url: str, model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    """One chat completion; returns the answer, reasoning size, and usage, or the error."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }).encode()  # fmt: skip
    request = urllib.request.Request(
        f"{base_url}/chat/completions", body, {"Content-Type": "application/json"}
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
        return {"error": f"{type(error).__name__}: {error}", "latency_s": time.monotonic() - started}
    choice = payload["choices"][0]
    message = choice.get("message") or {}
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    return {
        "response": message.get("content") or "",
        "reasoning_chars": len(reasoning),
        "finish_reason": choice.get("finish_reason"),
        "usage": payload.get("usage"),
        "latency_s": time.monotonic() - started,
    }


def _result(model: str, score: float, effective: int, error: str | None = None) -> dict[str, Any]:
    """The lm-eval style compatibility result."""
    result: dict[str, Any] = {
        "result_format": RESULT_FORMAT,
        "eval_adapter": "aa_lcr",
        "model_name": model,
        "results": {
            TASK_NAME: {"exact_match,strict-match": score, "exact_match_stderr,strict-match": 0.0}
        },
        "configs": {
            TASK_NAME: {
                "metric_list": [{"metric": "exact_match"}],
                "filter_list": [{"name": "strict-match"}],
            }
        },
        "n-samples": {TASK_NAME: {"original": effective, "effective": effective}},
        "source": {"repository": "ArtificialAnalysis/AA-LCR", "ref": REVISION},
    }
    if error is not None:
        result["integration_error"] = {"type": "aa_lcr", "message": error}
    return result


def _write(output_dir: Path, result: dict[str, Any], prefix: str = "aa_lcr") -> None:
    """Write the compatibility result."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S.%f")
    path = output_dir / f"results_{prefix}_{stamp}.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    """Generate ``--repeats`` answers per question and write samples plus the result."""
    questions = load_questions()
    jobs = [(q, r) for r in range(args.repeats) for q in questions]
    samples_path = args.output_dir / "samples_aa_lcr.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    answered = 0
    with (
        samples_path.open("w", encoding="utf-8") as samples,
        concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool,
    ):
        futures = {
            pool.submit(_ask, args.base_url, args.model, q["prompt"], args.max_tokens): (q, r)
            for q, r in jobs
        }
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            question, repeat = futures[future]
            outcome = future.result()
            answered += bool(outcome.get("response", "").strip())
            record = {k: v for k, v in question.items() if k != "prompt"}
            samples.write(json.dumps({**record, "repeat": repeat, **outcome}) + "\n")
            samples.flush()
            print(f"aa-lcr {done}/{len(jobs)} answered={answered}", flush=True)
    _write(args.output_dir, _result(args.model, answered / len(jobs), len(jobs)))
    return 0


def failure(args: argparse.Namespace) -> int:
    """Record an integration failure."""
    _write(args.output_dir, _result(args.model, 0.0, 0, args.message))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse ``run`` or ``failure`` and dispatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--base-url", required=True)
    run_parser.add_argument("--concurrency", type=int, required=True)
    run_parser.add_argument("--repeats", type=int, default=int(os.environ.get("AA_LCR_REPEATS", "3")))
    run_parser.add_argument(
        "--max-tokens", type=int, default=int(os.environ.get("AA_LCR_MAX_TOKENS", "32768"))
    )
    failure_parser = commands.add_parser("failure")
    failure_parser.add_argument("--message", required=True)
    for sub in (run_parser, failure_parser):
        sub.add_argument("--model", required=True)
        sub.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    return run(args) if args.command == "run" else failure(args)


if __name__ == "__main__":
    sys.exit(main())
