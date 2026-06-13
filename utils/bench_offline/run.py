#!/usr/bin/env python3
"""Tune and run the branch-only B300 TensorRT-LLM offline benchmark."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from io_utils import read_json, tail_text, write_json
from metrics import huawei_comparison
from prompts import INFINITEBENCH_REVISION, prepare_corpus, sha256_file
from trt_config import (
    FINAL_MEASURED_PASSES,
    INPUT_TOKENS,
    MAX_SEQ_LEN,
    MIN_WINNER_IMPROVEMENT,
    MTP_DRAFT_TOKENS,
    OUTPUT_TOKENS,
    PINNED_TRT_GLOBAL_SEED,
    SAMPLING_TEMPERATURE,
    SAMPLING_TOP_K,
    SAMPLING_TOP_P,
    TUNING_MEASURED_PASSES,
    WORLD_SIZE,
    CandidateConfig,
    balance_candidate,
    choose_winner,
    overlap_candidate,
    scheduler_candidates,
)


ALLOWED_CONCURRENCIES = (8, 32, 64, 128, 256, 512, 1024)
TRT_SOURCE_COMMIT = "c185066"
DEFAULT_IMAGE = (
    "ghcr.io#semianalysisai/"
    "trtllm-deepseek-v4:feat-deepseek_v4-c185066"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str | None:
    explicit_revision = (
        os.getenv("TRT_BENCH_GIT_REVISION") or os.getenv("GITHUB_SHA")
    )
    if explicit_revision:
        return explicit_revision
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def model_manifest(model_path: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        path = model_path / name
        if path.is_file():
            files[name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return {
        "path": str(model_path),
        "identity_files": files,
    }


def classify_failure(
    worker_result: dict[str, Any],
    log_text: str,
    timed_out: bool = False,
) -> str:
    if timed_out:
        return "timeout"
    text = "\n".join(
        [
            str(worker_result.get("error", "")),
            str(worker_result.get("traceback", "")),
            log_text,
        ]
    ).lower()
    graph_signatures = (
        "cuda graph",
        "cudagraph",
        "graph capture",
        "capture_begin",
        "cuda_error_illegal_address",
        "illegal memory access",
    )
    if any(signature in text for signature in graph_signatures):
        return "cuda_graph"
    oom_signatures = (
        "out of memory",
        "cuda_error_out_of_memory",
        "cuda error: out of memory",
        "std::bad_alloc",
        "cublas_status_alloc_failed",
    )
    if any(signature in text for signature in oom_signatures):
        return "oom"
    capacity_signatures = (
        "exceeds max_batch_size",
        "exceeds max_num_tokens",
        "exceeds max_seq_len",
        "kv cache",
        "no available memory",
        "insufficient memory",
        "cannot allocate",
    )
    if any(signature in text for signature in capacity_signatures):
        return "capacity"
    if "perfect-router" in text or "perfect_router" in text:
        return "perfect_router"
    if worker_result.get("phase") == "engine_init":
        return "initialization"
    return "runtime"


def terminate_process_group(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def compact_attempt(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: result.get(key)
        for key in (
            "attempt",
            "status",
            "mode",
            "candidate",
            "failure_kind",
            "phase",
            "error_type",
            "error",
            "return_code",
            "timed_out",
            "elapsed_seconds",
            "engine_init_seconds",
            "aggregate",
            "resolved_parallelism",
            "worker_output",
            "worker_log",
        )
        if result.get(key) is not None
    }


def run_worker(
    *,
    output_dir: Path,
    model_path: Path,
    corpus_path: Path,
    manifest_path: Path,
    candidate: CandidateConfig,
    attempt: int,
    mode: str,
    passes: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    label = f"{mode}_{attempt:02d}_{candidate.name}"
    candidate_path = output_dir / f"{label}_candidate.json"
    worker_output = output_dir / f"{label}_worker.json"
    worker_log = output_dir / f"{label}.log"
    marker_path = output_dir / f"{label}_perfect_router.jsonl"
    write_json(candidate_path, candidate.to_dict())
    marker_path.unlink(missing_ok=True)

    worker_script = Path(__file__).with_name("trt_worker.py")
    command = [
        sys.executable,
        str(worker_script),
        "--model-path",
        str(model_path),
        "--corpus",
        str(corpus_path),
        "--manifest",
        str(manifest_path),
        "--candidate",
        str(candidate_path),
        "--output",
        str(worker_output),
        "--mode",
        mode,
        "--passes",
        str(passes),
    ]
    environment = os.environ.copy()
    bench_dir = str(Path(__file__).resolve().parent)
    environment["PYTHONPATH"] = os.pathsep.join(
        item
        for item in (bench_dir, environment.get("PYTHONPATH"))
        if item
    )
    environment["ENABLE_PERFECT_ROUTER"] = "1"
    environment["TRTLLM_ENABLE_PERFECT_ROUTER"] = "1"
    environment["TRTLLM_PERFECT_ROUTER_MARKER"] = str(marker_path)

    started = time.perf_counter()
    timed_out = False
    with worker_log.open("w", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_process_group(process)
            return_code = process.returncode
    elapsed = time.perf_counter() - started

    if worker_output.exists():
        result = read_json(worker_output)
    else:
        result = {
            "schema_version": 1,
            "status": "failed",
            "phase": "worker_process",
            "error_type": "WorkerProcessError",
            "error": (
                f"Worker timed out after {timeout_seconds}s"
                if timed_out
                else f"Worker exited {return_code} without an output file"
            ),
        }
        write_json(worker_output, result)

    result.update(
        {
            "attempt": attempt,
            "candidate": candidate.to_dict(),
            "return_code": return_code,
            "timed_out": timed_out,
            "elapsed_seconds": elapsed,
            "worker_output": worker_output.name,
            "worker_log": worker_log.name,
        }
    )
    if result.get("status") != "success":
        result["failure_kind"] = classify_failure(
            result,
            tail_text(worker_log),
            timed_out=timed_out,
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-revision", default=INFINITEBENCH_REVISION)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-timeout", type=int, default=3600)
    parser.add_argument("--tuning-attempts", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    base_result: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": utc_now(),
        "benchmark": {
            "mode": "offline_in_process",
            "engine": "TensorRT-LLM PyTorch backend",
            "hardware": "B300",
            "world_size": WORLD_SIZE,
            "effective_parallelism": "DEP8",
            "input_tokens": INPUT_TOKENS,
            "generated_output_tokens": OUTPUT_TOKENS,
            "decode_tokens_per_request": OUTPUT_TOKENS - 1,
            "mtp_max_draft_len": MTP_DRAFT_TOKENS,
            "max_seq_len": MAX_SEQ_LEN,
            "concurrency": args.concurrency,
            "warmup_batches": 1,
            "tuning_measured_passes": TUNING_MEASURED_PASSES,
            "final_measured_passes": FINAL_MEASURED_PASSES,
            "sampling": {
                "temperature": SAMPLING_TEMPERATURE,
                "top_p": SAMPLING_TOP_P,
                "top_k": SAMPLING_TOP_K,
                "engine_global_seed": PINNED_TRT_GLOBAL_SEED,
                "seed_semantics": (
                    "Pinned TRT PyTorch sampler advances one engine-global "
                    "seed-42 generator and does not apply request-level "
                    "SamplingParams.seed"
                ),
            },
            "headline_tpot": (
                "arithmetic mean of each request's "
                "(last_token-first_token)/(output_tokens-1)"
            ),
            "derived_output_tput_per_gpu": (
                "concurrency / mean_token_tpot_seconds / 8"
            ),
            "wall_output_tput_per_gpu": (
                "all generated output tokens / measured wall seconds / 8"
            ),
            "huawei_conversion": (
                "published step throughput/chip multiplied by TRT observed "
                "output tokens/decode iteration; raw acceptance rate is not "
                "the multiplier"
            ),
            "effective_acceptance_rate": (
                "(observed output tokens/decode iteration - 1) / "
                "mtp_max_draft_len"
            ),
            "raw_acceptance_rate": (
                "TRT accepted/proposed draft counters when populated; null "
                "when the pinned PyTorch MTP path reports zero proposals"
            ),
        },
        "provenance": {
            "git_revision": git_revision(),
            "image": os.getenv("IMAGE", DEFAULT_IMAGE),
            "trt_source_commit": TRT_SOURCE_COMMIT,
            "model": model_manifest(args.model_path),
            "slurm_job_id": os.getenv("TRT_BENCH_SLURM_JOB_ID"),
            "slurm_node": os.getenv("TRT_BENCH_SLURM_NODE"),
        },
        "tuning": {
            "attempt_limit": args.tuning_attempts,
            "minimum_winner_improvement": MIN_WINNER_IMPROVEMENT,
            "attempts": [],
        },
    }
    write_json(result_path, base_result)

    try:
        if args.concurrency not in ALLOWED_CONCURRENCIES:
            raise ValueError(
                f"Concurrency must be one of {ALLOWED_CONCURRENCIES}, "
                f"got {args.concurrency}"
            )
        if not 1 <= args.tuning_attempts <= 6:
            raise ValueError("--tuning-attempts must be between 1 and 6")
        if not args.model_path.is_dir():
            raise FileNotFoundError(
                f"Model path does not exist: {args.model_path}"
            )
        if not args.dataset.is_file():
            raise FileNotFoundError(f"Dataset does not exist: {args.dataset}")

        corpus_manifest = prepare_corpus(
            args.dataset,
            str(args.model_path),
            args.concurrency,
            args.output_dir,
            dataset_revision=args.dataset_revision,
        )
        base_result["corpus"] = corpus_manifest
        write_json(result_path, base_result)

        corpus_path = args.output_dir / "corpus.bin"
        manifest_path = args.output_dir / "corpus_manifest.json"
        full_attempts: list[dict[str, Any]] = []
        scheduler_results: list[dict[str, Any]] = []
        force_graph_off = False
        stop_for_capacity = False
        fatal_failure: dict[str, Any] | None = None

        def execute(base_candidate: CandidateConfig) -> dict[str, Any] | None:
            nonlocal force_graph_off, stop_for_capacity, fatal_failure
            if len(full_attempts) >= args.tuning_attempts:
                return None
            candidate = (
                base_candidate.without_cuda_graph()
                if force_graph_off and base_candidate.cuda_graph
                else base_candidate
            )
            result = run_worker(
                output_dir=args.output_dir,
                model_path=args.model_path,
                corpus_path=corpus_path,
                manifest_path=manifest_path,
                candidate=candidate,
                attempt=len(full_attempts) + 1,
                mode="tune",
                passes=TUNING_MEASURED_PASSES,
                timeout_seconds=args.worker_timeout,
            )
            full_attempts.append(result)
            base_result["tuning"]["attempts"] = [
                compact_attempt(item) for item in full_attempts
            ]
            write_json(result_path, base_result)

            if (
                result.get("failure_kind") in {"cuda_graph", "oom"}
                and candidate.cuda_graph
                and len(full_attempts) < args.tuning_attempts
            ):
                force_graph_off = True
                fallback = run_worker(
                    output_dir=args.output_dir,
                    model_path=args.model_path,
                    corpus_path=corpus_path,
                    manifest_path=manifest_path,
                    candidate=candidate.without_cuda_graph(),
                    attempt=len(full_attempts) + 1,
                    mode="tune",
                    passes=TUNING_MEASURED_PASSES,
                    timeout_seconds=args.worker_timeout,
                )
                full_attempts.append(fallback)
                base_result["tuning"]["attempts"] = [
                    compact_attempt(item) for item in full_attempts
                ]
                write_json(result_path, base_result)
                result = fallback

            if (
                result.get("status") != "success"
                and result.get("failure_kind") in {"oom", "capacity"}
                and result.get("phase") in {"engine_init", "warmup"}
            ):
                stop_for_capacity = True
            if result.get("failure_kind") == "timeout":
                fatal_failure = result
            return result

        for candidate in scheduler_candidates():
            if (
                stop_for_capacity
                or fatal_failure is not None
                or len(full_attempts) >= args.tuning_attempts
            ):
                break
            scheduler_result = execute(candidate)
            if scheduler_result is not None:
                scheduler_results.append(scheduler_result)

        scheduler_winner = choose_winner(scheduler_results)
        if (
            scheduler_winner is not None
            and not stop_for_capacity
            and fatal_failure is None
            and len(full_attempts) < args.tuning_attempts
        ):
            execute(
                balance_candidate(
                    CandidateConfig.from_dict(scheduler_winner["candidate"])
                )
            )

        current_winner = choose_winner(full_attempts)
        if (
            current_winner is not None
            and not stop_for_capacity
            and fatal_failure is None
            and len(full_attempts) < args.tuning_attempts
        ):
            execute(
                overlap_candidate(
                    CandidateConfig.from_dict(current_winner["candidate"])
                )
            )

        winner = choose_winner(full_attempts)
        base_result["tuning"]["attempts"] = [
            compact_attempt(item) for item in full_attempts
        ]
        if fatal_failure is not None:
            base_result.update(
                {
                    "status": "failed",
                    "finished_at": utc_now(),
                    "error": "A TRT worker timed out; node state is not reusable",
                    "fatal_attempt": compact_attempt(fatal_failure),
                }
            )
            write_json(result_path, base_result)
            return 1
        if winner is None:
            failure_kinds = sorted(
                {
                    str(item.get("failure_kind"))
                    for item in full_attempts
                    if item.get("failure_kind")
                }
            )
            is_capacity = (
                args.concurrency >= 512
                and bool({"oom", "capacity"} & set(failure_kinds))
            )
            base_result.update(
                {
                    "status": (
                        "capacity_failure" if is_capacity else "failed"
                    ),
                    "finished_at": utc_now(),
                    "failure_kinds": failure_kinds,
                    "error": "No tuning candidate completed successfully",
                }
            )
            write_json(result_path, base_result)
            return 0 if is_capacity else 1

        winner_candidate = CandidateConfig.from_dict(winner["candidate"])
        base_result["tuning"]["winner"] = {
            "candidate": winner_candidate.to_dict(),
            "aggregate": winner["aggregate"],
            "attempt": winner["attempt"],
        }
        base_result["status"] = "final_measurement"
        write_json(result_path, base_result)

        final_result = run_worker(
            output_dir=args.output_dir,
            model_path=args.model_path,
            corpus_path=corpus_path,
            manifest_path=manifest_path,
            candidate=winner_candidate,
            attempt=1,
            mode="final",
            passes=FINAL_MEASURED_PASSES,
            timeout_seconds=args.worker_timeout,
        )
        if final_result.get("status") != "success":
            base_result.update(
                {
                    "status": "failed",
                    "finished_at": utc_now(),
                    "error": "Fresh final engine measurement failed",
                    "final": compact_attempt(final_result),
                }
            )
            write_json(result_path, base_result)
            return 1

        aggregate = final_result["aggregate"]
        base_result.update(
            {
                "status": "success",
                "finished_at": utc_now(),
                "winner": winner_candidate.to_dict(),
                "final": final_result,
                "aggregate": aggregate,
                "huawei": huawei_comparison(
                    args.concurrency,
                    float(aggregate["derived_output_tput_per_gpu"]),
                    float(aggregate["observed_tokens_per_step"]),
                ),
            }
        )
        write_json(result_path, base_result)
        return 0
    except BaseException as error:
        base_result.update(
            {
                "status": "failed",
                "finished_at": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        write_json(result_path, base_result)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
