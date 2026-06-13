#!/usr/bin/env python3
"""Tune and run the branch-only B300 TensorRT-LLM offline benchmark."""

from __future__ import annotations

import argparse
import json
import os
import re
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
    CANDIDATE_ENVIRONMENT_VARIABLES,
    CandidateConfig,
    balance_candidate,
    candidate_environment,
    choose_winner,
    overlap_candidate,
    scheduler_candidates,
)


ALLOWED_CONCURRENCIES = (4, 8, 16, 32, 64, 128, 256, 512, 1024)
TRT_SOURCE_COMMIT = "c185066"
DEFAULT_IMAGE = (
    "ghcr.io#semianalysisai/"
    "trtllm-deepseek-v4:feat-deepseek_v4-c185066"
)
PROGRESS_INTERVAL_SECONDS = 60.0
WORKER_PROGRESS_PREFIX = "[offline-trt-worker "


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_progress(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[offline-trt-controller {timestamp}] {message}", flush=True)


def aggregate_progress(aggregate: dict[str, Any]) -> str:
    fields: list[str] = []
    metrics = (
        ("mean_token_tpot_ms", "mean_tpot", ".3f", "ms"),
        (
            "derived_output_tput_per_gpu",
            "derived_output_tput_per_gpu",
            ".2f",
            "",
        ),
        (
            "derived_step_tput_per_gpu",
            "derived_step_tput_per_gpu",
            ".2f",
            "",
        ),
        ("observed_tokens_per_step", "tokens_per_step", ".3f", ""),
    )
    for key, label, format_spec, suffix in metrics:
        value = aggregate.get(key)
        if value is not None:
            fields.append(f"{label}={format(float(value), format_spec)}{suffix}")
    return " ".join(fields) or "metrics=unavailable"


def latest_worker_progress(worker_log: Path) -> str | None:
    for line in reversed(tail_text(worker_log, max_bytes=64_000).splitlines()):
        if line.startswith(WORKER_PROGRESS_PREFIX):
            _, separator, message = line.partition("] ")
            return message if separator else line
    return None


def wait_for_worker_process(
    process: subprocess.Popen[Any],
    *,
    label: str,
    worker_log: Path,
    started: float,
    timeout_seconds: int,
    heartbeat_seconds: float = PROGRESS_INTERVAL_SECONDS,
) -> int:
    while True:
        elapsed = time.perf_counter() - started
        remaining = timeout_seconds - elapsed
        if remaining <= 0:
            raise subprocess.TimeoutExpired(
                getattr(process, "args", label),
                timeout_seconds,
            )
        try:
            return process.wait(timeout=min(heartbeat_seconds, remaining))
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - started
            if elapsed >= timeout_seconds:
                raise subprocess.TimeoutExpired(
                    getattr(process, "args", label),
                    timeout_seconds,
                ) from None
            latest = latest_worker_progress(worker_log)
            detail = (
                f"; last_worker_progress={latest}"
                if latest is not None
                else ""
            )
            log_progress(
                f"{label}: still running elapsed={elapsed:.0f}s "
                f"timeout={timeout_seconds}s{detail}; "
                f"worker_log={worker_log.name}"
            )


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


def profile_trace_base(output_dir: Path, label: str) -> Path:
    return output_dir / f"{label}_torch_profile.json"


def collect_profile_artifacts(
    *,
    output_dir: Path,
    label: str,
    candidate: CandidateConfig,
) -> dict[str, Any] | None:
    if candidate.profile_iterations is None:
        return None
    trace_base = profile_trace_base(output_dir, label)
    traces = sorted(
        output_dir.glob(
            f"{trace_base.stem}-rank-*{trace_base.suffix}"
        )
    )
    files = []
    ranks = []
    for path in traces:
        match = re.search(r"-rank-(\d+)\.json$", path.name)
        rank = int(match.group(1)) if match else None
        if rank is not None:
            ranks.append(rank)
        files.append(
            {
                "path": path.name,
                "rank": rank,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    expected_ranks = list(range(candidate.active_gpu_count))
    manifest = {
        "profile_iterations": candidate.profile_iterations,
        "trace_base": trace_base.name,
        "expected_ranks": expected_ranks,
        "observed_ranks": sorted(ranks),
        "complete": (
            sorted(ranks) == expected_ranks
            and all(item["bytes"] > 0 for item in files)
        ),
        "files": files,
    }
    manifest_path = output_dir / f"{label}_profile_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = manifest_path.name
    return manifest


def classify_failure(
    worker_result: dict[str, Any],
    log_text: str,
    timed_out: bool = False,
) -> str:
    if timed_out:
        return "timeout"
    if worker_result.get("phase") == "profile_collection":
        return "profile"
    text = "\n".join(
        [
            str(worker_result.get("error", "")),
            str(worker_result.get("traceback", "")),
            log_text,
        ]
    ).lower()
    kernel_dtype_signatures = (
        "paged mqa logits dtype errors",
        "q must be float8_e4m3fn",
    )
    if any(signature in text for signature in kernel_dtype_signatures):
        return "kernel_dtype"
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
            "runtime_environment",
            "profile",
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
    for name in CANDIDATE_ENVIRONMENT_VARIABLES:
        environment.pop(name, None)
    for name in (
        "TLLM_PROFILE_LOG_RANKS",
        "TLLM_PROFILE_START_STOP",
        "TLLM_TORCH_PROFILE_TRACE",
    ):
        environment.pop(name, None)
    configured_environment = candidate_environment(candidate)
    environment.update(configured_environment)
    rank_environment = dict(configured_environment)
    if candidate.profile_iterations is not None:
        trace_base = profile_trace_base(output_dir, label)
        profile_environment = {
            "TLLM_PROFILE_LOG_RANKS": "0",
            "TLLM_PROFILE_START_STOP": candidate.profile_iterations,
            "TLLM_TORCH_PROFILE_TRACE": str(trace_base),
        }
        environment.update(profile_environment)
        rank_environment.update(profile_environment)
    environment["TRTLLM_BENCH_EXPECTED_RANK_ENV"] = json.dumps(
        rank_environment,
        sort_keys=True,
        separators=(",", ":"),
    )

    log_progress(
        f"{label}: worker start passes={passes} "
        f"wait_iters={candidate.batching_wait_iters} "
        f"timeout_iters={candidate.attention_dp_timeout_iters} "
        f"balance={'on' if candidate.attention_dp_balance else 'off'} "
        f"overlap={'on' if candidate.overlap_scheduler else 'off'} "
        f"cuda_graph={'on' if candidate.cuda_graph else 'off'} "
        f"adp_batch_mode={candidate.attention_dp_batch_mode} "
        "lm_head_tp="
        f"{'on' if candidate.enable_lm_head_tp_in_adp else 'off'} "
        "cute_dsl_mqa="
        f"{'on' if candidate.use_cute_dsl_paged_mqa_logits else 'off'} "
        "cute_dsl_topk="
        f"{'on' if candidate.use_cute_dsl_topk else 'off'} "
        "heuristic_topk="
        f"{'on' if candidate.enable_heuristic_topk else 'off'} "
        f"indexer_k_dtype={candidate.indexer_k_dtype or 'checkpoint'} "
        f"moe_backend={candidate.moe_backend} "
        "low_precision_moe_combine="
        f"{'on' if candidate.use_low_precision_moe_combine else 'off'} "
        f"force_moe_comm={candidate.force_moe_comm_method or 'auto'} "
        f"profile_iterations={candidate.profile_iterations or 'off'} "
        f"max_seq_len={candidate.max_seq_len} "
        f"trt_iter_log={'on' if candidate.print_iter_log else 'off'} "
        f"mtp={candidate.mtp_draft_tokens} "
        f"parallelism={candidate.effective_parallelism} "
        f"active_gpus={candidate.active_gpu_count}"
    )
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
            return_code = wait_for_worker_process(
                process,
                label=label,
                worker_log=worker_log,
                started=started,
                timeout_seconds=timeout_seconds,
            )
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
            "runtime_environment": {
                "candidate": configured_environment,
                "rank_expected": rank_environment,
            },
            "worker_output": worker_output.name,
            "worker_log": worker_log.name,
        }
    )
    profile = collect_profile_artifacts(
        output_dir=output_dir,
        label=label,
        candidate=candidate,
    )
    if profile is not None:
        result["profile"] = profile
        if result.get("status") == "success" and not profile["complete"]:
            result.update(
                {
                    "status": "failed",
                    "phase": "profile_collection",
                    "error_type": "ProfileArtifactError",
                    "error": (
                        "Profile traces did not cover every active rank: "
                        f"{profile['observed_ranks']} != "
                        f"{profile['expected_ranks']}"
                    ),
                }
            )
    if result.get("status") != "success":
        result["failure_kind"] = classify_failure(
            result,
            tail_text(worker_log),
            timed_out=timed_out,
        )
        log_progress(
            f"{label}: worker failed elapsed={elapsed:.1f}s "
            f"failure_kind={result['failure_kind']} "
            f"phase={result.get('phase', 'unknown')} "
            f"error={result.get('error', 'unknown')}"
        )
    else:
        aggregate = result.get("aggregate") or {}
        log_progress(
            f"{label}: worker complete elapsed={elapsed:.1f}s "
            f"{aggregate_progress(aggregate)} "
            f"profile_files={len((profile or {}).get('files', []))}"
        )
    write_json(worker_output, result)
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
    parser.add_argument("--experiment-config", type=Path)
    parser.add_argument("--experiment-id")
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
            "execution": (
                "single_candidate"
                if args.experiment_config is not None
                else "serial_tuning"
            ),
            "experiment_id": args.experiment_id,
            "engine": "TensorRT-LLM PyTorch backend",
            "request_path": "direct LLM.generate; no server or HTTP",
            "hardware": "B300",
            "world_size": WORLD_SIZE,
            "active_gpu_count": WORLD_SIZE,
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
            "derived_step_tput_per_gpu": (
                "concurrency / mean_step_tpot_seconds / 8"
            ),
            "wall_output_tput_per_gpu": (
                "all generated output tokens / measured wall seconds / 8"
            ),
            "huawei_conversion": (
                "at exact global batch, compare B300 decode steps per active "
                "GPU with Huawei decode steps per chip; also compare emitted "
                "output using Huawei's published 2.44 tokens/step"
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
    }
    if args.experiment_config is None:
        base_result["tuning"] = {
            "attempt_limit": args.tuning_attempts,
            "minimum_winner_improvement": MIN_WINNER_IMPROVEMENT,
            "attempts": [],
        }
    else:
        base_result["experiment"] = {
            "id": args.experiment_id,
            "config_path": str(args.experiment_config),
            "status": "pending",
        }
    write_json(result_path, base_result)
    log_progress(
        f"benchmark start concurrency={args.concurrency} "
        f"execution={base_result['benchmark']['execution']} "
        f"experiment_id={args.experiment_id or 'none'} "
        f"input_tokens={INPUT_TOKENS} output_tokens={OUTPUT_TOKENS} "
        f"tuning_attempts="
        f"{args.tuning_attempts if args.experiment_config is None else 0} "
        f"worker_timeout={args.worker_timeout}s "
        f"output_dir={args.output_dir}"
    )

    experiment_candidate: CandidateConfig | None = None
    try:
        if args.concurrency not in ALLOWED_CONCURRENCIES:
            raise ValueError(
                f"Concurrency must be one of {ALLOWED_CONCURRENCIES}, "
                f"got {args.concurrency}"
            )
        if (
            args.experiment_config is None
            and not 1 <= args.tuning_attempts <= 6
        ):
            raise ValueError("--tuning-attempts must be between 1 and 6")
        if not args.model_path.is_dir():
            raise FileNotFoundError(
                f"Model path does not exist: {args.model_path}"
            )
        if not args.dataset.is_file():
            raise FileNotFoundError(f"Dataset does not exist: {args.dataset}")
        if args.experiment_config is not None:
            if not args.experiment_config.is_file():
                raise FileNotFoundError(
                    f"Experiment config does not exist: "
                    f"{args.experiment_config}"
                )
            experiment_candidate = CandidateConfig.from_dict(
                read_json(args.experiment_config)
            )
            experiment_id = args.experiment_id or experiment_candidate.name
            base_result["benchmark"].update(
                {
                    "experiment_id": experiment_id,
                    "world_size": experiment_candidate.active_gpu_count,
                    "active_gpu_count": (
                        experiment_candidate.active_gpu_count
                    ),
                    "effective_parallelism": (
                        experiment_candidate.effective_parallelism
                    ),
                    "mtp_max_draft_len": (
                        experiment_candidate.mtp_draft_tokens
                    ),
                    "derived_output_tput_per_gpu": (
                        "concurrency / mean_token_tpot_seconds / "
                        f"{experiment_candidate.active_gpu_count}"
                    ),
                    "derived_step_tput_per_gpu": (
                        "concurrency / mean_step_tpot_seconds / "
                        f"{experiment_candidate.active_gpu_count}"
                    ),
                    "wall_output_tput_per_gpu": (
                        "all generated output tokens / measured wall "
                        "seconds / "
                        f"{experiment_candidate.active_gpu_count}"
                    ),
                }
            )
            base_result["experiment"] = {
                "id": experiment_id,
                "config_path": str(args.experiment_config),
                "status": "configured",
                "candidate": experiment_candidate.to_dict(),
            }
            write_json(result_path, base_result)
            log_progress(
                "single-candidate experiment configured "
                f"id={experiment_id} candidate={experiment_candidate.name} "
                f"parallelism={experiment_candidate.effective_parallelism} "
                f"wait_iters={experiment_candidate.batching_wait_iters} "
                "balance="
                f"{'on' if experiment_candidate.attention_dp_balance else 'off'} "
                "lm_head_tp="
                f"{'on' if experiment_candidate.enable_lm_head_tp_in_adp else 'off'} "
                "adp_batch_mode="
                f"{experiment_candidate.attention_dp_batch_mode} "
                "cute_dsl_mqa="
                f"{'on' if experiment_candidate.use_cute_dsl_paged_mqa_logits else 'off'} "
                "cute_dsl_topk="
                f"{'on' if experiment_candidate.use_cute_dsl_topk else 'off'} "
                "heuristic_topk="
                f"{'on' if experiment_candidate.enable_heuristic_topk else 'off'} "
                "indexer_k_dtype="
                f"{experiment_candidate.indexer_k_dtype or 'checkpoint'} "
                f"moe_backend={experiment_candidate.moe_backend} "
                "low_precision_moe_combine="
                f"{'on' if experiment_candidate.use_low_precision_moe_combine else 'off'} "
                "force_moe_comm="
                f"{experiment_candidate.force_moe_comm_method or 'auto'} "
                "profile_iterations="
                f"{experiment_candidate.profile_iterations or 'off'} "
                f"max_seq_len={experiment_candidate.max_seq_len} "
                "trt_iter_log="
                f"{'on' if experiment_candidate.print_iter_log else 'off'} "
                f"mtp={experiment_candidate.mtp_draft_tokens}"
            )

        log_progress(
            f"corpus preparation start dataset={args.dataset.name} "
            f"target_prompts={args.concurrency}"
        )
        corpus_manifest = prepare_corpus(
            args.dataset,
            str(args.model_path),
            args.concurrency,
            args.output_dir,
            dataset_revision=args.dataset_revision,
        )
        base_result["corpus"] = corpus_manifest
        write_json(result_path, base_result)
        adjusted_prompts = sum(
            1
            for item in corpus_manifest["context_tokenization"]
            if item.get("boundary_adjustment", "none") != "none"
            or int(item.get("context_tail_trimmed_characters", 0)) > 0
        )
        log_progress(
            f"corpus preparation complete "
            f"prompts={corpus_manifest['prompt_count']} "
            f"unique_contexts={corpus_manifest['unique_contexts']} "
            f"prompt_tokens={corpus_manifest['prompt_tokens']} "
            f"adjusted_prompts={adjusted_prompts}"
        )

        corpus_path = args.output_dir / "corpus.bin"
        manifest_path = args.output_dir / "corpus_manifest.json"
        if experiment_candidate is not None:
            log_progress(
                "single-candidate measurement start "
                f"candidate={experiment_candidate.name} "
                f"measured_passes={FINAL_MEASURED_PASSES}"
            )
            experiment_result = run_worker(
                output_dir=args.output_dir,
                model_path=args.model_path,
                corpus_path=corpus_path,
                manifest_path=manifest_path,
                candidate=experiment_candidate,
                attempt=1,
                mode="experiment",
                passes=FINAL_MEASURED_PASSES,
                timeout_seconds=args.worker_timeout,
            )
            base_result["experiment"].update(
                {
                    "status": experiment_result.get("status", "unknown"),
                    "result": compact_attempt(experiment_result),
                }
            )
            if experiment_result.get("status") != "success":
                failure_kind = experiment_result.get("failure_kind")
                is_capacity = (
                    args.concurrency >= 512
                    and failure_kind in {"oom", "capacity"}
                    and experiment_result.get("phase")
                    in {"engine_init", "warmup"}
                )
                base_result.update(
                    {
                        "status": (
                            "capacity_failure" if is_capacity else "failed"
                        ),
                        "finished_at": utc_now(),
                        "failure_kind": failure_kind,
                        "phase": experiment_result.get("phase"),
                        "error": (
                            "Single-candidate experiment failed "
                            f"({failure_kind or 'unknown'}) during "
                            f"{experiment_result.get('phase') or 'unknown'}: "
                            f"{experiment_result.get('error') or 'unknown'}"
                        ),
                        "final": compact_attempt(experiment_result),
                    }
                )
                write_json(result_path, base_result)
                log_progress(
                    f"benchmark finished status={base_result['status']} "
                    f"failure_kind={failure_kind or 'unknown'}"
                )
                return 0 if is_capacity else 1

            aggregate = experiment_result["aggregate"]
            base_result.update(
                {
                    "status": "success",
                    "finished_at": utc_now(),
                    "winner": experiment_candidate.to_dict(),
                    "final": experiment_result,
                    "aggregate": aggregate,
                    "huawei": huawei_comparison(
                        args.concurrency,
                        float(
                            aggregate["derived_output_tput_per_gpu"]
                        ),
                        float(aggregate["derived_step_tput_per_gpu"]),
                        float(aggregate["observed_tokens_per_step"]),
                        experiment_candidate.mtp_draft_tokens,
                        experiment_candidate.effective_parallelism,
                        experiment_candidate.active_gpu_count,
                    ),
                }
            )
            base_result["experiment"]["status"] = "success"
            write_json(result_path, base_result)
            log_progress(
                "benchmark complete status=success execution=single_candidate "
                f"{aggregate_progress(aggregate)}"
            )
            return 0

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
                log_progress(
                    f"{candidate.name}: retrying with CUDA graphs disabled "
                    f"after {result.get('failure_kind')} failure"
                )
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
                log_progress(
                    "stopping tuning after full-shape capacity failure "
                    f"phase={result.get('phase')} "
                    f"failure_kind={result.get('failure_kind')}"
                )
            if result.get("failure_kind") == "timeout":
                fatal_failure = result
            return result

        log_progress(
            f"scheduler tuning start "
            f"measured_passes={TUNING_MEASURED_PASSES}"
        )
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
        if scheduler_winner is not None:
            log_progress(
                "scheduler tuning winner "
                f"candidate={scheduler_winner['candidate']['name']} "
                f"{aggregate_progress(scheduler_winner['aggregate'])}"
            )
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
            log_progress(
                "benchmark failed because a worker timed out; "
                "node state is not reusable"
            )
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
            log_progress(
                f"benchmark finished status={base_result['status']} "
                f"failure_kinds={','.join(failure_kinds) or 'none'}"
            )
            return 0 if is_capacity else 1

        winner_candidate = CandidateConfig.from_dict(winner["candidate"])
        log_progress(
            f"tuning complete winner={winner_candidate.name} "
            f"attempt={winner['attempt']} "
            f"{aggregate_progress(winner['aggregate'])}"
        )
        base_result["tuning"]["winner"] = {
            "candidate": winner_candidate.to_dict(),
            "aggregate": winner["aggregate"],
            "attempt": winner["attempt"],
        }
        base_result["status"] = "final_measurement"
        write_json(result_path, base_result)

        log_progress(
            f"final measurement start candidate={winner_candidate.name} "
            f"measured_passes={FINAL_MEASURED_PASSES}"
        )
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
            log_progress(
                "benchmark failed because the fresh final measurement "
                f"failed phase={final_result.get('phase', 'unknown')}"
            )
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
                    float(aggregate["derived_step_tput_per_gpu"]),
                    float(aggregate["observed_tokens_per_step"]),
                    winner_candidate.mtp_draft_tokens,
                    winner_candidate.effective_parallelism,
                    winner_candidate.active_gpu_count,
                ),
            }
        )
        write_json(result_path, base_result)
        log_progress(
            "benchmark complete status=success "
            f"{aggregate_progress(aggregate)}"
        )
        return 0
    except BaseException as error:
        log_progress(
            f"benchmark failed error_type={type(error).__name__} "
            f"error={error}"
        )
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
