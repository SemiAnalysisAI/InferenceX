#!/usr/bin/env python3
"""Controller for the fixed-global-batch TRT offline benchmark."""

from __future__ import annotations

import argparse
import json
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
    ALLOWED_GLOBAL_BATCH_SIZES,
    CONTROLLED_ENVIRONMENT_VARIABLES,
    FIXED_BATCH_ARM_FILENAME,
    FIXED_BENCHMARK_CONFIG,
    HARDWARE_PROFILE,
    HUAWEI_MEASURED_DECODE_ROUNDS,
    HUAWEI_WARMUP_DECODE_ROUNDS,
    INPUT_TOKENS,
    MAX_SEQ_LEN,
    MEASURED_OUTPUT_TOKENS,
    MTP_DRAFT_TOKENS,
    PINNED_TRT_GLOBAL_SEED,
    SAMPLING_TEMPERATURE,
    SAMPLING_TOP_K,
    SAMPLING_TOP_P,
    WORLD_SIZE,
    attention_workspace_target_bytes,
    benchmark_environment,
    local_batch_size,
    max_num_tokens,
    validate_global_batch_size,
)


PROGRESS_INTERVAL_SECONDS = 60.0
WORKER_PROGRESS_PREFIX = "[offline-trt-worker "
MPI_PROGRESS_PREFIX = "[offline-trt-mpi] "
WORKER_FATAL_LOG_MARKERS = (
    "Fatal error detected, initiating shutdown",
)


class WorkerFatalLogError(RuntimeError):
    """A native TRT rank failed while its parent process remained alive."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_progress(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[offline-trt-controller {timestamp}] {message}", flush=True)


def aggregate_progress(aggregate: dict[str, Any]) -> str:
    fields: list[str] = []
    metrics = (
        ("decode_round_tpot_ms", "round_tpot", ".3f", "ms"),
        (
            "decode_step_tput_per_gpu",
            "step_tput_per_gpu",
            ".2f",
            "",
        ),
        ("output_tput_per_gpu", "output_tput_per_gpu", ".2f", ""),
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
        if line.startswith(MPI_PROGRESS_PREFIX):
            return line
    return None


def latest_rank_progress(marker_path: Path) -> str | None:
    if not marker_path.exists():
        return None
    latest_by_rank: dict[int, str] = {}
    for line in marker_path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            if row.get("source") != "trt_mpi_entry":
                continue
            rank = int(row["rank"])
            event = str(row["event"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        latest_by_rank[rank] = event
    if not latest_by_rank:
        return None
    ranks_by_event: dict[str, list[int]] = {}
    for rank, event in latest_by_rank.items():
        ranks_by_event.setdefault(event, []).append(rank)
    return ",".join(
        f"{event}:{sorted(ranks)}"
        for event, ranks in sorted(ranks_by_event.items())
    )


def latest_worker_fatal(worker_log: Path) -> str | None:
    for line in reversed(tail_text(worker_log).splitlines()):
        if any(marker in line for marker in WORKER_FATAL_LOG_MARKERS):
            return line.strip()
    return None


def wait_for_worker_process(
    process: subprocess.Popen[Any],
    *,
    label: str,
    worker_log: Path,
    marker_path: Path | None = None,
    started: float,
    timeout_seconds: int,
    heartbeat_seconds: float = PROGRESS_INTERVAL_SECONDS,
) -> int:
    while True:
        elapsed = time.perf_counter() - started
        remaining = timeout_seconds - elapsed
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout_seconds)
        try:
            return process.wait(timeout=min(heartbeat_seconds, remaining))
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - started
            if elapsed >= timeout_seconds:
                raise subprocess.TimeoutExpired(
                    process.args,
                    timeout_seconds,
                ) from None
            fatal_line = latest_worker_fatal(worker_log)
            if fatal_line is not None:
                raise WorkerFatalLogError(fatal_line)
            latest = latest_worker_progress(worker_log)
            details = []
            if latest is not None:
                details.append(f"last_worker_progress={latest}")
            rank_progress = (
                latest_rank_progress(marker_path)
                if marker_path is not None
                else None
            )
            if rank_progress is not None:
                details.append(f"rank_progress={rank_progress}")
            detail = f"; {'; '.join(details)}" if details else ""
            log_progress(
                f"{label}: still running elapsed={elapsed:.0f}s "
                f"timeout={timeout_seconds}s{detail}; "
                f"worker_log={worker_log.name}"
            )


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


def git_revision() -> str | None:
    explicit = os.getenv("TRT_BENCH_GIT_REVISION") or os.getenv("GITHUB_SHA")
    if explicit:
        return explicit
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
    return {"path": str(model_path), "identity_files": files}


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
    if (
        "fixed local batch" in text
        or "fixed-batch barrier" in text
        or "full-batch decode" in text
        or "full-local-batch prefill" in text
        or "prefill/decode barrier" in text
        or "mixed prefill and decode" in text
    ):
        return "fixed_batch_validation"
    if "out of memory" in text or "cuda_error_out_of_memory" in text:
        return "oom"
    if (
        "illegal memory access" in text
        or "cuda_error_illegal_address" in text
    ):
        return "cuda_illegal_address"
    if (
        "exceeds max_batch_size" in text
        or "exceeds max_num_tokens" in text
        or "kv cache" in text
        or "insufficient memory" in text
    ):
        return "capacity"
    if "cuda graph" in text or "graph capture" in text:
        return "cuda_graph"
    if "perfect-router" in text or "perfect_router" in text:
        return "perfect_router"
    if worker_result.get("phase") == "engine_init":
        return "initialization"
    return "runtime"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-revision", default=INFINITEBENCH_REVISION)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-timeout", type=int, default=7200)
    parser.add_argument("--experiment-id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    experiment_id = args.experiment_id or f"gbs{args.global_batch_size}"
    base_result: dict[str, Any] = {
        "schema_version": 2,
        "status": "running",
        "started_at": utc_now(),
        "benchmark": {
            "mode": "offline_fixed_global_batch_decode",
            "execution": "one_fresh_engine_one_measured_pass",
            "experiment_id": experiment_id,
            "engine": "TensorRT-LLM PyTorch backend",
            "request_path": "direct LLM.generate; no server or HTTP",
            "hardware": HARDWARE_PROFILE.hardware,
            "hardware_profile": HARDWARE_PROFILE.key,
            "renderer_hw": HARDWARE_PROFILE.renderer_hw,
            "world_size": WORLD_SIZE,
            "active_gpu_count": WORLD_SIZE,
            "physical_nodes": HARDWARE_PROFILE.physical_nodes,
            "gpus_per_node": HARDWARE_PROFILE.gpus_per_node,
            "is_multinode": HARDWARE_PROFILE.is_multinode,
            "effective_parallelism": (
                FIXED_BENCHMARK_CONFIG.parallelism
            ),
            "global_batch_size": args.global_batch_size,
            # Retained for the renderer and old result-discovery clients.
            "concurrency": args.global_batch_size,
            "local_batch_size": (
                args.global_batch_size // WORLD_SIZE
                if args.global_batch_size % WORLD_SIZE == 0
                else None
            ),
            "input_tokens": INPUT_TOKENS,
            "generated_output_tokens": MEASURED_OUTPUT_TOKENS,
            "mtp_max_draft_len": MTP_DRAFT_TOKENS,
            "max_seq_len": MAX_SEQ_LEN,
            "attention_workspace_target_bytes": None,
            "fp8_deep_gemm_max_rows": (
                FIXED_BENCHMARK_CONFIG.fp8_deep_gemm_max_rows
            ),
            "warmup_decode_rounds": HUAWEI_WARMUP_DECODE_ROUNDS,
            "measured_decode_rounds": HUAWEI_MEASURED_DECODE_ROUNDS,
            "sampling": {
                "temperature": SAMPLING_TEMPERATURE,
                "top_p": SAMPLING_TOP_P,
                "top_k": SAMPLING_TOP_K,
                "engine_global_seed": PINNED_TRT_GLOBAL_SEED,
            },
            "headline_tpot": (
                "mean TRT full-local-batch decode iteration latency after "
                "skipping the first round and removing upper-IQR outliers"
            ),
            "headline_throughput": (
                "global_batch_size / decode_round_tpot_seconds / "
                "active_gpu_count"
            ),
            "huawei_method_match": {
                "fixed_global_batch": True,
                "prefill_decode_barrier_required": True,
                "warmup_decode_rounds": HUAWEI_WARMUP_DECODE_ROUNDS,
                "measured_decode_rounds": HUAWEI_MEASURED_DECODE_ROUNDS,
                "skip_first_measured_round": True,
                "upper_iqr_outlier_filter": True,
                "mtp_yield_reported_separately": True,
            },
        },
        "config": FIXED_BENCHMARK_CONFIG.to_dict(),
        "provenance": {
            "git_revision": git_revision(),
            "image": os.getenv("IMAGE", HARDWARE_PROFILE.image),
            "trt_source_commit": os.getenv(
                "TRT_LLM_GIT_COMMIT",
                HARDWARE_PROFILE.trt_source_commit,
            ),
            "reference_pr": HARDWARE_PROFILE.reference_pr,
            "model": model_manifest(args.model_path),
            "slurm_job_id": os.getenv("TRT_BENCH_SLURM_JOB_ID"),
            "slurm_node": os.getenv("TRT_BENCH_SLURM_NODE"),
            "slurm_nodes": os.getenv("TRT_BENCH_SLURM_NODELIST"),
            "rank_map_artifact": os.getenv(
                "TRT_BENCH_RANK_MAP_ARTIFACT"
            ),
            "topology_artifact": os.getenv(
                "TRT_BENCH_TOPOLOGY_ARTIFACT"
            ),
        },
    }
    write_json(result_path, base_result)
    log_progress(
        f"benchmark start hardware={HARDWARE_PROFILE.hardware} "
        f"topology={FIXED_BENCHMARK_CONFIG.parallelism} "
        f"global_batch={args.global_batch_size} "
        f"experiment_id={experiment_id} input_tokens={INPUT_TOKENS} "
        f"warmup_rounds={HUAWEI_WARMUP_DECODE_ROUNDS} "
        f"measured_rounds={HUAWEI_MEASURED_DECODE_ROUNDS} "
        f"worker_timeout={args.worker_timeout}s"
    )

    try:
        validate_global_batch_size(args.global_batch_size)
        if not args.model_path.is_dir():
            raise FileNotFoundError(
                f"Model path does not exist: {args.model_path}"
            )
        if not args.dataset.is_file():
            raise FileNotFoundError(f"Dataset does not exist: {args.dataset}")
        local_batch = local_batch_size(args.global_batch_size)
        base_result["benchmark"]["local_batch_size"] = local_batch
        base_result["benchmark"]["max_num_tokens"] = max_num_tokens(
            args.global_batch_size
        )
        base_result["benchmark"]["attention_workspace_target_bytes"] = (
            attention_workspace_target_bytes(args.global_batch_size)
        )
        write_json(result_path, base_result)

        log_progress(
            f"corpus preparation start dataset={args.dataset.name} "
            f"global_batch={args.global_batch_size}"
        )
        corpus_manifest = prepare_corpus(
            args.dataset,
            str(args.model_path),
            args.global_batch_size,
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
            "corpus preparation complete "
            f"prompts={corpus_manifest['prompt_count']} "
            f"prompt_tokens={corpus_manifest['prompt_tokens']} "
            f"adjusted_prompts={adjusted_prompts}"
        )

        worker_output = args.output_dir / "worker_result.json"
        worker_log = args.output_dir / "worker.log"
        marker_path = args.output_dir / "perfect_router.jsonl"
        fixed_batch_arm_path = (
            args.output_dir / FIXED_BATCH_ARM_FILENAME
        ).resolve()
        fixed_batch_arm_path.unlink(missing_ok=True)
        worker_script = Path(__file__).with_name("trt_worker.py")
        command = [
            sys.executable,
            str(worker_script),
            "--model-path",
            str(args.model_path),
            "--corpus",
            str(args.output_dir / "corpus.bin"),
            "--manifest",
            str(args.output_dir / "corpus_manifest.json"),
            "--global-batch-size",
            str(args.global_batch_size),
            "--output",
            str(worker_output),
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
        for name in CONTROLLED_ENVIRONMENT_VARIABLES:
            environment.pop(name, None)
        configured_environment = benchmark_environment(
            args.global_batch_size,
            fixed_batch_arm_path,
        )
        environment.update(configured_environment)
        environment["TRTLLM_BENCH_EXPECTED_RANK_ENV"] = json.dumps(
            configured_environment,
            sort_keys=True,
            separators=(",", ":"),
        )

        log_progress(
            "worker launch "
            f"global_batch={args.global_batch_size} "
            f"local_batch={local_batch} "
            f"max_num_tokens={max_num_tokens(args.global_batch_size)} "
            "attention_workspace_bytes="
            f"{attention_workspace_target_bytes(args.global_batch_size)} "
            "fp8_deep_gemm_max_rows="
            f"{FIXED_BENCHMARK_CONFIG.fp8_deep_gemm_max_rows} "
            f"fixed_batch_arm_file={fixed_batch_arm_path}"
        )
        started = time.perf_counter()
        timed_out = False
        fatal_log_error: str | None = None
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
                    label=experiment_id,
                    worker_log=worker_log,
                    marker_path=marker_path,
                    started=started,
                    timeout_seconds=args.worker_timeout,
                )
            except subprocess.TimeoutExpired:
                timed_out = True
                terminate_process_group(process)
                return_code = process.returncode
            except WorkerFatalLogError as error:
                fatal_log_error = str(error)
                log_progress(
                    f"{experiment_id}: fatal TRT worker log detected; "
                    "terminating process group"
                )
                terminate_process_group(process)
                return_code = process.returncode
        elapsed = time.perf_counter() - started

        if worker_output.exists():
            worker_result = read_json(worker_output)
        else:
            worker_result = {
                "schema_version": 2,
                "status": "failed",
                "phase": "worker_process",
                "error_type": (
                    "WorkerFatalLogError"
                    if fatal_log_error
                    else "WorkerProcessError"
                ),
                "error": (
                    fatal_log_error
                    or (
                        f"Worker timed out after {args.worker_timeout}s"
                        if timed_out
                        else (
                            "Worker exited "
                            f"{return_code} without an output file"
                        )
                    )
                ),
            }
            write_json(worker_output, worker_result)
        worker_result.update(
            {
                "return_code": return_code,
                "timed_out": timed_out,
                "fatal_log_error": fatal_log_error,
                "elapsed_seconds": elapsed,
                "worker_output": worker_output.name,
                "worker_log": worker_log.name,
            }
        )
        write_json(worker_output, worker_result)

        if worker_result.get("status") != "success":
            failure_kind = classify_failure(
                worker_result,
                tail_text(worker_log),
                timed_out=timed_out,
            )
            base_result.update(
                {
                    "status": "failed",
                    "finished_at": utc_now(),
                    "failure_kind": failure_kind,
                    "phase": worker_result.get("phase"),
                    "error": worker_result.get("error"),
                    "final": worker_result,
                }
            )
            write_json(result_path, base_result)
            log_progress(
                f"benchmark failed elapsed={elapsed:.1f}s "
                f"failure_kind={failure_kind} "
                f"phase={worker_result.get('phase', 'unknown')}"
            )
            return 1

        aggregate = worker_result["aggregate"]
        base_result.update(
            {
                "status": "success",
                "finished_at": utc_now(),
                "final": worker_result,
                "aggregate": aggregate,
                "huawei": huawei_comparison(
                    args.global_batch_size,
                    aggregate,
                    hardware_key=HARDWARE_PROFILE.key,
                    hardware_label=HARDWARE_PROFILE.hardware,
                ),
            }
        )
        write_json(result_path, base_result)
        log_progress(
            "benchmark complete status=success "
            f"elapsed={elapsed:.1f}s {aggregate_progress(aggregate)}"
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
