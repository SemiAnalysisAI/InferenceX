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

from io_utils import read_json, read_locked_text, tail_text, write_json
from metrics import (
    apply_trt_host_step_timing,
    huawei_comparison,
    pr_reference_comparison,
)
from prompts import INFINITEBENCH_REVISION, prepare_corpus, sha256_file
from trt_config import (
    ALLOWED_GLOBAL_BATCH_SIZES,
    CONTROLLED_ENVIRONMENT_VARIABLES,
    FIXED_BATCH_ARM_FILENAME,
    FIXED_BENCHMARK_CONFIG,
    HARDWARE_PROFILE,
    INPUT_TOKENS,
    PINNED_TRT_GLOBAL_SEED,
    SAMPLING_TEMPERATURE,
    SAMPLING_TOP_K,
    SAMPLING_TOP_P,
    WORLD_SIZE,
    attention_workspace_target_bytes,
    benchmark_profile_key,
    benchmark_environment,
    engine_max_batch_size,
    engine_warmup_max_tokens,
    external_mpi_rank_environment,
    local_batch_size,
    max_num_tokens,
    measured_output_tokens,
    setup_prefill_iterations,
    validate_global_batch_size,
    warmup_output_tokens,
)


PROGRESS_INTERVAL_SECONDS = 60.0
TIMING_LOG_VISIBILITY_TIMEOUT_SECONDS = 10.0
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


def configure_perfect_router_environment(
    environment: dict[str, str],
    *,
    enabled: bool,
) -> None:
    """Set or remove both perfect-router aliases for the worker process."""
    if enabled:
        environment["ENABLE_PERFECT_ROUTER"] = "1"
        environment["TRTLLM_ENABLE_PERFECT_ROUTER"] = "1"
    else:
        environment.pop("ENABLE_PERFECT_ROUTER", None)
        environment.pop("TRTLLM_ENABLE_PERFECT_ROUTER", None)


def apply_external_trt_timing(
    aggregate: dict[str, Any],
    timing_log_path: Path,
    *,
    global_batch_size: int,
    num_gpus: int,
    skip_rounds: int,
    timeout_seconds: float = TIMING_LOG_VISIBILITY_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Read the external MPI log once all selected timing rows are visible."""
    deadline = time.monotonic() + timeout_seconds
    last_error: BaseException | None = None
    while True:
        try:
            log_text = timing_log_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
            return apply_trt_host_step_timing(
                aggregate,
                log_text,
                global_batch_size=global_batch_size,
                num_gpus=num_gpus,
                skip_rounds=skip_rounds,
            )
        except FileNotFoundError as error:
            last_error = error
        except RuntimeError as error:
            if "missing selected host-step rows" not in str(error):
                raise
            last_error = error
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "TRT timing log did not expose every selected host-step "
                f"row within {timeout_seconds:.1f}s: {timing_log_path}: "
                f"{last_error}"
            ) from last_error
        time.sleep(0.25)


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
    for line in read_locked_text(marker_path).splitlines():
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


def latest_rank_fatal(marker_path: Path) -> str | None:
    if not marker_path.exists():
        return None
    marker_tail = read_locked_text(marker_path)[-256_000:]
    for line in reversed(marker_tail.splitlines()):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        event = str(row.get("event", ""))
        if row.get("source") != "trt_mpi_entry" or not (
            event == "entry_failed" or event.endswith("_error")
        ):
            continue
        return (
            f"rank={row.get('rank')} event={event} "
            f"error_type={row.get('error_type')} "
            f"error={row.get('error')}"
        )
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
            rank_fatal = (
                latest_rank_fatal(marker_path)
                if marker_path is not None
                else None
            )
            if rank_fatal is not None:
                raise WorkerFatalLogError(rank_fatal)
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
    config = FIXED_BENCHMARK_CONFIG
    is_huawei = config.benchmark_mode == "huawei"
    base_result: dict[str, Any] = {
        "schema_version": 2,
        "status": "running",
        "started_at": utc_now(),
        "benchmark": {
            "mode": (
                "offline_fixed_global_batch_decode"
                if is_huawei
                else "offline_pr_config_decode_saturation"
            ),
            "execution": "one_fresh_engine_one_measured_pass",
            "experiment_id": experiment_id,
            "engine": "TensorRT-LLM PyTorch backend",
            "request_path": "direct LLM.generate; no server or HTTP",
            "hardware": HARDWARE_PROFILE.hardware,
            "hardware_profile": HARDWARE_PROFILE.key,
            "benchmark_profile": benchmark_profile_key(),
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
            "engine_max_batch_size": None,
            "input_tokens": INPUT_TOKENS,
            "generated_output_tokens": None,
            "mtp_max_draft_len": config.mtp_draft_tokens,
            "max_seq_len": config.max_seq_len,
            "engine_warmup_max_tokens": None,
            "attention_workspace_target_bytes": None,
            "fp8_deep_gemm_max_rows": (
                FIXED_BENCHMARK_CONFIG.fp8_deep_gemm_max_rows
            ),
            "warmup_decode_rounds": config.warmup_decode_rounds,
            "measured_decode_rounds": config.measured_decode_rounds,
            "request_warmup_enabled": config.run_request_warmup,
            "perfect_router_enabled": config.enable_perfect_router,
            "timing_source": config.timing_source,
            "latency_rounds_to_skip": config.latency_rounds_to_skip,
            "sampling": {
                "temperature": SAMPLING_TEMPERATURE,
                "top_p": SAMPLING_TOP_P,
                "top_k": SAMPLING_TOP_K,
                "engine_global_seed": PINNED_TRT_GLOBAL_SEED,
            },
            "headline_tpot": (
                (
                    "mean TRT full-local-batch decode iteration latency"
                    if is_huawei
                    else (
                        "mean rank-0 TRT host_step_time for saturated "
                        "full-local-batch decode iterations"
                    )
                )
                + " after startup-round skip and upper-IQR filtering"
            ),
            "headline_throughput": (
                "global_batch_size / decode_round_tpot_seconds / "
                "active_gpu_count"
            ),
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
            "reference_recipe_url": config.reference_recipe_url,
            "reference_active_global_batch": (
                config.reference_active_global_batch
            ),
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
            "external_world_log_artifact": (
                Path(os.environ["TRT_BENCH_EXTERNAL_WORLD_LOG"]).name
                if os.getenv("TRT_BENCH_EXTERNAL_WORLD_LOG")
                else None
            ),
            "fabric_cluster_uuid": os.getenv(
                "TRT_BENCH_FABRIC_CLUSTER_UUID"
            ),
            "fabric_clique_id": os.getenv(
                "TRT_BENCH_FABRIC_CLIQUE_ID"
            ),
        },
    }
    if is_huawei:
        base_result["benchmark"]["huawei_method_match"] = {
            "fixed_global_batch": True,
            "prefill_decode_barrier_required": True,
            "warmup_decode_rounds": config.warmup_decode_rounds,
            "measured_decode_rounds": config.measured_decode_rounds,
            "skip_first_measured_round": True,
            "upper_iqr_outlier_filter": True,
            "mtp_yield_reported_separately": True,
        }
    else:
        base_result["benchmark"]["pr_decode_config_match"] = {
            "decode_topology": True,
            "cuda_graph_batches": True,
            "kv_cache_fraction": True,
            "moe_backend_and_eplb": True,
            "overlap_scheduler": True,
            "mtp_depth": True,
            "attention_dp_config_default_none": True,
            "learned_model_router": not config.enable_perfect_router,
            "reference_decode_max_num_tokens": (
                config.reference_decode_max_num_tokens
            ),
            "offline_runtime_max_num_tokens": (
                config.runtime_max_num_tokens
            ),
            "offline_adaptation": (
                "The PR decode worker receives prefilled KV. This direct "
                "offline worker uses a 32768-token context budget to admit "
                "8K prompts in stages, then times only the saturated "
                "decode-only window."
            ),
            "measurement_instrumentation": (
                "enable_iter_perf_stats=True and max_stats_len=2048 are "
                "enabled only to prove the fixed decode window and measure "
                "MTP acceptance. The PR serving worker leaves iteration "
                "stats disabled."
            ),
        }
    write_json(result_path, base_result)
    log_progress(
        f"benchmark start hardware={HARDWARE_PROFILE.hardware} "
        f"profile={config.profile_key} "
        f"topology={FIXED_BENCHMARK_CONFIG.parallelism} "
        f"global_batch={args.global_batch_size} "
        f"experiment_id={experiment_id} input_tokens={INPUT_TOKENS} "
        f"warmup_rounds={config.warmup_decode_rounds} "
        f"measured_rounds={config.measured_decode_rounds} "
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
        base_result["benchmark"]["engine_max_batch_size"] = (
            engine_max_batch_size(args.global_batch_size)
        )
        base_result["benchmark"]["max_num_tokens"] = max_num_tokens(
            args.global_batch_size
        )
        base_result["benchmark"]["generated_output_tokens"] = (
            measured_output_tokens(args.global_batch_size)
        )
        base_result["benchmark"]["warmup_output_tokens"] = (
            warmup_output_tokens(args.global_batch_size)
        )
        base_result["benchmark"]["setup_prefill_iterations"] = (
            setup_prefill_iterations(args.global_batch_size)
        )
        base_result["benchmark"]["engine_warmup_max_tokens"] = (
            engine_warmup_max_tokens(args.global_batch_size)
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
        environment["TRTLLM_PERFECT_ROUTER_MARKER"] = str(marker_path)
        configured_environment = benchmark_environment(
            args.global_batch_size,
            fixed_batch_arm_path,
        )
        if os.getenv("TRT_BENCH_EXTERNAL_MPI") == "1":
            cute_cache_dir = environment.get(
                "TRTLLM_BENCH_CUTE_DSL_CACHE_DIR"
            )
            if not cute_cache_dir:
                raise RuntimeError(
                    "External MPI launch is missing "
                    "TRTLLM_BENCH_CUTE_DSL_CACHE_DIR"
                )
            expected_launch_environment = external_mpi_rank_environment(
                args.global_batch_size,
                fixed_batch_arm_path,
                marker_path,
                cute_cache_dir,
            )
            launch_mismatches = {
                name: {
                    "expected": expected,
                    "actual": environment.get(name),
                }
                for name, expected in expected_launch_environment.items()
                if environment.get(name) != expected
            }
            if launch_mismatches:
                raise RuntimeError(
                    "External MPI rank environment was not preseeded "
                    f"before trtllm-llmapi-launch: {launch_mismatches}"
                )
            unexpected_launch_environment = {
                name: environment[name]
                for name in CONTROLLED_ENVIRONMENT_VARIABLES
                if name not in expected_launch_environment
                and name in environment
            }
            if unexpected_launch_environment:
                raise RuntimeError(
                    "External MPI launch retained profile-incompatible "
                    "environment variables: "
                    f"{unexpected_launch_environment}"
                )
            log_progress(
                "validated preseeded external MPI rank environment "
                f"keys={len(expected_launch_environment)}"
            )
        for name in CONTROLLED_ENVIRONMENT_VARIABLES:
            environment.pop(name, None)
        environment.update(configured_environment)
        configure_perfect_router_environment(
            environment,
            enabled=config.enable_perfect_router,
        )
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
            "setup_prefill_iterations="
            f"{setup_prefill_iterations(args.global_batch_size)} "
            "measured_output_tokens="
            f"{measured_output_tokens(args.global_batch_size)} "
            "engine_warmup_max_tokens="
            f"{engine_warmup_max_tokens(args.global_batch_size)} "
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
        if config.overlap_scheduler:
            timing_log_path = Path(
                os.getenv(
                    "TRT_BENCH_EXTERNAL_WORLD_LOG",
                    str(worker_log),
                )
            )
            log_progress(
                "parsing rank-0 TRT iteration log for overlap-safe "
                f"host-step timing path={timing_log_path}"
            )
            aggregate = apply_external_trt_timing(
                aggregate,
                timing_log_path,
                global_batch_size=args.global_batch_size,
                num_gpus=WORLD_SIZE,
                skip_rounds=config.latency_rounds_to_skip,
            )
            aggregate["timing_log_file"] = timing_log_path.name
            worker_result["aggregate"] = aggregate
            write_json(worker_output, worker_result)
        base_result.update(
            {
                "status": "success",
                "finished_at": utc_now(),
                "final": worker_result,
                "aggregate": aggregate,
            }
        )
        if is_huawei:
            base_result["huawei"] = huawei_comparison(
                args.global_batch_size,
                aggregate,
                hardware_key=HARDWARE_PROFILE.key,
                hardware_label=HARDWARE_PROFILE.hardware,
            )
        else:
            reference_fields = (
                config.reference_concurrency,
                config.reference_active_global_batch,
                config.reference_output_tput_per_decode_gpu,
                config.reference_output_tput_per_total_gpu,
                config.reference_recipe_url,
            )
            if any(value is None for value in reference_fields):
                raise RuntimeError(
                    f"Profile {config.profile_key} lacks PR reference data"
                )
            base_result["pr_reference"] = pr_reference_comparison(
                aggregate,
                profile_name=config.profile_key,
                reference_concurrency=int(
                    config.reference_concurrency
                ),
                reference_active_global_batch=int(
                    config.reference_active_global_batch
                ),
                reference_prefill_gpu_count=(
                    config.reference_prefill_gpu_count
                ),
                reference_output_tput_per_decode_gpu=float(
                    config.reference_output_tput_per_decode_gpu
                ),
                reference_output_tput_per_total_gpu=float(
                    config.reference_output_tput_per_total_gpu
                ),
                reference_recipe_url=str(config.reference_recipe_url),
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
