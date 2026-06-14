#!/usr/bin/env python3
"""Run one fixed-global-batch TensorRT-LLM offline measurement."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from io_utils import write_json
from metrics import (
    select_full_batch_decode_rounds,
    summarize_decode_rounds,
    summarize_requests,
)
from prompts import load_corpus
from trt_config import (
    FIXED_BENCHMARK_CONFIG,
    HUAWEI_MEASURED_DECODE_ROUNDS,
    HUAWEI_WARMUP_DECODE_ROUNDS,
    MEASURED_OUTPUT_TOKENS,
    SAMPLING_TEMPERATURE,
    SAMPLING_TOP_K,
    SAMPLING_TOP_P,
    WARMUP_OUTPUT_TOKENS,
    WORLD_SIZE,
    build_llm_kwargs,
    fixed_environment,
    local_batch_size,
    resolved_parallelism,
    validate_global_batch_size,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_progress(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[offline-trt-worker {timestamp}] {message}", flush=True)


def perfect_router_source() -> dict[str, Any]:
    from tensorrt_llm._torch.modules.fused_moe import interface

    try:
        source = inspect.getsource(interface.MoE._init_perfect_router)
    except (OSError, TypeError):
        source = Path(interface.__file__).read_text(encoding="utf-8")
    supported = (
        "ENABLE_PERFECT_ROUTER" in source
        and "_enable_perfect_router" in source
    )
    return {
        "supported": supported,
        "module": str(Path(interface.__file__).resolve()),
        "environment": os.getenv("ENABLE_PERFECT_ROUTER"),
    }


def install_mpi_worker_entry() -> dict[str, str]:
    from tensorrt_llm.executor import proxy
    from trt_mpi_entry import worker_main

    proxy.worker_main = worker_main
    return {
        "proxy_module": str(Path(proxy.__file__).resolve()),
        "worker_entry": "trt_mpi_entry.worker_main",
    }


def expected_rank_environment() -> dict[str, str]:
    raw = os.getenv("TRTLLM_BENCH_EXPECTED_RANK_ENV", "{}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError(
            "TRTLLM_BENCH_EXPECTED_RANK_ENV must contain an object"
        )
    expected = {str(name): str(item) for name, item in value.items()}
    mismatches = {
        name: {"expected": expected_value, "actual": os.getenv(name)}
        for name, expected_value in expected.items()
        if os.getenv(name) != expected_value
    }
    if mismatches:
        raise RuntimeError(
            "Benchmark environment mismatch in controller worker: "
            f"{mismatches}"
        )
    return expected


def read_perfect_router_marker(path: Path) -> dict[str, Any]:
    processes: dict[int, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                processes[int(row["pid"])] = row
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
    mpi_entries = [
        row
        for row in processes.values()
        if row.get("source") == "trt_mpi_entry"
        and row.get("perfect_router") == "1"
    ]
    mpi_entries_with_cache = [
        row for row in mpi_entries if row.get("cute_dsl_cache_dir")
    ]
    return {
        "marker_path": str(path),
        "unique_processes": len(processes),
        "mpi_entry_processes": len(mpi_entries),
        "mpi_entry_ranks": sorted(
            int(row["rank"])
            for row in mpi_entries
            if row.get("rank") is not None
        ),
        "mpi_entry_cute_cache_processes": len(mpi_entries_with_cache),
        "mpi_entry_cute_cache_paths": sorted(
            {
                str(row["cute_dsl_cache_dir"])
                for row in mpi_entries_with_cache
            }
        ),
        "processes": sorted(processes.values(), key=lambda row: row["pid"]),
    }


def sampling_params(count: int, max_tokens: int) -> list[Any]:
    from tensorrt_llm import SamplingParams

    return [
        SamplingParams(
            temperature=SAMPLING_TEMPERATURE,
            top_p=SAMPLING_TOP_P,
            top_k=SAMPLING_TOP_K,
            max_tokens=max_tokens,
            ignore_eos=True,
            return_perf_metrics=True,
            detokenize=False,
            add_special_tokens=False,
        )
        for _ in range(count)
    ]


def generate(
    llm: Any,
    inputs: list[dict[str, list[int]]],
    *,
    label: str,
    max_tokens: int,
) -> tuple[list[Any], float, list[dict[str, Any]]]:
    log_progress(
        f"{label}: generation start global_batch={len(inputs)} "
        f"max_output_tokens={max_tokens}"
    )
    started = time.perf_counter()
    outputs = llm.generate(
        inputs,
        sampling_params=sampling_params(len(inputs), max_tokens),
        use_tqdm=False,
    )
    wall_seconds = time.perf_counter() - started
    if not isinstance(outputs, list):
        outputs = [outputs]
    if len(outputs) != len(inputs):
        raise RuntimeError(
            f"TRT returned {len(outputs)} requests for {len(inputs)} prompts"
        )
    iteration_stats = llm.get_stats(timeout=10)
    log_progress(
        f"{label}: generation complete wall={wall_seconds:.1f}s "
        f"iteration_stats={len(iteration_stats)}"
    )
    return outputs, wall_seconds, iteration_stats


def validate_rank_propagation(
    marker_path: Path,
    rank_environment: dict[str, str],
) -> dict[str, Any]:
    marker = read_perfect_router_marker(marker_path)
    expected_ranks = list(range(WORLD_SIZE))
    if marker["mpi_entry_processes"] != WORLD_SIZE:
        raise RuntimeError(
            "TRT spawned an unexpected number of active rank entrypoints: "
            f"{marker['mpi_entry_processes']} != {WORLD_SIZE}"
        )
    if marker["mpi_entry_ranks"] != expected_ranks:
        raise RuntimeError(
            "TRT active ranks do not match DEP8: "
            f"{marker['mpi_entry_ranks']!r} != {expected_ranks!r}"
        )
    environment_mismatches = {}
    for row in marker["processes"]:
        if row.get("source") != "trt_mpi_entry":
            continue
        if row.get("benchmark_environment") != rank_environment:
            environment_mismatches[str(row.get("rank"))] = {
                "expected": rank_environment,
                "actual": row.get("benchmark_environment"),
            }
    if environment_mismatches:
        raise RuntimeError(
            "Benchmark environment did not reach every TRT rank: "
            f"{environment_mismatches}"
        )
    expected_cache = os.getenv("CUTE_DSL_CACHE_DIR")
    if not expected_cache:
        raise RuntimeError("CUTE_DSL_CACHE_DIR is required")
    if marker["mpi_entry_cute_cache_processes"] != WORLD_SIZE:
        raise RuntimeError(
            "CuTe cache path did not reach every TRT rank: "
            f"{marker['mpi_entry_cute_cache_processes']} != {WORLD_SIZE}"
        )
    if marker["mpi_entry_cute_cache_paths"] != [expected_cache]:
        raise RuntimeError(
            "CuTe cache path did not reach every TRT rank: "
            f"{marker['mpi_entry_cute_cache_paths']!r}"
        )
    return marker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = utc_now()
    worker_started = time.perf_counter()
    phase = "startup"
    result: dict[str, Any] = {
        "schema_version": 2,
        "status": "running",
        "started_at": started_at,
    }
    try:
        validate_global_batch_size(args.global_batch_size)
        configured_environment = fixed_environment(args.global_batch_size)
        rank_environment = expected_rank_environment()
        prompts, corpus_manifest = load_corpus(args.corpus, args.manifest)
        if len(prompts) != args.global_batch_size:
            raise RuntimeError(
                f"Corpus has {len(prompts)} prompts; expected global batch "
                f"{args.global_batch_size}"
            )
        inputs = [{"prompt_token_ids": prompt} for prompt in prompts]
        local_batch = local_batch_size(args.global_batch_size)
        log_progress(
            f"worker start global_batch={args.global_batch_size} "
            f"local_batch={local_batch} topology=DEP8 "
            f"warmup_rounds={HUAWEI_WARMUP_DECODE_ROUNDS} "
            f"measured_rounds={HUAWEI_MEASURED_DECODE_ROUNDS}"
        )

        phase = "perfect_router_source"
        router_source = perfect_router_source()
        if not router_source["supported"]:
            raise RuntimeError(
                "Pinned TRT image does not expose ENABLE_PERFECT_ROUTER"
            )
        if os.getenv("ENABLE_PERFECT_ROUTER") != "1":
            raise RuntimeError("ENABLE_PERFECT_ROUTER is not set")
        mpi_worker_entry = install_mpi_worker_entry()

        from tensorrt_llm import LLM

        llm_kwargs = build_llm_kwargs(
            args.model_path,
            args.global_batch_size,
        )
        phase = "engine_init"
        log_progress(
            "engine initialization start "
            f"max_batch_size={llm_kwargs['max_batch_size']} "
            f"max_num_tokens={llm_kwargs['max_num_tokens']} "
            f"max_seq_len={llm_kwargs['max_seq_len']} "
            "overlap_scheduler=off iter_stats=on"
        )
        init_started = time.perf_counter()
        with LLM(**llm_kwargs) as llm:
            init_seconds = time.perf_counter() - init_started
            resolved = resolved_parallelism(
                llm.args,
                args.global_batch_size,
            )
            log_progress(
                "engine initialization complete "
                f"elapsed={init_seconds:.1f}s "
                f"local_batch={resolved['local_batch_size']} "
                f"cuda_graph_batch={resolved['cuda_graph_batch_sizes']} "
                f"max_num_tokens={resolved['max_num_tokens']}"
            )

            phase = "warmup"
            warmup_outputs, warmup_wall, warmup_stats = generate(
                llm,
                inputs,
                label="warmup",
                max_tokens=WARMUP_OUTPUT_TOKENS,
            )
            warmup_requests = summarize_requests(
                warmup_outputs,
                wall_seconds=warmup_wall,
                expected_output_tokens=WARMUP_OUTPUT_TOKENS,
                num_gpus=WORLD_SIZE,
            )
            warmup_rounds, warmup_schedule = (
                select_full_batch_decode_rounds(
                    warmup_stats,
                    local_batch_size=local_batch,
                    required_rounds=HUAWEI_WARMUP_DECODE_ROUNDS,
                )
            )
            write_json(
                args.output.parent / "warmup_iteration_stats.json",
                warmup_stats,
            )
            log_progress(
                "warmup validated "
                f"full_batch_rounds="
                f"{warmup_schedule['full_batch_decode_rounds_available']} "
                f"selected_iters={warmup_rounds[0]['iter']}-"
                f"{warmup_rounds[-1]['iter']}"
            )
            time.sleep(2.0)

            phase = "measured_generation"
            measured_outputs, measured_wall, measured_stats = generate(
                llm,
                inputs,
                label="measured",
                max_tokens=MEASURED_OUTPUT_TOKENS,
            )
            request_summary = summarize_requests(
                measured_outputs,
                wall_seconds=measured_wall,
                expected_output_tokens=MEASURED_OUTPUT_TOKENS,
                num_gpus=WORLD_SIZE,
            )
            request_aggregate = request_summary["aggregate"]
            decode_rounds = summarize_decode_rounds(
                measured_stats,
                global_batch_size=args.global_batch_size,
                local_batch_size=local_batch,
                num_gpus=WORLD_SIZE,
            )
            write_json(
                args.output.parent / "measured_iteration_stats.json",
                measured_stats,
            )
            log_progress(
                "measured decode window validated "
                f"iters="
                f"{decode_rounds['schedule_validation']['selected_first_iter']}"
                "-"
                f"{decode_rounds['schedule_validation']['selected_last_iter']} "
                f"round_tpot={decode_rounds['decode_round_tpot_ms']:.3f}ms "
                "step_tput_per_gpu="
                f"{decode_rounds['decode_step_tput_per_gpu']:.2f} "
                f"tokens_per_step="
                f"{decode_rounds['observed_tokens_per_step']:.3f}"
            )

            phase = "perfect_router_validation"
            marker_path = Path(
                os.environ["TRTLLM_PERFECT_ROUTER_MARKER"]
            )
            marker = validate_rank_propagation(
                marker_path,
                rank_environment,
            )
            log_progress(
                "rank propagation validated "
                f"ranks={marker['mpi_entry_ranks']}"
            )
            phase = "engine_shutdown"
            log_progress("engine shutdown start")

        log_progress("engine shutdown complete")
        aggregate = {
            **request_aggregate,
            **decode_rounds,
            "pass_count": 1,
        }
        result.update(
            {
                "status": "success",
                "finished_at": utc_now(),
                "global_batch_size": args.global_batch_size,
                "local_batch_size": local_batch,
                "config": FIXED_BENCHMARK_CONFIG.to_dict(),
                "llm_kwargs": llm_kwargs,
                "resolved_parallelism": resolved,
                "engine_init_seconds": init_seconds,
                "warmup": {
                    "decode_rounds_required": (
                        HUAWEI_WARMUP_DECODE_ROUNDS
                    ),
                    "output_tokens_per_request": WARMUP_OUTPUT_TOKENS,
                    "wall_seconds": warmup_wall,
                    "request_aggregate": warmup_requests["aggregate"],
                    "schedule_validation": warmup_schedule,
                },
                "measured_pass": {
                    "decode_rounds_required": (
                        HUAWEI_MEASURED_DECODE_ROUNDS
                    ),
                    "output_tokens_per_request": MEASURED_OUTPUT_TOKENS,
                    "wall_seconds": measured_wall,
                    "requests": request_summary["requests"],
                    "iteration_stats_file": (
                        "measured_iteration_stats.json"
                    ),
                },
                "aggregate": aggregate,
                "perfect_router": {
                    "source": router_source,
                    "mpi_worker_entry": mpi_worker_entry,
                    "propagation": marker,
                },
                "runtime_environment": {
                    "configured": configured_environment,
                    "rank_expected": rank_environment,
                },
                "corpus": corpus_manifest,
            }
        )
        write_json(args.output, result)
        log_progress(
            f"worker complete elapsed="
            f"{time.perf_counter() - worker_started:.1f}s "
            f"decode_round_tpot="
            f"{aggregate['decode_round_tpot_ms']:.3f}ms "
            f"decode_step_tput_per_gpu="
            f"{aggregate['decode_step_tput_per_gpu']:.2f} "
            f"output_tput_per_gpu={aggregate['output_tput_per_gpu']:.2f}"
        )
        return 0
    except BaseException as error:
        log_progress(
            f"worker failed phase={phase} "
            f"error_type={type(error).__name__} error={error}"
        )
        traceback.print_exc()
        marker_path_value = os.getenv("TRTLLM_PERFECT_ROUTER_MARKER")
        marker = (
            read_perfect_router_marker(Path(marker_path_value))
            if marker_path_value
            else None
        )
        result.update(
            {
                "status": "failed",
                "finished_at": utc_now(),
                "phase": phase,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "perfect_router_marker": marker,
            }
        )
        write_json(args.output, result)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
