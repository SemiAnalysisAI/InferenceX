#!/usr/bin/env python3
"""Run one fresh TensorRT-LLM offline candidate."""

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

from io_utils import read_json, write_json
from metrics import aggregate_passes, summarize_pass
from prompts import load_corpus
from trt_config import (
    OUTPUT_TOKENS,
    SAMPLING_TEMPERATURE,
    SAMPLING_TOP_K,
    SAMPLING_TOP_P,
    CandidateConfig,
    build_llm_kwargs,
    candidate_environment,
    resolved_parallelism,
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
            "TRTLLM_BENCH_EXPECTED_RANK_ENV must contain a JSON object"
        )
    expected = {str(name): str(item) for name, item in value.items()}
    mismatches = {
        name: {
            "expected": expected_value,
            "actual": os.getenv(name),
        }
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
    enabled = [
        row for row in processes.values() if row.get("perfect_router") == "1"
    ]
    mpi_entries = [
        row for row in enabled if row.get("source") == "trt_mpi_entry"
    ]
    mpi_entry_ranks = sorted(
        int(row["rank"])
        for row in mpi_entries
        if row.get("rank") is not None
    )
    mpi_entries_with_cache = [
        row for row in mpi_entries if row.get("cute_dsl_cache_dir")
    ]
    return {
        "marker_path": str(path),
        "unique_processes": len(processes),
        "enabled_processes": len(enabled),
        "mpi_entry_processes": len(mpi_entries),
        "mpi_entry_ranks": mpi_entry_ranks,
        "mpi_entry_cute_cache_processes": len(mpi_entries_with_cache),
        "mpi_entry_cute_cache_paths": sorted(
            {
                str(row["cute_dsl_cache_dir"])
                for row in mpi_entries_with_cache
            }
        ),
        "processes": sorted(processes.values(), key=lambda row: row["pid"]),
    }


def sampling_params(concurrency: int) -> list[Any]:
    from tensorrt_llm import SamplingParams

    return [
        SamplingParams(
            temperature=SAMPLING_TEMPERATURE,
            top_p=SAMPLING_TOP_P,
            top_k=SAMPLING_TOP_K,
            max_tokens=OUTPUT_TOKENS,
            ignore_eos=True,
            return_perf_metrics=True,
            detokenize=False,
            add_special_tokens=False,
        )
        for _ in range(concurrency)
    ]


def generate_pass(
    llm: Any,
    inputs: list[dict[str, list[int]]],
    *,
    label: str,
    candidate: CandidateConfig,
) -> dict[str, Any]:
    log_progress(f"{label}: generation start requests={len(inputs)}")
    started = time.perf_counter()
    outputs = llm.generate(
        inputs,
        sampling_params(concurrency=len(inputs)),
        use_tqdm=False,
    )
    wall_seconds = time.perf_counter() - started
    if not isinstance(outputs, list):
        outputs = [outputs]
    if len(outputs) != len(inputs):
        raise RuntimeError(
            f"TRT returned {len(outputs)} requests for {len(inputs)} prompts"
        )
    summary = summarize_pass(
        outputs,
        wall_seconds=wall_seconds,
        expected_output_tokens=OUTPUT_TOKENS,
        num_gpus=candidate.active_gpu_count,
        max_draft_tokens=candidate.mtp_draft_tokens,
    )
    aggregate = summary["aggregate"]
    log_progress(
        f"{label}: generation complete wall={wall_seconds:.1f}s "
        f"mean_tpot={float(aggregate['mean_token_tpot_ms']):.3f}ms "
        "derived_output_tput_per_gpu="
        f"{float(aggregate['derived_output_tput_per_gpu']):.2f} "
        f"derived_step_tput_per_gpu="
        f"{float(aggregate['derived_step_tput_per_gpu']):.2f} "
        f"wall_output_tput_per_gpu="
        f"{float(aggregate['wall_output_tput_per_gpu']):.2f} "
        f"tokens_per_step="
        f"{float(aggregate['observed_tokens_per_step']):.3f}"
    )
    return summary


def measured_pass_count(requested_passes: int) -> int:
    if requested_passes != 1:
        raise ValueError("--passes must be exactly 1 for this benchmark")
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("tune", "final", "experiment"),
        required=True,
    )
    parser.add_argument("--passes", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = utc_now()
    worker_started = time.perf_counter()
    phase = "startup"
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "mode": args.mode,
        "started_at": started_at,
    }
    try:
        phase = "load_candidate"
        candidate = CandidateConfig.from_dict(read_json(args.candidate))
        configured_environment = candidate_environment(candidate)
        rank_environment = expected_rank_environment()
        prompts, corpus_manifest = load_corpus(args.corpus, args.manifest)
        concurrency = len(prompts)
        if concurrency != int(corpus_manifest["concurrency"]):
            raise RuntimeError("Corpus concurrency does not match its manifest")
        inputs = [{"prompt_token_ids": prompt} for prompt in prompts]
        pass_count = measured_pass_count(args.passes)
        log_progress(
            f"worker start mode={args.mode} candidate={candidate.name} "
            f"concurrency={concurrency} measured_passes={pass_count} "
            f"parallelism={candidate.effective_parallelism} "
            f"mtp={candidate.mtp_draft_tokens} "
            f"adp_batch_mode={candidate.attention_dp_batch_mode} "
            "lm_head_tp="
            f"{'on' if candidate.enable_lm_head_tp_in_adp else 'off'} "
            "cute_dsl_topk="
            f"{'on' if candidate.use_cute_dsl_topk else 'off'} "
            "cute_dsl_mqa="
            f"{'on' if candidate.use_cute_dsl_paged_mqa_logits else 'off'} "
            "heuristic_topk="
            f"{'on' if candidate.enable_heuristic_topk else 'off'} "
            f"indexer_k_dtype={candidate.indexer_k_dtype or 'checkpoint'} "
            f"moe_backend={candidate.moe_backend} "
            "low_precision_moe_combine="
            f"{'on' if candidate.use_low_precision_moe_combine else 'off'} "
            f"force_moe_comm={candidate.force_moe_comm_method or 'auto'} "
            f"profile_iterations={candidate.profile_iterations or 'off'} "
            f"max_seq_len={candidate.max_seq_len} "
            f"trt_iter_log={'on' if candidate.print_iter_log else 'off'}"
        )

        phase = "perfect_router_source"
        log_progress("validating perfect-router support")
        router_source = perfect_router_source()
        if not router_source["supported"]:
            raise RuntimeError(
                "Pinned TRT image does not expose ENABLE_PERFECT_ROUTER"
            )
        if os.getenv("ENABLE_PERFECT_ROUTER") != "1":
            raise RuntimeError("ENABLE_PERFECT_ROUTER is not set in worker")

        log_progress("installing perfect-router MPI worker entry")
        mpi_worker_entry = install_mpi_worker_entry()
        from tensorrt_llm import LLM

        cute_dsl_available: bool | None = None
        if (
            candidate.use_cute_dsl_topk
            or candidate.use_cute_dsl_paged_mqa_logits
        ):
            phase = "cute_dsl_capability"
            from tensorrt_llm._torch.cute_dsl_utils import (
                IS_CUTLASS_DSL_AVAILABLE,
            )

            cute_dsl_available = bool(IS_CUTLASS_DSL_AVAILABLE)
            log_progress(
                "CuTE DSL capability "
                f"available={'yes' if cute_dsl_available else 'no'}"
            )
            if not cute_dsl_available:
                raise RuntimeError(
                    "A CuTE DSL kernel was requested, but the pinned TRT "
                    "image reports IS_CUTLASS_DSL_AVAILABLE=False"
                )

        llm_kwargs = build_llm_kwargs(
            args.model_path,
            concurrency,
            candidate,
        )
        phase = "engine_init"
        graph_batch_size = (
            llm_kwargs["cuda_graph_config"]["max_batch_size"]
            if llm_kwargs["cuda_graph_config"] is not None
            else None
        )
        log_progress(
            f"engine initialization start "
            f"max_batch_size={llm_kwargs['max_batch_size']} "
            f"max_seq_len={llm_kwargs['max_seq_len']} "
            f"max_num_tokens={llm_kwargs['max_num_tokens']} "
            f"cuda_graph={'on' if candidate.cuda_graph else 'off'} "
            f"cuda_graph_batch_size={graph_batch_size} "
            "cute_dsl_topk="
            f"{'on' if candidate.use_cute_dsl_topk else 'off'} "
            "cute_dsl_mqa="
            f"{'on' if candidate.use_cute_dsl_paged_mqa_logits else 'off'} "
            "heuristic_topk="
            f"{'on' if candidate.enable_heuristic_topk else 'off'} "
            f"indexer_k_dtype={candidate.indexer_k_dtype or 'checkpoint'} "
            f"moe_backend={candidate.moe_backend} "
            f"force_moe_comm={candidate.force_moe_comm_method or 'auto'} "
            f"profile_iterations={candidate.profile_iterations or 'off'} "
            f"trt_iter_log={'on' if candidate.print_iter_log else 'off'} "
            f"wait_iters={candidate.batching_wait_iters} "
            f"timeout_iters={candidate.attention_dp_timeout_iters}"
        )
        init_started = time.perf_counter()
        measured_passes: list[dict[str, Any]] = []
        with LLM(**llm_kwargs) as llm:
            init_seconds = time.perf_counter() - init_started
            resolved = resolved_parallelism(
                llm.args,
                candidate,
                concurrency,
            )
            log_progress(
                f"engine initialization complete elapsed={init_seconds:.1f}s "
                f"parallelism={resolved['effective_parallelism']} "
                f"max_batch_size={resolved['max_batch_size']} "
                f"max_seq_len={resolved['max_seq_len']} "
                "cuda_graph_batch_sizes="
                f"{resolved.get('cuda_graph_batch_sizes')} "
                "cute_dsl_topk="
                f"{'on' if resolved['use_cute_dsl_topk'] else 'off'} "
                "cute_dsl_mqa="
                f"{'on' if resolved['use_cute_dsl_paged_mqa_logits'] else 'off'} "
                "heuristic_topk="
                f"{'on' if resolved['enable_heuristic_topk'] else 'off'} "
                f"indexer_k_dtype={resolved['indexer_k_dtype']} "
                f"index_topk={resolved['index_topk']} "
                f"moe_backend={resolved['moe_backend']} "
                "low_precision_moe_combine="
                f"{'on' if resolved['use_low_precision_moe_combine'] else 'off'}"
            )

            phase = "warmup"
            warmup = generate_pass(
                llm,
                inputs,
                label="warmup",
                candidate=candidate,
            )
            time.sleep(2.0)

            phase = "measured_generation"
            for pass_index in range(pass_count):
                measured = generate_pass(
                    llm,
                    inputs,
                    label=f"measured pass {pass_index + 1}/{pass_count}",
                    candidate=candidate,
                )
                measured["pass_index"] = pass_index + 1
                measured_passes.append(measured)
                if pass_index + 1 < pass_count:
                    time.sleep(2.0)

            phase = "perfect_router_validation"
            log_progress("validating perfect-router rank propagation")
            marker_path = Path(
                os.environ["TRTLLM_PERFECT_ROUTER_MARKER"]
            )
            marker = read_perfect_router_marker(marker_path)
            expected_ranks = list(range(candidate.active_gpu_count))
            if marker["mpi_entry_processes"] != candidate.active_gpu_count:
                raise RuntimeError(
                    "TRT spawned an unexpected number of active rank "
                    "entrypoints: "
                    f"{marker['mpi_entry_processes']} != "
                    f"{candidate.active_gpu_count}"
                )
            if marker["mpi_entry_ranks"] != expected_ranks:
                raise RuntimeError(
                    "TRT active rank IDs do not match the requested "
                    f"topology: {marker['mpi_entry_ranks']!r} != "
                    f"{expected_ranks!r}"
                )
            environment_mismatches = {}
            for row in marker["processes"]:
                if row.get("source") != "trt_mpi_entry":
                    continue
                actual_environment = row.get("benchmark_environment")
                if actual_environment != rank_environment:
                    environment_mismatches[str(row.get("rank"))] = {
                        "expected": rank_environment,
                        "actual": actual_environment,
                    }
            if environment_mismatches:
                raise RuntimeError(
                    "Benchmark environment did not reach every TRT rank: "
                    f"{environment_mismatches}"
                )
            expected_cute_cache = os.getenv("CUTE_DSL_CACHE_DIR")
            if not expected_cute_cache:
                raise RuntimeError(
                    "CUTE_DSL_CACHE_DIR is required for fresh TRT engines"
                )
            if (
                marker["mpi_entry_cute_cache_processes"]
                != candidate.active_gpu_count
            ):
                raise RuntimeError(
                    "Persistent CuTe cache was not installed in every TRT "
                    "rank entrypoint: "
                    f"{marker['mpi_entry_cute_cache_processes']} != "
                    f"{candidate.active_gpu_count}"
                )
            if marker["mpi_entry_cute_cache_paths"] != [expected_cute_cache]:
                raise RuntimeError(
                    "Persistent CuTe cache was not installed in every TRT "
                    "rank entrypoint: "
                    f"{marker['mpi_entry_cute_cache_paths']!r}"
                )
            log_progress(
                "perfect-router validation complete "
                f"rank_processes={marker['mpi_entry_processes']} "
                f"ranks={marker['mpi_entry_ranks']} "
                f"cute_cache={expected_cute_cache} "
                f"environment_keys={sorted(rank_environment)}"
            )
            phase = "engine_shutdown"
            log_progress("engine shutdown start")

        log_progress("engine shutdown complete")
        phase = "aggregation"
        log_progress(f"aggregating {len(measured_passes)} measured passes")
        aggregate = aggregate_passes(
            measured_passes,
            candidate.active_gpu_count,
        )
        result.update(
            {
                "status": "success",
                "finished_at": utc_now(),
                "candidate": candidate.to_dict(),
                "concurrency": concurrency,
                "llm_kwargs": llm_kwargs,
                "resolved_parallelism": resolved,
                "engine_init_seconds": init_seconds,
                "warmup": warmup,
                "measured_passes": measured_passes,
                "aggregate": aggregate,
                "perfect_router": {
                    "source": router_source,
                    "mpi_worker_entry": mpi_worker_entry,
                    "propagation": marker,
                },
                "runtime_cache": {
                    "cute_dsl_cache_dir": expected_cute_cache,
                    "persistent": bool(expected_cute_cache),
                },
                "runtime_capabilities": {
                    "cutlass_dsl_available": cute_dsl_available,
                },
                "runtime_environment": {
                    "candidate": configured_environment,
                    "rank_expected": rank_environment,
                },
                "corpus": corpus_manifest,
            }
        )
        write_json(args.output, result)
        log_progress(
            f"worker complete elapsed="
            f"{time.perf_counter() - worker_started:.1f}s "
            f"mean_tpot={float(aggregate['mean_token_tpot_ms']):.3f}ms "
            "derived_output_tput_per_gpu="
            f"{float(aggregate['derived_output_tput_per_gpu']):.2f} "
            "derived_step_tput_per_gpu="
            f"{float(aggregate['derived_step_tput_per_gpu']):.2f}"
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
