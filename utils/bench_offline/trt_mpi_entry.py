"""MPI worker entry shim for the pinned TRT perfect-router environment."""

from __future__ import annotations

import json
import os
import queue
import time
from pathlib import Path
from typing import Any


def _emit_rank_event(event: str, **details: Any) -> None:
    rank = os.getenv("OMPI_COMM_WORLD_RANK", "unknown")
    _write_marker(event=event, **details)
    detail_text = " ".join(
        f"{name}={value}" for name, value in details.items()
    )
    suffix = f" {detail_text}" if detail_text else ""
    print(
        f"[offline-trt-mpi] rank={rank} event={event}{suffix}",
        flush=True,
    )


def _install_engine_warmup_shape_cap() -> dict[str, Any]:
    max_warmup_tokens = int(
        os.environ["TRTLLM_BENCH_ENGINE_WARMUP_MAX_TOKENS"]
    )
    if max_warmup_tokens <= 0:
        raise RuntimeError(
            "TRT engine warmup shape cap needs a positive limit"
        )

    from tensorrt_llm._torch.pyexecutor import model_engine

    model_engine_class = model_engine.PyTorchModelEngine
    create_warmup_request = model_engine_class._create_warmup_request
    cap_already_installed = getattr(
        create_warmup_request,
        "_offline_engine_warmup_shape_cap",
        False,
    )
    if not cap_already_installed:

        def bounded_warmup_request(
            engine: Any,
            resource_manager: Any,
            num_tokens: int,
            num_gen_requests: int,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            requested_tokens = int(num_tokens)
            bounded_tokens = (
                min(requested_tokens, max_warmup_tokens)
                if int(num_gen_requests) == 0
                else requested_tokens
            )
            if bounded_tokens != requested_tokens:
                _emit_rank_event(
                    "engine_warmup_shape_capped",
                    requested_tokens=requested_tokens,
                    tuned_tokens=bounded_tokens,
                )
            return create_warmup_request(
                engine,
                resource_manager,
                bounded_tokens,
                num_gen_requests,
                *args,
                **kwargs,
            )

        bounded_warmup_request._offline_engine_warmup_shape_cap = True
        model_engine_class._create_warmup_request = bounded_warmup_request

    warmup = model_engine_class.warmup
    trace_already_installed = getattr(
        warmup,
        "_offline_engine_warmup_trace",
        False,
    )
    if not trace_already_installed:

        def traced_warmup(
            engine: Any,
            resource_manager: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            runtime_max_num_tokens = int(engine.max_num_tokens)
            started_at = time.monotonic()
            _emit_rank_event(
                "engine_warmup_start",
                runtime_max_tokens=runtime_max_num_tokens,
                max_context_warmup_tokens=max_warmup_tokens,
            )
            warmup_error: BaseException | None = None
            try:
                return warmup(engine, resource_manager, *args, **kwargs)
            except BaseException as error:
                warmup_error = error
                raise
            finally:
                _emit_rank_event(
                    (
                        "engine_warmup_complete"
                        if warmup_error is None
                        else "engine_warmup_error"
                    ),
                    elapsed_seconds=f"{time.monotonic() - started_at:.3f}",
                    runtime_max_tokens=runtime_max_num_tokens,
                    **(
                        {}
                        if warmup_error is None
                        else {"error_type": type(warmup_error).__name__}
                    ),
                )

        traced_warmup._offline_engine_warmup_trace = True
        model_engine_class.warmup = traced_warmup

    return {
        "max_warmup_tokens": max_warmup_tokens,
        "target": (
            f"{model_engine_class.__name__}._create_warmup_request"
        ),
        "trace_target": f"{model_engine_class.__name__}.warmup",
        "already_installed": (
            cap_already_installed and trace_already_installed
        ),
    }


def _install_attention_workspace_preallocation() -> dict[str, Any]:
    target_bytes = int(
        os.environ["TRTLLM_BENCH_ATTENTION_WORKSPACE_BYTES"]
    )
    if target_bytes < 0:
        raise RuntimeError(
            "TRT attention workspace reservation cannot be negative"
        )
    if target_bytes == 0:
        return {
            "enabled": False,
            "target_bytes": 0,
            "target": None,
            "already_installed": False,
        }

    from tensorrt_llm._torch.pyexecutor import model_engine

    model_engine_class = model_engine.PyTorchModelEngine
    set_up_metadata = model_engine_class._set_up_attn_metadata
    installed_target = getattr(
        set_up_metadata,
        "_offline_attention_workspace_target_bytes",
        None,
    )
    if installed_target is not None:
        if int(installed_target) != target_bytes:
            raise RuntimeError(
                "TRT attention workspace hook already has target "
                f"{installed_target}, requested {target_bytes}"
            )
        return {
            "enabled": True,
            "target_bytes": target_bytes,
            "target": (
                f"{model_engine_class.__name__}._set_up_attn_metadata"
            ),
            "already_installed": True,
        }

    def set_up_metadata_with_reserved_workspace(
        engine: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        metadata = set_up_metadata(engine, *args, **kwargs)
        # No-cache metadata is transient. Reserve the large buffer only on the
        # cached runtime metadata used by context prefill and calibration.
        if getattr(engine, "attn_metadata", None) is not metadata:
            return metadata
        workspace = getattr(metadata, "workspace", None)
        if workspace is None:
            raise RuntimeError(
                "TRT attention metadata did not create an eager workspace"
            )
        element_size = int(workspace.element_size())
        if element_size <= 0:
            raise RuntimeError(
                "TRT attention workspace has an invalid element size"
            )
        previous_bytes = int(workspace.numel()) * element_size
        if previous_bytes >= target_bytes:
            return metadata
        target_elements = (
            target_bytes + element_size - 1
        ) // element_size
        reserved = workspace.new_empty((target_elements,))
        allocated_bytes = int(reserved.numel()) * int(
            reserved.element_size()
        )
        metadata.workspace = reserved
        cuda_graph_workspace = getattr(
            metadata,
            "cuda_graph_workspace",
            None,
        )
        cuda_graph_workspace_bytes = (
            0
            if cuda_graph_workspace is None
            else int(cuda_graph_workspace.numel())
            * int(cuda_graph_workspace.element_size())
        )
        _emit_rank_event(
            "attention_workspace_preallocated",
            runtime_max_tokens=int(engine.max_num_tokens),
            previous_bytes=previous_bytes,
            target_bytes=target_bytes,
            allocated_bytes=allocated_bytes,
            cuda_graph_workspace_bytes=cuda_graph_workspace_bytes,
        )
        return metadata

    setattr(
        set_up_metadata_with_reserved_workspace,
        "_offline_attention_workspace_target_bytes",
        target_bytes,
    )
    model_engine_class._set_up_attn_metadata = (
        set_up_metadata_with_reserved_workspace
    )
    return {
        "enabled": True,
        "target_bytes": target_bytes,
        "target": f"{model_engine_class.__name__}._set_up_attn_metadata",
        "already_installed": False,
    }


def _install_kv_prefill_memory_reserve() -> dict[str, Any]:
    reserve_bytes = int(
        os.environ["TRTLLM_BENCH_KV_PREFILL_RESERVE_BYTES"]
    )
    minimum_tokens = int(
        os.environ["TRTLLM_BENCH_MIN_RUNTIME_KV_TOKENS"]
    )
    if reserve_bytes < 0:
        raise RuntimeError("TRT KV prefill reserve cannot be negative")
    if minimum_tokens <= 0:
        raise RuntimeError(
            "TRT minimum runtime KV capacity needs a positive token count"
        )
    if reserve_bytes == 0:
        return {
            "enabled": False,
            "reserve_bytes": 0,
            "minimum_tokens": minimum_tokens,
            "target": None,
            "already_installed": False,
        }

    from tensorrt_llm._torch.pyexecutor import _util

    creator_class = _util.KvCacheCreator
    configure_capacity = creator_class.configure_kv_cache_capacity
    installed_reserve = getattr(
        configure_capacity,
        "_offline_kv_prefill_reserve_bytes",
        None,
    )
    installed_minimum = getattr(
        configure_capacity,
        "_offline_kv_prefill_minimum_tokens",
        None,
    )
    if installed_reserve is not None or installed_minimum is not None:
        if (
            installed_reserve != reserve_bytes
            or installed_minimum != minimum_tokens
        ):
            raise RuntimeError(
                "TRT KV prefill reserve hook already has reserve/minimum "
                f"{installed_reserve}/{installed_minimum}, requested "
                f"{reserve_bytes}/{minimum_tokens}"
            )
        return {
            "enabled": True,
            "reserve_bytes": reserve_bytes,
            "minimum_tokens": minimum_tokens,
            "target": (
                f"{creator_class.__name__}.configure_kv_cache_capacity"
            ),
            "already_installed": True,
        }

    def configure_with_prefill_reserve(
        creator: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        result = configure_capacity(creator, *args, **kwargs)
        configured_raw = creator._kv_cache_config.max_gpu_total_bytes
        if configured_raw is None:
            raise RuntimeError(
                "TRT KV capacity estimator did not set max_gpu_total_bytes"
            )
        configured_bytes = int(configured_raw)
        cache_cost = creator._get_kv_size_per_token()
        minimum_bytes = int(cache_cost.bytes_for_tokens(minimum_tokens))
        adjusted_bytes = configured_bytes - reserve_bytes
        if adjusted_bytes < minimum_bytes:
            raise RuntimeError(
                "Cannot reserve full-prefill transient memory while "
                "preserving the fixed-batch KV capacity: "
                f"configured_bytes={configured_bytes} "
                f"reserve_bytes={reserve_bytes} "
                f"adjusted_bytes={adjusted_bytes} "
                f"minimum_tokens={minimum_tokens} "
                f"minimum_bytes={minimum_bytes}"
            )
        creator._kv_cache_config.max_gpu_total_bytes = adjusted_bytes
        _emit_rank_event(
            "kv_prefill_reserve_applied",
            configured_bytes=configured_bytes,
            reserve_bytes=reserve_bytes,
            adjusted_bytes=adjusted_bytes,
            minimum_runtime_kv_tokens=minimum_tokens,
            minimum_runtime_kv_bytes=minimum_bytes,
        )
        return result

    setattr(
        configure_with_prefill_reserve,
        "_offline_kv_prefill_reserve_bytes",
        reserve_bytes,
    )
    setattr(
        configure_with_prefill_reserve,
        "_offline_kv_prefill_minimum_tokens",
        minimum_tokens,
    )
    creator_class.configure_kv_cache_capacity = (
        configure_with_prefill_reserve
    )
    return {
        "enabled": True,
        "reserve_bytes": reserve_bytes,
        "minimum_tokens": minimum_tokens,
        "target": f"{creator_class.__name__}.configure_kv_cache_capacity",
        "already_installed": False,
    }


def _install_large_prefill_fp8_quant_guard() -> dict[str, Any]:
    max_fused_rows = int(
        os.environ["TRTLLM_BENCH_FP8_FUSED_QUANT_MAX_ROWS"]
    )
    if max_fused_rows <= 0:
        raise RuntimeError(
            "Large-prefill FP8 quant guard needs a positive row limit"
        )

    from tensorrt_llm._torch.custom_ops import torch_custom_ops

    runner_class = torch_custom_ops.Fp8QuantKernelRunner
    tactic_selector = runner_class.get_valid_tactics
    tactic_filter_installed = getattr(
        tactic_selector,
        "_offline_large_prefill_guard",
        False,
    )
    if not tactic_filter_installed:

        def guarded_get_valid_tactics(
            runner: Any,
            inputs: list[Any],
            profile: Any,
            **kwargs: Any,
        ) -> list[Any]:
            tactics = tactic_selector(runner, inputs, profile, **kwargs)
            rows = int(inputs[0].shape[0])
            if rows <= max_fused_rows:
                return tactics
            triton_tactic = runner.TACTIC_TRITON
            if triton_tactic not in tactics:
                raise RuntimeError(
                    "TRT large-prefill FP8 quantization has no Triton tactic"
                )
            return [triton_tactic]

        guarded_get_valid_tactics._offline_large_prefill_guard = True
        runner_class.get_valid_tactics = guarded_get_valid_tactics

    quantize = torch_custom_ops._fp8_quantize_1x128_ue8m0
    dispatch_guard_installed = getattr(
        quantize,
        "_offline_large_prefill_guard",
        False,
    )
    if not dispatch_guard_installed:

        def guarded_quantize(input_tensor: Any, tactic: int) -> Any:
            if int(input_tensor.shape[0]) > max_fused_rows:
                tactic = runner_class.TACTIC_TRITON
            return quantize(input_tensor, tactic)

        guarded_quantize._offline_large_prefill_guard = True
        torch_custom_ops._fp8_quantize_1x128_ue8m0 = guarded_quantize

    return {
        "max_fused_rows": max_fused_rows,
        "already_installed": (
            tactic_filter_installed and dispatch_guard_installed
        ),
    }


def _install_large_prefill_fp8_gemm_chunking() -> dict[str, Any]:
    max_chunk_rows = int(
        os.environ["TRTLLM_BENCH_FP8_DEEP_GEMM_MAX_ROWS"]
    )
    if max_chunk_rows <= 0:
        raise RuntimeError(
            "Large-prefill FP8 GEMM chunking needs a positive row limit"
        )

    from tensorrt_llm._torch.custom_ops import torch_custom_ops

    runner_class = torch_custom_ops.fp8SwapABGemmRunner
    forward = runner_class.forward
    installed_limit = getattr(
        forward,
        "_offline_fp8_deep_gemm_max_rows",
        None,
    )
    if installed_limit is not None:
        if int(installed_limit) != max_chunk_rows:
            raise RuntimeError(
                "TRT large-prefill FP8 GEMM hook already has row limit "
                f"{installed_limit}, requested {max_chunk_rows}"
            )
        return {
            "max_chunk_rows": max_chunk_rows,
            "target": f"{runner_class.__name__}.forward",
            "synchronize_chunks": True,
            "already_installed": True,
        }

    reported_rows: set[int] = set()

    def chunked_forward(
        runner: Any,
        inputs: list[Any],
        tactic: int = -1,
    ) -> Any:
        input_tensor, weight, weight_scale = inputs
        rows = int(input_tensor.size(0))
        if rows <= max_chunk_rows:
            return forward(runner, inputs, tactic=tactic)

        output_features = int(weight.size(0))
        chunk_count = (
            rows + max_chunk_rows - 1
        ) // max_chunk_rows
        output = input_tensor.new_empty(
            (rows, output_features),
            dtype=runner.output_dtype,
        )
        trace_shape = rows not in reported_rows
        if trace_shape:
            _emit_rank_event(
                "fp8_prefill_gemm_chunking_start",
                rows=rows,
                input_features=int(input_tensor.size(1)),
                output_features=output_features,
                max_chunk_rows=max_chunk_rows,
                chunks=chunk_count,
                quant_tactic=int(runner.quant_tactic),
            )

        for chunk_index, start_row in enumerate(
            range(0, rows, max_chunk_rows)
        ):
            end_row = min(start_row + max_chunk_rows, rows)
            input_chunk = input_tensor[start_row:end_row]
            output_chunk = output[start_row:end_row]
            try:
                act, act_scale = (
                    torch_custom_ops._fp8_quantize_1x128_ue8m0(
                        input_chunk,
                        runner.quant_tactic,
                    )
                )
                torch_custom_ops.deep_gemm.fp8_gemm_nt(
                    (act, act_scale),
                    (weight, weight_scale),
                    output_chunk,
                    disable_ue8m0_cast=runner.disable_ue8m0_cast,
                )
                # This path is prefill-only. Synchronizing makes a pinned
                # kernel fault surface at the exact failing chunk instead of
                # a later allocation or sampler event.
                torch_custom_ops.torch.cuda.synchronize(
                    input_tensor.device
                )
            except BaseException as error:
                if trace_shape:
                    _emit_rank_event(
                        "fp8_prefill_gemm_chunk_error",
                        rows=rows,
                        chunk_index=chunk_index,
                        start_row=start_row,
                        end_row=end_row,
                        error_type=type(error).__name__,
                    )
                raise
            if trace_shape:
                _emit_rank_event(
                    "fp8_prefill_gemm_chunk_complete",
                    rows=rows,
                    chunk_index=chunk_index,
                    start_row=start_row,
                    end_row=end_row,
                )

        if trace_shape:
            reported_rows.add(rows)
            _emit_rank_event(
                "fp8_prefill_gemm_chunked",
                rows=rows,
                output_features=output_features,
                max_chunk_rows=max_chunk_rows,
                chunks=chunk_count,
                synchronized_chunks=True,
            )
        return output

    setattr(
        chunked_forward,
        "_offline_fp8_deep_gemm_max_rows",
        max_chunk_rows,
    )
    runner_class.forward = chunked_forward
    return {
        "max_chunk_rows": max_chunk_rows,
        "target": f"{runner_class.__name__}.forward",
        "synchronize_chunks": True,
        "already_installed": False,
    }


def _install_executor_lifecycle_trace() -> dict[str, Any]:
    from tensorrt_llm._torch.pyexecutor import py_executor

    executor_class = py_executor.PyExecutor
    installed: list[str] = []
    already_installed: list[str] = []
    for method_name, event_prefix in (
        ("_set_global_steady_clock_offset", "clock_sync"),
        ("start_worker", "executor_worker_start"),
    ):
        original = getattr(executor_class, method_name)
        if getattr(original, "_offline_lifecycle_trace", False):
            already_installed.append(method_name)
            continue

        def traced(
            executor: Any,
            *args: Any,
            _original: Any = original,
            _event_prefix: str = event_prefix,
            **kwargs: Any,
        ) -> Any:
            _emit_rank_event(f"{_event_prefix}_enter")
            try:
                result = _original(executor, *args, **kwargs)
            except BaseException as error:
                _emit_rank_event(
                    f"{_event_prefix}_error",
                    error_type=type(error).__name__,
                )
                raise
            _emit_rank_event(f"{_event_prefix}_exit")
            return result

        traced._offline_lifecycle_trace = True
        setattr(executor_class, method_name, traced)
        installed.append(method_name)
    return {
        "target": executor_class.__name__,
        "installed": installed,
        "already_installed": already_installed,
    }


def _install_fixed_batch_request_barrier() -> dict[str, Any]:
    expected = int(os.environ["TRTLLM_BENCH_GLOBAL_BATCH_SIZE"])
    arm_file_raw = os.getenv("TRTLLM_BENCH_FIXED_BATCH_ARM_FILE")
    if not arm_file_raw:
        raise RuntimeError(
            "TRTLLM_BENCH_FIXED_BATCH_ARM_FILE is required"
        )
    arm_file = Path(arm_file_raw)
    if not arm_file.is_absolute():
        raise RuntimeError(
            "Fixed-batch barrier arm file must use an absolute path"
        )
    timeout_seconds = float(
        os.getenv("TRTLLM_BENCH_FIXED_BATCH_TIMEOUT_SECONDS", "120")
    )
    if expected <= 0 or timeout_seconds <= 0:
        raise RuntimeError(
            "Fixed-batch request barrier needs positive size and timeout"
        )

    from tensorrt_llm._torch.pyexecutor import executor_request_queue

    request_queue_class = executor_request_queue.ExecutorRequestQueue
    original = request_queue_class.get_from_request_queue
    if getattr(original, "_offline_fixed_batch_barrier", False):
        return {
            "global_batch_size": expected,
            "timeout_seconds": timeout_seconds,
            "arm_file": str(arm_file),
            "armed_at_install": arm_file.is_file(),
            "already_installed": True,
        }

    armed = arm_file.is_file()

    def fixed_batch_get_from_request_queue(
        request_queue: Any,
        timeout: Any,
    ) -> list[Any]:
        nonlocal armed
        items = original(request_queue, timeout)
        # The MPI executor passes timeout=None only when there are no active or
        # waiting requests. That is the boundary between benchmark passes.
        if timeout is not None or not items:
            return items
        # TRT submits internal dummy requests while sizing the KV cache. The
        # parent creates this file only after LLM initialization has completed,
        # immediately before the benchmark's first real generate() call.
        if not armed:
            if not arm_file.is_file():
                return items
            armed = True
            _emit_rank_event(
                "fixed_batch_barrier_armed",
                arm_file=str(arm_file),
                global_batch_size=expected,
            )
        if any(not item.is_normal_request for item in items):
            return items
        if len(items) > expected:
            raise RuntimeError(
                "Fixed-batch barrier received too many requests: "
                f"{len(items)} > {expected}"
            )

        deadline = time.monotonic() + timeout_seconds
        while len(items) < expected:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    "Fixed-batch barrier timed out after "
                    f"{timeout_seconds}s with {len(items)}/{expected} "
                    "requests"
                )
            try:
                item = request_queue.request_queue.get(timeout=remaining)
            except queue.Empty as error:
                raise RuntimeError(
                    "Fixed-batch barrier timed out after "
                    f"{timeout_seconds}s with {len(items)}/{expected} "
                    "requests"
                ) from error
            if not item.is_normal_request:
                raise RuntimeError(
                    "Fixed-batch barrier received a control, cancellation, "
                    "or shutdown item before the batch was complete"
                )
            items.append(item)
        return items

    fixed_batch_get_from_request_queue._offline_fixed_batch_barrier = True
    request_queue_class.get_from_request_queue = (
        fixed_batch_get_from_request_queue
    )
    return {
        "global_batch_size": expected,
        "timeout_seconds": timeout_seconds,
        "arm_file": str(arm_file),
        "armed_at_install": armed,
        "already_installed": False,
    }


def _write_marker(event: str = "entry_ready", **details: Any) -> None:
    marker = os.getenv("TRTLLM_PERFECT_ROUTER_MARKER")
    if not marker:
        return
    expected_environment_raw = os.getenv(
        "TRTLLM_BENCH_EXPECTED_RANK_ENV",
        "{}",
    )
    try:
        expected_environment = json.loads(expected_environment_raw)
    except json.JSONDecodeError:
        expected_environment = {}
    if not isinstance(expected_environment, dict):
        expected_environment = {}
    payload = {
        "pid": os.getpid(),
        "rank": os.getenv("OMPI_COMM_WORLD_RANK"),
        "perfect_router": os.getenv("ENABLE_PERFECT_ROUTER"),
        "cute_dsl_cache_dir": os.getenv("CUTE_DSL_CACHE_DIR"),
        "benchmark_environment": {
            str(name): os.getenv(str(name))
            for name in expected_environment
        },
        "fixed_batch_global_size": os.getenv(
            "TRTLLM_BENCH_GLOBAL_BATCH_SIZE"
        ),
        "fixed_batch_arm_file": os.getenv(
            "TRTLLM_BENCH_FIXED_BATCH_ARM_FILE"
        ),
        "fixed_batch_barrier_armed": (
            Path(os.environ["TRTLLM_BENCH_FIXED_BATCH_ARM_FILE"]).is_file()
            if os.getenv("TRTLLM_BENCH_FIXED_BATCH_ARM_FILE")
            else False
        ),
        "engine_warmup_max_tokens": os.getenv(
            "TRTLLM_BENCH_ENGINE_WARMUP_MAX_TOKENS"
        ),
        "attention_workspace_target_bytes": os.getenv(
            "TRTLLM_BENCH_ATTENTION_WORKSPACE_BYTES"
        ),
        "kv_prefill_reserve_bytes": os.getenv(
            "TRTLLM_BENCH_KV_PREFILL_RESERVE_BYTES"
        ),
        "minimum_runtime_kv_tokens": os.getenv(
            "TRTLLM_BENCH_MIN_RUNTIME_KV_TOKENS"
        ),
        "fp8_fused_quant_max_rows": os.getenv(
            "TRTLLM_BENCH_FP8_FUSED_QUANT_MAX_ROWS"
        ),
        "fp8_deep_gemm_max_rows": os.getenv(
            "TRTLLM_BENCH_FP8_DEEP_GEMM_MAX_ROWS"
        ),
        "event": event,
        "source": "trt_mpi_entry",
    }
    payload.update(details)
    path = Path(marker)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def worker_main(*args: Any, **kwargs: Any) -> Any:
    """Restore benchmark aliases before importing TRT's real worker entry."""
    if os.getenv("TRTLLM_ENABLE_PERFECT_ROUTER") == "1":
        os.environ["ENABLE_PERFECT_ROUTER"] = "1"
    configurable_moe = os.getenv(
        "TRTLLM_BENCH_ENABLE_CONFIGURABLE_MOE"
    )
    if configurable_moe is not None:
        os.environ["ENABLE_CONFIGURABLE_MOE"] = configurable_moe
    cute_cache_dir = os.getenv("TRTLLM_BENCH_CUTE_DSL_CACHE_DIR")
    if cute_cache_dir:
        os.environ["CUTE_DSL_CACHE_DIR"] = cute_cache_dir
    warmup_cap = _install_engine_warmup_shape_cap()
    attention_workspace = _install_attention_workspace_preallocation()
    kv_prefill_reserve = _install_kv_prefill_memory_reserve()
    fp8_guard = _install_large_prefill_fp8_quant_guard()
    fp8_gemm_chunking = _install_large_prefill_fp8_gemm_chunking()
    lifecycle_trace = _install_executor_lifecycle_trace()
    fixed_batch_barrier = _install_fixed_batch_request_barrier()
    _write_marker()
    print(
        "[offline-trt-mpi] fixed-batch request barrier "
        f"global_batch={fixed_batch_barrier['global_batch_size']} "
        f"arm_file={fixed_batch_barrier['arm_file']} "
        f"armed_at_install={fixed_batch_barrier['armed_at_install']} "
        f"already_installed={fixed_batch_barrier['already_installed']}",
        flush=True,
    )
    print(
        "[offline-trt-mpi] synthetic engine warmup shape cap "
        f"max_tokens={warmup_cap['max_warmup_tokens']} "
        f"target={warmup_cap['target']} "
        f"trace_target={warmup_cap['trace_target']} "
        f"already_installed={warmup_cap['already_installed']}",
        flush=True,
    )
    print(
        "[offline-trt-mpi] eager attention workspace reservation "
        f"enabled={attention_workspace['enabled']} "
        f"target_bytes={attention_workspace['target_bytes']} "
        f"target={attention_workspace['target']} "
        f"already_installed={attention_workspace['already_installed']}",
        flush=True,
    )
    print(
        "[offline-trt-mpi] KV prefill transient reserve "
        f"enabled={kv_prefill_reserve['enabled']} "
        f"reserve_bytes={kv_prefill_reserve['reserve_bytes']} "
        f"minimum_tokens={kv_prefill_reserve['minimum_tokens']} "
        f"target={kv_prefill_reserve['target']} "
        f"already_installed={kv_prefill_reserve['already_installed']}",
        flush=True,
    )
    print(
        "[offline-trt-mpi] large-prefill FP8 quant guard "
        f"max_fused_rows={fp8_guard['max_fused_rows']} "
        f"already_installed={fp8_guard['already_installed']}",
        flush=True,
    )
    print(
        "[offline-trt-mpi] large-prefill FP8 GEMM chunking "
        f"max_chunk_rows={fp8_gemm_chunking['max_chunk_rows']} "
        f"target={fp8_gemm_chunking['target']} "
        f"synchronize_chunks={fp8_gemm_chunking['synchronize_chunks']} "
        f"already_installed={fp8_gemm_chunking['already_installed']}",
        flush=True,
    )
    print(
        "[offline-trt-mpi] executor lifecycle trace "
        f"target={lifecycle_trace['target']} "
        f"installed={lifecycle_trace['installed']} "
        f"already_installed={lifecycle_trace['already_installed']}",
        flush=True,
    )

    from tensorrt_llm.executor.worker import worker_main as trt_worker_main

    return trt_worker_main(*args, **kwargs)
