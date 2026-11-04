from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import operatorx_queued_timing as queued_timing

HELPER_PATH = Path(__file__).with_name("operatorx_ascend_worker.py")
HELPER_SHA256 = hashlib.sha256(HELPER_PATH.read_bytes()).hexdigest()
_SPEC = importlib.util.spec_from_file_location(
    "operatorx_fp8_dense_helper", HELPER_PATH
)
dense = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dense)

SCHEMA_VERSION = 1
DEFAULT_CACHE_WRITE_BYTES = 256 * 1024**2
DEFAULT_WORKSPACE_RESERVE_BYTES = 2 * 1024**3
DEFAULT_REFERENCE_SAMPLES = 64
QUANT_CHECK_CHUNK_ELEMENTS = 262144
SOURCE_URL = dense.FP32_REFERENCE_SOURCE
SOURCE_SHA256 = dense.REFERENCE_SOURCE_SHA256
IMPLEMENTATION = "native_dynamic_block_quant_fp32_scales_then_quant_matmul"


def classify_case(case):
    if not isinstance(case, dict):
        return dense._unsupported("invalid_case", "Case must be a JSON object")
    op_type = case.get("opType", case.get("op_type"))
    if "opType" in case and "op_type" in case and case["opType"] != case["op_type"]:
        return dense._unsupported("invalid_case", "Conflicting operator types")
    if op_type != "gemm":
        return dense._unsupported(
            "operator_not_implemented",
            "This worker implements FP32 block-scaled E4M3 GEMM only",
        )
    args = case.get("args")
    if not isinstance(args, dict) or set(args) - {
        "m",
        "n",
        "k",
        "a",
        "b",
        "out",
        "bias",
        "activation",
    }:
        return dense._unsupported("invalid_args", "Unknown or invalid GEMM arguments")
    if any(not dense._positive_int(args.get(key)) for key in ("m", "n", "k")):
        return dense._unsupported(
            "invalid_dimensions", "M, N and K must be positive integers"
        )
    for key, group, static in (("a", [1, 128], False), ("b", [128, 128], True)):
        operand = args.get(key)
        if (
            not isinstance(operand, dict)
            or set(operand) != {"dtype", "scale"}
            or operand["dtype"] != "e4m3"
        ):
            return dense._unsupported(
                "operand_format_not_implemented",
                "Only exact E4M3 descriptors with one FP32 block scale are implemented",
            )
        scale = operand["scale"]
        if (
            not isinstance(scale, dict)
            or set(scale) != {"dtype", "group", "static"}
            or scale["dtype"] != "fp32"
            or type(scale["static"]) is not bool
            or scale["static"] is not static
            or not isinstance(scale["group"], list)
            or len(scale["group"]) != 2
            or any(type(value) is not int for value in scale["group"])
            or scale["group"] != group
        ):
            return dense._unsupported(
                "scale_semantics_not_implemented",
                "Require dynamic A FP32 [1,128] and static W FP32 [128,128] scales",
            )
    if args.get("out", "bf16") != "bf16" or not isinstance(
        args.get("out", "bf16"), str
    ):
        return dense._unsupported(
            "output_not_implemented", "This FP8 path returns BF16"
        )
    if (
        type(args.get("bias", False)) is not bool
        or args.get("bias", False) is not False
        or args.get("activation") is not None
    ):
        return dense._unsupported(
            "epilogue_not_implemented", "Bias and activation are not implemented"
        )
    if args["k"] % 128:
        return dense._unsupported(
            "k_group_tail_requires_validation",
            "K must be divisible by 128; partial final K groups need separate validation",
        )
    return {
        "supported": True,
        "status": "supported_unvalidated",
        "reason_code": "native_fp8_block_candidate",
        "reason": "Exact declared block format; every device case still requires numerical validation",
        "normalized": {
            "m": args["m"],
            "n": args["n"],
            "k": args["k"],
            "dtype": "bf16",
            "dtype_out": "bf16",
            "tolerance_dtype": "bf16",
            "implementation": IMPLEMENTATION,
            "bias": False,
            "dtype_a_input": "bf16",
            "dtype_a_quantized": "e4m3",
            "dtype_b": "e4m3",
            "scale_dtype": "fp32",
            "group_sizes": [1, 128, 128],
            "layout": "A[M,K], W[N,K], Wscale[ceil(N/128),K/128]; compute A_quant @ W.T with FP32 scales",
        },
    }


def estimate_memory(
    case,
    *,
    cache_write_bytes=DEFAULT_CACHE_WRITE_BYTES,
    workspace_reserve_bytes=DEFAULT_WORKSPACE_RESERVE_BYTES,
):
    support = classify_case(case)
    if not support["supported"]:
        raise ValueError(support["reason"])
    if any(
        type(value) is not int or value < 0
        for value in (cache_write_bytes, workspace_reserve_bytes)
    ):
        raise ValueError("Cache/workspace budgets must be nonnegative integers")
    m, n, k = (support["normalized"][key] for key in ("m", "n", "k"))
    memory = {
        "a_bytes": 2 * m * k,
        "weight_bytes": n * k,
        "weight_scale_bytes": 4 * ((n + 127) // 128) * (k // 128),
        "quantized_a_bytes": m * k,
        "activation_scale_bytes": 4 * m * (k // 128),
        "output_bytes": 2 * m * n,
        "output_live_buffer_count": 2,
        "quantized_input_live_buffer_count": 2,
        "cache_write_bytes": cache_write_bytes,
        "workspace_reserve_bytes": workspace_reserve_bytes,
        "workspace_is_measured": False,
    }
    memory["tensor_and_cache_bytes"] = (
        memory["a_bytes"]
        + memory["weight_bytes"]
        + memory["weight_scale_bytes"]
        + cache_write_bytes
        + 2
        * (
            memory["quantized_a_bytes"]
            + memory["activation_scale_bytes"]
            + memory["output_bytes"]
        )
    )
    memory["input_generation_peak_bytes"] = max(
        4 * n * k,
        memory["a_bytes"] + memory["weight_bytes"] + memory["weight_scale_bytes"],
    )
    memory["estimated_required_bytes"] = (
        max(memory["tensor_and_cache_bytes"], memory["input_generation_peak_bytes"])
        + workspace_reserve_bytes
    )
    memory["required_bytes"] = memory["estimated_required_bytes"]
    memory["reference_cpu_working_bytes_estimate"] = (
        48 * max(k, QUANT_CHECK_CHUNK_ELEMENTS) + 32 * k
    )
    return memory


def _base_result(case):
    semantic = {
        "opType": case.get("opType", case.get("op_type")),
        "args": case.get("args"),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case.get("case_id", case.get("id")),
        "opType": semantic["opType"],
        "args": case.get("args"),
        "sources": case.get("sources", []),
        "testlists": case.get("testlists", []),
        "support": classify_case(case),
        "status": "unvalidated",
        "metrics": None,
        "warnings": [
            "Prepared candidate; a plan is not a hardware measurement.",
            "Native quantizer rounding and zero-group scale differ from the pinned CUDA quantizer; byte parity is not claimed.",
            "Cache writes do not prove cache eviction or public protocol parity.",
            "GEMM correctness uses sampled CPU FP64 dots; quantization quality and output finiteness scan all elements.",
        ],
        "provenance": {
            "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dense_helper_sha256": HELPER_SHA256,
            "case_semantic_sha256": hashlib.sha256(
                json.dumps(
                    semantic, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode()
            ).hexdigest(),
            "python": platform.python_version(),
            "machine": platform.machine(),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "native_kernel_result_not_rewritten_for_validation": True,
            "implementation": IMPLEMENTATION,
            "cuda_quantizer_byte_parity_claimed": False,
            "min_scale": 0.0,
            "native_zero_scale_for_zero_group_allowed": True,
        },
    }


def plan_case(
    case,
    *,
    cache_write_bytes=DEFAULT_CACHE_WRITE_BYTES,
    workspace_reserve_bytes=DEFAULT_WORKSPACE_RESERVE_BYTES,
):
    result = _base_result(case)
    if not result["support"]["supported"]:
        result["status"] = "unsupported"
    else:
        result["memory_estimate"] = estimate_memory(
            case,
            cache_write_bytes=cache_write_bytes,
            workspace_reserve_bytes=workspace_reserve_bytes,
        )
    return result


def native_parameters(torch):
    wanted = {
        "npu::npu_dynamic_block_quant": {
            "x",
            "min_scale",
            "round_mode",
            "dst_type",
            "row_block_size",
            "col_block_size",
        },
        "npu::npu_quant_matmul": {
            "x1",
            "x2",
            "scale",
            "offset",
            "pertoken_scale",
            "bias",
            "output_dtype",
            "group_sizes",
        },
    }
    extensions = {
        "npu::npu_dynamic_block_quant": {"dst_type_max"},
        "npu::npu_quant_matmul": {
            "x1_dtype",
            "x2_dtype",
            "pertoken_scale_dtype",
            "scale_dtype",
            "y_scale",
        },
    }
    metadata, names = {}, {}
    for op, required in wanted.items():
        schemas = [
            schema
            for schema in torch._C._jit_get_schemas_for_operator(op)
            if schema.overload_name == ""
        ]
        if len(schemas) != 1:
            raise ValueError(f"Missing or ambiguous native schema for {op}")
        schema = schemas[0]
        args = list(schema.arguments)
        names[op] = {arg.name for arg in args}
        leading = ["x"] if op.endswith("dynamic_block_quant") else ["x1", "x2", "scale"]
        if (
            [arg.name for arg in args[: len(leading)]] != leading
            or any(str(arg.type) != "Tensor" for arg in args[: len(leading)])
            or not required <= names[op]
            or names[op] - required - extensions[op]
            or any(
                not arg.has_default_value()
                for arg in args
                if arg.name in extensions[op]
            )
        ):
            raise ValueError(f"Unrecognized native operator signature for {op}")
        metadata[op] = str(schema)
    quant = {
        "min_scale": 0.0,
        "round_mode": "rint",
        "dst_type": torch.float8_e4m3fn,
        "row_block_size": 1,
        "col_block_size": 128,
    }
    if "dst_type_max" in names["npu::npu_dynamic_block_quant"]:
        quant["dst_type_max"] = 0.0
    mm = {
        "offset": None,
        "bias": None,
        "output_dtype": torch.bfloat16,
        "group_sizes": [1, 128, 128],
    }
    for name in names["npu::npu_quant_matmul"] & extensions["npu::npu_quant_matmul"]:
        mm[name] = None
    return quant, mm, metadata


def generate_inputs(torch, spec, *, device):
    weight = (
        torch.randn(spec["n"], spec["k"], device=device, dtype=torch.bfloat16) * 0.5
    ).to(torch.float8_e4m3fn)
    weight_scale = torch.full(
        ((spec["n"] + 127) // 128, spec["k"] // 128),
        0.01,
        device=device,
        dtype=torch.float32,
    )
    a = torch.randn(spec["m"], spec["k"], device=device, dtype=torch.bfloat16)
    recipe = {
        "source_url": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "source_functions": "_prepare_gemm -> _fill",
        "weight_recipe": "BF16 randn(N,K) * 0.5 in BF16, then E4M3 cast",
        "weight_scale_recipe": "FP32 full(ceil(N/128),K/128,0.01)",
        "a_recipe": "BF16 randn(M,K) after weight/scales",
        "generation_order": ["weight", "weight_scale", "a"],
        "public_input_distribution_reproduced": True,
        "cross_stack_rng_identity_claimed": False,
        "parameter_allocation_copy_omitted": True,
        "weights_static_outside_timer": True,
        "activation_quantization_inside_timer": True,
    }
    return a, weight, weight_scale, recipe


def execute_native(torch_npu, a, weight, weight_scale, quant_kwargs, mm_kwargs):
    quantized, scales = torch_npu.npu_dynamic_block_quant(a, **quant_kwargs)
    output = torch_npu.npu_quant_matmul(
        quantized, weight.t(), weight_scale.t(), pertoken_scale=scales, **mm_kwargs
    )
    return output, quantized, scales


def validate_quantization(
    torch, a, quantized, scales, *, chunk_elements=QUANT_CHECK_CHUNK_ELEMENTS
):
    if (
        a.dtype != torch.bfloat16
        or quantized.dtype != torch.float8_e4m3fn
        or scales.dtype != torch.float32
    ):
        raise ValueError("Unexpected activation/quantizer dtype")
    m, k = a.shape
    if (
        k % 128
        or list(quantized.shape) != [m, k]
        or list(scales.shape) != [m, k // 128]
    ):
        raise ValueError("Unexpected activation/quantizer shape")
    if not quantized.is_contiguous() or not scales.is_contiguous():
        raise ValueError("Unexpected native quantizer output strides")
    rows_per_chunk = max(1, chunk_elements // k)
    bad_elements = bad_groups = zero_groups = 0
    max_error = max_ratio = 0.0
    finite = True
    for start in range(0, m, rows_per_chunk):
        end = min(m, start + rows_per_chunk)
        original = (
            a[start:end].detach().cpu().float().reshape(end - start, k // 128, 128)
        )
        codes = (
            quantized[start:end]
            .detach()
            .cpu()
            .float()
            .reshape(end - start, k // 128, 128)
        )
        ss = scales[start:end].detach().cpu()
        chunk_finite = (
            bool(torch.isfinite(original).all())
            and bool(torch.isfinite(codes).all())
            and bool(torch.isfinite(ss).all())
        )
        finite &= chunk_finite
        if not chunk_finite:
            bad_elements += (end - start) * k
            continue
        zero = (ss == 0) & (original.abs().amax(-1) == 0) & (codes.abs().amax(-1) == 0)
        valid = (ss > 0) | zero
        bad_groups += int((~valid).sum().item())
        zero_groups += int(zero.sum().item())
        error = (codes * ss.unsqueeze(-1) - original).abs()
        bound = original.abs() / 16.0 + ss.unsqueeze(-1) / 1024.0
        bad_elements += int((error > bound).sum().item())
        max_error = max(max_error, float(error.max().item()))
        ratio = torch.where(
            bound > 0,
            error / bound,
            torch.where(
                error == 0,
                torch.zeros_like(error),
                torch.full_like(error, float("inf")),
            ),
        )
        max_ratio = max(max_ratio, float(ratio.max().item()))
    return {
        "passed": finite and bad_elements == 0 and bad_groups == 0,
        "all_quantized_inputs_finite": finite,
        "checked_elements": m * k,
        "checked_groups": m * (k // 128),
        "invalid_scale_groups": bad_groups,
        "zero_scale_zero_groups": zero_groups,
        "failed_elements": bad_elements,
        "max_abs_reconstruction_error": max_error if finite else None,
        "max_error_bound_ratio": max_ratio if math.isfinite(max_ratio) else None,
        "bound": "abs(decoded-original) <= abs(original)/16 + scale/1024; zero scale valid only for an all-zero original and payload group",
        "scope": "Every activation element checked on CPU in bounded chunks; CUDA payload/scale equality is not required",
    }


def sampled_decoded_reference(
    torch, quantized, scales, weight, weight_scale, coordinates
):
    values = []
    for row, col in coordinates:
        q = quantized[row].detach().cpu().to(torch.float64)
        s = scales[row].detach().cpu().to(torch.float64).repeat_interleave(128)
        w = weight[col].detach().cpu().to(torch.float64)
        ws = (
            weight_scale[col // 128]
            .detach()
            .cpu()
            .to(torch.float64)
            .repeat_interleave(128)
        )
        if any(not bool(torch.isfinite(x).all()) for x in (q, s, w, ws)):
            raise ValueError("Nonfinite sampled decoded operand")
        values.append(torch.dot(q * s, w * ws).item())
    return values


def validate_state(torch, a, weight, weight_scale, state, spec, coordinates):
    output, quantized, scales = state
    quant = validate_quantization(torch, a, quantized, scales)
    reference = sampled_decoded_reference(
        torch, quantized, scales, weight, weight_scale, coordinates
    )
    check = dense.validate_output(
        torch,
        output,
        normalized=spec,
        coordinates=coordinates,
        reference=reference,
        input_product_scale=0.005,
    )
    check["quantization"] = quant
    check["reference_scope"] = (
        "CPU FP64 full-K dots of this kernel call's E4M3 payloads with their FP32 block scales"
    )
    check["passed"] = check["passed"] and quant["passed"]
    return check


def _run_native(case, config, result):
    import torch
    import torch_npu

    npu = torch.npu
    if not npu.is_available():
        raise dense.WorkerFailure("hardware_unavailable", "No available Ascend NPU")
    quant_kwargs, mm_kwargs, schemas = native_parameters(torch)
    device = f"npu:{config['device']}"
    npu.set_device(device)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    npu.manual_seed(config["seed"])
    npu.synchronize()
    result["provenance"].update(
        torch=str(torch.__version__),
        torch_npu=str(torch_npu.__version__),
        native_schemas=schemas,
        quantizer_parameters={
            key: str(value) if key == "dst_type" else value
            for key, value in quant_kwargs.items()
        },
        matmul_parameters={
            key: str(value) if key == "output_dtype" else value
            for key, value in mm_kwargs.items()
        },
    )
    free, total = npu.mem_get_info(config["device"])
    budget = dense.check_memory_budget(
        result["memory_estimate"],
        available_bytes=int(free),
        memory_budget_bytes=config["memory_budget_bytes"],
    )
    result["memory_budget_check"] = {**budget, "total_device_bytes": int(total)}
    if not budget["fits"]:
        raise dense.WorkerFailure(
            "memory_budget_exceeded",
            "FP8 input/transient/output/cache/workspace estimate exceeds the available budget",
        )
    npu.reset_peak_memory_stats(config["device"])
    baseline = {
        "allocated_bytes": int(npu.memory_allocated(config["device"])),
        "reserved_bytes": int(npu.memory_reserved(config["device"])),
    }
    try:
        spec = result["support"]["normalized"]
        a, weight, weight_scale, recipe = generate_inputs(torch, spec, device=device)
        cache = torch.empty(
            config["cache_write_bytes"], device=device, dtype=torch.uint8
        )
        result["input_generation"] = {**recipe, "seed": config["seed"]}
        result["tensors"] = {
            "a": dense.tensor_descriptor(a),
            "weight": dense.tensor_descriptor(weight),
            "weight_scale": dense.tensor_descriptor(weight_scale),
            "cache_write": dense.tensor_descriptor(cache),
        }
        coordinates = dense.sample_coordinates(
            spec["m"], spec["n"], config["reference_samples"], config["seed"]
        )
        state = execute_native(
            torch_npu, a, weight, weight_scale, quant_kwargs, mm_kwargs
        )
        npu.synchronize()
        before = validate_state(
            torch, a, weight, weight_scale, state, spec, coordinates
        )
        result["correctness"] = {"before_timing": before, "after_timing": None}
        if not before["passed"]:
            raise dense.WorkerFailure(
                "correctness_failed", "Pre-timing quantization/GEMM reference failed"
            )
        del state  # Retain at most two live result buffers during queued execution.
        state, timing = queued_timing.time_op(
            npu,
            lambda: execute_native(
                torch_npu, a, weight, weight_scale, quant_kwargs, mm_kwargs
            ),
            cache,
            config,
        )
        timing.update(
            timed_implementation=IMPLEMENTATION,
            activation_quantization_inside_timer=True,
            static_weight_preparation_inside_timer=False,
        )
        result["timing"] = timing
        latency = dense.upper_median(timing["device_event_samples_us"])
        after = validate_state(torch, a, weight, weight_scale, state, spec, coordinates)
        result["correctness"]["after_timing"] = after
        result["tensors"].update(
            output=dense.tensor_descriptor(state[0]),
            quantized_a=dense.tensor_descriptor(state[1]),
            activation_scale=dense.tensor_descriptor(state[2]),
        )
        if not after["passed"]:
            raise dense.WorkerFailure(
                "correctness_failed", "Post-timing quantization/GEMM reference failed"
            )
        result["metrics"] = {
            "latency_us": latency,
            "queued_wall_latency_us_per_call": timing["queued_wall_us_per_call"],
            "tflops": 2 * spec["m"] * spec["n"] * spec["k"] / (latency * 1e6),
            "npu_graph": False,
        }
        result["status"] = "ok"
        result["warnings"].pop(0)
    finally:
        result["memory_high_water"] = {
            "baseline": baseline,
            "max_allocated_bytes": int(npu.max_memory_allocated(config["device"])),
            "max_reserved_bytes": int(npu.max_memory_reserved(config["device"])),
            "scope": "PyTorch NPU allocator, including generation and two live result tuples; opaque driver memory excluded",
        }


def run_case(
    case,
    *,
    memory_budget_bytes,
    device=0,
    samples=10,
    warmups=5,
    warmup_ms=25,
    seed=20261023,
    reference_samples=DEFAULT_REFERENCE_SAMPLES,
    cache_write_bytes=DEFAULT_CACHE_WRITE_BYTES,
    workspace_reserve_bytes=DEFAULT_WORKSPACE_RESERVE_BYTES,
):
    result = plan_case(
        case,
        cache_write_bytes=cache_write_bytes,
        workspace_reserve_bytes=workspace_reserve_bytes,
    )
    if result["status"] == "unsupported":
        return result
    config = {
        "memory_budget_bytes": memory_budget_bytes,
        "device": device,
        "samples": samples,
        "warmups": warmups,
        "warmup_ms": warmup_ms,
        "seed": seed,
        "reference_samples": reference_samples,
        "cache_write_bytes": cache_write_bytes,
        "workspace_reserve_bytes": workspace_reserve_bytes,
    }
    result["config"] = config
    try:
        if (
            not dense._positive_int(memory_budget_bytes)
            or type(device) is not int
            or device < 0
        ):
            raise ValueError(
                "Positive memory budget and nonnegative device are required"
            )
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("Invalid seed")
        if (
            type(samples) is not int
            or not 10 <= samples <= 10000
            or type(warmups) is not int
            or not 1 <= warmups <= 10000
        ):
            raise ValueError("Invalid timing sample/warmup count")
        if (
            type(warmup_ms) not in (int, float)
            or not math.isfinite(warmup_ms)
            or not 0 <= warmup_ms <= 10000
        ):
            raise ValueError("Invalid warmup duration")
        if type(reference_samples) is not int or not 1 <= reference_samples <= 4096:
            raise ValueError("Invalid reference sample count")
        _run_native(case, config, result)
    except Exception as exc:
        result["status"] = (
            "skipped_memory"
            if isinstance(exc, dense.WorkerFailure)
            and exc.kind == "memory_budget_exceeded"
            else "error"
        )
        result["metrics"] = None
        result["error"] = {
            "kind": exc.kind
            if isinstance(exc, dense.WorkerFailure)
            else "execution_failure",
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
        print(f"OperatorX FP8 worker {result['error']['kind']}: {exc}", file=sys.stderr)
    result["provenance"]["finished_at"] = datetime.now(timezone.utc).isoformat()
    return result
