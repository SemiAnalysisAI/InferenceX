#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import operatorx_queued_timing as queued_timing

SCHEMA_VERSION = 1
DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}
DEFAULT_CACHE_WRITE_BYTES = 256 * 1024**2
DEFAULT_WORKSPACE_RESERVE_BYTES = 512 * 1024**2
DEFAULT_REFERENCE_SAMPLES = 64
MAX_WARMUP_LAUNCHES = 1_000_000
FINITE_CHUNK_ELEMENTS = 1024 * 1024
LAYOUT = (
    "contiguous A[M,K], contiguous W[N,K], output[M,N]; mathematical A @ W.T + bias"
)
FP32_REFERENCE_SOURCE = "https://github.com/SemiAnalysisAI/InferenceX/blob/8f03d279200eedfae08027615fdeca9f43b81cc1/experimental/operatorx/runners/common/vllm/linear.py"
REFERENCE_SOURCE_SHA256 = (
    "6c4cea138b9eb99154fb3385a77f459e04eec14bdaf3ec33f3855cea047c15a6"
)
WEIGHT_STD = 0.02


def _unsupported(code, reason):
    return {
        "supported": False,
        "status": "unsupported",
        "reason_code": code,
        "reason": reason,
        "normalized": None,
    }


def _positive_int(value):
    return type(value) is int and value > 0


def classify_case(case):
    """Pure, fail-closed support classification; supported does not mean tested."""
    if not isinstance(case, dict):
        return _unsupported("invalid_case", "Case must be a JSON object")
    op_type = case.get("opType", case.get("op_type"))
    if "opType" in case and "op_type" in case and case["opType"] != case["op_type"]:
        return _unsupported("invalid_case", "Conflicting opType and op_type")
    if op_type != "gemm":
        return _unsupported(
            "operator_not_implemented", "This worker implements only GEMM"
        )
    args = case.get("args")
    if not isinstance(args, dict):
        return _unsupported("invalid_args", "args must be a JSON object")
    unknown = set(args) - {"m", "n", "k", "a", "b", "out", "bias", "activation"}
    if unknown:
        return _unsupported(
            "unknown_semantic_arguments",
            "Unimplemented GEMM argument(s): " + ", ".join(sorted(unknown)),
        )
    for name in ("m", "n", "k"):
        if not _positive_int(args.get(name)):
            return _unsupported(
                "invalid_dimensions", f"{name} must be a positive integer"
            )
    for name in ("a", "b"):
        desc = args.get(name)
        if not isinstance(desc, dict) or "dtype" not in desc:
            return _unsupported(
                "invalid_tensor_descriptor", f"{name} must contain a dtype descriptor"
            )
        if set(desc) != {"dtype"}:
            return _unsupported(
                "tensor_semantics_not_implemented",
                f"{name}: scales, quantization and additional descriptor fields require a separate exact adapter",
            )
        if not isinstance(desc["dtype"], str) or desc["dtype"] not in DTYPE_BYTES:
            return _unsupported(
                "dtype_not_implemented",
                f"{name}: only native bf16/fp16/fp32 are implemented",
            )
    out = args.get("out", "bf16")
    if not isinstance(out, str) or out not in DTYPE_BYTES:
        return _unsupported(
            "output_dtype_not_implemented",
            "Canonical out must be the string bf16/fp16/fp32, not a descriptor object",
        )
    dtype = args["a"]["dtype"]
    dtype_b = args["b"]["dtype"]
    bias = args.get("bias", False)
    if type(bias) is not bool:
        return _unsupported(
            "bias_semantics_not_implemented",
            "bias must be a boolean (native model-dtype vector of length N)",
        )
    if args.get("activation") is not None:
        return _unsupported(
            "activation_not_implemented", "Fused activation is not implemented"
        )
    implementation = "native_linear"
    tolerance_dtype = out
    if dtype == dtype_b == out:
        pass
    elif dtype == "bf16" and dtype_b in {"bf16", "fp32"} and out == "fp32":
        if dtype_b == "bf16" and not bias:
            implementation = "native_bf16_mm_fp32_output"
        elif dtype_b == "bf16":
            implementation = "source_defined_bf16_linear_then_fp32_cast"
            tolerance_dtype = "bf16"
        else:
            implementation = "source_defined_timed_fp32_input_promotion"
    else:
        return _unsupported(
            "mixed_precision_not_implemented",
            "No exact adapter for this input/output precision combination; output casts are not a substitute for native FP32 accumulation",
        )
    return {
        "supported": True,
        "status": "supported_unvalidated",
        "reason_code": "native_linear_candidate",
        "reason": "Unquantized GEMM candidate; hardware/API execution and correctness validation are still required",
        "normalized": {
            "m": args["m"],
            "n": args["n"],
            "k": args["k"],
            "dtype": out,
            "dtype_a": dtype,
            "dtype_b": dtype_b,
            "dtype_out": out,
            "tolerance_dtype": tolerance_dtype,
            "implementation": implementation,
            "bias": bias,
            "layout": LAYOUT,
            "fp32_semantics_source": FP32_REFERENCE_SOURCE
            if implementation != "native_linear"
            else None,
        },
    }


def estimate_memory(
    case,
    *,
    cache_write_bytes=DEFAULT_CACHE_WRITE_BYTES,
    workspace_reserve_bytes=DEFAULT_WORKSPACE_RESERVE_BYTES,
):
    """Conservative live tensor estimate; reserve is a budget, not measured workspace.

    Two output buffers cover Python reassignment while the old output is alive.
    Reference/finite validation transfers use CPU chunks, not full input clones.
    Library allocator fragmentation and opaque workspace can exceed the reserve.
    """
    support = classify_case(case)
    if not support["supported"]:
        raise ValueError(support["reason"])
    for name, value in (
        ("cache_write_bytes", cache_write_bytes),
        ("workspace_reserve_bytes", workspace_reserve_bytes),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    n = support["normalized"]
    width = DTYPE_BYTES[n["dtype_out"]]
    result = {
        "a_bytes": n["m"] * n["k"] * DTYPE_BYTES[n["dtype_a"]],
        "weight_bytes": n["n"] * n["k"] * DTYPE_BYTES[n["dtype_b"]],
        "bias_bytes": n["n"] * DTYPE_BYTES[n["dtype_b"]] if n["bias"] else 0,
        "output_bytes": n["m"] * n["n"] * width,
        "output_live_buffer_count": 2,
        "timed_input_promotion_bytes": n["m"] * n["k"] * 4
        if n["implementation"] == "source_defined_timed_fp32_input_promotion"
        else 0,
        "timed_intermediate_output_bytes": n["m"] * n["n"] * 2
        if n["implementation"] == "source_defined_bf16_linear_then_fp32_cast"
        else 0,
        "cache_write_bytes": cache_write_bytes,
        "workspace_reserve_bytes": workspace_reserve_bytes,
    }
    result["tensor_and_cache_bytes"] = (
        result["a_bytes"]
        + result["weight_bytes"]
        + result["bias_bytes"]
        + 2 * result["output_bytes"]
        + cache_write_bytes
        + result["timed_input_promotion_bytes"]
        + result["timed_intermediate_output_bytes"]
    )
    # Preserve the public full-tensor randn/scaling/cast arithmetic. For plain
    # linear, out-of-place scaling overlaps random and scaled model-dtype weights.
    # For _TorchMM, FP32 random weights are scaled in place before the final cast.
    if n["implementation"] == "native_linear":
        weight_generation_peak = 2 * result["weight_bytes"]
        bias_generation_peak = result["weight_bytes"] + 2 * result["bias_bytes"]
    else:
        weight_generation_peak = n["n"] * n["k"] * 4
        if n["dtype_b"] != "fp32":
            weight_generation_peak += result["weight_bytes"]
        bias_generation_peak = result["weight_bytes"] + result["bias_bytes"]
    result["input_generation_peak_bytes"] = max(
        weight_generation_peak,
        bias_generation_peak,
        result["a_bytes"] + result["weight_bytes"] + result["bias_bytes"],
    )
    result["generation_temporary_extra_bytes"] = max(
        0, result["input_generation_peak_bytes"] - result["tensor_and_cache_bytes"]
    )
    result["estimated_required_bytes"] = (
        max(result["tensor_and_cache_bytes"], result["input_generation_peak_bytes"])
        + workspace_reserve_bytes
    )
    result["required_bytes"] = result["estimated_required_bytes"]
    # At most two FP64 K-length reference rows and a bounded finite-check CPU chunk.
    result["reference_cpu_working_bytes_estimate"] = 32 * n[
        "k"
    ] + FINITE_CHUNK_ELEMENTS * (width + 1)
    result["workspace_is_measured"] = False
    return result


def check_memory_budget(estimate, *, available_bytes, memory_budget_bytes):
    for name, value in (
        ("available_bytes", available_bytes),
        ("memory_budget_bytes", memory_budget_bytes),
    ):
        if not _positive_int(value):
            raise ValueError(f"{name} must be a positive integer")
    limit = min(available_bytes, memory_budget_bytes)
    return {
        "fits": estimate["estimated_required_bytes"] <= limit,
        "available_bytes": available_bytes,
        "memory_budget_bytes": memory_budget_bytes,
        "effective_limit_bytes": limit,
        "estimated_required_bytes": estimate["estimated_required_bytes"],
    }


def upper_median(values):
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError(
            "Timing samples must be nonempty, finite and strictly positive"
        )
    return sorted(values)[len(values) // 2]


def sample_coordinates(m, n, count, seed):
    """Seeded coverage including boundaries; no allocation proportional to M*N."""
    if not all(_positive_int(x) for x in (m, n, count)):
        raise ValueError("Reference dimensions/count must be positive integers")
    count = min(count, m * n)
    indices = []
    for index in (0, n - 1, (m - 1) * n, m * n - 1, (m // 2) * n + n // 2):
        if index not in indices and len(indices) < count:
            indices.append(index)
    rng = random.Random(seed)
    while len(indices) < count:
        index = rng.randrange(m * n)
        if index not in indices:
            indices.append(index)
    return [(index // n, index % n) for index in indices]


def error_statistics(actual, reference, *, dtype, k, input_product_scale=1.0):
    """Compare native output values with independent FP64 dot products.

    Relative tolerance is two output epsilons (four for FP32). Absolute tolerance
    is input_product_scale * 1e-5 * max(1,sqrt(K)) for cancellation near zero. The absolute
    component scales like random-input accumulation error, not like BF16 output
    epsilon; an all-zero result or broad output quantization does not pass.
    """
    if dtype not in DTYPE_BYTES or not _positive_int(k):
        raise ValueError("Invalid reference dtype/K")
    if not math.isfinite(input_product_scale) or input_product_scale <= 0:
        raise ValueError("Reference input-product scale must be finite and positive")
    if not actual or len(actual) != len(reference):
        raise ValueError("Actual and reference samples must have equal nonzero lengths")
    epsilon = {"bf16": 2**-7, "fp16": 2**-10, "fp32": 2**-23}[dtype]
    rtol = epsilon * (4 if dtype == "fp32" else 2)
    atol = input_product_scale * 1e-5 * max(1.0, math.sqrt(k))
    finite = all(math.isfinite(x) for x in (*actual, *reference))
    stats = {
        "passed": False,
        "all_samples_finite": finite,
        "sample_count": len(actual),
        "atol": atol,
        "rtol": rtol,
        "reference_dtype": "fp64",
        "input_product_scale_for_tolerance": input_product_scale,
        "tolerance_rule": "abs(error) <= atol + rtol * abs(fp64_reference)",
    }
    if not finite:
        stats.update(
            {
                "failed_samples": len(actual),
                "max_abs_error": None,
                "max_rel_error": None,
                "rms_error": None,
                "normalized_rms_error": None,
                "max_tolerance_ratio": None,
            }
        )
        return stats
    errors = [abs(a - r) for a, r in zip(actual, reference)]
    limits = [atol + rtol * abs(r) for r in reference]
    failed = sum(e > limit for e, limit in zip(errors, limits))
    rms = math.sqrt(sum(e * e for e in errors) / len(errors))
    reference_rms = math.sqrt(sum(r * r for r in reference) / len(reference))
    stats.update(
        {
            "passed": failed == 0,
            "failed_samples": failed,
            "max_abs_error": max(errors),
            "max_rel_error": max(
                e / max(abs(r), atol) for e, r in zip(errors, reference)
            ),
            "rms_error": rms,
            "normalized_rms_error": rms / max(reference_rms, atol),
            "max_tolerance_ratio": max(e / limit for e, limit in zip(errors, limits)),
        }
    )
    return stats


def tensor_descriptor(tensor):
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "contiguous": tensor.is_contiguous(),
        "numel": tensor.numel(),
        "bytes": tensor.numel() * tensor.element_size(),
    }


def sampled_reference(torch, a, weight, bias, coordinates):
    """Copy only the requested rows, then multiply/sum them on CPU in FP64."""
    values = []
    for row, col in coordinates:
        # CPU transfer precedes the conversion: no accidental FP64 NPU operation.
        av = a[row].detach().cpu().to(dtype=torch.float64)
        wv = weight[col].detach().cpu().to(dtype=torch.float64)
        if not bool(torch.isfinite(av).all()) or not bool(torch.isfinite(wv).all()):
            raise ValueError("Nonfinite generated input in sampled reference rows")
        value = torch.dot(av, wv).item()
        if bias is not None:
            value += bias[col].detach().cpu().item()
        values.append(value)
    return values


def validate_output(
    torch, output, *, normalized, coordinates, reference, input_product_scale=1.0
):
    expected_shape = [normalized["m"], normalized["n"]]
    expected_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[normalized["dtype"]]
    if list(output.shape) != expected_shape or output.dtype != expected_dtype:
        raise ValueError(
            "Native kernel output shape/dtype does not match the requested case"
        )
    if not output.is_contiguous():
        raise ValueError("Native kernel returned a noncontiguous output")
    # Check every output element for NaN/Inf with bounded CPU transfer memory.
    flat = output.detach().view(-1)
    all_finite = True
    for offset in range(0, flat.numel(), FINITE_CHUNK_ELEMENTS):
        if not bool(
            torch.isfinite(flat[offset : offset + FINITE_CHUNK_ELEMENTS].cpu()).all()
        ):
            all_finite = False
            break
    actual = [output[row, col].detach().cpu().item() for row, col in coordinates]
    stats = error_statistics(
        actual,
        reference,
        dtype=normalized["tolerance_dtype"],
        k=normalized["k"],
        input_product_scale=input_product_scale,
    )
    stats["tolerance_dtype"] = normalized["tolerance_dtype"]
    stats["all_output_elements_finite"] = all_finite
    stats["passed"] = stats["passed"] and all_finite
    # Nonfinite values cannot appear in JSON; the failed check remains explicit.
    stats["samples"] = [
        {
            "row": row,
            "col": col,
            "actual": a if math.isfinite(a) else None,
            "reference_fp64": r if math.isfinite(r) else None,
        }
        for (row, col), a, r in zip(coordinates, actual, reference)
    ]
    return stats


def _provenance(case):
    semantic = {
        "opType": case.get("opType", case.get("op_type")),
        "args": case.get("args"),
    }
    return {
        "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "case_semantic_sha256": hashlib.sha256(
            json.dumps(
                semantic, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "native_kernel_result_not_rewritten_for_validation": True,
    }


def _base_result(case):
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case.get("case_id", case.get("id")),
        "opType": case.get("opType", case.get("op_type")),
        "args": case.get("args"),
        "sources": case.get("sources", []),
        "testlists": case.get("testlists", []),
        "support": classify_case(case),
        "status": "unvalidated",
        "metrics": None,
        "warnings": [
            "Prepared worker; an unvalidated plan is not a hardware result.",
            "Cache writes do not establish a cold-cache state or public protocol equivalence.",
            "Correctness uses sampled FP64 dot products and a full-output finite check; it is not an exhaustive reference comparison.",
        ],
        "provenance": _provenance(case),
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
        return result
    result["memory_estimate"] = estimate_memory(
        case,
        cache_write_bytes=cache_write_bytes,
        workspace_reserve_bytes=workspace_reserve_bytes,
    )
    return result


class WorkerFailure(Exception):
    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind


def execute_linear(torch, a, weight, bias, implementation):
    """The complete timed operation; there are no hidden casts in validation.

    FP32 mixed paths reproduce _TorchMM.forward in the pinned reference. In
    particular, the BF16+bias final cast is explicitly *not* FP32 accumulation.
    No fallback substitutes a cast if the native out_dtype API is unavailable.
    """
    if implementation == "native_linear":
        return torch.nn.functional.linear(a, weight, bias)
    if implementation == "native_bf16_mm_fp32_output":
        try:
            return torch.mm(a, weight.t(), out_dtype=torch.float32)
        except (TypeError, NotImplementedError) as exc:
            raise WorkerFailure(
                "native_output_dtype_api_unavailable", str(exc)
            ) from exc
    if implementation in {
        "source_defined_bf16_linear_then_fp32_cast",
        "source_defined_timed_fp32_input_promotion",
    }:
        return torch.nn.functional.linear(a.to(weight.dtype), weight, bias).float()
    raise WorkerFailure("invalid_implementation", "Unknown implementation path")


def generate_inputs(torch, spec, *, device):
    """Match pinned _fill/_TorchMM generation arithmetic outside timed regions.

    _prepare_gemm sets the default dtype to BF16 around _fill, so its unqualified
    randn and out-of-place scaling both use BF16. _TorchMM instead creates FP32
    random weights, scales in place and casts to the final dtype. Explicit dtypes
    here avoid depending on the caller's mutable default. Full tensor generation
    retains the source's RNG call shapes; no chunking changes the sample stream.
    Parameter wrapper construction/copy_ is omitted because it does not change
    values. Matching random values across CUDA and NPU generators is not claimed.
    """
    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    plain = spec["implementation"] == "native_linear"
    if plain:
        weight = (
            torch.randn(
                spec["n"], spec["k"], device=device, dtype=dtypes[spec["dtype_b"]]
            )
            * WEIGHT_STD
        )
        bias = (
            torch.randn(spec["n"], device=device, dtype=dtypes[spec["dtype_b"]])
            * WEIGHT_STD
            if spec["bias"]
            else None
        )
    else:
        weight = (
            torch.randn(spec["n"], spec["k"], device=device, dtype=torch.float32)
            .mul_(WEIGHT_STD)
            .to(dtypes[spec["dtype_b"]])
        )
        bias = (
            torch.zeros(spec["n"], device=device, dtype=dtypes[spec["dtype_b"]])
            if spec["bias"]
            else None
        )
    a = torch.randn(spec["m"], spec["k"], device=device, dtype=dtypes[spec["dtype_a"]])
    matched = (
        not plain or spec["dtype_a"] == spec["dtype_b"] == spec["dtype_out"] == "bf16"
    )
    recipe = {
        "source_url": FP32_REFERENCE_SOURCE,
        "source_sha256": REFERENCE_SOURCE_SHA256,
        "source_functions": "_prepare_gemm -> _fill"
        if plain
        else "_prepare_fp32_out -> _TorchMM.__init__",
        "a_recipe": f"torch.randn(M,K,device=device,dtype={spec['dtype_a']}) after weight and bias generation",
        "weight_recipe": (
            f"torch.randn(N,K,device=device,dtype={spec['dtype_b']}) * 0.02"
            if plain
            else f"torch.randn(N,K,device=device,dtype=fp32).mul_(0.02).to({spec['dtype_b']})"
        ),
        "bias_recipe": (
            f"torch.randn(N,device=device,dtype={spec['dtype_b']}) * 0.02"
            if plain
            else f"torch.zeros(N,device=device,dtype={spec['dtype_b']})"
        )
        if spec["bias"]
        else "none",
        "generation_order": ["weight", "bias" if spec["bias"] else "no_bias", "a"],
        "a_dtype": spec["dtype_a"],
        "weight_and_bias_dtype": spec["dtype_b"],
        "generated_device": device,
        "weight_standard_deviation_before_dtype_rounding": WEIGHT_STD,
        "public_input_distribution_reproduced": matched,
        "full_tensor_rng_calls": True,
        "cross_stack_rng_identity_claimed": False,
        "parameter_allocation_copy_omitted": True,
    }
    if not matched:
        recipe["qualification"] = (
            "Requested FP16/FP32 plain-linear dtype extends _fill; this exact dtype combination has no path in the pinned vLLM reference"
        )
    return a, weight, bias, recipe


def _run_native(case, config, result):
    # Deliberately local imports: classification/planning never load torch or NPU.
    import torch
    import torch_npu

    result["provenance"].update(
        {"torch": str(torch.__version__), "torch_npu": str(torch_npu.__version__)}
    )
    npu = torch.npu
    if not npu.is_available():
        raise WorkerFailure("hardware_unavailable", "torch.npu.is_available() is false")
    device = f"npu:{config['device']}"
    npu.set_device(device)
    # Preserve the advertised FP32 inputs. Reduced-precision HF32 is not requested.
    npu.matmul.allow_hf32 = False
    if bool(npu.matmul.allow_hf32):
        raise WorkerFailure("precision_configuration", "Cannot disable NPU matmul HF32")
    result["provenance"]["allow_hf32"] = False
    torch.set_grad_enabled(False)
    torch.manual_seed(config["seed"])
    npu.manual_seed(config["seed"])
    npu.synchronize()
    free, total = npu.mem_get_info(config["device"])
    budget = check_memory_budget(
        result["memory_estimate"],
        available_bytes=int(free),
        memory_budget_bytes=config["memory_budget_bytes"],
    )
    budget["total_device_bytes"] = int(total)
    result["memory_budget_check"] = budget
    if not budget["fits"]:
        raise WorkerFailure(
            "memory_budget_exceeded",
            "Estimated inputs, two outputs, cache writes and workspace reserve exceed the smaller of free device memory and configured budget",
        )
    npu.reset_peak_memory_stats(config["device"])
    baseline = {
        "allocated_bytes": int(npu.memory_allocated(config["device"])),
        "reserved_bytes": int(npu.memory_reserved(config["device"])),
    }
    try:
        spec = result["support"]["normalized"]
        a, weight, bias, generation = generate_inputs(torch, spec, device=device)
        cache = torch.empty(
            config["cache_write_bytes"], device=device, dtype=torch.uint8
        )
        npu.synchronize()
        result["tensors"] = {
            "a": tensor_descriptor(a),
            "weight": tensor_descriptor(weight),
            "bias": tensor_descriptor(bias) if bias is not None else None,
            "cache_write": tensor_descriptor(cache),
        }
        result["provenance"]["implementation"] = spec["implementation"]
        result["provenance"]["fp32_semantics_source"] = spec["fp32_semantics_source"]
        result["input_generation"] = {**generation, "seed": config["seed"]}
        coordinates = sample_coordinates(
            spec["m"], spec["n"], config["reference_samples"], config["seed"]
        )
        reference = sampled_reference(torch, a, weight, bias, coordinates)
        output = execute_linear(torch, a, weight, bias, spec["implementation"])
        npu.synchronize()
        result["tensors"]["output"] = tensor_descriptor(output)
        before = validate_output(
            torch,
            output,
            normalized=spec,
            coordinates=coordinates,
            reference=reference,
            input_product_scale=WEIGHT_STD,
        )
        result["correctness"] = {"before_timing": before, "after_timing": None}
        if not before["passed"]:
            raise WorkerFailure(
                "correctness_failed",
                "Before-timing FP64 reference or finite check failed",
            )
        del output  # Retain at most two live result buffers during queued execution.
        output, timing = queued_timing.time_op(
            npu,
            lambda: execute_linear(torch, a, weight, bias, spec["implementation"]),
            cache,
            config,
        )
        timing["timed_implementation"] = spec["implementation"]
        result["timing"] = timing
        latency_us = upper_median(timing["device_event_samples_us"])
        after = validate_output(
            torch,
            output,
            normalized=spec,
            coordinates=coordinates,
            reference=reference,
            input_product_scale=WEIGHT_STD,
        )
        result["correctness"]["after_timing"] = after
        if not after["passed"]:
            raise WorkerFailure(
                "correctness_failed",
                "After-timing FP64 reference or finite check failed",
            )
        result["metrics"] = {
            "latency_us": latency_us,
            "queued_wall_latency_us_per_call": timing["queued_wall_us_per_call"],
            "tflops": 2 * spec["m"] * spec["n"] * spec["k"] / (latency_us * 1e6),
            "npu_graph": False,
        }
        result["status"] = "ok"
        result["warnings"].pop(0)
    finally:
        result["memory_high_water"] = {
            "baseline": baseline,
            "max_allocated_bytes": int(npu.max_memory_allocated(config["device"])),
            "max_reserved_bytes": int(npu.max_memory_reserved(config["device"])),
            "scope": "PyTorch NPU allocator after initialization; opaque driver memory is not included",
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
        if not _positive_int(memory_budget_bytes):
            raise WorkerFailure(
                "invalid_config", "memory_budget_bytes must be a positive integer"
            )
        if (
            type(device) is not int
            or device < 0
            or type(seed) is not int
            or not 0 <= seed < 2**63
        ):
            raise WorkerFailure(
                "invalid_config",
                "device must be nonnegative and seed must be in [0,2**63)",
            )
        if type(samples) is not int or not 10 <= samples <= 10000:
            raise WorkerFailure(
                "invalid_config", "samples must be an integer in [10,10000]"
            )
        if type(warmups) is not int or not 1 <= warmups <= 10000:
            raise WorkerFailure(
                "invalid_config", "warmups must be an integer in [1,10000]"
            )
        if (
            type(warmup_ms) not in (int, float)
            or not math.isfinite(warmup_ms)
            or not 0 <= warmup_ms <= 10000
        ):
            raise WorkerFailure(
                "invalid_config", "warmup_ms must be finite and in [0,10000]"
            )
        if type(reference_samples) is not int or not 1 <= reference_samples <= 4096:
            raise WorkerFailure(
                "invalid_config", "reference_samples must be an integer in [1,4096]"
            )
        _run_native(case, config, result)
    except Exception as exc:
        result["status"] = (
            "skipped_memory"
            if isinstance(exc, WorkerFailure) and exc.kind == "memory_budget_exceeded"
            else "error"
        )
        result["metrics"] = None
        result["error"] = {
            "kind": exc.kind if isinstance(exc, WorkerFailure) else "execution_failure",
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
        print(
            f"OperatorX worker {result['error']['kind']}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
    result["provenance"]["finished_at"] = datetime.now(timezone.utc).isoformat()
    return result
