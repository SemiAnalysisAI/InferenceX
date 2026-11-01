from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys
import operatorx_queued_timing as queued_timing
from datetime import datetime, timezone


HELPER_PATH = Path(__file__).with_name("operatorx_ascend_worker.py")
HELPER_SHA256 = hashlib.sha256(HELPER_PATH.read_bytes()).hexdigest()
_SPEC = importlib.util.spec_from_file_location(
    "operatorx_mxfp8_128_dense_helper", HELPER_PATH
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
IMPLEMENTATION = (
    "triton_flat_exact_ue8m0_128_then_scale_expansion_32_then_cann_quant_matmul"
)
WEIGHT_SCALE_BYTE = 120
WEIGHT_SCALE_VALUE = 2**-7
WEIGHT_PRODUCT_STD = 0.5 * WEIGHT_SCALE_VALUE


def classify_case(case):
    if not isinstance(case, dict):
        return dense._unsupported("invalid_case", "Case must be a JSON object")
    op_type = case.get("opType", case.get("op_type"))
    if "opType" in case and "op_type" in case and case["opType"] != case["op_type"]:
        return dense._unsupported("invalid_case", "Conflicting operator types")
    if op_type != "gemm":
        return dense._unsupported(
            "operator_not_implemented",
            "This worker implements UE8M0 group-128 E4M3 GEMM only",
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
                "Only exact E4M3 descriptors with one UE8M0 block scale are implemented",
            )
        scale = operand["scale"]
        if (
            not isinstance(scale, dict)
            or set(scale) != {"dtype", "group", "static"}
            or scale["dtype"] != "ue8m0"
            or type(scale["static"]) is not bool
            or scale["static"] is not static
            or not isinstance(scale["group"], list)
            or len(scale["group"]) != 2
            or any(type(value) is not int for value in scale["group"])
            or scale["group"] != group
        ):
            return dense._unsupported(
                "scale_semantics_not_implemented",
                "Require dynamic A UE8M0 [1,128] and static W UE8M0 [128,128] scales",
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
            "k_alignment_not_implemented",
            "This flat quantizer requires K divisible by 128; partial groups are not supported",
        )
    return {
        "supported": True,
        "status": "supported_unvalidated",
        "reason_code": "native_mxfp8_128_candidate",
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
            "scale_dtype": "ue8m0",
            "activation_group_size": 128,
            "weight_group_size": [128, 128],
            "native_matmul_group_sizes": [1, 1, 32],
            "layout": "A quantized at128; W[N,K] with coarse128x128 scales; exact exponent repetition to native paired32 groups; payloads unchanged",
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
    groups = (k + 127) // 128
    tail = k % 128 != 0
    memory = {
        "a_bytes": 2 * m * k,
        "weight_bytes": n * k,
        "coarse_weight_scale_bytes": ((n + 127) // 128) * groups,
        "weight_scale_bytes": n * (k // 32),
        "static_scale_expansion_temporary_bytes": ((n + 127) // 128) * 128 * groups
        + (n * groups * 4 if tail else 0),
        "quantized_a_bytes": m * k,
        "raw_activation_scale_bytes": m * ((groups + 1) // 2) * 2,
        "activation_scale_bytes": m * (k // 32),
        "activation_scale_expansion_temporary_bytes": m * groups * 4 if tail else 0,
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
        + memory["coarse_weight_scale_bytes"]
        + memory["weight_scale_bytes"]
        + cache_write_bytes
        + 2
        * (
            memory["quantized_a_bytes"]
            + memory["raw_activation_scale_bytes"]
            + memory["activation_scale_bytes"]
            + memory["output_bytes"]
        )
        + memory["activation_scale_expansion_temporary_bytes"]
    )
    memory["input_generation_peak_bytes"] = max(
        4 * n * k,
        memory["weight_bytes"]
        + memory["coarse_weight_scale_bytes"]
        + memory["weight_scale_bytes"]
        + memory["static_scale_expansion_temporary_bytes"],
        memory["a_bytes"]
        + memory["weight_bytes"]
        + memory["coarse_weight_scale_bytes"]
        + memory["weight_scale_bytes"],
    )
    memory["estimated_required_bytes"] = (
        max(memory["tensor_and_cache_bytes"], memory["input_generation_peak_bytes"])
        + workspace_reserve_bytes
    )
    memory["required_bytes"] = memory["estimated_required_bytes"]
    cpu_chunk_elements = max(1, QUANT_CHECK_CHUNK_ELEMENTS // k) * groups * 128
    memory["reference_cpu_working_bytes_estimate"] = (
        80 * cpu_chunk_elements + 32 * k + memory["coarse_weight_scale_bytes"]
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
            "Triton quantizer follows pinned vLLM epsilon/scale recipe; CPU reference and device validation required; CUDA instruction parity is not claimed.",
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
            "activation_quantizer_group_size": 128,
            "scale_alg": 1,
            "static_weight_scale_byte": WEIGHT_SCALE_BYTE,
            "static_weight_scale_value": WEIGHT_SCALE_VALUE,
            "payload_requantization_during_scale_expansion": False,
            "public_gpu_tail_execution_parity_claimed": False,
            "gpu_input_padding_performed": False,
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


def native_parameters(torch, torch_npu):
    wanted = {
        "npu::npu_dynamic_mx_quant": {
            "input",
            "axis",
            "round_mode",
            "dst_type",
            "block_size",
            "scale_alg",
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
            "x1_dtype",
            "x2_dtype",
            "pertoken_scale_dtype",
            "scale_dtype",
        },
    }
    extensions = {
        "npu::npu_dynamic_mx_quant": {"dst_type_max"},
        "npu::npu_quant_matmul": {"y_scale"},
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
        leading = (
            ["input"] if op.endswith("dynamic_mx_quant") else ["x1", "x2", "scale"]
        )
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
    fp8_enum, scale_enum = torch_npu.float8_e4m3fn, torch_npu.float8_e8m0fnu
    if type(fp8_enum) is not int or type(scale_enum) is not int:
        raise ValueError("Explicit torch_npu dtype enums are required")
    quant = {
        "axis": -1,
        "round_mode": "rint",
        "dst_type": fp8_enum,
        "block_size": 128,
        "scale_alg": 1,
    }
    if "dst_type_max" in names["npu::npu_dynamic_mx_quant"]:
        quant["dst_type_max"] = 0.0
    mm = {
        "offset": None,
        "bias": None,
        "output_dtype": torch.bfloat16,
        "group_sizes": [1, 1, 32],
        "x1_dtype": None,
        "x2_dtype": None,
        "pertoken_scale_dtype": scale_enum,
        "scale_dtype": scale_enum,
    }
    for name in names["npu::npu_quant_matmul"] & extensions["npu::npu_quant_matmul"]:
        mm[name] = None
    return quant, mm, metadata


def generate_inputs(torch, spec, *, device):
    weight = (
        torch.randn(spec["n"], spec["k"], device=device, dtype=torch.bfloat16) * 0.5
    ).to(torch.float8_e4m3fn)
    coarse_weight_scale = torch.full(
        ((spec["n"] + 127) // 128, (spec["k"] + 127) // 128),
        WEIGHT_SCALE_BYTE,
        device=device,
        dtype=torch.uint8,
    )
    rows = coarse_weight_scale.repeat_interleave(128, dim=0)[: spec["n"]]
    full = rows.repeat_interleave(4, dim=1)
    compact = full[:, : spec["k"] // 32].clone() if spec["k"] % 128 else full
    weight_scale = compact.reshape(spec["n"], spec["k"] // 64, 2)
    del rows, full, compact
    a = torch.randn(spec["m"], spec["k"], device=device, dtype=torch.bfloat16)
    recipe = {
        "source_url": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "source_functions": "_prepare_gemm -> _fill",
        "weight_recipe": "BF16 randn(N,K) * 0.5 in BF16, then E4M3 cast",
        "weight_scale_recipe": "E8M0 fill_(0.01) has byte120/value2^-7; allocate uint8 coarse[ceil(N/128),ceil(K/128)] byte120 then repeat and crop exact bytes to native32 groups",
        "weight_scale_parameter_source": "https://github.com/vllm-project/vllm/blob/cd10ed6f9f6b37a8ace9cf380007e66fe12ec0c3/vllm/model_executor/layers/quantization/fp8.py#L347",
        "scale_cast_source": "https://github.com/pytorch/pytorch/blob/v2.13.0/torch/headeronly/util/Float8_e8m0fnu.h",
        "a_recipe": "BF16 randn(M,K) after weight/scales",
        "generation_order": ["weight", "weight_scale", "a"],
        "public_input_distribution_reproduced": True,
        "cross_stack_rng_identity_claimed": False,
        "parameter_allocation_copy_omitted": True,
        "weights_static_outside_timer": True,
        "activation_quantization_inside_timer": True,
        "activation_scale_expansion_inside_timer": True,
        "gpu_input_padding_performed": False,
        "original_k_preserved": True,
    }
    return a, weight, coarse_weight_scale, weight_scale, recipe


def expand_activation_scales(raw_scales, m, k):
    if type(k) is not int or k <= 0 or k % 64:
        raise ValueError("Scale expansion requires positive K divisible by64")
    groups = (k + 127) // 128
    valid = raw_scales.reshape(m, -1)[:, :groups]
    full = valid.repeat_interleave(4, dim=1)
    compact = full[:, : k // 32].clone() if k % 128 else full
    return compact.reshape(m, k // 64, 2)


def execute_native(torch_npu, a, weight, weight_scale, quant_kwargs, mm_kwargs):
    from operatorx_ue8m0_quant_triton import quantize

    quantized, raw_scales = quantize(a, block_rows=64)
    scales = expand_activation_scales(raw_scales, a.shape[0], a.shape[1])
    output = torch_npu.npu_quant_matmul(
        quantized,
        weight.t(),
        weight_scale.transpose(0, 1),
        pertoken_scale=scales,
        **mm_kwargs,
    )
    return output, quantized, raw_scales, scales


def decode_scale_bytes(torch, data):
    if data.device.type != "cpu" or data.dtype != torch.uint8:
        raise ValueError("Scale decoding requires CPU uint8 storage")
    values = torch.pow(
        torch.tensor(2.0, dtype=torch.float64, device="cpu"),
        data.to(torch.float64) - 127,
    )
    return values.masked_fill(data == 255, float("nan"))


def validate_static_scales(torch, coarse, expanded, n, k):
    if coarse.dtype != torch.uint8 or expanded.dtype != torch.uint8:
        raise ValueError("Static scales must retain UE8M0 uint8 storage")
    if list(coarse.shape) != [(n + 127) // 128, (k + 127) // 128] or list(
        expanded.shape
    ) != [n, k // 64, 2]:
        raise ValueError("Unexpected static scale shape")
    original = coarse.detach().cpu()
    if not bool((original == WEIGHT_SCALE_BYTE).all()):
        raise ValueError(
            "Static scales no longer match the source E8M0 fill_(0.01) recipe"
        )
    bad = 0
    rows_per_chunk = max(1, QUANT_CHECK_CHUNK_ELEMENTS // (k // 32))
    for start in range(0, n, rows_per_chunk):
        end = min(n, start + rows_per_chunk)
        observed = expanded[start:end].detach().cpu().reshape(end - start, k // 32)
        expected = original[torch.arange(start, end, device="cpu") // 128][
            :, torch.arange(k // 32, device="cpu") // 4
        ]
        bad += int((observed != expected).sum().item())
    return {
        "passed": bad == 0,
        "mismatched_expanded_bytes": bad,
        "checked_bytes": n * (k // 32),
        "source_coarse_group": [128, 128],
        "source_byte": WEIGHT_SCALE_BYTE,
        "payload_modified": False,
    }


def validate_quantization(
    torch,
    a,
    quantized,
    raw_scales,
    scales,
    *,
    chunk_elements=QUANT_CHECK_CHUNK_ELEMENTS,
):
    if (
        a.dtype != torch.bfloat16
        or quantized.dtype != torch.float8_e4m3fn
        or raw_scales.dtype != torch.uint8
        or scales.dtype != torch.uint8
    ):
        raise ValueError("Unexpected activation/quantizer dtype")
    m, k = a.shape
    groups = (k + 127) // 128
    if (
        k % 64
        or list(quantized.shape) != [m, k]
        or list(raw_scales.shape) != [m, (groups + 1) // 2, 2]
        or list(scales.shape) != [m, k // 64, 2]
    ):
        raise ValueError("Unexpected activation/quantizer shape")
    if (
        not quantized.is_contiguous()
        or not raw_scales.is_contiguous()
        or not scales.is_contiguous()
    ):
        raise ValueError("Unexpected native quantizer output strides")
    rows_per_chunk = max(1, chunk_elements // k)
    bad_elements = bad_groups = expansion_errors = 0
    max_error = max_ratio = 0.0
    finite = True
    padded_k = groups * 128
    valid_elements = (torch.arange(padded_k, device="cpu") < k).reshape(1, groups, 128)
    for start in range(0, m, rows_per_chunk):
        end = min(m, start + rows_per_chunk)
        original = a[start:end].detach().cpu().double()
        codes = quantized[start:end].detach().cpu().double()
        if padded_k != k:
            original = torch.nn.functional.pad(original, (0, padded_k - k))
            codes = torch.nn.functional.pad(codes, (0, padded_k - k))
        original = original.reshape(end - start, groups, 128)
        codes = codes.reshape(end - start, groups, 128)
        source = (
            raw_scales[start:end].detach().cpu().reshape(end - start, -1)[:, :groups]
        )
        expanded = scales[start:end].detach().cpu().reshape(end - start, k // 32)
        expected_expanded = source[:, torch.arange(k // 32, device="cpu") // 4]
        expansion_errors += int((expanded != expected_expanded).sum().item())
        ss = decode_scale_bytes(torch, source)
        chunk_finite = (
            bool(torch.isfinite(original).all())
            and bool(torch.isfinite(codes).all())
            and bool(torch.isfinite(ss).all())
        )
        finite &= chunk_finite
        if not chunk_finite:
            bad_elements += (end - start) * k
            continue
        valid = ss > 0
        bad_groups += int((~valid).sum().item())
        error = (codes * ss.unsqueeze(-1) - original).abs()
        bound = original.abs() / 16.0 + ss.unsqueeze(-1) / 1024.0
        bad_elements += int(((error > bound) & valid_elements).sum().item())
        max_error = max(
            max_error, float(torch.where(valid_elements, error, 0.0).max().item())
        )
        ratio = torch.where(
            bound > 0,
            error / bound,
            torch.where(
                error == 0,
                torch.zeros_like(error),
                torch.full_like(error, float("inf")),
            ),
        )
        max_ratio = max(
            max_ratio, float(torch.where(valid_elements, ratio, 0.0).max().item())
        )
    return {
        "passed": finite
        and bad_elements == 0
        and bad_groups == 0
        and expansion_errors == 0,
        "all_quantized_inputs_finite": finite,
        "checked_elements": m * k,
        "checked_groups": m * groups,
        "invalid_scale_groups": bad_groups,
        "masked_cpu_padding_elements": m * (padded_k - k),
        "last_group_valid_elements": k % 128 or 128,
        "gpu_input_padding_performed": False,
        "scale_expansion_mismatched_bytes": expansion_errors,
        "expanded_scale_bytes_checked": m * (k // 32),
        "raw_padding_byte_used_as_valid_group": False,
        "quantized_group_size": 128,
        "failed_elements": bad_elements,
        "max_abs_reconstruction_error": max_error if finite else None,
        "max_error_bound_ratio": max_ratio if math.isfinite(max_ratio) else None,
        "bound": "abs(decoded-original) <= abs(original)/16 + scale/1024; UE8M0 byte0 means2^-127 and byte255 is invalid",
        "scope": "Every activation element checked on CPU in bounded chunks; CUDA payload/scale equality is not required",
    }


def sampled_decoded_reference(
    torch, quantized, scales, weight, weight_scale, coordinates
):
    values = []
    for row, col in coordinates:
        q = quantized[row].detach().cpu().to(torch.float64)
        s = decode_scale_bytes(
            torch, scales[row].detach().cpu().reshape(-1)
        ).repeat_interleave(32)
        w = weight[col].detach().cpu().to(torch.float64)
        ws = decode_scale_bytes(
            torch, weight_scale[col].detach().cpu().reshape(-1)
        ).repeat_interleave(32)
        if any(not bool(torch.isfinite(x).all()) for x in (q, s, w, ws)):
            raise ValueError("Nonfinite sampled decoded operand")
        values.append(torch.dot(q * s, w * ws).item())
    return values


def validate_state(
    torch, a, weight, coarse_weight_scale, weight_scale, state, spec, coordinates
):
    output, quantized, raw_scales, scales = state
    quant = validate_quantization(torch, a, quantized, raw_scales, scales)
    static = validate_static_scales(
        torch, coarse_weight_scale, weight_scale, spec["n"], spec["k"]
    )
    reference = sampled_decoded_reference(
        torch, quantized, scales, weight, weight_scale, coordinates
    )
    check = dense.validate_output(
        torch,
        output,
        normalized=spec,
        coordinates=coordinates,
        reference=reference,
        input_product_scale=WEIGHT_PRODUCT_STD,
    )
    check["quantization"] = quant
    check["static_scale_expansion"] = static
    check["reference_scope"] = (
        "CPU FP64 full-K dots of this kernel call's actual E4M3 and expanded UE8M0 matmul inputs"
    )
    check["passed"] = check["passed"] and quant["passed"] and static["passed"]
    return check


def _run_native(case, config, result):
    import torch
    import torch_npu

    npu = torch.npu
    if not npu.is_available():
        raise dense.WorkerFailure("hardware_unavailable", "No available Ascend NPU")
    quant_kwargs, mm_kwargs, schemas = native_parameters(torch, torch_npu)
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
    import triton
    import operatorx_ue8m0_quant_triton as quantizer

    result["provenance"].update(
        quantizer="Triton",
        triton_version=triton.__version__,
        quantizer_parameters={
            "block_size": 128,
            "epsilon": quantizer.EPSILON,
            "block_rows": 64,
            "addressing": "flat groups, int64 offsets, group-uniform mask",
            "scale": "IEEE exponent bits implement ceil power-of-two of max(absmax,eps)*(1/448)",
            "rounding": "explicit E4M3 rtne; exact exponent-bit normalizer",
        },
        quantizer_source_sha256=hashlib.sha256(
            Path(quantizer.__file__).read_bytes()
        ).hexdigest(),
        quantizer_reference="https://github.com/vllm-project/vllm/blob/cd10ed6f9f6b37a8ace9cf380007e66fe12ec0c3/vllm/model_executor/layers/quantization/utils/fp8_utils.py",
    )
    free, total = npu.mem_get_info(config["device"])
    budget = dense.check_memory_budget(
        result["memory_estimate"],
        available_bytes=max(0, int(free) - 8 * 1024**3),
        memory_budget_bytes=config["memory_budget_bytes"],
    )
    result["memory_budget_check"] = {**budget, "total_device_bytes": int(total)}
    if not budget["fits"]:
        raise dense.WorkerFailure(
            "memory_budget_exceeded",
            "MXFP8 input/transient/scale-expansion/output/cache/workspace estimate exceeds the available budget",
        )
    npu.reset_peak_memory_stats(config["device"])
    baseline = {
        "allocated_bytes": int(npu.memory_allocated(config["device"])),
        "reserved_bytes": int(npu.memory_reserved(config["device"])),
    }
    try:
        spec = result["support"]["normalized"]
        a, weight, coarse_weight_scale, weight_scale, recipe = generate_inputs(
            torch, spec, device=device
        )
        cache = torch.empty(
            config["cache_write_bytes"], device=device, dtype=torch.uint8
        )
        result["input_generation"] = {**recipe, "seed": config["seed"]}
        result["tensors"] = {
            "a": dense.tensor_descriptor(a),
            "weight": dense.tensor_descriptor(weight),
            "coarse_weight_scale": dense.tensor_descriptor(coarse_weight_scale),
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
            torch,
            a,
            weight,
            coarse_weight_scale,
            weight_scale,
            state,
            spec,
            coordinates,
        )
        result["correctness"] = {"before_timing": before, "after_timing": None}
        if not before["passed"]:
            raise dense.WorkerFailure(
                "correctness_failed", "Pre-timing quantization/GEMM reference failed"
            )

        def fn():
            return execute_native(
                torch_npu, a, weight, weight_scale, quant_kwargs, mm_kwargs
            )

        state, timing = queued_timing.time_op(npu, fn, cache, config)
        result["timing"] = {
            **timing,
            "timed_implementation": IMPLEMENTATION,
            "activation_quantization_inside_timer": True,
            "activation_scale_expansion_inside_timer": True,
            "static_weight_preparation_inside_timer": False,
        }
        latency = dense.upper_median(timing["device_event_samples_us"])
        after = validate_state(
            torch,
            a,
            weight,
            coarse_weight_scale,
            weight_scale,
            state,
            spec,
            coordinates,
        )
        result["correctness"]["after_timing"] = after
        result["tensors"].update(
            output=dense.tensor_descriptor(state[0]),
            quantized_a=dense.tensor_descriptor(state[1]),
            raw_activation_scale_128=dense.tensor_descriptor(state[2]),
            activation_scale=dense.tensor_descriptor(state[3]),
        )
        if not after["passed"]:
            raise dense.WorkerFailure(
                "correctness_failed", "Post-timing quantization/GEMM reference failed"
            )
        result["metrics"] = {
            "latency_us": latency,
            "queued_wall_us_per_call": timing["queued_wall_us_per_call"],
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
        print(
            f"OperatorX MXFP8-128 worker {result['error']['kind']}: {exc}",
            file=sys.stderr,
        )
    result["provenance"]["finished_at"] = datetime.now(timezone.utc).isoformat()
    return result
