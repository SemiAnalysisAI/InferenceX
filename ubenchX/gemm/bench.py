from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from contextlib import suppress
from importlib import import_module
from pathlib import Path

import torch

WARMUP = 5
ITERS = 10
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dense GEMM latency and TFLOP/s on CUDA/ROCm or Ascend."
    )
    parser.add_argument("--device", choices=("cuda", "npu"), required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--shape", nargs=3, type=positive_int, action="append", metavar=("M", "K", "N")
    )
    inputs.add_argument("--shapes", type=Path, help="JSON list of [M, K, N] triples")
    parser.add_argument(
        "--flush-mib",
        type=positive_int,
        help="Scratch size; defaults to reported L2 bytes",
    )
    parser.add_argument(
        "--peak-tflops",
        type=positive_float,
        help="Dense peak for this dtype, used only for MFU",
    )
    args = parser.parse_args(argv)
    if args.device_index < 0:
        parser.error("--device-index must be nonnegative")
    if args.shapes is not None:
        try:
            shapes = json.loads(args.shapes.read_text())
        except (OSError, ValueError) as exc:
            parser.error(f"cannot read shapes: {exc}")
    else:
        shapes = args.shape
    if (
        not isinstance(shapes, list)
        or not shapes
        or any(
            not isinstance(shape, list)
            or len(shape) != 3
            or any(type(dim) is not int or dim <= 0 for dim in shape)
            for shape in shapes
        )
    ):
        parser.error(
            "shapes must be a nonempty JSON list of positive integer [M, K, N] triples"
        )
    args.shapes = [tuple(shape) for shape in shapes]
    return args


def select_device(kind: str, index: int):
    if kind == "npu":
        try:
            import_module("torch_npu")
        except ImportError as exc:
            raise RuntimeError("Ascend requires compatible torch_npu and CANN") from exc
    api = getattr(torch, kind)
    if not api.is_available():
        raise RuntimeError(f"{kind} device is unavailable")
    api.set_device(index)
    return api, f"{kind}:{index}", api.get_device_properties(index)


def measure(shape: tuple[int, int, int], dtype, device: str, api, flush) -> dict:
    m, k, n = shape
    a = torch.randn(m, k, dtype=dtype, device=device)
    b = torch.randn(k, n, dtype=dtype, device=device)
    for _ in range(WARMUP):
        torch.matmul(a, b)
    api.synchronize()
    starts = [api.Event(enable_timing=True) for _ in range(ITERS)]
    ends = [api.Event(enable_timing=True) for _ in range(ITERS)]
    for start, end in zip(starts, ends):
        flush.zero_()
        start.record()
        torch.matmul(a, b)
        end.record()
    api.synchronize()
    times = [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]
    if any(not math.isfinite(time) or time <= 0 for time in times):
        raise RuntimeError("device events returned invalid elapsed times")
    latency = sorted(times)[ITERS // 2]
    flops = 2 * m * k * n
    byte_count = (m * k + k * n + m * n) * a.element_size()
    return {
        "m": m,
        "k": k,
        "n": n,
        "latency_us": latency,
        "tflops": flops / (latency * 1e6),
        "ai": flops / byte_count,
        "timings_us": times,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        api, device, props = select_device(args.device, args.device_index)
        size = (
            args.flush_mib * 1024**2
            if args.flush_mib
            else getattr(props, "L2_cache_size", 0)
        )
        if not isinstance(size, int) or size <= 0:
            raise RuntimeError(
                "device did not report a positive L2 size; supply --flush-mib"
            )
        flush = torch.empty(size, dtype=torch.int8, device=device)
    except (ImportError, RuntimeError, ValueError, AttributeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    metadata = {
        "device": device,
        "device_name": props.name,
        "dtype": args.dtype,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "rocm": torch.version.hip,
        "flush_bytes": size,
        "warmup": WARMUP,
        "iterations": ITERS,
        "peak_tflops": args.peak_tflops,
    }
    if args.device == "npu":
        metadata["torch_npu"] = getattr(sys.modules["torch_npu"], "__version__", None)
        with suppress(ImportError, AttributeError, RuntimeError):
            version = import_module("torch_npu.utils").get_cann_version(module="CANN")
            if version:
                metadata["cann"] = version
        for name in ("allow_hf32", "cube_math_type"):
            with suppress(AttributeError, RuntimeError):
                value = getattr(api.matmul, name)
                metadata[name] = getattr(value, "name", value)
    else:
        with suppress(AttributeError, RuntimeError):
            metadata["float32_matmul_precision"] = torch.get_float32_matmul_precision()
        with suppress(AttributeError, RuntimeError):
            metadata["allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
    print(json.dumps(metadata), file=sys.stderr)
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=[
            "m",
            "k",
            "n",
            "dtype",
            "status",
            "latency_us",
            "tflops",
            "ai",
            "mfu_percent",
            "timings_us",
            "error",
        ],
    )
    writer.writeheader()
    failed = False
    for shape in args.shapes:
        row = dict(zip(("m", "k", "n"), shape), dtype=args.dtype)
        try:
            result = measure(shape, DTYPES[args.dtype], device, api, flush)
            row.update(result, status="ok", timings_us=json.dumps(result["timings_us"]))
            if args.peak_tflops is not None:
                row["mfu_percent"] = 100 * result["tflops"] / args.peak_tflops
        except (RuntimeError, ValueError) as exc:
            row.update(status="error", error=f"{type(exc).__name__}: {exc}")
            failed = True
        writer.writerow(row)
        sys.stdout.flush()
        # Exception tracebacks must release failed allocations before cache cleanup.
        api.empty_cache()
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
