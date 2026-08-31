#!/usr/bin/env python3
"""Validate the DSV4 gfx950 FlyDSL sparse-MLA MVP on one ROCm GPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch


HEAD_DIM = 512
NOPE_HEAD_DIM = 448
ROPE_HEAD_DIM = 64
DEFAULT_SCALE = HEAD_DIM**-0.5
EXPECTED_AITER_COMMIT = "59799f3a16d6dae44617346630aab7be27226789"
EXPECTED_VLLM_COMMIT = "91b654ac84638b582b62bb51921248a0d8023902"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _device_arch() -> str:
    props = torch.cuda.get_device_properties(0)
    return str(getattr(props, "gcnArchName", "")).split(":", 1)[0]


def _environment() -> dict[str, Any]:
    flydsl = importlib.import_module("flydsl")
    aiter = importlib.import_module("aiter")
    try:
        vllm = importlib.import_module("vllm")
        vllm_info = {
            "version": getattr(vllm, "__version__", None),
            "path": str(Path(vllm.__file__).resolve()),
        }
    except Exception as error:  # noqa: BLE001 - report optional baseline state
        vllm_info = {"import_error": repr(error)}

    props = torch.cuda.get_device_properties(0)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "device_name": props.name,
        "device_arch": _device_arch(),
        "device_memory_bytes": props.total_memory,
        "flydsl_version": getattr(flydsl, "__version__", None),
        "flydsl_path": str(Path(flydsl.__file__).resolve()),
        "aiter_path": str(Path(aiter.__file__).resolve()),
        "vllm": vllm_info,
        "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
    }


def _git_revision(tree: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(tree), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_flydsl_op(kernel_file: Path):
    module_name = "aiter.ops.flydsl.kernels.dsv4_sparse_mla_validation"
    module = _load_module(module_name, kernel_file)
    return module.flydsl_sparse_mla_prefill, module


def _load_triton_baseline(vllm_tree: Path):
    source = vllm_tree / "vllm/v1/attention/ops/rocm_aiter_mla_sparse.py"
    module = _load_module("dsv4_target_rocm_aiter_mla_sparse", source)
    return module._rocm_sparse_attn_prefill_ragged_triton, module


def _reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    indptr: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
) -> torch.Tensor:
    q_f32 = q.float()
    kv_f32 = kv.float()
    result = torch.zeros_like(q_f32)
    indices_cpu = indices.cpu().tolist()
    indptr_cpu = indptr.cpu().tolist()
    for query in range(q.shape[0]):
        row = [
            slot
            for slot in indices_cpu[indptr_cpu[query] : indptr_cpu[query + 1]]
            if 0 <= slot < kv.shape[0]
        ]
        if not row:
            continue
        selected = kv_f32[row]
        logits = torch.matmul(q_f32[query], selected.T) * scale
        if attn_sink is not None:
            logits = torch.cat([logits, attn_sink[: q.shape[1], None]], dim=1)
            probs = torch.softmax(logits, dim=1)[:, :-1]
        else:
            probs = torch.softmax(logits, dim=1)
        result[query] = torch.matmul(probs, selected)
    return result.to(torch.bfloat16)


def _error_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    absolute = (actual_f32 - expected_f32).abs()
    denominator = expected_f32.abs().clamp_min(1.0e-6)
    relative = absolute / denominator
    return {
        "max_abs": absolute.max().item() if absolute.numel() else 0.0,
        "mean_abs": absolute.mean().item() if absolute.numel() else 0.0,
        "max_rel": relative.max().item() if relative.numel() else 0.0,
    }


def _make_correctness_case(
    num_heads: int, with_sink: bool
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(1700 + num_heads + int(with_sink))
    q = (
        torch.randn(4, num_heads, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        * 0.125
    )
    if num_heads == 16:
        kv = torch.randn(97, HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.125
        q[:, :, 0] = 1.0
        kv[69, 0] = 128.0
        first_row = torch.arange(70, dtype=torch.int32, device="cuda")
        third_row = torch.arange(65, dtype=torch.int32, device="cuda") * 3 % 97
        third_row[7] = -1
        third_row[61] = 98
        indices = torch.cat(
            (
                first_row,
                torch.tensor([-1, 98], dtype=torch.int32, device="cuda"),
                third_row,
            )
        )
        indptr = torch.tensor(
            [0, 70, 72, 137, 137], dtype=torch.int32, device="cuda"
        )
    else:
        kv = torch.randn(7, HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.125
        indices = torch.tensor(
            [0, 2, -1, 8, 1, 3, 6], dtype=torch.int32, device="cuda"
        )
        indptr = torch.tensor([0, 2, 4, 7, 7], dtype=torch.int32, device="cuda")
    sink = (
        torch.linspace(-0.5, 0.5, num_heads, dtype=torch.float32, device="cuda")
        if with_sink
        else None
    )
    return q, kv, indices, indptr, sink


def _make_h16_split_boundary_case(
    with_sink: bool,
) -> tuple[torch.Tensor, ...]:
    lengths = [
        0,
        1,
        63,
        64,
        65,
        127,
        128,
        129,
        191,
        192,
        193,
        255,
        256,
        128,
        128,
        128,
    ]
    num_kv = 521
    torch.manual_seed(1943090 + int(with_sink))
    q = (
        torch.randn(
            len(lengths), 16, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        * 0.125
    )
    kv = (
        torch.randn(num_kv, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        * 0.125
    )
    rows: list[torch.Tensor] = []
    for query, length in enumerate(lengths):
        row = (
            torch.arange(length, dtype=torch.int32, device="cuda") * 13
            + query * 17
        ) % num_kv
        if length > 3:
            row[3] = -1
        if length > 67:
            row[67] = num_kv
        rows.append(row)

    # Exercise each split half independently, then make both halves invalid.
    rows[-3][:64] = -1
    rows[-2][64:] = num_kv
    rows[-1][:] = -1

    # Put the dominant score in group 1 so the final max-rescaled merge is
    # required for numerical correctness.
    late_query = 7
    late_slot = num_kv - 1
    rows[late_query][100] = late_slot
    q[late_query, :, 0] = 1.0
    kv[late_slot, 0] = 128.0

    indices = torch.cat(rows)
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    indptr = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    sink = (
        torch.linspace(-0.5, 0.5, 16, dtype=torch.float32, device="cuda")
        if with_sink
        else None
    )
    return q, kv, indices, indptr, sink


def _run_correctness(
    flydsl_op: Callable[..., torch.Tensor],
    triton_op: Callable[..., torch.Tensor],
    large_address_test: bool,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for num_heads, with_sink in (
        (1, False),
        (4, True),
        (7, True),
        (16, False),
        (16, True),
    ):
        q, kv, indices, indptr, sink = _make_correctness_case(
            num_heads, with_sink
        )
        provided_out = torch.empty_like(q)
        first_call_start = time.perf_counter()
        actual = flydsl_op(
            q=q,
            kv=kv,
            indices=indices,
            indptr=indptr,
            scale=DEFAULT_SCALE,
            attn_sink=sink,
            out=provided_out,
        )
        torch.cuda.synchronize()
        flydsl_first_call_ms = (time.perf_counter() - first_call_start) * 1000.0
        expected = _reference(q, kv, indices, indptr, DEFAULT_SCALE, sink)
        first_call_start = time.perf_counter()
        triton = triton_op(
            q=q,
            kv=kv,
            indices=indices,
            indptr=indptr,
            scale=DEFAULT_SCALE,
            attn_sink=sink,
            nope_head_dim=NOPE_HEAD_DIM,
            rope_head_dim=ROPE_HEAD_DIM,
        )
        torch.cuda.synchronize()
        triton_first_call_ms = (time.perf_counter() - first_call_start) * 1000.0

        torch.testing.assert_close(actual, expected, atol=3.0e-2, rtol=3.0e-2)
        torch.testing.assert_close(actual, triton, atol=3.0e-2, rtol=3.0e-2)
        assert actual.data_ptr() == provided_out.data_ptr()
        assert actual.dtype == torch.bfloat16
        assert torch.count_nonzero(actual[1]).item() == 0
        assert torch.count_nonzero(actual[3]).item() == 0
        cases.append(
            {
                "heads": num_heads,
                "sink": with_sink,
                "flydsl_first_call_ms": flydsl_first_call_ms,
                "triton_first_call_ms": triton_first_call_ms,
                "flydsl_vs_reference": _error_stats(actual, expected),
                "flydsl_vs_triton": _error_stats(actual, triton),
                "status": "pass",
            }
        )

    # vLLM's dense-to-ragged packer keeps capacity-sized storage while
    # indptr[-1] records only the logical nnz. Exercise that contract with
    # short rows, empty rows, every non-zero row-start residue modulo four,
    # and deterministic invalid values in the unused capacity.
    for with_sink in (False, True):
        torch.manual_seed(194309 + int(with_sink))
        q = (
            torch.randn(5, 16, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
            * 0.125
        )
        kv = torch.randn(9, HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.125
        logical_indices = torch.tensor(
            [0, 1, 2, 3, 4, 5, 6, -1, 9, 7],
            dtype=torch.int32,
            device="cuda",
        )
        indices = torch.full(
            (32,), 17, dtype=torch.int32, device="cuda"
        )
        indices[: logical_indices.numel()].copy_(logical_indices)
        indptr = torch.tensor(
            [0, 1, 3, 3, 6, 10], dtype=torch.int32, device="cuda"
        )
        sink = (
            torch.linspace(-0.5, 0.5, 16, dtype=torch.float32, device="cuda")
            if with_sink
            else None
        )
        provided_out = torch.empty_like(q)
        actual = flydsl_op(
            q=q,
            kv=kv,
            indices=indices,
            indptr=indptr,
            scale=DEFAULT_SCALE,
            attn_sink=sink,
            out=provided_out,
        )
        expected = _reference(q, kv, indices, indptr, DEFAULT_SCALE, sink)
        triton = triton_op(
            q=q,
            kv=kv,
            indices=indices,
            indptr=indptr,
            scale=DEFAULT_SCALE,
            attn_sink=sink,
            nope_head_dim=NOPE_HEAD_DIM,
            rope_head_dim=ROPE_HEAD_DIM,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, atol=3.0e-2, rtol=3.0e-2)
        torch.testing.assert_close(actual, triton, atol=3.0e-2, rtol=3.0e-2)
        assert actual.data_ptr() == provided_out.data_ptr()
        assert torch.count_nonzero(actual[2]).item() == 0
        cases.append(
            {
                "case": "overallocated_ragged_capacity",
                "heads": 16,
                "sink": with_sink,
                "logical_nnz": int(indptr[-1].item()),
                "physical_capacity": indices.numel(),
                "flydsl_vs_reference": _error_stats(actual, expected),
                "flydsl_vs_triton": _error_stats(actual, triton),
                "status": "pass",
            }
        )

    for with_sink in (False, True):
        q, kv, indices, indptr, sink = _make_h16_split_boundary_case(
            with_sink
        )
        actual = flydsl_op(
            q=q,
            kv=kv,
            indices=indices,
            indptr=indptr,
            scale=DEFAULT_SCALE,
            attn_sink=sink,
        )
        expected = _reference(q, kv, indices, indptr, DEFAULT_SCALE, sink)
        triton = triton_op(
            q=q,
            kv=kv,
            indices=indices,
            indptr=indptr,
            scale=DEFAULT_SCALE,
            attn_sink=sink,
            nope_head_dim=NOPE_HEAD_DIM,
            rope_head_dim=ROPE_HEAD_DIM,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, atol=3.0e-2, rtol=3.0e-2)
        torch.testing.assert_close(actual, triton, atol=3.0e-2, rtol=3.0e-2)
        assert torch.count_nonzero(actual[0]).item() == 0
        assert torch.count_nonzero(actual[-1]).item() == 0
        cases.append(
            {
                "case": "h16_split_boundaries",
                "heads": 16,
                "sink": with_sink,
                "row_lengths": [
                    int(value)
                    for value in (indptr[1:] - indptr[:-1]).cpu().tolist()
                ],
                "flydsl_vs_reference": _error_stats(actual, expected),
                "flydsl_vs_triton": _error_stats(actual, triton),
                "status": "pass",
            }
        )

    large_result: dict[str, Any] = {"requested": large_address_test}
    if large_address_test:
        row_bytes = HEAD_DIM * torch.tensor([], dtype=torch.bfloat16).element_size()
        high_slot = math.ceil((2**32) / row_bytes) + 1
        num_rows = high_slot + 1
        try:
            kv = torch.empty(
                (num_rows, HEAD_DIM), dtype=torch.bfloat16, device="cuda"
            )
            torch.manual_seed(194309)
            selected = torch.randn(
                HEAD_DIM, dtype=torch.bfloat16, device="cuda"
            ) * 0.125
            kv[high_slot].copy_(selected)
            indices = torch.full((4,), -1, dtype=torch.int32, device="cuda")
            indices[0] = high_slot
            indptr = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
            heads_tested = []
            for num_heads in (1, 16):
                q = torch.randn(
                    1,
                    num_heads,
                    HEAD_DIM,
                    dtype=torch.bfloat16,
                    device="cuda",
                ) * 0.125
                actual = flydsl_op(
                    q=q,
                    kv=kv,
                    indices=indices,
                    indptr=indptr,
                    scale=DEFAULT_SCALE,
                    attn_sink=None,
                )
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    actual[0], selected.expand(num_heads, -1), atol=0, rtol=0
                )
                heads_tested.append(num_heads)
            large_result.update(
                {
                    "status": "pass",
                    "heads_tested": heads_tested,
                    "high_slot": high_slot,
                    "row_byte_offset": high_slot * row_bytes,
                    "allocation_bytes": kv.numel() * kv.element_size(),
                }
            )
            del actual, indices, indptr, q, selected, kv
            torch.cuda.empty_cache()
        except torch.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            large_result.update({"status": "skipped_oom", "error": repr(error)})

    return {"cases": cases, "large_address": large_result}


def _make_uniform_ragged(
    num_queries: int, row_length: int, num_kv: int
) -> tuple[torch.Tensor, torch.Tensor]:
    query_base = (
        torch.arange(num_queries, dtype=torch.int64, device="cuda")[:, None] * 37
    )
    offsets = (
        torch.arange(row_length, dtype=torch.int64, device="cuda")[None, :] * 13
    )
    indices = ((query_base + offsets) % num_kv).to(torch.int32).reshape(-1)
    indptr = torch.arange(
        0,
        (num_queries + 1) * row_length,
        row_length,
        dtype=torch.int32,
        device="cuda",
    )
    return indices.contiguous(), indptr.contiguous()


def _measure_ms(
    function: Callable[[], Any], warmup: int, repeats: int
) -> tuple[float, list[float]]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return statistics.median(samples), samples


def _run_benchmarks(
    flydsl_op: Callable[..., torch.Tensor],
    triton_op: Callable[..., torch.Tensor],
    min_repeats: int = 0,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    shapes = (
        (1, 16, 256, 8192, 30),
        (16, 16, 512, 8192, 20),
        (64, 16, 2048, 16384, 10),
        (256, 16, 2048, 32768, 5),
    )
    for num_queries, num_heads, row_length, num_kv, repeats in shapes:
        repeats = max(repeats, min_repeats)
        torch.manual_seed(950 + num_queries + row_length)
        q = (
            torch.randn(
                num_queries,
                num_heads,
                HEAD_DIM,
                dtype=torch.bfloat16,
                device="cuda",
            )
            * 0.125
        )
        kv = (
            torch.randn(num_kv, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
            * 0.125
        )
        indices, indptr = _make_uniform_ragged(
            num_queries, row_length, num_kv
        )
        sink = torch.linspace(
            -0.5, 0.5, num_heads, dtype=torch.float32, device="cuda"
        )
        flydsl_out = torch.empty_like(q)

        compile_start = time.perf_counter()
        flydsl_op(
            q=q,
            kv=kv,
            indices=indices,
            indptr=indptr,
            scale=DEFAULT_SCALE,
            attn_sink=sink,
            out=flydsl_out,
        )
        torch.cuda.synchronize()
        flydsl_first_call_ms = (time.perf_counter() - compile_start) * 1000.0

        compile_start = time.perf_counter()
        triton_out = triton_op(
            q=q,
            kv=kv,
            indices=indices,
            indptr=indptr,
            scale=DEFAULT_SCALE,
            attn_sink=sink,
            nope_head_dim=NOPE_HEAD_DIM,
            rope_head_dim=ROPE_HEAD_DIM,
        )
        torch.cuda.synchronize()
        triton_first_call_ms = (time.perf_counter() - compile_start) * 1000.0
        torch.testing.assert_close(
            flydsl_out, triton_out, atol=4.0e-2, rtol=4.0e-2
        )

        def run_flydsl():
            return flydsl_op(
                q=q,
                kv=kv,
                indices=indices,
                indptr=indptr,
                scale=DEFAULT_SCALE,
                attn_sink=sink,
                out=flydsl_out,
            )

        def run_triton_kernel():
            return triton_op(
                q=q,
                kv=kv,
                indices=indices,
                indptr=indptr,
                scale=DEFAULT_SCALE,
                attn_sink=sink,
                nope_head_dim=NOPE_HEAD_DIM,
                rope_head_dim=ROPE_HEAD_DIM,
            )

        def run_triton_with_copy():
            return triton_out.copy_(
                triton_op(
                    q=q,
                    kv=kv,
                    indices=indices,
                    indptr=indptr,
                    scale=DEFAULT_SCALE,
                    attn_sink=sink,
                    nope_head_dim=NOPE_HEAD_DIM,
                    rope_head_dim=ROPE_HEAD_DIM,
                )
            )

        flydsl_ms, flydsl_samples = _measure_ms(run_flydsl, 3, repeats)
        triton_kernel_ms, triton_kernel_samples = _measure_ms(
            run_triton_kernel, 3, repeats
        )
        triton_ms, triton_samples = _measure_ms(
            run_triton_with_copy, 3, repeats
        )
        selected_tokens = num_queries * row_length
        results.append(
            {
                "num_queries": num_queries,
                "num_heads": num_heads,
                "row_length": row_length,
                "num_kv": num_kv,
                "selected_tokens": selected_tokens,
                "repeats": repeats,
                "flydsl_pretiming_call_ms": flydsl_first_call_ms,
                "triton_pretiming_call_ms": triton_first_call_ms,
                "flydsl_median_ms": flydsl_ms,
                "triton_median_ms": triton_ms,
                "speedup_vs_triton": triton_ms / flydsl_ms,
                "triton_kernel_median_ms": triton_kernel_ms,
                "speedup_vs_triton_kernel": triton_kernel_ms / flydsl_ms,
                "flydsl_selected_tokens_per_s": selected_tokens / (flydsl_ms / 1000),
                "triton_selected_tokens_per_s": selected_tokens / (triton_ms / 1000),
                "flydsl_samples_ms": flydsl_samples,
                "triton_samples_ms": triton_samples,
                "triton_kernel_samples_ms": triton_kernel_samples,
                "error": _error_stats(flydsl_out, triton_out),
            }
        )
        del q, kv, indices, indptr, sink, flydsl_out, triton_out
        torch.cuda.empty_cache()
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aiter-tree", type=Path, required=True)
    parser.add_argument("--vllm-tree", type=Path, required=True)
    parser.add_argument("--kernel-file", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--mode", choices=("all", "correctness", "benchmark"), default="all"
    )
    parser.add_argument("--large-address-test", action="store_true")
    parser.add_argument(
        "--force-split-kv",
        action="store_true",
        help="force the experimental two-group H=16 split-KV path",
    )
    parser.add_argument(
        "--benchmark-min-repeats",
        type=int,
        default=0,
        help="raise each benchmark shape to at least this many timed samples",
    )
    args = parser.parse_args()

    if args.benchmark_min_repeats < 0:
        parser.error("--benchmark-min-repeats must be non-negative")

    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("a ROCm GPU is required")
    if _device_arch() != "gfx950":
        raise RuntimeError(f"expected gfx950, got {_device_arch()}")

    os.environ.setdefault("AITER_TRITON_ONLY", "1")
    aiter_tree = args.aiter_tree.resolve()
    vllm_tree = args.vllm_tree.resolve()
    kernel_file = args.kernel_file.resolve()
    aiter_commit = _git_revision(aiter_tree)
    vllm_commit = _git_revision(vllm_tree)
    if aiter_commit != EXPECTED_AITER_COMMIT:
        raise RuntimeError(
            f"AITER commit mismatch: expected {EXPECTED_AITER_COMMIT}, "
            f"got {aiter_commit}"
        )
    if vllm_commit != EXPECTED_VLLM_COMMIT:
        raise RuntimeError(
            f"vLLM commit mismatch: expected {EXPECTED_VLLM_COMMIT}, "
            f"got {vllm_commit}"
        )

    # AITER's source-tree JIT helpers intentionally use legacy top-level
    # imports (for example ``from build_targets import ...``).  Installed
    # wheels arrange those modules on sys.path, while a checkout does not.
    # Add the helper directory explicitly so this validator can exercise the
    # exact target checkout without installing or modifying it.
    sys.path.insert(0, str(aiter_tree / "aiter/jit/utils"))
    sys.path.insert(0, str(aiter_tree))
    environment = _environment()
    flydsl_version = environment["flydsl_version"]
    if (
        not isinstance(flydsl_version, str)
        or flydsl_version.split("+")[0] != "0.3.1"
    ):
        raise RuntimeError(
            f"expected FlyDSL 0.3.1, got {flydsl_version}"
        )

    flydsl_op, flydsl_module = _load_flydsl_op(kernel_file)
    if args.force_split_kv:
        base_flydsl_op = flydsl_op

        def forced_split_flydsl_op(**kwargs):
            return base_flydsl_op(split_kv=True, **kwargs)

        flydsl_op = forced_split_flydsl_op
    triton_op, triton_module = _load_triton_baseline(vllm_tree)
    report: dict[str, Any] = {
        "environment": environment,
        "sources": {
            "aiter_commit": aiter_commit,
            "vllm_commit": vllm_commit,
            "flydsl_kernel_sha256": _sha256(kernel_file),
            "flydsl_kernel": str(Path(flydsl_module.__file__).resolve()),
            "triton_baseline": str(Path(triton_module.__file__).resolve()),
            "force_split_kv": args.force_split_kv,
        },
    }
    if args.mode in ("all", "correctness"):
        report["correctness"] = _run_correctness(
            flydsl_op, triton_op, args.large_address_test
        )
    if args.mode in ("all", "benchmark"):
        report["benchmarks"] = _run_benchmarks(
            flydsl_op, triton_op, args.benchmark_min_repeats
        )

    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded, flush=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
