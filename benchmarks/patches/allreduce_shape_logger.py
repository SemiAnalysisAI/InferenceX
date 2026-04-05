"""
Monkey-patch SGLang's GroupCoordinator.all_reduce to log tensor shapes
entering the custom allreduce kernel (cross_device_reduce_2stage).

Usage: Set PYTHONPATH to include the directory containing sitecustomize.py
which imports this module, OR call patch() directly before launching SGLang.

Logs are written to /workspace/allreduce_shapes.log (one line per call on rank 0).
After the run, the log can be post-processed to get unique shapes and counts.
"""

import atexit
import collections
import os

_shape_counts = collections.Counter()
_log_file = None
_original_all_reduce = None
_original_all_reduce_out_place = None
_patched = False
# Limit per-call logging to avoid flooding stdout; summary is printed at exit.
_MAX_LOG_LINES = 200
_log_line_count = 0


def _get_rank():
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


def _patched_all_reduce_out_place(self, input_, outplace_all_reduce_method):
    """Wrapper around _all_reduce_out_place that logs shapes for custom AR calls."""
    global _log_line_count
    rank = _get_rank()
    if rank == 0:
        shape_key = (tuple(input_.shape), str(input_.dtype), outplace_all_reduce_method)
        _shape_counts[shape_key] += 1
        if _log_line_count < _MAX_LOG_LINES:
            print(
                f"[AR_SHAPE] method={outplace_all_reduce_method} "
                f"shape={list(input_.shape)} dtype={input_.dtype} "
                f"numel={input_.numel()} bytes={input_.numel() * input_.element_size()}",
                flush=True,
            )
            _log_line_count += 1
    return _original_all_reduce_out_place(self, input_, outplace_all_reduce_method)


def _patched_all_reduce(self, input_):
    """Wrapper around all_reduce that logs shapes for ALL allreduce calls (including in-place/deterministic)."""
    global _log_line_count
    rank = _get_rank()
    if rank == 0 and _log_line_count < _MAX_LOG_LINES:
        shape_key = (tuple(input_.shape), str(input_.dtype), "all")
        _shape_counts[shape_key] += 1
        if _log_line_count < _MAX_LOG_LINES:
            print(
                f"[AR_SHAPE_ENTRY] shape={list(input_.shape)} dtype={input_.dtype} "
                f"numel={input_.numel()} bytes={input_.numel() * input_.element_size()}",
                flush=True,
            )
            _log_line_count += 1
    return _original_all_reduce(self, input_)


def _print_summary():
    """Print aggregated shape summary at process exit."""
    rank = _get_rank()
    if rank != 0 or not _shape_counts:
        return

    log_path = os.environ.get("AR_SHAPE_LOG", "/workspace/allreduce_shapes.log")
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("[AR_SHAPE_SUMMARY] AllReduce tensor shapes (rank 0):")
    lines.append(f"{'Count':>8}  {'Method':<12}  {'Shape':<30}  {'Dtype':<16}  {'Bytes':<12}")
    lines.append("-" * 80)

    for (shape, dtype, method), count in _shape_counts.most_common():
        import torch
        # Compute element size from dtype string
        elem_size = 2  # default bf16
        if "float32" in dtype:
            elem_size = 4
        elif "float16" in dtype or "bfloat16" in dtype:
            elem_size = 2
        elif "float8" in dtype:
            elem_size = 1
        numel = 1
        for s in shape:
            numel *= s
        nbytes = numel * elem_size
        lines.append(f"{count:>8}  {method:<12}  {str(list(shape)):<30}  {dtype:<16}  {nbytes:<12}")

    lines.append("=" * 80)
    summary = "\n".join(lines)
    print(summary, flush=True)

    try:
        with open(log_path, "w") as f:
            f.write(summary + "\n")
        print(f"[AR_SHAPE] Summary written to {log_path}", flush=True)
    except Exception as e:
        print(f"[AR_SHAPE] Failed to write log: {e}", flush=True)


def patch():
    """Apply the monkey-patch to GroupCoordinator."""
    global _original_all_reduce, _original_all_reduce_out_place, _patched
    if _patched:
        return

    try:
        from sglang.srt.distributed.parallel_state import GroupCoordinator
    except ImportError:
        print("[AR_SHAPE] Could not import GroupCoordinator, skipping patch", flush=True)
        return

    _original_all_reduce = GroupCoordinator.all_reduce
    _original_all_reduce_out_place = GroupCoordinator._all_reduce_out_place

    GroupCoordinator.all_reduce = _patched_all_reduce
    GroupCoordinator._all_reduce_out_place = _patched_all_reduce_out_place
    _patched = True

    atexit.register(_print_summary)
    print("[AR_SHAPE] Monkey-patch installed: logging allreduce tensor shapes on rank 0", flush=True)
