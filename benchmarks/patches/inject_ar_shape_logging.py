#!/usr/bin/env python3
"""
Inject allreduce shape logging into SGLang's parallel_state.py at runtime.

Patches GroupCoordinator._all_reduce_out_place to print tensor shapes on rank 0.
This patches the actual source file inside the container so that all worker
processes (forked by SGLang) pick up the change.

Usage: python3 inject_ar_shape_logging.py
"""
import importlib
import os
import re
import sys
import textwrap


def find_and_patch(module_path: str, target_method: str, log_tag: str) -> bool:
    """Find a Python module file and inject shape logging into a method."""
    try:
        mod = importlib.import_module(module_path)
        filepath = mod.__file__
    except (ImportError, AttributeError) as e:
        print(f"[AR_SHAPE] Could not import {module_path}: {e}")
        return False

    if not filepath or not os.path.exists(filepath):
        print(f"[AR_SHAPE] File not found for {module_path}")
        return False

    with open(filepath, "r") as f:
        src = f.read()

    # Look for the method definition
    # Match: "def <method_name>(self, <args>):"
    pattern = rf"(    def {re.escape(target_method)}\(self[^)]*\)[^:]*:.*\n)"
    match = re.search(pattern, src)
    if not match:
        print(f"[AR_SHAPE] Could not find {target_method} in {filepath}")
        return False

    # Check if already patched
    if "[AR_SHAPE_LOG]" in src:
        print(f"[AR_SHAPE] Already patched: {filepath}")
        return True

    # Find the first argument name after self (the tensor)
    sig_match = re.search(
        rf"def {re.escape(target_method)}\(self,\s*(\w+)", src
    )
    tensor_name = sig_match.group(1) if sig_match else "input_"

    # Build the logging code to insert after the method def line
    log_code = textwrap.dedent(f"""\
        # [AR_SHAPE_LOG] Injected shape logging
        try:
            import torch.distributed as _dist
            if not _dist.is_initialized() or _dist.get_rank() == 0:
                _s = list({tensor_name}.shape)
                _b = {tensor_name}.numel() * {tensor_name}.element_size()
                print(f"[AR_SHAPE] {log_tag} shape={{_s}} dtype={{{tensor_name}.dtype}} bytes={{_b}}", flush=True)
        except Exception:
            pass
    """)

    # Indent to match method body (8 spaces)
    indented_log = textwrap.indent(log_code, "        ")

    # Insert after the method definition line
    end_of_def = match.end()
    new_src = src[:end_of_def] + indented_log + src[end_of_def:]

    with open(filepath, "w") as f:
        f.write(new_src)
    print(f"[AR_SHAPE] Patched {target_method} in {filepath}")
    return True


def patch_parallel_state():
    """Patch GroupCoordinator._all_reduce_out_place in parallel_state.py."""
    return find_and_patch(
        "sglang.srt.distributed.parallel_state",
        "_all_reduce_out_place",
        "out_place",
    )


def patch_sglang_custom_ar():
    """Patch CustomAllreduce.all_reduce_unreg in sglang's custom_all_reduce.py."""
    return find_and_patch(
        "sglang.srt.distributed.device_communicators.custom_all_reduce",
        "all_reduce_unreg",
        "sglang_unreg",
    )


def patch_aiter_custom_ar():
    """Patch CustomAllreduce.all_reduce_unreg in aiter's custom_all_reduce.py."""
    return find_and_patch(
        "aiter.dist.device_communicators.custom_all_reduce",
        "all_reduce_unreg",
        "aiter_unreg",
    )


def patch_top_level_all_reduce():
    """Patch GroupCoordinator.all_reduce — the single entry point for all allreduce calls."""
    return find_and_patch(
        "sglang.srt.distributed.parallel_state",
        "all_reduce",
        "entry",
    )


if __name__ == "__main__":
    print("[AR_SHAPE] Starting allreduce shape logging injection...")

    # Patch the top-level entry point (catches ALL allreduce calls)
    patch_top_level_all_reduce()

    # Patch the out-of-place path (catches custom AR method selection)
    patch_parallel_state()

    # Patch the low-level unreg call in both sglang and aiter
    patch_sglang_custom_ar()
    patch_aiter_custom_ar()

    print("[AR_SHAPE] Done. Shape logs will appear as [AR_SHAPE] lines in server output.")
