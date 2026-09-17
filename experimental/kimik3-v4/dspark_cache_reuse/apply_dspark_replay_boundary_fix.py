#!/usr/bin/env python3
"""Backport the EAGLE/Mamba replay-boundary fix for Kimi-K3 DSpark."""

from __future__ import annotations

import os
from pathlib import Path
import py_compile
import subprocess


site = Path(
    os.environ.get(
        "VLLM_SITE",
        "/usr/local/lib/python3.12/dist-packages",
    )
)
patch = Path(
    os.environ.get(
        "DSPARK_REPLAY_PATCH",
        "/ppcompat/vllm-51295-replay-boundary.patch",
    )
)
marker = "K3 DSpark replay boundary retained across hybrid groups"


def apply_selected_patch() -> None:
    includes = (
        "vllm/v1/core/sched/scheduler.py",
        "vllm/v1/core/single_type_kv_cache_manager.py",
    )
    command = ["git", "apply"]
    for include in includes:
        command.append(f"--include={include}")
    command.append(str(patch))
    check = subprocess.run(
        [*command[:-1], "--check", command[-1]],
        cwd=site,
        capture_output=True,
        text=True,
    )
    if check.returncode == 0:
        subprocess.run(command, cwd=site, check=True)
        return
    reverse = subprocess.run(
        [*command[:-1], "--reverse", "--check", command[-1]],
        cwd=site,
        capture_output=True,
        text=True,
    )
    if reverse.returncode != 0:
        raise RuntimeError(
            f"cannot apply replay patch:\n{check.stderr}\n{reverse.stderr}"
        )


def replace_in_class_function(
    source: str,
    class_name: str,
    function_name: str,
    old: str,
    new: str,
) -> str:
    class_start = source.index(f"class {class_name}(")
    class_end = source.find("\nclass ", class_start + 7)
    if class_end < 0:
        class_end = len(source)
    block = source[class_start:class_end]
    function_start = block.index(f"    def {function_name}(")
    function_end = block.find("\n    def ", function_start + 8)
    if function_end < 0:
        function_end = len(block)
    function = block[function_start:function_end]
    if new in function:
        return source
    if function.count(old) != 1:
        raise RuntimeError(
            f"{class_name}.{function_name}: expected one anchor, "
            f"found {function.count(old)}"
        )
    function = function.replace(old, new, 1)
    block = block[:function_start] + function + block[function_end:]
    return source[:class_start] + block + source[class_end:]


apply_selected_patch()

coordinator = site / "vllm/v1/core/kv_cache_coordinator.py"
source = coordinator.read_text()
if marker not in source:
    helper_anchor = (
        "    def cache_blocks("
        "self, request: Request, num_computed_tokens: int) -> None:\n"
    )
    helper = f"""    def get_replay_boundary(self, request: Request) -> int:
        # {marker}. A hybrid hit is capped by its shortest group. Under
        # EAGLE/DSpark, retain every group's state one scheduler unit below
        # the prompt's last aligned boundary, where replay actually resumes.
        if not self.eagle_group_ids:
            return request.num_prompt_tokens - 1
        aligned = (
            request.num_prompt_tokens
            // self.scheduler_block_size
            * self.scheduler_block_size
        )
        return max(aligned - self.scheduler_block_size, 0)

"""
    if source.count(helper_anchor) < 1:
        raise RuntimeError("coordinator cache_blocks anchor missing")
    source = source.replace(helper_anchor, helper + helper_anchor, 1)

base_old = """        for manager in self.single_type_managers:
            # Only cache tokens with finalized KV. The last num_reprefillable_tokens
"""
base_new = """        replay_boundary = self.get_replay_boundary(request)
        for manager in self.single_type_managers:
            # Only cache tokens with finalized KV. The last num_reprefillable_tokens
"""
source = replace_in_class_function(
    source,
    "KVCacheCoordinator",
    "cache_blocks",
    base_old,
    base_new,
)
base_call_old = """                retention_interval=self.retention_interval,
            )
"""
base_call_new = """                retention_interval=self.retention_interval,
                replay_boundary=replay_boundary,
            )
"""
source = replace_in_class_function(
    source,
    "KVCacheCoordinator",
    "cache_blocks",
    base_call_old,
    base_call_new,
)

hybrid_old = """        cached_num_computed_tokens = self._align_cacheable(num_computed_tokens)
        for manager in self.single_type_managers:
"""
hybrid_new = """        cached_num_computed_tokens = self._align_cacheable(num_computed_tokens)
        replay_boundary = self.get_replay_boundary(request)
        for manager in self.single_type_managers:
"""
source = replace_in_class_function(
    source,
    "HybridKVCacheCoordinator",
    "cache_blocks",
    hybrid_old,
    hybrid_new,
)
source = replace_in_class_function(
    source,
    "HybridKVCacheCoordinator",
    "cache_blocks",
    base_call_old,
    base_call_new,
)

coordinator.write_text(source)
for relative in (
    "vllm/v1/core/kv_cache_coordinator.py",
    "vllm/v1/core/sched/scheduler.py",
    "vllm/v1/core/single_type_kv_cache_manager.py",
):
    py_compile.compile(str(site / relative), doraise=True)
    print(f"py_compile OK: {relative}")
print("Kimi-K3 DSpark replay-boundary fix ready")
