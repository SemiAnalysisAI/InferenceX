#!/usr/bin/env python3
"""Add concise startup diagnostics for DSpark KV-group propagation."""

from __future__ import annotations

import os
from pathlib import Path
import py_compile


site = Path(
    os.environ.get("VLLM_SITE", "/usr/local/lib/python3.12/dist-packages")
)

kv_utils = site / "vllm/v1/core/kv_cache_utils.py"
source = kv_utils.read_text()
marker = "K3 DSpark scheduler KV flags"
if marker not in source:
    function_start = source.index("def generate_scheduler_kv_cache_config(")
    function_end = source.index("\ndef ", function_start + 5)
    block = source[function_start:function_end]
    old = """    return cfg
"""
    new = f"""    logger.warning(
        "{marker}: inputs=%s merged=%s",
        [
            [group.is_eagle_group for group in worker_cfg.kv_cache_groups]
            for worker_cfg in kv_cache_configs
        ],
        [group.is_eagle_group for group in cfg.kv_cache_groups],
    )
    return cfg
"""
    if block.count(old) != 1:
        raise RuntimeError(f"{kv_utils}: scheduler diagnostic anchor missing")
    block = block.replace(old, new, 1)
    kv_utils.write_text(source[:function_start] + block + source[function_end:])

coordinator = site / "vllm/v1/core/kv_cache_coordinator.py"
source = coordinator.read_text()
marker = "K3 DSpark coordinator eagle groups"
if marker not in source:
    old = """        # During chunked prefill with EAGLE, the single next prefill lookahead
"""
    new = f"""        logger.warning(
            "{marker}: use_eagle=%s ids=%s flags=%s",
            use_eagle,
            sorted(self.eagle_group_ids),
            [group.is_eagle_group for group in kv_cache_config.kv_cache_groups],
        )

        # During chunked prefill with EAGLE, the single next prefill lookahead
"""
    if source.count(old) != 1:
        raise RuntimeError(f"{coordinator}: coordinator diagnostic anchor missing")
    coordinator.write_text(source.replace(old, new, 1))

for path in (kv_utils, coordinator):
    py_compile.compile(str(path), doraise=True)
    print(f"{path}: diagnostics patched and compiled")
