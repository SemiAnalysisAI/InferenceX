#!/usr/bin/env python3
"""Merge DSpark draft-group flags from every PP worker into scheduler KV config."""

from __future__ import annotations

import os
from pathlib import Path
import py_compile


site = Path(
    os.environ.get(
        "VLLM_SITE",
        "/usr/local/lib/python3.12/dist-packages",
    )
)
path = site / "vllm/v1/core/kv_cache_utils.py"
marker = "K3 PP scheduler merges global DSpark draft groups"
source = path.read_text()
if marker in source:
    print(f"{path}: already patched")
    raise SystemExit(0)

old = """    # All workers have the same kv_cache_config except layer names, so use
    # an arbitrary one to initialize the scheduler.
    cfg = copy.deepcopy(kv_cache_configs[0])
    for group in cfg.kv_cache_groups:
"""
new = f"""    # {marker}. Draft attention lives on the last PP stage only, so PP0
    # configs never carry is_eagle_group=True unless we merge across workers.
    cfg = copy.deepcopy(kv_cache_configs[0])
    for gidx, group in enumerate(cfg.kv_cache_groups):
        group.is_eagle_group = any(
            worker_cfg.kv_cache_groups[gidx].is_eagle_group
            for worker_cfg in kv_cache_configs
            if gidx < len(worker_cfg.kv_cache_groups)
        )
"""
if source.count(old) != 1:
    raise RuntimeError(
        f"{path}: expected one scheduler KV merge anchor, found {source.count(old)}"
    )
path.write_text(source.replace(old, new, 1))
py_compile.compile(str(path), doraise=True)
print(f"{path}: patched and compiled")
