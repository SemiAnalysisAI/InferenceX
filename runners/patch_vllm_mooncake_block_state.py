#!/usr/bin/env python3
"""Backport vLLM #54853 to the pinned DeepSeek V4.1 preview image.

Upstream: 0b066293f3c738a0cbd3a087bf893f2f4dcd61f2 (three production files).
Keep the missing-table assertion: resolve every scheduled request's real table.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import subprocess

PATCH = Path(__file__).with_name('vllm_mooncake_block_state.patch')


def apply_patch(root: Path) -> bool:
    """Check the complete upstream patch before modifying installed vLLM."""
    command = ['git', 'apply', '--check', str(PATCH)]
    check = subprocess.run(command, cwd=root, capture_output=True, text=True)
    if check.returncode:
        reverse = subprocess.run(
            ['git', 'apply', '--reverse', '--check', str(PATCH)],
            cwd=root, capture_output=True, text=True,
        )
        if reverse.returncode == 0:
            print('Mooncake block-state backport already applied')
            return False
        raise RuntimeError('Unsupported vLLM source; refusing partial backport:\n' + check.stderr)
    subprocess.run(['git', 'apply', str(PATCH)], cwd=root, check=True)
    for path in (
        'vllm/v1/core/sched/output.py',
        'vllm/v1/core/sched/scheduler.py',
        'vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py',
    ):
        ast.parse((root / path).read_text(), filename=path)
    print('Applied upstream vLLM #54853 Mooncake block-state fix')
    return True


if __name__ == '__main__':
    spec = importlib.util.find_spec('vllm')
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError('vLLM package not found')
    apply_patch(Path(next(iter(spec.submodule_search_locations))).parent)
