#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Candidate synchronous UCX progress for GLM GB200 SGLang v0.5.17 prefill.

The input hash identifies the upstream release source, not yet the installed
image. Unknown image contents fail closed. This workaround removes the progress
thread path reported in ai-dynamo/nixl#2102; GPU recovery remains unverified.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

SOURCE_PATH = "srt/disaggregation/nixl/conn.py"
SOURCE_HASH = "28da79ad06baa8c1b725bc98983ca570f82a73b49439873831505ab7f1eef601"
PATCHED_HASH = "c1c599fb66b35348ff7ed85b815f664f9b1d295b8c460589d953fe846606819f"


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError("unsupported SGLang source: patch anchor is not unique")
    return source.replace(old, new, 1)


def transform_source(source: str) -> str:
    source = replace_once(
        source,
        "        num_threads = 8 if disaggregation_mode == DisaggregationMode.PREFILL else 0\n",
        "        self._nixl_synchronous_ucx = (\n"
        '            backend == "UCX"\n'
        "            and disaggregation_mode == DisaggregationMode.PREFILL\n"
        "        )\n"
        "        num_threads = (\n"
        "            0 if self._nixl_synchronous_ucx\n"
        "            else 8 if disaggregation_mode == DisaggregationMode.PREFILL else 0\n"
        "        )\n",
    )
    source = replace_once(
        source,
        "            num_threads=num_threads,\n",
        "            num_threads=num_threads,\n"
        "            enable_prog_thread=not self._nixl_synchronous_ucx,\n",
    )
    source = replace_once(
        source,
        "        self.agent.create_backend(backend, backend_params)\n",
        "        if self._nixl_synchronous_ucx:\n"
        "            # Explicit backends do not inherit the agent's thread count.\n"
        '            backend_params["num_threads"] = "0"\n'
        "        self.agent.create_backend(backend, backend_params)\n",
    )
    return replace_once(
        source,
        "            try:\n"
        "                if self.check_status(room) == KVPoll.Failed:\n",
        "            try:\n"
        "                if self._nixl_synchronous_ucx:\n"
        "                    import torch\n\n"
        "                    # CUDA context is thread-local; selection errors must fail the room.\n"
        "                    torch.cuda.set_device(self.kv_args.gpu_id)\n"
        "                if self.check_status(room) == KVPoll.Failed:\n",
    )


def patch_package(package_root: Path) -> bool:
    path = package_root / SOURCE_PATH
    original = path.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    if digest == PATCHED_HASH:
        return False
    if digest != SOURCE_HASH:
        raise RuntimeError(
            f"unsupported SGLang source hash for {SOURCE_PATH}: {digest}"
        )
    patched = transform_source(original.decode())
    compile(patched, str(path), "exec")
    payload = patched.encode()
    if hashlib.sha256(payload).hexdigest() != PATCHED_HASH:
        raise RuntimeError("candidate patch output changed")
    path.write_bytes(payload)
    return True


def main(argv: list[str]) -> int:
    if len(argv) > 2:
        print(f"Usage: {argv[0]} [SGLANG_PACKAGE_ROOT]", file=sys.stderr)
        return 2
    try:
        if len(argv) == 2:
            root = Path(argv[1])
        else:
            spec = importlib.util.find_spec("sglang")
            if spec is None or not spec.submodule_search_locations:
                raise RuntimeError("sglang is not installed")
            root = Path(next(iter(spec.submodule_search_locations)))
        changed = patch_package(root)
    except (OSError, RuntimeError, UnicodeError, ValueError, SyntaxError) as error:
        print(f"ERROR: NIXL prefill candidate rejected: {error}", file=sys.stderr)
        return 1
    state = "Patched" if changed else "Already patched"
    print(f"{state} SGLang v0.5.17 UCX prefill synchronous-progress candidate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
