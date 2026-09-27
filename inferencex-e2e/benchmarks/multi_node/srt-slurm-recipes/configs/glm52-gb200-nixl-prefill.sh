#!/usr/bin/env bash
set -eo pipefail
bash /configs/install-torchao.sh

# Avoid the shared UCX worker's event-arm path in the unchanged GLM5.2 images.
# This is a scoped workaround, not the UCX event fix: openucx/ucx#11499.
python3 - <<'PY'
import hashlib
from pathlib import Path

path = Path("/sgl-workspace/sglang/python/sglang/srt/disaggregation/nixl/conn.py")
original = "28da79ad06baa8c1b725bc98983ca570f82a73b49439873831505ab7f1eef601"
patched = "00f857bc02cd54a1e869e44cd54d04bfb0dc8b2ada5e62a780e30dc6a113cf66"
source = path.read_text()
digest = hashlib.sha256(source.encode()).hexdigest()
if digest == patched:
    print(f"GLM5.2 UCX prefill setup already applied: {digest}")
    raise SystemExit(0)
if digest != original:
    raise SystemExit(f"Unexpected GLM5.2 NIXL conn.py: {digest}")
source = source.replace(
    '        num_threads = 8 if disaggregation_mode == DisaggregationMode.PREFILL else 0\n',
    '        num_threads = 8 if disaggregation_mode == DisaggregationMode.PREFILL else 0\n'
    '        synchronous_ucx = backend == "UCX" and disaggregation_mode == DisaggregationMode.PREFILL\n',
).replace(
    '            backends=[],\n',
    '            backends=[],\n            enable_prog_thread=not synchronous_ucx,\n',
).replace(
    '        self.agent.create_backend(backend, backend_params)\n',
    '        if synchronous_ucx:\n            backend_params["num_threads"] = "0"\n'
    '        self.agent.create_backend(backend, backend_params)\n',
)
if hashlib.sha256(source.encode()).hexdigest() != patched:
    raise SystemExit("Unexpected GLM5.2 NIXL patch output")
path.write_text(source)
print(f"GLM5.2 UCX prefill setup: {original} -> {patched}")
PY
