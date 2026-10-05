#!/usr/bin/env bash
set -eo pipefail
bash /configs/install-torchao.sh

# Test the upstream UCX event fix with SGLang's original asynchronous progress.
python3 - <<'PY'
import hashlib
from pathlib import Path

path = Path("/sgl-workspace/sglang/python/sglang/srt/disaggregation/nixl/conn.py")
expected = "28da79ad06baa8c1b725bc98983ca570f82a73b49439873831505ab7f1eef601"
actual = hashlib.sha256(path.read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit(f"Expected unmodified GLM5.2 NIXL conn.py: {actual}")
print(f"GLM5.2 original NIXL progress configuration: {actual}")
PY

# Preserve the image's Torch/NumPy and install only the matching CUDA 13 backend.
python3 -m pip install --no-deps --only-binary=:all: --require-hashes -r /dev/stdin <<'REQ'
nixl==1.4.0 --hash=sha256:aad5065c46ead71c96f485785a2cb6ef782b5a32cc6450aa7eff1735012d16c2
nixl-cu13==1.4.0 --hash=sha256:b9184de88d5919d1ec82b61b398c59396af31e1e34e6e023db4b5f01c6f07171
REQ

python3 - <<'PY'
import importlib.metadata
import json
import platform

versions = {
    name: importlib.metadata.version(name)
    for name in ("nixl", "nixl-cu13", "torch", "numpy")
}
if any(versions[name] != "1.4.0" for name in ("nixl", "nixl-cu13")):
    raise SystemExit(f"Unexpected NIXL installation: {versions}")
print("GLM5.2 upstream NIXL candidate: " + json.dumps({
    "versions": versions,
    "python": platform.python_version(),
    "architecture": platform.machine(),
    "sglang_conn_modified": False,
}))
PY
