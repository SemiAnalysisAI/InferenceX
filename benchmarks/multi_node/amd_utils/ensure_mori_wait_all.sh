#!/usr/bin/env bash
# Verify that job.slurm selected the source-built mori #341 image. Serving must
# never mutate site-packages or silently fall back to the Python polling path.
set -euo pipefail

python3 - <<'PY'
import mori
from mori.io import IOEngine, StatusCode

version = getattr(mori, "__version__", "?")
wait_all = hasattr(IOEngine, "wait_all")
print("[k3-mori-waitall] mori=", version, "at", mori.__file__)
print("[k3-mori-waitall] IOEngine.wait_all=", wait_all)
if not wait_all:
    raise RuntimeError(
        "IOEngine.wait_all missing: job.slurm must prepare the pinned "
        "ROCm/mori#341 derived image before serving"
    )
assert StatusCode is not None
PY
