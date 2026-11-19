#!/usr/bin/env bash
# GLM-5.2 MXFP4 on lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261118. The TP8
# points serve the Quark MXFP4 checkpoint on the sgl-amd-mono monokernel, which
# this image's sgl-amd-mono already accepts (sgl-amd-mono#123), so nothing is
# patched in; the one thing it cannot do without is FlyDSL 0.4.0. Missing it,
# SGLang drops back to its own forward and the server still answers, so check it
# here and fail the worker instead of reporting a quiet fallback as a result.
set -euo pipefail
python3 - <<'EOF'
from importlib.metadata import version

from packaging.version import Version

have = version("flydsl")
assert Version(have) >= Version("0.4.0"), f"glm5.2: FlyDSL {have}, the monokernel needs 0.4.0"
print(f"glm5.2: FlyDSL {have}, the monokernel can run")
EOF
