#!/usr/bin/env bash
set -eo pipefail
# Download, verify, and install the published wheel inside each backend container.
uv pip install --system --no-deps --reinstall --require-hashes \
    'https://test-files.pythonhosted.org/packages/f3/00/fb2847f5564864be132d0f70384dde5688689d9f91b2c81d2609632f36e4/mooncake_transfer_engine_cuda13-0.3.14.dev20260910-cp312-cp312-manylinux_2_28_aarch64.whl#sha256=c55fcc42cf189fcdecdcdf9d776ef84b348a9970a9bd3aad2fad64cb2d9e6f82'
