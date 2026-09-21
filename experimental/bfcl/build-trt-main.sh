#!/usr/bin/env bash
# Executed only while building the experimental image, never at server startup.
set -eo pipefail
source /infx/benchmarks/benchmark_lib.sh --validation-only
check_env_vars TRT_SOURCE_SHA TRT_BUILD_JOBS TRT_CUDA_ARCHS TRT_DEVEL_IMAGE
check_env_vars TRT_NVRTC_VERSION TRT_NVRTC_DEB_URL TRT_NVRTC_DEB_SHA256
[[ "$TRT_SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]]
[[ "$TRT_BUILD_JOBS" =~ ^[1-9][0-9]*$ ]]
command -v git
git lfs version
nvcc --version
python3 -m pip show torch

mkdir -p /trt-build/source
cd /trt-build/source
git init
git remote add origin https://github.com/NVIDIA/TensorRT-LLM.git
GIT_LFS_SKIP_SMUDGE=1 git fetch --depth=1 origin "$TRT_SOURCE_SHA"
GIT_LFS_SKIP_SMUDGE=1 git checkout --detach FETCH_HEAD
[[ "$(git rev-parse HEAD)" == "$TRT_SOURCE_SHA" ]]
git submodule update --init --recursive --depth=1
git lfs pull
git submodule status --recursive | tee /build-evidence/submodules.txt
if grep -Eq '^[-+U]' /build-evidence/submodules.txt; then
    echo 'Submodule checkout does not match the pinned source.' >&2
    exit 1
fi
git lfs fsck
git diff --exit-code
git diff --cached --exit-code
git show --no-patch --format=fuller HEAD > /build-evidence/upstream-commit.txt

# Reproduce main's CUDA dependency fix: the published devel image omits the
# NVRTC static archives despite reporting the pinned package version. Restore
# the official package payload inside this build container. Extraction avoids
# privileged dpkg maintainer scripts in the rootless Pyxis namespace.
grep -Fx "NVRTC_VER=\"$TRT_NVRTC_VERSION\"" docker/common/install_cuda_libs.sh
NVRTC_DEB=/trt-build/nvrtc-dev.deb
curl --fail --location --retry 3 "$TRT_NVRTC_DEB_URL" -o "$NVRTC_DEB"
printf '%s  %s\n' "$TRT_NVRTC_DEB_SHA256" "$NVRTC_DEB" | sha256sum --check
[[ "$(dpkg-deb -f "$NVRTC_DEB" Package)" == cuda-nvrtc-dev-13-4 ]]
[[ "$(dpkg-deb -f "$NVRTC_DEB" Version)" == "$TRT_NVRTC_VERSION" ]]
[[ "$(dpkg-deb -f "$NVRTC_DEB" Architecture)" == amd64 ]]
dpkg-deb --fsys-tarfile "$NVRTC_DEB" | tar --no-same-owner -xf - -C /
test -s /usr/local/cuda-13.4/targets/x86_64-linux/lib/libnvrtc_static.a
test -s /usr/local/cuda-13.4/targets/x86_64-linux/lib/libnvrtc-builtins_static.a
sha256sum "$NVRTC_DEB" > /build-evidence/nvrtc-package.sha256
dpkg-deb -f "$NVRTC_DEB" > /build-evidence/nvrtc-package-control.txt

# Build both Python and native extensions with the upstream build entrypoint.
# No precompiled reuse, fast-build kernel omissions, editable install, or source edits.
export TRTLLM_BUILD_SOURCE_COMMIT="$TRT_SOURCE_SHA"
export CONAN_HOME=/trt-build/conan PIP_CACHE_DIR=/trt-build/pip-cache
python3 scripts/build_wheel.py --build_root /trt-build/build --out-of-tree \
    --use_ccache -a "$TRT_CUDA_ARCHS" -j "$TRT_BUILD_JOBS"
git diff --exit-code
git diff --cached --exit-code

WHEELS=(/trt-build/build/dist/tensorrt_llm*.whl)
[[ ${#WHEELS[@]} == 1 && -f "${WHEELS[0]}" ]]
sha256sum "${WHEELS[0]}" > /build-evidence/wheel.sha256
# Preserve the NGC PyTorch ABI; do not let the wheel resolver replace it.
python3 -c 'import importlib.metadata as m; print("torch==" + m.version("torch"))' > /trt-build/torch-constraint.txt
python3 -m pip install -c /trt-build/torch-constraint.txt "${WHEELS[0]}[mx]"

# Match the upstream release stage's shared-library lookup.
mkdir -p /app/tensorrt_llm
LIB_DIR=$(python3 -c 'import site; print(site.getsitepackages()[0] + "/tensorrt_llm/libs")')
test -f "$LIB_DIR/libtensorrt_llm.so"
ln -s "$LIB_DIR" /app/tensorrt_llm/lib
echo /app/tensorrt_llm/lib > /etc/ld.so.conf.d/tensorrt_llm.conf
ldconfig

# Record installed source hashes and native-library hashes outside the checkout.
cd /
python3 - <<'PY'
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import tensorrt_llm
import torch

root = Path(tensorrt_llm.__file__).parent
source = Path('/trt-build/source/tensorrt_llm')
files = {}
for relative in (
    '_torch/attention/backends/sparse/minimax_m3/cache_manager.py',
    '_torch/attention/backends/sparse/minimax_m3/msa_backend.py',
    '_torch/pyexecutor/kv_cache/kv_cache_manager_v2.py',
):
    actual = (root / relative).read_bytes()
    assert actual == (source / relative).read_bytes(), relative
    files[relative] = hashlib.sha256(actual).hexdigest()
for path in sorted(root.rglob('*.so')):
    with path.open('rb') as stream:
        files[str(path.relative_to(root))] = hashlib.file_digest(stream, 'sha256').hexdigest()
assert any(name.endswith('.so') for name in files)
manifest = {
    'source_commit': os.environ['TRT_SOURCE_SHA'],
    'base_devel_image': os.environ['TRT_DEVEL_IMAGE'],
    'nvrtc_development_package': {
        'version': os.environ['TRT_NVRTC_VERSION'],
        'url': os.environ['TRT_NVRTC_DEB_URL'],
        'sha256': os.environ['TRT_NVRTC_DEB_SHA256'],
    },
    'cuda_archs': os.environ['TRT_CUDA_ARCHS'],
    'package_version': metadata.version('tensorrt_llm'),
    'torch_version': torch.__version__,
    'cuda_version': torch.version.cuda,
    'installed_files': files,
    'runtime_source_patches': False,
    'build_type': 'upstream source wheel on pinned NGC devel image',
}
text = json.dumps(manifest, indent=2) + '\n'
Path('/opt/inferencex-trt-main-build.json').write_text(text)
Path('/build-evidence/build-manifest.json').write_text(text)
print(text)
PY
python3 -m pip freeze > /build-evidence/installed-packages.txt
date -u +%FT%TZ > /build-evidence/build-completed-at.txt
