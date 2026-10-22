#!/usr/bin/env bash
set -eo pipefail

# The router uses a separate image without vLLM.
if ! python3 -c 'import importlib.metadata; importlib.metadata.version("vllm")' >/dev/null 2>&1; then
    exit 0
fi
source /infmax-workspace/benchmarks/check_env.sh
check_env_vars KIMIK3_KERNEL_BUILD_JOBS ROCM_PATH
[[ "$KIMIK3_KERNEL_BUILD_JOBS" =~ ^[1-9][0-9]*$ ]]

package=$(python3 -c 'import importlib.util; print(next(iter(importlib.util.find_spec("vllm").submodule_search_locations)))')
version=$(python3 -c 'import importlib.metadata; print(importlib.metadata.version("vllm"))')
revision=$(python3 -c 'import re, sys; match = re.search(r"\+g([0-9a-f]{7,40})(?:\.|$)", sys.argv[1]); assert match, f"Cannot resolve nightly source: {sys.argv[1]}"; print(match[1])' "$version")
patch_dir=$(cd "$(dirname "$0")/patches/kimik3-nightly" && pwd)
patches=("$patch_dir"/*.patch)
[[ -f "${patches[0]}" ]]
signature=$({ printf '%s\n' "$version"; sha256sum "${patches[@]}"; } | sha256sum | cut -d' ' -f1)
marker="$package/.infx-kimik3-kernels"
exec 9> /tmp/infx-kimik3-kernels.lock
flock 9
if [[ -f "$marker" && $(< "$marker") == "$signature" ]]; then
    echo "Kimi-K3 kernel patches already installed: $signature"
    exit 0
fi

build_root=$(mktemp -d /tmp/infx-kimik3-kernels.XXXXXX)
curl --fail --location --retry 3 --max-time 300 \
    "https://codeload.github.com/vllm-project/vllm/tar.gz/$revision" \
    -o "$build_root/source.tgz"
mkdir "$build_root/source"
tar -xzf "$build_root/source.tgz" --strip-components=1 -C "$build_root/source"
git -C "$build_root/source" init -q
applied=()
for patch in "${patches[@]}"; do
    if git -C "$build_root/source" apply --reverse --check "$patch" 2>/dev/null; then
        echo "Already included in $version: $(basename "$patch")"
    else
        git -C "$build_root/source" apply --check "$patch"
        git -C "$build_root/source" apply "$patch"
        applied+=("$patch")
    fi
done

if (( ${#applied[@]} )); then
    cp "$package"/*.so "$build_root/source/vllm/"
    cp "$package/_version.py" "$build_root/source/vllm/_version.py"
    uv venv --system-site-packages --python python3 "$build_root/venv"
    uv pip install --python "$build_root/venv/bin/python" cmake ninja
    export PATH="$build_root/venv/bin:$PATH"
    cmake -S "$build_root/source" -B "$build_root/build" -G Ninja \
        -DVLLM_TARGET_DEVICE=rocm \
        -DVLLM_PYTHON_EXECUTABLE="$build_root/venv/bin/python" \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld \
        -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld \
        -DCMAKE_INSTALL_PREFIX="$build_root/stage" \
        -DVLLM_GPU_ARCHES=gfx950 -DCMAKE_HIP_ARCHITECTURES=gfx950 \
        -DROCM_PATH="$ROCM_PATH"
    cmake --build "$build_root/build" --target _C_stable_libtorch -j "$KIMIK3_KERNEL_BUILD_JOBS"
    cmake --install "$build_root/build" --component _C_stable_libtorch
    install -m 755 "$build_root/stage/vllm/_C_stable_libtorch.abi3.so" "$package/"
    for patch in "${applied[@]}"; do
        while IFS=$'\t' read -r additions deletions path; do
            if [[ "$path" == vllm/*.py ]]; then
                install -D -m 644 "$build_root/source/$path" "$package/${path#vllm/}"
            fi
        done < <(git apply --numstat "$patch")
    done
    python3 -c 'import torch; import vllm._C_stable_libtorch'
fi
printf '%s\n' "$signature" > "$marker"
echo "Kimi-K3 kernel patches installed on $version; build evidence: $build_root"
