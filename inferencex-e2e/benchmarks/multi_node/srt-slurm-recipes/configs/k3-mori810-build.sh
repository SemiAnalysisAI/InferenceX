#!/usr/bin/env bash
set -eo pipefail
source /infmax-workspace/benchmarks/check_env.sh
check_env_vars MORI_SOURCE_COMMIT DYNAMO_VERSION DYNAMO_ENGINE_GRPC_VERSION
venv_path="$1"
src="$venv_path/mori-source"
if [[ ! -f /usr/include/grpcpp/grpcpp.h || ! -f /usr/include/google/protobuf/message.h ]] || ! command -v protoc >/dev/null || ! command -v grpc_cpp_plugin >/dev/null; then
    [[ $(id -u) == 0 ]]
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libgrpc-dev libgrpc++-dev libprotobuf-dev protobuf-compiler protobuf-compiler-grpc libibverbs-dev build-essential
fi
git init "$src"
git -C "$src" remote add origin https://github.com/ROCm/mori.git
git -C "$src" fetch --depth 1 origin "$MORI_SOURCE_COMMIT"
git -C "$src" checkout --detach FETCH_HEAD
[[ $(git -C "$src" rev-parse HEAD) == "$MORI_SOURCE_COMMIT" ]]
git -C "$src" submodule update --init --depth 1 3rdparty/msgpack-c 3rdparty/spdlog
"$venv_path/bin/python" -m pip install --disable-pip-version-check -r "$src/requirements-build.txt"
BUILD_UMBP=ON BUILD_TORCH_SYMM=OFF MORI_SKIP_PRECOMPILE=1 MORI_MORI_MORI_GPU_ARCHS=gfx950 "$venv_path/bin/python" -m pip install --disable-pip-version-check --no-build-isolation --no-deps "$src"
"$venv_path/bin/python" -m pip install --disable-pip-version-check "ai-dynamo==${DYNAMO_VERSION}" "ai-dynamo-runtime==${DYNAMO_VERSION}" "grpcio==${DYNAMO_ENGINE_GRPC_VERSION}" "grpcio-tools==${DYNAMO_ENGINE_GRPC_VERSION}"
