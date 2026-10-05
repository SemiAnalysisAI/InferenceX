#!/usr/bin/env bash
# Install smg-grpc-servicer 0.13.0 (the SMG v1.11.0 servicer) so `vllm serve --grpc`
# can start vLLM's gRPC server, which imports smg_grpc_servicer
# (vllm/entrypoints/launchers/grpc_server.py; vLLM's `grpc` extra is
# smg-grpc-servicer[vllm]>=0.5.2). Engine containers only need it; SMG itself does not.
#
# --no-deps keeps the image's vLLM, torch, grpcio and protobuf as they are. The image
# already ships grpcio, protobuf 6.x, pyzmq and msgspec; the remaining pure-Python deps
# are pinned to the protobuf-6 line, as SMG's own e2e install does:
# https://github.com/smg-project/smg/blob/3be823a700fabaff3add8a390cf78f163479d686/grpc_servicer/pyproject.toml
# https://github.com/smg-project/smg/blob/3be823a700fabaff3add8a390cf78f163479d686/crates/grpc_client/python/pyproject.toml#L1-L6
# https://github.com/smg-project/smg/blob/3be823a700fabaff3add8a390cf78f163479d686/scripts/ci_install_e2e_deps.sh#L28-L42
set -euo pipefail
# Ranks that share one container run this concurrently; serialize the install.
exec 9>/tmp/smg-grpc-servicer-0.13.0-install.lock
flock 9
pip_install=(python3 -m pip install)
if python3 -m pip install --help 2>/dev/null | grep -q -- --break-system-packages; then
    pip_install+=(--break-system-packages)
fi
"${pip_install[@]}" --quiet --no-deps \
    smg-grpc-servicer==0.13.0 \
    smg-grpc-proto==0.4.22 \
    grpcio-health-checking==1.81.1 \
    grpcio-reflection==1.81.1
# Fail here, not at engine start, if the image lacks a dependency.
python3 -c "import grpc_health.v1, grpc_reflection.v1alpha, smg_grpc_servicer.vllm.servicer"
