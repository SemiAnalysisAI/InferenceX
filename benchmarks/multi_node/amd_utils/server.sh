#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/../../benchmark_lib.sh" --validation-only
# Multi-Engine Disaggregated Server Dispatcher
# Dispatches to the engine-specific server launcher based on ENGINE env var.
#   ENGINE=sglang-disagg (default) -> server_sglang.sh (SGLang + MoRI)
#   ENGINE=vllm-disagg             -> server_vllm.sh  (vLLM + Nixl/MoRI-IO)
#   ENGINE=atom-disagg             -> server_atom.sh  (ATOM + mooncake)

check_env_vars ENGINE WS_PATH
if [[ -f /config/hicache_mc.env ]]; then
    set -a
    source /config/hicache_mc.env
    set +a
fi
export WS_PATH ENGINE

echo "[DISPATCHER] ENGINE=$ENGINE  WS_PATH=$WS_PATH"

if [[ "$ENGINE" == "vllm-disagg" ]]; then
    source "$WS_PATH/server_vllm.sh"
elif [[ "$ENGINE" == "atom-disagg" ]]; then
    export ATOM_WS_PATH="$WS_PATH"
    source "$WS_PATH/server_atom.sh"
else
    source "$WS_PATH/server_sglang.sh"
fi
